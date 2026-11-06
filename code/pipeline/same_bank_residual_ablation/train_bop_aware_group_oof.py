"""Nested object-disjoint development of BOP-aware GT-free update selection.

All imputation, calibration, residual guards and policy selection are learned
inside the outer training split. The inspected five-object set is never read.
"""
from __future__ import annotations

import os
for _name in ["OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"]:
    os.environ[_name] = "1"

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, log_loss, precision_score, recall_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

import train_gtfree_risk_calibrator as schema


INITIAL = schema.INITIAL
ACTIONS = schema.ACTIONS
KEYS = schema.KEYS
AR = ["AR_VSD_row", "AR_MSSD_row", "AR_MSPD_row"]
TARGETS = ["delta_AR_VSD", "delta_AR_MSSD", "delta_AR_MSPD", "delta_ADD_S_mm"]
CLASSES = ["adds_regression", "vsd_regression", "mssd_regression", "mspd_regression", "mean_ar_gain"]
POSE_R = [f"R_{i}{j}" for i in range(3) for j in range(3)]
POSE_T = ["t_x_mm", "t_y_mm", "t_z_mm"]
SEED = 20261019


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


def build_features(scores, trials, cohort):
    """Only an explicit observable whitelist is read; missing values stay missing."""
    methods = (INITIAL,) + ACTIONS
    scores = scores[scores.method.isin(methods)].copy()
    trials = trials[trials.method.isin(methods)].copy()
    if scores.duplicated(KEYS+["method"]).any() or trials.duplicated(KEYS+["method"]).any():
        raise ValueError("Duplicate candidate features")
    reference = trials[trials.method.eq(INITIAL)].set_index(KEYS).sort_index()
    index = reference.index
    score_tables = {method: scores[scores.method.eq(method)].set_index(KEYS).reindex(index) for method in methods}
    trial_tables = {method: trials[trials.method.eq(method)].set_index(KEYS).reindex(index) for method in methods}
    if any(len(table.dropna(subset=["method"])) != len(index) for table in [*score_tables.values(), *trial_tables.values()]):
        raise ValueError("Incomplete six-candidate feature bank")
    ranks = {}
    gaps = {}
    for name in schema.SCORE_FEATURES:
        matrix = pd.DataFrame({method: pd.to_numeric(score_tables[method].get(name, pd.Series(np.nan, index=index)), errors="coerce") for method in ACTIONS})
        matrix = matrix.replace([np.inf, -np.inf], np.nan)
        ranks[name] = matrix.rank(axis=1, method="average")/len(ACTIONS)
        gaps[name] = matrix.sub(matrix.min(axis=1), axis=0)
    records = []
    for action in ACTIONS:
        source = score_tables[action]
        values = {}
        for name in schema.SCORE_FEATURES:
            candidate = pd.to_numeric(source.get(name, pd.Series(np.nan, index=index)), errors="coerce")
            initial = pd.to_numeric(score_tables[INITIAL].get(name, pd.Series(np.nan, index=index)), errors="coerce")
            values[f"candidate_{name}"] = candidate.to_numpy()
            values[f"delta_{name}"] = (candidate-initial).to_numpy()
            values[f"rank_{name}"] = ranks[name][action].to_numpy()
            values[f"gap_{name}"] = gaps[name][action].to_numpy()
        for name in schema.TRIAL_FEATURES:
            values[name] = pd.to_numeric(reference.get(name, pd.Series(np.nan, index=index)), errors="coerce").to_numpy()
        aR = trial_tables[action][POSE_R].to_numpy(dtype=float).reshape(-1, 3, 3)
        iR = reference[POSE_R].to_numpy(dtype=float).reshape(-1, 3, 3)
        relative = aR @ np.transpose(iR, (0, 2, 1))
        degrees = np.rad2deg(np.arccos(np.clip((np.trace(relative, axis1=1, axis2=2)-1)/2, -1, 1)))
        tdist = np.linalg.norm(trial_tables[action][POSE_T].to_numpy()-reference[POSE_T].to_numpy(), axis=1)
        delta_R = np.max(np.abs(aR-iR), axis=(1, 2))
        # Compare saved poses directly; arccos near zero is numerically unstable.
        changed = (tdist > 1e-5) | (delta_R > 1e-8)
        degrees = np.where(changed, degrees, 0.)
        diameter = reference.diameter_mm.to_numpy(dtype=float)
        values["actual_translation_change_mm"] = tdist
        values["actual_rotation_change_deg"] = degrees
        values["translation_over_diameter"] = tdist/diameter
        values["surface_motion_bound_over_diameter"] = tdist/diameter + np.sin(np.deg2rad(degrees)/2)
        values["action_step"] = float(action.rsplit("s", 1)[-1])/100
        values["action_is_filtered"] = float("filtered" in action)
        valid_fraction = np.nan_to_num(source.signed_depth_valid_fraction.to_numpy(dtype=float), nan=0.)
        joint_support = np.nan_to_num(source.candidate_joint_support_fraction.to_numpy(dtype=float), nan=0.)
        support = np.sqrt(np.clip(valid_fraction, 0, 1)*np.clip(joint_support, 0, 1))
        part = pd.DataFrame(values).replace([np.inf, -np.inf], np.nan)
        part["observable_missing_fraction"] = part.isna().mean(axis=1)
        for key in KEYS:
            part[key] = reference.reset_index()[key].to_numpy()
        part["cohort"] = cohort
        part["action"] = action
        part["pose_changed"] = changed
        part["observation_support"] = support
        part["features_valid"] = np.isfinite(source.score_bidir.to_numpy(dtype=float)) & (reference.target_points.to_numpy() >= 20)
        records.append(part)
    excluded = set(KEYS + ["cohort", "action", "pose_changed", "observation_support", "features_valid"])
    feature_names = [c for c in records[0].columns if c not in excluded]
    return pd.concat(records, ignore_index=True), feature_names


def attach_labels(long, labels):
    key = ["cohort"]+KEYS
    reference = labels[labels.method.eq(INITIAL)][key+["ADD_S_mm"]+AR].rename(
        columns={"ADD_S_mm": "initial_ADD_S_mm", **{c:"initial_"+c for c in AR}})
    candidate = labels[labels.method.isin(ACTIONS)][key+["method", "ADD_S_mm"]+AR].rename(columns={"method":"action"})
    result = long.merge(candidate, on=key+["action"], validate="one_to_one").merge(reference, on=key, validate="many_to_one")
    if len(result) != len(long):
        raise ValueError("Feature/label join lost candidates")
    for column, target in zip(AR, TARGETS[:3]):
        result[target] = result[column]-result["initial_"+column]
    result["delta_ADD_S_mm"] = result.ADD_S_mm-result.initial_ADD_S_mm
    result["adds_regression"] = (result.delta_ADD_S_mm > 1e-9).astype(int)
    for target, column in zip(CLASSES[1:4], TARGETS[:3]):
        result[target] = (result[column] < -1e-12).astype(int)
    result["mean_ar_gain"] = (result[TARGETS[:3]].mean(axis=1) > 1e-12).astype(int)
    result["target_id"] = result.groupby(key, sort=False, dropna=False).ngroup()
    return result


def mask_fit(data):
    return data.pose_changed.to_numpy(dtype=bool) & data.features_valid.to_numpy(dtype=bool)


def transform(bundle, data, names):
    values = data[names].to_numpy(dtype=float)
    return np.clip(bundle["scaler"].transform(bundle["imputer"].transform(values)), -20., 20.)


def fit_raw(data, names):
    fitting = data.loc[mask_fit(data)]
    if len(fitting) < 12:
        raise ValueError("Insufficient moving/valid development candidates")
    imputer = SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)
    X = imputer.fit_transform(fitting[names].to_numpy(dtype=float))
    scaler = StandardScaler().fit(X)
    X = np.clip(scaler.transform(X), -20., 20.)
    regressor = Ridge(alpha=10.).fit(X, fitting[TARGETS].to_numpy(dtype=float))
    classifiers = {}
    for label in CLASSES:
        y = fitting[label].to_numpy(dtype=int)
        if len(np.unique(y)) == 1:
            classifiers[label] = float((y.sum()+.5)/(len(y)+1))
        else:
            classifiers[label] = LogisticRegression(C=1., max_iter=2000, random_state=SEED).fit(X, y)
    neighbors = NearestNeighbors(n_neighbors=min(2, len(X))).fit(X)
    radius = max(float(np.quantile(neighbors.kneighbors(X)[0][:, -1], .95)), 1.)
    return {"imputer":imputer, "scaler":scaler, "regressor":regressor,
            "classifiers":classifiers, "neighbors":neighbors, "domain_radius":radius,
            "feature_names":names, "training_objects":sorted(map(int, data.object_id.unique()))}


def predict_raw(bundle, data, names):
    X = transform(bundle, data, names)
    predicted = bundle["regressor"].predict(X)
    predicted[:, :3] = np.clip(predicted[:, :3], -1, 1)
    result = pd.DataFrame(predicted, columns=["pred_"+t for t in TARGETS], index=data.index)
    for label in CLASSES:
        model = bundle["classifiers"][label]
        result["raw_p_"+label] = model if isinstance(model, float) else model.predict_proba(X)[:, 1]
    distance = bundle["neighbors"].kneighbors(X, n_neighbors=1)[0][:, 0]
    result["training_support"] = 1/(1+distance/bundle["domain_radius"])
    return result


def folds(data, number):
    count = min(number, data.object_id.nunique())
    if count < 2:
        raise ValueError("At least two object groups required")
    return GroupKFold(n_splits=count).split(data, groups=data.object_id.to_numpy())


def base_oof(data, names, number=3):
    parts = []
    for train, valid in folds(data, number):
        fitting, held = data.iloc[train], data.iloc[valid]
        if set(fitting.object_id) & set(held.object_id):
            raise ValueError("Object leakage in base OOF")
        bundle = fit_raw(fitting, names)
        parts.append(predict_raw(bundle, held, names))
    return pd.concat(parts).sort_index().loc[data.index]


def logits(probabilities):
    p = np.clip(np.asarray(probabilities), 1e-5, 1-1e-5)
    return np.log(p/(1-p)).reshape(-1, 1)


def fit_calibration(data, raw):
    eligible = mask_fit(data)
    active = data.loc[eligible]
    predictions = raw.loc[eligible]
    calibrators = {}
    for label in CLASSES:
        y = active[label].to_numpy(dtype=int)
        if len(np.unique(y)) < 2:
            calibrators[label] = float((y.sum()+.5)/(len(y)+1))
        else:
            calibrators[label] = LogisticRegression(C=1., max_iter=1000, random_state=SEED).fit(logits(predictions["raw_p_"+label]), y)
    corrections = {}
    for action in ACTIONS:
        index = active.index[active.action.eq(action)]
        if len(index) < 8:
            index = active.index
        values = []
        for i, target in enumerate(TARGETS):
            # Empirical directional guards, not conditional coverage guarantees.
            residual = (predictions.loc[index, "pred_"+target]-active.loc[index, target]) if i < 3 else (active.loc[index, target]-predictions.loc[index, "pred_"+target])
            values.append(max(0., float(np.quantile(residual, .75))))
        corrections[action] = values
    return {"calibrators":calibrators, "corrections":corrections}


def calibrate(predictions, calibration, data):
    result = predictions.copy()
    for label in CLASSES:
        model = calibration["calibrators"][label]
        result["p_"+label] = model if isinstance(model, float) else model.predict_proba(logits(predictions["raw_p_"+label]))[:, 1]
    for i, target in enumerate(TARGETS):
        correction = data.action.map({a:v[i] for a,v in calibration["corrections"].items()}).to_numpy()
        result[("lcb_" if i < 3 else "ucb_")+target] = result["pred_"+target] + (-correction if i < 3 else correction)
    return result


def risk_bin_envelope(data, probabilities):
    edges = np.array([0., .1, .2, .4, .6, .8, 1.000001])
    valid = mask_fit(data)
    p = np.asarray(probabilities)[valid]
    active = data.loc[valid]
    ids = np.searchsorted(edges, p, side="right")-1
    objects = np.sort(active.object_id.unique())
    rng = np.random.default_rng(SEED)
    samples = rng.integers(0, len(objects), size=(400, len(objects)))
    upper = []
    for bin_id in range(len(edges)-1):
        selected = active.iloc[np.flatnonzero(ids == bin_id)]
        if len(selected) < 20 or selected.object_id.nunique() < 3:
            upper.append(1.)
            continue
        sums = selected.groupby("object_id").adds_regression.sum().reindex(objects, fill_value=0).to_numpy()
        counts = selected.groupby("object_id").size().reindex(objects, fill_value=0).to_numpy()
        denominator = counts[samples].sum(axis=1)
        rates = sums[samples].sum(axis=1)/np.maximum(denominator, 1)
        upper.append(float(np.quantile(rates[denominator > 0], .90)))
    return {"edges":edges, "upper":np.asarray(upper)}


def add_risk_bound(predictions, envelope):
    result = predictions.copy()
    p = result.p_adds_regression.to_numpy()
    bins = np.clip(np.searchsorted(envelope["edges"], p, side="right")-1, 0, len(envelope["upper"])-1)
    result["adds_risk_upper"] = np.maximum(p, envelope["upper"][bins])
    return result


def fit_bundle(data, names):
    raw = base_oof(data, names)
    calibration = fit_calibration(data, raw)
    calibrated = calibrate(raw, calibration, data)
    envelope = risk_bin_envelope(data, calibrated.p_adds_regression)
    return {"raw_model":fit_raw(data, names), "calibration":calibration, "risk_envelope":envelope,
            "feature_names":names, "training_objects":sorted(map(int, data.object_id.unique()))}


def predict_bundle(bundle, data):
    raw = predict_raw(bundle["raw_model"], data, bundle["feature_names"])
    return add_risk_bound(calibrate(raw, bundle["calibration"], data), bundle["risk_envelope"])


def calibrated_oof(data, names, number=3):
    parts = []
    for train, valid in folds(data, number):
        fitting, held = data.iloc[train], data.iloc[valid]
        if set(fitting.object_id) & set(held.object_id):
            raise ValueError("Object leakage in calibration/policy OOF")
        parts.append(predict_bundle(fit_bundle(fitting, names), held))
    return pd.concat(parts).sort_index().loc[data.index]


def select(data, predictions, risk_limit, component_loss, disabled=False):
    if not data.index.equals(predictions.index):
        raise ValueError("Prediction order mismatch")
    lcb = predictions[["lcb_"+t for t in TARGETS[:3]]].to_numpy()
    acceptable = (
        mask_fit(data) & (data.observation_support.to_numpy() >= .01)
        & (predictions.training_support.to_numpy() >= .2)
        & (predictions.adds_risk_upper.to_numpy() <= risk_limit)
        & (predictions.ucb_delta_ADD_S_mm.to_numpy() <= 0)
        & (lcb.min(axis=1) >= -component_loss) & (lcb.mean(axis=1) > 0)
    )
    if disabled:
        acceptable[:] = False
    result = []
    for target_id, indices in data.groupby("target_id", sort=True).groups.items():
        positions = data.index.get_indexer(indices)
        valid = positions[acceptable[positions]]
        if len(valid):
            chosen = min(valid, key=lambda j:(-lcb[j].mean(), predictions.iloc[j].adds_risk_upper, str(data.iloc[j].action)))
            row = data.iloc[chosen]
            selected = row.action
            add = row.ADD_S_mm
            ar = row[AR].to_numpy(dtype=float)
        else:
            row = data.iloc[positions[0]]
            selected = INITIAL
            add = row.initial_ADD_S_mm
            ar = row[["initial_"+c for c in AR]].to_numpy(dtype=float)
        result.append(dict(target_id=target_id, object_id=row.object_id, scene_id=row.scene_id,
                           img_id=row.img_id, inst_id=row.inst_id, frame_id=row.frame_id,
                           input_depth_condition=row.input_depth_condition, cohort=row.cohort,
                           selected_action=selected, ADD_S_mm=add, initial_ADD_S_mm=row.initial_ADD_S_mm,
                           delta_ADD_S_mm=add-row.initial_ADD_S_mm,
                           **{c:ar[i] for i,c in enumerate(AR)}, AR_mean_row=ar.mean()))
    return pd.DataFrame(result)


def tune_policy(data, predictions):
    base = data[data.action.eq("protocolF_se3_trust_p2p_s100")]
    initial_ar = base[["initial_"+c for c in AR]].to_numpy().mean()
    regression_limit = float((base.delta_ADD_S_mm > 1e-9).mean())
    records = []
    for risk_limit in [.1, .2, .3]:
        for component_loss in [0., .01]:
            selected = select(data, predictions, risk_limit, component_loss)
            record = dict(risk_limit=risk_limit, component_loss=component_loss,
                          selected_updates=int(selected.selected_action.ne(INITIAL).sum()),
                          mean_AR=selected.AR_mean_row.mean(),
                          adds_delta_mm=selected.delta_ADD_S_mm.mean(),
                          regression_rate=(selected.delta_ADD_S_mm > 1e-9).mean())
            record["passes_development_constraints"] = bool(
                record["selected_updates"] > 0 and record["mean_AR"] > initial_ar
                and record["adds_delta_mm"] <= 0 and record["regression_rate"] <= regression_limit)
            records.append(record)
    candidates = [r for r in records if r["passes_development_constraints"]]
    if not candidates:
        return {"risk_limit":.1, "component_loss":0., "disabled":True}, records
    best = min(candidates, key=lambda r:(-r["mean_AR"], r["regression_rate"], r["adds_delta_mm"], r["risk_limit"], r["component_loss"]))
    return {"risk_limit":best["risk_limit"], "component_loss":best["component_loss"], "disabled":False}, records


def baseline_rows(data, method):
    if method == INITIAL:
        rows = data.drop_duplicates("target_id").copy()
        rows["ADD_S_mm"] = rows.initial_ADD_S_mm
        for column in AR:
            rows[column] = rows["initial_"+column]
    else:
        rows = data[data.action.eq(method)].copy()
    result = rows[["target_id", "object_id", "scene_id", "img_id", "inst_id", "frame_id", "input_depth_condition", "cohort", "ADD_S_mm", "initial_ADD_S_mm"]+AR].copy()
    result["selected_action"] = method
    result["delta_ADD_S_mm"] = result.ADD_S_mm-result.initial_ADD_S_mm
    result["AR_mean_row"] = result[AR].mean(axis=1)
    return result


def metric_summary(rows):
    result = []
    scopes = [("all", "pooled", rows)]
    for column in ["cohort", "object_id", "input_depth_condition"]:
        scopes += [(column, str(value), group) for value,group in rows.groupby(column, sort=True)]
    for scope, stratum, group in scopes:
        for method, part in group.groupby("method", sort=False):
            result.append(dict(scope=scope, stratum=stratum, method=method, target_conditions=len(part),
                               mean_ADD_S_mm=part.ADD_S_mm.mean(), delta_ADD_S_mm=part.delta_ADD_S_mm.mean(),
                               regression_rate=(part.delta_ADD_S_mm > 1e-9).mean(),
                               update_fraction=part.selected_action.ne(INITIAL).mean(),
                               **{c:part[c].mean() for c in AR}, AR_mean=part.AR_mean_row.mean()))
    return pd.DataFrame(result)


def calibration_report(data, predictions):
    stats, bins = [], []
    valid = mask_fit(data)
    for label in CLASSES:
        y = data.loc[valid, label].to_numpy(dtype=int)
        for mode, column in [("raw", "raw_p_"+label), ("calibrated", "p_"+label)]:
            p = np.clip(predictions.loc[valid, column].to_numpy(), 1e-9, 1-1e-9)
            stats.append(dict(label=label, mode=mode, rows=len(y), positives=int(y.sum()), prevalence=y.mean(),
                              brier=brier_score_loss(y,p), log_loss=log_loss(y,p,labels=[0,1]),
                              precision_at_05=precision_score(y,p>=.5,zero_division=0),
                              recall_at_05=recall_score(y,p>=.5,zero_division=0)))
            ids = np.minimum((p*10).astype(int),9)
            for bin_id in range(10):
                selected = ids == bin_id
                bins.append(dict(label=label, mode=mode, bin=bin_id, rows=int(selected.sum()),
                                 predicted=float(p[selected].mean()) if selected.any() else np.nan,
                                 observed=float(y[selected].mean()) if selected.any() else np.nan))
    return pd.DataFrame(stats), pd.DataFrame(bins)


def object_bootstrap(rows):
    chosen = rows[rows.method.eq("bop_aware_nested_oof")]
    rng = np.random.default_rng(SEED)
    result = []
    for reference in [INITIAL, "protocolF_se3_trust_p2p_s100", "protocolF_se3_trust_filtered_s100"]:
        ref = rows[rows.method.eq(reference)]
        pair = chosen.merge(ref, on=["target_id","object_id"], suffixes=("_candidate","_reference"),validate="one_to_one")
        for column in ["ADD_S_mm", "AR_mean_row"]+AR:
            pair["delta"] = pair[column+"_candidate"]-pair[column+"_reference"]
            agg = pair.groupby("object_id").delta.agg(["sum","count"])
            indices = rng.integers(0,len(agg),size=(2000,len(agg)))
            distribution = agg["sum"].to_numpy()[indices].sum(axis=1)/agg["count"].to_numpy()[indices].sum(axis=1)
            low,high = np.quantile(distribution,[.025,.975])
            result.append(dict(reference=reference, metric=column, delta=pair.delta.mean(), ci_low=low, ci_high=high,
                               object_clusters=len(agg), scope="exploratory_development_nested_oof_not_final_confirmation"))
    return pd.DataFrame(result)


def make_plots(bins, summary, grid, output):
    fig, axes = plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for ax,label in zip(axes,["adds_regression","mean_ar_gain"]):
        for mode,color in [("raw","#666666"),("calibrated","#237e70")]:
            part = bins[(bins.label==label)&(bins["mode"]==mode)&(bins.rows>0)]
            ax.plot(part.predicted,part.observed,"o-",label=mode,color=color)
        ax.plot([0,1],[0,1],"--",color="#aaaaaa")
        ax.set(xlabel="Predicted probability",ylabel="Observed frequency",title=label,xlim=(0,1),ylim=(0,1))
        ax.legend()
    fig.savefig(output/"outer_oof_calibration.png",dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6,4),constrained_layout=True)
    ax.scatter(grid.regression_rate*100,grid.mean_AR*100,c="#237e70",label="Development policy grid")
    for row in summary[summary.scope.eq("all")].itertuples():
        ax.scatter(row.regression_rate*100,row.AR_mean*100,marker="x",s=60)
        ax.annotate(row.method.replace("protocolF_se3_trust_", ""),(row.regression_rate*100,row.AR_mean*100),fontsize=7)
    ax.set(xlabel="ADD-S regression rate (%)",ylabel="BOP-style mean AR (%)")
    ax.legend(fontsize=8)
    fig.savefig(output/"development_risk_ar_tradeoff.png",dpi=180)
    plt.close(fig)


def run(args):
    if args.output.exists():
        raise FileExistsError(f"Preserve existing output: {args.output}")
    state = json.loads((args.labels/"label_completion_state.json").read_text(encoding="utf-8"))
    if state["status"] != "COMPLETE":
        raise ValueError("Label completion has not passed")
    path = args.labels/"development_bop_labels.csv"
    if digest(path) != state["labels_sha256"]:
        raise ValueError("Development labels changed")
    labels = pd.read_csv(path)
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    all_long, names, source_hashes = [], None, {str(path):digest(path)}
    for cohort in plan["cohorts"]:
        trial_path = Path(cohort["trials"])
        score_path = trial_path.with_name("se3_trust_region_scores.csv")
        for source in [trial_path,score_path]:
            if digest(source) != state["source_hashes"][str(source)]:
                raise ValueError("Frozen development input changed")
            source_hashes[str(source)] = digest(source)
        trials, scores = pd.read_csv(trial_path), pd.read_csv(score_path)
        long, feature_names = build_features(scores,trials,cohort["cohort"])
        if names is not None and names != feature_names:
            raise ValueError("Feature schema differs between cohorts")
        names = feature_names
        # Label-blind replay: poisoned evaluation columns must not affect features.
        shadow = trials.copy()
        for column in ["ADD_S_mm","mask_iou","rotation_error_deg","translation_error_mm"]:
            if column in shadow:
                shadow[column] = np.nan
        replay, _ = build_features(scores,shadow,cohort["cohort"])
        if not np.allclose(long[names],replay[names],equal_nan=True):
            raise ValueError("GT-free shadow feature test failed")
        all_long.append(long)
    observable = pd.concat(all_long,ignore_index=True)
    data = attach_labels(observable,labels).reset_index(drop=True)
    if len(data)!=3396 or data.target_id.nunique()!=566 or data.object_id.nunique()!=22:
        raise ValueError("Development training dimensions differ from frozen plan")
    if set(data.object_id)&set(plan["excluded_inspected_objects"]):
        raise ValueError("Inspected holdout contamination")
    args.output.mkdir(parents=True)
    observable.to_csv(args.output/"observable_candidate_features.csv",index=False)
    data[["target_id","object_id","cohort","action"]+TARGETS+CLASSES].to_csv(args.output/"development_labels_long.csv",index=False)
    log = {"status":"NESTED_OOF_RUNNING", "training_objects":sorted(map(int,data.object_id.unique())),
           "excluded_inspected_objects":plan["excluded_inspected_objects"],"outer_folds":[],
           "evaluation_depth_policy":"native_sensor_depth", "feature_names":names,
           "model_hyperparameters":{"ridge_alpha":10.,"logistic_C":1.,"residual_guard_quantile":.75},
           "policy_grid":{"risk_limit":[.1,.2,.3],"component_loss":[0.,.01]},
           "environment_versions":{"python":sys.version,**{name:version(name) for name in ["numpy","pandas","scipy","scikit-learn","joblib","matplotlib"]}},
           "source_hashes":source_hashes,"started_at_utc":datetime.now(timezone.utc).isoformat()}
    dump(args.output/"training_state.json",log)
    outer_predictions, selected_parts, policy_records = [], [], []
    for fold_id,(train,valid) in enumerate(folds(data,5)):
        training,held = data.iloc[train],data.iloc[valid]
        train_objects=sorted(map(int,training.object_id.unique()))
        held_objects=sorted(map(int,held.object_id.unique()))
        if set(train_objects)&set(held_objects):
            raise ValueError("Outer object leakage")
        print(f"OUTER {fold_id+1}/5: train={train_objects}; held={held_objects}; inner calibration/policy fitting",flush=True)
        inner = calibrated_oof(training,names)
        policy,grid = tune_policy(training,inner)
        policy_records += [dict(row,outer_fold=fold_id,scope="inner_policy_selection") for row in grid]
        bundle = fit_bundle(training,names)
        predicted = predict_bundle(bundle,held)
        predicted["outer_fold"] = fold_id
        outer_predictions.append(predicted)
        selected = select(held,predicted,**policy)
        selected["outer_fold"] = fold_id
        selected_parts.append(selected)
        record = {"fold":fold_id,"training_objects":train_objects,"validation_objects":held_objects,
                  "selected_policy":policy,"held_target_conditions":len(selected),
                  "selected_updates":int(selected.selected_action.ne(INITIAL).sum())}
        train_images = set(map(tuple,training[["scene_id","img_id"]].drop_duplicates().to_numpy()))
        held_images = set(map(tuple,held[["scene_id","img_id"]].drop_duplicates().to_numpy()))
        record["shared_rgbd_images_between_different_objects"] = len(train_images & held_images)
        log["outer_folds"].append(record)
        joblib.dump({"bundle":bundle,"policy":policy,"fold":record},args.output/f"outer_fold_{fold_id}.joblib")
        dump(args.output/"training_state.json",log)
        print("OUTER COMPLETE",record,flush=True)
    oof = pd.concat(outer_predictions).sort_index().loc[data.index]
    oof_export = pd.concat([data[["target_id","object_id","cohort","action","pose_changed","features_valid"]],oof],axis=1)
    oof_export.to_csv(args.output/"outer_oof_candidate_predictions.csv",index=False)
    selected = pd.concat(selected_parts,ignore_index=True)
    selected["method"] = "bop_aware_nested_oof"
    tables = [selected]
    for method in [INITIAL,"protocolF_se3_trust_p2p_s100","protocolF_se3_trust_filtered_s100"]:
        part = baseline_rows(data,method)
        part["method"] = method
        tables.append(part)
    outcomes = pd.concat(tables,ignore_index=True)
    outcomes.to_csv(args.output/"nested_oof_selected_and_baselines.csv",index=False)
    summary = metric_summary(outcomes)
    summary.to_csv(args.output/"nested_oof_summary.csv",index=False)
    stats,bins = calibration_report(data,oof)
    stats.to_csv(args.output/"outer_oof_calibration_metrics.csv",index=False)
    bins.to_csv(args.output/"outer_oof_reliability_bins.csv",index=False)
    object_bootstrap(outcomes).to_csv(args.output/"paired_object_bootstrap.csv",index=False)
    print("Fitting all-development candidate artifact; choosing final policy from outer-OOF only",flush=True)
    final_policy,final_grid = tune_policy(data,oof)
    grid = pd.DataFrame(final_grid)
    grid.to_csv(args.output/"all_development_policy_grid_exploratory.csv",index=False)
    pd.DataFrame(policy_records).to_csv(args.output/"inner_policy_grid.csv",index=False)
    final_bundle = fit_bundle(data,names)
    joblib.dump({"bundle":final_bundle,"policy":final_policy,"status":"DEVELOPMENT_TRAINED_NOT_PROMOTED"},
                args.output/"bop_aware_development_candidate.joblib")
    # Batch invariance after training-only imputation, without any outcome fields.
    first = data[data.target_id.eq(data.target_id.iloc[0])]
    full = predict_bundle(final_bundle,data)
    one = predict_bundle(final_bundle,first)
    if not np.allclose(full.loc[first.index],one,atol=1e-9,rtol=1e-9):
        raise ValueError("Inference depends on evaluation batch")
    make_plots(bins,summary,grid,args.output)
    checks = {"complete_3962_pose_labels":len(labels)==3962,"all_22_development_objects":True,
              "no_inspected_five_objects_read":True,"outer_object_disjoint":True,
              "nested_group_calibration_and_policy_selection":True,"fold_local_imputation":True,
              "gt_shadow_feature_replay":True,"single_target_batch_invariance":True,
              "not_official_leaderboard":True,"method_promoted":False}
    pd.DataFrame([
        ("Simpson paradox","all object/cohort/depth summaries exported; pooled gain alone insufficient"),
        ("Ecological fallacy","no per-frame guarantees inferred from object means"),
        ("Selection/Berkson","conditional on matched external predictions; not full target-list recall"),
        ("Collider bias","GT outcomes are training labels only, never input filters"),
        ("Base rate neglect","calibration tables include prevalence and positive/negative classes"),
        ("Regression to mean","all methods evaluated on identical frozen initial conditions"),
        ("Survivorship","upstream missing-prediction coverage retained; no end-to-end success claim"),
        ("Look elsewhere","all six development policies exported; no selected p-values"),
        ("Forking paths","fixed model family and grid; outer OOF distinct from final policy search"),
        ("Causal overreach","development-only paired algorithm comparison; no platform claims"),
        ("Reverse causality","outcomes removed in GT shadow test; no outcome-based inference"),
    ],columns=["check","assessment"]).to_csv(args.output/"statistical_fallacy_scan_11.csv",index=False)
    log.update(status="COMPLETE_DEVELOPMENT_ONLY_NOT_PROMOTED",final_candidate_policy=final_policy,
               checks=checks,completed_at_utc=datetime.now(timezone.utc).isoformat())
    for source in [Path(__file__),Path(__file__).with_name("bop_aware_acceptance.py"),Path(__file__).with_name("train_gtfree_risk_calibrator.py")]:
        log["source_hashes"][str(source)]=digest(source)
    lines=["# 22 物体 BOP-aware 对象分组 OOF 训练与校准", "", "## Material Passport",
           "- Status: COMPLETE_DEVELOPMENT_ONLY_NOT_PROMOTED; Verification Status: ANALYZED",
           "- 3,962 个位姿标签；566 个目标条件；3,396 个动作条件行；22 个开发物体。",
           "- VSD 统一使用原始传感器深度；方法输入的 native/outlier 条件保留。不能与旧五物体使用受污染评价深度的 VSD 直接混比。",
           "- 外层 5 折按对象留出；每折内单独做对象分组 OOF、概率校准、经验残差修正与六组策略选择；外层验证标签不参与这些拟合。",
           "- 特征只来自白名单观测字段；缺失值和标准化在各训练折拟合；零运动不计为有效更新。",
           "", "## 外层 OOF 主结果", "```text", summary[summary.scope.eq("all")].to_string(index=False), "```",
           "", "## 外层 OOF 校准", "```text", stats.to_string(index=False), "```",
           "", "## 最终开发候选策略", "```text", json.dumps(final_policy,indent=2), "```",
           "- 此策略依据完整开发 OOF 选择；其在同一 OOF 上的策略搜索结果是开发拟合，不是新的独立验证。主表使用每个外层训练折内选择的策略。",
           "- LCB/UCB 是开发经验残差和对象 bootstrap 校准箱上界，不是有保证的逐样本安全置信界。",
           "- 未扩展尚未重算的渲染尾部残差特征；本轮复用观测几何并增加实际零运动/归一化运动特征。",
           "- 当前五个诊断物体未读取；没有真实平台实验，没有新的确认集分数，没有自动晋级论文主方法。",
           "- 配对 CI 按物体聚类，仅作探索性开发解释；不进行挑选性显著性声明。",
           "- 这是物体分组验证，不是场景/图像分组验证。不同物体可能来自同一 RGB-D 图像；各折共享图像数量记录在 training_state.json，不宣称图像与场景完全独立。",
           "", "## 审计", "```text", json.dumps(checks,indent=2), "```", ""]
    (args.output/"group_oof_training_report_zh.md").write_text("\n".join(lines),encoding="utf-8")
    snapshot=args.output/"source_snapshot"
    snapshot.mkdir()
    for name in ["train_bop_aware_group_oof.py","run_bop_aware_label_completion.py","bop_aware_acceptance.py",
                 "bop_thresholds.py","evaluate_bop_metrics_controlled.py","train_gtfree_risk_calibrator.py",
                 "test_bop_aware_group_oof.py","audit_bop_aware_development.py"]:
        shutil.copy2(Path(__file__).with_name(name),snapshot/name)
    log["artifact_hashes"]={str(p.relative_to(args.output)):digest(p) for p in args.output.rglob("*") if p.is_file() and p.name!="training_state.json"}
    dump(args.output/"training_state.json",log)
    print(summary[summary.scope.eq("all")].to_string(index=False),flush=True)
    print("TRAINING COMPLETE",args.output,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    root=Path(r"D:\6D_pose\codex_paper_workspace")
    parser.add_argument("--labels",type=Path,default=root/"outputs/bop_aware_development_labels_20261019_r1")
    parser.add_argument("--plan",type=Path,default=root/"outputs/bop_threshold_diagnosis_20261019_r1/development_label_plan/development_bop_label_plan.json")
    parser.add_argument("--output",type=Path,default=root/"outputs/bop_aware_group_oof_20261019_r1")
    run(parser.parse_args())
