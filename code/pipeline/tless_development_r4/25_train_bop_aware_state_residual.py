"""Exploratory object-OOF conditional residual guards and joint acceptance."""
from __future__ import annotations

import os
for _name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[_name] = "1"

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time
import traceback

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer

import train_bop_aware_group_oof as base
import train_bop_aware_tail_risk_v2 as v2
import train_bop_aware_selective_calibration as selective


STATE_NAMES = [
    "action_step", "action_is_filtered", "actual_translation_change_mm",
    "actual_rotation_change_deg", "translation_over_diameter",
    "model_motion_p95_over_diameter", "projected_motion_p95_640px",
    "candidate_candidate_joint_support_fraction", "candidate_front_free_space_fraction",
    "common_support_fraction", "common_abs_p95_delta_mm",
    "common_pixel_improved_fraction", "common_pixel_worsened_fraction",
    "tail_silhouette_iou", "tail_delta_silhouette_iou",
    "tail_depth_abs_mm_p95", "tail_delta_depth_abs_mm_p95",
    "tail_forward_nn_mm_p95", "tail_delta_forward_nn_mm_p95",
    "tail_reverse_nn_mm_p95", "tail_delta_reverse_nn_mm_p95",
    "tail_boundary_640px_p95", "tail_delta_boundary_640px_p95",
    "initializer_score", "mask_score", "observable_missing_fraction",
]
EXTRA_TARGETS = ["delta_AR_mean", "delta_joint_utility"]
ALL_TARGETS = base.TARGETS + EXTRA_TARGETS


def matrix(data, raw, names):
    if not data.index.equals(raw.index):
        raise ValueError("Residual feature/prediction index mismatch")
    result = data[names].astype(float).copy()
    for target in base.TARGETS:
        result["raw_pred_" + target] = raw["pred_" + target]
    for label in base.CLASSES:
        result["raw_probability_" + label] = raw["raw_p_" + label]
    return result.replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)


def expected_targets(raw, weight):
    expected = {target: raw["pred_" + target].to_numpy() for target in base.TARGETS}
    expected["delta_AR_mean"] = np.mean([expected[t] for t in base.TARGETS[:3]], axis=0)
    expected["delta_joint_utility"] = expected["delta_AR_mean"] - weight * expected["delta_ADD_S_mm"] / 5.
    return expected


def residual_targets(data, raw, weight):
    expected = expected_targets(raw, weight)
    observed = {target: data[target].to_numpy() for target in base.TARGETS}
    observed["delta_AR_mean"] = data[base.TARGETS[:3]].mean(axis=1).to_numpy()
    observed["delta_joint_utility"] = observed["delta_AR_mean"] - weight * observed["delta_ADD_S_mm"] / 5.
    return pd.DataFrame({t: observed[t] - expected[t] if t == "delta_ADD_S_mm"
                         else expected[t] - observed[t] for t in ALL_TARGETS}, index=data.index)


def fit_quantiles(data, raw, residuals, names, plan):
    imputer = SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)
    X = imputer.fit_transform(matrix(data, raw, names))
    weights = v2.object_frame_weights(data)
    models = {target: HistGradientBoostingRegressor(
        loss="quantile", quantile=plan["residual_quantile"], random_state=base.SEED,
        **plan["quantile_model"]).fit(X, residuals[target], sample_weight=weights)
        for target in ALL_TARGETS}
    return dict(imputer=imputer, models=models)


def predict_quantiles(model, data, raw, names):
    X = model["imputer"].transform(matrix(data, raw, names))
    return pd.DataFrame({t: m.predict(X) for t, m in model["models"].items()}, index=data.index)


def fit_residual_guard(data, raw, names, plan):
    active = data.loc[base.mask_fit(data)]
    raw = raw.loc[active.index]
    states = [name for name in STATE_NAMES if name in names]
    residuals = residual_targets(active, raw, plan["joint_continuous_weight"])
    crossfit = pd.DataFrame(index=active.index, columns=ALL_TARGETS, dtype=float)
    partitions = []
    for train, valid in base.folds(active, 3):
        training, held = active.iloc[train], active.iloc[valid]
        if set(training.object_id) & set(held.object_id):
            raise ValueError("Object leakage in residual crossfit")
        qmodel = fit_quantiles(training, raw.loc[training.index], residuals.loc[training.index], states, plan)
        crossfit.loc[held.index] = predict_quantiles(qmodel, held, raw.loc[held.index], states)
        partitions.append(dict(training_objects=sorted(map(int, training.object_id.unique())),
                               validation_objects=sorted(map(int, held.object_id.unique()))))
    weights = v2.object_frame_weights(active)
    offsets = {t: max(0., v2.weighted_quantile(residuals[t] - crossfit[t], weights,
                                             plan["residual_quantile"])) for t in ALL_TARGETS}
    globals_ = {}
    for action in base.ACTIONS:
        subset = active.loc[active.action.eq(action)]
        if len(subset) < 8:
            subset = active
        w = v2.object_frame_weights(subset)
        globals_[action] = {t: max(0., v2.weighted_quantile(residuals.loc[subset.index, t], w,
                                                          plan["residual_quantile"])) for t in ALL_TARGETS}
    return dict(model=fit_quantiles(active, raw, residuals, states, plan), states=states,
                offsets=offsets, global_guards=globals_, training_objects=sorted(map(int, data.object_id.unique())),
                residual_crossfit_partitions=partitions, low_support=plan["low_support_fallback"],
                joint_weight=plan["joint_continuous_weight"])


def apply_guard(data, predictions, guard):
    result = predictions.copy()
    conditional = predict_quantiles(guard["model"], data, predictions, guard["states"])
    expected = expected_targets(predictions, guard["joint_weight"])
    weak = predictions.training_support.to_numpy() < guard["low_support"]
    for target in ALL_TARGETS:
        width = np.maximum(0., conditional[target].to_numpy() + guard["offsets"][target])
        fallback = data.action.map({a: g[target] for a, g in guard["global_guards"].items()}).to_numpy()
        width = np.where(weak, np.maximum(width, fallback), width)
        result["guard_" + target] = width
        result[("ucb_" if target == "delta_ADD_S_mm" else "lcb_") + target] = (
            expected[target] + width if target == "delta_ADD_S_mm" else expected[target] - width)
    result["conditional_guard_used"] = (~weak).astype(float)
    return result


def fit_bundle(data, names, raw_plan, plan, existing=None):
    raw = v2.raw_oof(data, names, raw_plan)
    if existing is None:
        calibration = selective.fit_calibration(data, raw, names)
        probabilities = selective.calibrate(raw, data, calibration)
        active = data.loc[base.mask_fit(data)].copy()
        active["residual"] = active.adds_regression - probabilities.loc[active.index, "p_adds_regression"]
        biases = [float(np.average(group.residual, weights=v2.object_frame_weights(group)))
                  for _, group in active.groupby("object_id")]
        existing = dict(raw_model=v2.fit_raw(data, names, raw_plan), calibration=calibration,
                        risk_bias_guard=max(0., float(np.quantile(biases, .8))), feature_names=names,
                        training_objects=sorted(map(int, data.object_id.unique())))
    if existing["training_objects"] != sorted(map(int, data.object_id.unique())):
        raise ValueError("Frozen raw bundle training partition mismatch")
    return dict(base_bundle=existing, residual_guard=fit_residual_guard(data, raw, names, plan),
                training_objects=existing["training_objects"], feature_names=names)


def predict_bundle(bundle, data):
    return apply_guard(data, selective.predict_bundle(bundle["base_bundle"], data), bundle["residual_guard"])


def gate_arrays(data, p, policy, variant, plan):
    if not data.index.equals(p.index):
        raise ValueError("Decision index mismatch")
    lower = p[["lcb_" + t for t in base.TARGETS[:3]]].to_numpy()
    add = p.ucb_delta_ADD_S_mm.to_numpy()
    gates = dict(moving_valid=base.mask_fit(data), observation=data.observation_support.to_numpy() >= .01,
                 domain=p.training_support.to_numpy() >= .2,
                 risk=p.adds_risk_upper.to_numpy() <= policy["risk_limit"], ADD_upper=add <= 0,
                 AR_component=lower.min(axis=1) >= -policy["component_loss"])
    weight = policy["continuous_weight"]
    if variant == "conditional_residual_legacy":
        mean = lower.mean(axis=1)
        utility = mean - weight * add / 5.
        branch = np.where(mean > 0, "BOP", "continuous")
        gates["mean_AR_floor"] = mean >= (-.001 if weight > 0 else 0)
        gates["positive_utility"] = utility > 0
    elif variant == "conditional_residual_joint":
        mean = p.lcb_delta_AR_mean.to_numpy()
        if weight not in [0., plan["joint_continuous_weight"]]:
            raise ValueError("Joint residual was not calibrated for this utility weight")
        utility = mean if weight == 0 else p.lcb_delta_joint_utility.to_numpy()
        bop = mean > 0
        continuous = ((weight > 0) & (add <= -plan["continuous_minimum_ADD_gain_mm"]) & (mean >= -.001))
        gates["BOP_or_continuous_branch"] = bop | continuous
        gates["positive_joint_utility"] = utility > 0
        branch = np.where(bop, "BOP", "continuous")
    else:
        raise ValueError(variant)
    gates["enabled_policy"] = np.full(len(data), not policy.get("disabled", False))
    gates["finite_scores"] = np.isfinite(utility) & np.isfinite(lower).all(axis=1) & np.isfinite(add)
    return gates, utility, branch


def decisions(data, p, policy, variant, plan):
    gates, utility, _ = gate_arrays(data, p, policy, variant, plan)
    allowed = np.logical_and.reduce(list(gates.values()))
    result = {}
    for target, indices in data.groupby("target_id", sort=True).groups.items():
        positions = data.index.get_indexer(indices)
        eligible = positions[allowed[positions]]
        if len(eligible):
            winner = min(eligible, key=lambda i: (-utility[i], p.iloc[i].adds_risk_upper, str(data.iloc[i].action)))
            result[target] = data.iloc[winner].action
        else:
            result[target] = base.INITIAL
    return result


def tune_policy(data, predictions, variant, plan):
    initial = base.baseline_rows(data, base.INITIAL).set_index("target_id")
    records = []
    limits = plan["inner_constraints"]
    for risk in plan["risk_limits"]:
        for loss in plan["component_ar_loss_limits"]:
            for weight in plan["continuous_gain_weights"]:
                policy = dict(risk_limit=risk, component_loss=loss, continuous_weight=weight, disabled=False)
                selected = v2.evaluate_choices(data, decisions(data, predictions, policy, variant, plan))
                active = selected.loc[selected.selected_action.ne(base.INITIAL)]
                coverage = len(active) / len(selected)
                error = float(active.delta_ADD_S_mm.gt(1e-9).mean()) if len(active) else 1.
                objective, checks = [], True
                for _, group in selected.groupby("cohort"):
                    gain = float((group.set_index("target_id").AR_mean_row - initial.loc[group.target_id, "AR_mean_row"]).mean())
                    delta = float(group.delta_ADD_S_mm.mean())
                    checks &= gain >= -1e-12 and delta <= 1e-12
                    objective.append(gain - .005 * delta / 5.)
                passes = (coverage >= limits["minimum_target_update_fraction"]
                          and error <= limits["maximum_regression_fraction_among_accepted"] and checks)
                records.append(dict(**policy, coverage=coverage, accepted_regression_rate=error,
                                    every_cohort_nonregressed=bool(checks),
                                    equal_cohort_joint_gain=float(np.mean(objective)),
                                    passes_development_constraints=bool(passes)))
    feasible = [row for row in records if row["passes_development_constraints"]]
    if not feasible:
        return dict(risk_limit=.1, component_loss=0., continuous_weight=0., disabled=True), records
    best = min(feasible, key=lambda r: (-r["equal_cohort_joint_gain"], r["accepted_regression_rate"], -r["coverage"], r["risk_limit"]))
    return {k: best[k] for k in ["risk_limit", "component_loss", "continuous_weight", "disabled"]}, records


def diagnostic_tables(data, predictions, policies, plan):
    coverage, gates_rows = [], []
    for variant in plan["variants"]:
        accepted = set()
        for fold, policy in policies[variant].items():
            held = data.loc[predictions.outer_fold.eq(fold)]
            p = predictions.loc[held.index]
            gates, _, _ = gate_arrays(held, p, policy, variant, plan)
            choices = decisions(held, p, policy, variant, plan)
            accepted.update((target, action) for target, action in choices.items() if action != base.INITIAL)
            for cohort, group in [("all", held), *list(held.groupby("cohort"))]:
                pos = held.index.get_indexer(group.index)
                cumulative = np.ones(len(group), dtype=bool)
                for gate, mask in gates.items():
                    before = cumulative.sum()
                    cumulative &= mask[pos]
                    gates_rows.append(dict(variant=variant, outer_fold=fold, cohort=cohort, gate=gate,
                                           rows=len(group), individual_passes=int(mask[pos].sum()),
                                           cumulative_passes=int(cumulative.sum()), first_rejections=int(before-cumulative.sum())))
        chosen = np.array([(t, a) in accepted for t, a in zip(data.target_id, data.action)])
        active = base.mask_fit(data)
        subsets = [("all_moving", active), ("accepted", chosen)]
        subsets += [(c, active & data.cohort.eq(c).to_numpy()) for c in data.cohort.unique()]
        subsets += [("support_low", active & predictions.training_support.lt(.35).to_numpy()),
                    ("support_mid", active & predictions.training_support.between(.35, .65, inclusive="left").to_numpy()),
                    ("support_high", active & predictions.training_support.ge(.65).to_numpy())]
        observed = {t: data[t] for t in base.TARGETS}
        observed["delta_AR_mean"] = data[base.TARGETS[:3]].mean(axis=1)
        observed["delta_joint_utility"] = observed["delta_AR_mean"] - plan["joint_continuous_weight"] * observed["delta_ADD_S_mm"] / 5.
        for scope, mask in subsets:
            for target in ALL_TARGETS:
                values = observed[target].to_numpy()[mask]
                bound = predictions[("ucb_" if target == "delta_ADD_S_mm" else "lcb_") + target].to_numpy()[mask]
                covered = values <= bound + 1e-12 if target == "delta_ADD_S_mm" else values >= bound - 1e-12
                coverage.append(dict(variant=variant, scope=scope, target=target, rows=int(np.sum(mask)),
                                     objects=data.loc[mask].object_id.nunique(),
                                     empirical_coverage=float(np.mean(covered)) if len(values) else np.nan,
                                     median_guard=float(predictions.loc[mask, "guard_" + target].median()) if len(values) else np.nan))
    return pd.DataFrame(coverage), pd.DataFrame(gates_rows)


def outcome_summary(outcomes):
    rows = []
    initializer = outcomes.loc[outcomes.method.eq(base.INITIAL)].set_index("target_id")
    for method, full in outcomes.groupby("method", sort=False):
        for scope, group in [("all", full), *list(full.groupby("cohort"))]:
            update = group.selected_action.ne(base.INITIAL)
            regression = group.delta_ADD_S_mm.gt(1e-9)
            row = dict(method=method, scope=scope, rows=len(group), objects=group.object_id.nunique(),
                       updates=int(update.sum()), update_fraction=float(update.mean()),
                       accepted_regressions=int((update & regression).sum()),
                       accepted_regression_rate=float(regression[update].mean()) if update.any() else np.nan,
                       all_regression_rate=float(regression.mean()),
                       ADD_S_mm=float(group.ADD_S_mm.mean()), ADD_delta_mm=float(group.delta_ADD_S_mm.mean()),
                       AR_mean=float(group.AR_mean_row.mean()),
                       AR_gain=float((group.set_index("target_id").AR_mean_row-initializer.loc[group.target_id, "AR_mean_row"]).mean()))
            for metric in base.AR:
                row[metric] = group[metric].mean()
                row[metric + "_gain"] = float((group.set_index("target_id")[metric]-initializer.loc[group.target_id, metric]).mean())
            rows.append(row)
    return pd.DataFrame(rows)


def paired_bootstrap(outcomes, variants):
    rng = np.random.default_rng(base.SEED)
    records = []
    for variant in variants:
        chosen = outcomes.loc[outcomes.method.eq(variant)]
        for reference in [base.INITIAL, "tail_selective_calibration"]:
            other = outcomes.loc[outcomes.method.eq(reference)]
            pair = chosen.merge(other, on=["target_id", "object_id"], suffixes=("_candidate", "_reference"), validate="one_to_one")
            for metric in ["ADD_S_mm", "AR_mean_row"] + base.AR:
                pair["delta"] = pair[metric+"_candidate"] - pair[metric+"_reference"]
                agg = pair.groupby("object_id").delta.agg(["sum", "count"])
                indices = rng.integers(0, len(agg), size=(2000, len(agg)))
                distribution = agg["sum"].to_numpy()[indices].sum(axis=1)/agg["count"].to_numpy()[indices].sum(axis=1)
                low, high = np.quantile(distribution, [.025, .975])
                records.append(dict(method=variant, reference=reference, metric=metric, delta=float(pair.delta.mean()),
                                    ci_low=float(low), ci_high=float(high), object_clusters=len(agg),
                                    scope="exploratory_reused_object_folds_not_confirmation"))
    return pd.DataFrame(records)


def verify_bundle(bundle, features, data, plan):
    clean = predict_bundle(bundle, features)
    first = predict_bundle(bundle, features.iloc[:1])
    np.testing.assert_allclose(clean.iloc[:1], first, atol=1e-9, rtol=1e-9)
    shadow = data.copy()
    shadow[base.TARGETS+base.CLASSES+base.AR+["ADD_S_mm", "initial_ADD_S_mm"]] = 99999.
    np.testing.assert_allclose(clean, predict_bundle(bundle, shadow), atol=1e-9, rtol=1e-9)
    for variant in plan["variants"]:
        policy = dict(risk_limit=.3, component_loss=.01, continuous_weight=.005, disabled=False)
        if decisions(data, clean, policy, variant, plan) != decisions(shadow, clean, policy, variant, plan):
            raise ValueError("Outcome-dependent acceptance")
    return dict(label_free_prediction=True, label_blind_decisions=True, batch_invariance=True)


def run(args):
    resume = bool(args.resume)
    if args.output.exists() and not resume:
        raise FileExistsError("Preserve existing experiment")
    if resume and not (args.output/"state.json").is_file():
        raise FileNotFoundError("Resume requires an existing checkpoint")
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    raw_plan = json.loads(args.raw_plan.read_text(encoding="utf-8"))
    previous_state_path = args.previous/"selective_training_state.json"
    previous = json.loads(previous_state_path.read_text(encoding="utf-8"))
    if previous["status"] != "COMPLETE_EXPLORATORY_NOT_PROMOTED":
        raise ValueError("Previous experiment incomplete")
    for relative, expected in previous["artifact_hashes"].items():
        if base.digest(args.previous/relative) != expected:
            raise ValueError("Previous artifact hash mismatch: " + relative)
    sources = [args.features, args.labels, args.raw_plan, args.plan, previous_state_path,
               Path(__file__), Path(base.__file__), Path(v2.__file__), Path(selective.__file__)]
    for path, expected in previous["source_hashes"].items():
        if Path(path) in sources and base.digest(Path(path)) != expected:
            raise ValueError("Frozen source changed: " + path)
    features = pd.read_csv(args.features)
    data = base.attach_labels(features, pd.read_csv(args.labels)).reset_index(drop=True)
    if len(data) != 3396 or data.object_id.nunique() != 22 or set(data.object_id) & set(plan["excluded_objects"]):
        raise ValueError("Development partition mismatch")
    names = joblib.load(args.previous/"development_candidate.joblib")["bundle"]["feature_names"]
    if resume:
        state = json.loads((args.output/"state.json").read_text(encoding="utf-8"))
        if state["status"] != "RUNNING" or "error" in state:
            raise ValueError("Only interrupted nonfailed RUNNING checkpoints may resume")
        for path in sources:
            if path != Path(__file__) and base.digest(path) != state["source_hashes"][str(path)]:
                raise ValueError("Resume source changed: " + str(path))
        if json.loads((args.output/"locked_plan.json").read_text(encoding="utf-8")) != plan:
            raise ValueError("Resume plan changed")
        state["transport_recovery"] = dict(resumed_at_utc=datetime.now(timezone.utc).isoformat(),
            completed_folds_reused=[r["fold"] for r in state["folds"]],
            prior_source_hash=state["source_hashes"][str(Path(__file__))],
            note="Execution handle lost; no Python process or exception record; reuse completed folds without refitting")
        state["source_hashes"][str(Path(__file__))] = base.digest(Path(__file__))
    else:
        args.output.mkdir(parents=True)
        base.dump(args.output/"locked_plan.json", plan)
        state = dict(status="RUNNING", started_at_utc=datetime.now(timezone.utc).isoformat(), folds=[],
                     source_hashes={str(p): base.digest(p) for p in sources}, method_promoted=False,
                     new_final_holdout=False, excluded_objects=plan["excluded_objects"])
    base.dump(args.output/"state.json", state)
    try:
        saved = pd.read_csv(args.previous/"outer_oof_candidate_predictions.csv")
        if not saved[["target_id", "action", "object_id"]].equals(data[["target_id", "action", "object_id"]]):
            raise ValueError("Saved OOF alignment mismatch")
        outputs, selections, grids, policies = [], [], [], {v: {} for v in plan["variants"]}
        start = time.monotonic()
        completed = {r["fold"]: r for r in state["folds"]}
        for fold, (train, valid) in enumerate(base.folds(data, 5)):
            training, held = data.iloc[train], data.iloc[valid]
            old = joblib.load(args.previous/f"outer_fold_{fold}.joblib")
            if sorted(map(int, held.object_id.unique())) != old["fold"]["validation_objects"]:
                raise ValueError("Outer split differs from frozen control")
            print(f"STATE RESIDUAL OUTER {fold+1}/5", flush=True)
            if fold in completed:
                bundle = joblib.load(args.output/f"outer_fold_{fold}.joblib")["bundle"]
                inner_predictions = pd.read_csv(args.output/f"outer_fold_{fold}_inner_predictions.csv", index_col="candidate_index").loc[training.index]
                print("  reusing completed model and inner predictions; no refit", flush=True)
            else:
                parts = []
                for inner, (tr, va) in enumerate(base.folds(training, 3)):
                    bundle = fit_bundle(training.iloc[tr], names, raw_plan, plan)
                    parts.append(predict_bundle(bundle, training.iloc[va]))
                    print(f"  inner {inner+1}/3 complete; elapsed {time.monotonic()-start:.1f}s", flush=True)
                inner_predictions = pd.concat(parts).sort_index().loc[training.index]
                bundle = fit_bundle(training, names, raw_plan, plan, existing=old["bundle"])
            p = predict_bundle(bundle, held)
            frozen = selective.predict_bundle(old["bundle"], held)
            for column in frozen:
                if column.startswith(("pred_", "raw_p_", "p_")) or column in ["training_support", "adds_risk_upper"]:
                    np.testing.assert_allclose(p[column], saved.loc[held.index, column], atol=1e-9, rtol=1e-9)
            p["outer_fold"] = fold
            outputs.append(p)
            fold_record = dict(fold=fold, training_objects=sorted(map(int, training.object_id.unique())),
                               validation_objects=sorted(map(int, held.object_id.unique())), policies={},
                               residual_offsets=bundle["residual_guard"]["offsets"], frozen_head_predictions_unchanged=True)
            for variant in plan["variants"]:
                policy, grid = tune_policy(training, inner_predictions, variant, plan)
                policies[variant][fold] = policy
                if fold in completed and policy != completed[fold]["policies"][variant]["policy"]:
                    raise ValueError("Recovery policy differs from saved fold")
                selected = v2.evaluate_choices(held, decisions(held, p, policy, variant, plan))
                selected["method"] = variant
                selected["outer_fold"] = fold
                selections.append(selected)
                fold_record["policies"][variant] = dict(policy=policy, updates=int(selected.selected_action.ne(base.INITIAL).sum()))
                grids += [dict(row, variant=variant, outer_fold=fold) for row in grid]
            if fold not in completed:
                joblib.dump(dict(bundle=bundle, fold=fold_record), args.output/f"outer_fold_{fold}.joblib")
                inner_predictions.to_csv(args.output/f"outer_fold_{fold}_inner_predictions.csv", index_label="candidate_index")
                state["folds"].append(fold_record)
            elif fold_record != completed[fold]:
                raise ValueError("Recovery fold record differs from saved checkpoint")
            base.dump(args.output/"state.json", state)
            print("OUTER COMPLETE", json.dumps(fold_record), flush=True)
        p = pd.concat(outputs).sort_index().loc[data.index]
        pd.concat([data[["target_id", "object_id", "cohort", "action", "pose_changed", "features_valid"]], p], axis=1).to_csv(args.output/"outer_oof_predictions.csv", index=False)
        prior_outcomes = pd.read_csv(args.previous/"all_variant_oof_outcomes.csv")
        keep = [base.INITIAL, "protocolF_se3_trust_p2p_s100", "protocolF_se3_trust_filtered_s100", "tail_selective_calibration"]
        outcomes = pd.concat([prior_outcomes.loc[prior_outcomes.method.isin(keep)], *selections], ignore_index=True)
        outcomes.to_csv(args.output/"all_variant_oof_outcomes.csv", index=False)
        summary = outcome_summary(outcomes)
        summary.to_csv(args.output/"state_residual_summary.csv", index=False)
        pd.DataFrame(grids).to_csv(args.output/"inner_policy_grid.csv", index=False)
        coverage, funnels = diagnostic_tables(data, p, policies, plan)
        coverage.to_csv(args.output/"residual_bound_coverage.csv", index=False)
        funnels.to_csv(args.output/"acceptance_funnels.csv", index=False)
        paired_bootstrap(outcomes, plan["variants"]).to_csv(args.output/"paired_object_bootstrap.csv", index=False)
        print("Fitting all-development residual guard; final policy remains exploratory", flush=True)
        old = joblib.load(args.previous/"development_candidate.joblib")["bundle"]
        final_bundle = fit_bundle(data, names, raw_plan, plan, existing=old)
        final_policies, final_grids = {}, []
        for variant in plan["variants"]:
            policy, grid = tune_policy(data, p, variant, plan)
            final_policies[variant] = policy
            final_grids += [dict(row, variant=variant) for row in grid]
        pd.DataFrame(final_grids).to_csv(args.output/"final_policy_grid_exploratory.csv", index=False)
        checks = verify_bundle(final_bundle, features, data, plan)
        joblib.dump(dict(bundle=final_bundle, policies=final_policies, status="EXPLORATORY_NOT_PROMOTED"),
                    args.output/"development_candidate.joblib")
        snapshot = args.output/"source_snapshot"
        snapshot.mkdir()
        for path in sources:
            if path.suffix in [".py", ".json"]:
                shutil.copy2(path, snapshot/path.name)
        state.update(status="COMPLETE_EXPLORATORY_NOT_PROMOTED", completed_at_utc=datetime.now(timezone.utc).isoformat(),
                     training_objects=sorted(map(int, data.object_id.unique())), final_policies=final_policies,
                     checks=checks, elapsed_seconds=time.monotonic()-start)
        state["artifact_hashes"] = {str(path.relative_to(args.output)): base.digest(path) for path in args.output.rglob("*")
                                    if path.is_file() and path.name != "state.json"}
        base.dump(args.output/"state.json", state)
        print(summary.loc[summary.scope.eq("all")|summary.scope.str.endswith("_mild")].to_string(index=False), flush=True)
        print("STATE RESIDUAL COMPLETE", flush=True)
    except Exception:
        state.update(status="FAILED_PRESERVED_NO_AUTOMATIC_RETRY", error=traceback.format_exc())
        base.dump(args.output/"state.json", state)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    root = Path(r"D:\6D_pose\codex_paper_workspace")
    parser.add_argument("--features", type=Path, default=root/"outputs/bop_aware_tail_risk_v2_20261019_r1/observable_features_with_tails.csv")
    parser.add_argument("--labels", type=Path, default=root/"outputs/bop_aware_development_labels_20261019_r1/development_bop_labels.csv")
    parser.add_argument("--previous", type=Path, default=root/"outputs/bop_aware_selective_calibration_20261019_r1")
    parser.add_argument("--raw-plan", type=Path, default=Path(__file__).with_name("bop_aware_tail_risk_v2_plan_20261019.json"))
    parser.add_argument("--plan", type=Path, default=Path(__file__).with_name("bop_aware_state_residual_plan_20261019.json"))
    parser.add_argument("--output", type=Path, default=root/"outputs/bop_aware_state_residual_20261019_r1")
    parser.add_argument("--resume", action="store_true", help="Reuse completed nonfailed checkpoints after execution-channel interruption")
    run(parser.parse_args())
