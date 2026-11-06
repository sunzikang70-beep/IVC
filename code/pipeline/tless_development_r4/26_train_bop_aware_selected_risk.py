"""Winner-only risk calibration with object-nested policy validation."""
from __future__ import annotations

import os
for _name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[_name] = "1"

import argparse
from datetime import datetime, timezone
from itertools import product
import json
from pathlib import Path
import shutil
import time
import traceback

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import train_bop_aware_state_residual as res

base, v2 = res.base, res.v2
VARIANT = "conditional_residual_legacy"
STATE_NAMES = ["common_support_fraction", "common_pixel_improved_fraction", "common_pixel_worsened_fraction",
               "tail_delta_silhouette_iou", "tail_delta_depth_abs_mm_p95", "translation_over_diameter",
               "action_step", "action_is_filtered"]
RISK_NAMES = ["candidate_risk_logit", "original_risk_upper", "training_support", "ADD_safety_margin",
              "mean_AR_lower", "minimum_AR_lower", "winner_utility", "runner_up_gap", "eligible_candidates",
              "branch_is_continuous", "policy_risk_limit", "policy_component_loss", "policy_continuous_weight"] + STATE_NAMES
META = ["candidate_index", "target_id", "object_id", "scene_id", "img_id", "inst_id", "frame_id",
        "cohort", "input_depth_condition", "action", "policy_id", "branch"]


def policy_grid(plan):
    return [dict(risk_limit=r, component_loss=l, continuous_weight=w, disabled=False)
            for r, l, w in product(plan["risk_limits"], plan["component_ar_loss_limits"], plan["continuous_gain_weights"])]


def winner_pool(data, predictions, plan):
    rows = []
    for policy_id, policy in enumerate(policy_grid(plan)):
        gates, utility, branch = res.gate_arrays(data, predictions, policy, VARIANT, {})
        allowed = np.logical_and.reduce(list(gates.values()))
        for target, indices in data.groupby("target_id", sort=True).groups.items():
            positions = data.index.get_indexer(indices)
            eligible = positions[allowed[positions]]
            if not len(eligible):
                continue
            ordered = sorted(eligible, key=lambda i: (-utility[i], predictions.iloc[i].adds_risk_upper, str(data.iloc[i].action)))
            i = ordered[0]
            d, p = data.iloc[i], predictions.iloc[i]
            lower = p[["lcb_"+t for t in base.TARGETS[:3]]].to_numpy(dtype=float)
            row = {k: d[k] for k in META if k in data.columns}
            row.update(candidate_index=int(data.index[i]), policy_id=policy_id, branch=branch[i],
                       candidate_risk_logit=float(base.logits(np.array([p.p_adds_regression])).ravel()[0]),
                       original_risk_upper=float(p.adds_risk_upper), training_support=float(p.training_support),
                       ADD_safety_margin=-float(p.ucb_delta_ADD_S_mm), mean_AR_lower=float(lower.mean()),
                       minimum_AR_lower=float(lower.min()), winner_utility=float(utility[i]),
                       runner_up_gap=float(utility[i]-utility[ordered[1]]) if len(ordered)>1 else 0.,
                       eligible_candidates=len(eligible), branch_is_continuous=float(branch[i]=="continuous"),
                       policy_risk_limit=policy["risk_limit"], policy_component_loss=policy["component_loss"],
                       policy_continuous_weight=policy["continuous_weight"])
            for name in STATE_NAMES:
                row[name] = float(d[name]) if name in data.columns else np.nan
            rows.append(row)
    return pd.DataFrame(rows, columns=META+RISK_NAMES)


def fit_weights(pool):
    weights = v2.object_frame_weights(pool)
    return weights * pool.frame_id.nunique()/weights.sum()


def risk_matrix(pool):
    return pool[RISK_NAMES].astype(float).replace([np.inf, -np.inf], np.nan).to_numpy()


def fit_probability(pool, plan):
    if pool.empty:
        return 1.
    w = fit_weights(pool)
    y = pool.outcome.to_numpy(dtype=int)
    if len(np.unique(y)) < 2:
        return float((np.dot(w, y)+.5)/(w.sum()+1.))
    model = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
                          LogisticRegression(C=plan["logistic_C"], max_iter=2000, random_state=base.SEED))
    model.fit(risk_matrix(pool), y, logisticregression__sample_weight=w)
    return model


def probability(model, pool):
    if pool.empty:
        return np.empty(0)
    return np.full(len(pool), model) if isinstance(model, float) else model.predict_proba(risk_matrix(pool))[:, 1]


def fit_selected_risk(data, predictions, plan):
    pool = winner_pool(data, predictions, plan)
    pool["outcome"] = data.loc[pool.candidate_index, "adds_regression"].to_numpy(dtype=int)
    training_objects = sorted(map(int, data.object_id.unique()))
    records = []
    if pool.empty or pool.object_id.nunique() < 3:
        return dict(model=1., bias_guard=0., training_objects=training_objects,
                    winner_training_objects=sorted(map(int, pool.object_id.unique())),
                    partitions=[], fallback="insufficient_winner_objects_reject", feature_names=RISK_NAMES), pool
    crossfit = pd.Series(index=pool.index, dtype=float)
    for train, valid in base.folds(pool, 3):
        fitting, held = pool.iloc[train], pool.iloc[valid]
        if set(fitting.object_id) & set(held.object_id):
            raise ValueError("Object leakage in winner risk crossfit")
        crossfit.loc[held.index] = probability(fit_probability(fitting, plan), held)
        records.append(dict(training_objects=sorted(map(int, fitting.object_id.unique())),
                            validation_objects=sorted(map(int, held.object_id.unique()))))
    pool["risk_probability_crossfit"] = crossfit
    pool["risk_residual"] = pool.outcome - crossfit
    biases = [float(np.average(group.risk_residual, weights=v2.object_frame_weights(group)))
              for _, group in pool.groupby("object_id")]
    guard = max(0., float(np.quantile(biases, .8)))
    model = dict(model=fit_probability(pool, plan), bias_guard=guard, training_objects=training_objects,
                 winner_training_objects=sorted(map(int, pool.object_id.unique())), partitions=records,
                 fallback=None, feature_names=RISK_NAMES)
    return model, pool


def predict_selected_risk(model, pool):
    result = pool.copy()
    p = probability(model["model"], pool)
    upper = np.minimum(1., p + model["bias_guard"])
    result["selected_probability"] = p
    result["selected_risk_upper"] = upper
    result["postselection_risk_score"] = np.maximum(result.original_risk_upper.to_numpy(dtype=float), upper)
    return result


def choices(data, winners, policy_id, policy, variant):
    selected = {target: base.INITIAL for target in data.target_id.unique()}
    if policy.get("disabled", False):
        return selected
    pool = winners.loc[winners.policy_id.eq(policy_id)]
    if variant != "stratified_only":
        pool = pool.loc[pool.postselection_risk_score.le(policy["risk_limit"]) & np.isfinite(pool.postselection_risk_score)]
    if pool.target_id.duplicated().any():
        raise ValueError("Multiple winners for a target/policy")
    selected.update(dict(zip(pool.target_id, pool.action)))
    return selected


def selected_frame(data, winners, policy_id, policy, variant):
    selected = v2.evaluate_choices(data, choices(data, winners, policy_id, policy, variant))
    context = winners.loc[winners.policy_id.eq(policy_id), ["target_id", "branch"]]
    return selected.merge(context, on="target_id", how="left", validate="one_to_one")


def stratum_statistics(selected, variant, policy_id, scope):
    active = selected.loc[selected.selected_action.ne(base.INITIAL)].copy()
    records = []
    for axis in ["cohort", "input_depth_condition", "branch"]:
        for value, group in active.groupby(axis):
            bad = group.delta_ADD_S_mm.gt(1e-9).to_numpy()
            records.append(dict(variant=variant, policy_id=policy_id, scope=scope, axis=axis, stratum=str(value),
                                accepted=len(group), regressed=int(bad.sum()), objects=group.object_id.nunique(),
                                physical_instances=group.frame_id.nunique(), row_risk=float(bad.mean()),
                                balanced_risk=float(np.average(bad, weights=v2.object_frame_weights(group)))))
    return records


def coverage_preserved(selected, reference, plan):
    limits = plan["stratified_constraints"]
    records = []
    for cohort, group in selected.groupby("cohort"):
        old = reference.loc[reference.cohort.eq(cohort)].selected_action.ne(base.INITIAL).sum()
        new = group.selected_action.ne(base.INITIAL).sum()
        required = (int(np.ceil(limits["minimum_retained_reference_cohort_update_fraction"]*old))
                    if old >= limits["minimum_reference_updates_for_coverage_floor"] else 0)
        records.append(dict(cohort=cohort, reference_updates=int(old), updates=int(new), minimum_updates=required, passes=bool(new>=required)))
    return all(r["passes"] for r in records), records


def tune_policy(data, predictions, winners, variant, plan, residual_plan):
    reference_policy, _ = res.tune_policy(data, predictions, VARIANT, residual_plan)
    reference = v2.evaluate_choices(data, res.decisions(data, predictions, reference_policy, VARIANT, residual_plan))
    initial = base.baseline_rows(data, base.INITIAL).set_index("target_id")
    records, strata = [], []
    for policy_id, policy in enumerate(policy_grid(plan)):
        selected = selected_frame(data, winners, policy_id, policy, variant)
        active = selected.loc[selected.selected_action.ne(base.INITIAL)]
        coverage = len(active)/len(selected)
        risk = float(active.delta_ADD_S_mm.gt(1e-9).mean()) if len(active) else 1.
        nonregressed, objective = True, []
        for _, group in selected.groupby("cohort"):
            gain = float((group.set_index("target_id").AR_mean_row-initial.loc[group.target_id, "AR_mean_row"]).mean())
            delta = float(group.delta_ADD_S_mm.mean())
            nonregressed &= gain >= -1e-12 and delta <= 1e-12
            objective.append(gain - .005*delta/5.)
        rows = stratum_statistics(selected, variant, policy_id, "inner_policy")
        strata.extend(rows)
        limits = plan["stratified_constraints"]
        stratified = all(r["row_risk"] <= limits["maximum_row_regression_rate"]+1e-12
                         and r["balanced_risk"] <= limits["maximum_object_frame_balanced_regression_rate"]+1e-12 for r in rows)
        preserved, retention = coverage_preserved(selected, reference, plan)
        pooled = (coverage >= plan["pooled_constraints"]["minimum_target_update_fraction"]
                  and risk <= plan["pooled_constraints"]["maximum_accepted_regression_rate"] and nonregressed)
        passes = pooled and (variant == "selected_risk_only" or (stratified and preserved))
        records.append(dict(policy_id=policy_id, **policy, coverage=coverage, accepted=len(active),
                            accepted_regression_rate=risk, every_cohort_nonregressed=bool(nonregressed),
                            stratum_constraints=bool(stratified), reference_cohort_coverage_preserved=bool(preserved),
                            reference_coverage_details=json.dumps(retention),
                            equal_cohort_joint_gain=float(np.mean(objective)), passes_development_constraints=bool(passes)))
    feasible = [r for r in records if r["passes_development_constraints"]]
    if not feasible:
        return dict(policy_id=0, risk_limit=.1, component_loss=0., continuous_weight=0., disabled=True), records, strata
    best = min(feasible, key=lambda r: (-r["equal_cohort_joint_gain"], r["accepted_regression_rate"], -r["coverage"], r["risk_limit"]))
    return {k: best[k] for k in ["policy_id", "risk_limit", "component_loss", "continuous_weight", "disabled"]}, records, strata


def predict_deployment(bundle, features, plan, variant):
    if "target_id" not in features:
        features = features.copy()
        keys = (["cohort"] if "cohort" in features else []) + base.KEYS
        features["target_id"] = features.groupby(keys, sort=False, dropna=False).ngroup()
    predictions = res.predict_bundle(bundle["base_bundle"], features)
    pool = predict_selected_risk(bundle["selected_risk"], winner_pool(features, predictions, plan))
    policy = bundle["policies"][variant]
    return choices(features, pool, policy["policy_id"], policy, variant), pool


def check_manifest(directory, state):
    for relative, expected in state["artifact_hashes"].items():
        if base.digest(directory/relative) != expected:
            raise ValueError("Frozen artifact hash mismatch: "+relative)
    for path, expected in state["source_hashes"].items():
        if base.digest(Path(path)) != expected:
            raise ValueError("Frozen source hash mismatch: "+path)


def run(args):
    if args.output.exists():
        raise FileExistsError("Preserve previous experiment")
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    oldstate = json.loads((args.previous/"state.json").read_text(encoding="utf-8"))
    if oldstate["status"] != "COMPLETE_EXPLORATORY_NOT_PROMOTED":
        raise ValueError("Previous state incomplete")
    check_manifest(args.previous, oldstate)
    residual_plan = json.loads(args.residual_plan.read_text(encoding="utf-8"))
    raw_plan = json.loads(args.raw_plan.read_text(encoding="utf-8"))
    features = pd.read_csv(args.features)
    data = base.attach_labels(features, pd.read_csv(args.labels)).reset_index(drop=True)
    if len(data) != 3396 or data.object_id.nunique() != 22 or set(data.object_id)&set(plan["excluded_objects"]):
        raise ValueError("Development partition changed")
    original_p = pd.read_csv(args.previous/"outer_oof_predictions.csv")
    if not original_p[["target_id", "object_id", "action"]].equals(data[["target_id", "object_id", "action"]]):
        raise ValueError("OOF metadata alignment mismatch")
    names = joblib.load(args.previous/"development_candidate.joblib")["bundle"]["feature_names"]
    sources = [Path(__file__), Path(res.__file__), Path(base.__file__), Path(v2.__file__),
               Path(res.selective.__file__), args.plan, args.raw_plan, args.residual_plan,
               args.features, args.labels, args.previous/"state.json"]
    args.output.mkdir(parents=True)
    base.dump(args.output/"locked_plan.json", plan)
    state = dict(status="RUNNING", folds=[], started_at_utc=datetime.now(timezone.utc).isoformat(),
                 source_hashes={str(p): base.digest(p) for p in sources}, method_promoted=False, new_final_holdout=False)
    cache_hashes = {}
    if args.reuse_interrupted_output:
        cache_state = json.loads((args.reuse_interrupted_output/"state.json").read_text(encoding="utf-8"))
        if cache_state["status"] != "FAILED_PRESERVED_NO_AUTOMATIC_RETRY" or "bool_ is not JSON serializable" not in cache_state.get("error", ""):
            raise ValueError("Cache reuse is restricted to the explicitly approved recording failure")
        for path, expected in cache_state["source_hashes"].items():
            if Path(path) != Path(__file__) and base.digest(Path(path)) != expected:
                raise ValueError("Interrupted cache source changed: "+path)
        preserved_source = args.reuse_interrupted_output/"failed_source_before_record_fix.py"
        if base.digest(preserved_source) != cache_state["source_hashes"][str(Path(__file__))]:
            raise ValueError("Interrupted experiment source not preserved")
        for path in args.reuse_interrupted_output.rglob("*"):
            if path.is_file():
                cache_hashes[str(path)] = base.digest(path)
        state["approved_cache_reuse"] = dict(source=str(args.reuse_interrupted_output),
            approved_by_user=True, reused_models=[], frozen_cache_hashes=cache_hashes,
            change="native bool serialization and observable-only deployment target grouping; scientific plan unchanged")
    base.dump(args.output/"state.json", state)
    start = time.monotonic()
    try:
        selections, all_winners, grid_rows, stratum_rows, nested_records = [], [], [], [], []
        for fold, (train, valid) in enumerate(base.folds(data, 5)):
            training, held = data.iloc[train], data.iloc[valid]
            directory = args.output/f"fold_{fold}"
            directory.mkdir()
            print(f"SELECTED RISK OUTER {fold+1}/5", flush=True)
            inner_base = pd.read_csv(args.previous/f"outer_fold_{fold}_inner_predictions.csv", index_col="candidate_index").loc[training.index]
            inner_risk_parts = []
            for inner, (tr, va) in enumerate(base.folds(training, 3)):
                fitting, query = training.iloc[tr], training.iloc[va]
                deep_parts = []
                for deep, (a, b) in enumerate(base.folds(fitting, 3)):
                    deep_training, deep_held = fitting.iloc[a], fitting.iloc[b]
                    if set(deep_training.object_id)&set(deep_held.object_id) or set(fitting.object_id)&set(query.object_id):
                        raise ValueError("Nested calibration object leakage")
                    cached_path = (args.reuse_interrupted_output/f"fold_{fold}/inner_{inner}_deep_{deep}.joblib"
                                   if args.reuse_interrupted_output else None)
                    if cached_path is not None and cached_path.is_file():
                        cached = joblib.load(cached_path)
                        bundle = cached["bundle"]
                        if bundle["training_objects"] != sorted(map(int, deep_training.object_id.unique())) or cached["prediction_index"] != list(map(int, deep_held.index)):
                            raise ValueError("Cached nested model partition mismatch")
                        predicted = res.predict_bundle(bundle, deep_held)
                        saved = pd.read_csv(cached_path.with_name(f"inner_{inner}_calibration_oof.csv"), index_col="candidate_index")
                        np.testing.assert_allclose(predicted, saved.loc[deep_held.index, predicted.columns], atol=1e-9, rtol=1e-9)
                        state["approved_cache_reuse"]["reused_models"].append(str(cached_path))
                    else:
                        bundle = res.fit_bundle(deep_training, names, raw_plan, residual_plan)
                        predicted = res.predict_bundle(bundle, deep_held)
                    deep_parts.append(predicted)
                    path = directory/f"inner_{inner}_deep_{deep}.joblib"
                    joblib.dump(dict(bundle=bundle, prediction_index=list(map(int, deep_held.index))), path)
                    nested_records.append(dict(outer_fold=fold, inner_fold=inner, deep_fold=deep,
                        training_objects=sorted(map(int, deep_training.object_id.unique())),
                        validation_objects=sorted(map(int, deep_held.object_id.unique())),
                        policy_validation_objects=sorted(map(int, query.object_id.unique())),
                        outer_validation_objects=sorted(map(int, held.object_id.unique()))))
                    print(f"  inner {inner+1}/3 calibration OOF {deep+1}/3 complete; {time.monotonic()-start:.1f}s", flush=True)
                deep_p = pd.concat(deep_parts).sort_index().loc[fitting.index]
                deep_p.to_csv(directory/f"inner_{inner}_calibration_oof.csv", index_label="candidate_index")
                risk_model, training_winners = fit_selected_risk(fitting, deep_p, plan)
                training_winners.to_csv(directory/f"inner_{inner}_training_winners.csv", index=False)
                joblib.dump(risk_model, directory/f"inner_{inner}_risk.joblib")
                inner_scores = predict_selected_risk(risk_model, winner_pool(query, inner_base.loc[query.index], plan))
                inner_scores["inner_fold"] = inner
                inner_risk_parts.append(inner_scores)
            inner_risk = pd.concat(inner_risk_parts, ignore_index=True)
            inner_risk.to_csv(directory/"inner_validation_winners.csv", index=False)
            final_risk, outer_training_pool = fit_selected_risk(training, inner_base, plan)
            outer_training_pool.to_csv(directory/"outer_training_winners.csv", index=False)
            outer_scores = predict_selected_risk(final_risk, winner_pool(held, original_p.loc[held.index], plan))
            outer_scores["outer_fold"] = fold
            all_winners.append(outer_scores)
            record = dict(fold=fold, training_objects=sorted(map(int, training.object_id.unique())),
                          validation_objects=sorted(map(int, held.object_id.unique())),
                          risk_bias_guard=final_risk["bias_guard"], training_policy_winner_rows=len(outer_training_pool),
                          training_unique_physical_winners=outer_training_pool.frame_id.nunique(), policies={})
            for variant in plan["variants"]:
                policy, grid, strata = tune_policy(training, inner_base, inner_risk, variant, plan, residual_plan)
                selected = selected_frame(held, outer_scores, policy["policy_id"], policy, variant)
                selected["method"], selected["outer_fold"] = variant, fold
                selections.append(selected)
                record["policies"][variant] = dict(policy=policy, accepted=int(selected.selected_action.ne(base.INITIAL).sum()))
                grid_rows.extend(dict(row, variant=variant, outer_fold=fold) for row in grid)
                stratum_rows.extend(dict(row, outer_fold=fold) for row in strata)
                stratum_rows.extend(dict(row, outer_fold=fold) for row in stratum_statistics(selected, variant, policy["policy_id"], "outer_result"))
            joblib.dump(dict(selected_risk=final_risk, fold=record), directory/"outer_model.joblib")
            outer_scores.to_csv(directory/"outer_winners.csv", index=False)
            state["folds"].append(record)
            base.dump(args.output/"state.json", state)
            base.dump(args.output/"nested_object_partitions.json", nested_records)
            print("OUTER COMPLETE", json.dumps(record), flush=True)
        previous = pd.read_csv(args.previous/"all_variant_oof_outcomes.csv")
        keep = [base.INITIAL, "tail_selective_calibration", "conditional_residual_legacy"]
        outcomes = pd.concat([previous.loc[previous.method.isin(keep)], *selections], ignore_index=True)
        outcomes.to_csv(args.output/"all_variant_oof_outcomes.csv", index=False)
        summary = res.outcome_summary(outcomes)
        summary.to_csv(args.output/"selected_risk_summary.csv", index=False)
        pd.DataFrame(grid_rows).to_csv(args.output/"inner_policy_grid.csv", index=False)
        pd.DataFrame(stratum_rows).to_csv(args.output/"stratified_risk_tables.csv", index=False)
        winners = pd.concat(all_winners, ignore_index=True)
        winners.to_csv(args.output/"outer_oof_winners.csv", index=False)
        print("Fitting all-development winner calibrator (exploratory)", flush=True)
        final_risk, full_pool = fit_selected_risk(data, original_p, plan)
        full_pool.to_csv(args.output/"all_development_training_winners.csv", index=False)
        policies, final_grids = {}, []
        for variant in plan["variants"]:
            policy, grid, _ = tune_policy(data, original_p, winners, variant, plan, residual_plan)
            policies[variant] = policy
            final_grids.extend(dict(row, variant=variant) for row in grid)
        pd.DataFrame(final_grids).to_csv(args.output/"final_policy_grid_exploratory.csv", index=False)
        deployment = dict(base_bundle=joblib.load(args.previous/"development_candidate.joblib")["bundle"],
                          selected_risk=final_risk, policies=policies, status="EXPLORATORY_NOT_PROMOTED")
        joblib.dump(deployment, args.output/"development_candidate.joblib")
        shadow = data.copy()
        shadow[base.TARGETS+base.CLASSES+base.AR+["ADD_S_mm", "initial_ADD_S_mm"]] = 99999.
        for variant in plan["variants"]:
            clean, pool = predict_deployment(deployment, features, plan, variant)
            hidden, shadow_pool = predict_deployment(deployment, shadow, plan, variant)
            if clean != hidden or not np.allclose(pool[RISK_NAMES+["postselection_risk_score"]], shadow_pool[RISK_NAMES+["postselection_risk_score"]], equal_nan=True):
                raise ValueError("GT-dependent winner risk prediction")
        snapshot = args.output/"source_snapshot"
        snapshot.mkdir()
        for path in sources:
            if path.suffix in [".py", ".json"]:
                shutil.copy2(path, snapshot/path.name)
        state.update(status="COMPLETE_EXPLORATORY_NOT_PROMOTED", final_policies=policies,
                     checks=dict(label_blind_deployment=True, nested_policy_calibration=True, no_runner_up_reselection=True),
                     completed_at_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic()-start)
        state["artifact_hashes"] = {str(p.relative_to(args.output)): base.digest(p) for p in args.output.rglob("*")
                                    if p.is_file() and p != args.output/"state.json"}
        base.dump(args.output/"state.json", state)
        print(summary.loc[summary.scope.eq("all") | summary.scope.str.endswith("_mild")].to_string(index=False), flush=True)
        print("SELECTED RISK COMPLETE", flush=True)
    except Exception:
        state.update(status="FAILED_PRESERVED_NO_AUTOMATIC_RETRY", error=traceback.format_exc())
        base.dump(args.output/"state.json", state)
        raise


if __name__ == "__main__":
    root = Path(r"D:\6D_pose\codex_paper_workspace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--previous", type=Path, default=root/"outputs/bop_aware_state_residual_20261019_r1")
    parser.add_argument("--features", type=Path, default=root/"outputs/bop_aware_tail_risk_v2_20261019_r1/observable_features_with_tails.csv")
    parser.add_argument("--labels", type=Path, default=root/"outputs/bop_aware_development_labels_20261019_r1/development_bop_labels.csv")
    parser.add_argument("--residual-plan", type=Path, default=Path(__file__).with_name("bop_aware_state_residual_plan_20261019.json"))
    parser.add_argument("--raw-plan", type=Path, default=Path(__file__).with_name("bop_aware_tail_risk_v2_plan_20261019.json"))
    parser.add_argument("--plan", type=Path, default=Path(__file__).with_name("bop_aware_selected_risk_plan_20261019.json"))
    parser.add_argument("--output", type=Path, default=root/"outputs/bop_aware_selected_risk_20261019_r1")
    parser.add_argument("--reuse-interrupted-output", type=Path, help="User-approved reuse of the preserved JSON recording failure only")
    run(parser.parse_args())
