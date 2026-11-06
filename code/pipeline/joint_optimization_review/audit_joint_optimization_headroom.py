"""Read-only, post-hoc r4 diagnosis. Oracles use GT and are not methods."""
from __future__ import annotations

import os
for name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[name] = "1"

import argparse
from datetime import datetime, timezone
from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import run_same_bank_residual_ablation as ab

base, res = ab.base, ab.res
POLICY = dict(risk_limit=.3, component_loss=.01, continuous_weight=.005, disabled=False)
PLAN_PATH = Path(__file__).with_name("bop_aware_state_residual_plan_20261019.json")
PRIOR = ab.ROOT / "outputs/same_bank_residual_ablation_20261026_r1/per_target_outcomes.csv"


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def best_choices(data, allowed, score, risk=None):
    ranked = data[["target_id", "action", "delta_ADD_S_mm"]].copy()
    ranked["score"] = np.asarray(score)
    ranked = ranked.loc[np.asarray(allowed) & (ranked.score > 1e-12)]
    if risk is None:
        tie = "delta_ADD_S_mm"
    else:
        ranked["predicted_risk"] = np.asarray(risk)[ranked.index]
        tie = "predicted_risk"
    ranked = ranked.sort_values(["score", tie, "action"], ascending=[False, True, True], kind="stable")
    winners = ranked.drop_duplicates("target_id").set_index("target_id").action.to_dict()
    return {int(t): winners.get(t, base.INITIAL) for t in data.target_id.unique()}


def summarize(rows, dataset, method):
    result = []
    for scope, group in ab.scopes(rows):
        obj = group.groupby("object_id")[["AR_mean_row", "AR_gain", "delta_ADD_S_mm"]].mean()
        active = group.loc[group["update"].eq(1)]
        result.append(dict(dataset=dataset, scope=scope, method=method, targets=len(group), objects=len(obj),
            AR_macro=float(obj.AR_mean_row.mean()), AR_pooled=float(group.AR_mean_row.mean()),
            gain_macro=float(obj.AR_gain.mean()), gain_pooled=float(group.AR_gain.mean()),
            ADD_delta=float(group.delta_ADD_S_mm.mean()), updates=len(active),
            update_fraction=float(group["update"].mean()),
            accepted_ADD_risk=float(active.regression.mean()) if len(active) else np.nan,
            all_target_ADD_risk=float(group.regression.mean()),
            all_target_BOP_risk=float(group.bop_regression.mean())))
    return result


def auc(y, score):
    return float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else np.nan


def pair_ranking(data, score, mask):
    pairs, correct, ties = 0, 0., 0
    for _, group in data.loc[mask].groupby("target_id"):
        idx = group.index.to_numpy()
        for i in range(len(idx)):
            for j in range(i + 1, len(idx)):
                actual = data.loc[idx[i], "actual_gain"] - data.loc[idx[j], "actual_gain"]
                if abs(actual) <= 1e-12:
                    continue
                delta = score.loc[idx[i]] - score.loc[idx[j]]
                pairs += 1
                ties += int(abs(delta) <= 1e-12)
                correct += .5 if abs(delta) <= 1e-12 else float(actual * delta > 0)
    return dict(distinguishable_pairs=pairs, accuracy=correct / pairs if pairs else np.nan, score_ties=ties)


def diagnose(data, p, dataset, plan, prior):
    data = data.copy()
    data["actual_gain"] = data[base.TARGETS[:3]].mean(axis=1)
    gates, utility, _ = res.gate_arrays(data, p, POLICY, "conditional_residual_joint", plan)
    allowed = np.logical_and.reduce(list(gates.values()))
    basic = gates["moving_valid"] & gates["observation"]
    raw_utility = p[["pred_" + t for t in base.TARGETS[:3]]].mean(axis=1) - .001 * p.pred_delta_ADD_S_mm
    safe = basic & data.delta_ADD_S_mm.le(1e-9).to_numpy() & data[base.TARGETS[:3]].min(axis=1).ge(-.01 - 1e-12).to_numpy()
    oracle_masks = dict(GT_oracle_all=np.ones(len(data), bool), GT_oracle_observable=basic,
                        GT_oracle_ADD_component_safe=safe, GT_oracle_inside_joint_gates=allowed)
    methods = {name: best_choices(data, mask, data.actual_gain) for name, mask in oracle_masks.items()}
    methods["initializer"] = {int(t): base.INITIAL for t in data.target_id.unique()}
    methods["D_joint_full"] = res.decisions(data, p, POLICY, "conditional_residual_joint", plan)
    ap = p.copy()
    for name in base.TARGETS:
        ap[("ucb_" if name == "delta_ADD_S_mm" else "lcb_") + name] = p["pred_" + name]
    methods["A_delta_rank"] = ab.choose(data, ap, POLICY, "A_delta_rank", plan)
    methods["diagnostic_joint_score_basic_gates"] = best_choices(data, basic, utility, p.adds_risk_upper)
    methods["diagnostic_raw_score_basic_gates"] = best_choices(data, basic, raw_utility, p.adds_risk_upper)
    for gate in gates:
        if gate in ["moving_valid", "observation", "finite_scores", "enabled_policy", "positive_joint_utility"]:
            continue
        remaining = np.logical_and.reduce([v for k, v in gates.items() if k != gate])
        methods["diagnostic_remove_" + gate] = best_choices(data, remaining, utility, p.adds_risk_upper)
        methods["GT_oracle_remove_" + gate] = best_choices(data, remaining, data.actual_gain)
    outcomes, summaries = {}, []
    for name, choices in methods.items():
        rows = ab.evaluate(data, choices)
        rows["dataset"], rows["method"] = dataset, name
        outcomes[name] = rows
        summaries += summarize(rows, dataset, name)
    keys = ["cohort", *base.KEYS]
    for arm in ["D_joint_full", "A_delta_rank"]:
        old = prior.loc[(prior.dataset == dataset) & (prior.policy_mode == "shared_r4") & (prior.arm == arm)]
        merged = outcomes[arm].merge(old, on=keys, suffixes=("_now", "_old"), validate="one_to_one")
        assert len(merged) == len(outcomes[arm])
        assert merged.selected_action_now.tolist() == merged.selected_action_old.tolist()
        np.testing.assert_allclose(merged.AR_mean_row_now, merged.AR_mean_row_old, atol=1e-12, rtol=0)

    target = outcomes["D_joint_full"].copy()
    for method in oracle_masks:
        values = outcomes[method].set_index("target_id")
        target[method + "_gain"] = target.target_id.map(values.AR_gain)
    target["gate_blocking_gap"] = target.GT_oracle_all_gain - target.GT_oracle_inside_joint_gates_gain
    target["ranking_gap_inside_gates"] = target.GT_oracle_inside_joint_gates_gain - target.AR_gain
    target["total_oracle_gap"] = target.GT_oracle_all_gain - target.AR_gain
    np.testing.assert_allclose(target.gate_blocking_gap + target.ranking_gap_inside_gates, target.total_oracle_gap, atol=1e-12)
    assert target.gate_blocking_gap.min() >= -1e-12 and target.ranking_gap_inside_gates.min() >= -1e-12

    candidate = data[["target_id", "frame_id", "object_id", "cohort", "init_regime", "input_depth_condition", "action",
                      "actual_gain", *base.TARGETS, "actual_translation_change_mm", "actual_rotation_change_deg", "action_step"]].copy()
    candidate["dataset"] = dataset
    candidate["raw_utility"], candidate["joint_utility"] = np.asarray(raw_utility), utility
    candidate["training_support"] = p.training_support
    candidate["risk_score"] = p.adds_risk_upper
    candidate["weak_support"] = p.training_support.lt(.35)
    candidate["all_gates"] = allowed
    for name, values in gates.items():
        candidate["gate_" + name] = values
    choices = methods["D_joint_full"]
    candidate["chosen"] = [choices[int(t)] == action for t, action in zip(data.target_id, data.action)]
    for name in res.ALL_TARGETS:
        candidate["guard_" + name] = p["guard_" + name]
        bound = ("ucb_" if name == "delta_ADD_S_mm" else "lcb_") + name
        candidate[bound] = p[bound]
    gate_stats, rank_stats, risk_stats, guard_stats, candidate_stats = [], [], [], [], []
    best = best_choices(data, np.ones(len(data), bool), data.actual_gain)
    candidate["GT_best_positive"] = [best[int(t)] == action for t, action in zip(data.target_id, data.action)]
    for scope, group in ab.scopes(candidate):
        idx = group.index
        valid = group.loc[gates["moving_valid"][idx]]
        good = group.actual_gain.gt(1e-12)
        for gate in gates:
            others = np.logical_and.reduce([v[idx] for k, v in gates.items() if k != gate])
            gate_stats.append(dict(dataset=dataset, scope=scope, gate=gate, candidates=len(group),
                fail=int((~gates[gate][idx]).sum()), positive_candidates=int(good.sum()),
                positive_blocked=int((good & ~gates[gate][idx]).sum()),
                sole_block_positive=int((good & ~gates[gate][idx] & others).sum()),
                GT_best_blocked=int((group.GT_best_positive & ~gates[gate][idx]).sum())))
        for name, scores in [("raw_joint_utility", pd.Series(raw_utility, index=data.index)),
                             ("guarded_joint_utility", pd.Series(utility, index=data.index))]:
            rank_stats.append(dict(dataset=dataset, scope=scope, score=name,
                **pair_ranking(data.loc[idx], scores, pd.Series(basic, index=data.index).loc[idx])))
        for kind, subset in [("all_valid_candidates", valid), ("joint_chosen", group.loc[group.chosen]),
                             ("joint_gate_eligible", group.loc[group.all_gates])]:
            y = subset.delta_ADD_S_mm.gt(1e-9)
            risk_stats.append(dict(dataset=dataset, scope=scope, subset=kind, rows=len(subset),
                ADD_regression_prevalence=float(y.mean()) if len(y) else np.nan,
                mean_predicted_risk=float(subset.risk_score.mean()),
                ADD_regression_AUC=auc(y, subset.risk_score),
                positive_ADD_median=float(subset.loc[y, "delta_ADD_S_mm"].median()),
                positive_ADD_p95=float(subset.loc[y, "delta_ADD_S_mm"].quantile(.95)),
                positive_ADD_over_025mm=int((subset.delta_ADD_S_mm > .25).sum()),
                positive_ADD_over_1mm=int((subset.delta_ADD_S_mm > 1).sum()),
                BOP_regression_fraction=float(subset.actual_gain.lt(-1e-12).mean()) if len(subset) else np.nan,
                BOP_loss_over_0333=int((subset.actual_gain < -1 / 30 - 1e-12).sum())))
        for target_name in res.ALL_TARGETS:
            observed = group.actual_gain if target_name == "delta_AR_mean" else (
                group.actual_gain - .001 * group.delta_ADD_S_mm if target_name == "delta_joint_utility" else group[target_name])
            bound = group[("ucb_" if target_name == "delta_ADD_S_mm" else "lcb_") + target_name]
            covered = observed.le(bound + 1e-12) if target_name == "delta_ADD_S_mm" else observed.ge(bound - 1e-12)
            for kind, sel in [("all_valid_candidates", gates["moving_valid"][idx]), ("joint_chosen", group.chosen.to_numpy())]:
                guard = group.loc[sel, "guard_" + target_name]
                guard_stats.append(dict(dataset=dataset, scope=scope, target=target_name, subset=kind, rows=int(np.sum(sel)),
                    coverage=float(covered.loc[sel].mean()), guard_median=float(guard.median()),
                    guard_p90=float(guard.quantile(.9)), zero_guard_fraction=float(guard.le(1e-12).mean())))
        candidate_stats.append(dict(dataset=dataset, scope=scope, targets=group.target_id.nunique(), candidates=len(group),
            moving_valid_fraction=float(gates["moving_valid"][idx].mean()), weak_support_fraction=float(group.weak_support.mean()),
            outside_support_fraction=float(group.training_support.lt(.2).mean()),
            translation_at_cap_fraction=float((valid.actual_translation_change_mm >= 5 * valid.action_step * .999).mean()),
            rotation_at_cap_fraction=float((valid.actual_rotation_change_deg >= valid.action_step * .999).mean()),
            target_all_AR_deltas_zero_fraction=float(group.groupby("target_id").actual_gain.apply(lambda v: v.abs().max() <= 1e-12).mean())))
    return dict(outcomes=pd.concat(outcomes.values(), ignore_index=True), summary=pd.DataFrame(summaries),
        targets=target, candidates=candidate, gates=pd.DataFrame(gate_stats), rankings=pd.DataFrame(rank_stats),
        risks=pd.DataFrame(risk_stats), guards=pd.DataFrame(guard_stats), candidate_stats=pd.DataFrame(candidate_stats))


def run(output):
    if output.exists():
        raise FileExistsError(output)
    plan = ab.load(PLAN_PATH)
    files = [Path(__file__), PLAN_PATH, PRIOR, ab.OOF / "development_candidate.joblib", ab.OOF / "outer_oof_predictions.csv"]
    for folder, development in [(ab.DEV, True), *[(v, False) for v in ab.BANKS.values()]]:
        files += [folder / n for n in ["observable_candidate_features.csv", "locked_development_instances.csv" if development else "locked_holdout_instances.csv",
                    "development_bop_labels.csv" if development else "holdout_bop_labels.csv"]]
    files += [Path(m.__file__) for m in [ab, base, res, res.v2, res.selective]]
    files += [Path(__file__).with_name("test_joint_optimization_headroom.py")]
    hashes = {str(p): base.digest(p) for p in files}
    assert hashes[str(ab.OOF / "development_candidate.joblib")] == "a3836d3bfbb0dda4927578fdfb53389830fe0cff2bdc439f46de8cdcb2a5917c"
    output.mkdir(parents=True)
    dump(output / "input_hashes.json", hashes)
    dump(output / "scope_lock.json", dict(status="POSTHOC_DIAGNOSIS_ONLY", policy=POLICY, train=False,
        tune=False, candidate_generation=False, independent_confirmation=False, method_promotion=False,
        GT_oracles="unattainable reference ceilings; initializer always available", estimand="equal_object_AR",
        decomposition="oracle_all - D = (oracle_all - oracle_inside_gates) + (oracle_inside_gates - D)",
        output=str(output), started_at=datetime.now(timezone.utc).isoformat()))
    start = time.monotonic()
    frozen = joblib.load(ab.OOF / "development_candidate.joblib")
    assert frozen["policies"]["conditional_residual_joint"] == POLICY
    prior = ab.read_csv(PRIOR)
    results = []
    for dataset, folder, development in [("development22", ab.DEV, True), *[(k, v, False) for k, v in ab.BANKS.items()]]:
        data, _, _ = ab.bank(folder, development)
        p = ab.read_csv(ab.OOF / "outer_oof_predictions.csv", index_col="candidate_index") if development else res.predict_bundle(frozen["bundle"], data)
        assert data.index.tolist() == p.index.tolist()
        if development:
            for key in ["target_id", "object_id", "action", "cohort", "frame_id"]:
                assert data[key].tolist() == p[key].tolist()
        result = diagnose(data, p, dataset, plan, prior)
        results.append(result)
        print(f"{dataset}: {data.target_id.nunique()} targets; original A/D replay exact; elapsed {time.monotonic()-start:.1f}s", flush=True)
    merged = {}
    for key in results[0]:
        table = pd.concat([r[key] for r in results], ignore_index=True)
        if key in ["targets", "candidates", "outcomes"]:
            combined = table.loc[table.dataset.isin(ab.BANKS)].copy()
            combined["dataset"] = "combined8"
            table = pd.concat([table, combined], ignore_index=True)
        merged[key] = table
    combined_outcomes = merged["outcomes"].loc[merged["outcomes"].dataset.eq("combined8")]
    merged["summary"] = pd.concat([merged["summary"], pd.DataFrame([r for name, group in combined_outcomes.groupby("method") for r in summarize(group, "combined8", name)])], ignore_index=True)
    for key, table in merged.items():
        table.to_csv(output / (key + ".csv"), index=False)
    changed = [p for p, sha in hashes.items() if base.digest(Path(p)) != sha]
    assert not changed, changed
    dump(output / "checks.json", dict(input_files_unchanged=len(hashes), replay_A_D_exact=True,
        decomposition_exact=True, oracle_includes_initializer=True, no_training=True, no_threshold_tuning=True))
    dump(output / "state.json", dict(status="COMPLETE_POSTHOC_DIAGNOSIS_NOT_PROMOTED", verification="ANALYZED",
        elapsed_seconds=time.monotonic()-start, files={p.name: base.digest(p) for p in output.iterdir() if p.is_file()},
        finished_at=datetime.now(timezone.utc).isoformat()))
    print(merged["summary"].query("dataset == 'combined8' and scope == 'all'").to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
