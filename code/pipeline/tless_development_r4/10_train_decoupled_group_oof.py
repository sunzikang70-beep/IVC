"""Retrain existing candidate/winner calibration inside nested object partitions."""
from __future__ import annotations

import os
for _name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[_name] = "1"

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import traceback

import joblib
import numpy as np
import pandas as pd

import prepare_winner_instance_expansion as p
import train_bop_aware_state_residual as res
import train_bop_aware_selected_risk as sel
from pose_perturbation_protocols import DECOUPLED
from winner_training_data_io import read_csv

base, v2 = res.base, res.v2
RESIDUAL_VARIANTS = ["conditional_residual_legacy", "conditional_residual_joint"]
WINNER_VARIANTS = ["stratified_only", "selected_risk_only", "selected_risk_stratified"]


def partition_record(training, held, **context):
    a, b = sorted(map(int, training.object_id.unique())), sorted(map(int, held.object_id.unique()))
    if set(a)&set(b):
        raise ValueError("Object leakage")
    return dict(training_objects=a, validation_objects=b, **context)


def observable_shadow(data):
    shadow = data.copy()
    for key in base.TARGETS+base.CLASSES+base.AR+["ADD_S_mm", "initial_ADD_S_mm"]+["initial_"+c for c in base.AR]:
        shadow[key] = 99999.
    return shadow


def verify_inference(bundle, selected_risk, data, features, winner_plan, policies, residual_plan):
    clean = res.predict_bundle(bundle, features)
    shadow = observable_shadow(data)
    hidden = res.predict_bundle(bundle, shadow)
    np.testing.assert_allclose(clean, hidden, atol=1e-9, rtol=1e-9)
    target = data.loc[data.target_id.eq(data.target_id.iloc[0])]
    np.testing.assert_allclose(clean.loc[target.index], res.predict_bundle(bundle, target), atol=1e-9, rtol=1e-9)
    pool = sel.predict_selected_risk(selected_risk, sel.winner_pool(data, clean, winner_plan))
    fake = sel.predict_selected_risk(selected_risk, sel.winner_pool(shadow, hidden, winner_plan))
    pd.testing.assert_frame_equal(pool, fake)
    for variant in RESIDUAL_VARIANTS:
        if res.decisions(data, clean, policies[variant], variant, residual_plan) != res.decisions(shadow, hidden, policies[variant], variant, residual_plan):
            raise ValueError("Labels affected residual choices")
    for variant in WINNER_VARIANTS:
        policy = policies[variant]
        if sel.choices(data, pool, policy["policy_id"], policy, variant) != sel.choices(shadow, fake, policy["policy_id"], policy, variant):
            raise ValueError("Labels affected postselection choices")
    return dict(label_blind_candidate_predictions=True, label_blind_winners=True, label_blind_choices=True,
                single_target_batch_invariance=True)


def decorate(rows, metadata):
    return rows.merge(metadata, on="target_id", validate="many_to_one")


def planned_summary(outcomes, manifest):
    records = []
    dimensions = [("all", "all", manifest, outcomes)]
    dimensions += [("regime", str(value), group, outcomes.loc[outcomes.init_regime.eq(value)])
                   for value, group in manifest.groupby("init_regime")]
    dimensions += [("depth", condition, manifest, outcomes.loc[outcomes.input_depth_condition.eq(condition)])
                   for condition in outcomes.input_depth_condition.unique()]
    for axis, stratum, planned, chosen in dimensions:
        for method, group in chosen.groupby("method", sort=False):
            factor = 1 if axis == "depth" else 2
            total = len(planned)*factor
            active = group.selected_action.ne(base.INITIAL)
            bad = group.delta_ADD_S_mm.gt(1e-9)&active
            initial_ar = group[["initial_"+c for c in base.AR]].mean(axis=1)
            records.append(dict(axis=axis, stratum=stratum, method=method, planned_conditions=total,
                matched_conditions=len(group), physical_instances=group.frame_id.nunique(), objects=group.object_id.nunique(),
                coverage_fraction=len(group)/total, AR_matched=float(group.AR_mean_row.mean()),
                AR_all_planned_missing_zero=float(group.AR_mean_row.sum()/total),
                AR_gain_matched=float((group.AR_mean_row-initial_ar).mean()),
                AR_gain_all_planned_missing_zero=float((group.AR_mean_row-initial_ar).sum()/total),
                ADD_S_mm_matched=float(group.ADD_S_mm.mean()), ADD_delta_mm_matched=float(group.delta_ADD_S_mm.mean()),
                updates=int(active.sum()), update_fraction_matched=float(active.mean()), accepted_regressions=int(bad.sum()),
                accepted_regression_rate=float(bad.sum()/active.sum()) if active.any() else np.nan,
                all_regression_rate_matched=float(bad.mean()),
                **{c:float(group[c].mean()) for c in base.AR}))
    return pd.DataFrame(records)


def paired_bootstrap(outcomes):
    records = []
    rng = np.random.default_rng(base.SEED)
    chosen = outcomes.loc[outcomes.method.eq("selected_risk_stratified")]
    for reference in [base.INITIAL, "protocolF_se3_trust_p2p_s100", "protocolF_se3_trust_filtered_s100", "conditional_residual_legacy"]:
        paired = chosen.merge(outcomes.loc[outcomes.method.eq(reference)], on=["target_id", "object_id"],
                              suffixes=("_chosen", "_reference"), validate="one_to_one")
        for scope, group in [("all", paired), *list(paired.groupby("init_regime_chosen"))]:
            for metric in ["ADD_S_mm", "AR_mean_row", *base.AR]:
                group = group.copy()
                group["difference"] = group[metric+"_chosen"]-group[metric+"_reference"]
                agg = group.groupby("object_id").difference.agg(["sum", "count"])
                indexes = rng.integers(0, len(agg), size=(2000, len(agg)))
                draws = agg["sum"].to_numpy()[indexes].sum(axis=1)/agg["count"].to_numpy()[indexes].sum(axis=1)
                low, high = np.quantile(draws, [.025, .975])
                records.append(dict(reference=reference, scope=scope, metric=metric, delta=float(group.difference.mean()),
                    ci_low=float(low), ci_high=float(high), object_clusters=len(agg),
                    interpretation="exploratory_inspected_development_object_cluster_bootstrap_not_confirmation"))
    return pd.DataFrame(records)


def run(args):
    if args.output.exists():
        raise FileExistsError("Preserve prior model training")
    protocol = json.loads((args.data/"locked_protocol.json").read_text())
    if protocol["protocol"] != DECOUPLED:
        raise ValueError("Only new explicitly decoupled data allowed")
    work = Path(__file__).parent
    raw_plan = json.loads((work/"bop_aware_tail_risk_v2_plan_20261019.json").read_text())
    residual_plan = json.loads((work/"bop_aware_state_residual_plan_20261019.json").read_text())
    winner_plan = json.loads((work/"bop_aware_selected_risk_plan_20261019.json").read_text())
    features = read_csv(args.data/"observable_candidate_features.csv")
    labels = read_csv(args.data/"development_bop_labels.csv")
    data = base.attach_labels(features, labels).reset_index(drop=True)
    schema = json.loads((args.data/"feature_schema.json").read_text())
    names = schema["feature_names"]
    if len(names) != 191 or len(data) != 8796 or data.frame_id.nunique() != 629 or set(data.object_id) != set(p.OBJECTS):
        raise ValueError("New-protocol candidate coverage mismatch")
    if not labels.pose_protocol.eq(DECOUPLED).all() or not labels.evaluation_depth_policy.eq("native_sensor_depth").all():
        raise ValueError("Mixed protocol/reference depth labels")
    if not np.isfinite(data[names].to_numpy()[~data[names].isna().to_numpy()]).all():
        raise ValueError("Unnormalized nonfinite features")
    manifest = read_csv(args.data/"locked_development_instances.csv")
    regimes = manifest[["cohort", "frame_id", "init_regime"]]
    targets = data[["target_id", "cohort", "frame_id"]].drop_duplicates().merge(regimes, on=["cohort", "frame_id"], validate="many_to_one")
    target_metadata = targets[["target_id", "init_regime"]]
    initial_columns = data.drop_duplicates("target_id")[["target_id", *["initial_"+c for c in base.AR]]]
    args.output.mkdir(parents=True)
    state = dict(status="RUNNING", protocol=DECOUPLED, started_at_utc=datetime.now(timezone.utc).isoformat(),
        pid=os.getpid(), method_promoted=False, new_final_holdout=False, folds=[],
        source_hashes={str(path):p.digest(path) for path in [Path(__file__), args.data/"locked_protocol.json",
            args.data/"observable_candidate_features.csv", args.data/"development_bop_labels.csv", args.data/"feature_schema.json"]})
    p.save(args.output/"state.json", state)
    partitions, all_predictions, selections, winners, grids, strata, funnels = [], [], [], [], [], [], []
    start = time.monotonic()
    try:
        for fold, (a, b) in enumerate(base.folds(data, 5)):
            training, held = data.iloc[a], data.iloc[b]
            record = partition_record(training, held, fold=fold)
            expected = protocol["outer_folds"][fold]
            if record != expected:
                raise ValueError("Outer object partition differs from pre-result lock")
            directory = args.output/f"fold_{fold}"
            directory.mkdir()
            inner_predictions, inner_winners = [], []
            print(f"OUTER {fold+1}/5; held objects {record['validation_objects']}", flush=True)
            for inner, (c, d) in enumerate(base.folds(training, 3)):
                fitting, query = training.iloc[c], training.iloc[d]
                partition = partition_record(fitting, query, outer_fold=fold, inner_fold=inner, level="policy_validation")
                partitions.append(partition)
                deep_predictions = []
                for deep, (e, f) in enumerate(base.folds(fitting, 3)):
                    deep_train, deep_held = fitting.iloc[e], fitting.iloc[f]
                    entry = partition_record(deep_train, deep_held, outer_fold=fold, inner_fold=inner, deep_fold=deep,
                                             level="winner_training", policy_validation_objects=partition["validation_objects"],
                                             outer_validation_objects=record["validation_objects"])
                    if set(entry["training_objects"])&set(entry["policy_validation_objects"]+entry["outer_validation_objects"]):
                        raise ValueError("Deep winner-training leakage")
                    partitions.append(entry)
                    bundle = res.fit_bundle(deep_train, names, raw_plan, residual_plan)
                    deep_predictions.append(res.predict_bundle(bundle, deep_held))
                    joblib.dump(dict(bundle=bundle, partition=entry), directory/f"inner_{inner}_deep_{deep}.joblib")
                    print(f"  inner {inner+1}/3 deep {deep+1}/3; elapsed {time.monotonic()-start:.1f}s", flush=True)
                deep_p = pd.concat(deep_predictions).sort_index().loc[fitting.index]
                deep_p.to_csv(directory/f"inner_{inner}_winner_training_oof.csv", index_label="candidate_index")
                risk_model, pool = sel.fit_selected_risk(fitting, deep_p, winner_plan)
                pool.to_csv(directory/f"inner_{inner}_training_winners.csv", index=False)
                bundle = res.fit_bundle(fitting, names, raw_plan, residual_plan)
                predicted = res.predict_bundle(bundle, query)
                inner_predictions.append(predicted)
                scored = sel.predict_selected_risk(risk_model, sel.winner_pool(query, predicted, winner_plan))
                scored["inner_fold"] = inner
                inner_winners.append(scored)
                joblib.dump(dict(bundle=bundle, selected_risk=risk_model, partition=partition), directory/f"inner_{inner}.joblib")
            inner_p = pd.concat(inner_predictions).sort_index().loc[training.index]
            inner_pool = pd.concat(inner_winners, ignore_index=True)
            inner_p.to_csv(directory/"inner_policy_oof_predictions.csv", index_label="candidate_index")
            inner_pool.to_csv(directory/"inner_policy_oof_winners.csv", index=False)
            bundle = res.fit_bundle(training, names, raw_plan, residual_plan)
            risk_model, pool = sel.fit_selected_risk(training, inner_p, winner_plan)
            pool.to_csv(directory/"outer_training_winners.csv", index=False)
            predicted = res.predict_bundle(bundle, held)
            outer_scores = sel.predict_selected_risk(risk_model, sel.winner_pool(held, predicted, winner_plan))
            outer_scores["outer_fold"] = fold
            winners.append(outer_scores)
            policies = {}
            for variant in RESIDUAL_VARIANTS:
                policy, grid = res.tune_policy(training, inner_p, variant, residual_plan)
                policies[variant] = policy
                chosen = v2.evaluate_choices(held, res.decisions(held, predicted, policy, variant, residual_plan))
                chosen["method"], chosen["outer_fold"] = variant, fold
                selections.append(chosen)
                grids.extend(dict(row, method=variant, outer_fold=fold) for row in grid)
            for variant in WINNER_VARIANTS:
                policy, grid, stratum = sel.tune_policy(training, inner_p, inner_pool, variant, winner_plan, residual_plan)
                policies[variant] = policy
                chosen = sel.selected_frame(held, outer_scores, policy["policy_id"], policy, variant)
                chosen["method"], chosen["outer_fold"] = variant, fold
                selections.append(chosen)
                grids.extend(dict(row, method=variant, outer_fold=fold) for row in grid)
                strata.extend(dict(row, outer_fold=fold) for row in stratum)
                strata.extend(dict(row, outer_fold=fold) for row in sel.stratum_statistics(chosen, variant, policy["policy_id"], "outer_result"))
            checks = verify_inference(bundle, risk_model, held, features.loc[held.index], winner_plan, policies, residual_plan)
            predicted["outer_fold"] = fold
            all_predictions.append(predicted)
            joblib.dump(dict(bundle=bundle, selected_risk=risk_model, policies=policies, partition=record,
                             protocol=DECOUPLED), directory/"outer_model.joblib")
            for variant in RESIDUAL_VARIANTS:
                gates, _, _ = res.gate_arrays(held, predicted, policies[variant], variant, residual_plan)
                for gate, values in gates.items():
                    funnels.append(dict(outer_fold=fold, method=variant, gate=gate, rows=len(held), passes=int(values.sum())))
            record.update(policies=policies, checks=checks, winner_training_physical_instances=pool.frame_id.nunique(),
                          held_physical_instances=held.frame_id.nunique())
            state["folds"].append(record)
            p.save(args.output/"state.json", state)
            p.save(args.output/"nested_object_partitions.json", partitions)
            print("OUTER COMPLETE", fold+1, "elapsed", round(time.monotonic()-start, 1), flush=True)
        predictions = pd.concat(all_predictions).sort_index().loc[data.index]
        pd.concat([data[["candidate_index", "target_id", "object_id", "cohort", "frame_id", "action", "pose_changed", "features_valid"]], predictions], axis=1).to_csv(args.output/"outer_oof_predictions.csv", index=False)
        tables = [*selections]
        for method in [base.INITIAL, *base.ACTIONS]:
            baseline = base.baseline_rows(data, method)
            baseline["method"] = method
            tables.append(baseline)
        outcomes = pd.concat(tables, ignore_index=True)
        outcomes = decorate(outcomes, target_metadata).merge(initial_columns, on="target_id", validate="many_to_one")
        outcomes.to_csv(args.output/"all_method_oof_outcomes.csv", index=False)
        summary = planned_summary(outcomes, manifest)
        summary.to_csv(args.output/"matched_and_planned_summary.csv", index=False)
        pd.DataFrame(grids).to_csv(args.output/"inner_policy_grid.csv", index=False)
        pd.DataFrame(strata).to_csv(args.output/"stratified_risk_tables.csv", index=False)
        pd.DataFrame(funnels).to_csv(args.output/"acceptance_funnels.csv", index=False)
        oof_winners = pd.concat(winners, ignore_index=True)
        oof_winners.to_csv(args.output/"outer_oof_winners.csv", index=False)
        stats, bins = base.calibration_report(data, predictions)
        stats.to_csv(args.output/"outer_oof_probability_metrics.csv", index=False)
        bins.to_csv(args.output/"outer_oof_probability_bins.csv", index=False)
        calibration_by_regime = []
        for regime in p.REGIMES:
            ids = set(targets.loc[targets.init_regime.eq(regime), "target_id"])
            group = data.loc[data.target_id.isin(ids)]
            part, _ = base.calibration_report(group, predictions.loc[group.index])
            part["init_regime"] = regime
            calibration_by_regime.append(part)
        pd.concat(calibration_by_regime).to_csv(args.output/"regime_probability_metrics.csv", index=False)
        paired_bootstrap(outcomes).to_csv(args.output/"paired_object_bootstrap.csv", index=False)
        # The deployment policy is a development fit; only fold-local policies above define OOF performance.
        print("FINAL DEVELOPMENT REFIT; no new validation", flush=True)
        final_bundle = res.fit_bundle(data, names, raw_plan, residual_plan)
        final_risk, final_pool = sel.fit_selected_risk(data, predictions, winner_plan)
        final_pool.to_csv(args.output/"final_training_winners.csv", index=False)
        final_policies, final_grids = {}, []
        for variant in RESIDUAL_VARIANTS:
            policy, grid = res.tune_policy(data, predictions, variant, residual_plan)
            final_policies[variant] = policy
            final_grids.extend(dict(row, method=variant) for row in grid)
        for variant in WINNER_VARIANTS:
            policy, grid, _ = sel.tune_policy(data, predictions, oof_winners, variant, winner_plan, residual_plan)
            final_policies[variant] = policy
            final_grids.extend(dict(row, method=variant) for row in grid)
        pd.DataFrame(final_grids).to_csv(args.output/"final_policy_grid_exploratory.csv", index=False)
        verify_inference(final_bundle, final_risk, data, features, winner_plan, final_policies, residual_plan)
        joblib.dump(dict(bundle=final_bundle, selected_risk=final_risk, policies=final_policies, protocol=DECOUPLED,
                         status="DEVELOPMENT_ONLY_NOT_PROMOTED"), args.output/"development_candidate.joblib")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        methods = ["protocolF_se3_trust_p2p_s100", "conditional_residual_legacy", "selected_risk_stratified"]
        colors = ["#686868", "#237e70", "#c24652"]
        for ax, metric, title in zip(axes, ["AR_gain_matched", "accepted_regression_rate"], ["AR gain over own initializer", "ADD-S regression among accepted updates"]):
            for i, (method, color) in enumerate(zip(methods, colors)):
                group = summary.loc[summary.axis.eq("regime")&summary.method.eq(method)].set_index("stratum").reindex(p.REGIMES)
                ax.bar(np.arange(3)+(i-1)*.24, group[metric], width=.23, color=color, label=method.replace("protocolF_se3_trust_", ""))
            ax.set_xticks(np.arange(3), list(p.REGIMES))
            ax.set_title(title, fontsize=10)
            ax.axhline(0, color="#999999", linewidth=.6)
            ax.legend(fontsize=7)
        fig.savefig(args.output/"regime_gain_and_risk.png", dpi=180)
        plt.close(fig)
        scan = [
            ("Simpson", "Regime/cohort/depth and per-method tables; no pooled-only claim"),
            ("Ecological", "No per-instance guarantees from object means"),
            ("Selection/Berkson", "Matched-public-prediction conditional; all-planned AR also reported"),
            ("Collider", "GT labels do not filter which candidate conditions are evaluated"),
            ("Base rate", "Probability tables include prevalence and counts"),
            ("Regression to mean", "Comparisons use identical new-protocol initial poses"),
            ("Survivorship", "Missing ledger retained without replacement"),
            ("Look elsewhere", "All fixed methods and policy grids exported"),
            ("Forking paths", "Pre-result protocol and model lock; no outer-result retuning"),
            ("Causal overreach", "Development-only change; no real-platform/generalization claim"),
            ("Reverse causality", "Label poison tests pass at feature and deployment stages"),
        ]
        pd.DataFrame(scan, columns=["check", "assessment"]).to_csv(args.output/"statistical_fallacy_scan_11.csv", index=False)
        state.update(status="COMPLETE_NESTED_OBJECT_OOF_NOT_PROMOTED", completed_at_utc=datetime.now(timezone.utc).isoformat(),
            elapsed_seconds=time.monotonic()-start, final_policies=final_policies, checks=dict(outer_object_disjoint=True,
                inner_and_deeper_object_disjoint=True, from_scratch_new_protocol_models=True, no_old_calibrator_reuse=True,
                no_reserved_inference=True, full_candidate_BOP_labels=True, csv_round_trip=True))
        lines = ["# 解耦协议开发与对象分组 OOF 校准", "", "## Material Passport",
            "- Status: COMPLETE_NESTED_OBJECT_OOF_NOT_PROMOTED; Verification Status: ANALYZED",
            "- 22 个开发物体；629 个匹配物理实例；733 个实例/难度条件；1,466 个实例/难度/深度条件。",
            "- 全部 10,262 个位姿 BOP 标签；8,796 条六候选特征，未按赢家筛选标签。",
            "- 5 折外层对象留出；3 折内层策略验证；赢家风险训练另做 3 折更深对象 OOF。",
            "- 只改变旋转中心协议；原主干、六候选、特征、模型参数和策略网格不变。所有模型从头训练。",
            "", "## 主结果（相对新协议自身初始化）", "```text",
            summary.loc[summary.method.isin([base.INITIAL, *methods]) & summary.axis.isin(["all", "regime"])].to_string(index=False), "```",
            "", "## 使用边界",
            "- 本轮使用已经检查过的开发物体/实例，不是新的未见物体确认。88 个预留实例和五个排除物体未推理。",
            "- 旧相机原点压力协议和当前解耦协议不混合拟合；绝对初始化难度变化不能作为算法收益。",
            "- AR 是所选目标上的 BOP19 阈值平均，不是官方排行榜分数；VSD 使用原始传感器深度。",
            "- 物理实例、不同难度、两种深度和六候选不是独立样本；区间按物体聚类，仅作探索性描述。",
            "- 外层主表使用每折训练集内定出的策略；最终全开发策略是开发拟合，不把其回代结果冒充 OOF。",
            "- 对象划分不等于图像/场景完全独立；共享 RGB-D 图像只允许不同对象，后续另需独立确认。",
            "- 概率残差和 q75 状态残差界是经验校准，非逐实例安全保证或严格 conformal 保证。",
            "- 若内层约束无可行策略，保持初始化并明确报告；未放宽约束或根据外层结果二次调参。",
            "- 完整 CSV、各折模型、策略网格、赢家训练池、分区和校准表在本目录。", ""]
        (args.output/"report_zh.md").write_text("\n".join(lines), encoding="utf-8")
        for path, expected in state["source_hashes"].items():
            if p.digest(Path(path)) != expected:
                raise ValueError("Training source/data changed: "+path)
        state["artifact_hashes"] = {str(path.relative_to(args.output)):p.digest(path) for path in args.output.rglob("*")
            if path.is_file() and path != args.output/"state.json"}
        p.save(args.output/"state.json", state)
        print(summary.loc[summary.axis.isin(["all", "regime"]) & summary.method.eq("selected_risk_stratified")].to_string(index=False), flush=True)
        print("DECOUPLED OBJECT OOF COMPLETE", flush=True)
    except Exception:
        state.update(status="FAILED_PRESERVED_NO_AUTOMATIC_RETRY", error=traceback.format_exc())
        p.save(args.output/"state.json", state)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
