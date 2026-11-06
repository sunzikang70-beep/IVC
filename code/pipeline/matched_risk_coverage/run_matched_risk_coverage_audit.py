"""Frozen-r4 risk-budget transfer and label-blind matched-coverage audit."""
from __future__ import annotations

import os
for _key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[_key] = "1"

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time
import traceback

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import run_same_bank_residual_ablation as ref

ROOT, DEV, OOF = ref.ROOT, ref.DEV, ref.OOF
ARMS = ref.PRIMARY
REFERENCE = ROOT / "outputs/same_bank_residual_ablation_20261026_r1"
BUDGETS = [.05, .10, .15, .20]
COVERAGES = [.10, .20, .30]
BASES = ["utility", "predicted_risk"]
QUANTILES = np.linspace(0, 1, 41)
POLICY = dict(risk_limit=.3, component_loss=.01, continuous_weight=.005, disabled=False)
TIE_KEYS = ["object_id", "scene_id", "img_id", "inst_id", "cohort", "input_depth_condition"]
PAIRS = [("D_joint_full", "A_delta_rank"), ("C_state_residual", "A_delta_rank"),
         ("C_state_residual", "B_fixed_residual"), ("D_joint_full", "C_state_residual"),
         ("D_joint_full", "B_fixed_residual")]


def winners(data, p, bundle, arm, plan):
    ap = ref.adjusted(data, p, bundle, arm)
    choices = ref.choose(data, ap, POLICY, arm, plan)
    table = ref.evaluate(data, choices)
    mean = ap[["lcb_" + t for t in ref.base.TARGETS[:3]]].mean(axis=1).to_numpy()
    utility = ap.lcb_delta_joint_utility.to_numpy() if arm == "D_joint_full" else mean - .005 * ap.ucb_delta_ADD_S_mm.to_numpy() / 5
    scores = pd.DataFrame({"target_id": data.target_id.to_numpy(), "selected_action": data.action.to_numpy(),
                           "utility": utility, "predicted_risk": ap.adds_risk_upper.to_numpy()})
    table = table.merge(scores, on=["target_id", "selected_action"], how="left", validate="one_to_one")
    table["base_proposed_action"] = table.selected_action
    table["base_eligible"] = table["update"].eq(1)
    table["arm"] = arm
    return table


def confidence(table, basis):
    if basis == "utility":
        return table.utility.to_numpy(float)
    if basis == "predicted_risk":
        return -table.predicted_risk.to_numpy(float)
    raise ValueError(basis)


def reject(table, mask):
    mask = np.asarray(mask, bool)
    if mask.shape != (len(table),) or (mask & ~table.base_eligible.to_numpy(bool)).any():
        raise ValueError("Cannot accept an ineligible update")
    result = table.copy()
    refused = ~mask
    for c in ["ADD_S_mm", *ref.base.AR]:
        result.loc[refused, c] = result.loc[refused, "initial_" + c]
    result.loc[refused, "selected_action"] = ref.base.INITIAL
    result.loc[refused, "delta_ADD_S_mm"] = 0.
    result["AR_mean_row"] = result[ref.base.AR].mean(axis=1)
    result["AR_gain"] = result.AR_mean_row - result[["initial_" + c for c in ref.base.AR]].mean(axis=1)
    result["update"] = mask.astype(int)
    result["regression"] = result.delta_ADD_S_mm.gt(1e-9).astype(int)
    result["bop_regression"] = result.AR_gain.lt(-1e-12).astype(int)
    return result


def threshold_mask(table, policy):
    if policy.get("disabled", False):
        return np.zeros(len(table), bool)
    score = confidence(table, policy["basis"])
    return table.base_eligible.to_numpy(bool) & np.isfinite(score) & (score >= policy["threshold"])


def totals(table):
    updates, regressions = int(table["update"].sum()), int(table.regression.sum())
    obj = table.groupby("object_id")[["AR_mean_row", "AR_gain", "delta_ADD_S_mm", "update", "regression", "bop_regression"]].mean()
    return dict(targets=len(table), objects=len(obj), AR_equal_object=float(obj.AR_mean_row.mean()),
        AR_pooled=float(table.AR_mean_row.mean()), AR_gain_equal_object=float(obj.AR_gain.mean()),
        AR_gain_pooled=float(table.AR_gain.mean()), ADD_delta_mm=float(table.delta_ADD_S_mm.mean()),
        updates=updates, update_fraction=updates / len(table), regressions=regressions,
        accepted_risk=regressions / updates if updates else np.nan,
        all_target_regression_rate=regressions / len(table),
        accepted_risk_equal_object_weighted=float(obj.regression.mean() / obj["update"].mean()) if obj["update"].sum() else np.nan,
        bop_regression_rate=float(table.bop_regression.mean()),
        **{c: float(table[c].mean()) for c in ref.base.AR})


def calibration_grid(table):
    rows = []
    for basis in BASES:
        score = confidence(table, basis)
        values = score[table.base_eligible.to_numpy(bool) & np.isfinite(score)]
        if not len(values):
            continue
        for quantile in QUANTILES:
            threshold = float(np.quantile(values, quantile))
            policy = dict(basis=basis, threshold=threshold, disabled=False)
            chosen = reject(table, threshold_mask(table, policy))
            stats = totals(chosen)
            cohort_ok = all(g.AR_gain.mean() >= -1e-12 and g.delta_ADD_S_mm.mean() <= 1e-12 for _, g in chosen.groupby("cohort"))
            rows.append(dict(**policy, quantile=float(quantile), **stats,
                             cohort_nonregression=bool(cohort_ok), minimum_coverage=bool(stats["update_fraction"] >= .05)))
    return rows


def choose_risk_policy(grid, budget):
    feasible = [r for r in grid if r["minimum_coverage"] and r["cohort_nonregression"] and r["accepted_risk"] <= budget + 1e-12]
    if not feasible:
        return dict(disabled=True, basis="utility", threshold=None, budget=budget,
                    reason="no_development_policy_satisfies_risk_coverage_and_cohort_constraints")
    best = min(feasible, key=lambda r: (-r["AR_gain_equal_object"], r["accepted_risk"], -r["update_fraction"], r["basis"], r["quantile"]))
    return {**{k: best[k] for k in ["basis", "threshold", "quantile", "disabled"]}, "budget": budget,
            "development_targets": best["targets"], "development_updates": best["updates"],
            "development_risk": best["accepted_risk"], "development_AR_gain_equal_object": best["AR_gain_equal_object"]}


def rank_indices(table, basis):
    score = confidence(table, basis)
    ranked = table.loc[table.base_eligible & np.isfinite(score), TIE_KEYS + ["predicted_risk", "base_proposed_action"]].copy()
    ranked["_confidence"] = score[table.index.get_indexer(ranked.index)]
    return ranked.sort_values(["_confidence", "predicted_risk", "base_proposed_action", *TIE_KEYS],
                              ascending=[False, True, True] + [True] * len(TIE_KEYS), kind="stable").index.to_numpy()


def quota_masks(tables, basis, rate, mode):
    first = tables[ARMS[0]]
    for table in tables.values():
        if not table[TIE_KEYS].equals(first[TIE_KEYS]):
            raise ValueError("Quota tables differ in target identity/order")
    masks = {arm: np.zeros(len(first), bool) for arm in ARMS}
    records = []
    groups = [("all", first.index)] if mode == "pooled_exact" else first.groupby("object_id", sort=True).groups.items()
    for group, indices in groups:
        n = len(indices)
        requested = int(np.floor(rate * n + 1e-12))
        available = {arm: int(table.loc[indices].base_eligible.sum()) for arm, table in tables.items()}
        effective = requested if mode == "pooled_exact" else min(requested, *available.values())
        feasible = all(count >= effective for count in available.values())
        records.append(dict(group=str(group), target_count=n, requested=rate, requested_updates=requested,
                            actual_updates=effective, feasible=feasible, capped=effective < requested,
                            **{arm + "_available": count for arm, count in available.items()}))
        if not feasible:
            return None, records
        for arm in ARMS:
            ordered = rank_indices(tables[arm].loc[indices], basis)
            mask_positions = first.index.get_indexer(ordered[:effective])
            masks[arm][mask_positions] = True
    return masks, records


def decorate(table, dataset, family, point, basis, mode):
    table = table.copy()
    table["dataset"], table["family"], table["point"], table["ordering"], table["quota_mode"] = dataset, family, point, basis, mode
    return table


def summaries(outcomes):
    rows = []
    by = ["dataset", "family", "point", "ordering", "quota_mode", "arm"]
    for key, table in outcomes.groupby(by, sort=False, dropna=False):
        for scope, group in ref.scopes(table):
            stats = totals(group)
            rows.append(dict(zip(by, key), scope=scope, **stats,
                             actual_risk_within_budget=bool(stats["updates"] and stats["accepted_risk"] <= key[2] + 1e-12) if key[1] == "risk_budget" else None))
    return pd.DataFrame(rows)


def paired(outcomes, samples):
    rng = np.random.default_rng(ref.SEED + 1)
    rows = []
    keys = ["source_bank", "target_id", "object_id"]
    by = ["dataset", "family", "point", "ordering", "quota_mode"]
    for context, group in outcomes.groupby(by, sort=False, dropna=False):
        for candidate, reference in PAIRS:
            x, y = group.loc[group.arm.eq(candidate)], group.loc[group.arm.eq(reference)]
            p = x.merge(y, on=keys, suffixes=("_x", "_y"), validate="one_to_one")
            ref.same(len(p), len(x), "Paired target coverage mismatch")
            objects = sorted(p.object_id.unique())
            ix = rng.integers(0, len(objects), (samples, len(objects)))
            count = p.groupby("object_id").size().reindex(objects).to_numpy()
            for metric in ["AR_mean_row", *ref.base.AR, "ADD_S_mm", "regression", "update"]:
                delta = p[metric + "_x"] - p[metric + "_y"]
                means = delta.groupby(p.object_id).mean().reindex(objects).to_numpy()
                boot = means[ix].mean(axis=1)
                pooled = (means[ix] * count[ix]).sum(axis=1) / count[ix].sum(axis=1)
                lo, hi = np.quantile(boot, [.025, .975])
                plo, phi = np.quantile(pooled, [.025, .975])
                rows.append(dict(zip(by, context), candidate=candidate, reference=reference, metric=metric,
                    objects=len(objects), target_conditions=len(p), equal_object_difference=float(means.mean()),
                    ci_low=float(lo), ci_high=float(hi), pooled_difference=float(delta.mean()),
                    pooled_ci_low=float(plo), pooled_ci_high=float(phi), exploratory_pointwise=True))
            g = p.groupby("object_id")[["regression_x", "regression_y", "update_x", "update_y"]].sum().reindex(objects)
            with np.errstate(divide="ignore", invalid="ignore"):
                bx = g.regression_x.to_numpy()[ix].sum(axis=1) / g.update_x.to_numpy()[ix].sum(axis=1)
                by_risk = g.regression_y.to_numpy()[ix].sum(axis=1) / g.update_y.to_numpy()[ix].sum(axis=1)
                dist = bx - by_risk
            finite = dist[np.isfinite(dist)]
            lo, hi = np.quantile(finite, [.025, .975]) if len(finite) else [np.nan, np.nan]
            dx = g.regression_x.sum() / g.update_x.sum() if g.update_x.sum() else np.nan
            dy = g.regression_y.sum() / g.update_y.sum() if g.update_y.sum() else np.nan
            rows.append(dict(zip(by, context), candidate=candidate, reference=reference, metric="accepted_risk",
                objects=len(objects), target_conditions=len(p), equal_object_difference=np.nan,
                ci_low=np.nan, ci_high=np.nan, pooled_difference=float(dx - dy),
                pooled_ci_low=float(lo), pooled_ci_high=float(hi), finite_bootstrap_samples=len(finite), exploratory_pointwise=True))
    return pd.DataFrame(rows)


def draw(summary, output):
    colors = ["#687d8b", "#ca8046", "#21917b", "#a55471"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    for ax, dataset in zip(axes, ["holdout5", "added3", "combined8"]):
        for arm, color in zip(ARMS, colors):
            g = summary.loc[summary.scope.eq("all") & summary.dataset.eq(dataset) & summary.family.eq("matched_coverage")
                            & summary.ordering.eq("utility") & summary.quota_mode.eq("pooled_exact") & summary.arm.eq(arm)].sort_values("update_fraction")
            ax.plot(g.update_fraction * 100, g.AR_equal_object * 100, marker="o", color=color, label=arm[0])
        ax.set_title(dataset)
        ax.set_xlabel("Exactly matched update fraction (%)")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Equal-object AR (%)")
    axes[-1].legend(title="Arm")
    fig.savefig(output / "matched_update_accuracy.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    for arm, color in zip(ARMS, colors):
        g = summary.loc[summary.scope.eq("all") & summary.dataset.eq("combined8") & summary.family.eq("risk_budget") & summary.arm.eq(arm)].sort_values("point")
        axes[0].plot(g.point * 100, g.AR_equal_object * 100, marker="o", color=color, label=arm[0])
        axes[1].plot(g.point * 100, g.accepted_risk * 100, marker="o", color=color)
    axes[1].plot([5, 20], [5, 20], "--", color="#777777", lw=1)
    axes[0].set_ylabel("Equal-object AR (%)")
    axes[1].set_ylabel("Observed accepted ADD-S risk (%)")
    for ax in axes:
        ax.set_xlabel("Development risk budget (%)")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(title="Arm")
    fig.savefig(output / "risk_budget_transfer.png", dpi=180)
    plt.close(fig)


def make_report(output, summary, comparisons):
    lines = ["# 同更新率与同开发风险预算比较", "", "## Material Passport", "",
        "模式：本地冻结候选重放、统计分析及写作准备审查。Verification Status: ANALYZED。",
        "不训练、不重新生成候选、不上云、不改r4方法；原5与新增3此前已经检查，本次不是新的确认性验证。",
        "输入来源和哈希见input_hashes.json，实验方案见locked_plan.json，开发侧锁定阈值见frozen_risk_policies.json。", "",
        "## 严格区分两种公平比较", "",
        "1. 同更新率：只用GT-free分数排序，按固定批次配额选择相同数量的目标。utility是主排序，公共预测回归风险排序是对照。没有读取测试标签选择配额。",
        "2. 同风险预算：只在开发物体OOF上选阈值；留出真实回归率必须单独报告。相同预算不等于相同真实风险，不能强行写成测试风险已经匹配。",
        "风险统一定义为接受更新后ADD-S增加>1e-9的比例，不是绝对位姿失败率，也不是BOP退化率；后者另行报告。",
        "同更新率不是同计算量：六个候选和特征仍全部计算。拒绝更新仍输出初始化器，不计为消失或漏检。", "",
        "## 固定批次配额结果", "",
        "pooled_exact严格匹配全批次更新数；object_capped在每个物体内也匹配更新数，但只用共同可用候选容量封顶。",
        "某物体有一个方法完全拒绝时，该物体的匹配配额为0，所有方法都回退；该物体保留在精度分母中。",
        "跨批次使用top-k是已知批次预算的分析控制，不等价于已经部署的逐帧在线阈值。", "",
        "| 数据 | 配额 | 排序 | 目标更新率 | 方法 | 实际更新率 | AR等物体(%) | 接受后ADD-S风险 |",
        "|---|---|---|---:|---|---:|---:|---:|"]
    sel = summary.loc[summary.scope.eq("all") & summary.family.eq("matched_coverage") & summary.ordering.eq("utility") & summary.dataset.ne("development22")]
    for _, r in sel.iterrows():
        risk = "NA" if pd.isna(r.accepted_risk) else f"{r.accepted_risk:.2%}"
        lines.append(f"| {r.dataset} | {r.quota_mode} | {r.ordering} | {r.point:.0%} | {r.arm} | {r.update_fraction:.2%} | {100*r.AR_equal_object:.4f} | {risk} |")
    lines += ["", "## 开发侧风险预算迁移", "",
        "每个方法同等82组候选阈值：两种可观测排序，各41个开发侧分位点。选择最大等物体AR增益，要求开发接受后风险不超预算、更新率至少5%、每cohort AR与ADD-S不退化。",
        "外折开发报告使用内折物体OOF选阈值；最终留出阈值只使用22物体外折OOF。没有可行解时全部回退，风险为NA而不是0。", "",
        "| 数据 | 开发风险预算 | 方法 | AR等物体(%) | 更新率 | 实际接受后风险 | 实际是否达预算 |",
        "|---|---:|---|---:|---:|---:|---|"]
    sel = summary.loc[summary.scope.eq("all") & summary.family.eq("risk_budget") & summary.dataset.ne("development22")]
    for _, r in sel.iterrows():
        risk = "NA" if pd.isna(r.accepted_risk) else f"{r.accepted_risk:.2%}"
        lines.append(f"| {r.dataset} | {r.point:.0%} | {r.arm} | {100*r.AR_equal_object:.4f} | {r.update_fraction:.2%} | {risk} | {r.actual_risk_within_budget} |")
    lines += ["", "## 配对统计", "", "区间按物体簇bootstrap，等物体AR为主、目标池化为辅；少量物体、共享场景、重复测量与历史检查限制保留。",
        "所有区间均为探索性逐点95%区间，跨预算/配额/排序多重比较不作为显著性宣称。原5与新增3分开保留，合并8不代表全新的独立样本。",
        "paired_comparisons.csv包含D-A、C-A、C-B、D-C、D-B的精度和风险差值；全部正负结果均保留。", "",
        "## 可复现与解释边界", "",
        "开发阈值在holdout inference之前封存并核验。测试配额只依赖可观测分数与已知物体身份；没有按AR挑目标或按真实风险挑阈值。",
        "覆盖率是完整共同候选库中的条件更新率，新增3仍只有61/90计划实体目标匹配，不能当成全端到端覆盖。",
        "原PRL/MVA稿与当前模型并不相同，旧跨数据集/时间/候选数结果不可直接移入r4论文。见writing_readiness_zh.md。", ""]
    (output / "report_zh.md").write_text("\n".join(lines), encoding="utf-8")


def audit_files():
    return list(dict.fromkeys(ref.evidence_paths() + [REFERENCE / n for n in ["state.json", "per_target_outcomes.csv", "module_summary.csv", "interpretation_zh.md"]]
        + [Path(__file__), Path(__file__).with_name("test_matched_risk_coverage_audit.py"),
           ROOT / "manuscript_mva/main.tex", ROOT / "manuscript_prl/main.tex",
           ROOT / "outputs/novelty_audit_conditional_residual_joint_20261026/novelty_audit_zh.md"]))


def run(args):
    _, model, inventory = ref.preflight()
    previous = ref.load(REFERENCE / "state.json")
    for name, digest in previous["artifact_hashes"].items():
        ref.same(ref.base.digest(REFERENCE / name), digest, "Previous audit artifact changed")
    hashes = {str(p): ref.base.digest(p) for p in audit_files()}
    if args.preflight:
        print(json.dumps(dict(status="PREFLIGHT_OK", inputs=len(hashes), inventory=inventory), indent=2), flush=True)
        return
    if args.output.exists():
        raise FileExistsError("Preserve output: " + str(args.output))
    plan = ref.load(Path(ref.res.__file__).with_name("bop_aware_state_residual_plan_20261019.json"))
    args.output.mkdir(parents=True)
    ref.save(args.output / "locked_plan.json", dict(status="LOCKED_BEFORE_NEW_MATCHED_RESULTS", arms=ARMS,
        risk_budgets=BUDGETS, coverage_rates=COVERAGES, cutoff_quantiles=QUANTILES.tolist(), orderings=BASES,
        base_policy=POLICY, constraints=dict(minimum_coverage=.05, cohort_AR_delta_min=0, cohort_ADD_delta_max=0),
        selection_objective="maximize_equal_object_AR_gain_then_risk_then_coverage", candidate_threshold_budget=82,
        coverage_modes=["pooled_exact", "object_capped"], bootstrap_samples=args.bootstrap,
        holdout_tuning=False, risk_guarantee=False, same_realized_test_risk_claim=False,
        inference="posthoc_reused_holdout_exploratory", paired_contrasts=PAIRS, inventory=inventory))
    ref.save(args.output / "input_hashes.json", hashes)
    shutil.copy2(REFERENCE / "environment.json", args.output / "environment.json")
    state = dict(status="RUNNING", stage="nested_development", pid=os.getpid(),
                 started_at_utc=datetime.now(timezone.utc).isoformat(), method_promoted=False, status_scope="ANALYZED")
    ref.save(args.output / "state.json", state)
    start = time.monotonic()
    try:
        dev, _, _ = ref.bank(DEV, True)
        predictions = ref.read_csv(OOF / "outer_oof_predictions.csv", index_col="candidate_index")
        for col in ["target_id", "object_id", "cohort", "frame_id", "action"]:
            ref.same(predictions[col].tolist(), dev[col].tolist(), "Cached OOF identity changed")
        full_dev, inner_grids, outcomes, nested_policies = {arm: [] for arm in ARMS}, [], [], []
        for fold, (tr, va) in enumerate(ref.base.folds(dev, 5)):
            training, held = dev.iloc[tr], dev.iloc[va]
            folder = OOF / f"fold_{fold}"
            outer = joblib.load(folder / "outer_model.joblib")
            ref.same(sorted(held.object_id.unique()), outer["partition"]["validation_objects"], "Outer fold changed")
            if set(held.object_id) & set(outer["bundle"]["training_objects"]):
                raise ValueError("Outer leakage")
            ip = ref.read_csv(folder / "inner_policy_oof_predictions.csv", index_col="candidate_index")
            inner_tables = {arm: [] for arm in ARMS}
            for i in range(3):
                inner = joblib.load(folder / f"inner_{i}.joblib")
                valid_objects = inner["partition"]["validation_objects"]
                if set(valid_objects) & set(inner["bundle"]["training_objects"]):
                    raise ValueError("Inner leakage")
                group = training.loc[training.object_id.isin(valid_objects)]
                for arm in ARMS:
                    inner_tables[arm].append(winners(group, ip.loc[group.index], inner["bundle"], arm, plan))
            for arm in ARMS:
                inner_table = pd.concat(inner_tables[arm], ignore_index=True)
                ref.same(inner_table.target_id.nunique(), training.target_id.nunique(), "Inner target coverage changed")
                grid = calibration_grid(inner_table)
                inner_grids += [dict(row, arm=arm, outer_fold=fold) for row in grid]
                outer_table = winners(held, predictions.loc[held.index], outer["bundle"], arm, plan)
                full_dev[arm].append(outer_table)
                for budget in BUDGETS:
                    policy = choose_risk_policy(grid, budget)
                    nested_policies.append(dict(policy, arm=arm, outer_fold=fold))
                    selected = reject(outer_table, threshold_mask(outer_table, policy))
                    outcomes.append(decorate(selected, "development22", "risk_budget", budget, "development_selected", "not_applicable"))
            print(f"NESTED DEVELOPMENT {fold+1}/5; elapsed={time.monotonic()-start:.1f}s", flush=True)
        full_dev = {arm: pd.concat(pieces, ignore_index=True) for arm, pieces in full_dev.items()}
        final_policies = {}
        for arm, table in full_dev.items():
            grid = calibration_grid(table)
            inner_grids += [dict(row, arm=arm, outer_fold="final_development") for row in grid]
            final_policies[arm] = {str(b): choose_risk_policy(grid, b) for b in BUDGETS}
        ref.save(args.output / "frozen_risk_policies.json", final_policies)
        ref.save(args.output / "nested_risk_policies.json", nested_policies)
        pd.DataFrame(inner_grids).to_csv(args.output / "development_threshold_grids.csv", index=False)
        frozen_hash = ref.base.digest(args.output / "frozen_risk_policies.json")
        state.update(stage="holdout_no_tuning", frozen_risk_policy_sha256=frozen_hash)
        ref.save(args.output / "state.json", state)
        banks = {"development22": full_dev}
        replay, label_blind = {}, {}
        old = ref.read_csv(REFERENCE / "per_target_outcomes.csv")
        for name, folder in ref.BANKS.items():
            data, _, _ = ref.bank(folder)
            p = ref.res.predict_bundle(model["bundle"], data)
            tables = {arm: winners(data, p, model["bundle"], arm, plan) for arm in ARMS}
            for arm, table in tables.items():
                previous_rows = old.loc[old.dataset.eq(name) & old.policy_mode.eq("shared_r4") & old.arm.eq(arm)]
                joined = table.merge(previous_rows, on=["cohort", *ref.base.KEYS], suffixes=("_new", "_old"), validate="one_to_one")
                ref.same(len(joined), len(table), "Base replay target mismatch")
                ref.same(joined.selected_action_new.tolist(), joined.selected_action_old.tolist(), "Base replay choices differ")
                np.testing.assert_allclose(joined.AR_mean_row_new, joined.AR_mean_row_old, atol=1e-12)
                for budget in BUDGETS:
                    policy = final_policies[arm][str(budget)]
                    selected = reject(table, threshold_mask(table, policy))
                    outcomes.append(decorate(selected, name, "risk_budget", budget, "development_selected", "not_applicable"))
            banks[name] = tables
            replay[name] = dict(all_four_actions_identical=True, AR_atol=1e-12)
            print(f"{name} frozen-threshold inference complete; no threshold selection", flush=True)
        banks["combined8"] = {arm: pd.concat([banks["holdout5"][arm].assign(source_bank="holdout5"), banks["added3"][arm].assign(source_bank="added3")], ignore_index=True) for arm in ARMS}
        quota_receipts = []
        for name, tables in banks.items():
            for basis in BASES:
                for rate in COVERAGES:
                    for mode in ["pooled_exact", "object_capped"]:
                        masks, records = quota_masks(tables, basis, rate, mode)
                        quota_receipts += [dict(r, dataset=name, ordering=basis, quota_mode=mode) for r in records]
                        if masks is None:
                            continue
                        accepted_counts = []
                        fake_tables = {arm: table.assign(AR_mean_row=99999., AR_gain=-99999., delta_ADD_S_mm=99999., regression=1) for arm, table in tables.items()}
                        fake_masks, _ = quota_masks(fake_tables, basis, rate, mode)
                        for arm, table in tables.items():
                            np.testing.assert_array_equal(masks[arm], fake_masks[arm])
                            selected = reject(table, masks[arm])
                            accepted_counts.append(int(selected["update"].sum()))
                            if "source_bank" not in selected:
                                selected["source_bank"] = name
                            outcomes.append(decorate(selected, name, "matched_coverage", rate, basis, mode))
                        ref.same(len(set(accepted_counts)), 1, "Coverage not exactly matched")
            label_blind[name] = True
        # The combined risk row is the union of separately inferred holdout banks.
        for i, table in enumerate(outcomes):
            if "source_bank" not in table:
                outcomes[i] = table.assign(source_bank=table.dataset)
        all_rows = pd.concat(outcomes, ignore_index=True)
        risk_combined = all_rows.loc[all_rows.family.eq("risk_budget") & all_rows.dataset.isin(ref.BANKS)].copy()
        risk_combined["dataset"] = "combined8"
        all_rows = pd.concat([all_rows, risk_combined], ignore_index=True)
        all_rows.to_csv(args.output / "per_target_outcomes.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
        summary = summaries(all_rows)
        summary.to_csv(args.output / "matched_summary.csv", index=False)
        comparisons = paired(all_rows, args.bootstrap)
        comparisons.to_csv(args.output / "paired_comparisons.csv", index=False)
        pd.DataFrame(quota_receipts).to_csv(args.output / "quota_receipts.csv", index=False)
        unchanged = {path: ref.base.digest(Path(path)) == sha for path, sha in hashes.items()}
        if not all(unchanged.values()) or ref.base.digest(args.output / "frozen_risk_policies.json") != frozen_hash:
            raise ValueError("Input or frozen policy changed")
        ref.save(args.output / "checks.json", dict(inputs_unchanged=unchanged, base_replay=replay,
            label_blind_quotas=label_blind, all_feasible_coverage_counts_identical=True,
            risk_policies_frozen_before_holdout=True, no_claim_of_equal_realized_risk=True))
        ref.save(args.output / "statistical_fallacy_scan.json", {
            "status": "ANALYZED_EXPLORATORY", "coverage": "11/11",
            "1_Simpson": "Original5/new3, native/mild/hard, depth, equal-object/pooled preserved. Object-capped quotas assess changing object allocation.",
            "2_ecological": "Object-level extrapolation, not independent per-target samples; repeated conditions remain clustered.",
            "3_Berkson": "Matched initializer complete-bank conditioning limits the population.",
            "4_collider": "Acceptance is an output, not a covariate for causal interpretation. Full-target AR remains the primary accuracy denominator.",
            "5_base_rate": "Conditional accepted risk accompanied by update and full-target regression rates; zero updates give NA.",
            "6_regression_to_mean": "All methods paired to same initializer and candidate bank; no extreme-error target selection.",
            "7_survivorship": "Added3 61/90 matched; unmatched targets not silently treated as covered.",
            "8_look_elsewhere": "All budgets, quota levels and orderings retained; no best-test-threshold claims or significance claims.",
            "9_forking_paths": "Posthoc audited data; plan locked before new selections; no new independent confirmation.",
            "10_causality": "Mechanistic controlled replay, not a randomized deployment study or guaranteed safety result.",
            "11_reverse_causality": "GT labels only calibrate development/evaluate held-out metrics; poisoning does not change quota selections."})
        draw(summary, args.output)
        make_report(args.output, summary, comparisons)
        snapshot = args.output / "source_snapshot"
        snapshot.mkdir()
        for path in audit_files():
            if path.suffix == ".py":
                shutil.copy2(path, snapshot / path.name)
        state.update(status="COMPLETE_EXPLORATORY_NOT_PROMOTED", stage="complete", elapsed_seconds=time.monotonic()-start,
            completed_at_utc=datetime.now(timezone.utc).isoformat(), outcome_rows=len(all_rows))
        state["artifact_hashes"] = {str(p.relative_to(args.output)): ref.base.digest(p) for p in args.output.rglob("*") if p.is_file() and p.name != "state.json"}
        ref.save(args.output / "state.json", state)
        print(summary.loc[summary.scope.eq("all") & summary.dataset.eq("combined8")].to_string(index=False), flush=True)
        print("COMPLETE", args.output, flush=True)
    except Exception:
        state.update(status="FAILED_PRESERVED_NO_AUTOMATIC_RETRY", error=traceback.format_exc())
        ref.save(args.output / "state.json", state)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/matched_risk_coverage_20261026_r1")
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--preflight", action="store_true")
    run(parser.parse_args())
