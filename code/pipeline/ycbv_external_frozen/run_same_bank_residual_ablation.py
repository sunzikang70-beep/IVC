"""Post-hoc same-bank residual ablation; no fitting or candidate generation."""
from __future__ import annotations

import os
for _key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[_key] = "1"

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import itertools
import json
from pathlib import Path
import shutil
import sys
import time
import traceback

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import train_bop_aware_state_residual as res
from winner_training_data_io import read_csv

base, v2 = res.base, res.v2
ROOT = Path(r"D:\6D_pose\codex_paper_workspace")
DEV = ROOT / "outputs/decoupled_development_20261020_r4"
OOF = DEV / "group_oof"
BANKS = {
    "holdout5": ROOT / "outputs/decoupled_frozen_unseen5_20261021_r4",
    "added3": ROOT / "outputs/decoupled_frozen_unseen3_20261023_r1",
}
OBJECTS = {
    "development22": [2, 3, 4, 6, 7, 9, 10, 11, 12, 14, 15, 16, 17, 19, 20, 23, 24, 25, 26, 28, 29, 30],
    "holdout5": [5, 13, 18, 21, 27], "added3": [1, 8, 22],
}
PRIMARY = ["A_delta_rank", "B_fixed_residual", "C_state_residual", "D_joint_full"]
BRIDGES = ["C_state_joint_gate", "D_joint_legacy_gate"]
ARMS = PRIMARY + BRIDGES
PAIRS = [("B_fixed_residual", "A_delta_rank"), ("C_state_residual", "B_fixed_residual"),
         ("D_joint_full", "C_state_residual"), ("D_joint_legacy_gate", "C_state_residual"),
         ("C_state_joint_gate", "C_state_residual"), ("D_joint_full", "D_joint_legacy_gate"),
         ("D_joint_full", "C_state_joint_gate")]
METRICS = ["AR_mean_row", *base.AR, "ADD_S_mm", "update", "regression", "bop_regression"]
SEED = 20261026


def save(path, value):
    base.dump(path, value)


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def same(a, b, message):
    if a != b:
        raise ValueError(message)


def bank(folder, development=False):
    manifest = read_csv(folder / ("locked_development_instances.csv" if development else "locked_holdout_instances.csv"))
    features = read_csv(folder / "observable_candidate_features.csv")
    if "init_regime" not in features:
        features = features.merge(manifest[["cohort", "frame_id", "init_regime"]].drop_duplicates(),
                                  on=["cohort", "frame_id"], validate="many_to_one")
    labels = read_csv(folder / ("development_bop_labels.csv" if development else "holdout_bop_labels.csv"))
    data = base.attach_labels(features, labels).reset_index(drop=True)
    counts = data.groupby("target_id").action.agg(list)
    if not counts.map(lambda x: sorted(x) == sorted(base.ACTIONS)).all():
        raise ValueError("Incomplete/duplicate action bank")
    for _, group in data.groupby("target_id"):
        if group[base.KEYS + ["cohort", "init_regime"]].nunique().max() != 1:
            raise ValueError("Target identity is not constant across actions")
    return data, features, manifest


def adjusted(data, predictions, bundle, arm):
    p = predictions.copy()
    if arm in ["A_delta_rank", "B_fixed_residual"]:
        corrections = bundle["base_bundle"]["calibration"]["corrections"]
        widths = np.zeros((len(data), 4)) if arm == "A_delta_rank" else np.asarray([corrections[a] for a in data.action])
        for i, target in enumerate(base.TARGETS):
            prefix = "ucb_" if i == 3 else "lcb_"
            p[prefix + target] = p["pred_" + target] + (1 if i == 3 else -1) * widths[:, i]
    return p


def choose(data, p, policy, arm, plan):
    if not data.index.equals(p.index):
        raise ValueError("Candidate index mismatch")
    lower = p[["lcb_" + t for t in base.TARGETS[:3]]].to_numpy()
    add = p.ucb_delta_ADD_S_mm.to_numpy()
    weight = policy["continuous_weight"]
    joint_score = arm in ["D_joint_full", "D_joint_legacy_gate"]
    joint_gate = arm in ["D_joint_full", "C_state_joint_gate"]
    if joint_score:
        if weight not in [0., plan["joint_continuous_weight"]]:
            raise ValueError("Uncalibrated joint utility coefficient")
        mean = p.lcb_delta_AR_mean.to_numpy()
        utility = mean if weight == 0 else p.lcb_delta_joint_utility.to_numpy()
    else:
        mean = lower.mean(axis=1)
        utility = mean - weight * add / 5.
    gate = ((mean > 0) | ((weight > 0) & (add <= -plan["continuous_minimum_ADD_gain_mm"]) & (mean >= -.001))) if joint_gate else (mean >= (-.001 if weight > 0 else 0))
    allowed = (base.mask_fit(data) & (data.observation_support.to_numpy() >= .01)
               & (p.training_support.to_numpy() >= .2)
               & (p.adds_risk_upper.to_numpy() <= policy["risk_limit"])
               & (add <= 0) & (lower.min(axis=1) >= -policy["component_loss"])
               & gate & (utility > 0) & np.isfinite(utility) & np.isfinite(add)
               & np.isfinite(lower).all(axis=1) & (not policy.get("disabled", False)))
    # Stable global ordering gives the same utility/risk/action tie break as r4.
    ranked = pd.DataFrame({"target_id": data.target_id.to_numpy(), "action": data.action.to_numpy(),
                           "utility": utility, "risk": p.adds_risk_upper.to_numpy()})
    ranked = ranked.loc[allowed].sort_values(["utility", "risk", "action"], ascending=[False, True, True], kind="stable")
    winners = ranked.drop_duplicates("target_id").set_index("target_id").action.to_dict()
    return {int(t): winners.get(t, base.INITIAL) for t in data.target_id.unique()}


def evaluate(data, choices):
    initial = data.drop_duplicates("target_id").copy().set_index("target_id", drop=False)
    selected = initial.copy()
    selected["selected_action"] = base.INITIAL
    for c in ["ADD_S_mm", *base.AR]:
        selected[c] = initial["initial_" + c]
    selected["delta_ADD_S_mm"] = 0.
    requested = pd.DataFrame({"target_id": list(choices), "action": list(choices.values())})
    requested = requested.loc[requested.action.ne(base.INITIAL)]
    picked = requested.merge(data, on=["target_id", "action"], validate="one_to_one").set_index("target_id", drop=False)
    same(len(picked), len(requested), "Selected action absent from candidate bank")
    for c in ["ADD_S_mm", *base.AR, "delta_ADD_S_mm"]:
        selected.loc[picked.index, c] = picked[c]
    selected.loc[picked.index, "selected_action"] = picked.action
    keep = ["target_id", *base.KEYS, "cohort", "init_regime", "selected_action", "ADD_S_mm", "initial_ADD_S_mm",
            "delta_ADD_S_mm", *base.AR, *["initial_" + c for c in base.AR]]
    result = selected[keep].reset_index(drop=True)
    result["AR_mean_row"] = result[base.AR].mean(axis=1)
    result["AR_gain"] = result.AR_mean_row - result[["initial_" + c for c in base.AR]].mean(axis=1)
    result["update"] = result.selected_action.ne(base.INITIAL).astype(int)
    result["regression"] = result.delta_ADD_S_mm.gt(1e-9).astype(int)
    result["bop_regression"] = result.AR_gain.lt(-1e-12).astype(int)
    return result


def tune(data, p, arm, plan):
    records = []
    for risk, loss, weight in itertools.product(plan["risk_limits"], plan["component_ar_loss_limits"], plan["continuous_gain_weights"]):
        policy = dict(risk_limit=risk, component_loss=loss, continuous_weight=weight, disabled=False)
        rows = evaluate(data, choose(data, p, policy, arm, plan))
        active = rows.loc[rows["update"].eq(1)]
        risk_rate = float(active.regression.mean()) if len(active) else 1.
        groups = rows.groupby("cohort")
        cohort_check = all(g.AR_gain.mean() >= -1e-12 and g.delta_ADD_S_mm.mean() <= 1e-12 for _, g in groups)
        objective = float(np.mean([g.AR_gain.mean() - .005 * g.delta_ADD_S_mm.mean() / 5 for _, g in groups]))
        passes = rows["update"].mean() >= .05 and risk_rate <= .2 and cohort_check
        records.append(dict(**policy, coverage=float(rows["update"].mean()), accepted_regression_rate=risk_rate,
                            equal_cohort_joint_gain=objective, every_cohort_nonregressed=bool(cohort_check),
                            passes_development_constraints=bool(passes)))
    feasible = [r for r in records if r["passes_development_constraints"]]
    if not feasible:
        return dict(risk_limit=.1, component_loss=0., continuous_weight=0., disabled=True), records
    best = min(feasible, key=lambda r: (-r["equal_cohort_joint_gain"], r["accepted_regression_rate"], -r["coverage"], r["risk_limit"]))
    return {k: best[k] for k in ["risk_limit", "component_loss", "continuous_weight", "disabled"]}, records


def evidence_paths():
    files = [DEV / n for n in ["observable_candidate_features.csv", "development_bop_labels.csv", "locked_development_instances.csv", "feature_schema.json"]]
    files += [OOF / n for n in ["state.json", "development_candidate.joblib", "outer_oof_predictions.csv"]]
    for fold in range(5):
        f = OOF / f"fold_{fold}"
        files += [f / "outer_model.joblib", f / "inner_policy_oof_predictions.csv"]
        files += [f / f"inner_{i}.joblib" for i in range(3)]
    for folder in BANKS.values():
        files += [folder / n for n in ["observable_candidate_features.csv", "holdout_bop_labels.csv", "locked_holdout_instances.csv", "feature_schema.json", "frozen_outcomes.csv", "state.json", "frozen_model_receipt.json"]]
    files += [Path(m.__file__) for m in [base, v2, res, res.selective, base.schema]]
    files += [Path(__file__), Path(__file__).with_name("winner_training_data_io.py"),
              Path(__file__).with_name("test_same_bank_residual_ablation.py"),
              Path(__file__).with_name("bop_aware_state_residual_plan_20261019.json")]
    return list(dict.fromkeys(files))


def verify_saved(path, state_root):
    state = load(state_root / "state.json")
    expected = state.get("artifact_hashes", {}).get(str(path.relative_to(state_root)))
    if expected is None:
        raise ValueError("No original hash for artifact: " + str(path))
    same(base.digest(path), expected, "Original artifact hash mismatch: " + str(path))


def preflight():
    hashes = {str(p): base.digest(p) for p in evidence_paths()}
    for path in evidence_paths():
        if path.is_relative_to(OOF) and path.name != "state.json":
            verify_saved(path, OOF)
        for folder in BANKS.values():
            if path.is_relative_to(folder) and path.name not in ["state.json", "frozen_model_receipt.json"]:
                verify_saved(path, folder)
    frozen = joblib.load(OOF / "development_candidate.joblib")
    same(frozen["bundle"]["training_objects"], OBJECTS["development22"], "Training object set changed")
    same(base.digest(OOF / "development_candidate.joblib"), "a3836d3bfbb0dda4927578fdfb53389830fe0cff2bdc439f46de8cdcb2a5917c", "Frozen r4 model changed")
    inventory = []
    for name, folder in [("development22", DEV), *BANKS.items()]:
        data, _, _ = bank(folder, name == "development22")
        same(sorted(map(int, data.object_id.unique())), OBJECTS[name], "Bank object set mismatch")
        inventory.append(dict(bank=name, candidates=len(data), target_conditions=data.target_id.nunique(),
                              physical_target_instances=data.frame_id.nunique(),
                              unique_rgbd_images=len(data[["scene_id", "img_id"]].drop_duplicates()), objects=OBJECTS[name]))
    return hashes, frozen, inventory


def verify_replay(data, p, bundle, plan, original, policy):
    checks = {}
    for arm, variant in [("C_state_residual", "conditional_residual_legacy"), ("D_joint_full", "conditional_residual_joint")]:
        choices = choose(data, p, policy, arm, plan)
        same(choices, res.decisions(data, p, policy, variant, plan), "r4 decision replay differs")
        rows = evaluate(data, choices)
        prior = original.loc[original.method.eq(variant)]
        keys = ["cohort", *base.KEYS]
        pair = rows.merge(prior, on=keys, suffixes=("_new", "_old"), validate="one_to_one")
        same(len(pair), len(rows), "r4 outcome replay incomplete")
        same(pair.selected_action_new.tolist(), pair.selected_action_old.tolist(), "r4 chosen actions differ")
        np.testing.assert_allclose(pair[[c + "_new" for c in ["ADD_S_mm", *base.AR]]],
                                   pair[[c + "_old" for c in ["ADD_S_mm", *base.AR]]], atol=1e-10, rtol=1e-10)
        checks[arm] = dict(exact_chosen_actions=True, metrics_atol=1e-10, targets=len(rows))
    shadow = data.copy()
    for col in [*base.TARGETS, *base.CLASSES, *base.AR, "ADD_S_mm", "initial_ADD_S_mm", *["initial_" + c for c in base.AR]]:
        shadow[col] = 99999.
    observable = data.drop(columns=[c for c in shadow if c in [*base.TARGETS, *base.CLASSES, *base.AR, "ADD_S_mm", "initial_ADD_S_mm", *["initial_" + c for c in base.AR]]])
    clean = res.predict_bundle(bundle, observable)
    np.testing.assert_allclose(clean, res.predict_bundle(bundle, shadow), atol=1e-9, rtol=1e-9)
    first_target = data.loc[data.target_id.eq(data.target_id.iloc[0])]
    np.testing.assert_allclose(clean.loc[first_target.index], res.predict_bundle(bundle, first_target), atol=1e-9, rtol=1e-9)
    for arm in ARMS:
        adjusted_p = adjusted(data, clean, bundle, arm)
        same(choose(data, adjusted_p, policy, arm, plan), choose(shadow, adjusted_p, policy, arm, plan), "Label-dependent choices")
    checks.update(label_free_predictions=True, label_blind_choices_all_six_arms=True, single_target_batch_invariance=True)
    return checks


def produce(data, p, bundle, policies, plan, dataset, regime, fold=-1):
    outputs = []
    for arm in ARMS:
        ap = adjusted(data, p, bundle, arm)
        rows = evaluate(data, choose(data, ap, policies[arm], arm, plan))
        rows["arm"], rows["policy_mode"], rows["dataset"], rows["outer_fold"] = arm, regime, dataset, fold
        outputs.append(rows)
    initial = evaluate(data, {int(t): base.INITIAL for t in data.target_id.unique()})
    initial["arm"], initial["policy_mode"], initial["dataset"], initial["outer_fold"] = "initializer", regime, dataset, fold
    outputs.append(initial)
    return outputs


def scopes(table):
    yield "all", table
    for axis in ["init_regime", "input_depth_condition"]:
        for value, group in table.groupby(axis):
            yield axis + ":" + str(value), group


def summaries(outcomes):
    records = []
    for (dataset, mode, arm), full in outcomes.groupby(["dataset", "policy_mode", "arm"], sort=False):
        for scope, group in scopes(full):
            obj = group.groupby("object_id")[METRICS + ["AR_gain", "delta_ADD_S_mm"]].mean()
            updates, bad = int(group["update"].sum()), int(group.regression.sum())
            records.append(dict(dataset=dataset, policy_mode=mode, arm=arm, scope=scope,
                targets=len(group), objects=len(obj), physical_target_instances=group.frame_id.nunique(),
                unique_rgbd_images=len(group[["scene_id", "img_id"]].drop_duplicates()),
                AR_pooled=float(group.AR_mean_row.mean()), AR_equal_object=float(obj.AR_mean_row.mean()),
                AR_gain_pooled=float(group.AR_gain.mean()), AR_gain_equal_object=float(obj.AR_gain.mean()),
                ADD_S_mm=float(group.ADD_S_mm.mean()), ADD_delta_mm=float(group.delta_ADD_S_mm.mean()),
                updates=updates, update_fraction=float(group["update"].mean()), accepted_regressions=bad,
                accepted_regression_rate=bad / updates if updates else np.nan,
                all_target_regression_rate=float(group.regression.mean()),
                bop_regression_rate=float(group.bop_regression.mean()),
                **{c: float(group[c].mean()) for c in base.AR}))
    return pd.DataFrame(records)


def holm(values):
    a = np.asarray(values, float)
    order = np.argsort(a)
    result = np.empty(len(a))
    result[order] = np.minimum(1, np.maximum.accumulate(a[order] * np.arange(len(a), 0, -1)))
    return result


def paired_analysis(outcomes, samples):
    rng = np.random.default_rng(SEED)
    records, object_rows = [], []
    for (dataset, mode), full in outcomes.groupby(["dataset", "policy_mode"], sort=False):
        for scope, group in scopes(full):
            keys = ["source_bank", "target_id", "object_id"]
            for candidate, reference in PAIRS:
                x, y = group.loc[group.arm.eq(candidate)], group.loc[group.arm.eq(reference)]
                pair = x.merge(y, on=keys, suffixes=("_x", "_y"), validate="one_to_one")
                same(len(pair), len(x), "Paired comparison lost targets")
                objects = sorted(pair.object_id.unique())
                indices = rng.integers(0, len(objects), (samples, len(objects)))
                signs = np.asarray(list(itertools.product([-1., 1.], repeat=len(objects)))) if len(objects) <= 12 else rng.choice([-1., 1.], (samples, len(objects)))
                for metric in METRICS:
                    difference = pair[metric + "_x"] - pair[metric + "_y"]
                    means = difference.groupby(pair.object_id).mean().reindex(objects).to_numpy()
                    counts = pair.groupby("object_id").size().reindex(objects).to_numpy()
                    dist = means[indices].mean(axis=1)
                    pooled = (means[indices] * counts[indices]).sum(axis=1) / counts[indices].sum(axis=1)
                    lo, hi = np.quantile(dist, [.025, .975])
                    plo, phi = np.quantile(pooled, [.025, .975])
                    null = np.abs((signs * means).mean(axis=1))
                    exceed = int((null >= abs(means.mean()) - 1e-15).sum())
                    pvalue = exceed / len(signs) if len(objects) <= 12 else (exceed + 1) / (len(signs) + 1)
                    records.append(dict(dataset=dataset, policy_mode=mode, scope=scope, candidate=candidate, reference=reference,
                        metric=metric, object_clusters=len(objects), target_conditions=len(pair),
                        equal_object_difference=float(means.mean()), ci_low=float(lo), ci_high=float(hi),
                        pooled_difference=float(difference.mean()), pooled_ci_low=float(plo), pooled_ci_high=float(phi),
                        object_signflip_p=float(pvalue), signflip_exact=len(objects) <= 12,
                        candidate_better_objects=int((means > 1e-12).sum()) if metric not in ["ADD_S_mm", "regression", "bop_regression"] else int((means < -1e-12).sum()),
                        tied_objects=int((np.abs(means) <= 1e-12).sum()), inference="exploratory_posthoc_object_cluster"))
                    if scope == "all":
                        for oid, value, count in zip(objects, means, counts):
                            object_rows.append(dict(dataset=dataset, policy_mode=mode, candidate=candidate, reference=reference,
                                                    metric=metric, object_id=int(oid), paired_difference=float(value), targets=int(count)))
    frame = pd.DataFrame(records)
    frame["holm_p_primary_three_AR"] = np.nan
    primary_pairs = set(PAIRS[:3])
    for _, group in frame.loc[frame.scope.eq("all") & frame.metric.eq("AR_mean_row")].groupby(["dataset", "policy_mode"]):
        selected = group.loc[[tuple(row) in primary_pairs for row in group[["candidate", "reference"]].to_numpy()]]
        frame.loc[selected.index, "holm_p_primary_three_AR"] = holm(selected.object_signflip_p)
    return frame, pd.DataFrame(object_rows)


def transition_table(outcomes):
    rows = []
    for (dataset, mode), full in outcomes.groupby(["dataset", "policy_mode"], sort=False):
        for candidate, reference in PAIRS:
            x, y = full.loc[full.arm.eq(candidate)], full.loc[full.arm.eq(reference)]
            p = x.merge(y, on=["source_bank", "target_id", "object_id"], suffixes=("_x", "_y"), validate="one_to_one")
            kind = np.where(p.selected_action_x.eq(p.selected_action_y), "same_choice",
                            np.where(p.update_x.eq(0), "newly_rejected", np.where(p.update_y.eq(0), "newly_accepted", "different_action")))
            p["transition"] = kind
            for transition, g in p.groupby("transition"):
                rows.append(dict(dataset=dataset, policy_mode=mode, candidate=candidate, reference=reference,
                                 transition=transition, targets=len(g), objects=g.object_id.nunique(),
                                 AR_difference=float((g.AR_mean_row_x - g.AR_mean_row_y).mean()),
                                 ADD_difference_mm=float((g.ADD_S_mm_x - g.ADD_S_mm_y).mean()),
                                 candidate_regressions=int(g.regression_x.sum()), reference_regressions=int(g.regression_y.sum())))
    return pd.DataFrame(rows)


def plots(summary, object_rows, output):
    view = summary.loc[summary.dataset.eq("combined8") & summary.policy_mode.eq("shared_r4") & summary.scope.eq("all")].set_index("arm")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), constrained_layout=True)
    colors = ["#667782", "#cd8042", "#269781", "#a4526d"]
    for ax, metric, title, scale in zip(axes, ["AR_gain_equal_object", "update_fraction", "all_target_regression_rate"],
                                       ["AR gain vs initializer (pp)", "Update fraction (%)", "ADD-S regressions / all targets (%)"], [100, 100, 100]):
        ax.bar(["A", "B", "C", "D"], view.loc[PRIMARY, metric] * scale, color=colors)
        ax.set_title(title, fontsize=10)
        ax.axhline(0, color="black", lw=.6)
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(output / "same_bank_four_stages.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True, sharey=True)
    for ax, (candidate, reference) in zip(axes, PAIRS[:3]):
        group = object_rows.loc[object_rows.dataset.eq("combined8") & object_rows.policy_mode.eq("shared_r4")
                                & object_rows.metric.eq("AR_mean_row") & object_rows.candidate.eq(candidate)
                                & object_rows.reference.eq(reference)].sort_values("object_id")
        ax.barh(group.object_id.astype(str), group.paired_difference * 100, color=np.where(group.paired_difference >= 0, "#269781", "#b44f62"))
        ax.axvline(0, color="black", lw=.6)
        ax.set_title(candidate[0] + " minus " + reference[0])
        ax.set_xlabel("Paired AR difference (pp)")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Held-out object ID")
    fig.savefig(output / "paired_object_module_contributions.png", dpi=180)
    plt.close(fig)


def report(output, summary, comparisons, inventory, final_policies):
    lines = ["# 相同候选库下的模块独立贡献审计", "", "## 实验性质", "",
             "本次为已检查数据上的事后受控消融（ANALYZED），不是新的独立确认试验。未训练模型、未生成候选、未上云。",
             "所有方法共享六个候选、初始位姿、Mask/深度、191个可观测特征、增量回归头、概率校准和基础安全门。",
             "保留初始化器作为共同回退。A 是普通预测增量排序并保留共同安全门，不是无约束强制更新。",
             "B 使用每个 action 各自固定的训练侧 q75 残差；C 使用状态条件化边际残差；D 使用直接校准的 mean/joint 标量残差及原 joint 分支。",
             "这些都是经验性残差修正，不是形式化覆盖率、联合置信集或安全保证。", "", "## 控制与阈值", "",
             "shared_r4：六种对照使用完全相同的已冻结 r4 策略（risk=0.3, component_loss=0.01, weight=0.005）。这是模块归因的主比较。",
             "nested_equal_budget：每一方法使用相同的12组策略和原开发约束；外折策略只在内折物体OOF上选择，部署策略只在22物体外折OOF上选择。",
             "开发集 shared_r4 为回顾性诊断，因为该固定策略曾在整个开发集上选择；nested_equal_budget 才是策略选择也嵌套的开发估计。",
             "留出集从不选择策略；但这些留出结果此前已经检查，所以本次不能宣称预注册或再次确认泛化。", "", "## 数据清单", "",
             "| 数据库 | 物体 | 实体目标实例 | 独立RGB-D图像数 | 目标条件 | 候选行 |",
             "|---|---:|---:|---:|---:|---:|"]
    for r in inventory:
        lines.append(f"| {r['bank']} | {len(r['objects'])} | {r['physical_target_instances']} | {r['unique_rgbd_images']} | {r['target_conditions']} | {r['candidates']} |")
    lines += ["", "目标条件包含初始化难度和深度条件重复测量，不可当作独立样本。表中仅为共同完整候选库，缺失外部初始化器的计划实例没有被补成成功。",
              "新增3物体原计划90个目标、仅61个匹配；本次是匹配候选库条件下的消融，不是覆盖完整输入流水线的指标。", "", "## 四阶段结果", "",
              "AR 为原官方误差标签的逐目标阈值召回汇总；此处重新选择已有位姿并汇总，没有重跑渲染。AR数值0到1，提升列单位为百分点。",
              "| 数据/策略 | 阶段 | AR池化 | AR等物体 | 相对初始AR提升(pp,等物体) | ADD-S变化(mm) | 更新率 | 接受后ADD-S回归率 | 全目标ADD-S回归率 |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for _, r in summary.loc[summary.scope.eq("all") & summary.arm.isin(PRIMARY) & summary.dataset.ne("development22")].iterrows():
        risk = "NA" if pd.isna(r.accepted_regression_rate) else f"{r.accepted_regression_rate:.2%}"
        lines.append(f"| {r.dataset}/{r.policy_mode} | {r.arm} | {r.AR_pooled:.6f} | {r.AR_equal_object:.6f} | {100*r.AR_gain_equal_object:+.4f} | {r.ADD_delta_mm:+.4f} | {r.update_fraction:.2%} | {risk} | {r.all_target_regression_rate:.2%} |")
    lines += ["", "## 配对独立增量", "", "主分析以物体等权；95%区间为物体簇配对bootstrap。符号翻转p仅为探索性敏感度，不是随机实验因果推断。",
              "同一场景可出现多个物体，物体簇也不是完全独立采样；仅8个物体，区间和p值都应谨慎。",
              "| 数据/策略 | 对比 | AR增量(pp,等物体) | 95%区间(pp) | AR增量(pp,池化) | Holm p（三个主对比） |",
              "|---|---|---:|---:|---:|---:|"]
    pairs = comparisons.loc[comparisons.scope.eq("all") & comparisons.metric.eq("AR_mean_row") & comparisons.dataset.ne("development22")]
    for _, r in pairs.iterrows():
        h = "NA" if pd.isna(r.holm_p_primary_three_AR) else f"{r.holm_p_primary_three_AR:.4f}"
        lines.append(f"| {r.dataset}/{r.policy_mode} | {r.candidate} - {r.reference} | {100*r.equal_object_difference:+.4f} | [{100*r.ci_low:+.4f}, {100*r.ci_high:+.4f}] | {100*r.pooled_difference:+.4f} | {h} |")
    lines += ["", "## joint 拆解", "",
              "C_state_joint_gate 与 C_state_residual 比较：只更换接受分支，边际残差与排序分数不变。",
              "D_joint_legacy_gate 与 C_state_residual 比较：接受分支相同，只改成直接校准的 mean/joint 残差评分。",
              "D_joint_full 与 D_joint_legacy_gate 比较：直接校准的分数相同，只改变接受分支。",
              "D_joint_full 与 C_state_joint_gate 比较：joint 分支相同，比较直接标量残差与边际残差合成。",
              "这些是条件贡献，存在交互作用，不能把百分比简单相加或当成普适贡献比例。", "", "## 解释边界", "",
              "精度和风险一起判断：接受后回归率必须连同更新率与全目标回归率报告。更低回归率可能只是拒绝更多更新。",
              "若C对B稳定有利而D对C很小，论文主要证据应归于状态条件化残差，joint只作为辅助机制。若新增3物体方向不一致，不隐去该结果。",
              "不根据本次留出结果修改门限或挑选方法。不自动晋升方法。下一次任何设计调整都需要新的独立验证。", "", "## 材料护照", "",
              "输入：冻结r4 joblib、既有开发/留出候选CSV、官方BOP逐位姿误差汇总标签。",
              "变换：只替换残差边界/评分/分支，重放既有选择；策略重选只使用开发OOF。",
              "输出：逐目标选择、完整配对统计、物体级差值、分层汇总、选择转移和图。",
              "校验：原始artifact hash、源文件锁、训练/验证物体隔离、旧C/D重放、标签污染不变性、单目标batch不变性。",
              "来源与可复现性：locked_plan.json、input_hashes.json、checks.json、frozen_development_policies.json、source_snapshot/、state.json。",
              "统计问题清单：statistical_audit.json。所有小样本、多比较、选择偏倚和重复测量限制保留。", ""]
    (output / "report_zh.md").write_text("\n".join(lines), encoding="utf-8")


def run(args):
    hashes, frozen, inventory = preflight()
    if args.preflight:
        print(json.dumps(dict(status="PREFLIGHT_OK", artifacts=len(hashes), inventory=inventory), indent=2), flush=True)
        return
    if args.output.exists():
        raise FileExistsError("Preserve existing output: " + str(args.output))
    plan = load(Path(__file__).with_name("bop_aware_state_residual_plan_20261019.json"))
    common = frozen["policies"]["conditional_residual_joint"]
    same(common, frozen["policies"]["conditional_residual_legacy"], "Primary frozen policies differ")
    same(common, dict(risk_limit=.3, component_loss=.01, continuous_weight=.005, disabled=False), "Common frozen policy changed")
    locked = dict(status="LOCKED_BEFORE_NEW_ABLATION_SELECTIONS", arms=ARMS, primary=PRIMARY, paired_comparisons=PAIRS,
        fixed_policy=common, threshold_grid=12, raw_heads_refit=False, candidate_generation=False, holdout_tuning=False,
        bootstrap_samples=args.bootstrap, seed=SEED, inventory=inventory, state_residual_plan=plan,
        primary_estimand="equal_object_paired_AR_mean; pooled_secondary", joint_bridge="score_by_gate_2x2",
        inference_scope="retrospective_exploratory_not_new_confirmation", source_model_sha256=hashes[str(OOF / "development_candidate.joblib")])
    args.output.mkdir(parents=True)
    save(args.output / "locked_plan.json", locked)
    save(args.output / "input_hashes.json", hashes)
    save(args.output / "environment.json", dict(python=sys.version, executable=sys.executable,
         packages={k: version(k) for k in ["numpy", "pandas", "scipy", "scikit-learn", "joblib", "matplotlib"]}))
    state = dict(status="RUNNING", started_at_utc=datetime.now(timezone.utc).isoformat(), stage="development", pid=os.getpid(),
                 method_promoted=False, new_independent_confirmation=False, experiment_status="ANALYZED")
    save(args.output / "state.json", state)
    start = time.monotonic()
    try:
        outputs, grids, checks = [], [], {}
        dev, _, _ = bank(DEV, True)
        saved = read_csv(OOF / "outer_oof_predictions.csv", index_col="candidate_index")
        same(saved.index.tolist(), dev.index.tolist(), "Saved OOF index contract differs")
        for c in ["target_id", "object_id", "cohort", "frame_id", "action"]:
            same(saved[c].tolist(), dev[c].tolist(), "Saved OOF identity differs: " + c)
        final_inputs = {a: [] for a in ARMS}
        for fold, (tr, va) in enumerate(base.folds(dev, 5)):
            training, held = dev.iloc[tr], dev.iloc[va]
            folder = OOF / f"fold_{fold}"
            outer = joblib.load(folder / "outer_model.joblib")
            same(sorted(held.object_id.unique()), outer["partition"]["validation_objects"], "Outer split changed")
            same(sorted(training.object_id.unique()), outer["bundle"]["training_objects"], "Outer training changed")
            inner = read_csv(folder / "inner_policy_oof_predictions.csv", index_col="candidate_index").loc[training.index]
            per_arm = {a: [] for a in ARMS}
            for inner_id in range(3):
                m = joblib.load(folder / f"inner_{inner_id}.joblib")
                valid_objects = m["partition"]["validation_objects"]
                fit_objects = m["bundle"]["training_objects"]
                if set(valid_objects) & set(fit_objects) or not set(valid_objects + fit_objects) <= set(training.object_id):
                    raise ValueError("Inner object leakage")
                subset = training.loc[training.object_id.isin(valid_objects)]
                for arm in ARMS:
                    per_arm[arm].append(adjusted(subset, inner.loc[subset.index], m["bundle"], arm))
            nested = {}
            p = saved.loc[held.index]
            if not p.outer_fold.eq(fold).all():
                raise ValueError("Cached OOF fold assignment differs")
            for arm in ARMS:
                ip = pd.concat(per_arm[arm]).sort_index().loc[training.index]
                same(len(ip), len(training), "Inner policy predictions incomplete")
                same(ip.index.tolist(), training.index.tolist(), "Inner candidate identity differs")
                nested[arm], rows = tune(training, ip, arm, plan)
                grids += [dict(r, arm=arm, outer_fold=fold) for r in rows]
                final_inputs[arm].append(adjusted(held, p, outer["bundle"], arm))
            # The nested C/D policies must reproduce the original inner selection.
            for arm, variant in [("C_state_residual", "conditional_residual_legacy"), ("D_joint_full", "conditional_residual_joint")]:
                same(nested[arm], outer["policies"][variant], "Original nested policy replay differs")
                same(choose(held, p, nested[arm], arm, plan), res.decisions(held, p, nested[arm], variant, plan), "OOF decision replay differs")
            outputs += produce(held, p, outer["bundle"], {a: common for a in ARMS}, plan, "development22", "shared_r4", fold)
            outputs += produce(held, p, outer["bundle"], nested, plan, "development22", "nested_equal_budget", fold)
            print(f"DEVELOPMENT outer {fold+1}/5 complete; no model refit; elapsed={time.monotonic()-start:.1f}s", flush=True)
        final_policies = {}
        for arm in ARMS:
            op = pd.concat(final_inputs[arm]).sort_index().loc[dev.index]
            final_policies[arm], rows = tune(dev, op, arm, plan)
            grids += [dict(r, arm=arm, outer_fold="final_development_only") for r in rows]
        for arm, variant in [("C_state_residual", "conditional_residual_legacy"), ("D_joint_full", "conditional_residual_joint")]:
            same(final_policies[arm], frozen["policies"][variant], "Final original policy replay differs")
        save(args.output / "frozen_development_policies.json", final_policies)
        pd.DataFrame(grids).to_csv(args.output / "development_policy_grids.csv", index=False)
        state.update(stage="holdout_inference_no_tuning", development_policy_lock_sha256=base.digest(args.output / "frozen_development_policies.json"))
        save(args.output / "state.json", state)
        for name, folder in BANKS.items():
            data, _, _ = bank(folder)
            p = res.predict_bundle(frozen["bundle"], data)
            original = read_csv(folder / "frozen_outcomes.csv")
            checks[name] = verify_replay(data, p, frozen["bundle"], plan, original, common)
            outputs += produce(data, p, frozen["bundle"], {a: common for a in ARMS}, plan, name, "shared_r4")
            outputs += produce(data, p, frozen["bundle"], final_policies, plan, name, "nested_equal_budget")
            print(f"{name} all six arms evaluated; r4 replay and GT-blind checks passed", flush=True)
        outcomes = pd.concat(outputs, ignore_index=True)
        outcomes["source_bank"] = outcomes.dataset
        combined = outcomes.loc[outcomes.dataset.isin(BANKS)].copy()
        combined["dataset"] = "combined8"
        outcomes = pd.concat([outcomes, combined], ignore_index=True)
        outcomes.to_csv(args.output / "per_target_outcomes.csv", index=False)
        summary = summaries(outcomes)
        summary.to_csv(args.output / "module_summary.csv", index=False)
        comparisons, object_rows = paired_analysis(outcomes, args.bootstrap)
        comparisons.to_csv(args.output / "paired_module_contributions.csv", index=False)
        object_rows.to_csv(args.output / "paired_object_differences.csv", index=False)
        transition_table(outcomes).to_csv(args.output / "choice_transitions.csv", index=False)
        checks.update(input_hashes_unchanged={p: base.digest(Path(p)) == sha for p, sha in hashes.items()},
                      same_six_actions_per_target=True, cached_models_only=True, old_nested_policy_replay=True,
                      holdout_policy_file_unchanged=base.digest(args.output / "frozen_development_policies.json") == state["development_policy_lock_sha256"])
        if not all(checks["input_hashes_unchanged"].values()) or not checks["holdout_policy_file_unchanged"]:
            raise ValueError("Frozen input or policy changed during audit")
        save(args.output / "checks.json", checks)
        save(args.output / "statistical_audit.json", {
            "1_distribution_and_outliers": "Bounded AR; mm tails retained without winsorization. Paired cluster bootstrap, no Gaussian SE.",
            "2_independence": "Repeated depth/init conditions clustered by object; targets and images counted separately. Objects share scenes, residual dependence remains.",
            "3_sample_size": "22 development and 5+3 inspected holdout objects. Combined8 is descriptive; 3-object intervals fragile.",
            "4_hypothesis_and_estimand": "Three directional scientific comparisons, two-sided exploratory intervals and signflips. Equal-object primary, pooled secondary.",
            "5_multiple_testing": "Holm for three primary overall AR comparisons within each bank/policy. Bridges, subgroups, metrics exploratory uncorrected.",
            "6_effect_size": "Absolute paired AR pp, ADD-S mm, update and regression rates; no percent-of-gain claims.",
            "7_missingness": "Common complete candidate bank conditional on matched initializer; no coverage inflation. Added3 61/90 planned matched targets.",
            "8_leakage_and_selection": "No holdout tuning; same 12 development-only policies per arm, nested object folds. Shared fixed dev policy is retrospective.",
            "9_model_assumptions": "Raw/regression/probability heads held constant. q75 residual guards empirical, not confidence or conformal guarantees.",
            "10_robustness_and_negative_results": "Original5/new3 separate, native/mild/hard and depth strata, objectwise differences and adverse changes retained.",
            "11_reproducibility": "Input/source SHA256 before and after, same-bank identities, cached model partitions, old C/D replay, GT poisoning, fixed seed; no auto promotion.",
            "status": "ANALYZED_POSTHOC_NOT_PREREGISTERED", "training_reproduced": False})
        plots(summary, object_rows, args.output)
        report(args.output, summary, comparisons, inventory, final_policies)
        snapshot = args.output / "source_snapshot"
        snapshot.mkdir()
        for path in evidence_paths():
            if path.suffix == ".py":
                shutil.copy2(path, snapshot / path.name)
        state.update(status="COMPLETE_POSTHOC_ABLATION_NOT_PROMOTED", stage="complete",
                     completed_at_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic()-start,
                     input_files=len(hashes), inventory=inventory)
        state["artifact_hashes"] = {str(p.relative_to(args.output)): base.digest(p) for p in args.output.rglob("*") if p.is_file() and p.name != "state.json"}
        save(args.output / "state.json", state)
        print(summary.loc[summary.scope.eq("all") & summary.dataset.eq("combined8")].to_string(index=False), flush=True)
        print("COMPLETE", args.output, flush=True)
    except Exception:
        state.update(status="FAILED_PRESERVED_NO_AUTOMATIC_RETRY", error=traceback.format_exc())
        save(args.output / "state.json", state)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/same_bank_residual_ablation_20261026_r1")
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--preflight", action="store_true")
    run(parser.parse_args())
