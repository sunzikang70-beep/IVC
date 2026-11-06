"""Object-equal analysis of the frozen r4 YCB-V external validation.

numpy/pandas only, so it can be re-run anywhere. The estimands mirror
outputs/p0_main_table_baselines_20261104/source/build_p0_main_table.py, so the
YCB-V table and the 8-object T-LESS table are directly comparable:

* every arm is evaluated on the same matched target conditions
  (physical target x init regime x depth condition);
* AR is the mean of the per-row VSD/MSSD/MSPD recalls, averaged first within an
  object and then over objects (object-equal); pooled target means are secondary;
* paired differences are resampled by object (21 clusters), 20000 draws,
  seed 20261104;
* an accepted regression is an update whose ADD-S increases by more than 1e-9 mm.

Usage:  python analyze_decoupled_frozen_external_ycbv.py <run_output_dir>
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

INITIAL = "protocolB_e2e_initial_pose"
FIXED = {
    "protocolF_se3_trust_p2p_s025": "P2P s0.25",
    "protocolF_se3_trust_p2p_s050": "P2P s0.50",
    "protocolF_se3_trust_p2p_s100": "P2P s1.00",
    "protocolF_se3_trust_filtered_s025": "Filtered P2P s0.25",
    "protocolF_se3_trust_filtered_s050": "Filtered P2P s0.50",
    "protocolF_se3_trust_filtered_s100": "Filtered P2P s1.00",
}
FROZEN_NAMES = {
    INITIAL: "Initial",
    **FIXED,
    "selected_risk_stratified": "Locked r4 primary (selected_risk_stratified)",
    "conditional_residual_joint": "D joint (frozen run)",
}
ARM_NAMES = {
    "A_delta_rank": "A ordinary ranking",
    "B_fixed_residual": "B fixed residual",
    "C_state_residual": "C state residual",
    "D_joint_full": "D joint (ours)",
}
ORDER = ["Initial", "Locked r4 primary (selected_risk_stratified)", *FIXED.values(),
         "A ordinary ranking", "B fixed residual", "C state residual", "D joint (ours)",
         "D joint (frozen run)"]
OURS = "D joint (ours)"
KEY = ["object_id", "scene_id", "img_id", "inst_id", "input_depth_condition", "init_regime"]
AR_COLS = ["AR_VSD_row", "AR_MSSD_row", "AR_MSPD_row"]
DRAWS = 20000
SEED = 20261104
# Pre-specified in locked_protocol.json before any r4 output existed on YCB-V.
NONINFERIORITY_MARGIN_PP = -0.5


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load(out: Path) -> tuple[pd.DataFrame, dict]:
    fr = pd.read_csv(out / "frozen_outcomes.csv")
    fr = fr.loc[fr.method.isin(FROZEN_NAMES)].copy()
    fr["arm"] = fr.method.map(FROZEN_NAMES)
    fr["update"] = fr.selected_action.ne(INITIAL).astype(int)
    fr["regression"] = (fr["update"].eq(1) & fr.delta_ADD_S_mm.gt(1e-9)).astype(int)

    sb = pd.read_csv(out / "same_bank_outcomes.csv")
    sb = sb.loc[sb.policy_mode.eq("shared_r4") & sb.arm.isin(ARM_NAMES)].copy()
    sb["arm"] = sb.arm.map(ARM_NAMES)

    cols = KEY + ["arm", "selected_action", "AR_mean_row", *AR_COLS, *["initial_" + c for c in AR_COLS],
                  "ADD_S_mm", "initial_ADD_S_mm", "delta_ADD_S_mm", "update", "regression"]
    allr = pd.concat([fr[cols], sb[cols]], ignore_index=True)
    for c in ["object_id", "scene_id", "img_id", "inst_id"]:
        allr[c] = allr[c].astype(int)
    allr["init_AR_mean"] = allr[["initial_" + c for c in AR_COLS]].mean(axis=1)
    allr["AR_regression"] = (allr["update"].eq(1) & (allr.AR_mean_row < allr.init_AR_mean - 1e-12)).astype(int)

    if allr.duplicated(KEY + ["arm"]).any():
        raise ValueError("Duplicate target condition within an arm")
    counts = allr.groupby(KEY).arm.nunique()
    n_arms = allr.arm.nunique()
    common = counts[counts.eq(n_arms)].index
    allr = allr.set_index(KEY).loc[common].reset_index()

    checks = {"arms": int(n_arms), "common_target_conditions": int(len(common)),
              "physical_targets": int(allr[["object_id", "scene_id", "img_id", "inst_id"]].drop_duplicates().shape[0]),
              "objects": int(allr.object_id.nunique()),
              "conditions_per_arm": {k: int(v) for k, v in allr.groupby("arm").size().items()}}
    j1 = allr.loc[allr.arm.eq("D joint (frozen run)")].set_index(KEY)
    j2 = allr.loc[allr.arm.eq(OURS)].set_index(KEY).loc[j1.index]
    checks["joint_frozen_vs_samebank_max_abs_AR_diff"] = float((j1.AR_mean_row - j2.AR_mean_row).abs().max())
    checks["joint_frozen_vs_samebank_action_mismatch"] = int((j1.selected_action != j2.selected_action).sum())
    init = allr.loc[allr.arm.eq("Initial")].set_index(KEY)
    checks["initial_arm_equals_initial_columns_max_abs"] = float((init.AR_mean_row - init.init_AR_mean).abs().max())
    return allr, checks


def obj_means(d: pd.DataFrame) -> pd.DataFrame:
    g = d.groupby(["arm", "object_id"])
    return g.agg(AR=("AR_mean_row", "mean"), AR_VSD=("AR_VSD_row", "mean"), AR_MSSD=("AR_MSSD_row", "mean"),
                 AR_MSPD=("AR_MSPD_row", "mean"), ADD_S=("ADD_S_mm", "mean"), dADD=("delta_ADD_S_mm", "mean"),
                 upd=("update", "mean"), reg_all=("regression", "mean"), ar_reg_all=("AR_regression", "mean"),
                 n=("AR_mean_row", "size")).reset_index()


def summarize(d: pd.DataFrame, scope: str) -> list[dict]:
    om = obj_means(d)
    rows = []
    for arm, g in om.groupby("arm"):
        dd = d.loc[d.arm.eq(arm)]
        upd = int(dd["update"].sum())
        rows.append(dict(
            scope=scope, arm=arm, objects=len(g), conditions=len(dd),
            AR_pct=100 * g.AR.mean(), AR_VSD_pct=100 * g.AR_VSD.mean(),
            AR_MSSD_pct=100 * g.AR_MSSD.mean(), AR_MSPD_pct=100 * g.AR_MSPD.mean(),
            AR_pooled_pct=100 * dd.AR_mean_row.mean(),
            ADD_S_mm=g.ADD_S.mean(), dADD_S_mm=g.dADD.mean(),
            updates=upd, update_pct=100 * dd["update"].mean(),
            accepted_ADD_regression_pct=(100 * dd.regression.sum() / upd) if upd else np.nan,
            all_target_ADD_regression_pct=100 * dd.regression.mean(),
            accepted_AR_regression_pct=(100 * dd.AR_regression.sum() / upd) if upd else np.nan,
        ))
    return rows


def paired(d: pd.DataFrame, a: str, b: str, scope: str, metric: str, rng) -> dict:
    om = obj_means(d).pivot(index="object_id", columns="arm", values=metric)
    scale = 100 if metric.startswith("AR") else 1
    diff = (om[a] - om[b]).to_numpy() * scale
    k = len(diff)
    boot = diff[rng.integers(0, k, size=(DRAWS, k))].mean(axis=1)
    better = (diff > 0) if metric.startswith("AR") else (diff < 0)
    return dict(scope=scope, metric=metric, method=a, baseline=b, objects=k,
                mean_diff=float(diff.mean()), ci_low=float(np.quantile(boot, .025)),
                ci_high=float(np.quantile(boot, .975)),
                bootstrap_p_two_sided=float(min(1.0, 2 * min((boot <= 0).mean(), (boot >= 0).mean()))),
                objects_better=int(better.sum()), objects_worse=int(((diff < 0) if metric.startswith("AR") else (diff > 0)).sum()),
                per_object=";".join(f"{o}:{v:+.3f}" for o, v in zip(om.index, diff)))


def gate_diagnostics(out: Path) -> pd.DataFrame | None:
    path = out / "candidate_gates.csv.gz"
    if not path.exists():
        return None
    g = pd.read_csv(path)
    gate_cols = [c for c in g.columns if c.startswith("gate_")]
    records = []
    scopes = [("all", g)] + [(f"init_regime:{v}", x) for v, x in g.groupby("init_regime")] \
        + [(f"input_depth_condition:{v}", x) for v, x in g.groupby("input_depth_condition")]
    for scope, x in scopes:
        passed = x[gate_cols].astype(bool)
        failing = (~passed).sum(axis=1)
        allowed = failing.eq(0)
        per_target = allowed.groupby(x.target_id).any()
        rec = dict(scope=scope, candidates=len(x), targets=int(x.target_id.nunique()),
                   candidates_passing_all_pct=100 * allowed.mean(),
                   targets_with_eligible_candidate_pct=100 * per_target.mean())
        for c in gate_cols:
            rec[f"{c}_pass_pct"] = 100 * passed[c].mean()
            rec[f"{c}_sole_blocker_pct"] = 100 * ((~passed[c]) & failing.eq(1)).mean()
        records.append(rec)
    return pd.DataFrame(records)


def classify(primary: dict, regression_pct: float) -> str:
    if primary["ci_low"] > 0:
        return "EXTERNAL_GAIN: D improves AR over the initializer (95% object-cluster CI above 0)."
    if primary["ci_high"] < 0:
        return "EXTERNAL_HARM: D lowers AR relative to the initializer (95% CI below 0)."
    if primary["ci_low"] >= NONINFERIORITY_MARGIN_PP:
        return (f"NON_INFERIOR_NO_CLEAR_GAIN: CI includes 0 and its lower bound is above the "
                f"pre-specified margin {NONINFERIORITY_MARGIN_PP} pp.")
    return "INCONCLUSIVE: CI includes 0 and crosses the non-inferiority margin."


def fmt(v, nd=2):
    return "NA" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{nd}f}"


def report(out: Path, summ: pd.DataFrame, pairs: pd.DataFrame, checks: dict, best_fixed: str,
           gates: pd.DataFrame | None) -> None:
    def row(scope, arm):
        r = summ.loc[summ.scope.eq(scope) & summ.arm.eq(arm)]
        return None if r.empty else r.iloc[0]

    def pr(scope, baseline, metric="AR"):
        r = pairs.loc[pairs.scope.eq(scope) & pairs.baseline.eq(baseline) & pairs.metric.eq(metric)
                      & pairs.method.eq(OURS)]
        return None if r.empty else r.iloc[0]

    p = pr("all", "Initial")
    lines = ["# YCB-V 冻结外部验证（r4 conditional_residual_joint）", "",
             "## 材料护照",
             f"- 目标条件：{checks['common_target_conditions']}（{checks['physical_targets']} 个物理目标 × 3 初始化档位 × 2 深度条件），物体 {checks['objects']} 个。",
             "- 模型、阈值、策略与候选设计全部来自 22 个 T-LESS 开发物体；YCB-V 上没有任何拟合、调参或选择。",
             f"- 一致性：冻结运行 D 与同库 D 的 AR 最大差 {checks['joint_frozen_vs_samebank_max_abs_AR_diff']:.2e}，动作不一致 {checks['joint_frozen_vs_samebank_action_mismatch']} 个。",
             "", "## 主要终点（预先指定）",
             f"- D − Initial（物体等权 AR，全部条件）：{fmt(p.mean_diff)} pp，95% CI [{fmt(p.ci_low)}, {fmt(p.ci_high)}]，"
             f"{int(p.objects_better)}/{int(p.objects)} 个物体为正。",
             f"- 判定：{checks['primary_classification']}",
             "", "## 全部条件汇总（物体等权）", "",
             "| 方法 | AR % | VSD | MSSD | MSPD | 更新率 % | 被接受更新中 ADD-S 回退 % |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for arm in ORDER:
        r = row("all", arm)
        if r is None or arm == "D joint (frozen run)":
            continue
        lines.append(f"| {arm} | {fmt(r.AR_pct)} | {fmt(r.AR_VSD_pct)} | {fmt(r.AR_MSSD_pct)} | {fmt(r.AR_MSPD_pct)} | "
                     f"{fmt(r.update_pct, 1)} | {fmt(r.accepted_ADD_regression_pct, 1)} |")
    lines += ["", "## 分层：D − Initial 与 D − 最佳固定候选", "",
              f"最佳固定候选（在 YCB-V 全部条件上事后选出，对基线有利）：{best_fixed}", "",
              "| 范围 | D − Initial (pp) | 95% CI | D − 最佳固定 (pp) | 95% CI | D 更新率 % | D 回退 % |",
              "|---|---:|---|---:|---|---:|---:|"]
    for scope in ["all", "native", "mild", "hard", "native_depth", "local_outlier_30pct"]:
        a, b, r = pr(scope, "Initial"), pr(scope, best_fixed), row(scope, OURS)
        if a is None:
            continue
        lines.append(f"| {scope} | {fmt(a.mean_diff)} | [{fmt(a.ci_low)}, {fmt(a.ci_high)}] | {fmt(b.mean_diff)} | "
                     f"[{fmt(b.ci_low)}, {fmt(b.ci_high)}] | {fmt(r.update_pct, 1)} | {fmt(r.accepted_ADD_regression_pct, 1)} |")
    a = pr("all", "A ordinary ranking")
    if a is not None:
        lines += ["", f"- D − A（普通排序）：{fmt(a.mean_diff)} pp，95% CI [{fmt(a.ci_low)}, {fmt(a.ci_high)}]。"]
    if gates is not None:
        g = gates.loc[gates.scope.eq("all")].iloc[0]
        cols = [c for c in gates.columns if c.endswith("_pass_pct")]
        lines += ["", "## 门控诊断（D 的共享冻结策略，候选级）", "",
                  f"- 通过全部门控的候选：{fmt(g.candidates_passing_all_pct, 1)}%；至少有一个合格候选的目标：{fmt(g.targets_with_eligible_candidate_pct, 1)}%。"]
        for c in cols:
            name = c[len("gate_"):-len("_pass_pct")]
            lines.append(f"- {name}：通过 {fmt(g[c], 1)}%，唯一阻断 {fmt(g['gate_' + name + '_sole_blocker_pct'], 1)}%。")
    lines += ["", "## 解释边界",
              "- 这是新数据集、新传感器、新初始化器上的外部验证，不是 BOP 排行榜成绩；AR 为受控协议下的 BOP 风格 AR（VSD 使用原生深度）。",
              "- 主要终点之外的比较（分层、最佳固定候选、A–C 消融）均为探索性，未做多重比较校正；最佳固定候选是事后选出的。",
              "- 物体之间共享场景，物体聚类自助法不能完全消除残余相关。",
              "- YCB-V 曾用于更早的 Protocol G 方法的外部验证；r4 的特征、候选和策略均在 T-LESS 上确定，未在 YCB-V 上看过 r4 结果。"]
    (out / "analysis" / "report_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze(out: Path) -> dict:
    out = Path(out)
    (out / "analysis").mkdir(exist_ok=True)
    allr, checks = load(out)
    rng = np.random.default_rng(SEED)
    scopes = {"all": allr}
    for reg in ["native", "mild", "hard"]:
        scopes[reg] = allr.loc[allr.init_regime.eq(reg)]
    for dep in ["native_depth", "local_outlier_30pct"]:
        scopes[dep] = allr.loc[allr.input_depth_condition.eq(dep)]
    scopes = {k: v for k, v in scopes.items() if len(v)}

    om_all = obj_means(allr)
    fixed_means = om_all.loc[om_all.arm.isin(FIXED.values())].groupby("arm").AR.mean()
    best_fixed = str(fixed_means.idxmax())

    summ, pairs = [], []
    baselines = ["Initial", *FIXED.values(), "A ordinary ranking", "B fixed residual", "C state residual",
                 "Locked r4 primary (selected_risk_stratified)"]
    for scope, d in scopes.items():
        summ += summarize(d, scope)
        for b in baselines:
            for m in ["AR", "AR_VSD", "AR_MSSD", "AR_MSPD", "dADD"]:
                pairs.append(paired(d, OURS, b, scope, m, rng))
        pairs.append(paired(d, best_fixed, "Initial", scope, "AR", rng))
    summ = pd.DataFrame(summ)
    pairs = pd.DataFrame(pairs)
    summ["o"] = summ.arm.map({a: i for i, a in enumerate(ORDER)})
    summ = summ.sort_values(["scope", "o"]).drop(columns="o")

    primary = pairs.loc[pairs.scope.eq("all") & pairs.method.eq(OURS) & pairs.baseline.eq("Initial")
                        & pairs.metric.eq("AR")].iloc[0].to_dict()
    ours_all = summ.loc[summ.scope.eq("all") & summ.arm.eq(OURS)].iloc[0]
    checks.update(primary_endpoint={k: primary[k] for k in ["mean_diff", "ci_low", "ci_high", "objects", "objects_better",
                                                             "bootstrap_p_two_sided"]},
                  primary_classification=classify(primary, float(ours_all.accepted_ADD_regression_pct)),
                  noninferiority_margin_pp=NONINFERIORITY_MARGIN_PP,
                  best_fixed_post_hoc=best_fixed, bootstrap_draws=DRAWS, seed=SEED,
                  input_sha256={name: sha(out / name) for name in ["frozen_outcomes.csv", "same_bank_outcomes.csv"]})

    per_object = obj_means(allr).pivot(index="object_id", columns="arm", values="AR") * 100
    per_object = per_object[[a for a in ORDER if a in per_object.columns]]
    gates = gate_diagnostics(out)

    summ.to_csv(out / "analysis" / "main_table_summary.csv", index=False)
    pairs.to_csv(out / "analysis" / "paired_object_bootstrap.csv", index=False)
    om_all.to_csv(out / "analysis" / "object_level_means.csv", index=False)
    per_object.to_csv(out / "analysis" / "per_object_AR_pct.csv")
    if gates is not None:
        gates.to_csv(out / "analysis" / "gate_diagnostics.csv", index=False)
    (out / "analysis" / "checks.json").write_text(json.dumps(checks, indent=2, ensure_ascii=False, default=float),
                                                  encoding="utf-8")
    report(out, summ, pairs, checks, best_fixed, gates)
    return checks


if __name__ == "__main__":
    result = analyze(Path(sys.argv[1]))
    print(json.dumps({k: result[k] for k in ["common_target_conditions", "objects", "primary_endpoint",
                                               "primary_classification", "best_fixed_post_hoc",
                                               "joint_frozen_vs_samebank_max_abs_AR_diff",
                                               "joint_frozen_vs_samebank_action_mismatch"]},
                     indent=2, ensure_ascii=False, default=float))
    print("YCB-V EXTERNAL ANALYSIS COMPLETE")
