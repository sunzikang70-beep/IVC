"""Paper numbers with arm A (predicted-utility ranking, no residual correction) as the main method.

Read-only re-aggregation of frozen per-target outcomes; nothing is refit or retuned.
Same estimands as build_p0_main_table.py / analyze_decoupled_frozen_external_ycbv.py:
object-equal means, paired object-cluster bootstrap (20000 draws, seed 20261104).

Usage: python analyze_A_main_method.py <codex_paper_workspace> <output_dir>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(sys.argv[1])
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "work"))
import analyze_decoupled_frozen_external_ycbv as ycbv  # noqa: E402

KEY = ["object_id", "scene_id", "img_id", "inst_id", "input_depth_condition", "init_regime"]
OURS = "A ordinary ranking"
FIXED = ["P2P s0.25", "P2P s0.50", "P2P s1.00", "Filtered P2P s0.25", "Filtered P2P s0.50", "Filtered P2P s1.00"]
LEARNED = ["A ordinary ranking", "B fixed residual", "C state residual", "D joint (ours)"]
BASELINES = ["Initial", *FIXED, "B fixed residual", "C state residual", "D joint (ours)"]
METRICS = ["AR", "AR_VSD", "AR_MSSD", "AR_MSPD", "dADD"]
DRAWS, SEED = 20000, 20261104
rng = np.random.default_rng(SEED)


def obj_means(d):
    g = d.groupby(["arm", "object_id"])
    return g.agg(AR=("AR_mean_row", "mean"), AR_VSD=("AR_VSD_row", "mean"), AR_MSSD=("AR_MSSD_row", "mean"),
                 AR_MSPD=("AR_MSPD_row", "mean"), dADD=("delta_ADD_S_mm", "mean"),
                 n=("AR_mean_row", "size")).reset_index()


def summarize(d, dataset, scope):
    om = obj_means(d)
    init = om.loc[om.arm.eq("Initial")].set_index("object_id").AR
    rows = []
    for arm, g in om.groupby("arm"):
        dd = d.loc[d.arm.eq(arm)]
        upd = int(dd["update"].sum())
        gain = (g.set_index("object_id").AR - init.loc[g.object_id].to_numpy()).mean() * 100
        rows.append(dict(dataset=dataset, scope=scope, arm=arm, objects=len(g), conditions=len(dd),
                         AR_pct=100 * g.AR.mean(), dAR_pp=gain, AR_VSD_pct=100 * g.AR_VSD.mean(),
                         AR_MSSD_pct=100 * g.AR_MSSD.mean(), AR_MSPD_pct=100 * g.AR_MSPD.mean(),
                         dADD_S_mm=g.dADD.mean(), updates=upd, update_pct=100 * dd["update"].mean(),
                         accepted_ADD_regression_pct=(100 * dd.regression.sum() / upd) if upd else np.nan,
                         all_target_ADD_regression_pct=100 * dd.regression.mean(),
                         accepted_AR_regression_pct=(100 * dd.AR_regression.sum() / upd) if upd else np.nan))
    return rows


def paired(d, a, b, dataset, scope, metric):
    om = obj_means(d).pivot(index="object_id", columns="arm", values=metric)
    scale = 100 if metric.startswith("AR") else 1
    diff = (om[a] - om[b]).to_numpy() * scale
    boot = diff[rng.integers(0, len(diff), size=(DRAWS, len(diff)))].mean(axis=1)
    better = (diff > 1e-12) if metric.startswith("AR") else (diff < -1e-12)
    return dict(dataset=dataset, scope=scope, metric=metric, method=a, baseline=b, objects=len(diff),
                mean_diff=diff.mean(), ci_low=np.quantile(boot, .025), ci_high=np.quantile(boot, .975),
                objects_better=int(better.sum()),
                per_object=";".join(f"{o}:{v:+.3f}" for o, v in zip(om.index, diff)))


def scopes_for(d, dataset):
    s = {"all": d}
    for r in ["native", "mild", "hard"]:
        s[r] = d.loc[d.init_regime.eq(r)]
    for dep in ["native_depth", "local_outlier_30pct"]:
        s[dep] = d.loc[d.input_depth_condition.eq(dep)]
    if dataset == "T-LESS":
        s["original5"] = d.loc[~d.object_id.isin([1, 8, 22])]
        s["added3"] = d.loc[d.object_id.isin([1, 8, 22])]
    return s


# ----------------------------------------------------------------------------- data
tless = pd.read_csv(ROOT / "outputs/p0_main_table_baselines_20261104/merged_per_target.csv.gz")
tless = tless.loc[tless.arm.isin(["Initial", *FIXED, *LEARNED])].copy()
yv, ychecks = ycbv.load(ROOT / "outputs/decoupled_frozen_ycbv_external_20261104")
yv = yv.loc[yv.arm.isin(["Initial", *FIXED, *LEARNED])].copy()
data = {"T-LESS": tless, "YCB-V": yv}

summary, pairs, oracle = [], [], []
for name, d in data.items():
    for scope, part in scopes_for(d, name).items():
        summary += summarize(part, name, scope)
        for b in BASELINES:
            for m in METRICS:
                pairs.append(paired(part, OURS, b, name, scope, m))
        pairs.append(paired(part, "Filtered P2P s1.00", "Initial", name, scope, "AR"))
        pairs.append(paired(part, "D joint (ours)", "Initial", name, scope, "AR"))
        pairs.append(paired(part, "C state residual", "B fixed residual", name, scope, "AR"))
        pairs.append(paired(part, "D joint (ours)", "C state residual", name, scope, "AR"))
    # oracle headroom with A as the selective method
    w = d.pivot_table(index=KEY, columns="arm", values="AR_mean_row")
    delta = w[[*FIXED, *LEARNED]].sub(w["Initial"], axis=0)
    obj = delta.index.get_level_values("object_id")
    reg = delta.index.get_level_values("init_regime")
    for regime in ["native", "mild", "hard", "all"]:
        m = np.ones(len(delta), bool) if regime == "all" else (reg == regime)
        dd, o = delta[m], obj[m]
        oe = lambda s: 100 * s.groupby(o).mean().mean()  # noqa: E731
        F = dd["Filtered P2P s1.00"]
        oracle.append(dict(dataset=name, regime=regime, fixed_filtered_cap1=oe(F), ours_A=oe(dd[OURS]),
                           D_residual=oe(dd["D joint (ours)"]), oracle_veto=oe(F.clip(lower=0)),
                           oracle_best_of_7=oe(pd.Series(np.maximum(dd[FIXED].max(axis=1), 0), index=dd.index)),
                           fixed_rows_losing_AR_pct=100 * float((F < -1e-12).mean())))

summary = pd.DataFrame(summary)
pairs = pd.DataFrame(pairs)
oracle = pd.DataFrame(oracle)
oracle["headroom_veto_minus_fixed"] = oracle.oracle_veto - oracle.fixed_filtered_cap1
oracle["ours_minus_fixed"] = oracle.ours_A - oracle.fixed_filtered_cap1
oracle["recovered_share_pct"] = 100 * oracle.ours_minus_fixed / oracle.headroom_veto_minus_fixed
worst = summary.loc[summary.scope.isin(["native", "mild", "hard"])].groupby(["dataset", "arm"]).dAR_pp.min().unstack("dataset")

# ----------------------------------------------------------------------------- rule ablation for A (T-LESS)
c = pd.read_csv(ROOT / "outputs/joint_optimization_review_20261026_r2/candidates.csv")
c = c.loc[c.dataset.eq("combined8")].copy()
c["target_id"] = c.frame_id.astype(str) + "|" + c.cohort.astype(str) + "|" + c.input_depth_condition.astype(str)
for g in ["gate_moving_valid", "gate_observation", "gate_domain", "gate_risk", "gate_enabled_policy", "chosen"]:
    c[g] = c[g].astype(str).str.lower().isin(["true", "1", "1.0"])
fam = ["delta_AR_VSD", "delta_AR_MSSD", "delta_AR_MSPD"]
pred = np.column_stack([c["lcb_" + t] + c["guard_" + t] for t in fam])
pred_add = (c.ucb_delta_ADD_S_mm - c.guard_delta_ADD_S_mm).to_numpy()
mean = pred.mean(axis=1)
c["utility_A"] = mean - 0.005 * pred_add / 5.
c["gate_ADD_upper_A"] = pred_add <= 0
c["gate_AR_component_A"] = pred.min(axis=1) >= -0.01
c["gate_recall_floor_A"] = mean >= -0.001
c["gate_positive_utility_A"] = c.utility_A > 0
c["gate_finite_A"] = np.isfinite(c.utility_A) & np.isfinite(pred).all(axis=1) & np.isfinite(pred_add)
GATES = ["gate_moving_valid", "gate_observation", "gate_domain", "gate_risk", "gate_ADD_upper_A", "gate_AR_component_A",
         "gate_recall_floor_A", "gate_positive_utility_A", "gate_enabled_policy", "gate_finite_A"]
tgt = c.drop_duplicates("target_id")[["target_id", "object_id", "frame_id", "cohort", "init_regime", "input_depth_condition"]]


def select(gates, actions=None):
    d = c if actions is None else c.loc[c.action.isin(actions)]
    e = d.loc[d[gates].all(axis=1)].sort_values(["target_id", "utility_A", "risk_score", "action"],
                                                ascending=[True, False, True, True])
    return e.groupby("target_id").head(1).set_index("target_id")


def evaluate(chosen):
    r = tgt.set_index("target_id").copy()
    r["upd"] = r.index.isin(chosen.index).astype(int)
    for col in ["actual_gain", "delta_ADD_S_mm"]:
        r[col] = chosen[col].reindex(r.index).fillna(0.0)
    r["reg"] = ((r.upd == 1) & (r.delta_ADD_S_mm > 1e-9)).astype(int)
    ob = r.groupby("object_id")
    return r, dict(AR_gain_pp=100 * ob.actual_gain.mean().mean(), update_pct=100 * r.upd.mean(),
                   accepted_ADD_regression_pct=100 * r.reg.sum() / max(1, r.upd.sum()),
                   all_target_ADD_regression_pct=100 * r.reg.mean())


P2P = [a for a in c.action.unique() if "_p2p_" in a]
FIL = [a for a in c.action.unique() if "_filtered_" in a]
S100 = [a for a in c.action.unique() if a.endswith("s100")]
variants = {
    "full rule": (GATES, None),
    "filtered candidates (P2P only)": (GATES, P2P),
    "P2P candidates (filtered only)": (GATES, FIL),
    "smaller caps (cap 1 only)": (GATES, S100),
    "both ADD-S constraints": ([g for g in GATES if g not in ("gate_risk", "gate_ADD_upper_A")], None),
    "ADD-S risk score only": ([g for g in GATES if g != "gate_risk"], None),
    "ADD-S upper bound only": ([g for g in GATES if g != "gate_ADD_upper_A"], None),
    "per-metric recall floor": ([g for g in GATES if g != "gate_AR_component_A"], None),
    "mean-recall floor": ([g for g in GATES if g != "gate_recall_floor_A"], None),
    "support gates": ([g for g in GATES if g not in ("gate_observation", "gate_domain")], None),
}
rows, per_obj = [], {}
for name, (gates, acts) in variants.items():
    r, s = evaluate(select(gates, acts))
    rows.append(dict(variant=name, **s))
    per_obj[name] = r.groupby("object_id").actual_gain.mean() * 100
po = pd.DataFrame(per_obj)
abl = pd.DataFrame(rows)
cis = []
for name in variants:
    diff = (po[name] - po["full rule"]).to_numpy()
    b = diff[rng.integers(0, len(diff), size=(DRAWS, len(diff)))].mean(1)
    cis.append(dict(variant=name, AR_change_vs_full_pp=diff.mean(), ci_low=np.quantile(b, .025),
                    ci_high=np.quantile(b, .975), objects_lower=int((diff < -1e-12).sum()),
                    objects_higher=int((diff > 1e-12).sum())))
abl = abl.merge(pd.DataFrame(cis), on="variant")

# replay check: full rule must reproduce the same-bank arm A decisions on every condition
sb = pd.read_csv(ROOT / "outputs/same_bank_residual_ablation_20261026_r1/per_target_outcomes.csv")
sb = sb.loc[sb.dataset.eq("combined8") & sb.policy_mode.eq("shared_r4") & sb.arm.eq("A_delta_rank")].copy()
sb["target_id"] = sb.frame_id.astype(str) + "|" + sb.cohort.astype(str) + "|" + sb.input_depth_condition.astype(str)
full = select(GATES).action
expected = sb.set_index("target_id").selected_action
mine = pd.Series("protocolB_e2e_initial_pose", index=expected.index)
mine.loc[full.index.intersection(mine.index)] = full.loc[full.index.intersection(mine.index)]
checks = dict(ycbv=ychecks, rule_replay_targets=int(len(expected)),
              rule_replay_mismatches=int((mine != expected).sum()),
              rule_replay_extra_targets=int(len(set(full.index) - set(expected.index))),
              raw_utility_max_abs_diff=float(np.nanmax(np.abs(c.utility_A - c.raw_utility))) if "raw_utility" in c else None)

# ----------------------------------------------------------------------------- ADD-S (T-LESS) with A
dia = []
for dname in ["decoupled_frozen_unseen5_20261021_r4", "decoupled_frozen_unseen3_20261023_r1"]:
    x = pd.read_csv(ROOT / f"outputs/{dname}/bop/bop_pose_errors.csv", usecols=["object_id", "diameter_mm"])
    dia.append(x.drop_duplicates())
dia = pd.concat(dia).drop_duplicates().set_index("object_id").diameter_mm
m = tless.loc[tless.arm.isin(["Initial", "Filtered P2P s1.00", OURS, "D joint (ours)"])].copy()
m["adds_d"] = m.ADD_S_mm / m.object_id.map(dia)
m["init_adds_d"] = m.initial_ADD_S_mm / m.object_id.map(dia)
m["rec01"] = (m.adds_d < 0.1).astype(float)
adds_rows = []
for scope, d in {"all": m, "init<0.1d": m.loc[m.init_adds_d < 0.1]}.items():
    po_ = d.groupby(["arm", "object_id"]).agg(mean_adds_mm=("ADD_S_mm", "mean"), mean_adds_d=("adds_d", "mean"),
                                              rec01=("rec01", "mean"), n=("ADD_S_mm", "size")).reset_index()
    for metric in ["mean_adds_mm", "mean_adds_d", "rec01"]:
        pv = po_.pivot(index="object_id", columns="arm", values=metric)
        for arm in pv.columns:
            adds_rows.append(dict(scope=scope, metric=metric, arm=arm, value=pv[arm].mean() * (100 if metric != "mean_adds_mm" else 1),
                                  conditions=int(d.loc[d.arm.eq(arm)].shape[0])))
        for base in ["Initial", "Filtered P2P s1.00", "D joint (ours)"]:
            diff = (pv[OURS] - pv[base]).to_numpy() * (100 if metric != "mean_adds_mm" else 1)
            b = diff[rng.integers(0, len(diff), size=(DRAWS, len(diff)))].mean(1)
            better = (diff > 0) if metric == "rec01" else (diff < 0)
            adds_rows.append(dict(scope=scope, metric=metric, arm=f"{OURS} - {base}", value=diff.mean(),
                                  ci_low=np.quantile(b, .025), ci_high=np.quantile(b, .975), objects_better=int(better.sum())))
init_share = (m.loc[m.arm.eq("Initial"), "init_adds_d"] < 0.1).mean()
gross_share = (m.loc[m.arm.eq("Initial"), "init_adds_d"] >= 1.0).mean()
checks.update(init_lt_0p1d_share=float(init_share), init_gross_share=float(gross_share))

summary.to_csv(OUT / "summary.csv", index=False)
pairs.to_csv(OUT / "paired_bootstrap.csv", index=False)
oracle.to_csv(OUT / "oracle_headroom.csv", index=False)
worst.to_csv(OUT / "worst_regime.csv")
abl.to_csv(OUT / "rule_ablation_A.csv", index=False)
pd.DataFrame(adds_rows).to_csv(OUT / "adds_A.csv", index=False)
json.dump(checks, open(OUT / "checks.json", "w"), indent=2, default=float)
print(json.dumps(checks, indent=1, default=float))
