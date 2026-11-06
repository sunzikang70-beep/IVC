"""P0-1: 8 unseen objects main table with fixed-candidate baselines.

Read-only re-aggregation of existing frozen per-target outcomes; no model, threshold
or candidate is refit. Object-equal means; paired object-cluster bootstrap.
"""
import json, hashlib, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(sys.argv[1])           # codex_paper_workspace
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
SRC = {
    "frozen_unseen5": ROOT / "outputs/decoupled_frozen_unseen5_20261021_r4/frozen_outcomes.csv",
    "frozen_unseen3": ROOT / "outputs/decoupled_frozen_unseen3_20261023_r1/frozen_outcomes.csv",
    "same_bank": ROOT / "outputs/same_bank_residual_ablation_20261026_r1/per_target_outcomes.csv",
}
KEY = ["object_id", "scene_id", "img_id", "inst_id", "input_depth_condition", "init_regime"]
B = 20000
rng = np.random.default_rng(20261104)

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

# ---- frozen fixed candidates + initial + original locked branch
fr = []
for tag in ["frozen_unseen5", "frozen_unseen3"]:
    d = pd.read_csv(SRC[tag])
    d["source"] = tag
    fr.append(d)
fr = pd.concat(fr, ignore_index=True)
fr["init_AR_mean"] = fr[["initial_AR_VSD_row", "initial_AR_MSSD_row", "initial_AR_MSPD_row"]].mean(axis=1)
fr["update"] = (fr["selected_action"] != "protocolB_e2e_initial_pose").astype(int)
fr["regression"] = ((fr["update"] == 1) & (fr["delta_ADD_S_mm"] > 1e-9)).astype(int)
name_map = {
    "protocolB_e2e_initial_pose": "Initial",
    "protocolF_se3_trust_p2p_s025": "P2P s0.25",
    "protocolF_se3_trust_p2p_s050": "P2P s0.50",
    "protocolF_se3_trust_p2p_s100": "P2P s1.00",
    "protocolF_se3_trust_filtered_s025": "Filtered P2P s0.25",
    "protocolF_se3_trust_filtered_s050": "Filtered P2P s0.50",
    "protocolF_se3_trust_filtered_s100": "Filtered P2P s1.00",
    "selected_risk_stratified": "Locked r4 primary (selected_risk_stratified)",
    "conditional_residual_joint": "D joint (frozen run)",
}
fr = fr[fr.method.isin(name_map)].copy()
fr["arm"] = fr.method.map(name_map)

# ---- same-bank arms A-D on combined8, shared_r4 policy
sb = pd.read_csv(SRC["same_bank"])
sb = sb[(sb.dataset == "combined8") & (sb.policy_mode == "shared_r4")].copy()
arm_map = {"A_delta_rank": "A ordinary ranking", "B_fixed_residual": "B fixed residual",
           "C_state_residual": "C state residual", "D_joint_full": "D joint (ours)"}
sb = sb[sb.arm.isin(arm_map)].copy()
sb["arm"] = sb.arm.map(arm_map)
sb["init_AR_mean"] = sb[["initial_AR_VSD_row", "initial_AR_MSSD_row", "initial_AR_MSPD_row"]].mean(axis=1)

cols = KEY + ["arm", "selected_action", "AR_mean_row", "AR_VSD_row", "AR_MSSD_row", "AR_MSPD_row",
              "init_AR_mean", "initial_AR_VSD_row", "initial_AR_MSSD_row", "initial_AR_MSPD_row",
              "ADD_S_mm", "initial_ADD_S_mm", "delta_ADD_S_mm", "update", "regression"]
allr = pd.concat([fr[cols], sb[cols]], ignore_index=True)
for c in ["object_id", "scene_id", "img_id", "inst_id"]:
    allr[c] = allr[c].astype(int)

# common target set: those present for every arm
counts = allr.groupby(KEY).arm.nunique()
n_arms = allr.arm.nunique()
common = counts[counts == n_arms].index
allr = allr.set_index(KEY).loc[common].reset_index()
checks = {"arms": int(n_arms), "common_targets": int(len(common)),
          "targets_per_arm": allr.groupby("arm").size().to_dict()}

# consistency: frozen-run joint vs same-bank D per target
j1 = allr[allr.arm == "D joint (frozen run)"].set_index(KEY)
j2 = allr[allr.arm == "D joint (ours)"].set_index(KEY).loc[j1.index]
checks["joint_frozen_vs_samebank_max_abs_AR_diff"] = float((j1.AR_mean_row - j2.AR_mean_row).abs().max())
checks["joint_frozen_vs_samebank_action_mismatch"] = int((j1.selected_action != j2.selected_action).sum())

allr["AR_regression"] = ((allr["update"] == 1) & (allr.AR_mean_row < allr.init_AR_mean - 1e-12)).astype(int)
allr["cohort8"] = np.where(allr.object_id.isin([1, 8, 22]), "added3", "original5")

def obj_means(d):
    g = d.groupby(["arm", "object_id"])
    m = g.agg(AR=("AR_mean_row", "mean"), AR_VSD=("AR_VSD_row", "mean"), AR_MSSD=("AR_MSSD_row", "mean"),
              AR_MSPD=("AR_MSPD_row", "mean"), ADD_S=("ADD_S_mm", "mean"), dADD=("delta_ADD_S_mm", "mean"),
              upd=("update", "mean"), reg_all=("regression", "mean"), ar_reg_all=("AR_regression", "mean"),
              n=("AR_mean_row", "size"))
    return m.reset_index()

def summarize(d, scope):
    om = obj_means(d)
    rows = []
    for arm, g in om.groupby("arm"):
        dd = d[d.arm == arm]
        upd = dd["update"].sum()
        rows.append(dict(scope=scope, arm=arm, objects=len(g), conditions=len(dd),
                         AR_pct=100 * g.AR.mean(), AR_VSD_pct=100 * g.AR_VSD.mean(),
                         AR_MSSD_pct=100 * g.AR_MSSD.mean(), AR_MSPD_pct=100 * g.AR_MSPD.mean(),
                         ADD_S_mm=g.ADD_S.mean(), dADD_S_mm=g.dADD.mean(),
                         update_pct=100 * dd["update"].mean(),
                         accepted_ADD_regression_pct=(100 * dd.regression.sum() / upd) if upd else np.nan,
                         all_target_ADD_regression_pct=100 * dd.regression.mean(),
                         accepted_AR_regression_pct=(100 * dd.AR_regression.sum() / upd) if upd else np.nan))
    return rows

def paired(d, a, b, scope, metric="AR"):
    om = obj_means(d).pivot(index="object_id", columns="arm", values=metric)
    diff = (om[a] - om[b]).to_numpy() * (100 if metric.startswith("AR") else 1)
    k = len(diff)
    idx = rng.integers(0, k, size=(B, k))
    boot = diff[idx].mean(axis=1)
    return dict(scope=scope, metric=metric, method=a, baseline=b, objects=k,
                mean_diff=diff.mean(), ci_low=np.quantile(boot, .025), ci_high=np.quantile(boot, .975),
                objects_positive=int((diff > 0).sum()) if metric.startswith("AR") else int((diff < 0).sum()),
                per_object=";".join(f"{o}:{v:+.3f}" for o, v in zip(om.index, diff)))

summ, pairs = [], []
scopes = {"combined8": allr, "original5": allr[allr.cohort8 == "original5"], "added3": allr[allr.cohort8 == "added3"]}
for reg in ["native", "mild", "hard"]:
    scopes[f"combined8_{reg}"] = allr[allr.init_regime == reg]
for dep in ["native_depth", "local_outlier_30pct"]:
    scopes[f"combined8_{dep}"] = allr[allr.input_depth_condition == dep]
baselines = ["Initial", "P2P s0.25", "P2P s0.50", "P2P s1.00", "Filtered P2P s0.25", "Filtered P2P s0.50",
             "Filtered P2P s1.00", "A ordinary ranking", "B fixed residual", "C state residual"]
for s, d in scopes.items():
    summ += summarize(d, s)
    for b in baselines:
        for m in ["AR", "AR_VSD", "AR_MSSD", "AR_MSPD", "dADD"]:
            pairs.append(paired(d, "D joint (ours)", b, s, m))
    # best fixed candidate vs initial (for context)
    pairs.append(paired(d, "Filtered P2P s1.00", "Initial", s, "AR"))
    pairs.append(paired(d, "A ordinary ranking", "Filtered P2P s1.00", s, "AR"))

summ = pd.DataFrame(summ)
pairs = pd.DataFrame(pairs)
order = ["Initial", "Locked r4 primary (selected_risk_stratified)", "P2P s0.25", "P2P s0.50", "P2P s1.00",
         "Filtered P2P s0.25", "Filtered P2P s0.50", "Filtered P2P s1.00", "A ordinary ranking",
         "B fixed residual", "C state residual", "D joint (ours)", "D joint (frozen run)"]
summ["o"] = summ.arm.map({a: i for i, a in enumerate(order)})
summ = summ.sort_values(["scope", "o"]).drop(columns="o")
summ.to_csv(OUT / "main_table_summary.csv", index=False)
pairs.to_csv(OUT / "paired_object_bootstrap.csv", index=False)
obj_means(allr).to_csv(OUT / "object_level_means.csv", index=False)
allr.to_csv(OUT / "merged_per_target.csv.gz", index=False)
checks["bootstrap_draws"] = B
checks["seed"] = 20261104
checks["input_sha256"] = {k: sha(v) for k, v in SRC.items()}
json.dump(checks, open(OUT / "checks.json", "w"), indent=2, ensure_ascii=False)
print(json.dumps(checks, indent=1, ensure_ascii=False))
