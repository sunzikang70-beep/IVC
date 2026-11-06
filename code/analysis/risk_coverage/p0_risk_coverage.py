"""P0-4: continuous risk-coverage sweep on the 8 unseen objects.

For each ordering, eligible targets are ranked by a GT-free score; the top-k accept their
proposed action, all others keep the initializer. Outcomes of every action come from the
frozen per-target outcomes (P0-1 merge). Read-only; no fitting.
Orderings: A/B/C/D utilities (frozen r4 shared gates, from matched audit), and a classical
non-learned geometric selector (lowest bidirectional visibility/free-space cost among the six
candidates, ranked by relative cost improvement over the initializer).
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(sys.argv[1]); OUT = Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
KEY = ["object_id", "scene_id", "img_id", "inst_id", "input_depth_condition", "init_regime"]
INIT = "protocolB_e2e_initial_pose"
ACT = {"P2P s0.25": "protocolF_se3_trust_p2p_s025", "P2P s0.50": "protocolF_se3_trust_p2p_s050",
       "P2P s1.00": "protocolF_se3_trust_p2p_s100", "Filtered P2P s0.25": "protocolF_se3_trust_filtered_s025",
       "Filtered P2P s0.50": "protocolF_se3_trust_filtered_s050", "Filtered P2P s1.00": "protocolF_se3_trust_filtered_s100"}
m = pd.read_csv(ROOT / "outputs/p0_main_table_baselines_20261104/merged_per_target.csv.gz")
init = m[m.arm == "Initial"].set_index(KEY)
outc = {}
for arm, a in ACT.items():
    t = m[m.arm == arm].set_index(KEY)
    outc[a] = t
outc[INIT] = init

def outcome_table(chosen):  # chosen: Series index KEY -> action
    rows = []
    for a, idx in chosen.groupby(chosen).groups.items():
        t = outc[a].loc[idx, ["AR_mean_row", "ADD_S_mm", "delta_ADD_S_mm"]].copy()
        t["action"] = a
        rows.append(t)
    r = pd.concat(rows)
    r["init_AR"] = init.loc[r.index, "init_AR_mean"].to_numpy()
    r["upd"] = (r.action != INIT).astype(int)
    r["reg"] = ((r.upd == 1) & (r.delta_ADD_S_mm > 1e-9)).astype(int)
    r["ar_reg"] = ((r.upd == 1) & (r.AR_mean_row < r.init_AR - 1e-12)).astype(int)
    return r.reset_index()

# ---- learned orderings from matched audit (combined8)
mt = pd.read_csv(ROOT / "outputs/matched_risk_coverage_20261026_r1/per_target_outcomes.csv.gz")
mt = mt[(mt.dataset == "combined8") & (mt.family == "matched_coverage") & (mt.ordering == "utility")
        & (mt.quota_mode == "pooled_exact")]
for c in ["object_id", "scene_id", "img_id", "inst_id"]:
    mt[c] = mt[c].astype(int)
checks = {}
orders = {}
for arm, name in {"A_delta_rank": "A ordinary ranking", "B_fixed_residual": "B fixed residual",
                  "C_state_residual": "C state residual", "D_joint_full": "D joint (ours)"}.items():
    t = mt[mt.arm == arm]
    g = t.groupby(KEY).agg(utility=("utility", "first"), act=("base_proposed_action", "first"),
                           elig=("base_eligible", "first"), n_act=("base_proposed_action", "nunique"),
                           n_u=("utility", "nunique"))
    checks[f"{name}_consistent_across_points"] = bool((g.n_act <= 1).all() and (g.n_u <= 1).all())
    g = g[g.elig.astype(bool) & np.isfinite(g.utility) & g.act.ne(INIT)]
    orders[name] = g.sort_values("utility", ascending=False)[["act"]].assign(score=g.utility)

# ---- classical geometric selector
feats = []
for d in ["decoupled_frozen_unseen5_20261021_r4", "decoupled_frozen_unseen3_20261023_r1"]:
    f = pd.read_csv(ROOT / f"outputs/{d}/observable_candidate_features.csv",
                    usecols=["object_id", "scene_id", "img_id", "inst_id", "input_depth_condition", "cohort", "action",
                             "candidate_score_bidir_visibility_freespace", "delta_score_bidir_visibility_freespace"])
    feats.append(f)
f = pd.concat(feats)
f["init_regime"] = f.cohort.str.rsplit("_", n=1).str[-1]
f = f[f.action.isin(ACT.values())]
f["init_cost"] = f.candidate_score_bidir_visibility_freespace - f.delta_score_bidir_visibility_freespace
f["rel_gain"] = -f.delta_score_bidir_visibility_freespace / f.init_cost.abs().clip(lower=1e-12)
f = f[np.isfinite(f.rel_gain) & np.isfinite(f.candidate_score_bidir_visibility_freespace)]
best = f.sort_values(["candidate_score_bidir_visibility_freespace", "action"]).groupby(KEY).head(1).set_index(KEY)
best = best.loc[best.index.intersection(init.index)]
best = best[best.rel_gain > 0]
orders["Geometric cost selector (classical)"] = best.sort_values("rel_gain", ascending=False)[["action"]].rename(
    columns={"action": "act"}).assign(score=best.rel_gain)
checks["geometric_selector_targets_with_positive_gain"] = int(len(best))
checks["targets_total"] = int(len(init))

N = len(init)
grid = np.unique(np.round(np.linspace(0, 1, 101) * N).astype(int))
curves = []
for name, o in orders.items():
    acts = o.act
    for k in np.unique(np.append(grid[grid <= len(acts)], len(acts))):
        if k > len(acts):
            continue
        chosen = pd.Series(INIT, index=init.index)
        chosen.loc[acts.index[:k]] = acts.iloc[:k].to_numpy()
        r = outcome_table(chosen)
        ob = r.groupby("object_id").agg(AR=("AR_mean_row", "mean"), iAR=("init_AR", "mean"))
        upd = r.upd.sum()
        curves.append(dict(ordering=name, k=int(k), update_pct=100 * k / N,
                           AR_gain_pp=100 * (ob.AR - ob.iAR).mean(),
                           all_target_ADD_regression_pct=100 * r.reg.mean(),
                           accepted_ADD_regression_pct=100 * r.reg.sum() / upd if upd else np.nan,
                           accepted_AR_regression_pct=100 * r.ar_reg.sum() / upd if upd else np.nan,
                           max_k=len(acts)))
cur = pd.DataFrame(curves)
cur.to_csv(OUT / "risk_coverage_curves.csv", index=False)
# fixed candidates at 100% update for reference
fx = []
for arm, a in ACT.items():
    chosen = pd.Series(a, index=init.index)
    r = outcome_table(chosen)
    ob = r.groupby("object_id").agg(AR=("AR_mean_row", "mean"), iAR=("init_AR", "mean"))
    fx.append(dict(ordering=arm, update_pct=100.0, AR_gain_pp=100 * (ob.AR - ob.iAR).mean(),
                   all_target_ADD_regression_pct=100 * r.reg.mean(), accepted_ADD_regression_pct=100 * r.reg.mean()))
pd.DataFrame(fx).to_csv(OUT / "fixed_candidates_reference.csv", index=False)
json.dump(checks, open(OUT / "checks.json", "w"), indent=2)
print(json.dumps(checks, indent=1))
