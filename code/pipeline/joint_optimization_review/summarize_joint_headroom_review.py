"""Extend sealed diagnosis with ADD-S ceilings and reviewer-facing figures."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from audit_joint_optimization_headroom import best_choices, base

ROOT = Path(r"D:\6D_pose\codex_paper_workspace\outputs\joint_optimization_review_20261026_r2")


def main():
    state = json.loads((ROOT / "state.json").read_text(encoding="utf-8"))
    for name, digest in state["files"].items():
        assert base.digest(ROOT / name) == digest, name
    for path, digest in json.loads((ROOT / "input_hashes.json").read_text(encoding="utf-8")).items():
        assert base.digest(Path(path)) == digest, path
    candidates = pd.read_csv(ROOT / "candidates.csv")
    targets = pd.read_csv(ROOT / "targets.csv")
    original = pd.read_csv(ROOT / "outcomes.csv")
    rows = []
    for dataset in ["development22", "holdout5", "added3", "combined8"]:
        c = candidates.loc[candidates.dataset.eq(dataset)].reset_index(drop=True)
        t = targets.loc[targets.dataset.eq(dataset)].copy()
        # Combined banks have local target_id values; qualify by cohort and frame.
        keys = ["cohort", *base.KEYS]
        ckeys = ["cohort", "object_id", "frame_id", "input_depth_condition"]
        idmap = t[ckeys].reset_index(drop=True)
        idmap["qualified_id"] = np.arange(len(idmap))
        assert not idmap.duplicated(ckeys).any()
        c = c.merge(idmap, on=ckeys, validate="many_to_one")
        c["target_id"] = c.qualified_id
        masks = {"GT_oracle_min_ADD": np.ones(len(c), bool),
            "GT_oracle_min_ADD_BOP_nonregressed": c.actual_gain.ge(-1e-12).to_numpy() & c[base.TARGETS[:3]].min(axis=1).ge(-.01 - 1e-12).to_numpy()}
        for method, mask in masks.items():
            choices = best_choices(c, mask, -c.delta_ADD_S_mm)
            selected = c.loc[[choices[int(i)] == action for i, action in zip(c.target_id, c.action)]].set_index("qualified_id")
            out = t.reset_index(drop=True).copy()
            out["delta_ADD_S_mm"] = out.index.to_series().map(selected.delta_ADD_S_mm).fillna(0.)
            out["AR_gain"] = out.index.to_series().map(selected.actual_gain).fillna(0.)
            out["method"] = method
            out["ADD_S_mm"] = out.initial_ADD_S_mm + out.delta_ADD_S_mm
            out["update"] = out.index.isin(selected.index).astype(int)
            out["regression"] = out.delta_ADD_S_mm.gt(1e-9).astype(int)
            out["AR_mean_row"] = out[["initial_" + x for x in base.AR]].mean(axis=1) + out.AR_gain
            rows.append(out)
    augmented = pd.concat([original.loc[original.method.isin(["initializer", "A_delta_rank", "D_joint_full", "GT_oracle_all", "GT_oracle_ADD_component_safe"])], *rows], ignore_index=True)
    records = []
    for (dataset, method), full in augmented.groupby(["dataset", "method"]):
        for scope, group in [("all", full), *[("init_regime:" + str(k), v) for k, v in full.groupby("init_regime")]]:
            means = group.groupby("object_id")[["ADD_S_mm", "delta_ADD_S_mm", "AR_mean_row"]].mean()
            records.append(dict(dataset=dataset, method=method, scope=scope, targets=len(group),
                AR_macro=means.AR_mean_row.mean(), ADD_delta_macro=means.delta_ADD_S_mm.mean(),
                ADD_delta_pooled=group.delta_ADD_S_mm.mean(), ADD_mean_macro=means.ADD_S_mm.mean(),
                ADD_regression_rate=group.regression.mean(),
                positive_BOP_fraction=group.AR_gain.gt(1e-12).mean()))
    pd.DataFrame(records).to_csv(ROOT / "ADD_oracle_summary.csv", index=False)
    decomposition = targets.loc[targets.dataset.eq("combined8")].groupby("object_id")[["gate_blocking_gap", "ranking_gap_inside_gates", "total_oracle_gap"]].mean()
    decomposition.to_csv(ROOT / "object_headroom_decomposition.csv")
    totals = decomposition.mean()
    assert np.isclose(totals.gate_blocking_gap + totals.ranking_gap_inside_gates, totals.total_oracle_gap)

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    s = pd.read_csv(ROOT / "summary.csv")
    selected = s.loc[s.dataset.eq("combined8") & s.scope.eq("all")].set_index("method")
    names = ["D_joint_full", "A_delta_rank", "GT_oracle_inside_joint_gates", "GT_oracle_ADD_component_safe", "GT_oracle_all"]
    labels = ["Joint (frozen)", "Raw-delta baseline A", "GT oracle: joint gates kept", "GT oracle: true ADD/component safe", "GT oracle: any cached candidate"]
    ax = axes[0, 0]
    values = selected.loc[names, "AR_macro"] * 100
    ax.barh(labels[::-1], values.to_numpy()[::-1], color=["#a33236", "#bd7174", "#758e79", "#547b96", "#367f76"])
    ax.set_xlim(39, 42.45)
    for y, value in enumerate(values.to_numpy()[::-1]):
        ax.text(value + .025, y, f"{value:.3f}", va="center", fontsize=9)
    ax.set_xlabel("Equal-object AR (%)")
    ax.set_title("A. Same-bank empirical ceiling (8 objects)", loc="left")
    ax = axes[0, 1]
    g = targets.loc[targets.dataset.eq("combined8")].groupby(["init_regime", "object_id"])[["gate_blocking_gap", "ranking_gap_inside_gates"]].mean().groupby("init_regime").mean().loc[["native", "mild", "hard"]] * 100
    ax.bar(g.index, g.gate_blocking_gap, label="Unavailable behind current gates", color="#ba7843")
    ax.bar(g.index, g.ranking_gap_inside_gates, bottom=g.gate_blocking_gap, label="Wrong choice inside allowed bank", color="#547b96")
    ax.set_ylabel("Remaining oracle gap (AR percentage points)")
    ax.set_ylim(0, 1.25)
    ax.set_title("B. Exact decomposition, not a safety guarantee", loc="left")
    ax.legend(fontsize=8, loc="upper right")
    ax = axes[1, 0]
    ranking = pd.read_csv(ROOT / "rankings.csv")
    groups = ["development22", "holdout5", "added3"]
    x = np.arange(3)
    for i, (score, color, label) in enumerate([("raw_joint_utility", "#367f76", "Raw score"), ("guarded_joint_utility", "#ba7843", "Guarded joint score")]):
        v = ranking.loc[ranking.scope.eq("all") & ranking.score.eq(score)].set_index("dataset").loc[groups, "accuracy"].to_numpy() * 100
        ax.bar(x + (i - .5) * .34, v, .34, label=label, color=color)
    ax.set_xticks(x, ["Development OOF", "Original 5", "Additional 3"])
    ax.set_ylim(60, 95)
    ax.set_ylabel("Within-target pair ranking accuracy (%)")
    ax.set_title("C. Distinguishable AR pairs; descriptive only", loc="left")
    ax.legend(fontsize=8)
    ax = axes[1, 1]
    guard = pd.read_csv(ROOT / "guards.csv")
    for i, (subset, color, label) in enumerate([("all_valid_candidates", "#758e79", "All valid candidates"), ("joint_chosen", "#a33236", "Selected joint winners")]):
        v = guard.loc[guard.scope.eq("all") & guard.target.eq("delta_joint_utility") & guard.subset.eq(subset)].set_index("dataset").loc[groups, "coverage"].to_numpy() * 100
        ax.bar(x + (i - .5) * .34, v, .34, label=label, color=color)
    ax.axhline(75, linestyle="--", linewidth=1, color="#555555", label="Training quantile target (not guarantee)")
    ax.set_xticks(x, ["Development OOF", "Original 5", "Additional 3"])
    ax.set_ylim(35, 90)
    ax.set_ylabel("Empirical one-sided residual coverage (%)")
    ax.set_title("D. Selection and object-shift calibration gap", loc="left")
    ax.legend(fontsize=8, loc="lower left")
    fig.suptitle("Frozen joint: performance headroom and structural bottlenecks\nPost-hoc diagnosis; GT oracles are not deployable algorithms", fontsize=15)
    fig.savefig(ROOT / "joint_headroom_diagnosis.png", dpi=160)
    plt.close(fig)
    print(pd.DataFrame(records).loc[lambda x: x.dataset.eq("combined8") & x.scope.eq("all")].to_string(index=False))
    print("GATE_FRACTION_OF_AR_ORACLE_GAP", totals.gate_blocking_gap / totals.total_oracle_gap)


if __name__ == "__main__":
    main()
