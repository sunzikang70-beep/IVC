"""Qualitative case selection for arm A with the rule stated in the paper (fixed before looking at images):
local-outlier depth, initial ADD-S < 0.5 diameter, two cases per category from different objects,
ranked by effect size. Categories: accepted with largest AR gain; rejected where fixed filtered P2P cap 1
would lose most AR; accepted with largest AR loss."""
import sys
from pathlib import Path
import pandas as pd
ROOT = Path(sys.argv[1]); OUT = Path(sys.argv[2])
KEY = ["object_id", "scene_id", "img_id", "inst_id", "input_depth_condition", "init_regime"]
m = pd.read_csv(ROOT / "outputs/p0_main_table_baselines_20261104/merged_per_target.csv.gz")
dia = pd.concat([pd.read_csv(ROOT / f"outputs/{d}/bop/bop_pose_errors.csv", usecols=["object_id", "diameter_mm"]).drop_duplicates()
                 for d in ["decoupled_frozen_unseen5_20261021_r4", "decoupled_frozen_unseen3_20261023_r1"]]).drop_duplicates().set_index("object_id").diameter_mm
a = m[m.arm.eq("A ordinary ranking")].set_index(KEY)
f = m[m.arm.eq("Filtered P2P s1.00")].set_index(KEY).loc[a.index]
a = a.assign(F_gain=(f.AR_mean_row - f.init_AR_mean).to_numpy(), F_dADD=f.delta_ADD_S_mm.to_numpy(),
             AR_gain=a.AR_mean_row - a.init_AR_mean).reset_index()
a = a[a.input_depth_condition.eq("local_outlier_30pct") & (a.initial_ADD_S_mm < 0.5 * a.object_id.map(dia))]
def pick(d, col, asc):
    out, used = [], set()
    for r in d.sort_values([col, "object_id", "scene_id", "img_id"], ascending=[asc, True, True, True]).itertuples(index=False):
        if r.object_id in used: continue
        out.append(r); used.add(r.object_id)
        if len(out) == 2: break
    return pd.DataFrame(out)
succ = pick(a[a["update"].eq(1) & (a.AR_gain > 0)], "AR_gain", False).assign(case="success")
rej = pick(a[a["update"].eq(0) & (a.F_gain < 0)], "F_gain", True).assign(case="correct_rejection")
wrong = pick(a[a["update"].eq(1) & (a.AR_gain < 0)], "AR_gain", True).assign(case="wrong_acceptance")
sel = pd.concat([succ, rej, wrong])[["case", "object_id", "scene_id", "img_id", "inst_id", "init_regime", "selected_action",
                                      "init_AR_mean", "AR_mean_row", "AR_gain", "initial_ADD_S_mm", "ADD_S_mm", "F_gain", "F_dADD"]]
OUT.mkdir(parents=True, exist_ok=True)
sel.to_csv(OUT / "qualitative_selection_A.csv", index=False)
print(sel.round(3).to_string(index=False))
