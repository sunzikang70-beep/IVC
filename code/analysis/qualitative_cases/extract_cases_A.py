"""P0-6: gather poses (initial, method choice, fixed Filtered s1.00, GT) and camera for pre-selected cases."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(sys.argv[1]); TLESS = Path(sys.argv[2]); OUT = Path(sys.argv[3]); OUT.mkdir(parents=True, exist_ok=True)
FLIP = np.diag([1.0, -1.0, 1.0])
sel = pd.read_csv(ROOT / "outputs/a_main_method_qualitative_20261105/qualitative_selection_A.csv")
trials = pd.concat([pd.read_csv(ROOT / f"outputs/{d}/trials.csv") for d in
                    ["decoupled_frozen_unseen5_20261021_r4", "decoupled_frozen_unseen3_20261023_r1"]], ignore_index=True)
cases = []
for r in sel.itertuples(index=False):
    t = trials[(trials.object_id == r.object_id) & (trials.scene_id == r.scene_id) & (trials.img_id == r.img_id)
               & (trials.inst_id == r.inst_id) & (trials.input_depth_condition == "local_outlier_30pct")
               & (trials.init_regime == r.init_regime)]
    def pose(method):
        q = t[t.method == method]
        assert len(q) == 1, (r, method, len(q))
        q = q.iloc[0]
        R = np.array([[q[f"R_{i}{j}"] for j in range(3)] for i in range(3)])
        tt = np.array([q.t_x_mm, q.t_y_mm, q.t_z_mm])
        return dict(R=(FLIP @ R).tolist(), t=(FLIP @ tt).tolist(), ADD_S_mm=float(q.ADD_S_mm))
    scene = TLESS / "test_primesense" / f"{int(r.scene_id):06d}"
    gt = json.loads((scene / "scene_gt.json").read_text())[str(int(r.img_id))][int(r.inst_id)]
    assert int(gt["obj_id"]) == int(r.object_id)
    cam = json.loads((scene / "scene_camera.json").read_text())[str(int(r.img_id))]
    cases.append(dict(case=r.case, object_id=int(r.object_id), scene_id=int(r.scene_id), img_id=int(r.img_id),
                      inst_id=int(r.inst_id), init_regime=r.init_regime, selected_action=r.selected_action,
                      AR_init=float(r.init_AR_mean), AR_method=float(r.AR_mean_row), AR_fixed=float(r.init_AR_mean + r.F_gain),
                      K=cam["cam_K"], gt=dict(R=gt["cam_R_m2c"], t=gt["cam_t_m2c"]),
                      initial=pose("protocolB_e2e_initial_pose"), method=pose(r.selected_action),
                      fixed=pose("protocolF_se3_trust_filtered_s100"),
                      rgb=str((scene / "rgb" / f"{int(r.img_id):06d}.png").relative_to(TLESS)).replace("\\", "/")))
    # sanity: initial pose vs GT translation difference should be small (selected init < 0.5 d)
    cases[-1]["init_vs_gt_translation_mm"] = float(np.linalg.norm(np.array(cases[-1]["initial"]["t"]) - np.array(gt["cam_t_m2c"])))
json.dump(cases, open(OUT / "qualitative_cases.json", "w"), indent=1)
for c in cases:
    print(c["case"], c["object_id"], c["rgb"], round(c["init_vs_gt_translation_mm"], 2), c["initial"]["ADD_S_mm"])
