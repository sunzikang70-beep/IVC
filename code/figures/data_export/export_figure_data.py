"""Export per-condition data used by the added result figures (read-only re-aggregation).
Usage: python export_figure_data.py <codex_paper_workspace> <tless_models_info.json> <ycbv_models_info.json>"""
import json, sys
from pathlib import Path
import pandas as pd
ROOT=Path(sys.argv[1]); sys.path.insert(0,str(ROOT/'work'))
import analyze_decoupled_frozen_external_ycbv as ycbv
KEY=["object_id","scene_id","img_id","inst_id","input_depth_condition","init_regime"]
ARMS=["Initial","P2P s0.25","P2P s0.50","P2P s1.00","Filtered P2P s0.25","Filtered P2P s0.50","Filtered P2P s1.00","A ordinary ranking"]
out=ROOT/"outputs/figure_data_20261105"
t=pd.read_csv(ROOT/"outputs/p0_main_table_baselines_20261104/merged_per_target.csv.gz")
y,_=ycbv.load(ROOT/"outputs/decoupled_frozen_ycbv_external_20261104")
for name,d,info in [("T-LESS",t,sys.argv[2]),("YCB-V",y,sys.argv[3])]:
    d=d.loc[d.arm.isin(ARMS)]
    w=d.pivot_table(index=KEY,columns="arm",values="AR_mean_row")
    a=d.loc[d.arm.eq("A ordinary ranking")].set_index(KEY)
    w["A_selected_action"]=a.selected_action
    w["initial_ADD_S_mm"]=a.initial_ADD_S_mm
    w["A_delta_ADD_S_mm"]=a.delta_ADD_S_mm
    f=d.loc[d.arm.eq("Filtered P2P s1.00")].set_index(KEY)
    w["F_delta_ADD_S_mm"]=f.delta_ADD_S_mm
    mi=json.load(open(info)); w=w.reset_index()
    w["diameter_mm"]=w.object_id.map(lambda o: mi[str(int(o))]["diameter"])
    w["dataset"]=name
    w.to_csv(out/f"per_condition_{name.replace('-','').lower()}.csv",index=False)
    print(name,len(w),w.object_id.nunique())
