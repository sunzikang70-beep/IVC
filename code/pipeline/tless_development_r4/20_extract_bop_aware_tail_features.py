"""Replay method observations and extract GT-free geometric tail features."""
from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time

import cv2
import numpy as np
import open3d as o3d
import pandas as pd
from scipy.spatial import cKDTree

from evaluate_bop_metrics_controlled import Open3DRaycastRenderer, internal_pose_to_bop
from extract_protocolG_render_compare_features import compare, load_sensor_frame, boundary
from extract_real_gtfree_acceptance_features import masks_by_source_index
from protocolG_observation import prepare_depth, replay_method_depth, validate_condition_rows
from run_frozen_real_inputs_validation import decode_uncompressed_rle, MASK_JSON
import run_p2p_reliability_filter_development as support
from train_bop_aware_group_oof import digest, INITIAL, ACTIONS, KEYS


def tail_statistics(values, prefix):
    values=np.asarray(values,dtype=float)
    values=values[np.isfinite(values)]
    if not len(values):
        return {f"{prefix}_{name}":np.nan for name in ["p50","p90","p95","p99","trim90_mean","tail90_mean"]}
    q50,q90,q95,q99=np.quantile(values,[.5,.9,.95,.99])
    return {f"{prefix}_p50":q50,f"{prefix}_p90":q90,f"{prefix}_p95":q95,f"{prefix}_p99":q99,
            f"{prefix}_trim90_mean":float(values[values<=q90].mean()),
            f"{prefix}_tail90_mean":float(values[values>=q90].mean())}


def points_from_depth(depth_mm, mask, K, limit=2500):
    v,u=np.nonzero(mask & (depth_mm>0))
    stride=max(1,int(np.ceil(len(v)/limit)))
    v,u=v[::stride],u[::stride]
    z=depth_mm[v,u]
    return np.column_stack(((u-K[0,2])*z/K[0,0],(v-K[1,2])*z/K[1,1],z))


def geometry_features(rendered, observed, mask, K):
    values=compare(rendered,observed,mask)
    rendered_mask=rendered>0
    valid=rendered_mask & mask & (observed>0)
    residual=np.abs(observed[valid]-rendered[valid])
    values.update(tail_statistics(residual,"depth_abs_mm"))
    signed=observed[valid]-rendered[valid]
    values["signed_depth_p05_mm"]=float(np.quantile(signed,.05)) if len(signed) else np.nan
    values["signed_depth_p95_mm"]=float(np.quantile(signed,.95)) if len(signed) else np.nan
    source=points_from_depth(rendered,rendered_mask,K)
    target=points_from_depth(observed,mask,K)
    if min(len(source),len(target))>=20:
        forward=cKDTree(target).query(source,k=1)[0]
        reverse=cKDTree(source).query(target,k=1)[0]
    else:
        forward=reverse=np.empty(0)
    values.update(tail_statistics(forward,"forward_nn_mm"))
    values.update(tail_statistics(reverse,"reverse_nn_mm"))
    edge_a,edge_b=boundary(rendered_mask),boundary(mask)
    if edge_a.any() and edge_b.any():
        to_a=cv2.distanceTransform((~edge_a).astype(np.uint8),cv2.DIST_L2,3)
        to_b=cv2.distanceTransform((~edge_b).astype(np.uint8),cv2.DIST_L2,3)
        distances=np.concatenate([to_b[edge_a],to_a[edge_b]])*640/rendered.shape[1]
    else:
        distances=np.empty(0)
    values.update(tail_statistics(distances,"boundary_640px"))
    values["nn_source_sample_points"]=len(source)
    values["nn_target_sample_points"]=len(target)
    return values


def common_support(initial, candidate, observed, mask):
    common=(initial>0)&(candidate>0)&(observed>0)&mask
    result={"common_support_fraction":float(common.sum()/max(mask.sum(),1))}
    if common.any():
        before=np.abs(observed[common]-initial[common])
        after=np.abs(observed[common]-candidate[common])
        for q,name in [(.5,"p50"),(.9,"p90"),(.95,"p95")]:
            result[f"common_abs_{name}_delta_mm"]=float(np.quantile(after,q)-np.quantile(before,q))
        result["common_abs_mean_delta_mm"]=float(after.mean()-before.mean())
        result["common_pixel_improved_fraction"]=float((after<before).mean())
        result["common_pixel_worsened_fraction"]=float((after>before).mean())
    else:
        for name in ["common_abs_p50_delta_mm","common_abs_p90_delta_mm","common_abs_p95_delta_mm",
                     "common_abs_mean_delta_mm","common_pixel_improved_fraction","common_pixel_worsened_fraction"]:
            result[name]=np.nan
    return result


def model_motion(points, row, initial, K, diameter, width):
    R,t=internal_pose_to_bop(row)
    Ri,ti=internal_pose_to_bop(initial)
    a=points@R.T+t.reshape(1,3)
    b=points@Ri.T+ti.reshape(1,3)
    displacement=np.linalg.norm(a-b,axis=1)/diameter
    positive=(a[:,2]>0)&(b[:,2]>0)
    a,b=a[positive],b[positive]
    if len(a):
        ap=a@K.T; bp=b@K.T
        distance=np.linalg.norm(ap[:,:2]/ap[:,2:]-bp[:,:2]/bp[:,2:],axis=1)*640/width
    else:
        distance=np.array([np.nan])
    return {"model_motion_p95_over_diameter":float(np.quantile(displacement,.95)),
            "projected_motion_p95_640px":float(np.nanquantile(distance,.95)) if np.isfinite(distance).any() else np.nan}


def run(args):
    if args.output.exists():
        raise FileExistsError(f"Preserve existing feature run: {args.output}")
    plan=json.loads(args.plan.read_text(encoding="utf-8"))
    cohorts=[]; hashes={str(args.plan):digest(args.plan),str(args.mask_json):digest(args.mask_json)}
    for spec in plan["cohorts"]:
        path=Path(spec["trials"])
        if digest(path)!=spec["trials_sha256"]:
            raise ValueError("Frozen candidate trials changed")
        trials=pd.read_csv(path)
        if set(trials.object_id)&set(plan["excluded_inspected_objects"]):
            raise ValueError("Inspected objects cannot enter tail extraction")
        if args.smoke_targets:
            wanted=trials.frame_id.drop_duplicates().head(args.smoke_targets)
            trials=trials[trials.frame_id.isin(wanted)].copy()
        hashes[str(path)]=digest(path)
        cohorts.append((spec["cohort"],trials))
    wanted_masks=set().union(*[set(t.mask_source_index.astype(int)) for _,t in cohorts])
    print("Reading selected public predicted masks",flush=True)
    masks=masks_by_source_index(args.mask_json,wanted_masks)
    ns=support.load_experiment_namespace(plan["training_objects"],1,1,[.3],dataset_root=args.dataset)
    args.output.mkdir(parents=True)
    state={"status":"EXTRACTING","gt_free":True,"source_hashes":hashes,"completed_conditions":0,
           "reference_depth":"recorded_method_input_not_evaluation_depth",
           "excluded_inspected_objects":plan["excluded_inspected_objects"],"cohorts":[],
           "started_at_utc":datetime.now(timezone.utc).isoformat()}
    rows=[]; points_cache={}; observation_rows=[]; cache={}; renderer=None
    for name,trials in cohorts:
        for condition_index,((frame_id,condition),group) in enumerate(trials.groupby(["frame_id","input_depth_condition"],sort=False),1):
            validate_condition_rows(group)
            initial=group[group.method.eq(INITIAL)].iloc[0]
            if set(group.method)!={INITIAL,*ACTIONS}:
                raise ValueError("Incomplete seven-pose bank")
            predicted_mask=decode_uncompressed_rle(masks[int(initial.mask_source_index)])
            frame=load_sensor_frame(args.dataset,initial,"test_primesense",".png")
            if "observation_version" in group:
                prepared=replay_method_depth(ns,frame,predicted_mask,initial,int(initial.selected_frame_index))
                replay_status="RECORDED_V3_HASHES_VERIFIED"
            else:
                seeds={} if condition=="native_depth" else {"patch_seed":int(initial.patch_seed),"value_seed":int(initial.value_seed)}
                prepared=prepare_depth(ns,frame,predicted_mask,int(initial.object_id),int(initial.selected_frame_index),condition,float(initial.input_outlier_fraction),**seeds)
                if (prepared["center_u"]!=int(initial.patch_center_u) or prepared["center_v"]!=int(initial.patch_center_v)
                        or int(prepared["patch"].sum())!=int(initial.patch_pixels)):
                    raise ValueError("Legacy replay differs from recorded patch geometry")
                replay_status="LEGACY_SEEDS_AND_PATCH_CHECKED_NO_ORIGINAL_DEPTH_HASH"
            observed=prepared["depth"]*1000.
            K=frame["K"]
            if renderer is None:
                h,w=observed.shape
                renderer=Open3DRaycastRenderer(w,h,cache_size=16)
            if observed.shape!=(renderer.height,renderer.width):
                raise ValueError("Mixed image sizes")
            object_id=int(initial.object_id)
            if object_id not in points_cache:
                model=args.dataset/"models_eval"/f"obj_{object_id:06d}.ply"
                renderer.add_object(object_id,str(model))
                pts=np.asarray(o3d.io.read_triangle_mesh(str(model)).vertices,dtype=float)
                points_cache[object_id]=pts[::max(1,int(np.ceil(len(pts)/1000)))]
                hashes[str(model)]=digest(model)
            rendered={}; absolute={}
            for _,row in group.iterrows():
                R,t=internal_pose_to_bop(row)
                image_key=(int(row.scene_id),int(row.img_id),object_id,R.tobytes(),t.tobytes())
                if image_key in cache:
                    render=cache[image_key]
                else:
                    render=renderer.render_object(object_id,R,t,K[0,0],K[1,1],K[0,2],K[1,2])["depth"]
                    cache[image_key]=render
                    if len(cache)>32:
                        del cache[next(iter(cache))]
                rendered[row.method]=render
                absolute[row.method]=geometry_features(render,observed,predicted_mask,K)
            reference_features=absolute[INITIAL]
            for _,row in group.iterrows():
                values=absolute[row.method]
                record={**{key:row[key] for key in KEYS},"method":row.method,"cohort":name,
                        **prepared["metadata"],"replay_status":replay_status}
                for key,value in values.items():
                    record[f"tail_{key}"]=value
                    record[f"tail_delta_{key}"]=value-reference_features[key]
                record.update(common_support(rendered[INITIAL],rendered[row.method],observed,predicted_mask))
                record.update(model_motion(points_cache[object_id],row,initial,K,float(initial.diameter_mm),renderer.width))
                rows.append(record)
            observation_rows.append(dict(cohort=name,frame_id=frame_id,input_depth_condition=condition,
                                         replay_status=replay_status,**prepared["metadata"]))
            state["completed_conditions"]+=1
            if condition_index%20==0:
                print(f"{name}: {condition_index}/{group_count(trials)} conditions",flush=True)
                pd.DataFrame(rows).to_csv(args.output/"tail_feature_checkpoint.csv",index=False)
                (args.output/"feature_state.json").write_text(json.dumps(state,indent=2),encoding="utf-8")
        state["cohorts"].append({"cohort":name,"pose_rows":len(trials)})
    table=pd.DataFrame(rows)
    expected=sum(len(t) for _,t in cohorts)
    if len(table)!=expected or table.duplicated(["cohort",*KEYS,"method"]).any():
        raise ValueError("Tail feature coverage mismatch")
    table.to_csv(args.output/"tail_geometry_features.csv",index=False)
    pd.DataFrame(observation_rows).to_csv(args.output/"observation_replay_checks.csv",index=False)
    metadata={"cohort",*KEYS,"method","replay_status",*prepared["metadata"]}
    names=[c for c in table if c not in metadata]
    if any(c in names for c in ["ADD_S_mm","mask_iou","MSSD_normalized","init_difficulty"]):
        raise ValueError("Outcome field entered tail feature schema")
    snapshot=args.output/"source_snapshot"; snapshot.mkdir()
    for source in [Path(__file__),Path(__file__).with_name("protocolG_observation.py"),support.SOURCE]:
        hashes[str(source)]=digest(source)
        shutil.copy2(source,snapshot/source.name)
    state.update(status="SMOKE_COMPLETE" if args.smoke_targets else "COMPLETE",candidate_rows=len(table),
                 feature_names=names,source_hashes=hashes,feature_csv_sha256=digest(args.output/"tail_geometry_features.csv"),
                 completed_at_utc=datetime.now(timezone.utc).isoformat(),
                 legacy_replay_limitation="Original 1050 rows have no depth hash; seeds/patch checked, new hashes recorded, no false retrospective hash claim")
    (args.output/"feature_state.json").write_text(json.dumps(state,indent=2),encoding="utf-8")
    print("TAIL FEATURES COMPLETE",len(table),"pose rows;",len(names),"features",flush=True)


def group_count(trials):
    return len(trials[["frame_id","input_depth_condition"]].drop_duplicates())


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    root=Path(r"D:\6D_pose\codex_paper_workspace")
    parser.add_argument("--plan",type=Path,default=root/"outputs/bop_threshold_diagnosis_20261019_r1/development_label_plan/development_bop_label_plan.json")
    parser.add_argument("--dataset",type=Path,default=Path(r"D:\6D_pose\bop_datasets\tless"))
    parser.add_argument("--mask-json",type=Path,default=MASK_JSON)
    parser.add_argument("--output",type=Path,default=root/"outputs/bop_aware_tail_geometry_20261019_r1")
    parser.add_argument("--smoke-targets",type=int)
    run(parser.parse_args())
