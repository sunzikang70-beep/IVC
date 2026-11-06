"""Fixed-plan tail-feature and group-calibration development experiment."""
from __future__ import annotations

import os
for _name in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"]:
    os.environ[_name]="1"

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import train_bop_aware_group_oof as base


INITIAL=base.INITIAL
ACTIONS=base.ACTIONS
AR=base.AR
TARGETS=base.TARGETS
CLASSES=base.CLASSES
STATE_FEATURES=["candidate_candidate_joint_support_fraction","candidate_front_free_space_fraction",
                "actual_translation_change_mm","tail_silhouette_iou","tail_depth_abs_mm_p95",
                "tail_delta_depth_abs_mm_p95","common_support_fraction"]


def object_frame_weights(data):
    objects=data.object_id.nunique()
    frames=data.groupby("object_id").frame_id.nunique()
    counts=data.groupby(["object_id","frame_id"]).size()
    result=np.array([1/(objects*frames.loc[o]*counts.loc[(o,f)]) for o,f in zip(data.object_id,data.frame_id)])
    return result*len(result)/result.sum()


def weighted_quantile(values,weights,q):
    order=np.argsort(values)
    v=np.asarray(values)[order]
    w=np.asarray(weights)[order]
    return float(v[min(np.searchsorted(np.cumsum(w),q*w.sum(),side="left"),len(v)-1)])


def fit_raw(data,names,plan):
    active=data.loc[base.mask_fit(data)]
    imputer=SimpleImputer(strategy="median",add_indicator=True,keep_empty_features=True)
    X=imputer.fit_transform(active[names].to_numpy(dtype=float))
    scaler=StandardScaler().fit(X)
    X=np.clip(scaler.transform(X),-20,20)
    weights=object_frame_weights(active)
    params=dict(plan["boosting"],random_state=base.SEED)
    regressors={name:HistGradientBoostingRegressor(**params).fit(X,active[name],sample_weight=weights) for name in TARGETS}
    classifiers={}
    for name in CLASSES:
        y=active[name].to_numpy(dtype=int)
        if len(np.unique(y))<2:
            classifiers[name]=float((np.dot(weights,y)+.5)/(weights.sum()+1))
        else:
            classifiers[name]=HistGradientBoostingClassifier(**params).fit(X,y,sample_weight=weights)
    neighbors=NearestNeighbors(n_neighbors=2).fit(X)
    radius=max(float(np.quantile(neighbors.kneighbors(X)[0][:,-1],.95)),1.)
    return dict(imputer=imputer,scaler=scaler,regressors=regressors,classifiers=classifiers,
                neighbors=neighbors,domain_radius=radius,feature_names=names,
                training_objects=sorted(map(int,data.object_id.unique())))


def predict_raw(model,data):
    X=base.transform(model,data,model["feature_names"])
    result=pd.DataFrame(index=data.index)
    for target in TARGETS:
        values=model["regressors"][target].predict(X)
        result["pred_"+target]=np.clip(values,-1,1) if target!="delta_ADD_S_mm" else values
    for name in CLASSES:
        clf=model["classifiers"][name]
        result["raw_p_"+name]=clf if isinstance(clf,float) else clf.predict_proba(X)[:,1]
    result["training_support"]=1/(1+model["neighbors"].kneighbors(X,n_neighbors=1)[0][:,0]/model["domain_radius"])
    return result


def raw_oof(data,names,plan):
    output=[]
    for train,valid in base.folds(data,3):
        fitting,held=data.iloc[train],data.iloc[valid]
        if set(fitting.object_id)&set(held.object_id):
            raise ValueError("Object leakage in raw OOF")
        output.append(predict_raw(fit_raw(fitting,names,plan),held))
    return pd.concat(output).sort_index().loc[data.index]


def calibration_matrix(raw,data,label,states):
    matrix=pd.DataFrame({"raw_logit":base.logits(raw["raw_p_"+label]).reshape(-1)},index=data.index)
    for key in states:
        matrix[key]=data[key].to_numpy(dtype=float)
    return matrix.replace([np.inf,-np.inf],np.nan).to_numpy(dtype=float)


def fit_calibration(data,raw,names):
    valid=base.mask_fit(data)
    active=data.loc[valid]
    predictions=raw.loc[valid]
    weights=object_frame_weights(active)
    states=[name for name in STATE_FEATURES if name in names]
    calibrators={}
    for label in CLASSES:
        y=active[label].to_numpy(dtype=int)
        if len(np.unique(y))<2:
            calibrators[label]=float((np.dot(weights,y)+.5)/(weights.sum()+1))
        else:
            pipeline=make_pipeline(SimpleImputer(strategy="median",keep_empty_features=True),StandardScaler(),
                                   LogisticRegression(C=.1,max_iter=2000,random_state=base.SEED))
            pipeline.fit(calibration_matrix(predictions,active,label,states),y,
                         logisticregression__sample_weight=weights)
            calibrators[label]=pipeline
    corrections={}
    for action in ACTIONS:
        subset=active[active.action.eq(action)]
        if len(subset)<8:
            subset=active
        w=object_frame_weights(subset)
        guards=[]
        for i,target in enumerate(TARGETS):
            residual=(predictions.loc[subset.index,"pred_"+target]-subset[target]) if i<3 else (subset[target]-predictions.loc[subset.index,"pred_"+target])
            guards.append(max(0.,weighted_quantile(residual,w,.75)))
        corrections[action]=guards
    return dict(calibrators=calibrators,states=states,corrections=corrections)


def calibrate(raw,data,calibration):
    result=raw.copy()
    for label in CLASSES:
        model=calibration["calibrators"][label]
        result["p_"+label]=model if isinstance(model,float) else model.predict_proba(calibration_matrix(raw,data,label,calibration["states"]))[:,1]
    for i,target in enumerate(TARGETS):
        correction=data.action.map({a:g[i] for a,g in calibration["corrections"].items()}).to_numpy()
        result[("lcb_" if i<3 else "ucb_")+target]=result["pred_"+target]+(-correction if i<3 else correction)
    return result


def fit_bundle(data,names,plan):
    raw=raw_oof(data,names,plan)
    calibration=fit_calibration(data,raw,names)
    calibrated=calibrate(raw,data,calibration)
    active=data.loc[base.mask_fit(data)].copy()
    active["calibration_residual"]=active.adds_regression-calibrated.loc[active.index,"p_adds_regression"]
    group_bias=[]
    for _,group in active.groupby("object_id"):
        w=object_frame_weights(group)
        group_bias.append(float(np.average(group.calibration_residual,weights=w)))
    guard=max(0.,float(np.quantile(group_bias,.8)))
    return dict(raw_model=fit_raw(data,names,plan),calibration=calibration,risk_bias_guard=guard,
                feature_names=names,training_objects=sorted(map(int,data.object_id.unique())))


def predict_bundle(bundle,data):
    raw=predict_raw(bundle["raw_model"],data)
    result=calibrate(raw,data,bundle["calibration"])
    result["adds_risk_upper"]=np.minimum(1.,result.p_adds_regression+bundle["risk_bias_guard"])
    return result


def calibrated_oof(data,names,plan):
    parts=[]
    for train,valid in base.folds(data,3):
        training,held=data.iloc[train],data.iloc[valid]
        parts.append(predict_bundle(fit_bundle(training,names,plan),held))
    return pd.concat(parts).sort_index().loc[data.index]


def decisions(data,predictions,policy):
    if not data.index.equals(predictions.index):
        raise ValueError("Decision index mismatch")
    lower=predictions[["lcb_"+t for t in TARGETS[:3]]].to_numpy()
    mean=lower.mean(axis=1)
    add_upper=predictions.ucb_delta_ADD_S_mm.to_numpy()
    weight=policy["continuous_weight"]
    utility=mean-weight*add_upper/5.
    allowed=(base.mask_fit(data)&(data.observation_support.to_numpy()>=.01)
             &(predictions.training_support.to_numpy()>=.2)
             &(predictions.adds_risk_upper.to_numpy()<=policy["risk_limit"])
             &(add_upper<=0)&(lower.min(axis=1)>=-policy["component_loss"])
             &(mean>=(-.001 if weight>0 else 0))&(utility>0))
    if policy.get("disabled",False):
        allowed[:]=False
    selected={}
    for target,indices in data.groupby("target_id",sort=True).groups.items():
        positions=data.index.get_indexer(indices)
        eligible=positions[allowed[positions]]
        if len(eligible):
            position=min(eligible,key=lambda i:(-utility[i],predictions.iloc[i].adds_risk_upper,str(data.iloc[i].action)))
            selected[target]=data.iloc[position].action
        else:
            selected[target]=INITIAL
    return selected


def evaluate_choices(data,choices):
    rows=[]
    for target,group in data.groupby("target_id",sort=True):
        action=choices[target]
        row=group.iloc[0] if action==INITIAL else group[group.action.eq(action)].iloc[0]
        add=row.initial_ADD_S_mm if action==INITIAL else row.ADD_S_mm
        ar=row[["initial_"+c for c in AR]].to_numpy(dtype=float) if action==INITIAL else row[AR].to_numpy(dtype=float)
        rows.append(dict(target_id=target,object_id=row.object_id,scene_id=row.scene_id,img_id=row.img_id,
                         inst_id=row.inst_id,frame_id=row.frame_id,input_depth_condition=row.input_depth_condition,
                         cohort=row.cohort,selected_action=action,ADD_S_mm=add,initial_ADD_S_mm=row.initial_ADD_S_mm,
                         delta_ADD_S_mm=add-row.initial_ADD_S_mm,**{c:ar[i] for i,c in enumerate(AR)},AR_mean_row=ar.mean()))
    return pd.DataFrame(rows)


def tune_policy(data,predictions,plan):
    initial=base.baseline_rows(data,INITIAL).set_index("target_id")
    records=[]
    for risk in plan["risk_limits"]:
        for loss in plan["component_ar_loss_limits"]:
            for weight in plan["continuous_gain_weights"]:
                policy=dict(risk_limit=risk,component_loss=loss,continuous_weight=weight,disabled=False)
                selected=evaluate_choices(data,decisions(data,predictions,policy))
                active=selected[selected.selected_action.ne(INITIAL)]
                coverage=len(active)/len(selected)
                conditional_risk=float((active.delta_ADD_S_mm>1e-9).mean()) if len(active) else 1.
                objectives=[]; cohort_checks=True
                for _,group in selected.groupby("cohort"):
                    delta_ar=float((group.set_index("target_id").AR_mean_row-initial.loc[group.target_id,"AR_mean_row"]).mean())
                    add_delta=group.delta_ADD_S_mm.mean()
                    cohort_checks &= delta_ar>=-1e-12 and add_delta<=1e-12
                    objectives.append(delta_ar-.005*add_delta/5)
                passes=coverage>=.05 and conditional_risk<=.2 and cohort_checks
                records.append(dict(**policy,coverage=coverage,accepted_regression_rate=conditional_risk,
                                    mean_AR=selected.AR_mean_row.mean(),adds_delta_mm=selected.delta_ADD_S_mm.mean(),
                                    regression_rate=(selected.delta_ADD_S_mm>1e-9).mean(),
                                    equal_cohort_joint_gain=float(np.mean(objectives)),every_cohort_nonregressed=cohort_checks,
                                    passes_development_constraints=passes))
    feasible=[r for r in records if r["passes_development_constraints"]]
    if not feasible:
        return dict(risk_limit=.1,component_loss=0.,continuous_weight=0.,disabled=True),records
    chosen=min(feasible,key=lambda r:(-r["equal_cohort_joint_gain"],r["accepted_regression_rate"],-r["coverage"],r["risk_limit"]))
    return {key:chosen[key] for key in ["risk_limit","component_loss","continuous_weight","disabled"]},records


def discrimination(data,predictions,variant):
    records=[]
    for cohort,subset in [("pooled",data),*list(data.groupby("cohort"))]:
        active=subset.loc[base.mask_fit(subset)]
        for label in CLASSES:
            y=active[label]
            for mode,prefix in [("raw","raw_p_"),("calibrated","p_")]:
                p=predictions.loc[active.index,prefix+label]
                records.append(dict(variant=variant,cohort=cohort,label=label,mode=mode,rows=len(active),
                                    prevalence=y.mean(),AUROC=roc_auc_score(y,p) if y.nunique()==2 else np.nan,
                                    average_precision=average_precision_score(y,p) if y.sum()>0 else np.nan,
                                    brier=float(np.mean((y-p)**2))))
    return pd.DataFrame(records)


def run(args):
    if args.output.exists():
        raise FileExistsError(f"Preserve existing experiment: {args.output}")
    plan=json.loads(args.plan.read_text(encoding="utf-8"))
    tail_state=json.loads((args.features/"feature_state.json").read_text(encoding="utf-8"))
    label_state=json.loads((args.labels/"label_completion_state.json").read_text(encoding="utf-8"))
    old_state=json.loads((args.previous/"training_state.json").read_text(encoding="utf-8"))
    if tail_state["status"]!="COMPLETE" or label_state["status"]!="COMPLETE":
        raise ValueError("Full feature/label completion required")
    feature_path=args.features/"tail_geometry_features.csv"
    labels_path=args.labels/"development_bop_labels.csv"
    if base.digest(feature_path)!=tail_state["feature_csv_sha256"] or base.digest(labels_path)!=label_state["labels_sha256"]:
        raise ValueError("Feature/label hash mismatch")
    inventory=json.loads(args.inventory.read_text(encoding="utf-8"))
    parts=[]; names=None
    source_hashes={str(args.plan):base.digest(args.plan),str(feature_path):base.digest(feature_path),str(labels_path):base.digest(labels_path)}
    for spec in inventory["cohorts"]:
        path=Path(spec["trials"]); scores=path.with_name("se3_trust_region_scores.csv")
        if base.digest(path)!=spec["trials_sha256"] or base.digest(scores)!=label_state["source_hashes"][str(scores)]:
            raise ValueError("Frozen feature bank changed")
        long,current=base.build_features(pd.read_csv(scores),pd.read_csv(path),spec["cohort"])
        if names is not None and names!=current:
            raise ValueError("Base feature schemas differ")
        names=current; parts.append(long)
        source_hashes[str(path)]=base.digest(path); source_hashes[str(scores)]=base.digest(scores)
    features=pd.concat(parts,ignore_index=True)
    tails=pd.read_csv(feature_path)
    merge_keys=["cohort",*base.KEYS,"action"]
    tail_names=tail_state["feature_names"]
    features=features.merge(tails.rename(columns={"method":"action"})[merge_keys+tail_names],on=merge_keys,validate="one_to_one")
    features[tail_names]=features[tail_names].replace([np.inf,-np.inf],np.nan)
    data=base.attach_labels(features,pd.read_csv(labels_path)).reset_index(drop=True)
    if len(data)!=3396 or data.object_id.nunique()!=22 or set(data.object_id)&set(plan["excluded_objects"]):
        raise ValueError("Development split/coverage mismatch")
    args.output.mkdir(parents=True)
    features.to_csv(args.output/"observable_features_with_tails.csv",index=False)
    state=dict(status="V2_EXPLORATORY_NESTED_OOF_RUNNING",plan_sha256=base.digest(args.plan),
               source_hashes=source_hashes,variants={},training_objects=sorted(map(int,data.object_id.unique())),
               excluded_inspected_objects=plan["excluded_objects"],outer_results_previously_inspected=True,
               new_final_holdout_performed=False,method_promoted=False,started_at_utc=datetime.now(timezone.utc).isoformat())
    summaries=[]; all_discrimination=[]; all_outcomes=[]
    for variant in plan["variants"]:
        variant_names=names if variant=="base_group_boost" else names+tail_names
        directory=args.output/variant; directory.mkdir()
        linear=variant=="tail_linear_control"
        predictions=[]; selections=[]; grids=[]; fold_records=[]
        for fold,(train,valid) in enumerate(base.folds(data,5)):
            training,held=data.iloc[train],data.iloc[valid]
            expected=set(old_state["outer_folds"][fold]["validation_objects"])
            if set(held.object_id)!=expected or set(training.object_id)&set(held.object_id):
                raise ValueError("Outer partition changed or leaked")
            print(f"{variant} OUTER {fold+1}/5: inner group training/calibration",flush=True)
            if linear:
                inner=base.calibrated_oof(training,variant_names)
                policy,grid=base.tune_policy(training,inner)
                bundle=base.fit_bundle(training,variant_names)
                predicted=base.predict_bundle(bundle,held)
                selected=base.select(held,predicted,**policy)
            else:
                inner=calibrated_oof(training,variant_names,plan)
                policy,grid=tune_policy(training,inner,plan)
                bundle=fit_bundle(training,variant_names,plan)
                predicted=predict_bundle(bundle,held)
                selected=evaluate_choices(held,decisions(held,predicted,policy))
            predicted["outer_fold"]=fold
            predictions.append(predicted)
            selected["outer_fold"]=fold; selections.append(selected)
            grids += [dict(row,outer_fold=fold) for row in grid]
            record=dict(fold=fold,training_objects=sorted(map(int,training.object_id.unique())),
                        validation_objects=sorted(map(int,held.object_id.unique())),policy=policy,
                        selected_updates=int(selected.selected_action.ne(INITIAL).sum()))
            fold_records.append(record)
            joblib.dump(dict(bundle=bundle,policy=policy,fold=record),directory/f"outer_fold_{fold}.joblib")
            state["variants"][variant]=dict(status="RUNNING",folds=fold_records)
            base.dump(args.output/"v2_training_state.json",state)
            print("OUTER COMPLETE",record,flush=True)
        oof=pd.concat(predictions).sort_index().loc[data.index]
        selected=pd.concat(selections,ignore_index=True); selected["method"]=variant
        selected.to_csv(directory/"outer_oof_selected.csv",index=False)
        export=pd.concat([data[["target_id","object_id","cohort","action","pose_changed","features_valid"]],oof],axis=1)
        export.to_csv(directory/"outer_oof_candidate_predictions.csv",index=False)
        pd.DataFrame(grids).to_csv(directory/"inner_policy_grid.csv",index=False)
        stats,bins=base.calibration_report(data,oof)
        stats.to_csv(directory/"calibration_metrics.csv",index=False)
        bins.to_csv(directory/"reliability_bins.csv",index=False)
        disc=discrimination(data,oof,variant); all_discrimination.append(disc)
        disc.to_csv(directory/"cohort_discrimination.csv",index=False)
        if linear:
            final_policy,final_grid=base.tune_policy(data,oof)
            final_bundle=base.fit_bundle(data,variant_names)
            predictor=base.predict_bundle
        else:
            final_policy,final_grid=tune_policy(data,oof,plan)
            final_bundle=fit_bundle(data,variant_names,plan)
            predictor=predict_bundle
        joblib.dump(dict(bundle=final_bundle,policy=final_policy,status="EXPLORATORY_DEVELOPMENT_NOT_PROMOTED",variant=variant),directory/"development_candidate.joblib")
        pd.DataFrame(final_grid).to_csv(directory/"final_policy_grid_exploratory.csv",index=False)
        one=data[data.target_id.eq(data.target_id.iloc[0])]
        full_prediction=predictor(final_bundle,data)
        if not np.allclose(full_prediction.loc[one.index],predictor(final_bundle,one),atol=1e-9,rtol=1e-9):
            raise ValueError("Inference batch dependence")
        shadow=data.copy()
        for column in TARGETS+CLASSES+["ADD_S_mm","initial_ADD_S_mm"]+AR:
            shadow[column]=9999.
        if not np.allclose(full_prediction,predictor(final_bundle,shadow),atol=1e-9,rtol=1e-9):
            raise ValueError("Outcome-dependent inference")
        if not linear and decisions(data,full_prediction,final_policy)!=decisions(shadow,full_prediction,final_policy):
            raise ValueError("Outcome-dependent choices")
        all_outcomes.append(selected)
        state["variants"][variant]=dict(status="COMPLETE",folds=fold_records,feature_count=len(variant_names),
                                       final_development_policy=final_policy,batch_invariance=True,gt_shadow_inference=True)
        base.dump(args.output/"v2_training_state.json",state)
        print("VARIANT COMPLETE",variant,flush=True)
    for method in [INITIAL,"protocolF_se3_trust_p2p_s100","protocolF_se3_trust_filtered_s100"]:
        rows=base.baseline_rows(data,method); rows["method"]=method; all_outcomes.append(rows)
    old=pd.read_csv(args.previous/"nested_oof_selected_and_baselines.csv")
    old=old[old.method.eq("bop_aware_nested_oof")].copy(); old["method"]="v1_bop_aware_oof"
    all_outcomes.append(old)
    outcomes=pd.concat(all_outcomes,ignore_index=True)
    outcomes.to_csv(args.output/"all_variant_oof_outcomes.csv",index=False)
    summary=base.metric_summary(outcomes)
    conditional=[]
    for scope,stratum,group in [("all","pooled",outcomes)]+[("cohort",key,group) for key,group in outcomes.groupby("cohort")]:
        for method,part in group.groupby("method"):
            active=part[part.selected_action.ne(INITIAL)]
            conditional.append(dict(scope=scope,stratum=stratum,method=method,updates=len(active),
                                    regressions=int((active.delta_ADD_S_mm>1e-9).sum()),
                                    accepted_regression_rate=float((active.delta_ADD_S_mm>1e-9).mean()) if len(active) else np.nan))
    summary=summary.merge(pd.DataFrame(conditional),on=["scope","stratum","method"],how="left",validate="one_to_one")
    summary.to_csv(args.output/"v2_oof_comparison.csv",index=False)
    pd.concat(all_discrimination,ignore_index=True).to_csv(args.output/"all_cohort_discrimination.csv",index=False)
    write_report(args.output,summary,state)
    snapshot=args.output/"source_snapshot"; snapshot.mkdir()
    for name in [Path(__file__).name,"extract_bop_aware_tail_features.py","train_bop_aware_group_oof.py",
                 "protocolG_observation.py","test_bop_aware_tail_risk_v2.py",args.plan.name]:
        source=Path(__file__).with_name(name)
        state["source_hashes"][str(source)]=base.digest(source)
        shutil.copy2(source,snapshot/name)
    state.update(status="COMPLETE_EXPLORATORY_DEVELOPMENT_NOT_PROMOTED",completed_at_utc=datetime.now(timezone.utc).isoformat())
    state["artifact_hashes"]={str(path.relative_to(args.output)):base.digest(path) for path in args.output.rglob("*") if path.is_file() and path.name!="v2_training_state.json"}
    base.dump(args.output/"v2_training_state.json",state)
    print(summary[summary.scope.isin(["all","cohort"])].to_string(index=False),flush=True)
    print("V2 DEVELOPMENT COMPLETE",args.output,flush=True)


def write_report(output,summary,state):
    pooled=summary[summary.scope.eq("all")]
    mild=summary[summary.scope.eq("cohort") & summary.stratum.str.endswith("_mild")]
    lines=["# 尾部几何、对象均衡风险模型与跨物体校准开发实验", "", "## Material Passport",
           "- Status: EXPLORATORY_DEVELOPMENT_NOT_PROMOTED; Verification Status: ANALYZED",
           "- 同一 22 开发物体、同一候选位姿和原始传感器 VSD 标签；五个诊断物体未用于拟合或选择。",
           "- 本轮沿用已查看的外层对象划分，是探索性开发比较，不是新的独立确认。",
           "- 新特征使用记录的实际方法输入深度和预测 Mask，不使用评价参考深度、GT Mask 或 GT 位姿。",
           "- tail_linear_control 仅增加尾部特征，使用旧线性模型/校准/策略；base_group_boost 使用新模型和校准但无尾部特征；tail_group_boost 是完整组合。",
           "- 新模型为固定参数的浅层 sklearn Histogram Gradient Boosting；按物体和物理目标实例均衡样本权重。",
           "- 校准是基于对象 OOF 的状态条件化概率模型，既可能影响排序也影响概率，不冒称为纯单变量 Platt 变换。",
           "- 风险经验界使用非负对象校准残差修正，不是逐帧保证。",
           "- 新策略在内层同时要求每个 cohort 的 ADD-S 与 AR 不退化、至少 5% 更新、接受后退化率不高于 20%；失败则拒绝全部更新，不按外层结果重调。",
           "", "## 整体 OOF", "```text",pooled.to_string(index=False),"```",
           "", "## Mild OOF", "```text",mild.to_string(index=False),"```",
           "", "## 边界",
           "- 模型、校准、残差界和策略选择都只使用外层训练物体；填充器和标准化器按折拟合。",
           "- 报告整体/条件回归率和更新覆盖率，不能用大量保持初始化掩盖接受后的高退化率。",
           "- 同一对象的全部深度、初始化档和动作保持同组；不同对象可能共享图像，不能宣称场景独立。",
           "- 旧 1,050 条记录没有原始观测 hash，本次按记录种子、预测 Mask 和 patch 几何重建并记录新 hash，不追溯声称原始 hash 已验证。",
           "- 当前没有新独立确认集、云端扩量或真实平台验证；不能依据本轮开发结果自动升级论文主方法。",""]
    (output/"tail_risk_v2_report_zh.md").write_text("\n".join(lines),encoding="utf-8")
    fig,ax=plt.subplots(figsize=(9,4),constrained_layout=True)
    labels=[]
    for row in mild.itertuples():
        label=row.method.replace("protocolF_se3_trust_","").replace("protocolB_e2e_","")
        labels.append(label)
    colors=["#267f6c" if "group_boost" in label else "#727272" for label in labels]
    ax.barh(labels,mild.delta_ADD_S_mm.to_numpy(),color=colors)
    ax.axvline(0,color="#222222",linewidth=.8)
    ax.set_xlabel("Mild ADD-S change vs initializer (mm; lower is better)")
    fig.savefig(output/"mild_adds_comparison.png",dpi=180)
    plt.close(fig)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    root=Path(r"D:\6D_pose\codex_paper_workspace")
    parser.add_argument("--labels",type=Path,default=root/"outputs/bop_aware_development_labels_20261019_r1")
    parser.add_argument("--features",type=Path,default=root/"outputs/bop_aware_tail_geometry_20261019_r1")
    parser.add_argument("--previous",type=Path,default=root/"outputs/bop_aware_group_oof_20261019_r1")
    parser.add_argument("--inventory",type=Path,default=root/"outputs/bop_threshold_diagnosis_20261019_r1/development_label_plan/development_bop_label_plan.json")
    parser.add_argument("--plan",type=Path,default=Path(__file__).with_name("bop_aware_tail_risk_v2_plan_20261019.json"))
    parser.add_argument("--output",type=Path,default=root/"outputs/bop_aware_tail_risk_v2_20261019_r1")
    run(parser.parse_args())
