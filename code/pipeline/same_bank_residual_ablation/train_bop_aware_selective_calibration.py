"""Exploratory follow-up: group-OOF-select identity/Platt/state calibration."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import train_bop_aware_group_oof as base
import train_bop_aware_tail_risk_v2 as v2


def apply_probability(model,raw,data,label,states):
    if model is None:
        return raw["raw_p_"+label].to_numpy()
    if isinstance(model,float):
        return np.full(len(data),model)
    return model.predict_proba(v2.calibration_matrix(raw,data,label,states))[:,1]


def object_scores(data,probabilities,label):
    frame=data[["object_id",label]].copy()
    frame["p"]=probabilities
    frame["brier"]=(frame[label]-frame.p)**2
    brier=frame.groupby("object_id").brier.mean()
    auc=[]
    for _,group in frame.groupby("object_id"):
        if group[label].nunique()==2:
            auc.append(roc_auc_score(group[label],group.p))
    return brier,float(np.mean(auc)) if auc else np.nan


def fit_calibration(data,raw,names):
    active=data.loc[base.mask_fit(data)]
    active_raw=raw.loc[active.index]
    template=v2.fit_calibration(data,raw,names)
    modes={"identity":[],"platt":[],"state":[name for name in v2.STATE_FEATURES if name in names]}
    predictions={mode:pd.DataFrame(index=active.index,columns=base.CLASSES,dtype=float) for mode in modes}
    for train,valid in base.folds(active,3):
        training,held=active.iloc[train],active.iloc[valid]
        if set(training.object_id)&set(held.object_id):
            raise ValueError("Object leakage in calibrator selection")
        platt=v2.fit_calibration(training,active_raw.loc[training.index],[])
        state=v2.fit_calibration(training,active_raw.loc[training.index],names)
        for label in base.CLASSES:
            predictions["identity"].loc[held.index,label]=active_raw.loc[held.index,"raw_p_"+label]
            predictions["platt"].loc[held.index,label]=apply_probability(platt["calibrators"][label],active_raw.loc[held.index],held,label,[])
            predictions["state"].loc[held.index,label]=apply_probability(state["calibrators"][label],active_raw.loc[held.index],held,label,modes["state"])
    full_platt=v2.fit_calibration(data,raw,[])
    choices={}; records=[]
    for label in base.CLASSES:
        reference,reference_auc=object_scores(active,predictions["identity"][label].to_numpy(),label)
        eligible=[]
        for mode in modes:
            brier,auc=object_scores(active,predictions[mode][label].to_numpy(),label)
            preserves_auc=(not np.isfinite(reference_auc)) or (np.isfinite(auc) and auc>=reference_auc-.01)
            harm_fraction=float(((brier-reference)>1e-9).mean())
            accepted=(mode=="identity") or (brier.mean()<=reference.mean()-1e-4 and preserves_auc and harm_fraction<=.5)
            records.append(dict(label=label,mode=mode,object_macro_brier=brier.mean(),object_macro_auc=auc,
                                object_harm_fraction=harm_fraction,eligible=accepted))
            if accepted:
                eligible.append((float(brier.mean()),mode))
        chosen=min(eligible)[1]
        choices[label]=chosen
    models={label:(None if mode=="identity" else (full_platt if mode=="platt" else template)["calibrators"][label])
            for label,mode in choices.items()}
    return dict(models=models,modes=choices,states=modes["state"],corrections=template["corrections"],selection_records=records)


def calibrate(raw,data,calibration):
    result=raw.copy()
    for label,model in calibration["models"].items():
        states=calibration["states"] if calibration["modes"][label]=="state" else []
        result["p_"+label]=apply_probability(model,raw,data,label,states)
    for i,target in enumerate(base.TARGETS):
        guard=data.action.map({a:g[i] for a,g in calibration["corrections"].items()}).to_numpy()
        result[("lcb_" if i<3 else "ucb_")+target]=result["pred_"+target]+(-guard if i<3 else guard)
    return result


def fit_bundle(data,names,plan):
    raw=v2.raw_oof(data,names,plan)
    calibration=fit_calibration(data,raw,names)
    calibrated=calibrate(raw,data,calibration)
    active=data.loc[base.mask_fit(data)].copy()
    active["residual"]=active.adds_regression-calibrated.loc[active.index,"p_adds_regression"]
    biases=[float(np.average(group.residual,weights=v2.object_frame_weights(group))) for _,group in active.groupby("object_id")]
    return dict(raw_model=v2.fit_raw(data,names,plan),calibration=calibration,
                risk_bias_guard=max(0.,float(np.quantile(biases,.8))),feature_names=names,
                training_objects=sorted(map(int,data.object_id.unique())))


def predict_bundle(bundle,data):
    raw=v2.predict_raw(bundle["raw_model"],data)
    # Recompute the chosen neighbor's distance directly: BLAS norm expansion
    # can report different tiny nonzero distances for identical training rows.
    model=bundle["raw_model"]
    X=base.transform(model,data,model["feature_names"])
    nearest=model["neighbors"].kneighbors(X,n_neighbors=1)[1][:,0]
    distance=np.sqrt(np.sum((X-model["neighbors"]._fit_X[nearest])**2,axis=1))
    raw["training_support"]=1/(1+distance/model["domain_radius"])
    result=calibrate(raw,data,bundle["calibration"])
    result["adds_risk_upper"]=np.minimum(1.,result.p_adds_regression+bundle["risk_bias_guard"])
    return result


def calibrated_oof(data,names,plan):
    outputs=[]
    for train,valid in base.folds(data,3):
        outputs.append(predict_bundle(fit_bundle(data.iloc[train],names,plan),data.iloc[valid]))
    return pd.concat(outputs).sort_index().loc[data.index]


def run(args):
    if args.output.exists():
        raise FileExistsError("Preserve prior experiment")
    locked={"status":"LOCKED_BEFORE_SELECTIVE_CALIBRATION_OUTER_EVALUATION", "change":"training-only meta-group selection of identity/Platt/state calibration",
            "macro_brier_improvement_required":.0001,"max_macro_auc_loss":.01,"max_fraction_objects_harmed":.5,
            "raw_model_features_residual_guards_and_policy_grid":"unchanged_from_full_tail_v2",
            "scope":"exploratory_reusing_inspected_22_object_development_no_new_final_validation"}
    args.output.mkdir(parents=True)
    base.dump(args.output/"locked_followup_plan.json",locked)
    plan=json.loads(args.plan.read_text(encoding="utf-8"))
    features=pd.read_csv(args.v2/"observable_features_with_tails.csv")
    labels=pd.read_csv(args.labels/"development_bop_labels.csv")
    data=base.attach_labels(features,labels).reset_index(drop=True)
    v2state=json.loads((args.v2/"v2_training_state.json").read_text(encoding="utf-8"))
    if (base.digest(args.v2/"observable_features_with_tails.csv")!=v2state["artifact_hashes"]["observable_features_with_tails.csv"]
            or base.digest(args.labels/"development_bop_labels.csv")!=v2state["source_hashes"][str(args.labels/"development_bop_labels.csv")]
            or base.digest(args.plan)!=v2state["plan_sha256"]):
        raise ValueError("Frozen follow-up input hash mismatch")
    names=joblib.load(args.v2/"tail_group_boost/development_candidate.joblib")["bundle"]["feature_names"]
    if set(data.object_id)&set(v2state["excluded_inspected_objects"]):
        raise ValueError("Inspected objects entered follow-up")
    state=dict(status="SELECTIVE_CALIBRATION_RUNNING",folds=[],source_hashes={},method_promoted=False,
               new_final_holdout=False,started_at_utc=datetime.now(timezone.utc).isoformat())
    predictions=[]; selections=[]; grids=[]; calchoices=[]
    for fold,(train,valid) in enumerate(base.folds(data,5)):
        training,held=data.iloc[train],data.iloc[valid]
        print(f"SELECTIVE CALIBRATION OUTER {fold+1}/5",flush=True)
        inner=calibrated_oof(training,names,plan)
        policy,grid=v2.tune_policy(training,inner,plan)
        bundle=fit_bundle(training,names,plan)
        p=predict_bundle(bundle,held); p["outer_fold"]=fold; predictions.append(p)
        selected=v2.evaluate_choices(held,v2.decisions(held,p,policy)); selected["outer_fold"]=fold
        selections.append(selected); grids += [dict(row,outer_fold=fold) for row in grid]
        calchoices += [dict(row,outer_fold=fold) for row in bundle["calibration"]["selection_records"]]
        record=dict(fold=fold,training_objects=sorted(map(int,training.object_id.unique())),
                    validation_objects=sorted(map(int,held.object_id.unique())),policy=policy,
                    calibration_modes=bundle["calibration"]["modes"],updates=int(selected.selected_action.ne(base.INITIAL).sum()))
        state["folds"].append(record)
        joblib.dump(dict(bundle=bundle,policy=policy,fold=record),args.output/f"outer_fold_{fold}.joblib")
        base.dump(args.output/"selective_training_state.json",state)
        print("OUTER COMPLETE",record,flush=True)
    oof=pd.concat(predictions).sort_index().loc[data.index]
    selected=pd.concat(selections,ignore_index=True); selected["method"]="tail_selective_calibration"
    selected.to_csv(args.output/"outer_oof_selected.csv",index=False)
    pd.concat([data[["target_id","object_id","cohort","action","pose_changed","features_valid"]],oof],axis=1).to_csv(args.output/"outer_oof_candidate_predictions.csv",index=False)
    pd.DataFrame(grids).to_csv(args.output/"inner_policy_grid.csv",index=False)
    pd.DataFrame(calchoices).to_csv(args.output/"inner_calibrator_selection.csv",index=False)
    disc=v2.discrimination(data,oof,"tail_selective_calibration")
    disc.to_csv(args.output/"cohort_discrimination.csv",index=False)
    comparison=pd.concat([pd.read_csv(args.v2/"all_variant_oof_outcomes.csv"),selected],ignore_index=True)
    comparison.to_csv(args.output/"all_variant_oof_outcomes.csv",index=False)
    summary=base.metric_summary(comparison)
    summary.to_csv(args.output/"selective_oof_comparison.csv",index=False)
    final_policy,final_grid=v2.tune_policy(data,oof,plan)
    final_bundle=fit_bundle(data,names,plan)
    joblib.dump(dict(bundle=final_bundle,policy=final_policy,status="EXPLORATORY_NOT_PROMOTED"),args.output/"development_candidate.joblib")
    pd.DataFrame(final_grid).to_csv(args.output/"final_policy_grid_exploratory.csv",index=False)
    clean=predict_bundle(final_bundle,features)
    first=features.iloc[:1]
    if not np.allclose(clean.iloc[:1],predict_bundle(final_bundle,first),atol=1e-9,rtol=1e-9):
        raise ValueError("Batch-dependent calibrated inference")
    shadow=data.copy()
    for column in base.TARGETS+base.CLASSES+["ADD_S_mm","initial_ADD_S_mm"]+base.AR:
        shadow[column]=99999.
    predicted=predict_bundle(final_bundle,data)
    if not np.allclose(predicted,predict_bundle(final_bundle,shadow),atol=1e-9,rtol=1e-9) or v2.decisions(data,predicted,final_policy)!=v2.decisions(shadow,predicted,final_policy):
        raise ValueError("Outcome-based inference")
    for fold in state["folds"]:
        if set(fold["training_objects"])&set(fold["validation_objects"]):
            raise ValueError("Object leakage")
    snapshot=args.output/"source_snapshot"; snapshot.mkdir()
    for source in [Path(__file__),Path(v2.__file__),Path(base.__file__),args.plan]:
        state["source_hashes"][str(source)]=base.digest(source)
        shutil.copy2(source,snapshot/source.name)
    for source in [args.v2/"v2_training_state.json",args.v2/"observable_features_with_tails.csv",args.labels/"development_bop_labels.csv"]:
        state["source_hashes"][str(source)]=base.digest(source)
    state.update(status="COMPLETE_EXPLORATORY_NOT_PROMOTED",training_objects=sorted(map(int,data.object_id.unique())),
                 final_calibration_modes=final_bundle["calibration"]["modes"],final_policy=final_policy,
                 label_free_prediction=True,batch_invariance=True,gt_shadow=True,
                 completed_at_utc=datetime.now(timezone.utc).isoformat())
    state["artifact_hashes"]={str(p.relative_to(args.output)):base.digest(p) for p in args.output.rglob("*") if p.is_file() and p.name!="selective_training_state.json"}
    base.dump(args.output/"selective_training_state.json",state)
    print(summary[summary.scope.eq("all")|summary.stratum.str.endswith("_mild")].to_string(index=False),flush=True)
    print(disc[disc.cohort.str.endswith("_mild")&disc.label.eq("adds_regression")].to_string(index=False),flush=True)
    print("SELECTIVE CALIBRATION COMPLETE",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    root=Path(r"D:\6D_pose\codex_paper_workspace")
    parser.add_argument("--v2",type=Path,default=root/"outputs/bop_aware_tail_risk_v2_20261019_r1")
    parser.add_argument("--labels",type=Path,default=root/"outputs/bop_aware_development_labels_20261019_r1")
    parser.add_argument("--plan",type=Path,default=Path(v2.__file__).with_name("bop_aware_tail_risk_v2_plan_20261019.json"))
    parser.add_argument("--output",type=Path,default=root/"outputs/bop_aware_selective_calibration_20261019_r1")
    run(parser.parse_args())
