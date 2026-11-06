import numpy as np, pandas as pd, json, warnings
warnings.filterwarnings('ignore')
exec(open('stage2.py').read())
ag=pd.read_csv('/mnt/user-data/uploads/codex_paper_workspace/outputs/redesign_feasibility_20261105/dev_p2p_filtered_agreement.csv')
AG=['agree_rot_deg','agree_trans_mm','agree_norm','p2p_rot_deg','p2p_trans_mm','filt_rot_deg','filt_trans_mm','p2p_rmse','filt_rmse']
df=df.merge(ag[['cohort','frame_id','input_depth_condition']+AG],on=['cohort','frame_id','input_depth_condition'],how='left',validate='many_to_one')
names=json.load(open('/mnt/user-data/uploads/codex_paper_workspace/outputs/decoupled_development_20261020_r4/feature_schema.json'))['feature_names']
Z=STATE_NAMES+PREDS+PROBS
df['res_u']=df.predA-df.true_u
ens=np.load('ens_members.npy'); m=ens.mean(0); sd=ens.std(0); df['ens_m']=m; df['ens_sd']=sd; df['res_e']=m-df.true_u
V={}
V['A: uncorrected predicted utility (stored OOF)']=df.predA.values
V['D: SARef corrected bound (stored OOF)']=df.lcb_delta_joint_utility.values
V['R0: q75 width, 26 state + preds']=df.predA.values-np.maximum(0,fit_pred(Z,'res_u','quantile',.75))
V['R1: signed mean-bias correction, 26 state + preds']=df.predA.values-fit_pred(Z,'res_u')
V['R2: median correction']=df.predA.values-fit_pred(Z,'res_u','quantile',.5)
V['R3: stacked model on state + preds']=fit_pred(Z,'true_u')
V['R4: mean-bias correction, all 191 features']=df.predA.values-fit_pred(names+PREDS+PROBS,'res_u')
V['R5: object-bootstrap ensemble mean (8 models)']=m
V['R6: ensemble mean - 1 sd']=m-sd
V['R7: ensemble mean - q75(state, sd)']=m-np.maximum(0,fit_pred(STATE_NAMES+['ens_m','ens_sd'],'res_e','quantile',.75))
V['R8: mean-bias correction with P2P/filtered agreement']=df.predA.values-fit_pred(Z+AG,'res_u')
V['R9: q75 width with P2P/filtered agreement']=df.predA.values-np.maximum(0,fit_pred(Z+AG,'res_u','quantile',.75))
o=df.true_u.values
V['Oracle: true utility']=df.true_mean.values
res_obj=df.groupby('object_id').res_u.transform(lambda s: s[valid[s.index]].mean())
V['Oracle: per-object bias removed']=df.predA.values-res_obj.values
covs=np.linspace(0.05,1,20)
targets=df.drop_duplicates('target_id')[['target_id','object_id']].reset_index(drop=True)
objs=np.sort(targets.object_id.unique()); n=len(targets)
rng=np.random.default_rng(20261104); B=2000
draws=rng.integers(0,len(objs),(B,len(objs)))
def per_object_curve(s):
    d=df.assign(_s=np.where(valid,s,-np.inf))
    best=d.loc[d.groupby('target_id')._s.idxmax()]
    t=targets.merge(best[['target_id','_s','true_mean','true_add']],on='target_id',how='left')
    order=np.argsort(-t._s.values)
    G=np.zeros((len(covs),len(objs))); R=[]; Dd=[]
    for j,c in enumerate(covs):
        k=int(round(c*n)); acc=np.zeros(n,bool); acc[order[:k]]=True; acc&=np.isfinite(t._s.values)
        g=np.where(acc,t.true_mean.values,0.)
        G[j]=pd.Series(g).groupby(t.object_id.values).mean().reindex(objs).values*100
        R.append((t.true_add.values[acc]>1e-9).mean()); Dd.append((t.true_mean.values[acc]<0).mean())
    return G,np.array(R),np.array(Dd)
GA,_,_=per_object_curve(V['A: uncorrected predicted utility (stored OOF)'])
rows=[]
for k,s in V.items():
    G,R,Dd=per_object_curve(s)
    auc=G.mean(1).mean(); diff=(G-GA).mean(0)  # per-object AUC difference
    bs=diff[draws].mean(1)
    rows.append(dict(variant=k,AUC_gain_pp=auc,max_gain_pp=G.mean(1).max(),gain_at_50pct=G.mean(1)[9],ARdrop_at_50pct=Dd[9],ADDreg_at_50pct=R[9],
                     AUC_minus_A=diff.mean(),ci_lo=np.quantile(bs,.025),ci_hi=np.quantile(bs,.975),objects_better_than_A=int((diff>1e-12).sum())))
out=pd.DataFrame(rows); pd.set_option('display.width',250)
print(out.round(3).to_string(index=False))
out.to_csv('dev_feasibility_summary.csv',index=False)
