import pandas as pd, numpy as np
g=pd.read_csv('candidate_gates.csv.gz')
gates=[c for c in g.columns if c.startswith('gate_')]
g['allowed']=g[gates].all(axis=1)
g['pred_mean']=g[['p_pred_delta_AR_VSD','p_pred_delta_AR_MSSD','p_pred_delta_AR_MSPD']].mean(axis=1)
F='protocolF_se3_trust_filtered_s100'
f=g[g.action.eq(F)].set_index('target_id')
meta=f[['object_id','init_regime','input_depth_condition']]
def objeq(delta):  # delta indexed by target_id, pp
    d=meta.join(delta.rename('d'))
    return 100*d.groupby('object_id').d.mean().mean()
def obj(delta):
    d=meta.join(delta.rename('d')); return 100*d.groupby('object_id').d.mean()
mv=g[g.gate_moving_valid]
# D choice (all gates; max lcb joint utility)
al=g[g.allowed].sort_values(['target_id','joint_utility'],ascending=[True,False]).drop_duplicates('target_id').set_index('target_id')
dD=pd.Series(0.0,index=meta.index); dD.loc[al.index]=al.true_delta_AR_mean
print('D', round(objeq(dD),3), 'updates', round(len(al)/len(meta)*100,1))
print('F', round(objeq(f.true_delta_AR_mean),3))
# ungated argmax of utility / pred mean among moving candidates, full coverage
for score in ['p_lcb_delta_joint_utility','pred_mean','p_lcb_delta_AR_mean']:
    b=mv.sort_values(['target_id',score],ascending=[True,False]).drop_duplicates('target_id').set_index('target_id')
    d=pd.Series(0.0,index=meta.index); d.loc[b.index]=b.true_delta_AR_mean
    print('argmax',score,'full coverage', round(objeq(d),3), 'regress%', round(100*(b.true_delta_ADD_S_mm>1e-9).mean(),1))
    # risk-coverage: rank targets by score, accept top-k
    order=b.sort_values(score,ascending=False)
    for cov in [0.2,0.4,0.6,0.64,0.8,0.9,1.0]:
        k=int(round(cov*len(meta))); sel=order.index[:k]
        d=pd.Series(0.0,index=meta.index); d.loc[sel]=order.loc[sel,'true_delta_AR_mean']
        print(f'   cov {cov:.2f}: gain {objeq(d):.3f}  accepted-regress {100*(order.loc[sel,"true_delta_ADD_S_mm"]>1e-9).mean():.1f}%')
# fixed F with ranking by its own predicted utility (accept F only when predicted positive)
fx=f.sort_values('p_lcb_delta_joint_utility',ascending=False)
# per-object D-F
po=(obj(dD)-obj(f.true_delta_AR_mean)).round(2); print('D-F per object', po.to_dict(), 'D>F objects', int((po>0).sum()))
# F regime breakdown
for r in ['native','mild','hard']:
    m=meta.init_regime.eq(r); 
    dd=f.true_delta_AR_mean[m]; mm=meta[m]
    print(r,'F-init', round(100*mm.join(dd.rename('d')).groupby('object_id').d.mean().mean(),3),
          'F regress%', round(100*(f.true_delta_ADD_S_mm[m]>1e-9).mean(),1))
# Which actions does D choose
print(al.action.value_counts(normalize=True).round(3).to_dict())
