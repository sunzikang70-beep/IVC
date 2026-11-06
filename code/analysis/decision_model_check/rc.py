import numpy as np, pandas as pd
def per_target_pick(df, score, valid):
    d=df.assign(_s=np.where(valid, score, -np.inf))
    idx=d.groupby('target_id')['_s'].idxmax()
    best=d.loc[idx]
    best=best[np.isfinite(best._s)]
    return best
def curve(df, score, valid, covs=np.linspace(0.05,1,20), true_col='true_mean', reg_col='true_add'):
    best=per_target_pick(df, score, valid)
    targets=df.drop_duplicates('target_id')[['target_id','object_id']]
    n=len(targets)
    order=best.sort_values('_s',ascending=False)
    out=[]
    for c in covs:
        k=int(round(c*n)); acc=order.head(k)
        t=targets.merge(acc[['target_id',true_col,reg_col]],on='target_id',how='left').fillna({true_col:0.})
        gain=t.groupby('object_id')[true_col].mean().mean()*100
        reg=(acc[reg_col]>1e-9).mean() if len(acc) else np.nan
        drop=(acc[true_col]<0).mean() if len(acc) else np.nan
        out.append(dict(cov=k/n,gain=gain,reg=reg,ardrop=drop,thr=acc._s.min() if len(acc) else np.nan))
    return pd.DataFrame(out)
def at_threshold(df, score, valid, thr, true_col='true_mean', reg_col='true_add'):
    best=per_target_pick(df, score, valid)
    acc=best[best._s>thr]
    targets=df.drop_duplicates('target_id')[['target_id','object_id']]
    t=targets.merge(acc[['target_id',true_col,reg_col]],on='target_id',how='left').fillna({true_col:0.})
    return dict(cov=len(acc)/len(targets), gain=t.groupby('object_id')[true_col].mean().mean()*100,
                reg=(acc[reg_col]>1e-9).mean() if len(acc) else np.nan, ardrop=(acc[true_col]<0).mean() if len(acc) else np.nan)
