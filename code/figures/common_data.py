import numpy as np, pandas as pd, os
IN=os.environ.get('FIGDATA','/home/claude/newfigs/in')
FIXED='Filtered P2P s1.00'; OURS='A ordinary ranking'
CANDS=['P2P s0.25','P2P s0.50','P2P s1.00','Filtered P2P s0.25','Filtered P2P s0.50','Filtered P2P s1.00']
def load(name):
    d=pd.read_csv(os.path.join(IN,f"per_condition_{name}.csv"))
    d['e0']=d.initial_ADD_S_mm/d.diameter_mm
    for a in CANDS+[OURS]: d['d_'+a]=100*(d[a]-d['Initial'])
    d['d_veto']=d['d_'+FIXED].clip(lower=0)
    d['d_best7']=np.maximum(d[['d_'+a for a in CANDS]].max(axis=1),0)
    return d
EDGES=np.array([0,0.015,0.025,0.04,0.06,0.1,0.2,1.0,np.inf])
def binstats(d,col,edges=EDGES,draws=2000,seed=20261105):
    b=np.digitize(d.e0,edges[1:-1]); objs=d.object_id.unique(); rng=np.random.default_rng(seed)
    g=d.assign(b=b)
    mean=g.groupby('b')[col].mean().reindex(range(len(edges)-1))
    # object-cluster bootstrap of pooled bin means
    per=g.groupby(['object_id','b'])[col].agg(['sum','count']).unstack('b').fillna(0)
    S=per['sum'].reindex(columns=range(len(edges)-1),fill_value=0).to_numpy(); N=per['count'].reindex(columns=range(len(edges)-1),fill_value=0).to_numpy()
    idx=rng.integers(0,len(objs),size=(draws,len(objs)))
    bs=S[idx].sum(1); bn=N[idx].sum(1)
    with np.errstate(invalid='ignore',divide='ignore'): bm=bs/bn
    lo=np.nanquantile(bm,0.025,axis=0); hi=np.nanquantile(bm,0.975,axis=0)
    cnt=g.groupby('b').size().reindex(range(len(edges)-1),fill_value=0)
    return mean.to_numpy(),lo,hi,cnt.to_numpy()
