import numpy as np, pandas as pd, json, warnings, sys
warnings.filterwarnings('ignore')
from sklearn.ensemble import HistGradientBoostingRegressor
from rc import curve
df=pd.read_pickle('dev.pkl').copy()
df['true_u']=df.true_mean-0.005*df.true_add/5
names=json.load(open('/mnt/user-data/uploads/codex_paper_workspace/outputs/decoupled_development_20261020_r4/feature_schema.json'))['feature_names']
valid=((df.features_valid==1)&(df.pose_changed==1)&(df.observation_support>=.01)&(df.training_support>=.2)).values
fitmask=((df.features_valid==1)&(df.pose_changed==1)).values
ntar=df.groupby('object_id').target_id.transform('nunique'); w=(1./ntar).values
X=df[names].astype(float).replace([np.inf,-np.inf],np.nan)
P=dict(max_iter=60,max_leaf_nodes=7,min_samples_leaf=25,learning_rate=0.07,l2_regularization=5.0,random_state=0)
def prep(tr_idx, te_idx):
    Xtr=X.iloc[tr_idx]; med=Xtr.median(); keep=[c for c in names if Xtr[c].nunique()>1]
    return Xtr[keep].fillna(med[keep]).values, X.iloc[te_idx][keep].fillna(med[keep]).values
K=int(sys.argv[1]) if len(sys.argv)>1 else 8
rng=np.random.default_rng(20261105)
single=np.full(len(df),np.nan); ens=np.full((K,len(df)),np.nan)
for f in sorted(df.outer_fold.unique()):
    tr=np.where((df.outer_fold!=f).values & fitmask)[0]; te=np.where((df.outer_fold==f).values)[0]
    Xtr,Xte=prep(tr,te)
    single[te]=HistGradientBoostingRegressor(**P).fit(Xtr,df.true_u.values[tr],sample_weight=w[tr]).predict(Xte)
    objs=df.object_id.values[tr]; uo=np.unique(objs)
    for k in range(K):
        bo=rng.choice(uo,len(uo),replace=True)
        cnt=pd.Series(bo).value_counts()
        mult=pd.Series(objs).map(cnt).fillna(0).values
        sel=mult>0
        ens[k,te]=HistGradientBoostingRegressor(**{**P,'random_state':k}).fit(Xtr[sel],df.true_u.values[tr][sel],sample_weight=(w[tr]*mult)[sel]).predict(Xte)
    print('fold',f,flush=True)
np.save('ens_single.npy',single); np.save('ens_members.npy',ens)
