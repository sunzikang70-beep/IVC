import numpy as np, pandas as pd, json, warnings, time, sys
warnings.filterwarnings('ignore')
from sklearn.ensemble import HistGradientBoostingRegressor as HR, HistGradientBoostingClassifier as HC
from rc import curve
df=pd.read_pickle('dev.pkl').copy()
names=json.load(open('/mnt/user-data/uploads/codex_paper_workspace/outputs/decoupled_development_20261020_r4/feature_schema.json'))['feature_names']
AR=['delta_AR_VSD','delta_AR_MSSD','delta_AR_MSPD']; T4=AR+['delta_ADD_S_mm']
df['true_u']=df.true_mean-0.005*df.delta_ADD_S_mm/5
valid=((df.features_valid==1)&(df.pose_changed==1)&(df.observation_support>=.01)).values
fitm=((df.features_valid==1)&(df.pose_changed==1)).values
P=dict(max_iter=60,max_leaf_nodes=7,min_samples_leaf=25,learning_rate=0.07,l2_regularization=5.0,random_state=0)
def X_of(cols):
    X=df[cols].astype(float).replace([np.inf,-np.inf],np.nan)
    return X
def oof_reg(cols, targets=T4, params=P, utility=True):
    X=X_of(cols); out={t:np.full(len(df),np.nan) for t in targets}
    for f in range(5):
        tr=np.where((df.outer_fold!=f).values&fitm)[0]; te=np.where((df.outer_fold==f).values)[0]
        med=X.iloc[tr].median(); keep=[c for c in cols if X.iloc[tr][c].nunique()>1]
        Xtr=X.iloc[tr][keep].fillna(med[keep]).values; Xte=X.iloc[te][keep].fillna(med[keep]).values
        for t in targets:
            out[t][te]=HR(**params).fit(Xtr,df[t].values[tr]).predict(Xte)
    if utility: return np.mean([out[t] for t in AR],axis=0)-0.005*out['delta_ADD_S_mm']/5
    return out
def report(name,s):
    c=curve(df,s,valid); print(f'{name:45s} AUC {c.gain.mean():.3f} max {c.gain.max():.2f} @0.5 {c.gain.iloc[9]:.2f}/{c.ardrop.iloc[9]:.3f} @0.7 {c.gain.iloc[13]:.2f}/{c.ardrop.iloc[13]:.3f}',flush=True)
    return c
