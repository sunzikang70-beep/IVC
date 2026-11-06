import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from load import STATE_NAMES
from rc import curve
df=pd.read_pickle('dev.pkl').copy()
df['true_u']=df.true_mean-0.005*df.true_add/5
PREDS=['pred_delta_AR_VSD','pred_delta_AR_MSSD','pred_delta_AR_MSPD','pred_delta_ADD_S_mm']
PROBS=['raw_p_adds_regression','raw_p_vsd_regression','raw_p_mssd_regression','raw_p_mspd_regression','raw_p_mean_ar_gain']
valid=((df.features_valid==1)&(df.pose_changed==1)&(df.observation_support>=.01)&(df.training_support>=.2)).values
ntar=df.groupby('object_id').target_id.transform('nunique')
w=(1./ntar).values
def fit_pred(Xcols, ycol, loss='squared_error', q=None, params=None, mask_train=None):
    p=dict(max_iter=60,max_leaf_nodes=5,min_samples_leaf=40,learning_rate=0.06,l2_regularization=5.0,max_bins=64,random_state=0)
    if params: p.update(params)
    out=np.full(len(df),np.nan)
    for f in sorted(df.outer_fold.unique()):
        tr=(df.outer_fold!=f).values & valid
        te=(df.outer_fold==f).values
        kw=dict(loss=loss); 
        if q is not None: kw['quantile']=q
        Xtr=df.loc[tr,Xcols].astype(float).replace([np.inf,-np.inf],np.nan); med_=Xtr.median()
        keep=[c for c in Xcols if Xtr[c].nunique()>1]
        Xtr=Xtr[keep].fillna(med_[keep]); Xte=df.loc[te,keep].astype(float).replace([np.inf,-np.inf],np.nan).fillna(med_[keep])
        m=HistGradientBoostingRegressor(**kw,**p).fit(Xtr.values, df.loc[tr,ycol].values, sample_weight=w[tr])
        out[te]=m.predict(Xte.values)
    return out
