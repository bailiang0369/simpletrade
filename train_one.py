import time, gc, datetime, numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import sys, os; sys.path.insert(0,'/workspace'); import config

def ts_mask(ts,s,e):
    a=int(datetime.datetime.strptime(s,'%Y-%m-%d').replace(tzinfo=datetime.timezone.utc).timestamp())
    b=int(datetime.datetime.strptime(e,'%Y-%m-%d').replace(tzinfo=datetime.timezone.utc).timestamp())
    return (ts>=a)&(ts<b)
def tpd(n,b): return n*1440/b

SYM=sys.argv[1]; H=int(sys.argv[2])
OUT=f'/workspace/models/preds_{SYM}_h{H}.npz'

df=pd.read_parquet(f'{config.DS_DIR}/ds_{SYM}_h{H}.parquet')
FEAT=[c for c in df.columns if c not in ('label','soft_label','ret_future','ts')]
ts=df['ts'].values.astype(np.int64); y=df['label'].values.astype(np.int8); ret=df['ret_future'].values.astype(np.float32)
abs_ret=np.abs(ret)
X=df[FEAT].values.astype(np.float32)
tr_m=ts_mask(ts,'2020-01-01','2024-06-30'); es_m=ts_mask(ts,'2024-06-30','2024-09-30'); te_m=ts_mask(ts,'2025-09-30','2026-08-29')
keep_tr=abs_ret[tr_m]>=0.0005
Xtr=X[tr_m][keep_tr]; ytr=y[tr_m][keep_tr]; Xes=X[es_m]; yes=y[es_m]
Xte=X[te_m]; yte=y[te_m]; ret_te=ret[te_m]; total=len(yte)
print(f'{SYM} H={H}: TR={len(Xtr):,} TE={total:,}',flush=True)
del df,X,abs_ret; gc.collect()

SEEDS=12; LP=dict(num_leaves=63,learning_rate=0.02,min_data_in_leaf=200,feature_fraction=0.9,bagging_fraction=0.9,bagging_freq=5,lambda_l2=1.0,verbose=-1,num_threads=3,objective='binary',metric='auc')
t0=time.time(); preds=[]
for sd in range(42,42+SEEDS):
    dtr=lgb.Dataset(Xtr,label=ytr); des=lgb.Dataset(Xes,label=yes,reference=dtr)
    m=lgb.train({**LP,'seed':sd},dtr,num_boost_round=2000,valid_sets=[des],callbacks=[lgb.early_stopping(200,verbose=False)])
    preds.append(m.predict(Xte)); del m; gc.collect()
avg=np.mean(preds,axis=0); del preds,Xtr,Xes,Xte,ytr,yes; gc.collect()
auc=roc_auc_score(yte,avg); o=np.argsort(avg)
print(f'  AUC={auc:.4f} ({time.time()-t0:.0f}s)',flush=True)
for k in [0.005,0.008,0.01,0.015,0.02,0.03,0.05]:
    idx=o[-max(int(len(avg)*k),1):]; acc=(yte[idx]==1).mean(); tp=tpd(len(idx),total); r=ret_te[idx].mean()*10000
    flag='✅BOTH' if (acc>=0.65 and tp>=15) else ('✅ACC' if acc>=0.65 else ('✅TPD' if tp>=15 else ''))
    print(f'  top{k*100:.1f}%: acc={acc:.4f} tpd={tp:.1f} ret={r:.1f} {flag}',flush=True)
np.savez(OUT, pred=avg, y=yte, ret=ret_te, sym=SYM, h=H, auc=auc)
print(f'  Saved: {OUT}',flush=True)
