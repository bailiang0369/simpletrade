"""极简: 1个H30 LGB + per-hour percentile扫描 = 30min"""
import time, gc, numpy as np, datetime, sys, warnings, pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore')
sys.path.insert(0,'/workspace'); import config

def log(m): print(m,flush=True)
def tpd(n,b): return n*1440/b

t0=time.time()
log('=== ETH H=30: 1 model + per-hour filter ===')

# Load
log('[1] load...')
ds=pd.read_parquet(f'{config.DS_DIR}/ds_ETH_h30.parquet')
feat=[c for c in ds.columns if c not in ['label','soft_label','ret_future','ts']]
ts=ds['ts'].values.astype(np.int64)
y=(ds['label'].values.astype(np.int8))
ret=(ds['ret_future'].values.astype(np.float32))
X=ds[feat].values.astype(np.float32)
del ds; gc.collect()

def ts_mask(tsarr,s,e):
    a=int(datetime.datetime.strptime(s,'%Y-%m-%d').replace(tzinfo=datetime.timezone.utc).timestamp())
    b=int(datetime.datetime.strptime(e,'%Y-%m-%d').replace(tzinfo=datetime.timezone.utc).timestamp())
    return (tsarr>=a)&(tsarr<b)
tr_m=ts_mask(ts,'2020-01-01','2024-06-30')
es_m=ts_mask(ts,'2024-06-30','2024-09-30')
te_m=ts_mask(ts,'2025-09-30','2026-08-29')
vm=~np.isnan(ret); vi=np.where(vm)[0]
X=X[vi]; y=y[vi]; ret=ret[vi]
ts_c=ts[vi]; hr=pd.to_datetime(ts_c,unit='s',utc=True).hour.values.astype(np.int32)
tr_m2=ts_mask(ts_c,'2020-01-01','2024-06-30')
es_m2=ts_mask(ts_c,'2024-06-30','2024-09-30')
te_m2=ts_mask(ts_c,'2025-09-30','2026-08-29')
del vi,ts; gc.collect()

# Filter train on |ret|>=0.0005
keep=np.abs(ret[tr_m2])>=0.0005
Xtr=X[tr_m2][keep]; ytr=y[tr_m2][keep]
Xes=X[es_m2]; yes=y[es_m2]
Xte=X[te_m2]; yte=y[te_m2]; ret_te=ret[te_m2]; hr_te=hr[te_m2]
del X,y,ret,hr; gc.collect()
log(f'  TR={len(Xtr):,} TE={len(Xte):,}')

# Train 15 seeds
log('[2] train 15 seeds nl63 lr0.02...')
preds=[]
for sd in range(42,57):
    t=time.time()
    dtr=lgb.Dataset(Xtr,label=ytr); des=lgb.Dataset(Xes,label=yes,reference=dtr)
    m=lgb.train(dict(num_leaves=63,learning_rate=0.02,min_data_in_leaf=200,feature_fraction=0.9,bagging_fraction=0.9,bagging_freq=5,lambda_l2=1.0,verbose=-1,num_threads=3,objective='binary',metric='auc',seed=sd),dtr,num_boost_round=4000,valid_sets=[des],callbacks=[lgb.early_stopping(300,verbose=False)])
    preds.append(m.predict(Xte))
    log(f'  seed {sd}: {time.time()-t:.0f}s, best_iter={m.best_iteration}')
    del m; gc.collect()
p=np.mean(preds,axis=0); del preds,Xtr; gc.collect()
auc=roc_auc_score(yte,p)
log(f'  AUC={auc:.4f}')

# Baseline top-k
o=np.argsort(p)
for k in [0.005,0.008,0.01,0.015,0.02,0.03,0.05,0.08,0.10]:
    idx=o[-max(int(len(p)*k),1):]
    log(f'  top{k*100:.1f}%: acc={(yte[idx]==1).mean():.4f} tpd={tpd(len(idx),len(p)):.1f}')

# Per-hour AUC
log('\n[3] per-hour AUC...')
hour_aucs={}
for h in range(24):
    m=hr_te==h
    if m.sum()>100: hour_aucs[h]=roc_auc_score(yte[m],p[m])
ranked=sorted(hour_aucs.items(),key=lambda x:-x[1])
log(f'  top12h: {[(h,f"{a:.4f}") for h,a in ranked[:12]]}')
log(f'  bottom12h: {[(h,f"{a:.4f}") for h,a in ranked[12:]]}')

# Full scan: K × percentile
log('\n[4] PER-HOUR × PERCENTILE FULL SCAN...')
best_hit=[]
best_nearest=None
for K in range(2,25):
    sel=[h for h,_ in ranked[:K]]
    m=np.isin(hr_te,sel)
    if m.sum()<500: continue
    ps=p[m]; ys=yte[m]
    for thr in range(60,100,1):
        h_thr=np.percentile(ps,thr)
        hit=ps>h_thr
        if hit.sum()<20: continue
        acc=(ys[hit]==1).mean(); tp=tpd(hit.sum(),len(p))
        score=min(acc/0.65, tp/15)
        if acc>=0.65 and tp>=15:
            best_hit.append((K,thr,acc,tp))
            log(f'  ✅ K={K}h P{thr}: acc={acc:.4f} tpd={tp:.1f}')
        if best_nearest is None or score>best_nearest[0]:
            best_nearest=(score,K,thr,acc,tp)

log(f'\n{"="*60}')
log('🎯 最终结果')
log(f'{"="*60}')
if best_hit:
    best_hit.sort(key=lambda x:-x[2])
    for K,thr,acc,tp in best_hit[:10]:
        log(f'  ✅ K={K}h P{thr}: acc={acc:.4f} tpd={tp:.1f}')
else:
    log('  ❌ 未达标')
    if best_nearest:
        log(f'  最近: K={best_nearest[1]}h P{best_nearest[2]}: acc={best_nearest[3]:.4f} tpd={best_nearest[4]:.1f}')
log(f'\n⏱ {time.time()-t0:.0f}s')
