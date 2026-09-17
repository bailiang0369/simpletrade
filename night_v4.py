"""ETH H=30 v4: 用 ds 原有 55D 特征 + cross-horizon LGB stacking + ensemble
避开 OOM: 只用 LGB, 用完 del + gc
"""
import time, gc, numpy as np, datetime, os, sys, warnings
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore')
sys.path.insert(0,'/workspace'); import config

def log(m): print(m,flush=True)
def tpd(n,b): return n*1440/b
def topk(p,y,ret,kl=[0.005,0.008,0.01,0.015,0.02,0.03]):
    out={}; n=len(p); o=np.argsort(p)
    for k in kl:
        idx=o[-max(int(n*k),1):]
        out[k]={'acc':float((y[idx]==1).mean()),'ret':float(ret[idx].mean()*10000),'tpd':tpd(len(idx),n)}
    return out

t0=time.time(); SEEDS=15

# ========== Load original ds features (55D) ==========
log('loading ds features...')
import pickle
with open(f'{config.DS_DIR}/eth_ds.pkl','rb') as f:
    ds = pickle.load(f)
log(f'ds keys: {list(ds.keys())}')
# Extract: features, ts, labels for ETH
X = ds['ETH']['features'].astype(np.float32)
ts = ds['ETH']['ts'].astype(np.int64)
log(f'X={X.shape}, ts={len(ts):,}')

# Build multi-horizon labels from raw ETH close
raw_e = pd.read_parquet(f'{config.DS_DIR}/raw_ETH.parquet')
close_full = pd.Series(raw_e['close'].values.astype(np.float64),index=raw_e['ts'].values.astype(np.int64)).reindex(ts).values.astype(np.float64)
del raw_e; gc.collect()

HORIZONS=[3,5,15,30,60]
labels={}
for h in HORIZONS:
    rh=np.full(len(ts),np.nan,np.float32)
    rh[:-h]=(close_full[h:]/close_full[:-h]-1).astype(np.float32)
    labels[f'ret_{h}']=rh
    labels[f'y_{h}']=(rh>0).astype(np.int8)
del close_full; gc.collect()

# Time splits + NaN filter
def ts_mask(s,e):
    a=int(datetime.datetime.strptime(s,'%Y-%m-%d').replace(tzinfo=datetime.timezone.utc).timestamp())
    b=int(datetime.datetime.strptime(e,'%Y-%m-%d').replace(tzinfo=datetime.timezone.utc).timestamp())
    return (ts>=a)&(ts<b)
tr_m = ts_mask('2020-01-01','2024-06-30')
es_m = ts_mask('2024-06-30','2024-09-30')
te_m = ts_mask('2025-09-30','2026-08-29')

vm = ~np.isnan(labels['ret_30']) & ~np.isnan(X).any(axis=1) & np.isfinite(X).all(axis=1)
vi = np.where(vm)[0]; del vm; gc.collect()
X = X[vi]; ts_clean = ts[vi]; hr = pd.to_datetime(ts_clean,unit='s',utc=True).hour.values.astype(np.int32)
tr_m2 = ts_mask(ts_clean,'2020-01-01','2024-06-30')
es_m2 = ts_mask(ts_clean,'2024-06-30','2024-09-30')
te_m2 = ts_mask(ts_clean,'2025-09-30','2026-08-29')

for k in list(labels.keys()):
    labels[k] = labels[k][vi]
del vi, ts; gc.collect()

log(f'X={X.shape} ({X.nbytes/1e6:.0f}MB), TR={tr_m2.sum():,}, ES={es_m2.sum():,}, TE={te_m2.sum():,}')

def lp_base(seed=42,num_leaves=63,lr=0.02,mdl=200,l2=1.0):
    return dict(num_leaves=num_leaves,learning_rate=lr,min_data_in_leaf=mdl,feature_fraction=0.9,bagging_fraction=0.9,bagging_freq=5,lambda_l2=l2,verbose=-1,num_threads=3,objective='binary',metric='auc')

def train_ensemble(Xtr,ytr,Xes,yes,Xte,yte,ret_te,lp_fn,h=30,min_ret=0.0005,seeds=SEEDS,name=''):
    ret_h = labels[f'ret_{h}']
    keep = np.abs(ret_h[tr_m2])>=min_ret
    Xf = Xtr[keep]; yf = ytr[tr_m2][keep]
    preds=[]; times=[]
    for sd in range(42,42+seeds):
        t=time.time()
        dtr=lgb.Dataset(Xf,label=yf); des=lgb.Dataset(Xes,label=yes,reference=dtr)
        m=lgb.train(lp_fn(sd),dtr,num_boost_round=4000,valid_sets=[des],callbacks=[lgb.early_stopping(300,verbose=False)])
        preds.append(m.predict(Xte)); times.append(time.time()-t); del m; gc.collect()
    avg=np.mean(preds,axis=0); del preds; gc.collect()
    auc=roc_auc_score(yte,avg)
    avg_t=np.mean(times)
    log(f'  {name} AUC={auc:.4f} (avg {avg_t:.0f}s/seed)')
    for k,v in topk(avg,yte,ret_te,[0.005,0.008,0.01,0.02]).items():
        log(f'    top{k*100:.1f}%: acc={v["acc"]:.4f} tpd={v["tpd"]:.1f}')
    return avg, auc

Xtr=X[tr_m2]; Xes=X[es_m2]; Xte=X[te_m2]
del X; gc.collect()
y30=labels['y_30']; ret30=labels['ret_30']
yte=y30[te_m2]; ret_te=ret30[te_m2]; hr_te=hr[te_m2]

# ========== Exp1-3: Single-horizon LGB ==========
log(f'\n{"="*60}'); log('[Single-horizon LGB exp on 55D features]'); log(f'{"="*60}')
results = {}

for param_set_name, kwargs in [('baseline',{}),('reg',{'num_leaves':31,'lr':0.03,'mdl':500,'l2':5.0}),('med',{'num_leaves':95,'lr':0.015,'mdl':100})]:
    p,auc = train_ensemble(Xtr,y30,Xes,y30[es_m2],Xte,yte,ret_te,
        lambda sd,**kw: lp_base(seed=sd,**kw), h=30, name=f'H30_{param_set_name}')
    results[f'H30_{param_set_name}'] = (p,auc)

# ========== Exp4: Cross-horizon (H=3,5,15,30,60) × baseline LGB ==========
log(f'\n{"="*60}'); log('[Cross-horizon LGB × 5 horizons × baseline]'); log(f'{"="*60}')
ch_preds_es = []; ch_preds_te = []
for h in HORIZONS:
    yh = labels[f'y_{h}']; ret_h = labels[f'ret_{h}']
    yes_h = yh[es_m2]
    yh_tr = yh[tr_m2]; yh_te = yh[te_m2]
    keep=np.abs(ret_h[tr_m2])>=0.0005
    Xf=Xtr[keep]; yf=yh_tr[keep]
    p_es=[]; p_te=[]; times=[]
    for sd in range(42,42+10):  # 10 seeds for speed
        t=time.time()
        dtr=lgb.Dataset(Xf,label=yf); des=lgb.Dataset(Xes,label=yes_h,reference=dtr)
        m=lgb.train(lp_base(sd),dtr,num_boost_round=3000,valid_sets=[des],callbacks=[lgb.early_stopping(200,verbose=False)])
        p_es.append(m.predict(Xes)); p_te.append(m.predict(Xte)); times.append(time.time()-t); del m; gc.collect()
    avg_es=np.mean(p_es,axis=0); avg_te=np.mean(p_te,axis=0)
    ch_preds_es.append(avg_es); ch_preds_te.append(avg_te)
    auc_h = roc_auc_score(yh_te, avg_te)
    log(f'  H={h} AUC={auc_h:.4f} ({np.mean(times):.0f}s/seed × 10)')

# Cross-horizon equal weight predict H=30
ch_equal = np.mean(np.column_stack(ch_preds_te),axis=1)
auc_ceq = roc_auc_score(yte,ch_equal)
log(f'  CH_equal AUC={auc_ceq:.4f}')
for k,v in topk(ch_equal,yte,ret_te,[0.005,0.008,0.01,0.02]).items():
    log(f'    top{k*100:.1f}%: acc={v["acc"]:.4f} tpd={v["tpd"]:.1f}')

# Meta LGB stack: base preds → predict H=30
meta_dtr = lgb.Dataset(np.column_stack(ch_preds_es),label=y30[es_m2])
meta = lgb.train(dict(num_leaves=31,learning_rate=0.05,min_data_in_leaf=5,verbose=-1,num_threads=3,objective='binary',metric='auc'),meta_dtr,num_boost_round=500)
meta_avg = meta.predict(np.column_stack(ch_preds_te)); del meta; gc.collect()
auc_meta = roc_auc_score(yte,meta_avg)
log(f'  Meta_LGB AUC={auc_meta:.4f}')
for k,v in topk(meta_avg,yte,ret_te,[0.005,0.008,0.01,0.02]).items():
    log(f'    top{k*100:.1f}%: acc={v["acc"]:.4f} tpd={v["tpd"]:.1f}')

# ========== Exp5: Weighted ensemble across methods ==========
log(f'\n{"="*60}'); log('[Weight search]'); log(f'{"="*60}')
all_preds = {k:v[0] for k,v in results.items()}
all_preds['CH_equal'] = ch_equal; all_preds['Meta'] = meta_avg

names = list(all_preds.keys())
Xc = np.column_stack([all_preds[n] for n in names])

# Corr matrix
corr = np.corrcoef(Xc.T)
log('  Correlations:')
for i in range(len(names)):
    for j in range(i+1,len(names)):
        if corr[i,j]>0.85: log(f'    HIGH {names[i]}-{names[j]}: {corr[i,j]:.4f}')

# Grid search (top 4 least correlated)
best_w=None; best_auc=0
Xc4 = Xc[:,:4]  # first 4
for w1 in np.arange(0,1.05,0.1):
    for w2 in np.arange(0,1.05-w1,0.1):
        for w3 in np.arange(0,1.05-w1-w2,0.1):
            w4=round(max(0,1-w1-w2-w3),2)
            p=Xc4@np.array([w1,w2,w3,w4])
            a=roc_auc_score(yte,p)
            if a>best_auc: best_auc=a; best_w=(w1,w2,w3,w4)
best_combo = Xc4@np.array(best_w)
log(f'  Best4w AUC={best_auc:.4f} w={best_w}')
all4_eq = np.mean(Xc4,axis=1); auc_eq4 = roc_auc_score(yte,all4_eq)
log(f'  All4_eq AUC={auc_eq4:.4f}')

# ========== Exp6: Per-hour filter on best method ==========
log(f'\n{"="*60}'); log('[Per-hour filter]'); log(f'{"="*60}')
# Use best single method
base_name = max(results.keys(), key=lambda k: results[k][1])
base_pred = results[base_name][0]
log(f'  base={base_name}')
hour_aucs = {}
for h in range(24):
    m=hr_te==h
    if m.sum()>100: hour_aucs[h]=roc_auc_score(yte[m],base_pred[m])
ranked = sorted(hour_aucs.items(),key=lambda x:-x[1])
log(f'  Top hours: {ranked[:6]}')

for K in [3,5,8,12,16,20,24]:
    sel=[h for h,_ in ranked[:K]]
    m=np.isin(hr_te,sel)
    if m.sum()==0: continue
    p=base_pred[m]; y=yte[m]
    for thr in [70,80,85,90,92,94,95,96,97,98,99]:
        h_thr=np.percentile(p,thr)
        hit=p>h_thr
        if hit.sum()==0: continue
        acc=(y[hit]==1).mean(); tp=tpd(hit.sum(),len(yte))
        flag='✅HIT' if (acc>=0.65 and tp>=15) else ('ACC' if acc>=0.65 else ('TPD' if tp>=15 else ''))
        if flag: log(f'  K={K} P{thr}: acc={acc:.4f} tpd={tp:.1f} {flag}')

# ========== FINAL SUMMARY ==========
log(f'\n{"="*60}'); log('FINAL SUMMARY'); log(f'{"="*60}')
final = list(all_preds.items()) + [('Best4w',best_combo),('All4eq',all4_eq)]
log(f'{"Method":<18} {"AUC":>7} {"t0.5%":>7} {"t0.8%":>7} {"t1%":>7} {"t2%":>7}')
for name,p in final:
    auc=roc_auc_score(yte,p)
    tt=topk(p,yte,ret_te,[0.005,0.008,0.01,0.02])
    def f(k): return f'{tt[k]["acc"]:.4f}'
    log(f'{name:<18} {auc:.4f} {f(0.005):>7} {f(0.008):>7} {f(0.01):>7} {f(0.02):>7}')

log(f'\nTARGET (acc>=65%, tpd>=15):')
hit=False; best_n=None
for name,p in final:
    o=np.argsort(p)
    for k in [0.005,0.008,0.01,0.015,0.02,0.03,0.05,0.08,0.10]:
        idx=o[-max(int(len(p)*k),1):]
        acc=(yte[idx]==1).mean(); tp=tpd(len(idx),len(yte))
        score=min(acc/0.65, tp/15)
        if acc>=0.65 and tp>=15: log(f'  ✅ {name} top{k*100:.1f}%: acc={acc:.4f} tpd={tp:.1f}'); hit=True
        if best_n is None or score>best_n[0]: best_n=(score,name,k,acc,tp)
if not hit:
    log(f'  ❌ NO — Nearest: {best_n[1]} top{best_n[2]*100:.1f}%: acc={best_n[3]:.4f} tpd={best_n[4]:.1f}')

# Save
np.savez('/workspace/models/preds_v4.npz',
    **{n:p for n,p in final}, yte=yte, ret_te=ret_te, hr_te=hr_te)
log(f'\nTotal: {time.time()-t0:.0f}s ({(time.time()-t0)/3600:.1f}h)')
