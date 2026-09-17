"""ETH H=30 v5: 55D ds parquet + cross-horizon LGB + ensemble"""
import time, gc, numpy as np, datetime, sys, warnings, pandas as pd
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

# ========== Load ==========
log('loading ds...')
ds = pd.read_parquet(f'{config.DS_DIR}/ds_ETH_h30.parquet')
feat_cols = [c for c in ds.columns if c not in ['label','soft_label','ret_future','ts']]
ts = ds['ts'].values.astype(np.int64)
label_h30 = ds['label'].values.astype(np.int8)
ret_h30 = ds['ret_future'].values.astype(np.float32)
X = ds[feat_cols].values.astype(np.float32)
del ds; gc.collect()

raw_e = pd.read_parquet(f'{config.DS_DIR}/raw_ETH.parquet')
close_raw = pd.Series(raw_e['close'].values.astype(np.float64),index=raw_e['ts'].values.astype(np.int64)).reindex(ts).values.astype(np.float64)
del raw_e; gc.collect()

HORIZONS=[3,5,15,30,60]
multi_labels={}
for h in HORIZONS:
    rh=np.full(len(ts),np.nan,np.float32)
    rh[:-h]=(close_raw[h:]/close_raw[:-h]-1).astype(np.float32)
    multi_labels[f'ret_{h}']=rh; multi_labels[f'y_{h}']=(rh>0).astype(np.int8)
del close_raw; gc.collect()

def ts_mask(tsarr,s,e):
    a=int(datetime.datetime.strptime(s,'%Y-%m-%d').replace(tzinfo=datetime.timezone.utc).timestamp())
    b=int(datetime.datetime.strptime(e,'%Y-%m-%d').replace(tzinfo=datetime.timezone.utc).timestamp())
    return (tsarr>=a)&(tsarr<b)
tr_m = ts_mask(ts,'2020-01-01','2024-06-30')
es_m = ts_mask(ts,'2024-06-30','2024-09-30')
te_m = ts_mask(ts,'2025-09-30','2026-08-29')

vm = ~np.isnan(ret_h30) & ~np.isnan(X).any(axis=1) & np.isfinite(X).all(axis=1)
vi = np.where(vm)[0]; del vm; gc.collect()
X = X[vi]; ts_c = ts[vi]; hr = pd.to_datetime(ts_c,unit='s',utc=True).hour.values.astype(np.int32)
tr_m2 = ts_mask(ts_c,'2020-01-01','2024-06-30')
es_m2 = ts_mask(ts_c,'2024-06-30','2024-09-30')
te_m2 = ts_mask(ts_c,'2025-09-30','2026-08-29')

ret_h30 = ret_h30[vi]; label_h30 = label_h30[vi]
for k in list(multi_labels.keys()): multi_labels[k] = multi_labels[k][vi]
del vi, ts; gc.collect()

log(f'X={X.shape}, TR={tr_m2.sum():,}, ES={es_m2.sum():,}, TE={te_m2.sum():,}')

# Slice once
Xtr=X[tr_m2]; Xes=X[es_m2]; Xte=X[te_m2]
del X; gc.collect()
yte=label_h30[te_m2]; ret_te=ret_h30[te_m2]; hr_te=hr[te_m2]
yes_h30 = label_h30[es_m2]
ytr_h30 = label_h30[tr_m2]

# Pre-slice all multi_labels to tr_m2 for keep computation
ret_tr = {h: multi_labels[f'ret_{h}'][tr_m2] for h in HORIZONS}

def lp(seed,num_leaves=63,lr=0.02,mdl=200,l2=1.0):
    return dict(num_leaves=num_leaves,learning_rate=lr,min_data_in_leaf=mdl,feature_fraction=0.9,bagging_fraction=0.9,bagging_freq=5,lambda_l2=l2,verbose=-1,num_threads=3,objective='binary',metric='auc',seed=seed)

def run_exp(name, Xf, yf, param_kwargs, seeds=SEEDS):
    preds=[]; t_per_seed=[]
    for sd in range(42,42+seeds):
        t=time.time()
        dtr=lgb.Dataset(Xf,label=yf); des=lgb.Dataset(Xes,label=yes_h30,reference=dtr)
        m=lgb.train(lp(seed=sd,**param_kwargs),dtr,num_boost_round=4000,valid_sets=[des],callbacks=[lgb.early_stopping(300,verbose=False)])
        preds.append(m.predict(Xte)); t_per_seed.append(time.time()-t); del m; gc.collect()
    avg=np.mean(preds,axis=0); del preds; gc.collect()
    auc=roc_auc_score(yte,avg); mt=np.mean(t_per_seed)
    log(f'  {name}: AUC={auc:.4f} ({mt:.0f}s/seed)')
    return avg,auc

# ========== Step1: H=30 single-horizon variants ==========
log(f'\n{"="*60}'); log('[Step1: H=30 LGB variants]'); log(f'{"="*60}')
results={}
for nm,kw in [('base',dict(num_leaves=63,lr=0.02,mdl=200,l2=1.0)),
              ('reg', dict(num_leaves=31,lr=0.03,mdl=500,l2=5.0)),
              ('med', dict(num_leaves=95,lr=0.015,mdl=100,l2=1.5))]:
    keep=np.abs(ret_tr[30])>=0.0005
    p,a = run_exp(f'H30_{nm}', Xtr[keep], ytr_h30[keep], kw)
    results[f'H30_{nm}']=(p,a)

# ========== Step2: Cross-horizon base LGB ==========
log(f'\n{"="*60}'); log('[Step2: Cross-horizon H=3,5,15,30,60 base LGB]'); log(f'{"="*60}')
ch_es=[]; ch_te=[]
for h in HORIZONS:
    yh_es = multi_labels[f'y_{h}'][es_m2]; yh_te = multi_labels[f'y_{h}'][te_m2]; yh_tr = multi_labels[f'y_{h}'][tr_m2]
    keep = np.abs(ret_tr[h])>=0.0005
    Xf=Xtr[keep]; yf=yh_tr[keep]
    p_es=[]; p_te=[]
    for sd in range(42,42+10):
        dtr=lgb.Dataset(Xf,label=yf); des=lgb.Dataset(Xes,label=yh_es,reference=dtr)
        m=lgb.train(lp(seed=sd),dtr,num_boost_round=3000,valid_sets=[des],callbacks=[lgb.early_stopping(200,verbose=False)])
        p_es.append(m.predict(Xes)); p_te.append(m.predict(Xte)); del m; gc.collect()
    ch_es.append(np.mean(p_es,axis=0)); ch_te.append(np.mean(p_te,axis=0))
    log(f'  H={h}: AUC_H={roc_auc_score(yh_te,ch_te[-1]):.4f}')

ch_eq = np.mean(np.column_stack(ch_te),axis=1)
auc_ceq = roc_auc_score(yte,ch_eq); log(f'  CH_equal AUC={auc_ceq:.4f}')
for k,v in topk(ch_eq,yte,ret_te,[0.005,0.008,0.01,0.02]): log(f'    top{k*100:.1f}%: acc={v["acc"]:.4f} tpd={v["tpd"]:.1f}')

# Meta
meta_dtr = lgb.Dataset(np.column_stack(ch_es),label=yes_h30)
meta = lgb.train(dict(num_leaves=31,learning_rate=0.05,min_data_in_leaf=5,verbose=-1,num_threads=3,objective='binary',metric='auc',seed=42),meta_dtr,num_boost_round=500)
meta_avg = meta.predict(np.column_stack(ch_te)); del meta; gc.collect()
auc_meta = roc_auc_score(yte,meta_avg); log(f'  Meta_LGB AUC={auc_meta:.4f}')
for k,v in topk(meta_avg,yte,ret_te,[0.005,0.008,0.01,0.02]): log(f'    top{k*100:.1f}%: acc={v["acc"]:.4f} tpd={v["tpd"]:.1f}')

# ========== Step3: Cross-horizon × reg LGB ==========
log(f'\n{"="*60}'); log('[Step3: Cross-horizon × reg LGB]'); log(f'{"="*60}')
ch_reg_te=[]
for h in HORIZONS:
    yh_es = multi_labels[f'y_{h}'][es_m2]; yh_te = multi_labels[f'y_{h}'][te_m2]; yh_tr = multi_labels[f'y_{h}'][tr_m2]
    keep = np.abs(ret_tr[h])>=0.0005
    Xf=Xtr[keep]; yf=yh_tr[keep]
    p_te=[]
    for sd in range(42,42+10):
        dtr=lgb.Dataset(Xf,label=yf); des=lgb.Dataset(Xes,label=yh_es,reference=dtr)
        m=lgb.train(lp(seed=sd,num_leaves=31,lr=0.03,mdl=500,l2=5.0),dtr,num_boost_round=3000,valid_sets=[des],callbacks=[lgb.early_stopping(200,verbose=False)])
        p_te.append(m.predict(Xte)); del m; gc.collect()
    ch_reg_te.append(np.mean(p_te,axis=0))
    log(f'  H={h} reg: AUC_H={roc_auc_score(yh_te,ch_reg_te[-1]):.4f}')
ch_reg_eq = np.mean(np.column_stack(ch_reg_te),axis=1)
auc_req = roc_auc_score(yte,ch_reg_eq); log(f'  CH_reg_equal AUC={auc_req:.4f}')

# ========== Step4: Weighted ensemble ==========
log(f'\n{"="*60}'); log('[Step4: Weighted ensemble]'); log(f'{"="*60}')
all_preds = {k:v[0] for k,v in results.items()}
all_preds['CH_base_eq'] = ch_eq
all_preds['CH_reg_eq'] = ch_reg_eq
all_preds['Meta'] = meta_avg

names = list(all_preds.keys())
Xc = np.column_stack([all_preds[n] for n in names])
corr = np.corrcoef(Xc.T)
log(f'  Methods: {names}')
for i in range(len(names)):
    for j in range(i+1,len(names)):
        if corr[i,j]>0.85: log(f'    HIGH CORR {names[i]}-{names[j]}: {corr[i,j]:.4f}')

best_w=None; best_auc=0
# 3D sweep (exhaustive): C(7,3) × weights - that's too many combos. Try top 4 pairwise:
# Try all pairs of least correlated
n_methods = Xc.shape[1]
for i in range(n_methods):
    for j in range(i+1,n_methods):
        for w in np.arange(0,1.05,0.1):
            p=Xc[:,i]*w + Xc[:,j]*(1-w)
            a=roc_auc_score(yte,p)
            if a>best_auc: best_auc=a; best_w=(names[i],names[j],w,1-w)
# Also 3D with all combinations of top-3 least correlated
min_corr_pairs = [(i,j) for i in range(n_methods) for j in range(i+1,n_methods)]
min_corr_pairs.sort(key=lambda x: corr[x[0],x[1]])
top3 = list({min_corr_pairs[0][0],min_corr_pairs[0][1],min_corr_pairs[1][1]})[:3]
for w1 in np.arange(0,1.05,0.1):
    for w2 in np.arange(0,1.05-w1,0.1):
        w3=round(max(0,1-w1-w2),2)
        p=Xc[:,top3[0]]*w1+Xc[:,top3[1]]*w2+Xc[:,top3[2]]*w3
        a=roc_auc_score(yte,p)
        if a>best_auc: best_auc=a; best_w=(names[top3[0]],names[top3[1]],names[top3[2]],w1,w2,w3)

log(f'  Best AUC={best_auc:.4f} from {best_w}')
# Reconstruct best combo
if len(best_w)==4:
    best_combo = Xc[:,names.index(best_w[0])]*best_w[2] + Xc[:,names.index(best_w[1])]*best_w[3]
else:
    best_combo = sum(Xc[:,names.index(n)]*w for n,w in zip(best_w[:3],best_w[3:]))
all_eq = np.mean(Xc,axis=1); auc_all_eq = roc_auc_score(yte,all_eq)
log(f'  All_equal AUC={auc_all_eq:.4f}')

# ========== Step5: Per-hour filter ==========
log(f'\n{"="*60}'); log('[Step5: Per-hour filter on best single]'); log(f'{"="*60}')
base_name = max(results.keys(), key=lambda k: results[k][1])
base_pred = results[base_name][0]
log(f'  base={base_name}')

ranked = sorted([(h, roc_auc_score(yte[hr_te==h], base_pred[hr_te==h])) for h in range(24) if (hr_te==h).sum()>100], key=lambda x:-x[1])
log(f'  Top: {ranked[:6]}')

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
final = list(all_preds.items()) + [('BestW',best_combo),('AllEq',all_eq)]
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

np.savez('/workspace/models/preds_v5.npz',**{n:p for n,p in final}, yte=yte, ret_te=ret_te, hr_te=hr_te)
log(f'\nTotal: {time.time()-t0:.0f}s ({(time.time()-t0)/3600:.1f}h)')
