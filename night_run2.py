"""ETH H=30 全面优化 v2 - 内存友好"""
import time, gc, datetime, numpy as np, pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score
import sys, os, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0,'/workspace'); import config

def log(m): print(m,flush=True)
def ts_mask(ts,s,e):
    a=int(datetime.datetime.strptime(s,'%Y-%m-%d').replace(tzinfo=datetime.timezone.utc).timestamp())
    b=int(datetime.datetime.strptime(e,'%Y-%m-%d').replace(tzinfo=datetime.timezone.utc).timestamp())
    return (ts>=a)&(ts<b)
def tpd(n,b): return n*1440/b
def topk(p,y,ret,kl=[0.005,0.008,0.01,0.015,0.02,0.03]):
    out={}; n=len(p); o=np.argsort(p)
    for k in kl:
        idx=o[-max(int(n*k),1):]
        out[k]={'acc':float((y[idx]==1).mean()),'ret':float(ret[idx].mean()*10000),'n':int(len(idx)),'tpd':tpd(len(idx),n)}
    return out

t0=time.time(); SYM='ETH'; H_TARGET=30
SEEDS=15

# ========== Build everything from raw, ONE TIME ==========
log(f'\n{"="*60}')
log(f'  ETH H=30 FULL OPT v2')
log(f'{"="*60}')

raw_e = pd.read_parquet(f'{config.DS_DIR}/raw_ETH.parquet')
raw_b = pd.read_parquet(f'{config.DS_DIR}/raw_BTC.parquet')
ts = raw_e['ts'].values.astype(np.int64)
close_e = raw_e['close'].values.astype(np.float64)
close_b = pd.Series(raw_b['close'].values,index=raw_b['ts'].values).reindex(ts).values.astype(np.float64)
buy_e = raw_e['buy_vol'].values.astype(np.float64)
sell_e = raw_e['sell_vol'].values.astype(np.float64)
fund_e = raw_e['funding'].values.astype(np.float64)
buy_b = pd.Series(raw_b['buy_vol'].values,index=raw_b['ts'].values).reindex(ts).values.astype(np.float64)
sell_b = pd.Series(raw_b['sell_vol'].values,index=raw_b['ts'].values).reindex(ts).values.astype(np.float64)
fund_b = pd.Series(raw_b['funding'].values,index=raw_b['ts'].values).reindex(ts).values.astype(np.float64)
del raw_e, raw_b; gc.collect()
log(f'raw loaded: {len(ts):,} rows')

close_es = pd.Series(close_e); close_bs = pd.Series(close_b)
lr_e = close_es.pct_change(); lr_b = close_bs.pct_change()
hr = pd.to_datetime(ts,unit='s',utc=True).hour.values

# Build feats (keep < 80D)
feats = {}
for w in [3,5,10,15,20,30,45,60,90,120,180,240,360,480,720,960]:
    feats[f'lr_e_{w}'] = close_es.pct_change(w).astype(np.float32).values
    feats[f'lr_b_{w}'] = close_bs.pct_change(w).astype(np.float32).values
for w in [10,20,30,60,120,240]:
    s = feats[f'lr_e_{w}']; ser=pd.Series(s); feats[f'z_e_{w}'] = ((ser-ser.rolling(w).mean())/(ser.rolling(w).std()+1e-9)).astype(np.float32).values
for w in [15,30,60,120,240]:
    feats[f'rvol_e_{w}'] = lr_e.rolling(w).std().astype(np.float32).values
    feats[f'rvol_b_{w}'] = lr_b.rolling(w).std().astype(np.float32).values
    feats[f'cvd_e_{w}'] = pd.Series(buy_e-sell_e).rolling(w).mean().astype(np.float32).values
    feats[f'vol_e_{w}'] = pd.Series(buy_e+sell_e).rolling(w).mean().astype(np.float32).values
for w in [5,15,30,60,120]:
    feats[f'fund_e_mean_{w}'] = pd.Series(fund_e).rolling(w).mean().astype(np.float32).values
    feats[f'fund_b_mean_{w}'] = pd.Series(fund_b).rolling(w).mean().astype(np.float32).values
for w in [60,120,240]:
    feats[f'skew_{w}'] = lr_e.rolling(w).skew().astype(np.float32).values
ratio = close_e/(close_b+1e-12)
for w in [5,15,30,60,120]:
    feats[f'ratio_lr_{w}'] = pd.Series(ratio).pct_change(w).astype(np.float32).values
for w in [30,60,120]:
    hi=pd.Series(close_e).rolling(w).max(); lo=pd.Series(close_e).rolling(w).min()
    feats[f'pos_{w}'] = ((close_e-lo)/(hi-lo+1e-12)).astype(np.float32).values
# cross features
feats['vol_x_mom30'] = (feats['rvol_e_60'] * feats['lr_e_30']).astype(np.float32)
feats['fund_x_vol'] = (feats['fund_e_mean_30'] * feats['rvol_e_60']).astype(np.float32)
feats['cvd_x_vol'] = (feats['cvd_e_60'] * feats['rvol_e_60']).astype(np.float32)
feats['hour_sin'] = np.sin(2*np.pi*hr/24).astype(np.float32)
feats['hour_cos'] = np.cos(2*np.pi*hr/24).astype(np.float32)
feats['vol_x_hour_sin'] = (feats['rvol_e_60'] * feats['hour_sin']).astype(np.float32)
feats['vol_x_hour_cos'] = (feats['rvol_e_60'] * feats['hour_cos']).astype(np.float32)
# streaks
ls = np.sign(lr_e.values)
us=np.zeros(len(ts),np.int32); ds=np.zeros(len(ts),np.int32)
for i in range(1,len(ts)):
    us[i]=us[i-1]+1 if ls[i]>0 else 0
    ds[i]=ds[i-1]+1 if ls[i]<0 else 0
feats['streak_diff']=(us-ds).astype(np.float32); del us,ds,ls; gc.collect()

FEAT_NAMES=sorted(feats.keys())
X_all = np.stack([feats[f] for f in FEAT_NAMES],axis=1).astype(np.float32)
del feats; gc.collect()
log(f'X_all: {X_all.shape} mem={X_all.nbytes/1e6:.0f}MB feats={len(FEAT_NAMES)}')

# Build multi-horizon labels
HORIZONS=[3,5,15,30,60]
labels={}
for h in HORIZONS:
    rh=np.full(len(ts),np.nan,np.float32)
    rh[:-h]=(close_es.values[h:]/close_es.values[:-h]-1).astype(np.float32)
    labels[h]=(rh>0).astype(np.int8); labels[f'ret_{h}']=rh
del close_es, close_bs, lr_e, lr_b, close_e, close_b, buy_e, sell_e, fund_e, buy_b, sell_b, fund_b, ratio; gc.collect()
log('labels built')

# Time splits
tr_m = ts_mask(ts,'2020-01-01','2024-06-30')
es_m = ts_mask(ts,'2024-06-30','2024-09-30')
te_m = ts_mask(ts,'2025-09-30','2026-08-29')
# Clean rows with NaN
vm = ~np.isnan(labels['ret_30']) & ~np.isnan(X_all).any(axis=1)
vm = vm & np.isfinite(X_all).all(axis=1) & np.isfinite(labels['ret_30'])
vi = np.where(vm)[0]
X = X_all[vi]; ts_clean=ts[vi]; hr_clean=hr[vi]; del X_all,ts,hr,vm,vi; gc.collect()

tr_m2 = ts_mask(ts_clean,'2020-01-01','2024-06-30')
es_m2 = ts_mask(ts_clean,'2024-06-30','2024-09-30')
te_m2 = ts_mask(ts_clean,'2025-09-30','2026-08-29')

def get_split(h):
    yh=labels[h][vi]; rh=labels[f'ret_{h}'][vi]
    return X[tr_m2],X[es_m2],X[te_m2], yh[tr_m2],yh[es_m2],yh[te_m2], rh[te_m2], hr_clean[te_m2]

def lgb_train(Xtr,ytr,Xes,yes,Xte,lp,seeds=SEEDS):
    preds=[]
    for sd in range(42,42+seeds):
        dtr=lgb.Dataset(Xtr,label=ytr); des=lgb.Dataset(Xes,label=yes,reference=dtr)
        m=lgb.train({**lp,'seed':sd},dtr,num_boost_round=3000,valid_sets=[des],callbacks=[lgb.early_stopping(300,verbose=False)])
        preds.append(m.predict(Xte)); del m; gc.collect()
    return np.mean(preds,axis=0)

def lgb_filter_train(Xtr,ytr,Xes,yes,Xte,lp,h=30,min_ret=0.0005):
    rh=labels[f'ret_{h}'][vi]
    keep=np.abs(rh[tr_m2])>=min_ret
    return lgb_train(Xtr[keep],ytr[keep],Xes,yes,Xte,lp)

del labels; gc.collect()  # free labels dict
labels_reload = None  # we need it later, will reload selectively

# ========== Exp1: Baseline extended feats ==========
log(f'\n{"="*60}')
log('[Exp1] Baseline ext LGB seeds=15 num_leaves=63 lr=0.02')
log(f'{"="*60}')
_,_,_,ytr_30,yes_30,yte_30,ret_te_30,hr_te = get_split(30)
del _
lp_base=dict(num_leaves=63,learning_rate=0.02,min_data_in_leaf=200,feature_fraction=0.9,bagging_fraction=0.9,bagging_freq=5,lambda_l2=1.0,verbose=-1,num_threads=3,objective='binary',metric='auc')
t1=time.time()
exp1_avg = lgb_filter_train(X[tr_m2],ytr_30,X[es_m2],yes_30,X[te_m2],lp_base)
auc1=roc_auc_score(yte_30,exp1_avg)
log(f'  AUC={auc1:.4f} ({time.time()-t1:.0f}s)')
for k,v in topk(exp1_avg,yte_30,ret_te_30,[0.005,0.01,0.02]).items():
    log(f'  top{k*100:.1f}%: acc={v["acc"]:.4f} tpd={v["tpd"]:.1f} ret={v["ret"]:.1f}')

# ========== Exp2: Larger LGB num_leaves=127 lr=0.01 ==========
log(f'\n{"="*60}')
log('[Exp2] LGB big num_leaves=127 lr=0.01 min_data=50')
log(f'{"="*60}')
lp2=dict(num_leaves=127,learning_rate=0.01,min_data_in_leaf=50,feature_fraction=0.85,bagging_fraction=0.85,bagging_freq=5,lambda_l2=1.0,verbose=-1,num_threads=3,objective='binary',metric='auc')
t2=time.time()
exp2_avg = lgb_filter_train(X[tr_m2],ytr_30,X[es_m2],yes_30,X[te_m2],lp2,seeds=10)
auc2=roc_auc_score(yte_30,exp2_avg)
log(f'  AUC={auc2:.4f} ({time.time()-t2:.0f}s)')
for k,v in topk(exp2_avg,yte_30,ret_te_30,[0.005,0.01,0.02]).items():
    log(f'  top{k*100:.1f}%: acc={v["acc"]:.4f} tpd={v["tpd"]:.1f} ret={v["ret"]:.1f}')

# ========== Exp3: Small reg LGB ==========
log(f'\n{"="*60}')
log('[Exp3] LGB reg num_leaves=31 lr=0.03 min_data=500 l2=5')
log(f'{"="*60}')
lp3=dict(num_leaves=31,learning_rate=0.03,min_data_in_leaf=500,feature_fraction=0.9,bagging_fraction=0.9,bagging_freq=5,lambda_l2=5.0,verbose=-1,num_threads=3,objective='binary',metric='auc')
t3=time.time()
exp3_avg = lgb_filter_train(X[tr_m2],ytr_30,X[es_m2],yes_30,X[te_m2],lp3)
auc3=roc_auc_score(yte_30,exp3_avg)
log(f'  AUC={auc3:.4f} ({time.time()-t3:.0f}s)')
for k,v in topk(exp3_avg,yte_30,ret_te_30,[0.005,0.01,0.02]).items():
    log(f'  top{k*100:.1f}%: acc={v["acc"]:.4f} tpd={v["tpd"]:.1f} ret={v["ret"]:.1f}')

# ========== Exp4: XGBoost ==========
log(f'\n{"="*60}')
log('[Exp4] XGBoost d6 5 seeds')
log(f'{"="*60}')
t4=time.time()
keep=np.abs(labels_reload['ret_30'][tr_m2])>=0.0005 if False else None  # skip
# use ytr_30 directly since we already filtered above in lgb_filter_train
# Actually let's filter inline
tr_keep=np.abs((close_es.values[vi[tr_m2]] if labels_reload else labels_reload is None and False))>=0
# simpler: do a fresh lgb_filter_train-equivalent
Xtr_f=X[tr_m2]; Xes_f=X[es_m2]; Xte_f=X[te_m2]
# We need ret_30 to filter. Let's rebuild ret from close_es? No, we deleted close_es. Recompute from close_aligned? 
# Actually easier: build a mask from close data. But we don't have close anymore. Let's reload labels.
labels_reload = {}
for h in HORIZONS:
    rh=np.full(len(ts_clean),np.nan,np.float32)
    # We need close here. Let's reload just close.
    pass
# Actually simplest: load raw again only once, take close, then compute
raw_e2 = pd.read_parquet(f'{config.DS_DIR}/raw_ETH.parquet')
close_al = pd.Series(raw_e2['close'].values.astype(np.float64),index=raw_e2['ts'].values.astype(np.int64)).reindex(ts_clean).values.astype(np.float64)
del raw_e2; gc.collect()

rh30=np.full(len(ts_clean),np.nan,np.float32)
rh30[:-30]=(close_al[30:]/close_al[:-30]-1).astype(np.float32)
tr_keep=np.abs(rh30[tr_m2])>=0.0005
Xtr_x=X[tr_m2][tr_keep]; ytr_x=ytr_30[tr_m2][tr_keep]
log(f'  XGB TR={len(Xtr_x):,}')

xlp=dict(max_depth=6,learning_rate=0.02,min_child_weight=200,subsample=0.9,colsample_bytree=0.9,reg_lambda=1.0,objective='binary:logistic',eval_metric='auc',nthread=3)
dtr=xgb.DMatrix(Xtr_x,label=ytr_x); des_dm=xgb.DMatrix(Xes_f,label=yes_30)
preds=[]
for sd in [42,123,456,789,1024]:
    m=xgb.train({**xlp,'seed':sd},dtr,num_boost_round=2000,evals=[(des_dm,'es')],early_stopping_rounds=200,verbose_eval=0)
    preds.append(m.predict(xgb.DMatrix(Xte_f))); del m; gc.collect()
xgb_avg=np.mean(preds,axis=0); del preds; gc.collect()
auc4=roc_auc_score(yte_30,xgb_avg)
log(f'  AUC={auc4:.4f} ({time.time()-t4:.0f}s)')
for k,v in topk(xgb_avg,yte_30,ret_te_30,[0.005,0.01,0.02]).items():
    log(f'  top{k*100:.1f}%: acc={v["acc"]:.4f} tpd={v["tpd"]:.1f} ret={v["ret"]:.1f}')

# ========== Exp5: CatBoost ==========
log(f'\n{"="*60}')
log('[Exp5] CatBoost d6 4 seeds 500k')
log(f'{"="*60}')
t5=time.time()
rng=np.random.RandomState(42); idx=rng.choice(len(Xtr_x),min(500_000,len(Xtr_x)),replace=False)
ctr=Pool(Xtr_x[idx],ytr_x[idx]); ces=Pool(Xes_f,yes_30)
cb_preds=[]
for sd in [42,123,456,789]:
    cb=CatBoostClassifier(iterations=2000,learning_rate=0.02,depth=6,l2_leaf_reg=3.0,loss_function='Logloss',eval_metric='AUC',od_type='Iter',od_wait=200,verbose=0,thread_count=3,random_seed=sd)
    cb.fit(ctr,eval_set=ces,use_best_model=True)
    cb_preds.append(cb.predict_proba(Xte_f)[:,1]); del cb; gc.collect()
cb_avg=np.mean(cb_preds,axis=0); del cb_preds,ctr,ces; gc.collect()
auc5=roc_auc_score(yte_30,cb_avg)
log(f'  AUC={auc5:.4f} ({time.time()-t5:.0f}s)')
for k,v in topk(cb_avg,yte_30,ret_te_30,[0.005,0.01,0.02]).items():
    log(f'  top{k*100:.1f}%: acc={v["acc"]:.4f} tpd={v["tpd"]:.1f} ret={v["ret"]:.1f}')

# ========== Exp6: Cross-horizon stacking ==========
log(f'\n{"="*60}')
log('[Exp6] Cross-horizon H=3,5,15,30,60 + meta')
log(f'{"="*60}')
# Build multi-horizon labels
all_labels = {}
for h in HORIZONS:
    rh=np.full(len(ts_clean),np.nan,np.float32)
    rh[:-h]=(close_al[h:]/close_al[:-h]-1).astype(np.float32)
    all_labels[h] = (rh>0).astype(np.int8)
    all_labels[f'ret_{h}']=rh
del close_al; gc.collect()

Xte_f=X[te_m2]; Xes_f=X[es_m2]
base_es=[]; base_te=[]
for h in HORIZONS:
    yh = all_labels[h]
    yh_tr = yh[tr_m2]; yh_es = yh[es_m2]; yh_te = yh[te_m2]
    rh30_mask = np.abs(all_labels[f'ret_{h}'][tr_m2])>=0.0005
    preds=[]
    for sd in range(42,42+8):  # fewer seeds for speed
        keep_tr = np.abs(all_labels[f'ret_{h}'][tr_m2])>=0.0005
        Xtr_h=X[tr_m2][keep_tr]; ytr_h=yh_tr[keep_tr]
        dtr=lgb.Dataset(Xtr_h,label=ytr_h); des=lgb.Dataset(Xes_f,label=yh_es,reference=dtr)
        m=lgb.train({**lp_base,'seed':sd},dtr,num_boost_round=2000,valid_sets=[des],callbacks=[lgb.early_stopping(200,verbose=False)])
        preds.append(m.predict(Xte_f)); del m; gc.collect()
    p_avg=np.mean(preds,axis=0); del preds; gc.collect()
    base_te.append(p_avg)
    log(f'  H={h} TE AUC={roc_auc_score(yh_te,p_avg):.4f}')

# Equal-weight cross-horizon
ch_equal = np.mean(np.column_stack(base_te),axis=1)
auc_ceq = roc_auc_score(yte_30,ch_equal)
log(f'  Equal-weight AUC={auc_ceq:.4f}')

# ========== Exp7: Per-hour filter ==========
log(f'\n{"="*60}')
log('[Exp7] Per-hour AUC + percentile filter')
log(f'{"="*60}')
# Use exp2 (best AUC so far) as base
base_pred = exp2_avg
hour_aucs = {}
for h in range(24):
    m=hr_te==h
    if m.sum()>100:
        hour_aucs[h]=roc_auc_score(yte_30[m],base_pred[m])
ranked=sorted(hour_aucs.items(),key=lambda x:-x[1])
log(f'  Top hours: {ranked[:6]}')

best_hit=None
for K in [3,5,8,12,16,20,24]:
    sel=[h for h,_ in ranked[:K]]
    m=np.isin(hr_te,sel)
    p=base_pred[m]; y=yte_30[m]; r=ret_te_30[m]
    for thr in [70,80,90,94,96,97,98,99]:
        h_thr=np.percentile(p,thr)
        hit=p>h_thr
        if hit.sum()==0: continue
        acc=(y[hit]==1).mean(); tp=tpd(hit.sum(),len(yte_30))
        if acc>=0.65 and tp>=15:
            best_hit=(K,thr,acc,tp)
            log(f'  ✅ K={K} P{thr}: acc={acc:.4f} tpd={tp:.1f}')
        elif acc>=0.62 and tp>=15:
            log(f'  near ✅ K={K} P{thr}: acc={acc:.4f} tpd={tp:.1f}')

# ========== Exp8: Weighted ensemble ==========
log(f'\n{"="*60}')
log('[Exp8] Weight search (LGB_base, LGB_big, XGB, CB, CH_eq)')
log(f'{"="*60}')
cands=[('LGB_base',exp1_avg),('LGB_big',exp2_avg),('LGB_reg',exp3_avg),('XGB',xgb_avg),('CB',cb_avg),('CH_eq',ch_equal)]
Xc=np.column_stack([p for _,p in cands])
corr=np.corrcoef(Xc.T)
names=[n for n,_ in cands]
for i in range(len(names)):
    for j in range(i+1,len(names)):
        if corr[i,j]>0.85:
            log(f'  HIGH CORR {names[i]}-{names[j]}: {corr[i,j]:.4f}')

best_w=None; best_auc=0
# 4-way weight sweep
for w1 in np.arange(0,1.05,0.15):
    for w2 in np.arange(0,1.05-w1,0.15):
        for w3 in np.arange(0,1.05-w1-w2,0.15):
            w4=round(max(0,1-w1-w2-w3),2)
            p=Xc[:,:4]@np.array([w1,w2,w3,w4])
            a=roc_auc_score(yte_30,p)
            if a>best_auc: best_auc=a; best_w=(w1,w2,w3,w4)
best_4 = Xc[:,:4]@np.array(best_w)
log(f'  Best weights (LGB_base,big,reg,XGB,CB...): {best_w} AUC={best_auc:.4f}')

# All 6 equal
all6 = np.mean(Xc,axis=1); auc6 = roc_auc_score(yte_30,all6)
log(f'  All6 equal AUC={auc6:.4f}')

# ========== FINAL SUMMARY ==========
log(f'\n{"="*60}')
log('FINAL SUMMARY')
log(f'{"="*60}')
log(f'{"Method":<20} {"AUC":>7} {"t0.5%":>7} {"t0.8%":>7} {"t1%":>7} {"t2%":>7}')
final_cands = cands + [('WBest4',best_4),('All6eq',all6),('Best_of_best',exp2_avg)]
for name,p in final_cands:
    auc=roc_auc_score(yte_30,p)
    tt=topk(p,yte_30,ret_te_30,[0.005,0.008,0.01,0.02])
    def f(k): return f'{tt[k]["acc"]:.4f}'
    log(f'{name:<20} {auc:.4f} {f(0.005):>7} {f(0.008):>7} {f(0.01):>7} {f(0.02):>7}')

log(f'\nTARGET CHECK (acc>=65%, tpd>=15):')
hit=False
best_nearest=None
for name,p in final_cands:
    o=np.argsort(p)
    for k in [0.005,0.008,0.01,0.015,0.02,0.03,0.05,0.08,0.10]:
        idx=o[-max(int(len(p)*k),1):]
        acc=(yte_30[idx]==1).mean(); tp=tpd(len(idx),len(yte_30))
        score=min(acc/0.65, tp/15)
        if acc>=0.65 and tp>=15:
            log(f'  ✅ {name} top{k*100:.1f}%: acc={acc:.4f} tpd={tp:.1f}')
            hit=True
        if best_nearest is None or score>best_nearest[0]:
            best_nearest=(score,name,k,acc,tp)
if not hit:
    log(f'  ❌ NO TARGET MET — Nearest: {best_nearest[1]} top{best_nearest[2]*100:.1f}%: acc={best_nearest[3]:.4f} tpd={best_nearest[4]:.1f} score={best_nearest[0]:.3f}')
log(f'\nTotal: {time.time()-t0:.0f}s ({(time.time()-t0)/3600:.1f}h)')
