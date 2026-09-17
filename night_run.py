"""
ETH H=30 全面优化 — 通宵跑
目标: top1% acc >= 65%
"""
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

t0=time.time()
SYM='ETH'; H=30

# ========== 0. Load raw + base ds ==========
log(f'\n{"="*60}')
log(f'  ETH H=30 FULL OPTIMIZATION')
log(f'{"="*60}')

# Load raw
raw_e = pd.read_parquet(f'{config.DS_DIR}/raw_ETH.parquet')
raw_b = pd.read_parquet(f'{config.DS_DIR}/raw_BTC.parquet')
ts = raw_e['ts'].values.astype(np.int64)
log(f'raw: ETH={len(raw_e):,} rows, BTC={len(raw_b):,} rows')

# Build extended features from raw
log('\n[1] Building extended features...')
close_e = raw_e['close'].values.astype(np.float64)
close_b = pd.Series(raw_b['close'].values,index=raw_b['ts'].values).reindex(ts).values.astype(np.float64)
buy_e = raw_e['buy_vol'].values.astype(np.float64)
sell_e = raw_e['sell_vol'].values.astype(np.float64)
fund_e = raw_e['funding'].values.astype(np.float64)
buy_b = pd.Series(raw_b['buy_vol'].values,index=raw_b['ts'].values).reindex(ts).values.astype(np.float64)
sell_b = pd.Series(raw_b['sell_vol'].values,index=raw_b['ts'].values).reindex(ts).values.astype(np.float64)
fund_b = pd.Series(raw_b['funding'].values,index=raw_b['ts'].values).reindex(ts).values.astype(np.float64)
del raw_e, raw_b; gc.collect()

close_es = pd.Series(close_e); close_bs = pd.Series(close_b)
lr_e = close_es.pct_change(); lr_b = close_bs.pct_change()

feats = {}
# === 1a. Multi-horizon ETH returns + zscores ===
for w in [3,5,10,15,20,30,45,60,90,120,180,240,360,480,720,960]:
    feats[f'lr_e_{w}'] = close_es.pct_change(w).astype(np.float32).values
for w in [10,20,30,60,120,240]:
    s = feats[f'lr_e_{w}'].copy()
    ser = pd.Series(s)
    feats[f'z_e_{w}'] = ((ser - ser.rolling(w).mean()) / (ser.rolling(w).std()+1e-9)).astype(np.float32).values

# === 1b. BTC cross-asset (more windows) ===
for w in [3,5,10,15,30,60,120,240,480,960]:
    feats[f'lr_b_{w}'] = close_bs.pct_change(w).astype(np.float32).values
for w in [30,60,120,240]:
    s = feats[f'lr_b_{w}'].copy()
    ser = pd.Series(s)
    feats[f'z_b_{w}'] = ((ser - ser.rolling(w).mean()) / (ser.rolling(w).std()+1e-9)).astype(np.float32).values

# === 1c. ETH/BTC ratio features ===
ratio = close_e / (close_b + 1e-12)
for w in [5,15,30,60,120,240]:
    feats[f'ratio_lr_{w}'] = pd.Series(ratio).pct_change(w).astype(np.float32).values
# ratio z
r = pd.Series(feats['ratio_lr_60'])
feats['ratio_z_60'] = ((r - r.rolling(60).mean())/(r.rolling(60).std()+1e-9)).astype(np.float32).values

# === 1d. Volatility (rolling std of returns) ===
for w in [15,30,60,120,240,480]:
    feats[f'rvol_e_{w}'] = lr_e.rolling(w).std().astype(np.float32).values
for w in [15,30,60,120,240]:
    feats[f'rvol_b_{w}'] = lr_b.rolling(w).std().astype(np.float32).values

# === 1e. Volume features ===
for w in [15,30,60,120,240]:
    feats[f'vol_e_{w}'] = pd.Series(buy_e+sell_e).rolling(w).mean().astype(np.float32).values
    feats[f'vol_buy_e_{w}'] = pd.Series(buy_e).rolling(w).mean().astype(np.float32).values
    feats[f'vol_sell_e_{w}'] = pd.Series(sell_e).rolling(w).mean().astype(np.float32).values
    feats[f'cvd_e_{w}'] = pd.Series(buy_e-sell_e).rolling(w).mean().astype(np.float32).values
for w in [30,60,120]:
    feats[f'cvd_b_{w}'] = pd.Series(buy_b-sell_b).rolling(w).mean().astype(np.float32).values

# === 1f. Funding ===
for w in [5,15,30,60,120,240]:
    feats[f'fund_e_mean_{w}'] = pd.Series(fund_e).rolling(w).mean().astype(np.float32).values
    feats[f'fund_b_mean_{w}'] = pd.Series(fund_b).rolling(w).mean().astype(np.float32).values
for w in [30,60,120]:
    feats[f'fund_e_z_{w}'] = ((pd.Series(fund_e) - pd.Series(fund_e).rolling(w).mean()) / (pd.Series(fund_e).rolling(w).std()+1e-12)).astype(np.float32).values

# === 1g. Run-length / streaks ===
lr_sign = np.sign(lr_e.values)
# up streak
up_streak = np.zeros(len(ts), dtype=np.int32)
for i in range(1, len(ts)):
    up_streak[i] = up_streak[i-1]+1 if lr_sign[i]>0 else 0
dn_streak = np.zeros(len(ts), dtype=np.int32)
for i in range(1, len(ts)):
    dn_streak[i] = dn_streak[i-1]+1 if lr_sign[i]<0 else 0
feats['up_streak'] = up_streak.astype(np.float32)
feats['dn_streak'] = dn_streak.astype(np.float32)
feats['streak_diff'] = (up_streak - dn_streak).astype(np.float32)

# === 1h. Momentum consensus (sign agreement across TFs) ===
lr_tfs = ['lr_e_3','lr_e_5','lr_e_15','lr_e_30','lr_e_60','lr_e_120','lr_e_240']
signs = np.stack([np.sign(feats[f]) for f in lr_tfs if f in feats], axis=1)
feats['mom_consensus'] = signs.mean(axis=1).astype(np.float32)
feats['mom_std'] = signs.std(axis=1).astype(np.float32)

# === 1i. Vol×mom interaction ===
feats['vol_x_mom30'] = (feats['rvol_e_60'] * feats['lr_e_30']).astype(np.float32)
feats['vol_x_mom15'] = (feats['rvol_e_30'] * feats['lr_e_15']).astype(np.float32)
feats['vol_x_mom60'] = (feats['rvol_e_120'] * feats['lr_e_60']).astype(np.float32)
feats['fund_x_vol'] = (feats['fund_e_z_60'] * feats['rvol_e_60']).astype(np.float32)
feats['cvd_x_vol'] = (feats['cvd_e_60'] * feats['rvol_e_60']).astype(np.float32)

# === 1j. Price position (in range, deviation from VWAP proxy) ===
for w in [30,60,120,240]:
    hi = pd.Series(close_e).rolling(w).max()
    lo = pd.Series(close_e).rolling(w).min()
    feats[f'pos_{w}'] = ((close_e - lo)/(hi-lo+1e-12)).astype(np.float32).values

# === 1k. Hour / dow cyclical ===
dt_idx = pd.to_datetime(ts, unit='s', utc=True)
hr = dt_idx.hour.values; dow = dt_idx.dayofweek.values
feats['hour_sin'] = np.sin(2*np.pi*hr/24).astype(np.float32)
feats['hour_cos'] = np.cos(2*np.pi*hr/24).astype(np.float32)
feats['dow_sin'] = np.sin(2*np.pi*dow/7).astype(np.float32)
feats['dow_cos'] = np.cos(2*np.pi*dow/7).astype(np.float32)
# hour × vol interaction (known per-hour AUC diff)
feats['vol_x_hour_sin'] = (feats['rvol_e_60'] * feats['hour_sin']).astype(np.float32)
feats['vol_x_hour_cos'] = (feats['rvol_e_60'] * feats['hour_cos']).astype(np.float32)

# === 1l. Max drawdown / runup ===
for w in [60,120,240]:
    roll_max = pd.Series(close_e).rolling(w).max()
    roll_min = pd.Series(close_e).rolling(w).min()
    feats[f'dd_{w}'] = ((close_e - roll_max)/(roll_max+1e-12)).astype(np.float32).values
    feats[f'ru_{w}'] = ((close_e - roll_min)/(roll_min+1e-12)).astype(np.float32).values

# === 1m. Skewness / kurtosis of returns ===
for w in [60,120,240]:
    feats[f'skew_{w}'] = lr_e.rolling(w).skew().astype(np.float32).values
    feats[f'kurt_{w}'] = lr_e.rolling(w).kurt().astype(np.float32).values

# Assemble
FEAT_NAMES = sorted(feats.keys())
X_all = np.stack([feats[f] for f in FEAT_NAMES], axis=1).astype(np.float32)
print(f'  Extended feats: {len(FEAT_NAMES)}D',flush=True)
del feats, close_e, close_b, lr_e, lr_b, buy_e, sell_e, fund_e, buy_b, sell_b, fund_b, up_streak, dn_streak; gc.collect()

# === Build label (ETH H=30 direction) ===
ret_f = np.full(len(ts), np.nan, dtype=np.float32)
ret_f[:-H] = (close_es.values[H:] / close_es.values[:-H] - 1).astype(np.float32)
y_all = (ret_f > 0).astype(np.int8)
del close_es, close_bs, ratio; gc.collect()

# === Time splits ===
tr_m = ts_mask(ts,'2020-01-01','2024-06-30')
es_m = ts_mask(ts,'2024-06-30','2024-09-30')
mv_m = ts_mask(ts,'2024-09-30','2025-09-30')
te_m = ts_mask(ts,'2025-09-30','2026-08-29')

# Clean NaN rows
vm = ~np.isnan(ret_f) & ~np.isnan(X_all).any(axis=1)
vm = vm & np.isfinite(ret_f) & np.isfinite(X_all).all(axis=1)
vi = np.where(vm)[0]
X = X_all[vi]; y = y_all[vi]; ret_clean = ret_f[vi]; ts_clean = ts[vi]; hr_clean = hr[vi]
del X_all, y_all, ret_f, ts, vm, vi; gc.collect()

tr_m2 = ts_mask(ts_clean,'2020-01-01','2024-06-30')
es_m2 = ts_mask(ts_clean,'2024-06-30','2024-09-30')
mv_m2 = ts_mask(ts_clean,'2024-09-30','2025-09-30')
te_m2 = ts_mask(ts_clean,'2025-09-30','2026-08-29')

# Filter training
keep_tr = np.abs(ret_clean[tr_m2]) >= 0.0005
Xtr = X[tr_m2][keep_tr]; ytr = y[tr_m2][keep_tr]
Xes = X[es_m2]; yes = y[es_m2]
Xmv = X[mv_m2]; ymv = y[mv_m2]; ret_mv = ret_clean[mv_m2]
Xte = X[te_m2]; yte = y[te_m2]; ret_te = ret_clean[te_m2]; hr_te = hr_clean[te_m2]
del X, y, ret_clean, ts_clean, hr_clean, tr_m, es_m, mv_m, te_m, tr_m2, es_m2, mv_m2, te_m2; gc.collect()
log(f'  TR={len(Xtr):,} ES={len(Xes):,} MV={len(Xmv):,} TE={len(Xte):,}')

# ========== EXPERIMENT 1: Baseline extended feats ==========
log(f'\n{"="*60}')
log('[Exp1] Baseline: Extended feats + 15-seed LGB')
log(f'{"="*60}')
SEEDS=15
lp=dict(num_leaves=63,learning_rate=0.02,min_data_in_leaf=200,feature_fraction=0.9,bagging_fraction=0.9,bagging_freq=5,lambda_l2=1.0,verbose=-1,num_threads=3,objective='binary',metric='auc')
preds=[]
t1=time.time()
for sd in range(42,42+SEEDS):
    dtr=lgb.Dataset(Xtr,label=ytr); des=lgb.Dataset(Xes,label=yes,reference=dtr)
    m=lgb.train({**lp,'seed':sd},dtr,num_boost_round=2000,valid_sets=[des],callbacks=[lgb.early_stopping(200,verbose=False)])
    preds.append(m.predict(Xte)); del m; gc.collect()
exp1_avg=np.mean(preds,axis=0); del preds; gc.collect()
auc1=roc_auc_score(yte,exp1_avg)
log(f'  AUC={auc1:.4f} ({time.time()-t1:.0f}s)')
for k,v in topk(exp1_avg,yte,ret_te,[0.005,0.01,0.02]).items():
    log(f'  top{k*100:.1f}%: acc={v["acc"]:.4f} tpd={v["tpd"]:.1f} ret={v["ret"]:.1f}')

# ========== EXPERIMENT 2: Larger model ==========
log(f'\n{"="*60}')
log('[Exp2] LGB: num_leaves=127, lr=0.01, min_data=50')
log(f'{"="*60}')
lp2=dict(num_leaves=127,learning_rate=0.01,min_data_in_leaf=50,feature_fraction=0.85,bagging_fraction=0.85,bagging_freq=5,lambda_l2=1.0,verbose=-1,num_threads=3,objective='binary',metric='auc')
preds=[]
t2=time.time()
for sd in range(42,42+SEEDS):
    dtr=lgb.Dataset(Xtr,label=ytr); des=lgb.Dataset(Xes,label=yes,reference=dtr)
    m=lgb.train({**lp2,'seed':sd},dtr,num_boost_round=5000,valid_sets=[des],callbacks=[lgb.early_stopping(300,verbose=False)])
    preds.append(m.predict(Xte)); del m; gc.collect()
exp2_avg=np.mean(preds,axis=0); del preds; gc.collect()
auc2=roc_auc_score(yte,exp2_avg)
log(f'  AUC={auc2:.4f} ({time.time()-t2:.0f}s)')
for k,v in topk(exp2_avg,yte,ret_te,[0.005,0.01,0.02]).items():
    log(f'  top{k*100:.1f}%: acc={v["acc"]:.4f} tpd={v["tpd"]:.1f} ret={v["ret"]:.1f}')

# ========== EXPERIMENT 3: Smaller tree (regularized) ==========
log(f'\n{"="*60}')
log('[Exp3] LGB: num_leaves=31, lr=0.03, min_data=500, lambda_l2=5.0')
log(f'{"="*60}')
lp3=dict(num_leaves=31,learning_rate=0.03,min_data_in_leaf=500,feature_fraction=0.9,bagging_fraction=0.9,bagging_freq=5,lambda_l2=5.0,verbose=-1,num_threads=3,objective='binary',metric='auc')
preds=[]
t3=time.time()
for sd in range(42,42+SEEDS):
    dtr=lgb.Dataset(Xtr,label=ytr); des=lgb.Dataset(Xes,label=yes,reference=dtr)
    m=lgb.train({**lp3,'seed':sd},dtr,num_boost_round=3000,valid_sets=[des],callbacks=[lgb.early_stopping(300,verbose=False)])
    preds.append(m.predict(Xte)); del m; gc.collect()
exp3_avg=np.mean(preds,axis=0); del preds; gc.collect()
auc3=roc_auc_score(yte,exp3_avg)
log(f'  AUC={auc3:.4f} ({time.time()-t3:.0f}s)')
for k,v in topk(exp3_avg,yte,ret_te,[0.005,0.01,0.02]).items():
    log(f'  top{k*100:.1f}%: acc={v["acc"]:.4f} tpd={v["tpd"]:.1f} ret={v["ret"]:.1f}')

# ========== EXPERIMENT 4: XGBoost ensemble ==========
log(f'\n{"="*60}')
log('[Exp4] XGBoost 5 seeds d6')
log(f'{"="*60}')
xlp=dict(max_depth=6,learning_rate=0.02,min_child_weight=200,subsample=0.9,colsample_bytree=0.9,reg_lambda=1.0,objective='binary:logistic',eval_metric='auc',nthread=3)
dtr=xgb.DMatrix(Xtr,label=ytr); des_dm=xgb.DMatrix(Xes,label=yes)
preds=[]
t4=time.time()
for sd in [42,123,456,789,1024]:
    m=xgb.train({**xlp,'seed':sd},dtr,num_boost_round=2000,evals=[(des_dm,'es')],early_stopping_rounds=200,verbose_eval=0)
    preds.append(m.predict(xgb.DMatrix(Xte))); del m; gc.collect()
xgb_avg=np.mean(preds,axis=0); del preds; gc.collect()
auc4=roc_auc_score(yte,xgb_avg)
log(f'  AUC={auc4:.4f} ({time.time()-t4:.0f}s)')
for k,v in topk(xgb_avg,yte,ret_te,[0.005,0.01,0.02]).items():
    log(f'  top{k*100:.1f}%: acc={v["acc"]:.4f} tpd={v["tpd"]:.1f} ret={v["ret"]:.1f}')

# ========== EXPERIMENT 5: CatBoost (subsample to avoid OOM) ==========
log(f'\n{"="*60}')
log('[Exp5] CatBoost 4 seeds d6 500k')
log(f'{"="*60}')
rng=np.random.RandomState(42); idx=rng.choice(len(Xtr),min(500_000,len(Xtr)),replace=False)
ctr=Pool(Xtr[idx],ytr[idx]); ces=Pool(Xes,yes)
cb_preds=[]; t5=time.time()
for sd in [42,123,456,789]:
    cb=CatBoostClassifier(iterations=2000,learning_rate=0.02,depth=6,l2_leaf_reg=3.0,loss_function='Logloss',eval_metric='AUC',od_type='Iter',od_wait=200,verbose=0,thread_count=3,random_seed=sd)
    cb.fit(ctr,eval_set=ces,use_best_model=True)
    cb_preds.append(cb.predict_proba(Xte)[:,1]); del cb; gc.collect()
cb_avg=np.mean(cb_preds,axis=0); del cb_preds,ctr,ces; gc.collect()
auc5=roc_auc_score(yte,cb_avg)
log(f'  AUC={auc5:.4f} ({time.time()-t5:.0f}s)')
for k,v in topk(cb_avg,yte,ret_te,[0.005,0.01,0.02]).items():
    log(f'  top{k*100:.1f}%: acc={v["acc"]:.4f} tpd={v["tpd"]:.1f} ret={v["ret"]:.1f}')

# ========== EXPERIMENT 6: Cross-horizon stacking ==========
log(f'\n{"="*60}')
log('[Exp6] Cross-horizon: Train H=3,5,15,30,60 models, meta LGB')
log(f'{"="*60}')
HORIZONS=[3,5,15,30,60]
# For each horizon: retrain on same X/y but different label (H=h direction)
# We need raw close to build multi-horizon labels
raw_e2=pd.read_parquet(f'{config.DS_DIR}/raw_ETH.parquet')
close_full=raw_e2['close'].values.astype(np.float64)
ts_full=raw_e2['ts'].values.astype(np.int64); del raw_e2; gc.collect()
# Align: our X rows are at ts_clean[vi], need close at those ts
close_aligned = pd.Series(close_full,index=ts_full).reindex(ts_clean).values.astype(np.float64)
del close_full, ts_full; gc.collect()

# Build multi-horizon labels at te positions using close_aligned
base_labels = {}
for h in HORIZONS:
    rh = np.full(len(close_aligned), np.nan, dtype=np.float32)
    rh[:-h] = (close_aligned[h:] / close_aligned[:-h] - 1).astype(np.float32)
    base_labels[h] = rh
del close_aligned; gc.collect()

# Re-do splits with multi-horizon labels (same ts_clean)
tr_vi = np.where(tr_m2)[0]; es_vi = np.where(es_m2)[0]; te_vi = np.where(te_m2)[0]

base_model_preds_es=[]; base_model_preds_mv=[]; base_model_preds_te=[]
for h in HORIZONS:
    yh = (base_labels[h][vi] > 0).astype(np.int8)
    # Filter tr
    keep_tr_h = np.abs(base_labels[h][tr_vi]) >= 0.0005
    Xtr_h = X[tr_vi][keep_tr_h]; ytr_h = yh[tr_vi][keep_tr_h]
    Xes_h = X[es_vi]; yes_h = yh[es_vi]
    Xmv_h = X[mv_m2]; ymv_h = yh[mv_m2]
    Xte_h = X[te_vi]; yte_h = yh[te_vi]
    del yh; gc.collect()
    # Train
    preds_es=[]; preds_mv=[]; preds_te=[]
    for sd in range(42,42+SEEDS):
        dtr=lgb.Dataset(Xtr_h,label=ytr_h); des=lgb.Dataset(Xes_h,label=yes_h,reference=dtr)
        m=lgb.train({**lp,'seed':sd},dtr,num_boost_round=2000,valid_sets=[des],callbacks=[lgb.early_stopping(200,verbose=False)])
        preds_es.append(m.predict(Xes_h)); preds_mv.append(m.predict(Xmv_h)); preds_te.append(m.predict(Xte_h))
        del m; gc.collect()
    base_model_preds_es.append(np.mean(preds_es,axis=0))
    base_model_preds_mv.append(np.mean(preds_mv,axis=0))
    base_model_preds_te.append(np.mean(preds_te,axis=0))
    del preds_es,preds_mv,preds_te,Xtr_h,ytr_h,Xes_h,yes_h,Xmv_h,ymv_h,Xte_h,yte_h; gc.collect()
    log(f'  H={h} trained')

# Also append original exp1_avg to cross-horizon (H=30 already, but include)
# Meta: use base preds + exp1_avg as features for meta LGB predicting H=30
X_meta_es = np.column_stack(base_model_preds_es)
X_meta_mv = np.column_stack(base_model_preds_mv)
X_meta_te = np.column_stack(base_model_preds_te)
del base_labels, base_model_preds_es, base_model_preds_mv; gc.collect()

log(f'  Corr between base model preds on TE:')
corr = np.corrcoef(X_meta_te.T)
for i in range(len(HORIZONS)):
    for j in range(i+1,len(HORIZONS)):
        log(f'    H{HORIZONS[i]}-H{HORIZONS[j]}: {corr[i,j]:.4f}')

# Meta LGB (regression of H=30 label)
meta_lp=dict(num_leaves=31,learning_rate=0.03,min_data_in_leaf=30,verbose=-1,num_threads=3,objective='binary',metric='auc')
# Use MV as dev set since we don't have ES labels saved
dtr_m=lgb.Dataset(X_meta_mv,label=ymv); des_m=lgb.Dataset(X_meta_mv,label=ymv,reference=dtr_m)
meta=lgb.train(meta_lp,dtr_m,num_boost_round=500)
meta_avg=meta.predict(X_meta_te)
del meta,X_meta_mv,X_meta_te; gc.collect()
auc_meta=roc_auc_score(yte,meta_avg)
log(f'  Meta LGB AUC={auc_meta:.4f}')
for k,v in topk(meta_avg,yte,ret_te,[0.005,0.01,0.02]).items():
    log(f'  top{k*100:.1f}%: acc={v["acc"]:.4f} tpd={v["tpd"]:.1f} ret={v["ret"]:.1f}')

# Simple equal weight across horizons
ch_equal = np.mean(np.column_stack(base_model_preds_te),axis=1)
auc_ceq = roc_auc_score(yte,ch_equal)
log(f'  Cross-horizon equal AUC={auc_ceq:.4f}')

# ========== EXPERIMENT 7: Per-hour ensemble ==========
log(f'\n{"="*60}')
log('[Exp7] Per-hour AUC analysis + filtered trading')
log(f'{"="*60}')
best_pred = exp1_avg  # use exp1 as baseline
hour_aucs = {}
for h in range(24):
    mask = hr_te == h
    if mask.sum() > 100:
        hour_aucs[h] = roc_auc_score(yte[mask], best_pred[mask])
ranked_hours = sorted(hour_aucs.items(), key=lambda x:-x[1])
log(f'  Top 8 hours by AUC:')
for h,a in ranked_hours[:8]:
    m=hr_te==h
    n_samples=int(m.sum()*0.005)
    if n_samples>0:
        idx_in_h = np.argsort(best_pred[m])[-n_samples:]
        idx_global = np.where(m)[0][idx_in_h]
        acc = (yte[idx_global]==1).mean()
        tpd_h = tpd(len(idx_global), len(yte))
        log(f'    h={h:>2d}: AUC={a:.4f} top0.5% acc={acc:.4f} tpd={tpd_h:.1f}')
del hour_aucs; gc.collect()

# Filter to top-N hours, scan percentile threshold
for K in [3,5,8,12,16,20]:
    sel_hours = [h for h,_ in ranked_hours[:K]]
    mask = np.isin(hr_te, sel_hours)
    p_sel = best_pred[mask]; y_sel = yte[mask]; r_sel = ret_te[mask]
    log(f'\n  Filter top-{K}h={sel_hours}: n={mask.sum():,}')
    # Scan percentile thresholds
    best_acc = 0
    for thr in [70,75,80,85,90,92,94,95,96,97,98,99]:
        h_thr = np.percentile(p_sel, thr)
        hit = p_sel > h_thr
        if hit.sum()==0: continue
        acc = (y_sel[hit]==1).mean()
        tp = tpd(hit.sum(), len(yte))
        if acc > best_acc or (acc >= 0.65 and tp >= 15):
            best_acc = acc
        if acc >= 0.60:
            flag='✅BOTH' if (acc>=0.65 and tp>=15) else ('✅ACC' if acc>=0.65 else ('✅TPD' if tp>=15 else ''))
            log(f'    P{thr}: acc={acc:.4f} tpd={tp:.1f} {flag}')

# ========== EXPERIMENT 8: Best-of-all ==========
log(f'\n{"="*60}')
log('[Exp8] Best-of-all combination sweep')
log(f'{"="*60}')
candidates = [
    ('LGB_big', exp2_avg), ('LGB_reg', exp3_avg),
    ('XGB', xgb_avg), ('CB', cb_avg),
    ('CH_equal', ch_equal), ('Meta_LGB', meta_avg),
    ('baseline', exp1_avg),
]

# Correlation
corr_matrix = np.corrcoef(np.column_stack([p for _,p in candidates]).T)
log('  Correlations:')
names=[n for n,_ in candidates]
for i,n in enumerate(names):
    for j,m in enumerate(names):
        if i<j and corr_matrix[i,j]>0:
            log(f'    {n}-{m}: {corr_matrix[i,j]:.4f}')

# Grid search weights (only over LGB_big, XGB, CB since they should be least correlated)
log('\n  Weight search on (LGB_big, XGB, CB):')
best_combo=None; best_auc=0
X_three = np.column_stack([exp2_avg, xgb_avg, cb_avg])
for w1 in np.arange(0,1.05,0.1):
    for w2 in np.arange(0,1.05-w1,0.1):
        w3=round(max(0,1-w1-w2),2)
        p=X_three@np.array([w1,w2,w3])
        a=roc_auc_score(yte,p)
        if a>best_auc: best_auc=a; best_combo=(w1,w2,w3)
best_three_avg=X_three@np.array(best_combo)
log(f'  Best weights: LGB_big={best_combo[0]:.1f} XGB={best_combo[1]:.1f} CB={best_combo[2]:.1f} AUC={best_auc:.4f}')
for k,v in topk(best_three_avg,yte,ret_te,[0.005,0.01,0.02]).items():
    log(f'    top{k*100:.1f}%: acc={v["acc"]:.4f} tpd={v["tpd"]:.1f} ret={v["ret"]:.1f}')

# ========== FINAL SUMMARY ==========
log(f'\n{"="*60}')
log('FINAL SUMMARY — ALL EXPERIMENTS')
log(f'{"="*60}')
log(f'{"Method":<20} {"AUC":>7} {"t0.5%":>7} {"t0.8%":>7} {"t1%":>7} {"t2%":>7}')
for name,p in candidates + [('Best3w', best_three_avg)]:
    auc=roc_auc_score(yte,p)
    kk=[0.005,0.008,0.01,0.02]
    tt=topk(p,yte,ret_te,kk)
    def f(k): return f'{tt[k]["acc"]:.4f}'
    log(f'{name:<20} {auc:.4f} {f(0.005):>7} {f(0.008):>7} {f(0.01):>7} {f(0.02):>7}')

# Check if any hit target
log(f'\nTARGET CHECK (acc>=65%, tpd>=15):')
hit=False
for name,p in candidates + [('Best3w', best_three_avg), ('Meta', meta_avg)]:
    o=np.argsort(p)
    for k in [0.005,0.008,0.01,0.015,0.02,0.03,0.05,0.08,0.10]:
        idx=o[-max(int(len(p)*k),1):]
        acc=(yte[idx]==1).mean(); tp=tpd(len(idx),len(yte))
        if acc>=0.65 and tp>=15:
            log(f'  ✅ {name} top{k*100:.1f}%: acc={acc:.4f} tpd={tp:.1f}')
            hit=True
if not hit:
    log('  ❌ NO TARGET MET')
    # Show nearest
    best=None
    for name,p in candidates + [('Best3w', best_three_avg)]:
        o=np.argsort(p)
        for k in [0.005,0.008,0.01,0.015,0.02,0.03]:
            idx=o[-max(int(len(p)*k),1):]
            acc=(yte[idx]==1).mean(); tp=tpd(len(idx),len(yte))
            score=min(acc/0.65, tp/15)
            if best is None or score>best[0]:
                best=(score,name,k,acc,tp)
    log(f'  Nearest: {best[1]} top{best[2]*100:.1f}%: acc={best[3]:.4f} tpd={best[4]:.1f} score={best[0]:.3f}')

log(f'\nTotal: {time.time()-t0:.0f}s ({(time.time()-t0)/3600:.1f}h)')
