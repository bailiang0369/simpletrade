"""快速对比: LGB baseline vs +funding vs +CatBoost vs +ensemble
不 cross-asset, 专注 BTC 自身特征 + 新模型
"""
import numpy as np, pandas as pd, time, gc, sys, warnings, os
import lightgbm as lgb
import catboost as cb
import polars as pl
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore'); sys.stdout.reconfigure(line_buffering=True)

T0=time.time(); PAR='data/datasets'; NPY='data/splits_npy'
SEEDS=[42,49,56,63,70]; TRAIN_END=1722556800; ES_END=1725148800; META_END=1759363200
def log(*a): print(' '.join(str(x) for x in a), flush=True)

# ========== Step 0: Build new BTC h15 dataset WITH funding features ==========
log("[0] Build BTC h15 + funding features...")
raw = pl.read_parquet(f'{PAR}/raw_BTC.parquet').sort('ts')
# Load base ds features
import features as F
feats_pl = F.build_features(raw)
log(f"  base feats from features.py: {feats_pl.width}")

# Now add funding features using polars
fund = raw['funding'].to_numpy().astype(np.float64)
fund_rolling = pd.Series(fund)  # easier rolling with pandas

f_new = {}
# Raw funding values (注意 funding 是 8h 更新但 1min 频率)
f_new['funding'] = fund.astype(np.float32)
# z-scores 不同窗口
for w in (60, 120, 360, 1440):  # 1h, 2h, 6h, 24h in minutes
    mu = fund_rolling.rolling(w, min_periods=5).mean().to_numpy()
    sd = fund_rolling.rolling(w, min_periods=5).std().to_numpy()
    f_new[f'funding_z{w}'] = np.where(sd>1e-9, (fund-mu)/sd, 0.0).astype(np.float32)
# funding trend (change over windows)
for w in (60, 360, 1440):
    f_new[f'funding_d{w}'] = (fund - np.roll(fund, w)).astype(np.float32)
# funding extremes
f_new['funding_hi_flag'] = (fund > 30).astype(np.float32)  # extreme bull funding (~3σ)
f_new['funding_lo_flag'] = (fund < -30).astype(np.float32)
# funding moving avg vs raw
for w in (60, 360):
    f_new[f'funding_ma{w}'] = fund_rolling.rolling(w, min_periods=5).mean().to_numpy().astype(np.float32)

log(f"  funding feats: {len(f_new)}")

# Build labels same as ds_h15
close = raw['close'].to_numpy()
label = np.zeros(len(close), dtype=np.int8)
h=15
label[:-h] = (close[h:] / close[:-h] - 1 > 0).astype(np.int8)
ret_future = np.full(len(close), np.nan, dtype=np.float32)
ret_future[:-h] = (close[h:] / close[:-h] - 1).astype(np.float32)

# Merge features
base_feats = feats_pl.to_numpy()  # already float32
new_feats = np.stack([f_new[k] for k in sorted(f_new.keys())], axis=1).astype(np.float32)
log(f"  base={base_feats.shape} funding={new_feats.shape}")

# Merge base + funding
X_all = np.hstack([base_feats, new_feats]).astype(np.float32)
log(f"  combined: {X_all.shape}")
del base_feats, new_feats, f_new, fund_rolling; gc.collect()

# Null mask (warmup + label end)
feat_valid = ~np.isnan(X_all).any(axis=1)
ret_valid = ~np.isnan(ret_future)
ok = feat_valid & ret_valid
log(f"  valid rows: {ok.sum():,} (drop {(~ok).sum():,})")

X_all = X_all[ok]; label = label[ok]; ret_future = ret_future[ok]
ts_all = raw['ts'].to_numpy()[ok].astype(np.int64)
del raw, feats_pl; gc.collect()

# Split
def save_split(split, tlo, thi, X, y, ret, ts_arr):
    m = (ts_arr>=tlo)&(ts_arr<thi)
    np.save(f'{NPY}/BTC_h15f_{split}_X.npy', X[m])
    np.save(f'{NPY}/BTC_h15f_{split}_y.npy', y[m])
    np.save(f'{NPY}/BTC_h15f_{split}_ret.npy', ret[m])
    return m.sum()

os.makedirs(NPY, exist_ok=True)
for split, tlo, thi in [('train',0,TRAIN_END),('early_stop',TRAIN_END,ES_END),
                         ('meta_val',ES_END,META_END),('test',META_END,10**18)]:
    n = save_split(split, tlo, thi, X_all, label, ret_future, ts_all)
    log(f'  {split}: {n:,}')

# Also save base-only split for fair comparison
# Use existing npy to know the correct order... actually let's just compare both side by side
log(f"\n[0.5] BTC h15 + funding feats saved. Now train 5-way.")

# ========== Step 1: Train 5-way ==========
def load(sym, h, suff):
    X=np.load(f'{NPY}/{sym}_h{h}{suff}_train_X.npy').astype(np.float32)
    y=np.load(f'{NPY}/{sym}_h{h}{suff}_train_y.npy').astype(np.int32)
    Xes=np.load(f'{NPY}/{sym}_h{h}{suff}_early_stop_X.npy').astype(np.float32)
    yes=np.load(f'{NPY}/{sym}_h{h}{suff}_early_stop_y.npy').astype(np.int32)
    Xte=np.load(f'{NPY}/{sym}_h{h}{suff}_test_X.npy').astype(np.float32)
    yte=np.load(f'{NPY}/{sym}_h{h}{suff}_test_y.npy').astype(np.int32)
    return X,y,Xes,yes,Xte,yte

def run_lgb(Xtr,ytr,Xes,yes,Xte,yte,sw=None,label=""):
    params={'objective':'binary','metric':'auc','learning_rate':0.05,'num_leaves':63,
            'min_child_samples':200,'feature_fraction':0.8,'bagging_fraction':0.8,
            'bagging_freq':5,'lambda_l2':0.1,'verbose':-1,'n_jobs':4}
    pte,pes=[],[]; t0=time.time()
    for sd in SEEDS:
        params['seed']=sd
        tr=lgb.Dataset(Xtr,label=ytr,weight=sw); es=lgb.Dataset(Xes,label=yes,reference=tr)
        m=lgb.train(params,tr,5000,[es],callbacks=[lgb.early_stopping(100),lgb.log_evaluation(0)])
        pte.append(m.predict(Xte)); pes.append(m.predict(Xes))
    aes=roc_auc_score(yes,np.mean(pes,0)); ate=roc_auc_score(yte,np.mean(pte,0))
    log(f"  LGB {label:25s}: es={aes:.4f} te={ate:.4f} ({time.time()-t0:.0f}s)")
    return aes,ate, np.mean(pte,0)

def run_cb(Xtr,ytr,Xes,yes,Xte,yte,label=""):
    pte,pes=[],[]; t0=time.time()
    for sd in SEEDS:
        m=cb.CatBoostClassifier(iterations=2000, learning_rate=0.05, depth=6,
                                l2_leaf_reg=3, random_seed=sd, verbose=0,
                                early_stopping_rounds=100, task_type='CPU',
                                thread_count=4)
        m.fit(Xtr,ytr,eval_set=(Xes,yes), use_best_model=True)
        pte.append(m.predict_proba(Xte)[:,1]); pes.append(m.predict_proba(Xes)[:,1])
    aes=roc_auc_score(yes,np.mean(pes,0)); ate=roc_auc_score(yte,np.mean(pte,0))
    log(f"  CB  {label:25s}: es={aes:.4f} te={ate:.4f} ({time.time()-t0:.0f}s)")
    return aes,ate, np.mean(pte,0)

log("\n[1] Train 5 models...")

# Compute negw weights
ret_b=np.load(f'{NPY}/BTC_h15_train_ret.npy')
sw_b=np.where(np.abs(ret_b)>=np.quantile(np.abs(ret_b),0.90),0.3,1.0).astype(np.float32)
ret_f=np.load(f'{NPY}/BTC_h15f_train_ret.npy')
sw_f=np.where(np.abs(ret_f)>=np.quantile(np.abs(ret_f),0.90),0.3,1.0).astype(np.float32)

Xtr_b,ytr_b,Xes_b,yes_b,Xte_b,yte_b = load('BTC',15,'')
Xtr_f,ytr_f,Xes_f,yes_f,Xte_f,yte_f = load('BTC',15,'f_')

results={}
# 1) baseline LGB
_,_,pte_base = run_lgb(Xtr_b,ytr_b,Xes_b,yes_b,Xte_b,yte_b,None,    'baseline LGB 56')
# 2) baseline LGB + negw
_,_,pte_base_nw = run_lgb(Xtr_b,ytr_b,Xes_b,yes_b,Xte_b,yte_b,sw_b,  'baseline+negw')
# 3) funding LGB
_,_,pte_fund = run_lgb(Xtr_f,ytr_f,Xes_f,yes_f,Xte_f,yte_f,None,     'funding LGB 73')
# 4) funding LGB + negw
_,_,pte_fund_nw = run_lgb(Xtr_f,ytr_f,Xes_f,yes_f,Xte_f,yte_f,sw_f,  'funding+negw')
# 5) funding CatBoost
_,_,pte_cb = run_cb(Xtr_f,ytr_f,Xes_f,yes_f,Xte_f,yte_f,             'funding CB')

# ========== Step 2: Ensemble ==========
log("\n[2] Ensembles...")
y_test = yte_f
names = ['baseline','baseline+negw','funding','funding+negw','fundingCB']
ptes = [pte_base, pte_base_nw, pte_fund, pte_fund_nw, pte_cb]
# Simple avg of all 5
p_avg = np.mean(np.column_stack(ptes), axis=1)
a_avg = roc_auc_score(y_test, p_avg)
log(f"  all 5 avg: te={a_avg:.4f}")
# Just best 2 (negw+CB)
p_best2 = (pte_fund_nw + pte_cb)/2
log(f"  funding+negw + CB: te={roc_auc_score(y_test,p_best2):.4f}")
# All LGB
p_lgb_avg = np.mean(np.column_stack([pte_base, pte_base_nw, pte_fund, pte_fund_nw]), axis=1)
log(f"  all LGB avg: te={roc_auc_score(y_test,p_lgb_avg):.4f}")

# Summary
log(f"\n{'='*55}\nFINAL SUMMARY  (ETH h15 ref te=0.5432)\n{'='*55}")
best_te = max(roc_auc_score(y_test,p) for p in ptes)
log(f"  Single model best: {best_te:.4f}  gap={0.5432-best_te:.4f}")
log(f"  Ensemble best:     {max(a_avg, roc_auc_score(y_test,p_best2), roc_auc_score(y_test,p_lgb_avg)):.4f}")
log(f"\nTOTAL: {time.time()-T0:.0f}s")
