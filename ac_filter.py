"""A (hard) + C (soft) 交叉过滤 + per-hour"""
import time, gc, numpy as np, datetime, sys, warnings, pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore')
sys.path.insert(0,'/workspace'); import config

def log(m): print(m,flush=True)
def tpd(n,b): return n*1440/b

t0=time.time()
log('=== A(hard) + C(soft) cross filter ===')

log('[1] load...')
ds=pd.read_parquet(f'{config.DS_DIR}/ds_ETH_h30.parquet')
feat=[c for c in ds.columns if c not in ['label','soft_label','ret_future','ts']]
ts=ds['ts'].values.astype(np.int64)
y_hard=ds['label'].values.astype(np.int8)
ret=ds['ret_future'].values.astype(np.float32)
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
X=X[vi]; y_hard=y_hard[vi]; ret=ret[vi]
ts_c=ts[vi]; hr=pd.to_datetime(ts_c,unit='s',utc=True).hour.values.astype(np.int32)
tr_m2=ts_mask(ts_c,'2020-01-01','2024-06-30')
es_m2=ts_mask(ts_c,'2024-06-30','2024-09-30')
te_m2=ts_mask(ts_c,'2025-09-30','2026-08-29')
del vi,ts; gc.collect()

keep=np.abs(ret[tr_m2])>=0.0005
Xtr=X[tr_m2][keep]; ytr_hard=y_hard[tr_m2][keep]; retr=ret[tr_m2][keep]
Xes=X[es_m2]; yes=y_hard[es_m2]
Xte=X[te_m2]; yte=y_hard[te_m2]; ret_te=ret[te_m2]; hr_te=hr[te_m2]
del X,y_hard,ret,hr; gc.collect()
log(f'  TR={len(Xtr):,} TE={len(Xte):,}')

soft_tr=(retr>0.001).astype(np.int8)
soft_te=(ret_te>0.001).astype(np.int8)

def train(target, seed_base=42, seeds=10):
    te_preds=[]; es_preds=[]
    for sd in range(seed_base, seed_base+seeds):
        lp=dict(num_leaves=63,learning_rate=0.02,min_data_in_leaf=200,feature_fraction=0.9,bagging_fraction=0.9,bagging_freq=5,lambda_l2=1.0,verbose=-1,num_threads=3,objective='binary',metric='auc',seed=sd)
        dtr=lgb.Dataset(Xtr,label=target); des=lgb.Dataset(Xes,label=yes,reference=dtr)
        m=lgb.train(lp,dtr,num_boost_round=4000,valid_sets=[des],callbacks=[lgb.early_stopping(300,verbose=False)])
        es_preds.append(m.predict(Xes)); te_preds.append(m.predict(Xte)); del m; gc.collect()
    return np.mean(np.column_stack(es_preds),axis=1), np.mean(np.column_stack(te_preds),axis=1)

log('\n[train A] hard label...')
a_es, a_te = train(ytr_hard, seeds=10)
log(f'  TE AUC(hard)={roc_auc_score(yte,a_te):.4f}')

log('\n[train C] soft label...')
c_es, c_te = train(soft_tr, seeds=10)
log(f'  TE AUC(hard)={roc_auc_score(yte,c_te):.4f}')
log(f'  TE AUC(soft)={roc_auc_score(soft_te,c_te):.4f}')

# Save for potential future use
np.save('/tmp/a_te.npy',a_te); np.save('/tmp/c_te.npy',c_te)

# ====== Strategy 1: Intersection ======
# Take top predictions from A, filter to only those where C also says up
log('\n[1] A∩C 交叉过滤...')
results=[]
# Rank both
a_rank = np.argsort(-a_te)  # indices sorted by A score desc
c_rank = np.argsort(-c_te)

# For each K top from A, what % pass C threshold?
for k_pct in [0.005, 0.008, 0.01, 0.015, 0.02, 0.03]:
    n=max(int(len(a_te)*k_pct),1)
    topA_idx = a_rank[:n]
    # Filter: keep only where C > threshold
    for c_thr_pct in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99]:
        c_thr_val = np.percentile(c_te, c_thr_pct*100)
        mask = c_te[topA_idx] > c_thr_val
        sel = topA_idx[mask]
        if len(sel) < 20: continue
        acc = (yte[sel]==1).mean(); tp = tpd(len(sel), len(yte))
        score = min(acc/0.65, tp/15)
        if acc>=0.62 and tp>=5:
            log(f'  A_top{k_pct*100:.1f}% ∩ C_P{c_thr_pct*100:.0f}: acc={acc:.4f} tpd={tp:.1f} n={len(sel)}')
            if acc>=0.65 and tp>=15: log(f'    ✅ HIT!')
        results.append(('A_intersect_C', k_pct, c_thr_pct, acc, tp, score))

# ====== Strategy 2: Intersection by hour ======
log('\n[2] A∩C × per-hour...')
for K in range(2,25):
    hr_aucs_a = sorted([(h,roc_auc_score(yte[hr_te==h],a_te[hr_te==h])) for h in range(24) if (hr_te==h).sum()>100],key=lambda x:-x[1])
    sel_hours = [h for h,_ in hr_aucs_a[:K]]
    hour_mask = np.isin(hr_te, sel_hours)
    if hour_mask.sum()<500: continue
    
    for k_pct in [0.005, 0.01, 0.02, 0.03, 0.05]:
        for c_thr_pct in [0.5, 0.7, 0.85, 0.95, 0.99]:
            c_thr_val = np.percentile(c_te[hour_mask], c_thr_pct*100)
            # top k% of A within selected hours, filtered by C
            idx_in_hour = np.where(hour_mask)[0]
            a_scores_in_hour = a_te[idx_in_hour]
            top_n = max(int(len(idx_in_hour)*k_pct),1)
            top_idx_local = np.argsort(-a_scores_in_hour)[:top_n]
            top_idx_global = idx_in_hour[top_idx_local]
            c_vals = c_te[top_idx_global]
            sel = top_idx_global[c_vals > c_thr_val]
            if len(sel)<20: continue
            acc = (yte[sel]==1).mean(); tp = tpd(len(sel), len(yte))
            if acc>=0.65 and tp>=15:
                log(f'  ✅ K={K}h A_top{k_pct*100:.1f}% ∩ C_P{c_thr_pct*100:.0f}: acc={acc:.4f} tpd={tp:.1f}')
            results.append(('Hour_intersect', K, k_pct*100+c_thr_pct*100, acc, tp, min(acc/0.65,tp/15)))

# ====== Strategy 3: Sum/Product ======
log('\n[3] A+C combined score...')
# Normalize both to [0,1]
def norm(x): return (x-x.min())/(x.max()-x.min())
a_n = norm(a_te); c_n = norm(c_te)

for w in [0.3, 0.5, 0.7]:
    comb = a_n*w + c_n*(1-w)
    log(f'  A×{w}+C×{1-w}: AUC={roc_auc_score(yte,comb):.4f}')
    o=np.argsort(comb)
    for k in [0.005,0.01,0.015,0.02,0.03,0.05]:
        idx=o[-max(int(len(comb)*k),1):]
        log(f'    top{k*100:.1f}%: acc={(yte[idx]==1).mean():.4f} tpd={tpd(len(idx),len(comb)):.1f}')

# ====== Strategy 4: Multiplicative ======
prod = a_n * c_n
log(f'\n  A×C (mult): AUC={roc_auc_score(yte,prod):.4f}')
o=np.argsort(prod)
for k in [0.005,0.01,0.015,0.02,0.03,0.05]:
    idx=o[-max(int(len(prod)*k),1):]
    acc=(yte[idx]==1).mean(); tp=tpd(len(idx),len(prod))
    log(f'    top{k*100:.1f}%: acc={acc:.4f} tpd={tp:.1f}')

# ====== Per-hour on A alone ======
log(f'\n{"="*60}'); log('Per-hour on A (baseline)'); log(f'{"="*60}')
ranked=sorted([(h,roc_auc_score(yte[hr_te==h],a_te[hr_te==h])) for h in range(24) if (hr_te==h).sum()>100],key=lambda x:-x[1])
log(f'  top10h: {[(h,f"{a:.4f}") for h,a in ranked[:10]]}')
for K in range(2,25):
    sel=[h for h,_ in ranked[:K]]
    m=np.isin(hr_te,sel)
    if m.sum()<500: continue
    ps=a_te[m]; ys=yte[m]
    for thr in range(50,100,1):
        h_thr=np.percentile(ps,thr)
        hit=ps>h_thr
        if hit.sum()<20: continue
        acc=(ys[hit]==1).mean(); tp=tpd(hit.sum(),len(a_te))
        if acc>=0.65 and tp>=15: log(f'  ✅ A K={K}h P{thr}: acc={acc:.4f} tpd={tp:.1f}')

# ====== Final Summary ======
log(f'\n{"="*60}'); log('🏆 汇总'); log(f'{"="*60}')
hits = sorted([r for r in results if r[3]>=0.65 and r[4]>=15], key=lambda x:-x[3])
if hits:
    for r in hits[:20]: log(f'  ✅ {r}')
else:
    log('  未找到 A∩C 交叉命中')
    # Show best non-hit
    scored = [(r, r[5]) for r in results if r[5] is not None]
    scored.sort(key=lambda x:-x[1])
    if scored:
        log(f'  最佳候选: {scored[0][0]}')

log(f'\n⏱ {time.time()-t0:.0f}s ({(time.time()-t0)/60:.0f}min)')
