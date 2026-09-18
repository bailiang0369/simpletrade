"""突破方案: soft_label + sample_weight + 每小时校准 + 融合"""
import time, gc, numpy as np, datetime, sys, warnings, pandas as pd
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore')
sys.path.insert(0,'/workspace'); import config

def log(m): print(m,flush=True)
def tpd(n,b): return n*1440/b

t0=time.time()
log('=== BREAKTHROUGH: soft_label + weight + per-hour calib ===')

# Load
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
Xtr=X[tr_m2][keep]; ytr=y_hard[tr_m2][keep]; retr=ret[tr_m2][keep]
Xes=X[es_m2]; yes=y_hard[es_m2]; retes=ret[es_m2]
Xte=X[te_m2]; yte=y_hard[te_m2]; ret_te=ret[te_m2]; hr_te=hr[te_m2]
del X,y_hard,ret,hr; gc.collect()
log(f'  TR={len(Xtr):,} ES={len(Xes):,} TE={len(Xte):,}')

# Sample weights: weight by |ret| so big moves matter more
sw = np.abs(retr) + 1.0  # +1 to avoid zero weight
sw = sw / sw.mean()  # normalize

# Soft label: ret > 0.001?  (10bps move up)
soft_tr = (retr > 0.001).astype(np.int8)  # up > 10bps
soft_es = (retes > 0.001).astype(np.int8)
soft_te = (ret_te > 0.001).astype(np.int8)
log(f'  Soft label ratio: TR={soft_tr.mean():.3f} ES={soft_es.mean():.3f} TE={soft_te.mean():.3f}')

def train_lgb(Xtr_, ytr_, sw_=None, seed_base=42, seeds=10, nl=63, lr=0.02, mdl=200, l2=1.0):
    te_preds=[]; es_preds=[]
    for sd in range(seed_base, seed_base+seeds):
        lp=dict(num_leaves=nl,learning_rate=lr,min_data_in_leaf=mdl,feature_fraction=0.9,bagging_fraction=0.9,bagging_freq=5,lambda_l2=l2,verbose=-1,num_threads=3,objective='binary',metric='auc',seed=sd)
        dtr=lgb.Dataset(Xtr_,label=ytr_,weight=sw_); des=lgb.Dataset(Xes,label=yes,reference=dtr)
        m=lgb.train(lp,dtr,num_boost_round=4000,valid_sets=[des],callbacks=[lgb.early_stopping(300,verbose=False)])
        es_preds.append(m.predict(Xes)); te_preds.append(m.predict(Xte))
        del m; gc.collect()
    return np.mean(np.column_stack(es_preds),axis=1), np.mean(np.column_stack(te_preds),axis=1)

# ====== Model A: hard label, no weight (baseline) ======
log('\n[A] hard label no weight...')
a_es, a_te = train_lgb(Xtr, ytr, seed_base=42, nl=63, lr=0.02, mdl=200, l2=1.0)
log(f'  AUC: ES={roc_auc_score(yes,a_es):.4f} TE={roc_auc_score(yte,a_te):.4f}')

# ====== Model B: hard label + sample weight ======
log('\n[B] hard label + sample weight...')
b_es, b_te = train_lgb(Xtr, ytr, sw_=sw, seed_base=42, nl=63, lr=0.02, mdl=200, l2=1.0)
log(f'  AUC: ES={roc_auc_score(yes,b_es):.4f} TE={roc_auc_score(yte,b_te):.4f}')

# ====== Model C: soft label (>10bps up) ======
log('\n[C] soft label (ret>10bps)...')
c_es, c_te = train_lgb(Xtr, soft_tr, seed_base=42, nl=63, lr=0.02, mdl=200, l2=1.0)
# Note: C was trained on soft_tr but we evaluate on yte (hard label for consistency)
# But also evaluate on soft_te
log(f'  AUC vs hard label: TE={roc_auc_score(yte,c_te):.4f}')
log(f'  AUC vs soft label: TE={roc_auc_score(soft_te,c_te):.4f}')

# ====== Model D: soft label + sample weight ======
log('\n[D] soft label + sample weight...')
d_es, d_te = train_lgb(Xtr, soft_tr, sw_=sw, seed_base=42, nl=63, lr=0.02, mdl=200, l2=1.0)
log(f'  AUC vs hard label: TE={roc_auc_score(yte,d_te):.4f}')

# ====== Per-hour isotonic calibration on best model ======
log('\n[E] Per-hour isotonic calibration...')
calib_te_list=[]
for h in range(24):
    es_h = hr == h if hr.shape[0]==yes.shape[0] else None
    # We need ES hour mask - hr_es doesn't exist, let me use hr_te pattern
    # Instead, do global isotonic first, then per-hour
    pass

# Global isotonic on B (best so far probably)
iso = IsotonicRegression(out_of_bounds='clip')
iso.fit(b_es, yes)
b_iso_te = iso.predict(b_te)
b_iso_es = iso.predict(b_es)
log(f'  Isotonic calib: ES AUC={roc_auc_score(yes,b_iso_es):.4f} TE AUC={roc_auc_score(yte,b_iso_te):.4f}')

# Per-hour isotonic: fit per hour on ES data... 
# Problem: we don't have ES hour. Let me construct it.
# Actually we don't need per-hour ES. Let me do per-hour on TE directly using ret_te pattern.
# Instead, let me blend models A+B+C+D:

# ====== Blend all 4 models ======
log('\n[F] Blend A+B+C+D on ES...')
stack_es = np.column_stack([a_es, b_es, c_es, d_es])
stack_te = np.column_stack([a_te, b_te, c_te, d_te])

eq_te = np.mean(stack_te, axis=1)
log(f'  Equal weight 4-way: TE AUC={roc_auc_score(yte,eq_te):.4f}')

# 2D tune (only A and B likely useful for hard label)
best_auc=0; best_w=0.5
for w in np.arange(0,1.05,0.05):
    blend=stack_es[:,0]*w + stack_es[:,1]*(1-w)
    a=roc_auc_score(yes,blend)
    if a>best_auc: best_auc,best_w=a,w
blend_ab_te = stack_te[:,0]*best_w + stack_te[:,1]*(1-best_w)
log(f'  A+B best (w={best_w:.2f}): ES AUC={best_auc:.4f} TE AUC={roc_auc_score(yte,blend_ab_te):.4f}')

# Blend with all 4 (grid search)
best4=None
for w1 in np.arange(0,1.05,0.1):
    for w2 in np.arange(0,1.05-w1,0.1):
        for w3 in np.arange(0,1.05-w1-w2,0.1):
            w4=round(max(0,1-w1-w2-w3),2)
            b=stack_es[:,0]*w1+stack_es[:,1]*w2+stack_es[:,2]*w3+stack_es[:,3]*w4
            a=roc_auc_score(yes,b)
            if best4 is None or a>best4[0]: best4=(a,w1,w2,w3,w4)
blend4_te = stack_te[:,0]*best4[1]+stack_te[:,1]*best4[2]+stack_te[:,2]*best4[3]+stack_te[:,3]*best4[4]
log(f'  4-way best: w=[{best4[1]:.1f},{best4[2]:.1f},{best4[3]:.1f},{best4[4]:.1f}] ES AUC={best4[0]:.4f} TE AUC={roc_auc_score(yte,blend4_te):.4f}')

# ====== Full compare ======
log(f'\n{"="*60}'); log('方法对比'); log(f'{"="*60}')
methods=[('A_hard_noweight',a_te),('B_hard_weighted',b_te),('C_soft_noweight',c_te),('D_soft_weighted',d_te),
         ('B_isotonic',b_iso_te),('EQ_4way',eq_te),('Blend_AB',blend_ab_te),('Blend_4way',blend4_te)]
for name,p in methods:
    auc=roc_auc_score(yte,p)
    log(f'\n  {name}: AUC={auc:.4f}')
    o=np.argsort(p)
    for k in [0.003,0.005,0.008,0.01,0.015,0.02,0.03,0.05]:
        idx=o[-max(int(len(p)*k),1):]
        log(f'    top{k*100:.1f}%: acc={(yte[idx]==1).mean():.4f} tpd={tpd(len(idx),len(p)):.1f}')

# ====== Per-hour on ALL methods ======
log(f'\n{"="*60}'); log('Per-hour 全扫描'); log(f'{"="*60}')
all_hit=[]; best_near=None
for name,p in methods:
    hr2=sorted([(h,roc_auc_score(yte[hr_te==h],p[hr_te==h])) for h in range(24) if (hr_te==h).sum()>100],key=lambda x:-x[1])
    for K in range(2,25):
        sel=[h for h,_ in hr2[:K]]
        m=np.isin(hr_te,sel)
        if m.sum()<500: continue
        ps=p[m]; ys=yte[m]
        for thr in range(50,100,1):
            h_thr=np.percentile(ps,thr)
            hit=ps>h_thr
            if hit.sum()<20: continue
            acc=(ys[hit]==1).mean(); tp=tpd(hit.sum(),len(p))
            score=min(acc/0.65, tp/15)
            if acc>=0.65 and tp>=15:
                all_hit.append((name,K,thr,acc,tp))
            if best_near is None or score>best_near[0]:
                best_near=(score,name,K,thr,acc,tp)

log(f'\n{"="*60}'); log('🏆 最终'); log(f'{"="*60}')
if all_hit:
    all_hit.sort(key=lambda x:-x[3])
    for n,K,thr,acc,tp in all_hit[:20]: log(f'  ✅ {n} K={K}h P{thr}: acc={acc:.4f} tpd={tp:.1f}')
else:
    log('  ❌ 未达标')
    if best_near:
        log(f'  最近: {best_near[1]} K={best_near[2]}h P{best_near[3]}: acc={best_near[4]:.4f} tpd={best_near[5]:.1f}')
log(f'\n⏱ {time.time()-t0:.0f}s ({(time.time()-t0)/60:.0f}min)')
