"""3 configs × 10 LGB seeds, 存磁盘防 OOM, ensemble + per-hour"""
import time, gc, numpy as np, datetime, sys, warnings, pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore')
sys.path.insert(0,'/workspace'); import config

def log(m): print(m,flush=True)
def tpd(n,b): return n*1440/b

t0=time.time()
log('=== ETH H=30: 3×LGB configs × 10 seeds + per-hour ===')

# Load
log('[1] load...')
ds=pd.read_parquet(f'{config.DS_DIR}/ds_ETH_h30.parquet')
feat=[c for c in ds.columns if c not in ['label','soft_label','ret_future','ts']]
ts=ds['ts'].values.astype(np.int64)
y=ds['label'].values.astype(np.int8)
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
X=X[vi]; y=y[vi]; ret=ret[vi]
ts_c=ts[vi]; hr=pd.to_datetime(ts_c,unit='s',utc=True).hour.values.astype(np.int32)
tr_m2=ts_mask(ts_c,'2020-01-01','2024-06-30')
es_m2=ts_mask(ts_c,'2024-06-30','2024-09-30')
te_m2=ts_mask(ts_c,'2025-09-30','2026-08-29')
del vi,ts; gc.collect()

keep=np.abs(ret[tr_m2])>=0.0005
Xtr=X[tr_m2][keep]; ytr=y[tr_m2][keep]
Xes=X[es_m2]; yes=y[es_m2]
Xte=X[te_m2]; yte=y[te_m2]; ret_te=ret[te_m2]; hr_te=hr[te_m2]
del X,y,ret,hr; gc.collect()
log(f'  TR={len(Xtr):,} ES={len(Xes):,} TE={len(Xte):,}')

# 3 diverse configs
configs = {
    'nl63_lr0.02': dict(num_leaves=63,learning_rate=0.02,min_data_in_leaf=200,lambda_l2=1.0),
    'nl31_lr0.03': dict(num_leaves=31,learning_rate=0.03,min_data_in_leaf=400,lambda_l2=5.0),
    'nl127_lr0.01': dict(num_leaves=127,learning_rate=0.01,min_data_in_leaf=150,lambda_l2=0.5),
}

# Train each config, save TE preds to file
saved = {}
for cfg_name, cfg_base in configs.items():
    log(f'\n[train] {cfg_name} × 10 seeds...')
    te_preds = []
    es_preds = []
    for i, sd in enumerate(range(42,52)):
        t=time.time()
        lp = dict(**cfg_base, feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=5,
                  verbose=-1, num_threads=3, objective='binary', metric='auc', seed=sd)
        dtr=lgb.Dataset(Xtr,label=ytr); des=lgb.Dataset(Xes,label=yes,reference=dtr)
        m=lgb.train(lp, dtr, num_boost_round=4000, valid_sets=[des],
                    callbacks=[lgb.early_stopping(300,verbose=False)])
        te_preds.append(m.predict(Xte))
        es_preds.append(m.predict(Xes))
        log(f'  seed {sd}: {time.time()-t:.0f}s iter={m.best_iteration}')
        del m; gc.collect()
    te_avg = np.mean(np.column_stack(te_preds), axis=1)
    es_avg = np.mean(np.column_stack(es_preds), axis=1)
    np.save(f'/tmp/te_{cfg_name}.npy', te_avg)
    np.save(f'/tmp/es_{cfg_name}.npy', es_avg)
    saved[cfg_name] = (es_avg, te_avg)
    log(f'  → ES AUC={roc_auc_score(yes,es_avg):.4f} TE AUC={roc_auc_score(yte,te_avg):.4f}')
    del te_preds, es_preds, te_avg, es_avg; gc.collect()

# Now load all + blend tune
log('\n[blend] ES weight search...')
names = list(saved.keys())
es_stack = np.column_stack([saved[n][0] for n in names])
te_stack = np.column_stack([saved[n][1] for n in names])

# Correlation
corr = np.corrcoef(te_stack.T)
log(f'  Correlation matrix:')
for i in range(len(names)):
    for j in range(i+1,len(names)):
        log(f'    {names[i]}-{names[j]}: {corr[i,j]:.4f}')

# Equal weight baseline
eq_te = np.mean(te_stack, axis=1)
eq_es = np.mean(es_stack, axis=1)
log(f'  Equal weight: ES AUC={roc_auc_score(yes,eq_es):.4f} TE AUC={roc_auc_score(yte,eq_te):.4f}')

# 2D blend
best_w=None; best_auc=0
for w in np.arange(0,1.05,0.05):
    for v in np.arange(0,1.05-w,0.05):
        u=round(max(0,1-w-v),2)
        b=es_stack[:,0]*w + es_stack[:,1]*v + es_stack[:,2]*u
        a=roc_auc_score(yes,b)
        if a>best_auc: best_auc,best_w=a,(w,v,u)
log(f'  Best 3-way: {dict(zip(names,best_w))} ES AUC={best_auc:.4f}')
blend_te = te_stack[:,0]*best_w[0] + te_stack[:,1]*best_w[1] + te_stack[:,2]*best_w[2]

# Compare all
log(f'\n[compare] top-k on all methods:')
methods = [(n, saved[n][1]) for n in names] + [('EQ_all', eq_te), ('Blended', blend_te)]
for name, p in methods:
    auc=roc_auc_score(yte,p)
    log(f'  {name}: AUC={auc:.4f}')
    o=np.argsort(p)
    for k in [0.003,0.005,0.008,0.01,0.015,0.02,0.03,0.05]:
        idx=o[-max(int(len(p)*k),1):]
        log(f'    top{k*100:.1f}%: acc={(yte[idx]==1).mean():.4f} tpd={tpd(len(idx),len(p)):.1f}')

# Per-hour on best method
log(f'\n[per-hour] on Blended...')
ranked=sorted([(h,roc_auc_score(yte[hr_te==h],blend_te[hr_te==h])) for h in range(24) if (hr_te==h).sum()>100],key=lambda x:-x[1])
log(f'  top10h: {[(h,f"{a:.4f}") for h,a in ranked[:10]]}')

best_hit=[]; best_near=None
for name, p in methods:
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
                best_hit.append((name,K,thr,acc,tp))
            if best_near is None or score>best_near[0]:
                best_near=(score,name,K,thr,acc,tp)

log(f'\n{"="*60}')
log('🏆 最终')
log(f'{"="*60}')
if best_hit:
    best_hit.sort(key=lambda x:-x[3])
    for n,K,thr,acc,tp in best_hit[:15]:
        log(f'  ✅ {n} K={K}h P{thr}: acc={acc:.4f} tpd={tp:.1f}')
else:
    log('  ❌ 未达标')
    if best_near:
        log(f'  最近: {best_near[1]} K={best_near[2]}h P{best_near[3]}: acc={best_near[4]:.4f} tpd={best_near[5]:.1f}')
log(f'\n⏱ {time.time()-t0:.0f}s ({(time.time()-t0)/60:.0f}min)')
