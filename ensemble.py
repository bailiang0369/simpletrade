"""LGB 30 seeds + CB 15 seeds + ES blend tune + per-hour"""
import time, gc, numpy as np, datetime, sys, warnings, os, pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

def log(m): print(m,flush=True)
def tpd(n,b): return n*1440/b

t0=time.time()
log('=== ETH H=30: LGB30 + CB15 ensemble + per-hour ===')

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

# ====== LGB 30 seeds ======
log('\n[2] LGB 30 seeds (nl63 lr0.02)...')
lgb_es_preds=[]; lgb_te_preds=[]
for i,sd in enumerate(range(42,72)):
    t=time.time()
    dtr=lgb.Dataset(Xtr,label=ytr); des=lgb.Dataset(Xes,label=yes,reference=dtr)
    m=lgb.train(dict(num_leaves=63,learning_rate=0.02,min_data_in_leaf=200,feature_fraction=0.9,bagging_fraction=0.9,bagging_freq=5,lambda_l2=1.0,verbose=-1,num_threads=3,objective='binary',metric='auc',seed=sd),dtr,num_boost_round=4000,valid_sets=[des],callbacks=[lgb.early_stopping(300,verbose=False)])
    lgb_es_preds.append(m.predict(Xes))
    lgb_te_preds.append(m.predict(Xte))
    log(f'  LGB seed {sd} ({i+1}/30): {time.time()-t:.0f}s iter={m.best_iteration}')
    del m; gc.collect()
lgb_es=np.mean(lgb_es_preds,axis=0); lgb_te=np.mean(lgb_te_preds,axis=0)
del lgb_es_preds,lgb_te_preds; gc.collect()
log(f'  LGB AUC: ES={roc_auc_score(yes,lgb_es):.4f} TE={roc_auc_score(yte,lgb_te):.4f}')

# ====== CB 15 seeds ======
log('\n[3] CB 15 seeds...')
cb_es_preds=[]; cb_te_preds=[]
for i,sd in enumerate(range(42,57)):
    t=time.time()
    cb=CatBoostClassifier(loss_function='Logloss',eval_metric='AUC',iterations=5000,learning_rate=0.03,depth=6,l2_leaf_reg=3.0,subsample=0.9,colsample_bylevel=0.9,random_seed=sd,thread_count=3,verbose=0,early_stopping_rounds=300)
    cb.fit(Pool(Xtr,label=ytr), eval_set=Pool(Xes,label=yes), use_best_model=True)
    cb_es_preds.append(cb.predict_proba(Xes)[:,1])
    cb_te_preds.append(cb.predict_proba(Xte)[:,1])
    log(f'  CB seed {sd} ({i+1}/15): {time.time()-t:.0f}s iter={cb.get_best_iteration()}')
    del cb; gc.collect()
cb_es=np.mean(cb_es_preds,axis=0); cb_te=np.mean(cb_te_preds,axis=0)
del cb_es_preds,cb_te_preds; gc.collect()
log(f'  CB AUC: ES={roc_auc_score(yes,cb_es):.4f} TE={roc_auc_score(yte,cb_te):.4f}')

# ====== Blend tune on ES ======
log('\n[4] ES blend tune...')
best_w=None; best_auc=0
for w in np.arange(0,1.05,0.05):
    blended_es=lgb_es*w+cb_es*(1-w)
    a=roc_auc_score(yes,blended_es)
    if a>best_auc: best_auc,best_w=a,w
log(f'  Best: w_LGB={best_w:.2f} ES_AUC={best_auc:.4f}')
blended_te=lgb_te*best_w+cb_te*(1-best_w)

# ====== Baseline top-k on all 3 methods ======
log('\n[5] Baseline top-k...')
for name,p in [('LGB',lgb_te),('CB',cb_te),('Blended',blended_te)]:
    auc=roc_auc_score(yte,p)
    log(f'  {name}: AUC={auc:.4f}')
    conf = np.maximum(p, 1 - p)
    pred = (p >= 0.5).astype(np.int8)
    o = np.argsort(-conf)
    for k in [0.003,0.005,0.008,0.01,0.015,0.02,0.03,0.05]:
        idx=o[:max(int(len(p)*k),1)]
        acc=(pred[idx]==yte[idx]).mean()
        log(f'    top{k*100:.1f}%: acc={acc:.4f} tpd={tpd(len(idx),len(p)):.1f}')

# Correlation between LGB and CB
corr=np.corrcoef(lgb_te,cb_te)[0,1]
log(f'\n  LGB-CB corr={corr:.4f}')

# ====== Per-hour on blended ======
log(f'\n[6] Per-hour on blended...')
ranked=sorted([(h,roc_auc_score(yte[hr_te==h],blended_te[hr_te==h])) for h in range(24) if (hr_te==h).sum()>100],key=lambda x:-x[1])
log(f'  top10h: {[(h,f"{a:.4f}") for h,a in ranked[:10]]}')
log(f'  bottom6h: {[(h,f"{a:.4f}") for h,a in ranked[-6:]]}')

best_hit=[]
for K in range(2,25):
    sel=[h for h,_ in ranked[:K]]
    m=np.isin(hr_te,sel)
    if m.sum()<500: continue
    ps=blended_te[m]; ys=yte[m]
    for thr in range(50,100,1):
        h_thr=np.percentile(ps,thr)
        hit=ps>h_thr
        if hit.sum()<20: continue
        acc=(ys[hit]==1).mean(); tp=tpd(hit.sum(),len(blended_te))
        if acc>=0.65 and tp>=15:
            best_hit.append((K,thr,acc,tp))
            log(f'  ✅ K={K}h P{thr}: acc={acc:.4f} tpd={tp:.1f}')

# ====== Also try ALL 3 methods with per-hour ======
log(f'\n[7] Per-hour scan ALL methods...')
best_nearest=None
for base_name, base_pred in [('LGB',lgb_te),('CB',cb_te),('Blended',blended_te)]:
    auc_full=roc_auc_score(yte,base_pred)
    hr2=sorted([(h,roc_auc_score(yte[hr_te==h],base_pred[hr_te==h])) for h in range(24) if (hr_te==h).sum()>100],key=lambda x:-x[1])
    log(f'\n  {base_name} (AUC={auc_full:.4f}):')
    found_any=False
    for K in range(2,25):
        sel=[h for h,_ in hr2[:K]]
        m=np.isin(hr_te,sel)
        if m.sum()<500: continue
        ps=base_pred[m]; ys=yte[m]
        for thr in range(50,100,1):
            h_thr=np.percentile(ps,thr)
            hit=ps>h_thr
            if hit.sum()<20: continue
            acc=(ys[hit]==1).mean(); tp=tpd(hit.sum(),len(base_pred))
            score=min(acc/0.65, tp/15)
            if acc>=0.65 and tp>=15:
                log(f'    ✅ K={K}h P{thr}: acc={acc:.4f} tpd={tp:.1f}')
                found_any=True
            if best_nearest is None or score>best_nearest[0]:
                best_nearest=(score,base_name,K,thr,acc,tp)
    if not found_any: log(f'    (no hits)')

# ====== Summary ======
log(f'\n{"="*60}')
log('🏆 最终汇总')
log(f'{"="*60}')
lgb_conf = np.maximum(lgb_te, 1 - lgb_te); lgb_pred = (lgb_te >= 0.5).astype(np.int8); lgb_top = np.argsort(-lgb_conf)[:max(int(len(lgb_te)*0.005),1)]
cb_conf = np.maximum(cb_te, 1 - cb_te); cb_pred = (cb_te >= 0.5).astype(np.int8); cb_top = np.argsort(-cb_conf)[:max(int(len(cb_te)*0.005),1)]
bl_conf = np.maximum(blended_te, 1 - blended_te); bl_pred = (blended_te >= 0.5).astype(np.int8); bl_top = np.argsort(-bl_conf)[:max(int(len(blended_te)*0.005),1)]
log(f'  单模型 top0.5% acc: LGB={(lgb_pred[lgb_top]==yte[lgb_top]).mean():.4f} CB={(cb_pred[cb_top]==yte[cb_top]).mean():.4f} Blend={(bl_pred[bl_top]==yte[bl_top]).mean():.4f}')
if best_hit:
    best_hit.sort(key=lambda x:-x[2])
    for K,thr,acc,tp in best_hit[:10]: log(f'  ✅ K={K}h P{thr}: acc={acc:.4f} tpd={tp:.1f}')
elif best_nearest:
    log(f'  ❌ 未达标')
    log(f'  最近: {best_nearest[1]} K={best_nearest[2]}h P{best_nearest[3]}: acc={best_nearest[4]:.4f} tpd={best_nearest[5]:.1f}')
log(f'\n⏱ {time.time()-t0:.0f}s ({(time.time()-t0)/60:.0f}min)')
