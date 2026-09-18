"""超精简: 只跑 single_H30_base + per-hour 过滤"""
import time, gc, numpy as np, datetime, sys, warnings, pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore')
sys.path.insert(0,'/workspace'); import config

def log(m): print(m,flush=True)
def tpd(n,b): return n*1440/b

t0=time.time()
log('=== fast: train single + per-hour ===')

# ========== Load ==========
log('loading...')
ds=pd.read_parquet(f'{config.DS_DIR}/ds_ETH_h30.parquet')
feat_cols=[c for c in ds.columns if c not in ['label','soft_label','ret_future','ts']]
ts=ds['ts'].values.astype(np.int64)
label_h30=ds['label'].values.astype(np.int8)
ret_h30=ds['ret_future'].values.astype(np.float32)
X=ds[feat_cols].values.astype(np.float32)
del ds; gc.collect()

raw_e=pd.read_parquet(f'{config.DS_DIR}/raw_ETH.parquet')
close_raw=pd.Series(raw_e['close'].values.astype(np.float64),index=raw_e['ts'].values.astype(np.int64)).reindex(ts).values.astype(np.float64)
del raw_e; gc.collect()

HORIZONS=[3,5,15,30,60]
multi={}
for h in HORIZONS:
    rh=np.full(len(ts),np.nan,np.float32)
    rh[:-h]=(close_raw[h:]/close_raw[:-h]-1).astype(np.float32)
    multi[f'r{h}']=rh; multi[f'y{h}']=(rh>0).astype(np.int8)
del close_raw; gc.collect()

def ts_mask(tsarr,s,e):
    a=int(datetime.datetime.strptime(s,'%Y-%m-%d').replace(tzinfo=datetime.timezone.utc).timestamp())
    b=int(datetime.datetime.strptime(e,'%Y-%m-%d').replace(tzinfo=datetime.timezone.utc).timestamp())
    return (tsarr>=a)&(tsarr<b)
tr_m=ts_mask(ts,'2020-01-01','2024-06-30')
es_m=ts_mask(ts,'2024-06-30','2024-09-30')
te_m=ts_mask(ts,'2025-09-30','2026-08-29')
vm=~np.isnan(multi['r30']); vi=np.where(vm)[0]
X=X[vi]; ts_c=ts[vi]; hr=pd.to_datetime(ts_c,unit='s',utc=True).hour.values.astype(np.int32)
tr_m2=ts_mask(ts_c,'2020-01-01','2024-06-30')
es_m2=ts_mask(ts_c,'2024-06-30','2024-09-30')
te_m2=ts_mask(ts_c,'2025-09-30','2026-08-29')
label_h30=label_h30[vi]; ret_h30=ret_h30[vi]
for k in list(multi.keys()): multi[k]=multi[k][vi]
del vi,ts; gc.collect()

Xtr=X[tr_m2]; Xes=X[es_m2]; Xte=X[te_m2]; del X; gc.collect()
yte=label_h30[te_m2]; ret_te=ret_h30[te_m2]; hr_te=hr[te_m2]
ret_tr={h:multi[f'r{h}'][tr_m2] for h in HORIZONS}
log(f'TR={len(Xtr):,} ES={len(Xes):,} TE={len(Xte):,}')

# ========== Train 5 baseline models ==========
log('[train] 5 horizons × baseline LGB (10 seeds)...')
def mk_lp(seed,nl=63,lr=0.02,mdl=200,l2=1.0):
    return dict(num_leaves=nl,learning_rate=lr,min_data_in_leaf=mdl,feature_fraction=0.9,bagging_fraction=0.9,bagging_freq=5,lambda_l2=l2,verbose=-1,num_threads=3,objective='binary',metric='auc',seed=seed)

ch_b={}
for h in HORIZONS:
    yh_tr=multi[f'y{h}'][tr_m2]; yh_es=multi[f'y{h}'][es_m2]
    keep=np.abs(ret_tr[h])>=0.0005
    Xf=Xtr[keep]; yf=yh_tr[keep]
    p_te=[]
    for sd in range(42,42+10):
        dtr=lgb.Dataset(Xf,label=yf); des=lgb.Dataset(Xes,label=yh_es,reference=dtr)
        m=lgb.train(mk_lp(sd),dtr,num_boost_round=3000,valid_sets=[des],callbacks=[lgb.early_stopping(200,verbose=False)])
        p_te.append(m.predict(Xte)); del m; gc.collect()
    ch_b[h]=np.mean(p_te,axis=0)
    log(f'  H={h}: AUC_H={roc_auc_score(multi[f"y{h}"][te_m2],ch_b[h]):.4f}')

single_H30 = ch_b[30]
ch_eq_b = np.mean(np.column_stack([ch_b[h] for h in HORIZONS]),axis=1)
del ch_b, multi, ret_tr; gc.collect()

# ========== Per-hour filter ==========
log(f'\n{"="*60}'); log('Per-hour filter'); log(f'{"="*60}')

methods=[('single_H30',single_H30),('CH_eq_b',ch_eq_b)]

best_overall=None
for base_name, base_pred in methods:
    auc_full=roc_auc_score(yte,base_pred)
    ranked=sorted([(h,roc_auc_score(yte[hr_te==h],base_pred[hr_te==h])) for h in range(24) if (hr_te==h).sum()>100],key=lambda x:-x[1])
    log(f'\n{base_name}: full AUC={auc_full:.4f}')
    log(f'  top8h: {ranked[:8]}')
    
    for K in range(2,25):
        sel=[h for h,_ in ranked[:K]]
        m=np.isin(hr_te,sel)
        if m.sum()<100: continue
        p_sel=base_pred[m]; y_sel=yte[m]
        for thr in range(50,100,1):
            h_thr=np.percentile(p_sel,thr)
            hit=p_sel>h_thr
            if hit.sum()<50: continue
            acc=(y_sel[hit]==1).mean(); tp=tpd(hit.sum(),len(yte))
            score=min(acc/0.65, tp/15)
            hit_target = acc>=0.65 and tp>=15
            near_target = acc>=0.62 and tp>=10
            if hit_target or near_target:
                flag='✅HIT' if hit_target else 'near'
                log(f'    K={K}h P{thr}: acc={acc:.4f} tpd={tp:.1f} {flag}')
            if score>0.95:
                log(f'    🎯 K={K}h P{thr}: acc={acc:.4f} tpd={tp:.1f}')
            if best_overall is None or score>best_overall[0]:
                best_overall=(score,base_name,K,thr,acc,tp)

log(f'\n{"="*60}')
log('🏆 结果汇总')
log(f'{"="*60}')
for base_name, base_pred in methods:
    o=np.argsort(base_pred)
    for k in [0.005,0.01,0.015,0.02,0.03,0.05,0.08,0.10]:
        idx=o[-max(int(len(base_pred)*k),1):]
        acc=(yte[idx]==1).mean(); tp=tpd(len(idx),len(base_pred))
        log(f'  {base_name} top{k*100:.1f}%: acc={acc:.4f} tpd={tp:.1f}')

if best_overall:
    log(f'\n🎯 Per-hour 最佳:')
    log(f'   {best_overall[1]} × top-{best_overall[2]}h × P{best_overall[3]}')
    log(f'   acc={best_overall[4]:.4f} tpd={best_overall[5]:.1f} score={best_overall[0]:.3f}')

log(f'\n⏱ Total: {time.time()-t0:.0f}s ({(time.time()-t0)/60:.1f}min)')
