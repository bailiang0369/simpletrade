"""HYBRID = ret(56) + pyramid(800) → LGBM"""
import numpy as np, pandas as pd, time, gc, sys, warnings, os
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
sys.stdout.reconfigure(line_buffering=True); warnings.filterwarnings('ignore')
T0=time.time(); SEEDS=[42,49,56,63,70]; NPYR=800

# ============= 1. Pyramid mmap + ret align =============
print('[1] Load pyramid + align ret...')
p_tr=np.load('data/pyramid/tr.npy', mmap_mode='r'); p_es=np.load('data/pyramid/es.npy', mmap_mode='r'); p_te=np.load('data/pyramid/te.npy', mmap_mode='r')
y_tr=np.load('data/pyramid/tr_y.npy'); y_es=np.load('data/pyramid/es_y.npy'); y_te=np.load('data/pyramid/te_y.npy')

TRAIN_END=1722556800; ES_END=1725148800; META_END=1759363200
ds=pd.read_parquet('data/datasets/ds_ETH_h15.parquet').sort_values('ts')
fc=[c for c in ds.columns if c not in ('ts','label','ret_future','soft_label')]
ret_all=ds[fc].to_numpy().astype(np.float32); ret_fut=ds['ret_future'].to_numpy().astype(np.float64)
ts_all=ds['ts'].to_numpy().astype(np.int64); del ds; gc.collect()

m_tr=ts_all<TRAIN_END; full_idx=np.where(m_tr)[0]
np.random.seed(42); sub=np.random.choice(len(full_idx),500000,replace=False); sub2=np.random.choice(500000,200000,replace=False)
pos=full_idx[sub[sub2]]; pos_es=np.where((ts_all>=TRAIN_END)&(ts_all<ES_END))[0][:43200]; pos_te=np.where(ts_all>=META_END)[0][:478066]
del ts_all,m_tr,full_idx,sub,sub2; gc.collect()

def hb(ret_a, p):
    py=p.reshape(-1,NPYR).astype(np.float32); del p
    return np.hstack([ret_a, py])

t0=time.time(); Xtr=hb(ret_all[pos],p_tr); del p_tr; gc.collect()
print(f'  HYBRID tr: {Xtr.shape} ({time.time()-t0:.0f}s)', flush=True)
t0=time.time(); Xes=hb(ret_all[pos_es],p_es); del p_es; gc.collect()
print(f'  HYBRID es: {Xes.shape} ({time.time()-t0:.0f}s)', flush=True)

sw=np.where(np.abs(ret_fut[pos])>=np.quantile(np.abs(ret_fut[pos]),0.90),0.3,1.0).astype(np.float32)
del ret_all,ret_fut,pos,pos_es; gc.collect()

# ============= 2. Train =============
print(f'\n[2] Train HYBRID ({Xtr.shape[1]} feats)...')
params={'objective':'binary','metric':'auc','learning_rate':0.05,'num_leaves':127,'min_child_samples':200,
        'feature_fraction':0.5,'bagging_fraction':0.8,'bagging_freq':5,'lambda_l2':0.1,'verbose':-1,'n_jobs':4}
models=[]; t0=time.time()
for sd in SEEDS:
    params['seed']=sd
    tr=lgb.Dataset(Xtr,label=y_tr,weight=sw); es_d=lgb.Dataset(Xes,label=y_es,reference=tr)
    m=lgb.train(params,tr,5000,[es_d],callbacks=[lgb.early_stopping(100),lgb.log_evaluation(0)])
    models.append(m); print(f'  sd={sd} iter={m.best_iteration} es={m.best_score["valid_0"]["auc"]:.4f}', flush=True)
del Xtr,Xes,sw; gc.collect()
print(f'  Train: {time.time()-t0:.0f}s', flush=True)

# ============= 3. Predict te =============
print(f'\n[3] Predict te...')
Xte=hb(pd.read_parquet('data/datasets/ds_ETH_h15.parquet', columns=fc).to_numpy().astype(np.float32)[pos_te], p_te)
del p_te,pos_te; gc.collect()
pte=[]
for m in models: pte.append(m.predict(Xte))
del Xte; gc.collect()
p_hyb=np.mean(pte,axis=0); auc_hyb=roc_auc_score(y_te,p_hyb)

# Ret baseline
print(f'\n[4] Ret baseline (2.4M)...')
auc_lgbm=0.5428
if os.path.exists('data/splits_npy/ETH_h15_train_X.npy'):
    Xtrr=np.load('data/splits_npy/ETH_h15_train_X.npy').astype(np.float32)
    ytrr=np.load('data/splits_npy/ETH_h15_train_y.npy').astype(np.int32)
    Xesr=np.load('data/splits_npy/ETH_h15_early_stop_X.npy').astype(np.float32)
    yesr=np.load('data/splits_npy/ETH_h15_early_stop_y.npy').astype(np.int32)
    Xter=np.load('data/splits_npy/ETH_h15_test_X.npy').astype(np.float32)
    yter=np.load('data/splits_npy/ETH_h15_test_y.npy').astype(np.int32)
    ret_w=np.load('data/splits_npy/ETH_h15_train_ret.npy')
    sw_r=np.where(np.abs(ret_w)>=np.quantile(np.abs(ret_w),0.90),0.3,1.0).astype(np.float32); del ret_w
    params_r={'objective':'binary','metric':'auc','learning_rate':0.05,'num_leaves':63,'min_child_samples':200,'feature_fraction':0.8,'bagging_fraction':0.8,'bagging_freq':5,'lambda_l2':0.1,'verbose':-1,'n_jobs':4}
    pl=[]
    for sd in SEEDS:
        params_r['seed']=sd
        tr=lgb.Dataset(Xtrr,label=ytrr,weight=sw_r); esd=lgb.Dataset(Xesr,label=yesr,reference=tr)
        m=lgb.train(params_r,tr,5000,[esd],callbacks=[lgb.early_stopping(100),lgb.log_evaluation(0)])
        pl.append(m.predict(Xter))
    del Xtrr,Xesr; gc.collect()
    p_lgbm=np.mean(pl,axis=0); auc_lgbm=roc_auc_score(yter,p_lgbm)

# Pyramid-only
auc_pyr=0.5400

print(f'\n{"="*60}')
print(f'  ★ TEST AUC 对比')
print(f'{"="*60}')
print(f'  Ret LGBM+negw (56 feats, 2.4M):    {auc_lgbm:.4f}  tpd14.4≈61.6%')
print(f'  Pyramid LGBM (800 feats, 200k):      {auc_pyr:.4f}  tpd14.4≈57.6%')
print(f'  ★ HYBRID Ret+Pyramid (856 feats):   {auc_hyb:.4f}  ({(auc_hyb-auc_lgbm)*100:+.2f}pp)')
print(f'  CNN pyramid (500k, 3 seeds):          0.5415')
print(f'{"="*60}')
print(f'\nTOTAL: {time.time()-T0:.0f}s')
