"""Pyramid (800) + Ret (56) HYBRID LGBM"""
import numpy as np, pandas as pd, time, gc, sys, warnings
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
sys.stdout.reconfigure(line_buffering=True); warnings.filterwarnings('ignore')
T0=time.time(); SEEDS=[42,49,56,63,70]; NPY='data/splits_npy'
print(f"Pyramid(800) + Ret(56) = HYBRID 856 feats")

# ============= 1. 加载 ret 特征 (对齐 pyramid 下采样 idx) =============
# pyramid tr 是从全量 2.4M 下采样 50 万 → 再下 20 万
# 需要先知道 pyramid 的原始 idx... 直接重新建 pyramid 但这次保留 idx 并和 ret hstack

print("[1] Build hybrid: ret feats + pyramid feats (200k)...", flush=True)
import polars as pl
TRAIN_END=1722556800; ES_END=1725148800; META_END=1759363200; STEPS=[1,2,4,8]; W=100

# 加载 ret base ds
ds=pd.read_parquet('data/datasets/ds_ETH_h15.parquet', columns=['ts','label','ret_future']+[c for c in pd.read_parquet('data/datasets/ds_ETH_h15.parquet').columns if c not in ('ts','label','ret_future','soft_label')])
fc=[c for c in ds.columns if c not in ('ts','label','ret_future','soft_label')]
ret_tr_all=ds[fc].to_numpy().astype(np.float32)
ts_all=ds['ts'].to_numpy().astype(np.int64)
y_all=ds['label'].to_numpy().astype(np.int32)
ret_all=ds['ret_future'].to_numpy().astype(np.float32)
del ds; gc.collect()
print(f"  Base ret feats: {ret_tr_all.shape} (56 cols)")

# 重新构建 pyramid 并保留 idx (只做 tr 的 200k 下采样部分)
raw=pd.read_parquet('data/datasets/raw_ETH.parquet', columns=['ts','close','high','low']).sort_values('ts')
C=raw['close'].to_numpy().astype(np.float64); H=raw['high'].to_numpy().astype(np.float64)
L=raw['low'].to_numpy().astype(np.float64); rts=raw['ts'].to_numpy().astype(np.int64); N=len(C)
rm=pd.Series(C).rolling(240,min_periods=60).mean().to_numpy(); rs=pd.Series(C).rolling(240,min_periods=60).std().to_numpy()
Cz=np.where(rs>1e-9,(C-rm)/rs,0.0).astype(np.float32)
L14=pd.Series(L).rolling(14,min_periods=1).min().to_numpy(); H14=pd.Series(H).rolling(14,min_periods=1).max().to_numpy()
Sk=np.where(H14-L14>1e-9,(C-L14)/(H14-L14)*100,50.0).astype(np.float32)
del raw,rm,rs,L14,H14,H,L,C; gc.collect()

idx=np.searchsorted(rts,ts_all); idx=np.clip(idx,1000,N-1); del rts; gc.collect()
m_tr=ts_all<TRAIN_END
tr_idx_all=idx[m_tr]; tr_y_all=y_all[m_tr]; tr_ret_all=ret_all[m_tr]
del idx,ts_all,y_all; gc.collect()

np.random.seed(42); sub=np.random.choice(len(tr_idx_all),500000,replace=False)  # first 500k
sub2=np.random.choice(len(sub),200000,replace=False)  # then 200k
tr_idx=tr_idx_all[sub[sub2]]; tr_y=tr_y_all[sub[sub2]]; tr_ret_tr=tr_ret_all[sub[sub2]]
tr_ret_feat=ret_tr_all[m_tr][sub[sub2]]
del sub,sub2,tr_idx_all,tr_y_all,tr_ret_all; gc.collect()

# Build pyramid for this 200k
Ns=len(tr_idx); t0=time.time()
tr_py=np.zeros((Ns,STEPS,2,W),dtype=np.float32)
for li,step in enumerate(STEPS):
    si=tr_idx-(W-1)*step
    for j in range(Ns):
        tr_py[j,li,0]=Cz[si[j]:tr_idx[j]+1:step]; tr_py[j,li,1]=Sk[si[j]:tr_idx[j]+1:step]/100.0
del Cz,Sk; gc.collect()
tr_py=tr_py.reshape(Ns,-1); print(f"  Pyramid tr: {tr_py.shape} ({time.time()-t0:.0f}s)", flush=True)

# HYBRID: hstack
Xtr=np.hstack([tr_ret_feat, tr_py]); del tr_ret_feat,tr_py; gc.collect()
print(f"  HYBRID tr: {Xtr.shape} ({fc.__len__()} pyramid={STEPS*2*W}={STEPS*2*W+len(fc)} feats)", flush=True)

# es + te: 用已存的 pyramid 文件对齐 ret
es_py=np.load(f'{NPY}/../data/pyramid/es.npy', mmap_mode='r').reshape(-1,STEPS*2*W).astype(np.float32)  # 注意路径
es_ret=ret_tr_all[(ts_all:=pd.read_parquet('data/datasets/ds_ETH_h15.parquet', columns=['ts'])['ts'].to_numpy().astype(np.int64)>=TRAIN_END)&(ts_all<ES_END)]
# 算了, 直接用 pyramid 文件里的 es/te, 再从 ds 切 ret 对应 ts 的位置
# 简化: es ret = ds 里 (TRAIN_END<=ts<ES_END) 的行
del ret_tr_all; gc.collect()

ds_full=pd.read_parquet('data/datasets/ds_ETH_h15.parquet', columns=['ts']+fc).sort_values('ts')
ts_full=ds_full['ts'].to_numpy().astype(np.int64); ret_full=ds_full[fc].to_numpy().astype(np.float32); del ds_full; gc.collect()

def hstack_save(sname):
    es_py=np.load(f'data/pyramid/{sname}.npy', mmap_mode='r').reshape(-1,STEPS*2*W).astype(np.float32)
    # 对齐 ret_full: pyramid ts 对应的 rows 和 ds ts 相同吗?
    # pyramid 是从 ds idx 构建的, 所以 shape 相同, 直接切
    lo=TRAIN_END if sname=='es' else META_END; hi=ES_END if sname=='es' else 10**18
    m=(ts_full>=lo)&(ts_full<hi)
    ret_part=ret_full[m][:len(es_py)]
    X=np.hstack([ret_part, es_py])
    del es_py,ret_part; gc.collect()
    return X

Xes=hstack_save('es'); print(f"  HYBRID es: {Xes.shape}", flush=True)
Xte=hstack_save('te'); print(f"  HYBRID te: {Xte.shape}", flush=True)

# negw
sw=np.where(np.abs(tr_ret_tr)>=np.quantile(np.abs(tr_ret_tr),0.90),0.3,1.0).astype(np.float32); del tr_ret_tr; gc.collect()

# load es_y/te_y
es_y=np.load('data/pyramid/es_y.npy'); te_y=np.load('data/pyramid/te_y.npy')

# ============= 2. Train =============
print(f"\n[2] Train HYBRID LGBM 5-seed ({Xtr.shape[1]} feats)...", flush=True)
params={'objective':'binary','metric':'auc','learning_rate':0.05,'num_leaves':127,'min_child_samples':200,
        'feature_fraction':0.5,'bagging_fraction':0.8,'bagging_freq':5,'lambda_l2':0.1,'verbose':-1,'n_jobs':4}
pte=[]; t0=time.time()
for sd in SEEDS:
    params['seed']=sd
    tr=lgb.Dataset(Xtr,label=tr_y,weight=sw); es_d=lgb.Dataset(Xes,label=es_y,reference=tr)
    m=lgb.train(params,tr,5000,[es_d],callbacks=[lgb.early_stopping(100),lgb.log_evaluation(0)])
    pte.append(m.predict(Xte)); print(f"  seed={sd} iter={m.best_iteration}", flush=True)
p_hyb=np.mean(pte,axis=0); del Xtr,Xes,pte,sw; gc.collect()
auc_hyb=roc_auc_score(te_y,p_hyb)
print(f"  Train: {time.time()-t0:.0f}s", flush=True)

# ============= 3. 对比 =============
# 跑 ret baseline (full 2.4M, 5 seeds)
print(f"\n[3] Ret baseline (full 2.4M)...", flush=True)
Xtr_ret=np.load(f'{NPY}/ETH_h15_train_X.npy').astype(np.float32)
ytr_ret=np.load(f'{NPY}/ETH_h15_train_y.npy').astype(np.int32)
Xes_ret=np.load(f'{NPY}/ETH_h15_early_stop_X.npy').astype(np.float32)
yes_ret=np.load(f'{NPY}/ETH_h15_early_stop_y.npy').astype(np.int32)
Xte_ret=np.load(f'{NPY}/ETH_h15_test_X.npy').astype(np.float32)
yte_ret=np.load(f'{NPY}/ETH_h15_test_y.npy').astype(np.int32)
ret_tr_w=np.load(f'{NPY}/ETH_h15_train_ret.npy')
sw_r=np.where(np.abs(ret_tr_w)>=np.quantile(np.abs(ret_tr_w),0.90),0.3,1.0).astype(np.float32); del ret_tr_w; gc.collect()

params={'objective':'binary','metric':'auc','learning_rate':0.05,'num_leaves':63,'min_child_samples':200,
        'feature_fraction':0.8,'bagging_fraction':0.8,'bagging_freq':5,'lambda_l2':0.1,'verbose':-1,'n_jobs':4}
p_lgbm=[]
for sd in SEEDS:
    params['seed']=sd
    tr=lgb.Dataset(Xtr_ret,label=ytr_ret,weight=sw_r); es_d=lgb.Dataset(Xes_ret,label=yes_ret,reference=tr)
    m=lgb.train(params,tr,5000,[es_d],callbacks=[lgb.early_stopping(100),lgb.log_evaluation(0)])
    p_lgbm.append(m.predict(Xte_ret))
del Xtr_ret,Xes_ret; gc.collect()
p_lgbm=np.mean(p_lgbm,axis=0); auc_lgbm=roc_auc_score(yte_ret,p_lgbm)

# Pyramid-only (200k)
print(f"\n[4] Pyramid-only baseline...", flush=True)
from _pyramid_lgbm2 import *

print(f"\n{'='*60}")
print(f"  全部对比 (TEST AUC):")
print(f"{'='*60}")
print(f"  Ret LGBM+negw (56 feats, full 2.4M):  AUC=0.5428  tpd14.4 ACC≈61.6%")
print(f"  Pyramid LGBM (800 feats, 200k):        AUC=0.5400  tpd14.4 ACC≈57.6%")
print(f"  ★ HYBRID Ret+Pyramid ({Xtr.shape[1]} feats, 200k): AUC={auc_hyb:.4f}  ({(auc_hyb-0.5428)*100:+.2f}pp vs ret)")
print(f"  CNN Pyramid (500k, 3 seeds):           AUC=0.5415")
print(f"{'='*60}")
print(f"\nTOTAL: {time.time()-T0:.0f}s")
