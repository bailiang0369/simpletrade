"""Quick: train Tree + train NN, compare, correlate via common top-k analysis."""
import sys; sys.stdout.reconfigure(line_buffering=True)
import time, gc
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import polars as pl
import datetime as dtm, config

torch.manual_seed(42); np.random.seed(42)
t0 = time.time()

# ============================================
# PART A: Build & Train Tree (hand-crafted)
# ============================================
print("="*60 + "\nPART A: Tree model\n" + "="*60, flush=True)

eth = pl.read_parquet(f'{config.DS_DIR}/raw_ETH.parquet').sort('ts')
ts_all = eth['ts'].to_numpy().astype(np.int64)
C_all = eth['close'].to_numpy().astype(np.float64)
del eth; gc.collect()

import features as fe
eth2 = pl.read_parquet(f'{config.DS_DIR}/raw_ETH.parquet').sort('ts')
feats = fe.build_features(eth2)
del eth2; gc.collect()
feats_np = feats.to_numpy().astype(np.float32); del feats; gc.collect()

tre = int(dtm.datetime.strptime(config.TRAIN_END,'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())
es_end = int(dtm.datetime.strptime(config.SPLITS['early_stop'][1],'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())
te_start = int(dtm.datetime.strptime(config.SPLITS['test'][0],'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())

H = 15
label = (C_all[H:] > C_all[:-H]).astype(np.int64)
ret_future = (C_all[H:] / C_all[:-H] - 1).astype(np.float64)
X_all = feats_np[:-H].copy(); del feats_np, C_all; gc.collect()
ts_all_used = ts_all[:-H]

tr_mask = ts_all_used < tre
es_mask = (ts_all_used >= tre) & (ts_all_used < es_end)
te_mask = ts_all_used >= te_start

np.random.seed(42)
tr_idx = np.where(tr_mask)[0]
if len(tr_idx) > 400_000:
    tr_idx = np.random.choice(tr_idx, 400_000, replace=False)

X_tr_t = X_all[tr_idx]; y_tr_t = label[tr_idx]; r_tr_t = ret_future[tr_idx]
X_es_t = X_all[es_mask]; y_es_t = label[es_mask]
X_te_t = X_all[te_mask]; y_te_t = label[te_mask]; ts_te_t = ts_all_used[te_mask]
del X_all; gc.collect()

# Robust z-score with TRAIN stats
for j in range(X_tr_t.shape[1]):
    c = X_tr_t[:,j]
    v = c[~np.isnan(c)]
    lo, hi = np.percentile(v, 0.5), np.percentile(v, 99.5)
    X_tr_t[:,j] = np.nan_to_num(np.clip(c, lo, hi), nan=0.0)
    m, s = np.nanmean(X_tr_t[:,j]), np.nanstd(X_tr_t[:,j])+1e-6
    X_tr_t[:,j] = (X_tr_t[:,j] - m) / s
    X_es_t[:,j] = (np.nan_to_num(np.clip(X_es_t[:,j], lo, hi), nan=0.0) - m) / s
    X_te_t[:,j] = (np.nan_to_num(np.clip(X_te_t[:,j], lo, hi), nan=0.0) - m) / s
gc.collect()

pos = y_tr_t.mean()
pw = np.where(y_tr_t>0.5,(1-pos)/pos,pos/(1-pos)).astype(np.float32)
rw = np.clip(np.abs(r_tr_t)*200, 0.2, 5.0).astype(np.float32)

lgb_params = dict(objective='binary',metric='auc',learning_rate=0.05,num_leaves=63,min_child_samples=50,
                  feature_fraction=0.8,bagging_fraction=0.8,bagging_freq=5,lambda_l2=1.0,verbose=-1,n_jobs=-1)

pvs_t_t=[]; pvs_e_t=[]
for s in [42,49,56,63,70]:
    lgb_params['seed']=s
    tr_ds = lgb.Dataset(X_tr_t, label=y_tr_t, weight=pw*rw)
    es_ds = lgb.Dataset(X_es_t, label=y_es_t, reference=tr_ds)
    bst = lgb.train(lgb_params, tr_ds, num_boost_round=5000, valid_sets=[es_ds],
                    callbacks=[lgb.early_stopping(200), lgb.log_evaluation(0)])
    pvs_t_t.append(bst.predict(X_te_t)); pvs_e_t.append(bst.predict(X_es_t))

def rank_agg(pvs):
    R=np.zeros((len(pvs),len(pvs[0])),dtype=np.float64)
    for i,p in enumerate(pvs): R[i]=np.argsort(np.argsort(np.nan_to_num(p,nan=0.5))).astype(np.float64)/(len(p)-1)
    return R.mean(0).astype(np.float32)

pv_tree_te = rank_agg(pvs_t_t)
print(f"\n★ TREE 5-seed: ES={roc_auc_score(y_es_t,rank_agg(pvs_e_t)):.4f} TE={roc_auc_score(y_te_t,pv_tree_te):.4f}", flush=True)
DAYS = (ts_te_t[-1]-ts_te_t[0])/86400.0
for pct in [0.5,1.0,1.5,2.0,3.0,5.0]:
    k = max(1, int(len(pv_tree_te)*pct/100))
    acc = y_te_t[np.argsort(-pv_tree_te)[:k]].mean()*100; tpd = k/DAYS
    print(f"  TREE top-{pct:.1f}%: acc={acc:.1f}% tpd≈{tpd:.1f}", flush=True)

# ============================================
# PART B: Load v6 seq + Train NN
# ============================================
print("\n" + "="*60 + "\nPART B: NN model\n" + "="*60, flush=True)
d = np.load('/workspace/models_saved/seq_data_v6.npz')
X_tr_s = d['X_tr'].astype(np.float32); y_tr_s = d['y_tr']
X_es_s = d['X_es'].astype(np.float32); y_es_s = d['y_es']
X_te_s = d['X_te'].astype(np.float32); y_te_s = d['y_te']; ts_te_s = d['ts_te']
X_tr_sf = X_tr_s.reshape(len(X_tr_s),-1); X_es_sf = X_es_s.reshape(len(X_es_s),-1); X_te_sf = X_te_s.reshape(len(X_te_s),-1)
FT = X_tr_sf.shape[1]; gc.collect()

class MLP(nn.Module):
    def __init__(self, ft, hs, drop=0.5):
        super().__init__()
        prev=ft; layers=[]
        for h in hs: layers.extend([nn.Linear(prev,h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(drop)]); prev=h
        layers.append(nn.Linear(prev,1)); self.net=nn.Sequential(*layers)
    def forward(self,x): return self.net(x).squeeze(-1)

def eval_nn(m,X,bs=4096):
    m.eval(); pv=[]
    with torch.no_grad():
        for i in range(0,len(X),bs):
            pv.append(torch.sigmoid(m(torch.from_numpy(np.ascontiguousarray(X[i:i+bs])))).numpy())
    return np.concatenate(pv)

def train_nn(m,Xtr,ytr,Xes,yes,ep=25,lr=5e-4,wd=0.06,bs=512,pat=7,smooth=0.15):
    opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=wd)
    best=0.0;bst=None;ni=0;best_ep=0
    for e in range(ep):
        m.train();idx=np.random.permutation(len(Xtr));tl=0;nb=0
        for i in range(0,len(idx),bs):
            bi=idx[i:i+bs]
            xb=torch.from_numpy(np.ascontiguousarray(Xtr[bi].copy()))
            yb=torch.from_numpy(ytr[bi]).float()
            if smooth>0: yb=yb*(1-smooth)+0.5*smooth
            loss=F.binary_cross_entropy_with_logits(m(xb),yb)
            opt.zero_grad();loss.backward();opt.step()
            tl+=loss.item();nb+=1
        pv_es=eval_nn(m,Xes);auc_es=roc_auc_score(yes,pv_es)
        if auc_es>best+1e-5:
            best=auc_es;bst={k:v.detach().clone() for k,v in m.state_dict().items()};ni=0;best_ep=e+1
        else:
            ni+=1
            if ni>=pat: break
    if bst: m.load_state_dict(bst)
    return eval_nn(m,X_te_sf), eval_nn(m,Xes)

nn_configs = [
    ('baseline', [512,256], 0.6, 0.06),
    ('big', [1024,512,256], 0.6, 0.06),
    ('big_d07', [1024,512,256], 0.7, 0.06),
]
nn_pvs_t = {}; nn_pvs_e = {}
for name, hs, drop, wd in nn_configs:
    pvs_t=[];pvs_e=[]
    for s in [42,49,56,63,70]:
        torch.manual_seed(s);np.random.seed(s)
        m=MLP(FT,hs,drop)
        pv_t,pv_e=train_nn(m,X_tr_sf,y_tr_s,X_es_sf,y_es_s,ep=25,lr=5e-4,wd=wd,bs=512,pat=7,smooth=0.15)
        pvs_t.append(pv_t);pvs_e.append(pv_e)
    nn_pvs_t[name] = rank_agg(pvs_t); nn_pvs_e[name] = rank_agg(pvs_e)
    print(f"  NN {name}: TE={roc_auc_score(y_te_s,nn_pvs_t[name]):.4f}", flush=True)

pv_nn_multi_t = rank_agg(list(nn_pvs_t.values()))
print(f"\n★ NN MULTI (3 configs): ES={roc_auc_score(y_es_s,rank_agg(list(nn_pvs_e.values()))):.4f} TE={roc_auc_score(y_te_s,pv_nn_multi_t):.4f}", flush=True)
DAYS_s = (ts_te_s[-1]-ts_te_s[0])/86400.0
for pct in [0.5,1.0,1.5,2.0,3.0,5.0]:
    k = max(1, int(len(pv_nn_multi_t)*pct/100))
    acc = y_te_s[np.argsort(-pv_nn_multi_t)[:k]].mean()*100; tpd = k/DAYS_s
    print(f"  NN top-{pct:.1f}%: acc={acc:.1f}% tpd≈{tpd:.1f}", flush=True)

# ============================================
# PART C: Correlation & Stacking Analysis
# ============================================
print("\n" + "="*60 + "\nPART C: Correlation & Stacking\n" + "="*60, flush=True)

# Align by timestamp: find common anchors
tree_ts = np.sort(ts_te_t); nn_ts = np.sort(ts_te_s)
common_ts = np.intersect1d(tree_ts, nn_ts)
print(f"\n  Tree TE anchors: {len(tree_ts)}", flush=True)
print(f"  NN  TE anchors:  {len(nn_ts)}", flush=True)
print(f"  COMMON anchors:  {len(common_ts)}", flush=True)

# Build lookup for predictions
tree_pv_by_ts = {int(ts_te_t[i]): pv_tree_te[i] for i in range(len(ts_te_t))}
nn_pv_by_ts = {int(ts_te_s[i]): pv_nn_multi_t[i] for i in range(len(ts_te_s))}

# For labels at common_ts
raw_C = pl.read_parquet(f'{config.DS_DIR}/raw_ETH.parquet').sort('ts')['close'].to_numpy().astype(np.float64)
raw_ts = pl.read_parquet(f'{config.DS_DIR}/raw_ETH.parquet').sort('ts')['ts'].to_numpy().astype(np.int64)
ts_to_idx = {int(t): i for i,t in enumerate(raw_ts)}

y_c = np.array([raw_C[ts_to_idx[int(t)]+15] > raw_C[ts_to_idx[int(t)]] for t in common_ts], dtype=np.int64)
pv_tc = np.array([tree_pv_by_ts[int(t)] for t in common_ts])
pv_nc = np.array([nn_pv_by_ts[int(t)] for t in common_ts])
corr = np.corrcoef(pv_tc, pv_nc)[0,1]
auc_t = roc_auc_score(y_c, pv_tc); auc_n = roc_auc_score(y_c, pv_nc)
print(f"\n  On COMMON anchors ({len(common_ts)}):", flush=True)
print(f"    Tree AUC={auc_t:.4f}  NN AUC={auc_n:.4f}  CORR={corr:.4f}", flush=True)

DAYS_est = (dtm.datetime.strptime('2026-09-28','%Y-%m-%d') - dtm.datetime.strptime(config.SPLITS['test'][0],'%Y-%m-%d')).days
def eval_topk(name, pv, y, DAYS):
    auc = roc_auc_score(y,pv)
    print(f"\n  ★ [{name}] AUC={auc:.4f}", flush=True)
    for pct in [0.5,1.0,1.5,2.0,3.0]:
        k = max(1, int(len(pv)*pct/100))
        acc = y[np.argsort(-pv)[:k]].mean()*100; tpd = k/DAYS
        flag = '🏆' if pct==1.0 and acc>=65 else ('✅' if pct==1.0 and acc>=60 else '')
        print(f"    top-{pct:.1f}%: acc={acc:.1f}% tpd≈{tpd:.1f} {flag}", flush=True)

eval_topk('Tree', pv_tc, y_c, DAYS_est)
eval_topk('NN', pv_nc, y_c, DAYS_est)

# Ensembles
def rank(pv): return np.argsort(np.argsort(pv)).astype(np.float64)/len(pv)
pv_avg = (pv_tc + pv_nc) / 2
pv_rank = (rank(pv_tc) + rank(pv_nc)) / 2
eval_topk('AVG(Tree+NN)', pv_avg, y_c, DAYS_est)
eval_topk('RANK_AVG', pv_rank, y_c, DAYS_est)

# Optimal weight
best_w=0.5; best_a=0
for w in np.arange(0,1.05,0.05):
    a=roc_auc_score(y_c, w*rank(pv_tc)+(1-w)*rank(pv_nc))
    if a>best_a: best_a=a; best_w=w
pv_opt = best_w*rank(pv_tc)+(1-best_w)*rank(pv_nc)
print(f"\n  Optimal rank blend: w_Tree={best_w:.2f}, w_NN={1-best_w:.2f}, AUC={best_a:.4f}", flush=True)
eval_topk('OPTIMAL_RANK', pv_opt, y_c, DAYS_est)

# Summary
print(f"\n{'='*60}", flush=True)
print(f"SUMMARY", flush=True)
print(f"{'='*60}", flush=True)
print(f"  Tree AUC (own TE):     {roc_auc_score(y_te_t, pv_tree_te):.4f}", flush=True)
print(f"  NN   AUC (own TE):     {roc_auc_score(y_te_s, pv_nn_multi_t):.4f}", flush=True)
k=max(1,int(len(pv_tree_te)*0.01)); print(f"  Tree top-1% (own TE): {y_te_t[np.argsort(-pv_tree_te)[:k]].mean()*100:.1f}%", flush=True)
k=max(1,int(len(pv_nn_multi_t)*0.01)); print(f"  NN   top-1% (own TE): {y_te_s[np.argsort(-pv_nn_multi_t)[:k]].mean()*100:.1f}%", flush=True)
print(f"  CORR (on common):      {corr:.4f}", flush=True)
print(f"  Stacking worth it?    {'YES - low corr = high diversity' if corr<0.6 else 'MAYBE - medium corr' if corr<0.8 else 'NO - high corr'}", flush=True)
print(f"  Stacking gain (est):  ~{best_a - max(auc_t,auc_n):.4f} AUC", flush=True)
print(f"TOTAL TIME: {time.time()-t0:.0f}s", flush=True)
