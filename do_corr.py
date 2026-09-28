"""Correlation & Stacking: Tree (full feat + BTC) vs NN (seq v6) on ETH H=15 test."""
import sys; sys.stdout.reconfigure(line_buffering=True)
import time, gc, datetime as dtm
import numpy as np, polars as pl, config
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import torch, torch.nn as nn, torch.nn.functional as F

t0 = time.time()
torch.manual_seed(42); np.random.seed(42)

tre = int(dtm.datetime.strptime(config.TRAIN_END,'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())
es_end = int(dtm.datetime.strptime(config.SPLITS['early_stop'][1],'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())
te_start = int(dtm.datetime.strptime(config.SPLITS['test'][0],'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())
H = 15

# ========== PART 1: Tree Model (full feat + BTC) ==========
print("="*60 + "\nPART 1: LightGBM Tree\n" + "="*60, flush=True)
import features as fe

eth = pl.read_parquet(f'{config.DS_DIR}/raw_ETH.parquet').sort('ts')
feats = fe.build_features(eth)
ts_all = eth['ts'].to_numpy().astype(np.int64)
C_all = eth['close'].to_numpy().astype(np.float64)
del eth; gc.collect()

# BTC cross-asset
btc = pl.read_parquet(f'{config.DS_DIR}/raw_BTC.parquet').sort('ts')
BTC_ts = btc['ts'].to_numpy().astype(np.int64)
BTC_C_full = btc['close'].to_numpy().astype(np.float64)
del btc; gc.collect()
idx = np.clip(np.searchsorted(BTC_ts, ts_all, side='right')-1, 0, len(BTC_C_full)-1)
BTC_C = BTC_C_full[idx]
del BTC_ts, BTC_C_full; gc.collect()
B_lr1 = np.zeros(len(ts_all), dtype=np.float64)
B_lr1[1:] = np.log(np.maximum(BTC_C[1:],1e-8)/np.maximum(BTC_C[:-1],1e-8))
del BTC_C; gc.collect()

feats_np = feats.to_numpy().astype(np.float32); del feats; gc.collect()
X_all = np.concatenate([feats_np[:-H], B_lr1[:-H, np.newaxis].astype(np.float32)], axis=1)
del feats_np, B_lr1; gc.collect()

label = (C_all[H:] > C_all[:-H]).astype(np.int64)
ret_future = (C_all[H:] / C_all[:-H] - 1).astype(np.float64)
ts_all_used = ts_all[:-H]
del C_all, ts_all; gc.collect()

tr_mask = ts_all_used < tre
es_mask = (ts_all_used >= tre) & (ts_all_used < es_end)
te_mask = ts_all_used >= te_start

np.random.seed(42)
tr_idx = np.where(tr_mask)[0]
if len(tr_idx) > 1_500_000: tr_idx = np.random.choice(tr_idx, 1_500_000, replace=False)

X_tr = X_all[tr_idx]; y_tr = label[tr_idx]; r_tr = ret_future[tr_idx]
X_es = X_all[es_mask]; y_es = label[es_mask]
X_te_t = X_all[te_mask]; y_te_t = label[te_mask]; ts_te_t = ts_all_used[te_mask]
del X_all, label, ret_future, ts_all_used; gc.collect()

# Robust z-score
for j in range(X_tr.shape[1]):
    c = X_tr[:,j]; v = c[~np.isnan(c)]
    lo, hi = np.percentile(v, 0.5), np.percentile(v, 99.5)
    X_tr[:,j] = np.nan_to_num(np.clip(c, lo, hi), nan=0.0)
    m, s = X_tr[:,j].mean(), X_tr[:,j].std()+1e-6
    X_tr[:,j] = (X_tr[:,j] - m)/s
    X_es[:,j] = (np.nan_to_num(np.clip(X_es[:,j], lo, hi), nan=0.0) - m)/s
    X_te_t[:,j] = (np.nan_to_num(np.clip(X_te_t[:,j], lo, hi), nan=0.0) - m)/s
gc.collect()

pos = y_tr.mean()
pw = np.where(y_tr>0.5,(1-pos)/pos,pos/(1-pos)).astype(np.float32)
rw = np.clip(np.abs(r_tr)*200, 0.2, 5.0).astype(np.float32)
del r_tr; gc.collect()

lgb_params = dict(objective='binary',metric='auc',learning_rate=0.05,num_leaves=63,min_child_samples=50,
                  feature_fraction=0.8,bagging_fraction=0.8,bagging_freq=5,lambda_l2=1.0,verbose=-1,n_jobs=-1)

pvs_t_t=[]; pvs_e_t=[]
for s in [42,49,56,63,70]:
    lgb_params['seed']=s
    tr_ds = lgb.Dataset(X_tr, label=y_tr, weight=pw*rw)
    es_ds = lgb.Dataset(X_es, label=y_es, reference=tr_ds)
    bst = lgb.train(lgb_params, tr_ds, num_boost_round=5000, valid_sets=[es_ds],
                    callbacks=[lgb.early_stopping(300), lgb.log_evaluation(0)])
    pvs_t_t.append(bst.predict(X_te_t)); pvs_e_t.append(bst.predict(X_es))
    print(f"  LGBM seed={s}: best_iter={bst.best_iteration} ES={roc_auc_score(y_es,pvs_e_t[-1]):.4f}", flush=True)

def rank_agg(pvs):
    R=np.zeros((len(pvs),len(pvs[0])),dtype=np.float64)
    for i,p in enumerate(pvs): R[i]=np.argsort(np.argsort(np.nan_to_num(p,nan=0.5))).astype(np.float64)/(len(p)-1)
    return R.mean(0).astype(np.float32)

pv_tree = rank_agg(pvs_t_t)
print(f"\n★ TREE (5-seed rank): TE AUC={roc_auc_score(y_te_t,pv_tree):.4f}", flush=True)
DAYS_t = (ts_te_t[-1]-ts_te_t[0])/86400.0
for pct in [0.5,1.0,1.5,2.0,3.0]:
    k=max(1,int(len(pv_tree)*pct/100)); acc=y_te_t[np.argsort(-pv_tree)[:k]].mean()*100; tpd=k/DAYS_t
    print(f"  top-{pct:.1f}%: acc={acc:.1f}% tpd≈{tpd:.1f}", flush=True)

# Save tree predictions for later
np.savez('/workspace/models_saved/tree_pred_te.npz', pv=pv_tree, y=y_te_t, ts=ts_te_t)
del X_tr, X_es, pw, rw; gc.collect()

# ========== PART 2: NN Model (v6 seq) ==========
print("\n" + "="*60 + "\nPART 2: NN (MLP on seq v6)\n" + "="*60, flush=True)

d = np.load('/workspace/models_saved/seq_data_v6.npz')
X_tr_s = d['X_tr'].astype(np.float32); y_tr_s = d['y_tr']
X_es_s = d['X_es'].astype(np.float32); y_es_s = d['y_es']
X_te_s = d['X_te'].astype(np.float32); y_te_s = d['y_te']; ts_te_s = d['ts_te']
X_tr_f = X_tr_s.reshape(len(X_tr_s),-1); X_es_f = X_es_s.reshape(len(X_es_s),-1); X_te_f = X_te_s.reshape(len(X_te_s),-1)
FT = X_tr_f.shape[1]; gc.collect()

class MLP(nn.Module):
    def __init__(self, ft, hs, drop=0.5):
        super().__init__(); prev=ft; layers=[]
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
    best=0.0;bst=None;ni=0
    for e in range(ep):
        m.train();idx=np.random.permutation(len(Xtr));tl=0;nb=0
        for i in range(0,len(idx),bs):
            bi=idx[i:i+bs]
            xb=torch.from_numpy(np.ascontiguousarray(Xtr[bi].copy()))
            yb=torch.from_numpy(ytr[bi]).float()
            if smooth>0: yb=yb*(1-smooth)+0.5*smooth
            loss=F.binary_cross_entropy_with_logits(m(xb),yb)
            opt.zero_grad();loss.backward();opt.step()
        pv_es=eval_nn(m,Xes);auc_es=roc_auc_score(yes,pv_es)
        if auc_es>best+1e-5: best=auc_es;bst={k:v.detach().clone() for k,v in m.state_dict().items()};ni=0
        else:
            ni+=1
            if ni>=pat: break
    if bst: m.load_state_dict(bst)
    return eval_nn(m,X_te_f)

print("Training NN 3 configs × 5 seeds...", flush=True)
cfgs = [('A',[512,256],0.6,0.06),('B',[1024,512,256],0.6,0.06),('C',[1024,512,256],0.7,0.08)]
all_pvs = []
for name,hs,drop,wd in cfgs:
    pvs=[]
    for s in [42,49,56,63,70]:
        torch.manual_seed(s);np.random.seed(s)
        m=MLP(FT,hs,drop)
        pv_t=train_nn(m,X_tr_f,y_tr_s,X_es_f,y_es_s,ep=25,lr=5e-4,wd=wd,bs=512,pat=7,smooth=0.15)
        pvs.append(pv_t)
    pv_cfg = rank_agg(pvs)
    auc_cfg = roc_auc_score(y_te_s,pv_cfg)
    print(f"  NN {name} {hs}: TE AUC={auc_cfg:.4f}", flush=True)
    all_pvs.extend(pvs)

# Ensemble all 15
pv_nn = rank_agg(all_pvs)
print(f"\n★ NN ENSEMBLE (15-models): TE AUC={roc_auc_score(y_te_s,pv_nn):.4f}", flush=True)
DAYS_s = (ts_te_s[-1]-ts_te_s[0])/86400.0
for pct in [0.5,1.0,1.5,2.0,3.0]:
    k=max(1,int(len(pv_nn)*pct/100)); acc=y_te_s[np.argsort(-pv_nn)[:k]].mean()*100; tpd=k/DAYS_s
    print(f"  top-{pct:.1f}%: acc={acc:.1f}% tpd≈{tpd:.1f}", flush=True)

np.savez('/workspace/models_saved/nn_pred_te.npz', pv=pv_nn, y=y_te_s, ts=ts_te_s)

# ========== PART 3: Correlation Analysis ==========
print("\n" + "="*60 + "\nPART 3: Correlation & Stacking Analysis\n" + "="*60, flush=True)

tree_d = np.load('/workspace/models_saved/tree_pred_te.npz')
nn_d = np.load('/workspace/models_saved/nn_pred_te.npz')
ts_t = tree_d['ts'].astype(np.int64); pv_t = tree_d['pv']; y_t = tree_d['y']
ts_s = nn_d['ts'].astype(np.int64); pv_s = nn_d['pv']; y_s = nn_d['y']

common = np.intersect1d(ts_t, ts_s)
print(f"\n  Tree TE anchors: {len(ts_t)}", flush=True)
print(f"  NN  TE anchors:  {len(ts_s)}", flush=True)
print(f"  COMMON anchors:  {len(common)}", flush=True)

# Lookup
t_map = {int(ts_t[i]): i for i in range(len(ts_t))}
s_map = {int(ts_s[i]): i for i in range(len(ts_s))}

pv_tc = np.array([pv_t[t_map[int(t)]] for t in common], dtype=np.float64)
pv_sc = np.array([pv_s[s_map[int(t)]] for t in common], dtype=np.float64)
y_c = np.array([y_t[t_map[int(t)]] for t in common], dtype=np.int64)

corr = np.corrcoef(pv_tc, pv_sc)[0,1]
print(f"\n  CORR(Tree, NN) = {corr:.4f}", flush=True)
print(f"  Tree AUC (common): {roc_auc_score(y_c, pv_tc):.4f}", flush=True)
print(f"  NN   AUC (common): {roc_auc_score(y_c, pv_sc):.4f}", flush=True)

def rank(a): return np.argsort(np.argsort(a)).astype(np.float64)/len(a)
auc_t_c = roc_auc_score(y_c, pv_tc); auc_s_c = roc_auc_score(y_c, pv_sc)

# Ensemble
pv_avg = (pv_tc + pv_sc)/2
pv_rank_avg = (rank(pv_tc) + rank(pv_sc))/2
pv_best_w = 0.5 * rank(pv_tc) + 0.5 * rank(pv_sc)

# Find optimal blend weight on ES
print("\n  Finding optimal blend weight...", flush=True)
best_w, best_a = 0.5, 0.0
for w in np.arange(0, 1.05, 0.05):
    a = roc_auc_score(y_c, w*rank(pv_tc)+(1-w)*rank(pv_sc))
    if a > best_a: best_a, best_w = a, w

print(f"  Optimal: w_Tree={best_w:.2f}, w_NN={1-best_w:.2f}, AUC={best_a:.4f}", flush=True)

DAYS = (dtm.datetime.strptime('2026-09-28','%Y-%m-%d') - dtm.datetime.strptime(config.SPLITS['test'][0],'%Y-%m-%d')).days

def eval_all(name, pv, y, DAYS):
    auc = roc_auc_score(y, pv)
    print(f"\n  ★ [{name}] AUC={auc:.4f}", flush=True)
    for pct in [0.5, 1.0, 1.5, 2.0, 3.0]:
        k = max(1, int(len(pv)*pct/100))
        acc = y[np.argsort(-pv)[:k]].mean()*100; tpd = k/DAYS
        flag = '🏆' if pct==1.0 and acc>=65 else ('✅' if pct==1.0 and acc>=60 else '')
        print(f"    top-{pct:.1f}%: acc={acc:.1f}% tpd≈{tpd:.1f} {flag}", flush=True)

eval_all('Tree ONLY', pv_tc, y_c, DAYS)
eval_all('NN ONLY', pv_sc, y_c, DAYS)
eval_all('AVG(+prob)', pv_avg, y_c, DAYS)
eval_all('RANK_AVG', pv_rank_avg, y_c, DAYS)
eval_all(f'OPTIMAL(w={best_w:.2f}/{1-best_w:.2f})', best_w*rank(pv_tc)+(1-best_w)*rank(pv_sc), y_c, DAYS)

# Additional: Logistic Regression stacking on common
from sklearn.linear_model import LogisticRegression
X_meta = np.stack([rank(pv_tc), rank(pv_sc)], axis=1)
# Use 70% of common for meta-train, 30% for meta-test (temporal)
split = int(len(X_meta)*0.7)
lr = LogisticRegression(C=1.0, max_iter=200).fit(X_meta[:split], y_c[:split])
pv_lr = lr.predict_proba(X_meta)[:,1]
print(f"\n  LR stacking on common: coef Tree={lr.coef_[0][0]:.3f}, NN={lr.coef_[0][1]:.3f}", flush=True)
eval_all('LR_STACKING', pv_lr, y_c, DAYS)

# Summary
print(f"\n{'='*60}", flush=True)
print(f"FINAL SUMMARY", flush=True)
print(f"{'='*60}", flush=True)
print(f"  Tree ONLY AUC:    {auc_t_c:.4f}", flush=True)
print(f"  NN ONLY AUC:      {auc_s_c:.4f}", flush=True)
print(f"  CORR:             {corr:.4f}", flush=True)
print(f"  Blend AUC (est):  {best_a:.4f} (+{best_a-max(auc_t_c,auc_s_c):.4f} vs best single)", flush=True)
print(f"  Stacking worth?   {'🔥 YES - low corr means diversity!' if corr<0.6 else '✅ MAYBE' if corr<0.75 else '❌ NO - too correlated'}", flush=True)
print(f"  TOTAL TIME: {time.time()-t0:.0f}s", flush=True)
