"""Final correlation & stacking: Tree (baseline params) vs NN (seq v6)."""
import sys; sys.stdout.reconfigure(line_buffering=True)
import time, gc, datetime as dtm
import numpy as np, polars as pl, config
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import torch, torch.nn as nn, torch.nn.functional as F

t0 = time.time()
tre = int(dtm.datetime.strptime(config.TRAIN_END,'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())
es_end = int(dtm.datetime.strptime(config.SPLITS['early_stop'][1],'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())
te_start = int(dtm.datetime.strptime(config.SPLITS['test'][0],'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())
H = 15

# ========== Tree with BASELINE_AUC params ==========
print("="*60 + "\nPART 1: LightGBM (baseline_auc params, 5 seeds)\n" + "="*60, flush=True)
import features as fe
eth = pl.read_parquet(f'{config.DS_DIR}/raw_ETH.parquet').sort('ts')
feats = fe.build_features(eth)
ts = eth['ts'].to_numpy().astype(np.int64)
C = eth['close'].to_numpy().astype(np.float64)
del eth; gc.collect()

btc = pl.read_parquet(f'{config.DS_DIR}/raw_BTC.parquet').sort('ts')
BTC_ts = btc['ts'].to_numpy().astype(np.int64)
BTC_C_full = btc['close'].to_numpy().astype(np.float64)
del btc; gc.collect()
idx = np.clip(np.searchsorted(BTC_ts, ts, side='right')-1, 0, len(BTC_C_full)-1)
BTC_C = BTC_C_full[idx]; del BTC_ts, BTC_C_full; gc.collect()
B_lr1 = np.zeros(len(ts), dtype=np.float64)
B_lr1[1:] = np.log(np.maximum(BTC_C[1:],1e-8)/np.maximum(BTC_C[:-1],1e-8))
del BTC_C; gc.collect()

X = np.concatenate([feats.to_numpy().astype(np.float32)[:-H], B_lr1[:-H, np.newaxis].astype(np.float32)], axis=1)
del feats, B_lr1; gc.collect()
y = (C[H:] > C[:-H]).astype(np.int64); del C; gc.collect()
ts_u = ts[:-H]; del ts; gc.collect()

tr_m = ts_u < tre; es_m = (ts_u >= tre) & (ts_u < es_end); te_m = ts_u >= te_start
np.random.seed(42)
tr_idx = np.where(tr_m)[0]
if len(tr_idx) > 1_500_000: tr_idx = np.random.choice(tr_idx, 1_500_000, replace=False)
X_tr = X[tr_idx]; y_tr = y[tr_idx]
X_es = X[es_m]; y_es = y[es_m]
X_te = X[te_m]; y_te = y[te_m]; ts_te = ts_u[te_m]
del X, y, ts_u, tr_m, es_m, te_m; gc.collect()

for j in range(X_tr.shape[1]):
    c = X_tr[:,j]; v = c[~np.isnan(c)]
    lo, hi = np.percentile(v, 0.5), np.percentile(v, 99.5)
    X_tr[:,j] = np.nan_to_num(np.clip(c, lo, hi), nan=0.0)
    m, s = X_tr[:,j].mean(), X_tr[:,j].std()+1e-6
    X_tr[:,j] = (X_tr[:,j] - m)/s
    X_es[:,j] = (np.nan_to_num(np.clip(X_es[:,j], lo, hi), nan=0.0) - m)/s
    X_te[:,j] = (np.nan_to_num(np.clip(X_te[:,j], lo, hi), nan=0.0) - m)/s
gc.collect()
print(f"Data built in {time.time()-t0:.0f}s", flush=True)

params = dict(objective='binary',metric='auc',learning_rate=0.03,num_leaves=63,min_child_samples=200,
              feature_fraction=0.8,bagging_fraction=0.8,bagging_freq=5,lambda_l2=0.1,verbose=-1,n_jobs=-1)
pvs_t=[]; pvs_e=[]
for s in [42,49,56,63,70]:
    params['seed']=s
    bst = lgb.train(params, lgb.Dataset(X_tr, label=y_tr), num_boost_round=5000,
                    valid_sets=[lgb.Dataset(X_es, label=y_es)],
                    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
    pvs_t.append(bst.predict(X_te)); pvs_e.append(bst.predict(X_es))
    print(f"  seed={s}: best={bst.best_iteration} ES={roc_auc_score(y_es,pvs_e[-1]):.4f} TE={roc_auc_score(y_te,pvs_t[-1]):.4f}", flush=True)

def rank_agg(pvs):
    R=np.zeros((len(pvs),len(pvs[0])),dtype=np.float64)
    for i,p in enumerate(pvs): R[i]=np.argsort(np.argsort(np.nan_to_num(p,nan=0.5))).astype(np.float64)/(len(p)-1)
    return R.mean(0).astype(np.float32)

pv_tree = rank_agg(pvs_t)
auc_tree = roc_auc_score(y_te, pv_tree)
print(f"\n★ TREE rank-agg: TE AUC={auc_tree:.4f}", flush=True)

DAYS = (ts_te[-1]-ts_te[0])/86400.0
print(f"\nTree on own TE ({len(y_te)} anchors, ~{DAYS:.0f} days):")
for pct in [0.5,1.0,1.5,2.0,3.0]:
    k=max(1,int(len(pv_tree)*pct/100)); acc=y_te[np.argsort(-pv_tree)[:k]].mean()*100; tpd=k/DAYS
    print(f"  top-{pct:.1f}%: acc={acc:.1f}% tpd≈{tpd:.1f}", flush=True)
gc.collect()

# ========== NN on seq v6 ==========
print("\n" + "="*60 + "\nPART 2: NN MLP on v6 seq (3 configs × 5 seeds)\n" + "="*60, flush=True)
torch.manual_seed(42); np.random.seed(42)
d = np.load('/workspace/models_saved/seq_data_v6.npz')
X_tr_s = d['X_tr'].astype(np.float32); y_tr_s = d['y_tr']
X_es_s = d['X_es'].astype(np.float32); y_es_s = d['y_es']
X_te_s = d['X_te'].astype(np.float32); y_te_s = d['y_te']; ts_te_s = d['ts_te']
X_tr_f = X_tr_s.reshape(len(X_tr_s),-1); X_es_f = X_es_s.reshape(len(X_es_s),-1); X_te_f = X_te_s.reshape(len(X_te_s),-1)
FT = X_tr_f.shape[1]; del d, X_tr_s, X_es_s, X_te_s; gc.collect()

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
        m.train();idx=np.random.permutation(len(Xtr))
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

all_pvs = []
cfgs = [('A',[512,256],0.6,0.06),('B',[1024,512,256],0.6,0.06),('C',[1024,512,256],0.7,0.08)]
for name,hs,drop,wd in cfgs:
    pvs=[]
    for s in [42,49,56,63,70]:
        torch.manual_seed(s);np.random.seed(s)
        m=MLP(FT,hs,drop)
        pv_t=train_nn(m,X_tr_f,y_tr_s,X_es_f,y_es_s,ep=25,lr=5e-4,wd=wd,bs=512,pat=7,smooth=0.15)
        pvs.append(pv_t)
    all_pvs.extend(pvs)
    auc_cfg = roc_auc_score(y_te_s, rank_agg(pvs))
    print(f"  NN {name}: TE AUC={auc_cfg:.4f}", flush=True)

pv_nn = rank_agg(all_pvs)
auc_nn = roc_auc_score(y_te_s, pv_nn)
print(f"\n★ NN ENSEMBLE (15 models): TE AUC={auc_nn:.4f}", flush=True)

DAYS_s = (ts_te_s[-1]-ts_te_s[0])/86400.0
print(f"\nNN on own TE ({len(y_te_s)} anchors, ~{DAYS_s:.0f} days):")
for pct in [0.5,1.0,1.5,2.0,3.0]:
    k=max(1,int(len(pv_nn)*pct/100)); acc=y_te_s[np.argsort(-pv_nn)[:k]].mean()*100; tpd=k/DAYS_s
    print(f"  top-{pct:.1f}%: acc={acc:.1f}% tpd≈{tpd:.1f}", flush=True)

# ========== Correlation & Stacking ==========
print("\n" + "="*60 + "\nPART 3: Correlation & Stacking Analysis\n" + "="*60, flush=True)

tree_ts = ts_te.astype(np.int64); tree_pv = pv_tree; tree_y = y_te
nn_ts = ts_te_s.astype(np.int64); nn_pv = pv_nn; nn_y = y_te_s

common = np.intersect1d(tree_ts, nn_ts)
print(f"\n  Tree TE anchors: {len(tree_ts)}", flush=True)
print(f"  NN  TE anchors:  {len(nn_ts)}", flush=True)
print(f"  COMMON anchors:  {len(common)}", flush=True)

t_map = {int(tree_ts[i]): i for i in range(len(tree_ts))}
s_map = {int(nn_ts[i]): i for i in range(len(nn_ts))}
pv_tc = np.array([tree_pv[t_map[int(t)]] for t in common], dtype=np.float64)
pv_sc = np.array([nn_pv[s_map[int(t)]] for t in common], dtype=np.float64)
y_c = np.array([tree_y[t_map[int(t)]] for t in common], dtype=np.int64)

corr = np.corrcoef(pv_tc, pv_sc)[0,1]
auc_t_c = roc_auc_score(y_c, pv_tc); auc_s_c = roc_auc_score(y_c, pv_sc)
print(f"\n  ★ CORR(Tree rank, NN rank) = {corr:.4f}", flush=True)
print(f"  Tree AUC (common): {auc_t_c:.4f}", flush=True)
print(f"  NN   AUC (common): {auc_s_c:.4f}", flush=True)

def rank(a): return np.argsort(np.argsort(a)).astype(np.float64)/len(a)
rank_t = rank(pv_tc); rank_s = rank(pv_sc)

# Ensemble methods
ensembles = {
    'Tree ONLY': (pv_tc, auc_t_c),
    'NN ONLY': (pv_sc, auc_s_c),
    'PROB_AVG': ((pv_tc+pv_sc)/2, roc_auc_score(y_c,(pv_tc+pv_sc)/2)),
    'RANK_AVG': ((rank_t+rank_s)/2, roc_auc_score(y_c,(rank_t+rank_s)/2)),
}

# Optimal blend
best_w, best_a = 0.5, 0.0
for w in np.arange(0, 1.05, 0.05):
    a = roc_auc_score(y_c, w*rank_t+(1-w)*rank_s)
    if a > best_a: best_a, best_w = a, w
ensembles[f'OPT_BLEND(w={best_w:.2f}/{1-best_w:.2f})'] = (best_w*rank_t+(1-best_w)*rank_s, best_a)

# LR stacking
from sklearn.linear_model import LogisticRegression
split = int(len(y_c)*0.7)
lr = LogisticRegression(C=1.0, max_iter=200).fit(np.stack([rank_t,rank_s])[:split], y_c[:split])
pv_lr = lr.predict_proba(np.stack([rank_t,rank_s]))[:,1]
ensembles['LR_STACKING'] = (pv_lr, roc_auc_score(y_c, pv_lr))
print(f"  LR coef: Tree={lr.coef_[0][0]:.3f}, NN={lr.coef_[0][1]:.3f}", flush=True)

# Evaluate all
print(f"\n  Ensemble comparison on COMMON anchors ({len(common)}):", flush=True)
DAYS_common = (dtm.datetime.strptime('2026-09-28','%Y-%m-%d') - dtm.datetime.strptime(config.SPLITS['test'][0],'%Y-%m-%d')).days
for name, (pv, auc) in ensembles.items():
    print(f"\n  ★ [{name}] AUC={auc:.4f}", flush=True)
    for pct in [0.5, 1.0, 1.5, 2.0, 3.0]:
        k = max(1, int(len(pv)*pct/100))
        acc = y_c[np.argsort(-pv)[:k]].mean()*100; tpd = k/DAYS_common
        flag = '🏆' if pct==1.0 and acc>=65 else ('✅' if pct==1.0 and acc>=60 else '')
        print(f"    top-{pct:.1f}%: acc={acc:.1f}% tpd≈{tpd:.1f} {flag}", flush=True)

# ====== KEY SUMMARY ======
print(f"\n{'='*60}", flush=True)
print(f"📊 FINAL CORRELATION & STACKING REPORT", flush=True)
print(f"{'='*60}", flush=True)
print(f"  Tree model:        LightGBM 5-seed rank-agg", flush=True)
print(f"    AUC (own TE):    {auc_tree:.4f}", flush=True)
print(f"  NN model:          MLP(15 models: 3 cfgs × 5 seeds)", flush=True)
print(f"    AUC (own TE):    {auc_nn:.4f}", flush=True)
print(f"  CORR (rank):       {corr:.4f}", flush=True)
print(f"  Max single AUC:    {max(auc_t_c, auc_s_c):.4f}", flush=True)
print(f"  Best ensemble AUC: {best_a:.4f}", flush=True)
print(f"  Stacking gain:     +{best_a-max(auc_t_c,auc_s_c):.4f} AUC", flush=True)
print(f"  {'='*60}", flush=True)
if corr < 0.4:
    print(f"  🔥 CORR < 0.4 — STRONG diversity! Stacking WILL help!", flush=True)
elif corr < 0.6:
    print(f"  ✅ CORR 0.4-0.6 — Good diversity, stacking worthwhile", flush=True)
else:
    print(f"  ⚠️  CORR > 0.6 — Limited diversity, stacking may not help much", flush=True)

print(f"\nTOTAL TIME: {time.time()-t0:.0f}s", flush=True)
