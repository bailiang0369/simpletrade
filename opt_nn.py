"""Part 2: NN MLP 训练 + 堆叠分析"""
import sys; sys.stdout.reconfigure(line_buffering=True)
import time, gc, os
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from itertools import combinations

t0 = time.time()

# 0. 加载 Tree 预测 + 数据
print("[0] 加载 Tree 预测 + 数据...", flush=True)
D = np.load('/workspace/models_saved/tree_preds.npz')
pv_tree_mv = D['pv_tree_mv'].astype(np.float32)
pv_tree_te = D['pv_tree_te'].astype(np.float32)
y_mv = D['y_mv'].astype(np.int64)
y_te = D['y_te'].astype(np.int64)
ts_te = D['ts_te'].astype(np.int64)
X_tr = D['X_tr'].astype(np.float32)
X_es = D['X_es'].astype(np.float32)
X_mv = D['X_mv'].astype(np.float32)
X_te = D['X_te'].astype(np.float32)
y_tr = D['y_tr'].astype(np.int64)
print(f"  X_tr={X_tr.shape} X_te={X_te.shape}", flush=True)

# 子采样 NN train → 800K (省内存)
np.random.seed(42)
if len(X_tr) > 800_000:
    sel = np.random.choice(len(X_tr), 800_000, replace=False)
    X_tr_nn = X_tr[sel].copy(); y_tr_nn = y_tr[sel].copy()
    del X_tr, y_tr; gc.collect()
else:
    X_tr_nn = X_tr.copy(); y_tr_nn = y_tr.copy()

# 1. NN 特征构造 (显式交互)
print("\n[1] NN 特征构造...", flush=True)

def build_nn_features(X_tr, X_es, X_mv, X_te, y_tr, topk_feat=10):
    corrs = np.array([np.corrcoef(X_tr[:,j][~np.isnan(X_tr[:,j])], y_tr[~np.isnan(X_tr[:,j])])[0,1]
                       if np.std(X_tr[:,j][~np.isnan(X_tr[:,j])]) > 1e-8 else 0.0
                       for j in range(X_tr.shape[1])])
    corrs = np.nan_to_num(corrs, nan=0.0)
    topk = np.argsort(-np.abs(corrs))[:topk_feat]
    print(f"  Top {topk_feat} corr: {[f'{corrs[j]:.3f}' for j in topk[:5]]}...", flush=True)
    
    def _build_one(X):
        parts = [X]
        parts.append(X[:, topk] ** 2)
        for i,j in combinations(topk, 2):
            parts.append((X[:,i] * X[:,j])[:, np.newaxis])
        return np.concatenate(parts, axis=1).astype(np.float32)
    
    return _build_one(X_tr), _build_one(X_es), _build_one(X_mv), _build_one(X_te)

X_tr_nn, X_es_nn, X_mv_nn, X_te_nn = build_nn_features(X_tr_nn, X_es, X_mv, X_te, y_tr_nn, topk_feat=10)
FT_NN = X_tr_nn.shape[1]
print(f"  NN 特征维度: {FT_NN}", flush=True)
gc.collect()

# 2. 训练 MLP 8-seed
print("\n[2] MLP 8-seed 训练 [256,128] + drop=0.3", flush=True)

class MLP(nn.Module):
    def __init__(self, ft, hs, drop=0.5):
        super().__init__()
        prev=ft; layers=[]
        for h in hs: layers.extend([nn.Linear(prev,h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(drop)]); prev=h
        layers.append(nn.Linear(prev,1)); self.net=nn.Sequential(*layers)
    def forward(self,x): return self.net(x).squeeze(-1)

def evaluate_nn(m, X, bs=8192):
    m.eval(); pv=[]
    with torch.no_grad():
        for i in range(0,len(X),bs):
            pv.append(torch.sigmoid(m(torch.from_numpy(np.ascontiguousarray(X[i:i+bs])))).numpy())
    return np.concatenate(pv)

def train_mlp(m, Xtr, ytr, Xes, yes, Xmv, pat=6):
    opt = torch.optim.AdamW(m.parameters(), lr=5e-4, weight_decay=1e-3)
    best_auc = 0.0; bst_state = None; ni = 0
    for e in range(30):
        m.train(); idx = np.random.permutation(len(Xtr))
        for i in range(0, len(idx), 1024):
            bi = idx[i:i+1024]
            xb = torch.from_numpy(np.ascontiguousarray(Xtr[bi].copy()))
            yb = torch.from_numpy(ytr[bi]).float()
            yb = yb*(1-0.05) + 0.5*0.05
            logits = m(xb)
            loss = F.binary_cross_entropy_with_logits(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
        pv_es = evaluate_nn(m, Xes)
        auc_es = roc_auc_score(yes, pv_es)
        if auc_es > best_auc + 1e-5:
            best_auc = auc_es
            bst_state = {k:v.detach().clone() for k,v in m.state_dict().items()}
            ni = 0
        else:
            ni += 1
            if ni >= pat: break
    if bst_state: m.load_state_dict(bst_state)
    return evaluate_nn(m, Xes), evaluate_nn(m, Xmv), evaluate_nn(m, X_te_nn)

def rank_agg(pvs):
    R = np.zeros((len(pvs), len(pvs[0])), dtype=np.float64)
    for i,p in enumerate(pvs):
        R[i] = np.argsort(np.argsort(np.nan_to_num(p,nan=0.5))).astype(np.float64)/(len(p)-1)
    return R.mean(0).astype(np.float32)

NN_SEEDS = [42, 49, 56, 63, 70, 101, 108, 115]
pvs_nn_es = []; pvs_nn_mv = []; pvs_nn_te = []
for s in NN_SEEDS:
    torch.manual_seed(s); np.random.seed(s)
    m = MLP(FT_NN, [256, 128], drop=0.3)
    pv_es, pv_mv, pv_te = train_mlp(m, X_tr_nn, y_tr_nn, X_es_nn, D['y_es'].astype(np.int64), X_mv_nn)
    pvs_nn_es.append(pv_es); pvs_nn_mv.append(pv_mv); pvs_nn_te.append(pv_te)
    auc_te = roc_auc_score(y_te, pv_te)
    print(f"  NN seed{s}: TE AUC={auc_te:.4f}", flush=True)
    del m; gc.collect()

pv_nn_es = rank_agg(pvs_nn_es)
pv_nn_mv = rank_agg(pvs_nn_mv)
pv_nn_te = rank_agg(pvs_nn_te)
print(f"\n  ★ NN {len(NN_SEEDS)}-seed: MV AUC={roc_auc_score(y_mv, pv_nn_mv):.4f} TE AUC={roc_auc_score(y_te, pv_nn_te):.4f}", flush=True)
del pvs_nn_es, pvs_nn_mv; gc.collect()

# 3. 相关性分析
print("\n[3] Tree vs NN 相关性", flush=True)
corr_all = np.corrcoef(pv_tree_te, pv_nn_te)[0,1]
print(f"  整体 CORR: {corr_all:.4f}", flush=True)

for tail_pct in [0.5, 1.0, 2.0, 5.0]:
    k = max(1, int(len(pv_tree_te)*tail_pct/100))
    tree_top_idx = np.argsort(-pv_tree_te)[:k]
    nn_top_idx = np.argsort(-pv_nn_te)[:k]
    inter = len(np.intersect1d(tree_top_idx, nn_top_idx))
    jaccard = inter / (2*k - inter)
    tree_top_vals = pv_tree_te[tree_top_idx]
    nn_at_tree_top = pv_nn_te[tree_top_idx]
    corr_tail = np.corrcoef(tree_top_vals, nn_at_tree_top)[0,1]
    print(f"  tail top-{tail_pct}%: Jaccard={jaccard:.3f}, corr(NN@Tree top)={corr_tail:.4f}", flush=True)

# 4. Stacking 权重搜索
print("\n[4] Stacking 权重搜索 (meta_val)", flush=True)

def to_rank(pv): return np.argsort(np.argsort(pv)).astype(np.float64) / len(pv)

best_w = 0.5; best_mv_auc = 0.0
for w in np.arange(0.0, 1.05, 0.05):
    pv_blend_mv = w * to_rank(pv_tree_mv) + (1-w) * to_rank(pv_nn_mv)
    auc_mv = roc_auc_score(y_mv, pv_blend_mv)
    if auc_mv > best_mv_auc:
        best_mv_auc = auc_mv
        best_w = w

print(f"  最优权重: Tree={best_w:.2f}, NN={1-best_w:.2f}, MV AUC={best_mv_auc:.4f}", flush=True)

pv_stack_mv = best_w * to_rank(pv_tree_mv) + (1-best_w) * to_rank(pv_nn_mv)
pv_stack_te = best_w * to_rank(pv_tree_te) + (1-best_w) * to_rank(pv_nn_te)
print(f"  Stack TE AUC: {roc_auc_score(y_te, pv_stack_te):.4f}", flush=True)

# 5. 全方法 TOP-K 评估
print("\n[5] 全方法 TOP-K 评估", flush=True)
DAYS = (ts_te[-1] - ts_te[0]) / 86400.0
print(f"  Test: {DAYS:.0f} days", flush=True)

methods = {
    'Tree_5seed': pv_tree_te,
    'NN_8seed': pv_nn_te,
    f'Stack({best_w:.2f}T/{1-best_w:.2f}N)': pv_stack_te,
    'EQUAL_AVG': (to_rank(pv_tree_te) + to_rank(pv_nn_te)) / 2,
    'TREE_ONLY': to_rank(pv_tree_te),
    'NN_ONLY': to_rank(pv_nn_te),
}

for name, pv in methods.items():
    auc = roc_auc_score(y_te, pv)
    line = f"  [{name:>22s}] AUC={auc:.4f}"
    for pct in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
        k = max(1, int(len(pv)*pct/100))
        idx = np.argsort(-pv)[:k]
        acc = y_te[idx].mean() * 100
        tpd = k / DAYS
        line += f'  top{pct:>3}%={acc:.1f}%/tpd={tpd:.0f}'
    print(line, flush=True)

# 6. Regime-aware 堆叠
print("\n[6] Regime-aware 堆叠 (meta_val 选权重)", flush=True)
# 用 rvol_60 (特征第 11 列左右)
reg_mv = X_mv[:, 11] if X_mv.shape[1] > 11 else np.abs(X_mv[:, 0])
reg_te = X_te[:, 11] if X_te.shape[1] > 11 else np.abs(X_te[:, 0])
reg_thresh = np.quantile(reg_mv, 0.7)
high_vol_mv = reg_mv > reg_thresh
high_vol_te = reg_te > reg_thresh
print(f"  高波动 regime: MV={high_vol_mv.mean()*100:.1f}% TE={high_vol_te.mean()*100:.1f}", flush=True)

# 分别找最优权重
best_w_hv = 0.5; best_hv_auc = 0; best_w_lv = 0.5; best_lv_auc = 0
for w in np.arange(0.0, 1.05, 0.05):
    pv_hv = w * to_rank(pv_tree_mv[high_vol_mv]) + (1-w) * to_rank(pv_nn_mv[high_vol_mv])
    a_hv = roc_auc_score(y_mv[high_vol_mv], pv_hv)
    if a_hv > best_hv_auc: best_hv_auc = a_hv; best_w_hv = w
    pv_lv = w * to_rank(pv_tree_mv[~high_vol_mv]) + (1-w) * to_rank(pv_nn_mv[~high_vol_mv])
    a_lv = roc_auc_score(y_mv[~high_vol_mv], pv_lv)
    if a_lv > best_lv_auc: best_lv_auc = a_lv; best_w_lv = w
print(f"  高波动最优: Tree={best_w_hv:.2f}, MV AUC={best_hv_auc:.4f}", flush=True)
print(f"  低波动最优: Tree={best_w_lv:.2f}, MV AUC={best_lv_auc:.4f}", flush=True)

pv_regime_te = np.where(high_vol_te,
    best_w_hv * to_rank(pv_tree_te) + (1-best_w_hv) * to_rank(pv_nn_te),
    best_w_lv * to_rank(pv_tree_te) + (1-best_w_lv) * to_rank(pv_nn_te))
print(f"  Regime Stack TE AUC: {roc_auc_score(y_te, pv_regime_te):.4f}", flush=True)

# 加入 regime 结果
print(f"\n  [Regime_Stack] AUC={roc_auc_score(y_te, pv_regime_te):.4f}", flush=True)
for pct in [0.5, 1.0, 1.5, 2.0, 3.0]:
    k = max(1, int(len(pv_regime_te)*pct/100))
    idx = np.argsort(-pv_regime_te)[:k]
    acc = y_te[idx].mean() * 100; tpd = k/DAYS
    print(f"    top{pct:.1f}%: acc={acc:.1f}% tpd≈{tpd:.1f}", flush=True)

# 7. 保存所有预测
print(f"\n[7] 保存结果", flush=True)
np.savez('/workspace/models_saved/nn_and_stack_preds.npz',
    pv_nn_mv=pv_nn_mv, pv_nn_te=pv_nn_te,
    pv_stack_mv=pv_stack_mv, pv_stack_te=pv_stack_te,
    pv_regime_mv=np.where(high_vol_mv,
        best_w_hv * to_rank(pv_tree_mv) + (1-best_w_hv) * to_rank(pv_nn_mv),
        best_w_lv * to_rank(pv_tree_mv) + (1-best_w_lv) * to_rank(pv_nn_mv)),
    pv_regime_te=pv_regime_te,
    y_mv=y_mv, y_te=y_te, ts_te=ts_te,
    X_mv=X_mv, X_te=X_te)
print(f"  已保存", flush=True)

print(f"\n⏱ 总耗时: {time.time()-t0:.0f}s", flush=True)
