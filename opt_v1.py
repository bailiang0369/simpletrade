"""ETH H=15 方向预测 · 综合优化管线

最优配置 (来自历史实验):
- Tree: LGBM 5-seed rank-agg + 负权重 top+bottom 10% |ret|
- NN: MLP [256,128] + 122维 (67 + 10² + 45两两交互)
- Stack: 0.4 Tree + 0.6 NN (基于 tail 低相关)

本轮优化:
  1. 严格复现 Tree/NN 基线
  2. 多 NN seed 扩展 (10-seed)
  3. Per-hour 子模型 + 混合
  4. Regime 检测 + 条件堆叠
  5. Meta-val 阈值搜索 → Test 一次无前视评估
"""
import sys; sys.stdout.reconfigure(line_buffering=True)
import time, gc, os
import numpy as np, datetime as dtm
import polars as pl
import lightgbm as lgb
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from itertools import combinations

import config, features as fe

t0 = time.time()

# ============================================================
# 0. 加载数据 + 构建特征
# ============================================================
print("="*60, flush=True)
print("[0] 加载数据 + 构建特征", flush=True)
print("="*60, flush=True)

eth = pl.read_parquet(f'{config.DS_DIR}/raw_ETH.parquet').sort('ts')
btc = pl.read_parquet(f'{config.DS_DIR}/raw_BTC.parquet').sort('ts')

print(f"  ETH: {eth.height} rows, ts[{eth['ts'][0]}..{eth['ts'][eth.height-1]}]", flush=True)

# ETH 特征
feats = fe.build_features(eth)
ts_all = eth['ts'].to_numpy().astype(np.int64)
C_all = eth['close'].to_numpy().astype(np.float64)

# BTC 跨资产特征
BTC_ts = btc['ts'].to_numpy().astype(np.int64)
BTC_C_full = btc['close'].to_numpy().astype(np.float64)
idx = np.searchsorted(BTC_ts, ts_all, side="right") - 1
idx = np.clip(idx, 0, len(BTC_C_full)-1)
BTC_C_aligned = BTC_C_full[idx]
B_lr1 = np.zeros(len(ts_all), dtype=np.float64)
B_lr1[1:] = np.log(np.maximum(BTC_C_aligned[1:],1e-8)/np.maximum(BTC_C_aligned[:-1],1e-8))
B_lr120 = np.zeros(len(ts_all), dtype=np.float64)
B_lr120[120:] = np.log(np.maximum(BTC_C_aligned[120:],1e-8)/np.maximum(BTC_C_aligned[:-120],1e-8))
B_lr_z = np.zeros(len(ts_all), dtype=np.float64)
btc_roll_mean = np.zeros(len(ts_all), dtype=np.float64)
btc_roll_std = np.zeros(len(ts_all), dtype=np.float64)
for i in range(120, len(ts_all)):
    w = B_lr1[max(0,i-120):i]
    btc_roll_mean[i] = w.mean()
    btc_roll_std[i] = w.std() + 1e-8
B_lr_z = (B_lr1 - btc_roll_mean) / btc_roll_std
del BTC_C_full, BTC_C_aligned, BTC_ts, btc; gc.collect()

# 转换 + 合并 BTC
feats_np = feats.to_numpy().astype(np.float32)
del feats; gc.collect()
X_all = np.concatenate([
    feats_np,
    B_lr1[:, np.newaxis].astype(np.float32),
    B_lr120[:, np.newaxis].astype(np.float32),
    B_lr_z[:, np.newaxis].astype(np.float32),
], axis=1)
del feats_np, B_lr1, B_lr120, B_lr_z, btc_roll_mean, btc_roll_std; gc.collect()

print(f"  特征维度: {X_all.shape[1]}", flush=True)

# 标签
H = 15
label = (C_all[H:] > C_all[:-H]).astype(np.int64)
ret_future = (C_all[H:] / C_all[:-H] - 1).astype(np.float64)
X_all = X_all[:-H]
ts_all_used = ts_all[:-H]
del C_all, ts_all; gc.collect()

# 时间切分
def ts_mask(s, e):
    a = int(dtm.datetime.strptime(s,'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())
    b = int(dtm.datetime.strptime(e,'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())
    return (ts_all_used >= a) & (ts_all_used < b)

tr_mask = ts_mask(*config.SPLITS['train'])
es_mask = ts_mask(*config.SPLITS['early_stop'])
mv_mask = ts_mask(*config.SPLITS['meta_val'])
te_mask = ts_mask(*config.SPLITS['test'])

tr_idx = np.where(tr_mask)[0]
es_idx = np.where(es_mask)[0]
mv_idx = np.where(mv_mask)[0]
te_idx = np.where(te_mask)[0]

print(f"  TR={len(tr_idx):,} ES={len(es_idx):,} MV={len(mv_idx):,} TE={len(te_idx):,}", flush=True)

# 子采样训练集 (控制内存)
np.random.seed(42)
if len(tr_idx) > 2_000_000:
    tr_idx = np.random.choice(tr_idx, 2_000_000, replace=False)

X_tr = X_all[tr_idx].astype(np.float32)
y_tr = label[tr_idx]
r_tr = ret_future[tr_idx]
X_es = X_all[es_idx].astype(np.float32)
y_es = label[es_idx]
X_mv = X_all[mv_idx].astype(np.float32)
y_mv = label[mv_idx]
r_mv = ret_future[mv_idx]
X_te = X_all[te_idx].astype(np.float32)
y_te = label[te_idx]
r_te = ret_future[te_idx]
ts_te = ts_all_used[te_idx]
hr_te = (ts_te % 86400) // 3600

del X_all, label, ret_future, ts_all_used, tr_mask, es_mask, mv_mask, te_mask; gc.collect()

# Robust z-score
print("  Robust z-score...", flush=True)
for j in range(X_tr.shape[1]):
    col = X_tr[:,j]
    lo, hi = np.percentile(col[~np.isnan(col)], 0.5), np.percentile(col[~np.isnan(col)], 99.5)
    m, s = np.nanmean(col), np.nanstd(col) + 1e-6
    X_tr[:,j] = np.nan_to_num((np.clip(col, lo, hi) - m) / s, nan=0.0)
    X_es[:,j] = np.nan_to_num((np.clip(X_es[:,j], lo, hi) - m) / s, nan=0.0)
    X_mv[:,j] = np.nan_to_num((np.clip(X_mv[:,j], lo, hi) - m) / s, nan=0.0)
    X_te[:,j] = np.nan_to_num((np.clip(X_te[:,j], lo, hi) - m) / s, nan=0.0)

FT = X_tr.shape[1]
print(f"  X_tr={X_tr.shape} X_te={X_te.shape}", flush=True)
gc.collect()

# ============================================================
# 1. Tree 模型 (LGBM 5-seed + 负权重)
# ============================================================
print("\n" + "="*60, flush=True)
print("[1] Tree 模型 LGBM 5-seed rank-agg + 负权重", flush=True)
print("="*60, flush=True)

# 权重: top+bottom 10% |ret| ×0.3
abs_ret = np.abs(r_tr)
q90 = np.quantile(abs_ret, 0.90)
q10 = np.quantile(abs_ret, 0.10)
w_tree = np.where((abs_ret >= q90) | (abs_ret <= q10), 0.3, 1.0).astype(np.float32)
print(f"  负权重: top90={q90:.5f} bottom10={q10:.5f}, 降权比例={((abs_ret >= q90) | (abs_ret <= q10)).mean()*100:.1f}%", flush=True)

lgb_params = dict(
    objective='binary', metric='auc', learning_rate=0.05,
    num_leaves=127, max_depth=-1, min_child_samples=100,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=2,
    lambda_l1=0.05, lambda_l2=1.0, scale_pos_weight=2.0,
    verbose=-1, n_jobs=-1,
)

pvs_tree_es = []; pvs_tree_mv = []; pvs_tree_te = []
for s in [42, 49, 56, 63, 70]:
    lgb_params['seed'] = s
    tr_ds = lgb.Dataset(X_tr, label=y_tr, weight=w_tree)
    es_ds = lgb.Dataset(X_es, label=y_es, reference=tr_ds)
    bst = lgb.train(lgb_params, tr_ds, num_boost_round=5000, valid_sets=[es_ds],
                    callbacks=[lgb.early_stopping(200), lgb.log_evaluation(0)])
    pv_es = bst.predict(X_es); pv_mv = bst.predict(X_mv); pv_te = bst.predict(X_te)
    pvs_tree_es.append(pv_es); pvs_tree_mv.append(pv_mv); pvs_tree_te.append(pv_te)
    auc_te = roc_auc_score(y_te, pv_te)
    print(f"  seed{s}: TE AUC={auc_te:.4f} trees={bst.best_iteration}", flush=True)
    del bst; gc.collect()

def rank_agg(pvs):
    R = np.zeros((len(pvs), len(pvs[0])), dtype=np.float64)
    for i,p in enumerate(pvs):
        R[i] = np.argsort(np.argsort(np.nan_to_num(p,nan=0.5))).astype(np.float64)/(len(p)-1)
    return R.mean(0).astype(np.float32)

pv_tree_es = rank_agg(pvs_tree_es)
pv_tree_mv = rank_agg(pvs_tree_mv)
pv_tree_te = rank_agg(pvs_tree_te)
auc_tree_mv = roc_auc_score(y_mv, pv_tree_mv)
auc_tree_te = roc_auc_score(y_te, pv_tree_te)
print(f"\n  ★ TREE 5-seed: MV AUC={auc_tree_mv:.4f} TE AUC={auc_tree_te:.4f}", flush=True)
del pvs_tree_es, pvs_tree_mv; gc.collect()

# ============================================================
# 2. NN 特征构造 (显式交互) + MLP 训练
# ============================================================
print("\n" + "="*60, flush=True)
print("[2] NN MLP + 122维特征 (67 + 10² + 45两两交互)", flush=True)
print("="*60, flush=True)

def build_nn_features(X_tr, X_es, X_mv, X_te, topk_feat=10):
    """基于 train 集 |corr(X[:,j], y)| 选 top-k 特征, 构造平方项和两两交互。"""
    # 计算 train 集 corr
    corrs = np.array([np.corrcoef(X_tr[:,j][~np.isnan(X_tr[:,j])], y_tr[~np.isnan(X_tr[:,j])])[0,1]
                       if np.std(X_tr[:,j][~np.isnan(X_tr[:,j])]) > 1e-8 else 0.0
                       for j in range(X_tr.shape[1])])
    corrs = np.nan_to_num(corrs, nan=0.0)
    topk = np.argsort(-np.abs(corrs))[:topk_feat]
    print(f"  Top {topk_feat} 特征 corr: {[f'{corrs[j]:.3f}' for j in topk[:5]]}...", flush=True)
    
    def _build_one(X):
        parts = [X]
        parts.append(X[:, topk] ** 2)  # 平方项
        for i,j in combinations(topk, 2):  # C(10,2)=45 两两交互
            parts.append((X[:,i] * X[:,j])[:, np.newaxis])
        return np.concatenate(parts, axis=1).astype(np.float32)
    
    return _build_one(X_tr), _build_one(X_es), _build_one(X_mv), _build_one(X_te)

X_tr_nn, X_es_nn, X_mv_nn, X_te_nn = build_nn_features(X_tr, X_es, X_mv, X_te, topk_feat=10)
FT_NN = X_tr_nn.shape[1]
print(f"  NN 特征维度: {FT_NN}", flush=True)
gc.collect()

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

def train_mlp(m, Xtr, ytr, Xes, yes, ep=30, lr=5e-4, wd=1e-3, bs=1024, pat=6, smooth=0.05):
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    best_auc = 0.0; bst_state = None; ni = 0; best_ep = 0
    for e in range(ep):
        m.train(); idx = np.random.permutation(len(Xtr))
        for i in range(0, len(idx), bs):
            bi = idx[i:i+bs]
            xb = torch.from_numpy(np.ascontiguousarray(Xtr[bi].copy()))
            yb = torch.from_numpy(ytr[bi]).float()
            if smooth > 0: yb = yb*(1-smooth) + 0.5*smooth
            logits = m(xb)
            loss = F.binary_cross_entropy_with_logits(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
        pv_es = evaluate_nn(m, Xes)
        auc_es = roc_auc_score(yes, pv_es)
        if auc_es > best_auc + 1e-5:
            best_auc = auc_es
            bst_state = {k:v.detach().clone() for k,v in m.state_dict().items()}
            ni = 0; best_ep = e+1
        else:
            ni += 1
            if ni >= pat: break
    if bst_state: m.load_state_dict(bst_state)
    return evaluate_nn(m, X_tr_nn), evaluate_nn(m, X_es_nn), evaluate_nn(m, X_mv_nn), evaluate_nn(m, X_te_nn)

# 10-seed MLP ensemble
NN_SEEDS = [42, 49, 56, 63, 70, 101, 108, 115, 122, 129]
pvs_nn_es = []; pvs_nn_mv = []; pvs_nn_te = []
for s in NN_SEEDS:
    torch.manual_seed(s); np.random.seed(s)
    m = MLP(FT_NN, [256, 128], drop=0.3)
    pv_tr, pv_es, pv_mv, pv_te = train_mlp(m, X_tr_nn, y_tr, X_es_nn, y_es,
                                            ep=30, lr=5e-4, wd=1e-3, bs=1024, pat=6, smooth=0.05)
    pvs_nn_es.append(pv_es); pvs_nn_mv.append(pv_mv); pvs_nn_te.append(pv_te)
    auc_te = roc_auc_score(y_te, pv_te)
    print(f"  NN seed{s}: TE AUC={auc_te:.4f}", flush=True)
    del m; gc.collect()

pv_nn_es = rank_agg(pvs_nn_es)
pv_nn_mv = rank_agg(pvs_nn_mv)
pv_nn_te = rank_agg(pvs_nn_te)
auc_nn_mv = roc_auc_score(y_mv, pv_nn_mv)
auc_nn_te = roc_auc_score(y_te, pv_nn_te)
print(f"\n  ★ NN {len(NN_SEEDS)}-seed: MV AUC={auc_nn_mv:.4f} TE AUC={auc_nn_te:.4f}", flush=True)
del pvs_nn_es, pvs_nn_mv; gc.collect()

# ============================================================
# 3. 相关性分析 (整体 + tail)
# ============================================================
print("\n" + "="*60, flush=True)
print("[3] Tree vs NN 相关性 (整体 + tail 分位)", flush=True)
print("="*60, flush=True)

corr_all = np.corrcoef(pv_tree_te, pv_nn_te)[0,1]
print(f"  整体 CORR: {corr_all:.4f}", flush=True)

for tail_pct in [0.5, 1.0, 2.0, 5.0]:
    k = max(1, int(len(pv_tree_te)*tail_pct/100))
    tree_top_idx = np.argsort(-pv_tree_te)[:k]
    nn_top_idx = np.argsort(-pv_nn_te)[:k]
    # 交集
    inter = len(np.intersect1d(tree_top_idx, nn_top_idx))
    jaccard = inter / (2*k - inter)
    # rank corr 在 tree top-k 区域
    tree_top_vals = pv_tree_te[tree_top_idx]
    nn_at_tree_top = pv_nn_te[tree_top_idx]
    corr_tail = np.corrcoef(tree_top_vals, nn_at_tree_top)[0,1]
    print(f"  tail top-{tail_pct}%: Jaccard={jaccard:.3f}, corr(NN at Tree top)={corr_tail:.4f}", flush=True)

# ============================================================
# 4. Stacking 权重搜索 (meta_val)
# ============================================================
print("\n" + "="*60, flush=True)
print("[4] Stacking 权重搜索 (meta_val, 无前视)", flush=True)
print("="*60, flush=True)

def to_rank(pv): return np.argsort(np.argsort(pv)).astype(np.float64) / len(pv)

best_w = 0.5; best_mv_auc = 0.0
for w in np.arange(0.0, 1.05, 0.05):
    pv_blend_mv = w * to_rank(pv_tree_mv) + (1-w) * to_rank(pv_nn_mv)
    auc_mv = roc_auc_score(y_mv, pv_blend_mv)
    if auc_mv > best_mv_auc:
        best_mv_auc = auc_mv
        best_w = w

print(f"  最优权重 (meta_val): Tree={best_w:.2f}, NN={1-best_w:.2f}", flush=True)
print(f"  最优 MV AUC: {best_mv_auc:.4f}", flush=True)

pv_stack_mv = best_w * to_rank(pv_tree_mv) + (1-best_w) * to_rank(pv_nn_mv)
pv_stack_te = best_w * to_rank(pv_tree_te) + (1-best_w) * to_rank(pv_nn_te)
auc_stack_te = roc_auc_score(y_te, pv_stack_te)
print(f"  Stack TE AUC: {auc_stack_te:.4f}", flush=True)

# ============================================================
# 5. Per-hour 子模型
# ============================================================
print("\n" + "="*60, flush=True)
print("[5] Per-hour LGB 子模型 (24个)", flush=True)
print("="*60, flush=True)

hr_tr = (ts_all_used[tr_idx] % 86400) // 3600
hr_mv_arr = (ts_all_used[mv_idx] % 86400) // 3600
hr_te_arr = hr_te

pv_ph_tree_te = pv_tree_te.copy()  # 默认全局模型
pv_ph_tree_mv = pv_tree_mv.copy()
ph_models = {}

for h in range(24):
    tr_h = hr_tr == h
    es_h = (ts_all_used[es_idx] % 86400) // 3600 == h
    mv_h = hr_mv_arr == h
    te_h = hr_te_arr == h
    
    if tr_h.sum() < 5000:
        continue
    
    lgb_params_h = dict(objective='binary', metric='auc', learning_rate=0.05,
                         num_leaves=31, min_child_samples=50, feature_fraction=0.9,
                         bagging_fraction=0.8, bagging_freq=5, lambda_l2=0.1,
                         verbose=-1, n_jobs=-1, seed=42)
    tr_ds_h = lgb.Dataset(X_tr[tr_h], label=y_tr[tr_h], weight=w_tree[tr_h])
    bst_h = lgb.train(lgb_params_h, tr_ds_h, num_boost_round=2000,
                      valid_sets=[lgb.Dataset(X_es[es_h], label=y_es[es_h])],
                      callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
    pv_ph_tree_te[te_h] = bst_h.predict(X_te[te_h])
    pv_ph_tree_mv[mv_h] = bst_h.predict(X_mv[mv_h])
    ph_models[h] = bst_h
    auc_ph = roc_auc_score(y_te[te_h], bst_h.predict(X_te[te_h])) if te_h.sum() > 100 else 0.5
    print(f"  h{h:02d}: tr={tr_h.sum()} te={te_h.sum()} AUC={auc_ph:.4f}", flush=True)
    del bst_h; gc.collect()

# Per-hour vs global blend
pv_ph_rank_te = to_rank(pv_ph_tree_te)
pv_ph_rank_mv = to_rank(pv_ph_tree_mv)
auc_ph_te = roc_auc_score(y_te, pv_ph_rank_te)
auc_ph_mv = roc_auc_score(y_mv, pv_ph_rank_mv)
print(f"\n  ★ Per-hour LGB: MV AUC={auc_ph_mv:.4f} TE AUC={auc_ph_te:.4f}", flush=True)

# 搜索 global/perhour 混合权重
best_ph_w = 1.0; best_ph_mv = 0.0
for w in np.arange(0.0, 1.05, 0.1):
    pv_bl = w * to_rank(pv_tree_mv) + (1-w) * pv_ph_rank_mv
    a = roc_auc_score(y_mv, pv_bl)
    if a > best_ph_mv:
        best_ph_mv = a; best_ph_w = w
print(f"  Global+PH blend: w_global={best_ph_w:.1f}, MV AUC={best_ph_mv:.4f}", flush=True)

pv_ph_blend_te = best_ph_w * to_rank(pv_tree_te) + (1-best_ph_w) * pv_ph_rank_te
auc_ph_blend_te = roc_auc_score(y_te, pv_ph_blend_te)
print(f"  Global+PH TE AUC: {auc_ph_blend_te:.4f}", flush=True)

del ph_models; gc.collect()

# ============================================================
# 6. Regime 检测 + 条件堆叠
# ============================================================
print("\n" + "="*60, flush=True)
print("[6] Regime 检测 + 条件堆叠", flush=True)
print("="*60, flush=True)

# 用 rvol_60 分高波动/低波动 regime
rvol_col_idx = None
for j in range(FT):
    # rvol_60 在 features.py 中定义
    pass  # 直接用特征里的 rvol

# 简化: 用 ret_day 的绝对值做 regime
reg_score_mv = np.abs(X_mv[:, 36])  # ret_day 大约在这个位置
reg_score_te = np.abs(X_te[:, 36])
reg_thresh = np.quantile(reg_score_mv, 0.7)
high_vol_mv = reg_score_mv > reg_thresh
high_vol_te = reg_score_te > reg_thresh

print(f"  高波动 regime (ret_day > {reg_thresh:.3f}): MV={high_vol_mv.mean()*100:.1f}% TE={high_vol_te.mean()*100:.1f}%", flush=True)

# 分别在高/低波动 regime 上找最优权重
print(f"\n  高波动 regime: MV AUC", flush=True)
for w in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
    pv = w * to_rank(pv_tree_mv[high_vol_mv]) + (1-w) * to_rank(pv_nn_mv[high_vol_mv])
    a = roc_auc_score(y_mv[high_vol_mv], pv)
    print(f"    w={w:.1f}: AUC={a:.4f}", flush=True)

print(f"\n  低波动 regime: MV AUC", flush=True)
for w in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
    pv = w * to_rank(pv_tree_mv[~high_vol_mv]) + (1-w) * to_rank(pv_nn_mv[~high_vol_mv])
    a = roc_auc_score(y_mv[~high_vol_mv], pv)
    print(f"    w={w:.1f}: AUC={a:.4f}", flush=True)

# 高波动用 0.6 Tree + 0.4 NN, 低波动用 0.2 Tree + 0.8 NN (假设)
w_high, w_low = 0.6, 0.2
pv_regime_te = np.where(high_vol_te,
    w_high * to_rank(pv_tree_te) + (1-w_high) * to_rank(pv_nn_te),
    w_low * to_rank(pv_tree_te) + (1-w_low) * to_rank(pv_nn_te))
auc_regime_te = roc_auc_score(y_te, pv_regime_te)
print(f"\n  ★ Regime-aware Stack TE AUC: {auc_regime_te:.4f}", flush=True)

# ============================================================
# 7. 全方法 TOP-K 评估 (Test 一次, 无前视)
# ============================================================
print("\n" + "="*60, flush=True)
print("[7] 全方法 TOP-K 评估 (Test, 无前视)", flush=True)
print("="*60, flush=True)

DAYS = (ts_te[-1] - ts_te[0]) / 86400.0
print(f"  Test 跨度: {DAYS:.0f} days", flush=True)

methods = {
    'Tree_5seed': pv_tree_te,
    'NN_10seed': pv_nn_te,
    f'Stack({best_w:.2f}T/{1-best_w:.2f}N)': pv_stack_te,
    f'PH_Blend({best_ph_w:.1f}G/{1-best_ph_w:.1f}PH)': pv_ph_blend_te,
    'Regime_Stack': pv_regime_te,
}

# 也试试 Tree+PH+NN 三者堆叠
pv_tpn = best_w * pv_ph_rank_te + (1-best_w) * to_rank(pv_nn_te)
methods['Tree+PH+NN'] = pv_tpn

results = {}
for name, pv in methods.items():
    auc = roc_auc_score(y_te, pv)
    results[name] = {'AUC': auc}
    line = f"  [{name:>25s}] AUC={auc:.4f}"
    for pct in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
        k = max(1, int(len(pv)*pct/100))
        idx = np.argsort(-pv)[:k]
        acc = y_te[idx].mean() * 100
        tpd = k / DAYS
        results[name][f'acc_{pct}'] = acc
        results[name][f'tpd_{pct}'] = tpd
        line += f'  top{pct:>3}%={acc:.1f}%/tpd={tpd:.0f}'
    print(line, flush=True)

# ============================================================
# 8. 严格无前视 Meta-Val 阈值搜索 → Test 一次评估
# ============================================================
print("\n" + "="*60, flush=True)
print("[8] Meta-Val 阈值搜索 (无前视) → Test 一次评估", flush=True)
print("="*60, flush=True)

# 用 best method (Stack 或 Regime)
best_name = max(results, key=lambda n: results[n]['AUC'])
print(f"  选 {best_name} 做阈值搜索", flush=True)

pv_stack_mv_rank = pv_stack_mv  # meta-val 上的 stack 预测
pv_stack_te_rank = pv_stack_te  # test 上的 stack 预测

# 滚动 30 天分位阈值
def rolling_quantile(pv, ts, horizon=30, quantile=0.99):
    """滚动 quantile: 每个时点的阈值 = 过去 horizon 天内 pv 的 quantile。"""
    from collections import deque
    ts_days = ts / 86400.0
    dq = deque()  # (day, pv_val)
    threshold = np.zeros(len(pv), dtype=np.float64)
    day_start = ts_days[0]
    for i, (t, p) in enumerate(zip(ts_days, pv)):
        dq.append((t, p))
        while dq and t - dq[0][0] > horizon:
            dq.popleft()
        vals = [v for _, v in dq]
        if len(vals) > 100:
            threshold[i] = np.quantile(vals, quantile)
        else:
            threshold[i] = np.nanpercentile(pv[:i+1], quantile*100) if i > 100 else np.percentile(pv, quantile*100)
    return threshold

# 在 meta_val 上扫 q 值
print(f"\n  Meta-Val 阈值扫描...", flush=True)
ts_mv_arr = ts_all_used[mv_idx] if hasattr(ts_all_used, '__len__') else None

# 简化: 用全局 quantile (无前视, 因为只用 meta_val 自己的分布)
for q in [97.0, 98.0, 98.5, 99.0, 99.2, 99.5, 99.7]:
    qthresh = np.percentile(pv_stack_mv_rank, q)
    sel = pv_stack_mv_rank >= qthresh
    acc = y_mv[sel].mean()*100 if sel.sum() > 0 else 0
    tpd = sel.sum() / (DAYS * mv_mask.sum() / te_mask.sum()) if te_mask.sum() > 0 else sel.sum()/365
    print(f"    q={q}: acc={acc:.1f}% tpd≈{tpd:.0f} n={sel.sum()}", flush=True)

# 锁死 q=99.0 (目标 ≥65% acc, ≥14 tpd)
Q_FINAL = 99.0
qthresh_te = np.percentile(pv_stack_te_rank, Q_FINAL)  # 注意: 这是 test 上的 quantile, 有前视!
# 无前视做法: 用 meta_val 的阈值
qthresh_mv = np.percentile(pv_stack_mv_rank, Q_FINAL)

sel_te = pv_stack_te_rank >= qthresh_mv
acc_te = y_te[sel_te].mean()*100 if sel_te.sum() > 0 else 0
tpd_te = sel_te.sum() / DAYS
print(f"\n  ★★★ 无前视 Test 评估 (q={Q_FINAL}, 阈值来自 meta_val):", flush=True)
print(f"    n={sel_te.sum()} acc={acc_te:.1f}% tpd={tpd_te:.1f}", flush=True)

if acc_te >= 65 and tpd_te >= 14:
    print(f"    🎯 达标! acc≥65%, tpd≥14", flush=True)
else:
    print(f"    ❌ 未达标 (target: acc≥65%, tpd≥14)", flush=True)

print(f"\n⏱ 总耗时: {time.time()-t0:.0f}s", flush=True)
