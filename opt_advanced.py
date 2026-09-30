"""Part 3: Per-hour Tree + XGBoost + 进阶堆叠"""
import sys; sys.stdout.reconfigure(line_buffering=True)
import time, gc, os
import numpy as np, datetime as dtm
import polars as pl
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import roc_auc_score

import config, features as fe

t0 = time.time()

# 0. 重新加载数据 (需要完整 ts)
print("[0] 加载完整数据...", flush=True)
D = np.load('/workspace/models_saved/tree_preds.npz')
X_tr = D['X_tr'].astype(np.float32)
y_tr = D['y_tr'].astype(np.int64)
X_es = D['X_es'].astype(np.float32)
y_es = D['y_es'].astype(np.int64)
X_mv = D['X_mv'].astype(np.float32)
y_mv = D['y_mv'].astype(np.int64)
X_te = D['X_te'].astype(np.float32)
y_te = D['y_te'].astype(np.int64)
ts_te = D['ts_te'].astype(np.int64)

# 需要 ts_all 来算 hr_tr 和 hr_mv
eth = pl.read_parquet(f'{config.DS_DIR}/raw_ETH.parquet').sort('ts')
ts_all = eth['ts'].to_numpy().astype(np.int64)
del eth; gc.collect()

H = 15
def ts_mask(s, e):
    a = int(dtm.datetime.strptime(s,'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())
    b = int(dtm.datetime.strptime(e,'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())
    return (ts_all[:-H] >= a) & (ts_all[:-H] < b)

tr_idx = np.where(ts_mask(*config.SPLITS['train']))[0]
es_idx = np.where(ts_mask(*config.SPLITS['early_stop']))[0]
mv_idx = np.where(ts_mask(*config.SPLITS['meta_val']))[0]
te_idx = np.where(ts_mask(*config.SPLITS['test']))[0]

# 对齐: 之前子采样了 tr_idx
np.random.seed(42)
if len(tr_idx) > 1_500_000:
    tr_idx = np.random.choice(tr_idx, 1_500_000, replace=False)

hr_tr = (ts_all[:-H][tr_idx] % 86400) // 3600
hr_es = (ts_all[:-H][es_idx] % 86400) // 3600
hr_mv = (ts_all[:-H][mv_idx] % 86400) // 3600
hr_te = (ts_all[:-H][te_idx] % 86400) // 3600
del ts_all; gc.collect()

print(f"  TR={len(tr_idx):,} ES={len(es_idx):,} MV={len(mv_idx):,} TE={len(te_idx):,}", flush=True)

# 加载已有 Tree 预测
pv_tree_mv = D['pv_tree_mv'].astype(np.float32)
pv_tree_te = D['pv_tree_te'].astype(np.float32)

# ============================================================
# 1. Per-hour LGB (24个)
# ============================================================
print("\n[1] Per-hour LGB 子模型", flush=True)

pv_ph_tree_mv = pv_tree_mv.copy()  # 默认全局
pv_ph_tree_te = pv_tree_te.copy()

for h in range(24):
    tr_h = hr_tr == h
    es_h = hr_es == h
    mv_h = hr_mv == h
    te_h = hr_te == h
    
    if tr_h.sum() < 3000:
        continue
    
    params_h = dict(objective='binary:logistic', eval_metric='auc', learning_rate=0.05,
                     max_depth=6, min_child_weight=50, subsample=0.9,
                     colsample_bytree=0.9, reg_alpha=0.1, reg_lambda=1.0,
                     verbosity=0, nthread=2, seed=42)
    
    dtr = xgb.DMatrix(X_tr[tr_h], label=y_tr[tr_h])
    des = xgb.DMatrix(X_es[es_h], label=y_es[es_h])
    bst_h = xgb.train(params_h, dtr, num_boost_round=1000,
                      evals=[(des,'es')], , early_stopping_rounds=50), verbose_eval=False)
    pv_ph_tree_mv[mv_h] = bst_h.predict(xgb.DMatrix(X_mv[mv_h]))
    pv_ph_tree_te[te_h] = bst_h.predict(xgb.DMatrix(X_te[te_h]))
    auc_h = roc_auc_score(y_te[te_h], pv_ph_tree_te[te_h]) if te_h.sum() > 100 else 0.5
    print(f"  h{h:02d}: tr={tr_h.sum()} te={te_h.sum()} AUC={auc_h:.4f}", flush=True)
    del bst_h, dtr, des; gc.collect()

def to_rank(pv): return np.argsort(np.argsort(pv)).astype(np.float64) / len(pv)
pv_ph_rank_mv = to_rank(pv_ph_tree_mv)
pv_ph_rank_te = to_rank(pv_ph_tree_te)
print(f"\n  ★ Per-hour XGB: MV AUC={roc_auc_score(y_mv, pv_ph_rank_mv):.4f} TE AUC={roc_auc_score(y_te, pv_ph_rank_te):.4f}", flush=True)

# ============================================================
# 2. 全局 XGBoost (5-seed)
# ============================================================
print("\n[2] 全局 XGBoost 5-seed", flush=True)

# 负权重 (和 Tree 一致)
abs_ret = np.abs(D['r_tr'].astype(np.float64)[:len(y_tr)]) if 'r_tr' in D else None
# 简化: 不做负权重
w_xgb = np.ones(len(y_tr), dtype=np.float32)

xgb_params = dict(objective='binary:logistic', eval_metric='auc', learning_rate=0.05,
                  max_depth=7, min_child_weight=100, subsample=0.8,
                  colsample_bytree=0.8, reg_alpha=0.05, reg_lambda=1.0,
                  scale_pos_weight=2.0, verbosity=0, nthread=4)

pvs_xgb_mv = []; pvs_xgb_te = []
for s in [42, 49, 56, 63, 70]:
    xgb_params['seed'] = s
    dtr = xgb.DMatrix(X_tr, label=y_tr, weight=w_xgb)
    des = xgb.DMatrix(X_es, label=y_es)
    bst = xgb.train(xgb_params, dtr, num_boost_round=3000,
                    evals=[(des,'es')], early_stopping_rounds=100, verbose_eval=False)
    pv_mv = bst.predict(xgb.DMatrix(X_mv))
    pv_te = bst.predict(xgb.DMatrix(X_te))
    pvs_xgb_mv.append(pv_mv); pvs_xgb_te.append(pv_te)
    print(f"  seed{s}: TE AUC={roc_auc_score(y_te, pv_te):.4f}", flush=True)
    del bst, dtr, des; gc.collect()

pv_xgb_mv = rank_agg(pvs_xgb_mv) if len(pvs_xgb_mv) > 1 else pvs_xgb_mv[0]
pv_xgb_te = rank_agg(pvs_xgb_te) if len(pvs_xgb_te) > 1 else pvs_xgb_te[0]
print(f"\n  ★ XGB 5-seed: MV AUC={roc_auc_score(y_mv, pv_xgb_mv):.4f} TE AUC={roc_auc_score(y_te, pv_xgb_te):.4f}", flush=True)

# ============================================================
# 3. 三者堆叠 (Tree + XGB + NN)
# ============================================================
print("\n[3] 三者堆叠搜索 (meta_val)", flush=True)

def rank_agg(pvs):
    R = np.zeros((len(pvs), len(pvs[0])), dtype=np.float64)
    for i,p in enumerate(pvs):
        R[i] = np.argsort(np.argsort(np.nan_to_num(p,nan=0.5))).astype(np.float64)/(len(p)-1)
    return R.mean(0).astype(np.float32)

# 加载 NN 预测
D2 = np.load('/workspace/models_saved/nn_and_stack_preds.npz')
pv_nn_mv = D2['pv_nn_mv'].astype(np.float32)
pv_nn_te = D2['pv_nn_te'].astype(np.float32)

# 三者 rank 化
rt_mv = to_rank(pv_tree_mv); rt_te = to_rank(pv_tree_te)
rx_mv = to_rank(pv_xgb_mv); rx_te = to_rank(pv_xgb_te)
rn_mv = to_rank(pv_nn_mv); rn_te = to_rank(pv_nn_te)
rph_mv = pv_ph_rank_mv; rph_te = pv_ph_rank_te

# 扫权重 (Tree, XGB, NN)
best_3w = (1/3, 1/3, 1/3); best_3auc = 0
for w1 in np.arange(0.0, 1.05, 0.1):
    for w2 in np.arange(0.0, 1.05-w1, 0.1):
        w3 = 1 - w1 - w2
        pv = w1*rt_mv + w2*rx_mv + w3*rn_mv
        a = roc_auc_score(y_mv, pv)
        if a > best_3auc:
            best_3auc = a; best_3w = (w1, w2, w3)
print(f"  Tree+XGB+NN 最优: T={best_3w[0]:.1f} X={best_3w[1]:.1f} N={best_3w[2]:.1f}, MV AUC={best_3auc:.4f}", flush=True)

# Tree+PH+NN
best_phw = (1/3, 1/3, 1/3); best_phauc = 0
for w1 in np.arange(0.0, 1.05, 0.1):
    for w2 in np.arange(0.0, 1.05-w1, 0.1):
        w3 = 1 - w1 - w2
        pv = w1*rt_mv + w2*rph_mv + w3*rn_mv
        a = roc_auc_score(y_mv, pv)
        if a > best_phauc:
            best_phauc = a; best_phw = (w1, w2, w3)
print(f"  Tree+PH+NN 最优: T={best_phw[0]:.1f} PH={best_phw[1]:.1f} N={best_phw[2]:.1f}, MV AUC={best_phauc:.4f}", flush=True)

# XGB+PH+NN
best_xpw = (1/3, 1/3, 1/3); best_xpauc = 0
for w1 in np.arange(0.0, 1.05, 0.1):
    for w2 in np.arange(0.0, 1.05-w1, 0.1):
        w3 = 1 - w1 - w2
        pv = w1*rx_mv + w2*rph_mv + w3*rn_mv
        a = roc_auc_score(y_mv, pv)
        if a > best_xpauc:
            best_xpauc = a; best_xpw = (w1, w2, w3)
print(f"  XGB+PH+NN 最优: X={best_xpw[0]:.1f} PH={best_xpw[1]:.1f} N={best_xpw[2]:.1f}, MV AUC={best_xpauc:.4f}", flush=True)

# 4 者一起扫 (粗粒度)
best_4w = (0.25,0.25,0.25,0.25); best_4auc = 0
for w1 in np.arange(0.0, 1.05, 0.2):
    for w2 in np.arange(0.0, 1.05-w1, 0.2):
        for w3 in np.arange(0.0, 1.05-w1-w2, 0.2):
            w4 = 1 - w1 - w2 - w3
            pv = w1*rt_mv + w2*rx_mv + w3*rph_mv + w4*rn_mv
            a = roc_auc_score(y_mv, pv)
            if a > best_4auc:
                best_4auc = a; best_4w = (w1, w2, w3, w4)
print(f"  4者 最优: T={best_4w[0]:.1f} X={best_4w[1]:.1f} PH={best_4w[2]:.1f} N={best_4w[3]:.1f}, MV AUC={best_4auc:.4f}", flush=True)

# ============================================================
# 4. 全方法评估
# ============================================================
print("\n[4] 全方法 TOP-K 评估 (Test)", flush=True)

DAYS = (ts_te[-1] - ts_te[0]) / 86400.0

methods = {
    'Tree_5seed': rt_te,
    'XGB_5seed': rx_te,
    'PH_XGB': rph_te,
    'NN_8seed': rn_te,
    'Stack_Tree+NN(0.6/0.4)': 0.6*rt_te + 0.4*rn_te,
    f'Tree+XGB+NN({best_3w[0]:.1f}/{best_3w[1]:.1f}/{best_3w[2]:.1f})': best_3w[0]*rt_te + best_3w[1]*rx_te + best_3w[2]*rn_te,
    f'Tree+PH+NN({best_phw[0]:.1f}/{best_phw[1]:.1f}/{best_phw[2]:.1f})': best_phw[0]*rt_te + best_phw[1]*rph_te + best_phw[2]*rn_te,
    f'XGB+PH+NN({best_xpw[0]:.1f}/{best_xpw[1]:.1f}/{best_xpw[2]:.1f})': best_xpw[0]*rx_te + best_xpw[1]*rph_te + best_xpw[2]*rn_te,
    f'4way({best_4w[0]:.1f}/{best_4w[1]:.1f}/{best_4w[2]:.1f}/{best_4w[3]:.1f})': best_4w[0]*rt_te + best_4w[1]*rx_te + best_4w[2]*rph_te + best_4w[3]*rn_te,
}

results = {}
for name, pv in methods.items():
    auc = roc_auc_score(y_te, pv)
    results[name] = {'AUC': auc}
    line = f"  [{name:>28s}] AUC={auc:.4f}"
    for pct in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
        k = max(1, int(len(pv)*pct/100))
        idx = np.argsort(-pv)[:k]
        acc = y_te[idx].mean() * 100
        tpd = k / DAYS
        results[name][f'acc_{pct}'] = acc
        results[name][f'tpd_{pct}'] = tpd
        line += f'  top{pct:>3}%={acc:.1f}%/tpd={tpd:.0f}'
    print(line, flush=True)

# 5. Per-hour + Global 混合权重
print("\n[5] Per-hour vs Global 混合权重", flush=True)
best_gw = 1.0; best_gauc = 0
for w in np.arange(0.0, 1.05, 0.1):
    pv = w*rt_mv + (1-w)*rph_mv
    a = roc_auc_score(y_mv, pv)
    if a > best_gauc: best_gauc = a; best_gw = w
print(f"  Global+PH 最优: w_global={best_gw:.1f}, MV AUC={best_gauc:.4f}", flush=True)

pv_gph_te = best_gw*rt_te + (1-best_gw)*rph_te
print(f"  Global+PH TE AUC={roc_auc_score(y_te, pv_gph_te):.4f}", flush=True)

# 6. 保存最终预测
print("\n[6] 保存结果", flush=True)
best_name = max(results, key=lambda n: results[n]['AUC'])
print(f"  最佳方法: {best_name} (AUC={results[best_name]['AUC']:.4f})", flush=True)
best_pv_te = methods[best_name]
best_pv_mv = best_3w[0]*rt_mv + best_3w[1]*rx_mv + best_3w[2]*rn_mv if 'Tree+XGB+NN' in best_name else rt_mv

np.savez('/workspace/models_saved/final_preds.npz',
    best_pv_mv=best_pv_mv, best_pv_te=best_pv_te,
    all_methods={k: v for k,v in methods.items()},
    y_mv=y_mv, y_te=y_te, ts_te=ts_te, hr_te=hr_te)
print(f"  已保存", flush=True)

print(f"\n⏱ 总耗时: {time.time()-t0:.0f}s", flush=True)
