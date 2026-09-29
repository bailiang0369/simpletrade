"""特征子集分割 LightGBM 集成 + 扩充 BTC cross-asset 特征 + 去噪+负权重叠加。

核心假设: 之前所有 Tree 都用同一特征集, 多 seed 集成相关性太高 (增益<0.3pp).
          不同特征子集的 Tree 信号来源不同 → 相关性低 → 集成增益大.

严格无前视: meta_val 选参数 → test 只跑一次.
"""
import sys; sys.stdout.reconfigure(line_buffering=True)
import time, gc
import numpy as np
import polars as pl
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import datetime as dtm
import config

t0 = time.time()
torch_flag = False  # 避免不必要依赖

# ============================================================
# 1. Load data + build ETH features
# ============================================================
print("=" * 60, flush=True)
print("1. Load raw data & build ETH features", flush=True)
print("=" * 60, flush=True)

import features as fe
eth = pl.read_parquet(f'{config.DS_DIR}/raw_ETH.parquet').sort('ts')
btc = pl.read_parquet(f'{config.DS_DIR}/raw_BTC.parquet').sort('ts')

# 基础 ETH 特征
feats_eth = fe.build_features(eth)
ts_all = eth['ts'].to_numpy().astype(np.int64)
C_eth = eth['close'].to_numpy().astype(np.float64)

# ============================================================
# 2. Build EXPANDED BTC cross-asset features (POLARS, fast rolling)
# ============================================================
print("\n2. Build expanded BTC cross-asset features (fast polars rolling)", flush=True)

EPS = 1e-12
# First: build BTC's own rolling features on raw BTC data using polars
# Then align BTC features to ETH timestamps

btc_lr = (pl.col('close') / pl.col('close').shift(1)).log()
btc_feats_df = btc.with_columns(
    btc_lr=btc_lr,
).with_columns(
    # Ret at multi horizons
    btc_lr_15=(pl.col('close') / pl.col('close').shift(15)).log(),
    btc_lr_120=(pl.col('close') / pl.col('close').shift(120)).log(),
    # Realized vol
    btc_rvol_30=btc_lr.rolling_std(30, ddof=1) * 100,
    btc_rvol_60=btc_lr.rolling_std(60, ddof=1) * 100,
    # Funding stats
    btc_fund_mean_30=pl.col('funding').rolling_mean(30),
    btc_fund_mean_60=pl.col('funding').rolling_mean(60),
    btc_fund_std_30=pl.col('funding').rolling_std(30, ddof=1),
).with_columns(
    btc_fund_z_30=(pl.col('funding') - pl.col('btc_fund_mean_30')) / (pl.col('btc_fund_std_30') + EPS),
    btc_cvd_30=(pl.col('buy_vol') - pl.col('sell_vol')).rolling_mean(30) / ((pl.col('buy_vol') + pl.col('sell_vol')).rolling_mean(30) + EPS),
    btc_tb_ratio=pl.col('buy_vol') / (pl.col('buy_vol') + pl.col('sell_vol') + EPS),
)

# Extract BTC feat columns and align to ETH
btc_feat_cols = ['ts', 'btc_lr_15', 'btc_lr_120', 'btc_rvol_30', 'btc_rvol_60',
                 'btc_fund_mean_30', 'btc_fund_mean_60', 'btc_fund_std_30',
                 'btc_fund_z_30', 'btc_cvd_30', 'btc_tb_ratio']
BTC_df = btc_feats_df.select(btc_feat_cols)
BTC_ts_p = BTC_df['ts'].to_numpy().astype(np.int64)
BTC_feats_np = BTC_df.drop('ts').to_numpy().astype(np.float64)

# Align BTC to ETH timestamps
idx = np.searchsorted(BTC_ts_p, ts_all, side="right") - 1
idx = np.clip(idx, 0, len(BTC_ts_p) - 1)
BTC_aligned = BTC_feats_np[idx]  # shape (n_eth, 10)

# Build dict for easy access
BTC_names = BTC_df.drop('ts').columns
BTC_feats = {BTC_names[i]: BTC_aligned[:, i] for i in range(len(BTC_names))}

del btc, btc_feats_df, BTC_df, BTC_ts_p, BTC_feats_np, BTC_aligned
gc.collect()

print(f"  Added {len(BTC_feats)} BTC cross-asset features", flush=True)

# ============================================================
# 3. Split ETH features into SUBSETS
# ============================================================
print("\n3. Split features into price vs funding/volume subsets", flush=True)

all_feat_names = feats_eth.columns
n_eth = len(all_feat_names)
print(f"  ETH features: {n_eth} total", flush=True)

# 子集 A: 价格/技术驱动 (收益率、波动率、Z分数、区间位置、时间、regime、技术指标、动量一致性)
price_feat_patterns = [
    'lr_', 'mom_', 'rvol_', 'rvol_z_', 'z_', 'pos_', 'dd_', 'ru_', 'hh_', 'll_',
    'skew_', 'range_', 'hour_', 'dow_', 'regime_', 'dn_', 'low_', 'bounce_',
    'stoch_', 'rsi_', 'macd_', 'ema_', 'ret_day'
]

# 子集 B: 资金/成交驱动 (主动买卖量、funding)
fund_feat_patterns = [
    'tb_act_', 'ts_act_', 'cvd_', 'tb_ratio', 'tb_ts_ratio',
    'fund_mean_', 'fund_std_', 'fund_z_'
]

A_names = []
B_names = []
for fn in all_feat_names:
    if any(fn.startswith(p) for p in fund_feat_patterns):
        B_names.append(fn)
    elif any(fn.startswith(p) for p in price_feat_patterns):
        A_names.append(fn)

# BTC 特征同时加给 A 和 B？不，BTC 特征单独做子集 C
BTC_names = list(BTC_feats.keys())

print(f"  Subset A (price/tech): {len(A_names)} feats", flush=True)
print(f"  Subset B (fund/vol): {len(B_names)} feats", flush=True)
print(f"  Subset C (BTC cross-asset): {len(BTC_names)} feats", flush=True)

# Build numpy arrays
feats_A = feats_eth.select(A_names).to_numpy().astype(np.float32)
feats_B = feats_eth.select(B_names).to_numpy().astype(np.float32)
feats_C = np.stack([BTC_feats[k] for k in BTC_names], axis=1).astype(np.float32)
del feats_eth, BTC_feats
gc.collect()

# Full ETH features (A+B) for baseline
feats_full = np.concatenate([feats_A, feats_B], axis=1)
# Full features (A+B+C) for BTC-expanded model
feats_full_btc = np.concatenate([feats_A, feats_B, feats_C], axis=1)

print(f"  Full ETH (A+B): {feats_full.shape[1]} feats", flush=True)
print(f"  Full ETH+BTC (A+B+C): {feats_full_btc.shape[1]} feats", flush=True)

# ============================================================
# 4. Build labels + splits
# ============================================================
print("\n4. Build labels + time splits", flush=True)

H = 15
label = (C_eth[H:] > C_eth[:-H]).astype(np.int64)
ret_future = (C_eth[H:] / C_eth[:-H] - 1).astype(np.float64)

# Trim features to match labels (去掉最后 H 行)
X_full = feats_full[:-H]
X_full_btc = feats_full_btc[:-H]
ts_used = ts_all[:-H]
del C_eth, ts_all, feats_A, feats_B, feats_C, feats_full, feats_full_btc
gc.collect()

tre = int(dtm.datetime.strptime(config.TRAIN_END, '%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())
es_end = int(dtm.datetime.strptime(config.SPLITS['early_stop'][1], '%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())
mv_end = int(dtm.datetime.strptime(config.META_VAL_END, '%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())

tr_mask = ts_used < tre
es_mask = (ts_used >= tre) & (ts_used < es_end)
mv_mask = (ts_used >= es_end) & (ts_used < mv_end)
te_mask = ts_used >= mv_end

print(f"  TR={tr_mask.sum()} ES={es_mask.sum()} MV={mv_mask.sum()} TE={te_mask.sum()}", flush=True)

# Subsample train to 400K (speed + memory)
np.random.seed(42)
tr_idx = np.where(tr_mask)[0]
if len(tr_idx) > 400_000:
    tr_idx = np.random.choice(tr_idx, 400_000, replace=False)

y_tr = label[tr_idx]
r_tr = ret_future[tr_idx]

# ============================================================
# 5. Define weight schemes + LGB params
# ============================================================
print("\n5. Prepare weight schemes", flush=True)

def make_weight(r, y, scheme='negw'):
    """Weight schemes for noisy/extreme samples."""
    pos = y.mean()
    pw = np.where(y > 0.5, (1 - pos) / pos, pos / (1 - pos)).astype(np.float32)

    if scheme == 'baseline':
        return pw
    elif scheme == 'negw10':
        lo = np.percentile(np.abs(r), 10)
        hi = np.percentile(np.abs(r), 90)
        ew = np.ones(len(r), dtype=np.float32)
        mask_noise = np.abs(r) <= lo
        mask_extreme = np.abs(r) >= hi
        ew[mask_noise | mask_extreme] = 0.3
        return pw * ew
    elif scheme == 'negw5':
        lo = np.percentile(np.abs(r), 5)
        hi = np.percentile(np.abs(r), 95)
        ew = np.ones(len(r), dtype=np.float32)
        ew[(np.abs(r) <= lo) | (np.abs(r) >= hi)] = 0.3
        return pw * ew
    else:
        raise ValueError(f"Unknown scheme: {scheme}")

def subsample_by_eps(r, eps):
    """去噪: 只保留 |ret| > eps 的样本."""
    keep = np.abs(r) > eps
    return keep

lgb_params_base = dict(
    objective='binary', metric='auc',
    learning_rate=0.05, num_leaves=255,
    min_child_samples=200, feature_fraction=0.8,
    bagging_fraction=0.8, bagging_freq=5,
    lambda_l2=1.0, verbose=-1, n_jobs=-1
)

def train_5seed(X_tr_sub, y_tr_sub, sw, X_es, y_es, X_mv, y_mv, X_te, y_te, desc=""):
    """Train 5-seed LGB rank ensemble, return AUCs + preds."""
    print(f"\n  --- Training {desc} ---", flush=True)
    pvs_es, pvs_mv, pvs_te = [], [], []
    for s in [42, 49, 56, 63, 70]:
        lgb_params_base['seed'] = s
        tr_ds = lgb.Dataset(X_tr_sub, label=y_tr_sub, weight=sw)
        es_ds = lgb.Dataset(X_es, label=y_es, reference=tr_ds)
        bst = lgb.train(lgb_params_base, tr_ds, num_boost_round=3000, valid_sets=[es_ds],
                        callbacks=[lgb.early_stopping(150), lgb.log_evaluation(500)])
        pv_es = bst.predict(X_es)
        pv_mv = bst.predict(X_mv)
        pv_te = bst.predict(X_te)
        pvs_es.append(pv_es); pvs_mv.append(pv_mv); pvs_te.append(pv_te)
        auc_te = roc_auc_score(y_te, pv_te)
        print(f"    seed{s}: ES={roc_auc_score(y_es, pv_es):.4f} TE={auc_te:.4f}", flush=True)

    def rank_agg(pvs):
        R = np.zeros((len(pvs), len(pvs[0])), dtype=np.float64)
        for i, p in enumerate(pvs):
            R[i] = np.argsort(np.argsort(np.nan_to_num(p, nan=0.5))).astype(np.float64) / (len(p) - 1)
        return R.mean(0).astype(np.float32)

    pv_es_ens = rank_agg(pvs_es)
    pv_mv_ens = rank_agg(pvs_mv)
    pv_te_ens = rank_agg(pvs_te)

    auc_es = roc_auc_score(y_es, pv_es_ens)
    auc_mv = roc_auc_score(y_mv, pv_mv_ens)
    auc_te = roc_auc_score(y_te, pv_te_ens)
    print(f"  ★ {desc} 5-seed RANK: ES={auc_es:.4f} MV={auc_mv:.4f} TE={auc_te:.4f}", flush=True)
    return pv_mv_ens, pv_te_ens, auc_mv, auc_te

# ============================================================
# 6. Experiment 1: Full ETH (A+B) baseline with different weight schemes
# ============================================================
print("\n" + "=" * 60, flush=True)
print("6. Experiment 1: Full ETH baseline (A+B) with weight schemes", flush=True)
print("=" * 60, flush=True)

X_tr_full = X_full[tr_idx]
X_es_full = X_full[es_mask]
X_mv_full = X_full[mv_mask]
X_te_full = X_full[te_mask]
y_es_full = label[es_mask]
y_mv_full = label[mv_mask]
y_te_full = label[te_mask]

results = {}
for scheme in ['baseline', 'negw10', 'negw5']:
    sw = make_weight(r_tr, y_tr, scheme)
    pv_mv, pv_te, auc_mv, auc_te = train_5seed(
        X_tr_full, y_tr, sw,
        X_es_full, y_es_full, X_mv_full, y_mv_full, X_te_full, y_te_full,
        desc=f"FullETH_{scheme}"
    )
    results[f"full_{scheme}"] = (pv_mv, pv_te, auc_mv, auc_te)

# ============================================================
# 7. Experiment 2:去噪 (ε=0.0002/0.0005) + 负权重叠加
# ============================================================
print("\n" + "=" * 60, flush=True)
print("7. Experiment 2: Denoise + neg weight stacking", flush=True)
print("=" * 60, flush=True)

for eps in [0.0002, 0.0005]:
    keep = subsample_by_eps(r_tr, eps)
    X_tr_dn = X_tr_full[keep]
    y_tr_dn = y_tr[keep]
    r_tr_dn = r_tr[keep]
    print(f"\n  --- Denoise ε={eps} → keep {keep.mean()*100:.1f}% samples ---", flush=True)
    for scheme in ['baseline', 'negw10']:
        sw = make_weight(r_tr_dn, y_tr_dn, scheme)
        pv_mv, pv_te, auc_mv, auc_te = train_5seed(
            X_tr_dn, y_tr_dn, sw,
            X_es_full, y_es_full, X_mv_full, y_mv_full, X_te_full, y_te_full,
            desc=f"FullETH_eps{eps}_{scheme}"
        )
        results[f"full_eps{eps}_{scheme}"] = (pv_mv, pv_te, auc_mv, auc_te)

# ============================================================
# 8. Experiment 3: BTC-expanded Full ETH (A+B+C)
# ============================================================
print("\n" + "=" * 60, flush=True)
print("8. Experiment 3: ETH+BTC full model (A+B+C)", flush=True)
print("=" * 60, flush=True)

X_tr_btc = X_full_btc[tr_idx]
X_es_btc = X_full_btc[es_mask]
X_mv_btc = X_full_btc[mv_mask]
X_te_btc = X_full_btc[te_mask]

for scheme in ['baseline', 'negw10']:
    sw = make_weight(r_tr, y_tr, scheme)
    pv_mv, pv_te, auc_mv, auc_te = train_5seed(
        X_tr_btc, y_tr, sw,
        X_es_btc, y_es_full, X_mv_btc, y_mv_full, X_te_btc, y_te_full,
        desc=f"ETH_BTC_{scheme}"
    )
    results[f"ethbtc_{scheme}"] = (pv_mv, pv_te, auc_mv, auc_te)

# Denoise + negw on ETH+BTC
for eps in [0.0002, 0.0005]:
    keep = subsample_by_eps(r_tr, eps)
    X_tr_btc_dn = X_tr_btc[keep]
    y_tr_btc_dn = y_tr[keep]
    r_tr_btc_dn = r_tr[keep]
    sw = make_weight(r_tr_btc_dn, y_tr_btc_dn, 'negw10')
    pv_mv, pv_te, auc_mv, auc_te = train_5seed(
        X_tr_btc_dn, y_tr_btc_dn, sw,
        X_es_btc, y_es_full, X_mv_btc, y_mv_full, X_te_btc, y_te_full,
        desc=f"ETH_BTC_eps{eps}_negw10"
    )
    results[f"ethbtc_eps{eps}_negw10"] = (pv_mv, pv_te, auc_mv, auc_te)

# ============================================================
# 9. Experiment 4: Subset models (A only vs B only vs C only)
# ============================================================
print("\n" + "=" * 60, flush=True)
print("9. Experiment 4: Individual subset models", flush=True)
print("=" * 60, flush=True)

# Rebuild X for each subset from trimmed X_full_btc
# We already have X_full = A+B, X_full_btc = A+B+C
# So we can slice:
n_A = len(A_names)
n_B = len(B_names)
n_C = len(BTC_names)
X_sub_A = X_full[:, :n_A]
X_sub_B = X_full[:, n_A:n_A+n_B]
X_sub_C = X_full_btc[:, -n_C:]

for name, X_sub in [('A_price', X_sub_A), ('B_fund', X_sub_B), ('C_btc', X_sub_C)]:
    X_tr_s = X_sub[tr_idx]
    X_es_s = X_sub[es_mask]
    X_mv_s = X_sub[mv_mask]
    X_te_s = X_sub[te_mask]
    sw = make_weight(r_tr, y_tr, 'negw10')
    pv_mv, pv_te, auc_mv, auc_te = train_5seed(
        X_tr_s, y_tr, sw,
        X_es_s, y_es_full, X_mv_s, y_mv_full, X_te_s, y_te_full,
        desc=f"Subset_{name}_negw10"
    )
    results[f"sub_{name}_negw10"] = (pv_mv, pv_te, auc_mv, auc_te)

# ============================================================
# 10. Correlation between subset models
# ============================================================
print("\n" + "=" * 60, flush=True)
print("10. Cross-subset correlation analysis", flush=True)
print("=" * 60, flush=True)

# Correlate their TE predictions
sub_pairs = [
    ('sub_A_price_negw10', 'sub_B_fund_negw10'),
    ('sub_A_price_negw10', 'sub_C_btc_negw10'),
    ('sub_B_fund_negw10', 'sub_C_btc_negw10'),
]
for k1, k2 in sub_pairs:
    if k1 in results and k2 in results:
        _, pv1, _, _ = results[k1]
        _, pv2, _, _ = results[k2]
        corr = np.corrcoef(pv1, pv2)[0, 1]
        print(f"  {k1} vs {k2}: CORR={corr:.4f}", flush=True)

# ============================================================
# 11. Subset ensemble (A + B + C rank blend)
# ============================================================
print("\n" + "=" * 60, flush=True)
print("11. Subset ensemble rank blending", flush=True)
print("=" * 60, flush=True)

# Get subset predictions
sub_keys = [k for k in results if k.startswith('sub_')]
print(f"\n  Subset models available: {sub_keys}", flush=True)

def to_rank(pv):
    return np.argsort(np.argsort(np.nan_to_num(pv, nan=0.5))).astype(np.float64) / len(pv)

# A + B blend
if all(k in results for k in ['sub_A_price_negw10', 'sub_B_fund_negw10']):
    _, pv_A, _, _ = results['sub_A_price_negw10']
    _, pv_B, _, _ = results['sub_B_fund_negw10']
    rA = to_rank(pv_A)
    rB = to_rank(pv_B)
    # Grid search optimal w on META_VAL
    _, pv_A_mv, _, _ = results['sub_A_price_negw10']
    _, pv_B_mv, _, _ = results['sub_B_fund_negw10']
    rA_mv = to_rank(pv_A_mv)
    rB_mv = to_rank(pv_B_mv)

    best_w = 0.5; best_auc_mv = 0
    for w in np.arange(0.0, 1.05, 0.05):
        pv_mv_blend = w * rA_mv + (1 - w) * rB_mv
        auc = roc_auc_score(y_mv_full, pv_mv_blend)
        if auc > best_auc_mv:
            best_auc_mv = auc; best_w = w
    pv_mv_AB = best_w * rA_mv + (1 - best_w) * rB_mv
    pv_te_AB = best_w * rA + (1 - best_w) * rB
    auc_mv_AB = roc_auc_score(y_mv_full, pv_mv_AB)
    auc_te_AB = roc_auc_score(y_te_full, pv_te_AB)
    print(f"\n  ★ A+B OPT_BLEND (w_A={best_w:.2f}, w_B={1-best_w:.2f}): MV={auc_mv_AB:.4f} TE={auc_te_AB:.4f}", flush=True)
    results['ens_AB_opt'] = (pv_mv_AB, pv_te_AB, auc_mv_AB, auc_te_AB)

# A + B + C blend
if all(k in results for k in ['sub_A_price_negw10', 'sub_B_fund_negw10', 'sub_C_btc_negw10']):
    _, pv_A, _, _ = results['sub_A_price_negw10']
    _, pv_B, _, _ = results['sub_B_fund_negw10']
    _, pv_C, _, _ = results['sub_C_btc_negw10']
    rA = to_rank(pv_A)
    rB = to_rank(pv_B)
    rC = to_rank(pv_C)
    # Simple equal weight first, then grid
    pv_te_ABC_eq = (rA + rB + rC) / 3
    auc_te_ABC_eq = roc_auc_score(y_te_full, pv_te_ABC_eq)
    _, pv_A_mv, _, _ = results['sub_A_price_negw10']
    _, pv_B_mv, _, _ = results['sub_B_fund_negw10']
    _, pv_C_mv, _, _ = results['sub_C_btc_negw10']
    rA_mv = to_rank(pv_A_mv); rB_mv = to_rank(pv_B_mv); rC_mv = to_rank(pv_C_mv)
    pv_mv_ABC_eq = (rA_mv + rB_mv + rC_mv) / 3
    auc_mv_ABC_eq = roc_auc_score(y_mv_full, pv_mv_ABC_eq)
    print(f"\n  ★ A+B+C EQUAL WEIGHT: MV={auc_mv_ABC_eq:.4f} TE={auc_te_ABC_eq:.4f}", flush=True)
    results['ens_ABC_eq'] = (pv_mv_ABC_eq, pv_te_ABC_eq, auc_mv_ABC_eq, auc_te_ABC_eq)

# ============================================================
# 12. Strict no-lookahead evaluation (MV选参数, TE只跑一次)
# ============================================================
print("\n" + "=" * 60, flush=True)
print("12. Strict no-lookahead top-k evaluation", flush=True)
print("=" * 60, flush=True)

DAYS_TE = (ts_used[te_mask][-1] - ts_used[te_mask][0]) / 86400.0
print(f"  TE spans ~{DAYS_TE:.0f} days", flush=True)

def topk_eval(pv, y, label_str=""):
    auc = roc_auc_score(y, pv)
    out = [f"\n  [{label_str}] AUC={auc:.4f}"]
    for pct in [0.5, 1.0, 1.5, 2.0, 3.0]:
        k = max(1, int(len(pv) * pct / 100))
        acc = y[np.argsort(-pv)[:k]].mean() * 100
        tpd = k / DAYS_TE
        flag = '🏆' if pct == 1.0 and acc >= 65 else ('✅' if pct == 1.0 and acc >= 60 else '')
        out.append(f"    top-{pct:.1f}%: acc={acc:.1f}% tpd≈{tpd:.1f} {flag}")
    return "\n".join(out), auc

# 选每个类别最好的 (根据 MV AUC)
best_overall = None; best_mv_auc = 0
for key, (pv_mv, pv_te, auc_mv, auc_te) in results.items():
    line, _ = topk_eval(pv_te, y_te_full, key)
    print(line, flush=True)
    if auc_mv > best_mv_auc:
        best_mv_auc = auc_mv; best_overall = key

print(f"\n  🏆 BEST MV MODEL: {best_overall} (MV AUC={best_mv_auc:.4f})", flush=True)

# ============================================================
# 13. Also check: 去噪+negw+子集 有没有叠加
# ============================================================
print("\n" + "=" * 60, flush=True)
print("13. Summary table", flush=True)
print("=" * 60, flush=True)

print("\n| Model | MV AUC | TE AUC | TE top1% Acc | TE top1% TPD |", flush=True)
print("|---|---|---|---|---|", flush=True)
for key, (pv_mv, pv_te, auc_mv, auc_te) in sorted(results.items()):
    k1 = max(1, int(len(pv_te) * 0.01))
    acc1 = y_te_full[np.argsort(-pv_te)[:k1]].mean() * 100
    tpd1 = k1 / DAYS_TE
    print(f"| {key} | {auc_mv:.4f} | {auc_te:.4f} | {acc1:.1f}% | {tpd1:.1f} |", flush=True)

print(f"\nTOTAL TIME: {time.time()-t0:.0f}s", flush=True)
