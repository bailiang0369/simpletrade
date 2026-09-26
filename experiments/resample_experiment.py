"""多分辨率 K 线融合集成实验 (Multi-Resolution Time-Frame Stacking & Voting):

验证思路:
  在同一时刻 t，同时使用:
    - 1min 基础 K 线预测未来 30 分钟涨跌 (H=30 根 1m K)
    - 2min 基础 K 线预测未来 30 分钟涨跌 (H=15 根 2m K)
    - 3min 基础 K 线预测未来 30 分钟涨跌 (H=10 根 3m K)
  由于三者预测的是同一个未来 30 分钟的目标标签 P(C_{t+30m} > C_t)，
  通过概率秩归一化 (Rank Uniformization) 与元学习器 (Logistic Stacking) 进行融合，
  观察是否能进一步平滑噪声、提升总体准确率与月度稳定性！
"""
import os, sys, gc, time, datetime, warnings
import numpy as np
import pandas as pd
import polars as pl
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

def resample_ohlcv_phase(raw_df, period_min=2, offset_min=0):
    p_df = pl.from_pandas(raw_df)
    p_df = p_df.with_columns(pl.from_epoch(pl.col('ts'), time_unit='s').alias('dt'))

    resampled = (
        p_df.group_by_dynamic('dt', every=f'{period_min}m', offset=f'{offset_min}m')
        .agg([
            pl.col('ts').first().alias('ts'),
            pl.col('open').first().alias('open'),
            pl.col('high').max().alias('high'),
            pl.col('low').min().alias('low'),
            pl.col('close').last().alias('close'),
            pl.col('buy_vol').sum().alias('buy_vol'),
            pl.col('sell_vol').sum().alias('sell_vol'),
            pl.col('funding').last().alias('funding'),
        ])
        .sort('ts')
    )
    return resampled.to_pandas()

def build_resampled_features(df_target, df_other):
    p_target = pl.from_pandas(df_target[['ts', 'close', 'buy_vol', 'sell_vol']])
    p_other = pl.from_pandas(df_other[['ts', 'close']])

    ts_t = p_target['ts'].to_numpy()
    ts_o = p_other['ts'].to_numpy()
    close_o = p_other['close'].to_numpy()

    idx_o = np.searchsorted(ts_o, ts_t, side='right') - 1
    idx_o = np.clip(idx_o, 0, len(ts_o) - 1)
    close_o_aligned = close_o[idx_o]

    exprs = []
    # 1. 动量对数收益率 (多尺度)
    for w in [1, 2, 5, 10, 20, 40, 80, 160]:
        exprs.append((pl.col('close').log() - pl.col('close').log().shift(w)).fill_null(0.0).alias(f'lr_{w}'))

    # 2. 随机指标 Stoch
    for w in [10, 20, 40]:
        lw = pl.col('close').rolling_min(window_size=w)
        hw = pl.col('close').rolling_max(window_size=w)
        fast_k = (pl.col('close') - lw) / (hw - lw + 1e-9) * 100
        slow_k = (fast_k.rolling_mean(window_size=2).fill_null(50.0) / 100.0).alias(f'stoch_{w}')
        exprs.append(slow_k)

    # 3. CVD 比率
    d = pl.col('buy_vol') - pl.col('sell_vol')
    tot = pl.col('buy_vol') + pl.col('sell_vol')
    for w in [15, 30, 60]:
        cvd_w = (d.rolling_sum(window_size=w) / (tot.rolling_sum(window_size=w) + 1e-9)).fill_null(0.0).alias(f'cvd_{w}')
        exprs.append(cvd_w)

    # 4. MACD Hist
    for fast, slow, sig in [(6, 13, 5), (12, 26, 9)]:
        e_fast = pl.col('close').ewm_mean(span=fast, adjust=False)
        e_slow = pl.col('close').ewm_mean(span=slow, adjust=False)
        macd = e_fast - e_slow
        signal = macd.ewm_mean(span=sig, adjust=False)
        hist = (macd - signal).fill_null(0.0).alias(f'macd_{fast}_{slow}')
        exprs.append(hist)

    feat_df = p_target.with_columns(exprs)
    feat_cols = [c for c in feat_df.columns if c not in ['ts', 'close', 'buy_vol', 'sell_vol']]
    feats = feat_df[feat_cols].to_numpy().astype(np.float32)

    # 5. 时段编码与相对强度
    hour = (ts_t % 86400) // 3600
    h_sin = np.sin(hour * 2 * np.pi / 24).astype(np.float32)[:, None]
    h_cos = np.cos(hour * 2 * np.pi / 24).astype(np.float32)[:, None]

    close_t = p_target['close'].to_numpy()
    ratio = close_t / (close_o_aligned + 1e-12)
    r_cols = []
    for w in [5, 20, 80]:
        r_ret = np.zeros(len(close_t), dtype=np.float32)
        r_ret[w:] = np.log(np.maximum(ratio[w:], 1e-12) / np.maximum(ratio[:-w], 1e-12)).astype(np.float32)
        r_cols.append(r_ret)
    r_mat = np.stack(r_cols, axis=1)

    all_feats = np.hstack([feats, h_sin, h_cos, r_mat]).astype(np.float32)
    all_feats = np.nan_to_num(all_feats, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return all_feats

def eval_r2_daily(p, y, ts_arr):
    n = len(p)
    pred = (p >= 0.5).astype(np.int8)
    conf = np.maximum(p, 1 - p)
    sec_arr = ts_arr.astype(np.int64)
    day = sec_arr // 86400
    days = np.unique(day)
    sm = np.zeros(n, bool)

    for d in days:
        md = day == d
        kd = max(1, int(np.ceil(int(md.sum()) * 0.01)))
        sub = np.where(md)[0]
        sm[sub[np.argsort(-conf[sub])[:kd]]] = True

    sel = np.where(sm)[0]
    mts = sec_arr[sel].astype("datetime64[s]").astype("datetime64[M]")
    uniq = np.unique(mts)
    acc_m = {str(u)[:7]: float((pred[sel] == y[sel])[mts == u].mean()) for u in uniq if (mts == u).sum() >= 5}
    min_a = min(acc_m.values()) if len(acc_m) > 0 else 0.0
    bad_m = sum(1 for a in acc_m.values() if a < 0.55)
    overall_acc = float((pred[sel] == y[sel]).mean()) if len(sel) > 0 else 0.0
    tpd = float(sel.size) / len(days)
    return overall_acc, min_a, bad_m, tpd, acc_m

def train_single_resampled_model(period_min, bars_ahead, symbol='ETH'):
    raw_e = pd.read_parquet(os.path.join(config.DS_DIR, "raw_ETH.parquet")).sort_values('ts').reset_index(drop=True)
    raw_b = pd.read_parquet(os.path.join(config.DS_DIR, "raw_BTC.parquet")).sort_values('ts').reset_index(drop=True)

    res_e = resample_ohlcv_phase(raw_e, period_min=period_min, offset_min=0) if period_min > 1 else raw_e
    res_b = resample_ohlcv_phase(raw_b, period_min=period_min, offset_min=0) if period_min > 1 else raw_b

    feats = build_resampled_features(res_e, res_b) if symbol == 'ETH' else build_resampled_features(res_b, res_e)
    close = res_e['close'].values if symbol == 'ETH' else res_b['close'].values
    ts = res_e['ts'].values.astype(np.int64) if symbol == 'ETH' else res_b['ts'].values.astype(np.int64)

    n = len(ts)
    ret_fut = np.full(n, np.nan, dtype=np.float32)
    ret_fut[:-bars_ahead] = (close[bars_ahead:] / close[:-bars_ahead] - 1.0).astype(np.float32)
    label = (ret_fut > 0).astype(np.int8)

    valid = ~np.isnan(ret_fut) & (np.arange(n) >= 40)

    def ts_mask(ts_arr, s, e):
        a = int(datetime.datetime.strptime(s, '%Y-%m-%d').replace(tzinfo=datetime.timezone.utc).timestamp())
        b = int(datetime.datetime.strptime(e, '%Y-%m-%d').replace(tzinfo=datetime.timezone.utc).timestamp())
        return (ts_arr >= a) & (ts_arr < b)

    tr_m = ts_mask(ts, '2020-01-01', '2024-06-30') & valid
    es_m = ts_mask(ts, '2024-06-30', '2024-09-30') & valid
    mv_m = ts_mask(ts, '2024-09-30', '2025-09-30') & valid
    te_m = ts_mask(ts, '2025-09-30', '2026-08-29') & valid

    Xtr, ytr, retr = feats[tr_m], label[tr_m], ret_fut[tr_m]
    Xes, yes = feats[es_m], label[es_m]
    Xmv = feats[mv_m]
    Xte, yte, ts_te = feats[te_m], label[te_m], ts[te_m]

    sw = np.clip(np.abs(retr) * 50, 0.5, 5.0)

    dtr = lgb.Dataset(Xtr, ytr, weight=sw)
    des = lgb.Dataset(Xes, yes, reference=dtr)

    params = dict(
        objective='binary', metric='auc', learning_rate=0.02, num_leaves=31,
        feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=2,
        min_child_samples=200, verbose=-1, seed=42
    )
    model = lgb.train(params, dtr, num_boost_round=1000, valid_sets=[des],
                      callbacks=[lgb.early_stopping(100, verbose=False)])

    p_mv = model.predict(Xmv)
    p_te = model.predict(Xte)
    ts_mv = ts[mv_m]
    return p_mv, p_te, ts_mv, ts_te, label[mv_m], yte

def run_multi_resolution_stacking(symbol='ETH'):
    print(f"\n" + "=" * 65, flush=True)
    print(f"  【多分辨率 K 线融合】{symbol} 训练 1m(H=30) + 2m(H=15) + 3m(H=10) 预测未来 30min", flush=True)
    print("=" * 65, flush=True)

    # 训练 3 个不同分辨率的模型
    p1_mv, p1_te, ts1_mv, ts1_te, y1_mv, y1_te = train_single_resampled_model(1, 30, symbol)
    p2_mv, p2_te, ts2_mv, ts2_te, _, _ = train_single_resampled_model(2, 15, symbol)
    p3_mv, p3_te, ts3_mv, ts3_te, _, _ = train_single_resampled_model(3, 10, symbol)

    # 依时间戳对齐 2m 和 3m 预测到 1m 序列
    idx2_mv = np.clip(np.searchsorted(ts2_mv, ts1_mv), 0, len(ts2_mv) - 1)
    idx2_te = np.clip(np.searchsorted(ts2_te, ts1_te), 0, len(ts2_te) - 1)

    idx3_mv = np.clip(np.searchsorted(ts3_mv, ts1_mv), 0, len(ts3_mv) - 1)
    idx3_te = np.clip(np.searchsorted(ts3_te, ts1_te), 0, len(ts3_te) - 1)

    p2_mv_a, p2_te_a = p2_mv[idx2_mv], p2_te[idx2_te]
    p3_mv_a, p3_te_a = p3_mv[idx3_mv], p3_te[idx3_te]

    # Rank 归一化
    def to_rank(p): return (np.argsort(np.argsort(p)) / (len(p) - 1)).astype(np.float32)

    r1_mv, r1_te = to_rank(p1_mv), to_rank(p1_te)
    r2_mv, r2_te = to_rank(p2_mv_a), to_rank(p2_te_a)
    r3_mv, r3_te = to_rank(p3_mv_a), to_rank(p3_te_a)

    # 1. 独立单模型评估
    a1, min1, b1, tpd1, _ = eval_r2_daily(r1_te, y1_te, ts1_te)
    a2, min2, b2, tpd2, _ = eval_r2_daily(r2_te, y1_te, ts1_te)
    a3, min3, b3, tpd3, _ = eval_r2_daily(r3_te, y1_te, ts1_te)

    print(f"  单独 1min(H=30)  -> 总体准确率: {a1*100:.2f}% | 最低月: {min1*100:.1f}% | 坏月: {b1}个")
    print(f"  单独 2min(H=15)  -> 总体准确率: {a2*100:.2f}% | 最低月: {min2*100:.1f}% | 坏月: {b2}个")
    print(f"  单独 3min(H=10)  -> 总体准确率: {a3*100:.2f}% | 最低月: {min3*100:.1f}% | 坏月: {b3}个")

    # 2. 多分辨率 Rank Voting (1m:50%, 2m:30%, 3m:20%)
    r_fused_voting = r1_te * 0.5 + r2_te * 0.3 + r3_te * 0.2
    auc_voting = roc_auc_score(y1_te, r_fused_voting)
    acc_voting, min_voting, bad_voting, tpd_voting, monthly_voting = eval_r2_daily(r_fused_voting, y1_te, ts1_te)

    # 3. Logistic Stacking 元学习器
    X_meta_train = np.column_stack([r1_mv, r2_mv, r3_mv])
    X_meta_test = np.column_stack([r1_te, r2_te, r3_te])

    meta_model = LogisticRegression(C=0.1, max_iter=500, random_state=42)
    meta_model.fit(X_meta_train, y1_mv)

    p_stacking = meta_model.predict_proba(X_meta_test)[:, 1]
    auc_stacking = roc_auc_score(y1_te, p_stacking)
    acc_stacking, min_stacking, bad_stacking, tpd_stacking, monthly_stacking = eval_r2_daily(p_stacking, y1_te, ts1_te)

    print(f"\n【多分辨率融合结果】")
    print(f"1. 多分辨率秩投票 (Rank Voting) -> Test AUC: {auc_voting:.4f} | 总体Top1%准确率: {acc_voting*100:.2f}% | 最低月: {min_voting*100:.1f}% | 坏月: {bad_voting}个 | TPD: {tpd_voting:.1f}笔/天")
    print(f"2. Stacking 元学习器           -> Test AUC: {auc_stacking:.4f} | 总体Top1%准确率: {acc_stacking*100:.2f}% | 最低月: {min_stacking*100:.1f}% | 坏月: {bad_stacking}个 | TPD: {tpd_stacking:.1f}笔/天")
    print(f"元学习器解出权重: 1min={meta_model.coef_[0][0]:.3f}, 2min={meta_model.coef_[0][1]:.3f}, 3min={meta_model.coef_[0][2]:.3f}")
    print(f"\n{symbol} 多分辨率 Stacking 逐月明细:\n{monthly_stacking}\n")

def main():
    print("=================================================================", flush=True)
    print("  多分辨率 K 线 (1min + 2min + 3min) 预测未来 30min 融合集成", flush=True)
    print("=================================================================", flush=True)

    run_multi_resolution_stacking('ETH')
    run_multi_resolution_stacking('BTC')

if __name__ == "__main__":
    main()
