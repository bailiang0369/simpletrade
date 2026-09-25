"""K线重采样实验 (2分钟 / 3分钟基础 K 线预测未来 30 分钟涨跌):
  - 实验 1: 1min 基础 K 线 -> 预测未来 30 根 K 线 (H=30min) [基线对照]
  - 实验 2: 2min 基础 K 线 -> 预测未来 15 根 K 线 (H=15 2min bars = 30min)
  - 实验 3: 3min 基础 K 线 -> 预测未来 10 根 K 线 (H=10 3min bars = 30min)

验证目的: 观察将基础 K 线聚合降噪后，对未来 30 分钟涨跌的预测准确率与坏月稳定性的影响。
"""
import os, sys, gc, time, datetime, warnings
import numpy as np
import pandas as pd
import polars as pl
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

def resample_ohlcv(raw_df, period_min=2):
    """将 1min K 线表重采样为 2min / 3min 基础 K 线"""
    t0 = time.time()
    p_df = pl.from_pandas(raw_df)
    p_df = p_df.with_columns(pl.from_epoch(pl.col('ts'), time_unit='s').alias('dt'))

    resampled = (
        p_df.group_by_dynamic('dt', every=f'{period_min}m')
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
    print(f"  [Resample {period_min}min K] 原始 {len(raw_df):,} 行 -> 重采样后 {resampled.height:,} 行 ({time.time()-t0:.2f}s)", flush=True)
    return resampled.to_pandas()

def build_resampled_features(df_target, df_other):
    """在重采样后的 2min / 3min K 线上构建技术特征"""
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
    for w in [5, 10, 20]:
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

def run_resample_experiment(period_min, bars_ahead, symbol='ETH'):
    print(f"\n" + "=" * 65, flush=True)
    print(f"  【实验】{symbol} 使用 {period_min}min 基础 K 线 -> 预测未来 {bars_ahead} 根 K 线 ({period_min*bars_ahead}分钟)", flush=True)
    print("=" * 65, flush=True)

    raw_e = pd.read_parquet(os.path.join(config.DS_DIR, "raw_ETH.parquet")).sort_values('ts').reset_index(drop=True)
    raw_b = pd.read_parquet(os.path.join(config.DS_DIR, "raw_BTC.parquet")).sort_values('ts').reset_index(drop=True)

    if period_min > 1:
        res_e = resample_ohlcv(raw_e, period_min=period_min)
        res_b = resample_ohlcv(raw_b, period_min=period_min)
    else:
        res_e = raw_e
        res_b = raw_b

    # 提取特征
    feats_e = build_resampled_features(res_e, res_b) if symbol == 'ETH' else build_resampled_features(res_b, res_e)
    close = res_e['close'].values if symbol == 'ETH' else res_b['close'].values
    ts = res_e['ts'].values.astype(np.int64) if symbol == 'ETH' else res_b['ts'].values.astype(np.int64)

    n = len(ts)
    ret_fut = np.full(n, np.nan, dtype=np.float32)
    ret_fut[:-bars_ahead] = (close[bars_ahead:] / close[:-bars_ahead] - 1.0).astype(np.float32)
    label = (ret_fut > 0).astype(np.int8)

    valid = ~np.isnan(ret_fut) & (np.arange(n) >= 40)

    def ts_mask(s, e):
        a = int(datetime.datetime.strptime(s, '%Y-%m-%d').replace(tzinfo=datetime.timezone.utc).timestamp())
        b = int(datetime.datetime.strptime(e, '%Y-%m-%d').replace(tzinfo=datetime.timezone.utc).timestamp())
        return (ts >= a) & (ts < b)

    tr_m = ts_mask('2020-01-01', '2024-06-30') & valid
    es_m = ts_mask('2024-06-30', '2024-09-30') & valid
    te_m = ts_mask('2025-09-30', '2026-08-29') & valid

    Xtr, ytr, retr = feats_e[tr_m], label[tr_m], ret_fut[tr_m]
    Xes, yes = feats_e[es_m], label[es_m]
    Xte, yte, ts_te = feats_e[te_m], label[te_m], ts[te_m]

    # 加权样本收益率
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

    p_te = model.predict(Xte)
    auc = roc_auc_score(yte, p_te)
    overall_acc, min_a, bad_m, tpd, monthly = eval_r2_daily(p_te, yte, ts_te)

    print(f"  Test AUC: {auc:.4f}")
    print(f"  总体 Top 1% 信号预测准确率: {overall_acc*100:.2f}%")
    print(f"  日均交易次数 (TPD): {tpd:.1f} 笔/天")
    print(f"  坏月份数 (<55%): {bad_m} 个 (最低单月准确率: {min_a*100:.2f}%)")
    print(f"  逐月明细:\n  {monthly}\n")
    return overall_acc, min_a, bad_m, tpd

def main():
    print("=================================================================", flush=True)
    print("  K 线重采样对预测未来 30 分钟涨跌影响的对比实验", flush=True)
    print("=================================================================", flush=True)

    # 预测标的: ETH (以太坊)
    run_resample_experiment(period_min=1, bars_ahead=30, symbol='ETH')  # 1min 基础 K -> 预测 30 根
    run_resample_experiment(period_min=2, bars_ahead=15, symbol='ETH')  # 2min 基础 K -> 预测 15 根
    run_resample_experiment(period_min=3, bars_ahead=10, symbol='ETH')  # 3min 基础 K -> 预测 10 根

    # 预测标的: BTC (比特币)
    run_resample_experiment(period_min=1, bars_ahead=30, symbol='BTC')  # 1min 基础 K -> 预测 30 根
    run_resample_experiment(period_min=2, bars_ahead=15, symbol='BTC')  # 2min 基础 K -> 预测 15 根
    run_resample_experiment(period_min=3, bars_ahead=10, symbol='BTC')  # 3min 基础 K -> 预测 10 根

if __name__ == "__main__":
    main()
