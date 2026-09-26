"""Real-Time Multi-Timeframe Feature Engine & Model Evaluation

设计原理:
在同一 1min 时间节点 t (例如 18:47:00)，直接基于包含当前时刻在内过去 $N$ 根 1min K 线
实时计算 5m / 15m / 30m / 60m 的实时高周期特征 (包括 Rolling High/Low 突破位置, 实时 CVD 比例,
Stoch 随机指标, EMA 偏离度, 实时 Realized Volatility 等)。

全周期 1min 频次无需等待高周期 K 线闭合，在实盘/测试中每 1 分钟触发即获得最前沿实时微观与宏观形态。
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

def compute_realtime_multitf_features(df_self, df_other=None, higher_tf_minutes=[5, 15, 30, 60]):
    """1min 频次实时多周期特征引擎 (基于最新 1m 滚动窗口零迟滞生成)"""
    p_self = pl.from_pandas(df_self) if isinstance(df_self, pd.DataFrame) else df_self

    exprs = []
    close = pl.col('close')
    high = pl.col('high')
    low = pl.col('low')
    buy = pl.col('buy_vol')
    sell = pl.col('sell_vol')
    d_vol = buy - sell
    tot_vol = buy + sell

    # 1. 基础 1min 收益率与 Log Close
    exprs.append((close.log() - close.log().shift(1)).fill_null(0.0).alias('lr1_1m'))
    exprs.append((d_vol / (tot_vol + 1e-9)).fill_null(0.0).alias('cvd_1m'))

    # 2. 针对 5m, 15m, 30m, 60m 逐级生成 1m 实时滑动特征
    for tf in higher_tf_minutes:
        # (A) 实时高周期最高/最低突破位置
        roll_max = high.rolling_max(window_size=tf)
        roll_min = low.rolling_min(window_size=tf)
        range_tf = roll_max - roll_min + 1e-9
        pos_range = ((close - roll_min) / range_tf).fill_null(0.5).alias(f'pos_range_{tf}m')
        exprs.append(pos_range)

        # (B) 实时高周期突破距离
        breakout_up = ((close - roll_max) / (close + 1e-9)).fill_null(0.0).alias(f'breakout_up_{tf}m')
        breakout_dn = ((roll_min - close) / (close + 1e-9)).fill_null(0.0).alias(f'breakout_dn_{tf}m')
        exprs.append(breakout_up)
        exprs.append(breakout_dn)

        # (C) 实时高周期对数收益率
        lr_tf = (close.log() - close.log().shift(tf)).fill_null(0.0).alias(f'lr_{tf}m')
        exprs.append(lr_tf)

        # (D) 实时高周期 CVD 比例
        cvd_tf = (d_vol.rolling_sum(window_size=tf) / (tot_vol.rolling_sum(window_size=tf) + 1e-9)).fill_null(0.0).alias(f'cvd_{tf}m')
        exprs.append(cvd_tf)

        # (E) 实时高周期 Stoch 随机指标
        stoch_k = ((close - roll_min) / range_tf * 100.0).fill_null(50.0)
        stoch_d = (stoch_k.rolling_mean(window_size=max(2, tf // 5)).fill_null(50.0) / 100.0).alias(f'stoch_{tf}m')
        exprs.append(stoch_d)

        # (F) 实时高周期 Realized Volatility
        lr1 = (close.log() - close.log().shift(1)).fill_null(0.0)
        rvol = (lr1.rolling_std(window_size=tf).fill_null(0.0) * 100.0).alias(f'rvol_{tf}m')
        exprs.append(rvol)

        # (G) 实时高周期 EMA 偏离度
        ema_tf = close.ewm_mean(span=tf, adjust=False)
        bias_tf = ((close - ema_tf) / (close + 1e-9)).fill_null(0.0).alias(f'bias_{tf}m')
        exprs.append(bias_tf)

    # 3. 跨资产 (Cross-Asset) 实时特征对齐
    if df_other is not None:
        p_other = pl.from_pandas(df_other) if isinstance(df_other, pd.DataFrame) else df_other
        ts_s = p_self['ts'].to_numpy()
        ts_o = p_other['ts'].to_numpy()
        close_o = p_other['close'].to_numpy()

        idx_o = np.searchsorted(ts_o, ts_s, side='right') - 1
        idx_o = np.clip(idx_o, 0, len(ts_o) - 1)
        close_o_aligned = close_o[idx_o]

        close_s = p_self['close'].to_numpy()
        ratio = close_s / (close_o_aligned + 1e-12)

        cross_exprs = []
        for tf in higher_tf_minutes:
            r_ret = np.zeros(len(close_s), dtype=np.float32)
            r_ret[tf:] = np.log(np.maximum(ratio[tf:], 1e-12) / np.maximum(ratio[:-tf], 1e-12)).astype(np.float32)
            exprs.append(pl.Series(f'cross_rret_{tf}m', r_ret))

    feat_df = p_self.with_columns(exprs)
    feat_cols = [c for c in feat_df.columns if c not in ['ts', 'open', 'high', 'low', 'close', 'buy_vol', 'sell_vol', 'funding']]
    feats = feat_df[feat_cols].to_numpy().astype(np.float32)
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return feats, feat_cols

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

def train_and_eval_realtime_multitf(symbol='ETH', horizon=30):
    print(f"\n" + "=" * 65, flush=True)
    print(f"  【实时多周期特征 LightGBM 训练与评估】{symbol} HORIZON={horizon}m", flush=True)
    print("=" * 65, flush=True)

    raw_e = pd.read_parquet(os.path.join(config.DS_DIR, "raw_ETH.parquet")).sort_values('ts').reset_index(drop=True)
    raw_b = pd.read_parquet(os.path.join(config.DS_DIR, "raw_BTC.parquet")).sort_values('ts').reset_index(drop=True)

    if symbol == 'ETH':
        feats, feat_names = compute_realtime_multitf_features(raw_e, raw_b, higher_tf_minutes=[5, 15, 30, 60, 120])
        ts = raw_e['ts'].values.astype(np.int64)
        close = raw_e['close'].values
    else:
        feats, feat_names = compute_realtime_multitf_features(raw_b, raw_e, higher_tf_minutes=[5, 15, 30, 60, 120])
        ts = raw_b['ts'].values.astype(np.int64)
        close = raw_b['close'].values

    n = len(ts)
    ret_fut = np.full(n, np.nan, dtype=np.float32)
    ret_fut[:-horizon] = (close[horizon:] / close[:-horizon] - 1.0).astype(np.float32)
    label = (ret_fut > 0).astype(np.int8)

    valid = ~np.isnan(ret_fut) & (np.arange(n) >= 120)

    def ts_mask(ts_arr, s, e):
        a = int(datetime.datetime.strptime(s, '%Y-%m-%d').replace(tzinfo=datetime.timezone.utc).timestamp())
        b = int(datetime.datetime.strptime(e, '%Y-%m-%d').replace(tzinfo=datetime.timezone.utc).timestamp())
        return (ts_arr >= a) & (ts_arr < b)

    tr_m = ts_mask(ts, '2020-01-01', '2024-06-30') & valid
    es_m = ts_mask(ts, '2024-06-30', '2024-09-30') & valid
    mv_m = ts_mask(ts, '2024-09-30', '2025-09-30') & valid
    te_m = ts_mask(ts, '2025-09-30', '2026-08-29') & valid

    Xtr, ytr, retr = feats[tr_m][::3], label[tr_m][::3], ret_fut[tr_m][::3]
    Xes, yes = feats[es_m], label[es_m]
    Xte, yte, ts_te = feats[te_m], label[te_m], ts[te_m]

    sw = np.clip(np.abs(retr) * 50, 0.5, 5.0)

    dtr = lgb.Dataset(Xtr, ytr, weight=sw)
    des = lgb.Dataset(Xes, yes, reference=dtr)

    params = dict(
        objective='binary', metric='auc', learning_rate=0.03, num_leaves=31,
        feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=2,
        min_child_samples=200, verbose=-1, seed=42
    )
    model = lgb.train(params, dtr, num_boost_round=1000, valid_sets=[des],
                      callbacks=[lgb.early_stopping(100, verbose=False)])

    p_te = model.predict(Xte)
    auc = roc_auc_score(yte, p_te)
    acc, min_a, bad_m, tpd, acc_m = eval_r2_daily(p_te, yte, ts_te)

    monthly_mean = np.mean(list(acc_m.values()))

    print(f"【{symbol} 评估结果】")
    print(f"  Test AUC: {auc:.4f}")
    print(f"  Top 1% 总体准确率: {acc*100:.2f}%")
    print(f"  月均准确率: {monthly_mean*100:.2f}%")
    print(f"  最低月准确率: {min_a*100:.2f}%")
    print(f"  坏月 (<55%): {bad_m} 个")
    print(f"  每日交易笔数: {tpd:.1f} 笔/天")
    print(f"\n  逐月明细:\n  {acc_m}\n")

def main():
    print("=================================================================", flush=True)
    print("  全实时多周期 (5m, 15m, 30m, 60m, 120m) 1min Tick 深度训练评估", flush=True)
    print("=================================================================", flush=True)

    train_and_eval_realtime_multitf('ETH', horizon=30)
    train_and_eval_realtime_multitf('BTC', horizon=30)

if __name__ == "__main__":
    main()
