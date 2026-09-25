"""Real-Time Multi-Timeframe Model Horizon Comparison (H=15m vs H=30m vs H=60m)

测试与评估说明:
保持相同的全实时 1min Tick 多周期特征引擎 (5m, 15m, 30m, 60m, 120m, 240m 滑动窗口)，
对比模型在不同未来预测目标下的准确率:
  - H = 15m (预测未来 15 分钟涨跌)
  - H = 30m (预测未来 30 分钟涨跌)
  - H = 60m (预测未来 60 分钟涨跌)
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

from realtime_three_families_ensemble import build_realtime_multitf_features, eval_r2_daily, to_rank

def eval_horizon_realtime(symbol='ETH', horizon=15):
    raw_e = pd.read_parquet(os.path.join(config.DS_DIR, "raw_ETH.parquet")).sort_values('ts').reset_index(drop=True)
    raw_b = pd.read_parquet(os.path.join(config.DS_DIR, "raw_BTC.parquet")).sort_values('ts').reset_index(drop=True)

    if symbol == 'ETH':
        feats, feat_names = build_realtime_multitf_features(raw_e, raw_b)
        ts = raw_e['ts'].values.astype(np.int64)
        close = raw_e['close'].values
    else:
        feats, feat_names = build_realtime_multitf_features(raw_b, raw_e)
        ts = raw_b['ts'].values.astype(np.int64)
        close = raw_b['close'].values

    n = len(ts)
    ret_fut = np.full(n, np.nan, dtype=np.float32)
    ret_fut[:-horizon] = (close[horizon:] / close[:-horizon] - 1.0).astype(np.float32)
    label = (ret_fut > 0).astype(np.int8)

    valid = ~np.isnan(ret_fut) & (np.arange(n) >= 240)

    def ts_mask(ts_arr, s, e):
        a = int(datetime.datetime.strptime(s, '%Y-%m-%d').replace(tzinfo=datetime.timezone.utc).timestamp())
        b = int(datetime.datetime.strptime(e, '%Y-%m-%d').replace(tzinfo=datetime.timezone.utc).timestamp())
        return (ts_arr >= a) & (ts_arr < b)

    tr_m = ts_mask(ts, '2020-01-01', '2024-06-30') & valid
    es_m = ts_mask(ts, '2024-06-30', '2024-09-30') & valid
    te_m = ts_mask(ts, '2025-09-30', '2026-08-29') & valid

    Xtr, ytr, retr = feats[tr_m][::3], label[tr_m][::3], ret_fut[tr_m][::3]
    Xes, yes = feats[es_m], label[es_m]
    Xte, yte, ts_te = feats[te_m], label[te_m], ts[te_m]

    sw = np.clip(np.abs(retr) * 50, 0.5, 5.0)

    dtr = lgb.Dataset(Xtr, ytr, weight=sw)
    des = lgb.Dataset(Xes, yes, reference=dtr)

    params_lgb = dict(
        objective='binary', metric='auc', learning_rate=0.03, num_leaves=31,
        feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=2,
        min_child_samples=200, verbose=-1, seed=42
    )
    m_lgb = lgb.train(params_lgb, dtr, num_boost_round=1000, valid_sets=[des],
                      callbacks=[lgb.early_stopping(100, verbose=False)])

    p_te = m_lgb.predict(Xte)
    auc = roc_auc_score(yte, p_te)
    acc, min_a, bad_m, tpd, acc_m = eval_r2_daily(p_te, yte, ts_te)

    m_mean = np.mean(list(acc_m.values()))
    return {
        'symbol': symbol,
        'horizon': horizon,
        'auc': auc,
        'acc': acc,
        'm_mean': m_mean,
        'min_a': min_a,
        'bad_m': bad_m,
        'tpd': tpd,
        'acc_m': acc_m
    }

def main():
    print("=================================================================", flush=True)
    print("  全实时多周期模型预测目标对比: H=15m vs H=30m vs H=60m", flush=True)
    print("=================================================================\n", flush=True)

    results = []
    for sym in ['ETH', 'BTC']:
        for h in [15, 30, 60]:
            print(f"正在评估 {sym} 未来 H={h} 分钟涨跌预测...", flush=True)
            res = eval_horizon_realtime(sym, h)
            results.append(res)
            print(f"  --> {sym} H={h}m | AUC: {res['auc']:.4f} | Top1%准确率: {res['acc']*100:.2f}% | 月均: {res['m_mean']*100:.2f}% | 最低月: {res['min_a']*100:.1f}% | 坏月: {res['bad_m']}个", flush=True)

    print("\n" + "=" * 70, flush=True)
    print("  【不同预测 Horizon (15m vs 30m vs 60m) 汇总对比表】", flush=True)
    print("=" * 70, flush=True)
    for r in results:
        print(f"[{r['symbol']} 未来{r['horizon']}m涨跌] -> Test AUC: {r['auc']:.4f} | Top1%准确率: {r['acc']*100:.2f}% | 月均: {r['m_mean']*100:.2f}% | 最低月: {r['min_a']*100:.1f}% | 坏月: {r['bad_m']}个")

if __name__ == "__main__":
    main()
