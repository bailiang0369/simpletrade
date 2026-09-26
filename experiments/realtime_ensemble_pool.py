"""Real-Time Multi-Timeframe 3-Model Ensemble System (三大核心模型 LightGBM + XGBoost + CatBoost 聚合)

设计原理:
1. 采用全实时 1min Tick 频次计算多周期特征 (5m, 15m, 30m, 60m, 120m, 240m)，
   任何时刻 t (如 18:47) 零迟滞获取当前未闭合与过去的微观/宏观组合形态。
2. 聚合 SimpleTrade 3 大核心模型族:
   - LightGBM (树模型 GBDT)
   - XGBoost (极端梯度提升树)
   - CatBoost (类别/数值对称树)
3. 概率 Rank Uniformization (秩归一化) + 3 大模型族平均/加权投票。
4. 全量 1min K 线测试段评估 (Top 1% 每日信号筛选, 验证总体准确率与逐月坏月)。
"""

import os, sys, gc, time, datetime, warnings
import numpy as np
import pandas as pd
import polars as pl

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

# ========== 1. 1min Tick 级别全实时多周期特征构建 ==========
def build_realtime_multitf_features(df_self, df_other=None, higher_tf_minutes=[5, 15, 30, 60, 120, 240]):
    p_self = pl.from_pandas(df_self) if isinstance(df_self, pd.DataFrame) else df_self

    exprs = []
    close = pl.col('close')
    high = pl.col('high')
    low = pl.col('low')
    buy = pl.col('buy_vol')
    sell = pl.col('sell_vol')
    d_vol = buy - sell
    tot_vol = buy + sell

    # (1) 1min 微观基础特征
    exprs.append((close.log() - close.log().shift(1)).fill_null(0.0).alias('lr1_1m'))
    exprs.append((d_vol / (tot_vol + 1e-9)).fill_null(0.0).alias('cvd_1m'))

    # (2) 多周期 1min 实时滑动特征
    for tf in higher_tf_minutes:
        # A. 区间相对位置
        roll_max = high.rolling_max(window_size=tf)
        roll_min = low.rolling_min(window_size=tf)
        range_tf = roll_max - roll_min + 1e-9
        exprs.append(((close - roll_min) / range_tf).fill_null(0.5).alias(f'pos_range_{tf}m'))

        # B. 突破幅度
        exprs.append(((close - roll_max) / (close + 1e-9)).fill_null(0.0).alias(f'breakout_up_{tf}m'))
        exprs.append(((roll_min - close) / (close + 1e-9)).fill_null(0.0).alias(f'breakout_dn_{tf}m'))

        # C. 动量对数收益率
        exprs.append((close.log() - close.log().shift(tf)).fill_null(0.0).alias(f'lr_{tf}m'))

        # D. 实时 CVD 比率
        exprs.append((d_vol.rolling_sum(window_size=tf) / (tot_vol.rolling_sum(window_size=tf) + 1e-9)).fill_null(0.0).alias(f'cvd_{tf}m'))

        # E. Stoch 随机指标 (K/D)
        stoch_k = ((close - roll_min) / range_tf * 100.0).fill_null(50.0)
        exprs.append((stoch_k.rolling_mean(window_size=max(2, tf // 5)).fill_null(50.0) / 100.0).alias(f'stoch_{tf}m'))

        # F. 波动率 Realized Volatility
        lr1 = (close.log() - close.log().shift(1)).fill_null(0.0)
        exprs.append((lr1.rolling_std(window_size=tf).fill_null(0.0) * 100.0).alias(f'rvol_{tf}m'))

        # G. EMA 偏离度
        ema_tf = close.ewm_mean(span=tf, adjust=False)
        exprs.append(((close - ema_tf) / (close + 1e-9)).fill_null(0.0).alias(f'bias_{tf}m'))

    # (3) 跨资产对手盘实时对齐
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

        for tf in higher_tf_minutes:
            r_ret = np.zeros(len(close_s), dtype=np.float32)
            r_ret[tf:] = np.log(np.maximum(ratio[tf:], 1e-12) / np.maximum(ratio[:-tf], 1e-12)).astype(np.float32)
            exprs.append(pl.Series(f'cross_rret_{tf}m', r_ret))

    feat_df = p_self.with_columns(exprs)
    feat_cols = [c for c in feat_df.columns if c not in ['ts', 'open', 'high', 'low', 'close', 'buy_vol', 'sell_vol', 'funding']]
    feats = feat_df[feat_cols].to_numpy().astype(np.float32)
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return feats, feat_cols

# ========== 2. 评估函数 ==========
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

def to_rank(p):
    return (np.argsort(np.argsort(p)) / (len(p) - 1)).astype(np.float32)

# ========== 3. 训练与 3 模型聚合同步评估 ==========
def run_realtime_ensemble_pool(symbol='ETH', horizon=30):
    print(f"\n" + "=" * 70, flush=True)
    print(f"  【实时多周期三大核心模型 (LGB + XGB + CAT) 聚合训练】{symbol} H={horizon}m", flush=True)
    print("=" * 70, flush=True)

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
    mv_m = ts_mask(ts, '2024-09-30', '2025-09-30') & valid
    te_m = ts_mask(ts, '2025-09-30', '2026-08-29') & valid

    Xtr, ytr, retr = feats[tr_m][::3], label[tr_m][::3], ret_fut[tr_m][::3]
    Xes, yes = feats[es_m], label[es_m]
    Xte, yte, ts_te = feats[te_m], label[te_m], ts[te_m]

    sw = np.clip(np.abs(retr) * 50, 0.5, 5.0)

    # ---------------- 1. LightGBM ----------------
    print("\n[1/3] 训练 LightGBM 实时多周期模型...", flush=True)
    dtr = lgb.Dataset(Xtr, ytr, weight=sw)
    des = lgb.Dataset(Xes, yes, reference=dtr)
    params_lgb = dict(
        objective='binary', metric='auc', learning_rate=0.03, num_leaves=31,
        feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=2,
        min_child_samples=200, verbose=-1, seed=42
    )
    m_lgb = lgb.train(params_lgb, dtr, num_boost_round=1000, valid_sets=[des],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
    p_lgb = m_lgb.predict(Xte)

    # ---------------- 2. XGBoost ----------------
    print("[2/3] 训练 XGBoost 实时多周期模型...", flush=True)
    dx_tr = xgb.DMatrix(Xtr, label=ytr, weight=sw)
    dx_es = xgb.DMatrix(Xes, label=yes)
    dx_te = xgb.DMatrix(Xte)
    params_xgb = dict(
        objective='binary:logistic', eval_metric='auc', learning_rate=0.03,
        max_depth=6, subsample=0.8, colsample_bytree=0.8, min_child_weight=200,
        tree_method='hist', seed=42
    )
    m_xgb = xgb.train(params_xgb, dx_tr, num_boost_round=1000, evals=[(dx_es, 'val')],
                      early_stopping_rounds=100, verbose_eval=False)
    p_xgb = m_xgb.predict(dx_te)

    # ---------------- 3. CatBoost ----------------
    print("[3/3] 训练 CatBoost 实时多周期模型...", flush=True)
    cb_tr = Pool(Xtr, ytr, weight=sw)
    cb_es = Pool(Xes, yes)
    m_cb = CatBoostClassifier(
        iterations=1000, learning_rate=0.03, depth=6, eval_metric='AUC',
        random_seed=42, verbose=False
    )
    m_cb.fit(cb_tr, eval_set=cb_es, early_stopping_rounds=100)
    p_cat = m_cb.predict_proba(Xte)[:, 1]

    # ---------------- 4. 秩归一化与 3 模型聚合 ----------------
    r_lgb = to_rank(p_lgb)
    r_xgb = to_rank(p_xgb)
    r_cat = to_rank(p_cat)

    p_ens = (r_lgb + r_xgb + r_cat) / 3.0

    # 评估单个模型与 聚合模型
    auc_lgb = roc_auc_score(yte, p_lgb)
    acc_lgb, min_lgb, bad_lgb, _, _ = eval_r2_daily(p_lgb, yte, ts_te)

    auc_xgb = roc_auc_score(yte, p_xgb)
    acc_xgb, min_xgb, bad_xgb, _, _ = eval_r2_daily(p_xgb, yte, ts_te)

    auc_cat = roc_auc_score(yte, p_cat)
    acc_cat, min_cat, bad_cat, _, _ = eval_r2_daily(p_cat, yte, ts_te)

    auc_ens = roc_auc_score(yte, p_ens)
    acc_ens, min_ens, bad_ens, tpd_ens, acc_m_ens = eval_r2_daily(p_ens, yte, ts_te)
    m_mean_ens = np.mean(list(acc_m_ens.values()))

    print("\n" + "=" * 70, flush=True)
    print(f"  【实时多周期三大模型聚合 ({symbol}) 最终结果】", flush=True)
    print("=" * 70, flush=True)
    print(f"1. LightGBM  -> AUC: {auc_lgb:.4f} | 总体Top1%准确率: {acc_lgb*100:.2f}% | 最低月: {min_lgb*100:.1f}% | 坏月: {bad_lgb}个")
    print(f"2. XGBoost   -> AUC: {auc_xgb:.4f} | 总体Top1%准确率: {acc_xgb*100:.2f}% | 最低月: {min_xgb*100:.1f}% | 坏月: {bad_xgb}个")
    print(f"3. CatBoost  -> AUC: {auc_cat:.4f} | 总体Top1%准确率: {acc_cat*100:.2f}% | 最低月: {min_cat*100:.1f}% | 坏月: {bad_cat}个")
    print("-" * 70, flush=True)
    print(f"★ 三大模型聚合 (Rank Ensemble) -> Test AUC: {auc_ens:.4f}")
    print(f"  • 总体 Top 1% 准确率: {acc_ens*100:.2f}%")
    print(f"  • 12 个月月均准确率: {m_mean_ens*100:.2f}%")
    print(f"  • 最低月份准确率: {min_ens*100:.2f}%")
    print(f"  • 坏月 (<55%) 数量: {bad_ens} 个")
    print(f"  • 每日交易频率: {tpd_ens:.1f} 笔/天")
    print(f"\n  {symbol} 三大模型聚合 12 个月逐月准确率明细:\n  {acc_m_ens}\n")

def main():
    print("=================================================================", flush=True)
    print("  SimpleTrade 3 大核心模型族 (LGB+XGB+CAT) 实时多周期聚合评估", flush=True)
    print("=================================================================", flush=True)

    run_realtime_ensemble_pool('ETH', horizon=30)
    run_realtime_ensemble_pool('BTC', horizon=30)

if __name__ == "__main__":
    main()
