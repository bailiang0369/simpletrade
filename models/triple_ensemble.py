"""三模型异构集成 (Triple Heterogeneous Ensemble: LightGBM + CatBoost + XGBoost)

全新无前视三模型融合 + 跨周期多重共振筛选 (Multi-Horizon Consensus) 策略。
"""

import time
import os
import gc
import numpy as np
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool
import xgboost as xgb

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from data_store import AssetContext
from evaluate import evaluate_topk, check_all
from validate_eth_quick import compute_extra_raw, get_X


def train_triple_ensemble(s, h=30):
    t0 = time.time()
    ctx = AssetContext(s, horizon=h)
    extra = compute_extra_raw(ctx)
    trm = ctx.split_rows['train']
    esm = ctx.split_rows['early_stop']
    mvm = ctx.split_rows['meta_val']
    tem = ctx.split_rows['test']

    tr_idx = np.where(trm)[0]
    rng = np.random.default_rng(42)
    tr_idx_sub = rng.choice(tr_idx, size=1_000_000, replace=False)
    train_mask = np.zeros_like(trm, dtype=bool)
    train_mask[tr_idx_sub] = True

    Xtr = get_X(ctx, extra, train_mask)
    ytr = ctx.label[train_mask].astype(np.float64)
    Xes = get_X(ctx, extra, esm)
    yes = ctx.label[esm].astype(np.float64)
    w = np.clip(np.abs(ctx.retf('train')[np.isin(tr_idx, tr_idx_sub)]).astype(np.float64) * 50, 0.5, 5.0)

    # 1. Model 1: LightGBM (3 seeds)
    lgb_preds_te = []
    lgb_preds_mv = []
    for sd in [42, 49, 56]:
        dtr = lgb.Dataset(Xtr, ytr, weight=w)
        des = lgb.Dataset(Xes, yes, reference=dtr)
        m_lgb = lgb.train(
            dict(objective='binary', metric='auc', learning_rate=0.04,
                 num_leaves=63, max_depth=-1, feature_fraction=0.8,
                 bagging_fraction=0.8, bagging_freq=2, min_data_in_leaf=100,
                 verbosity=-1, seed=sd, num_threads=3),
            dtr, num_boost_round=800, valid_sets=[des],
            callbacks=[lgb.early_stopping(80, verbose=False)]
        )
        lgb_preds_mv.append(m_lgb.predict(get_X(ctx, extra, mvm)))
        lgb_preds_te.append(m_lgb.predict(get_X(ctx, extra, tem)))

    p_lgb_mv = np.mean(lgb_preds_mv, axis=0)
    p_lgb_te = np.mean(lgb_preds_te, axis=0)

    # 2. Model 2: CatBoost
    cb = CatBoostClassifier(iterations=600, learning_rate=0.03, depth=6, random_seed=42, thread_count=3, verbose=False)
    cb.fit(Pool(Xtr, ytr, weight=w), eval_set=Pool(Xes, yes), early_stopping_rounds=80)
    p_cb_mv = cb.predict_proba(get_X(ctx, extra, mvm))[:, 1]
    p_cb_te = cb.predict_proba(get_X(ctx, extra, tem))[:, 1]

    # 3. Model 3: XGBoost
    dtrain_xgb = xgb.DMatrix(Xtr, label=ytr, weight=w)
    deval_xgb = xgb.DMatrix(Xes, label=yes)
    dmeta_xgb = xgb.DMatrix(get_X(ctx, extra, mvm))
    dtest_xgb = xgb.DMatrix(get_X(ctx, extra, tem))

    params_xgb = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'learning_rate': 0.04,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'seed': 42,
        'nthread': 3
    }
    m_xgb = xgb.train(params_xgb, dtrain_xgb, num_boost_round=800, evals=[(deval_xgb, 'es')], early_stopping_rounds=80, verbose_eval=False)
    p_xgb_mv = m_xgb.predict(dmeta_xgb)
    p_xgb_te = m_xgb.predict(dtest_xgb)

    # Rank Normalization
    def rank_norm(p):
        return np.argsort(np.argsort(p)).astype(float) / (len(p) - 1)

    p_triple_mv = rank_norm(p_lgb_mv) * 0.4 + rank_norm(p_cb_mv) * 0.3 + rank_norm(p_xgb_mv) * 0.3
    p_triple_te = rank_norm(p_lgb_te) * 0.4 + rank_norm(p_cb_te) * 0.3 + rank_norm(p_xgb_te) * 0.3

    print(f"[Triple Ensemble] {s} H={h} 三模型 (LGB+CB+XGB) 训练完成 (耗时 {time.time()-t0:.0f}s)", flush=True)
    return ctx, p_triple_mv, p_triple_te


if __name__ == "__main__":
    ctx_b, p_b_mv, p_b_te = train_triple_ensemble("BTC", h=30)
    ctx_e, p_e_mv, p_e_te = train_triple_ensemble("ETH", h=30)

    # Evaluate 1% Coverage
    r_b = evaluate_topk(p_b_te, ctx_b.y("test"), ctx_b.retf("test"), ctx_b.times("test"), coverage=0.01)
    r_e = evaluate_topk(p_e_te, ctx_e.y("test"), ctx_e.retf("test"), ctx_e.times("test"), coverage=0.01)

    print("=== 三模型异构集成 (LGB+CB+XGB) 全 Test 集 1% 覆盖率原始胜率 ===")
    print(f"BTC H=30: Acc = {r_b['accuracy']*100:.2f}% ({r_b['trades_per_day']:.1f}单/天)")
    print(f"ETH H=30: Acc = {r_e['accuracy']*100:.2f}% ({r_e['trades_per_day']:.1f}单/天)")
