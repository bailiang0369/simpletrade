"""
mh_union.py — Multi-Horizon 全局模型 Union

策略: 对 ETH 同时训练多个 horizon 的 LGB 模型 (H=3,5,15,30,60),
每个 horizon 独立预测后按各自 threshold 选 top K% 信号, 做并集。

目标: 在不增加数据源的情况下, 用已有 1min OHLCV + funding + BTC cross-asset,
通过多 horizon 覆盖不同周期市场信号, 提高 tpd 覆盖度 (acc 提升有限)。

验证重点:
  1. 每个 horizon 的 top1%/top5%/top10% acc & ret
  2. 多 horizon union 后的 tpd & acc
  3. 是否能接近 tpd=15 + acc=65% (之前单 horizon 达不到)
"""
import os, sys, time, gc, datetime, json, pickle
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from pathlib import Path

sys.path.insert(0, '/workspace')
import config

OUT_DIR = Path('/workspace/mh_experiments')
OUT_DIR.mkdir(exist_ok=True)

# ──────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────
def ts_mask(ts, start, end):
    a = int(datetime.datetime(*start, tzinfo=datetime.timezone.utc).timestamp())
    b = int(datetime.datetime(*end, tzinfo=datetime.timezone.utc).timestamp())
    return (ts >= a) & (ts < b)

def topk_stats(preds, y_true, ret_true, k_list=[0.01, 0.02, 0.05, 0.10]):
    out = {}
    n = len(preds)
    for k in k_list:
        idx = np.argsort(preds)[-max(int(n * k), 1):]
        acc = (y_true[idx] == 1).mean()
        ret_bps = ret_true[idx].mean() * 10000
        out[k] = {'acc': float(acc), 'ret_bps': float(ret_bps), 'n': int(len(idx))}
    return out

def train_lgb(Xtr, ytr, Xes, yes, seed=42, nthreads=3, leaves=63, lr=0.02):
    p = {
        'num_leaves': leaves, 'learning_rate': lr, 'min_data_in_leaf': 200,
        'feature_fraction': 1.0, 'bagging_fraction': 1.0, 'bagging_freq': 1,
        'lambda_l2': 1.0, 'verbose': -1, 'num_threads': nthreads,
        'objective': 'binary', 'metric': 'auc', 'seed': seed,
    }
    dtr = lgb.Dataset(Xtr, label=ytr)
    des = lgb.Dataset(Xes, label=yes, reference=dtr)
    return lgb.train(p, dtr, num_boost_round=1500, valid_sets=[des],
                     callbacks=[lgb.early_stopping(150, verbose=False)])

def subsample(X, y, n, seed=42):
    if len(X) <= n: return X, y
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(X), n, replace=False)
    return X[idx], y[idx]

# ──────────────────────────────────────────────
# 数据集构建 (统一在 1min 粒度)
# ──────────────────────────────────────────────
def build_dataset_all_horizons(horizons):
    """一次加载 raw, 计算所有 horizon 的 label + 共享特征, 返回 dict"""
    t0 = time.time()
    raw_eth = pd.read_parquet('data/datasets/raw_ETH.parquet')
    raw_eth['lr'] = raw_eth['close'].pct_change().astype(np.float32)
    raw_btc = pd.read_parquet('data/datasets/raw_BTC.parquet')
    raw_btc['lr'] = raw_btc['close'].pct_change().astype(np.float32)
    N = len(raw_eth)
    ts = raw_eth['ts'].values
    
    print(f"[load] raw_eth={N:,}, raw_btc={len(raw_btc):,} ({time.time()-t0:.1f}s)", flush=True)
    
    # 统一构建 DataFrame 避免 off-by-one
    feats = pd.DataFrame(index=raw_eth.index)
    # 价格收益窗口
    for w in [5, 15, 30, 60, 120, 240, 480, 720]:
        feats[f'lr_{w}'] = raw_eth['close'].pct_change(w).astype(np.float32)
        if w >= 30:
            lr_s = feats[f'lr_{w}']
            feats[f'z_{w}'] = ((lr_s - lr_s.rolling(w).mean()) / (lr_s.rolling(w).std() + 1e-9)).astype(np.float32)
    feats['rvol_60'] = raw_eth['lr'].rolling(60).std().astype(np.float32)
    feats['rvol_240'] = raw_eth['lr'].rolling(240).std().astype(np.float32)
    # CVD / 订单流代理
    feats['cvd_60'] = (raw_eth['buy_vol'] - raw_eth['sell_vol']).rolling(60).sum().astype(np.float32)
    feats['cvd_240'] = (raw_eth['buy_vol'] - raw_eth['sell_vol']).rolling(240).sum().astype(np.float32)
    feats['tb_act_60'] = raw_eth['buy_vol'].rolling(60).sum().astype(np.float32)
    # Funding 特征 (我们有!)
    feats['funding_now'] = raw_eth['funding'].astype(np.float32)
    feats['funding_z_60'] = ((raw_eth['funding'] - raw_eth['funding'].rolling(60).mean()) / (raw_eth['funding'].rolling(60).std() + 1e-12)).astype(np.float32)
    feats['funding_z_240'] = ((raw_eth['funding'] - raw_eth['funding'].rolling(240).mean()) / (raw_eth['funding'].rolling(240).std() + 1e-12)).astype(np.float32)
    feats['funding_delta_60'] = (raw_eth['funding'] - raw_eth['funding'].shift(60)).astype(np.float32)
    # BTC cross-asset
    for w in [5, 15, 30, 60, 120, 240, 480]:
        feats[f'BTC_lr_{w}'] = raw_btc['close'].pct_change(w).astype(np.float32)
    feats['BTC_rvol_60'] = raw_btc['lr'].rolling(60).std().astype(np.float32)
    feats['BTC_funding_z_60'] = ((raw_btc['funding'] - raw_btc['funding'].rolling(60).mean()) / (raw_btc['funding'].rolling(60).std() + 1e-12)).astype(np.float32)
    # 时间
    dt_idx = pd.to_datetime(raw_eth['ts'], unit='s', utc=True)
    feats['hour_sin'] = np.sin(2 * np.pi * dt_idx.dt.hour.values / 24).astype(np.float32)
    feats['hour_cos'] = np.cos(2 * np.pi * dt_idx.dt.hour.values / 24).astype(np.float32)
    feats['dow_sin'] = np.sin(2 * np.pi * dt_idx.dt.dayofweek.values / 7).astype(np.float32)
    feats['dow_cos'] = np.cos(2 * np.pi * dt_idx.dt.dayofweek.values / 7).astype(np.float32)
    
    print(f"[feats] {feats.shape} ({time.time()-t0:.1f}s)", flush=True)
    
    feat_cols = list(feats.columns)
    
    # 为每个 horizon 生成 label
    datasets = {}
    for h in horizons:
        ret_future = (raw_eth['close'].shift(-h) / raw_eth['close'] - 1).astype(np.float32)
        label = (ret_future > 0).astype(np.int8)
        
        # 对每个 horizon 单独 dropna
        valid_mask = feats.notna().all(axis=1) & ret_future.notna()
        valid_idx = np.where(valid_mask.values)[0]
        
        datasets[h] = {
            'feat_cols': feat_cols,
            'feats': feats.values[valid_idx].astype(np.float32),
            'label': label.values[valid_idx],
            'ret_future': ret_future.values[valid_idx],
            'ts': ts[valid_idx],
            'hour': dt_idx.dt.hour.values[valid_idx],
            'n': len(valid_idx),
        }
        print(f"  H={h:>3d}: valid={len(valid_idx):,}", flush=True)
    
    del feats, raw_eth, raw_btc; gc.collect()
    print(f"[build] done ({time.time()-t0:.1f}s)", flush=True)
    return datasets

# ──────────────────────────────────────────────
# 单 horizon 全时段模型训练 + 评估
# ──────────────────────────────────────────────
def train_single_horizon(h_dataset, horizon, subsample_n=800_000):
    """训练一个完整 horizon 模型, 返回各 split 的 preds"""
    d = h_dataset
    ts = d['ts']
    m_tr = ts_mask(ts, (2020, 1, 1), (2024, 6, 30))
    m_es = ts_mask(ts, (2024, 6, 30), (2024, 9, 30))
    m_mv = ts_mask(ts, (2024, 9, 30), (2025, 9, 30))
    m_te = ts_mask(ts, (2025, 9, 30), (2026, 8, 29))
    
    Xtr, ytr = subsample(d['feats'][m_tr], d['label'][m_tr], subsample_n)
    Xes, yes = d['feats'][m_es], d['label'][m_es]
    
    model = train_lgb(Xtr, ytr, Xes, yes)
    
    # 预测各 split
    preds = {}
    for name, mask in [('train', m_tr), ('es', m_es), ('meta_val', m_mv), ('test', m_te)]:
        preds[name] = {
            'pred': model.predict(d['feats'][mask]),
            'y': d['label'][mask],
            'ret': d['ret_future'][mask],
            'ts': ts[mask],
            'hour': d['hour'][mask],
        }
    
    bi = model.best_iteration
    es_auc = roc_auc_score(yes, preds['es']['pred'])
    te_auc = roc_auc_score(preds['test']['y'], preds['test']['pred'])
    del model, Xtr, ytr, Xes, yes; gc.collect()
    
    return preds, bi, es_auc, te_auc

# ──────────────────────────────────────────────
# Union 评估
# ──────────────────────────────────────────────
def evaluate_union(all_test_preds, horizons, k_per_horizon):
    """
    每个 horizon 取 top k_per_horizon (float, e.g. 0.05 = top 5%),
    然后并集评估 acc & ret & tpd
    
    注意: 不同 horizon 的 ts 不完全对齐 (H=30 的 test 集比 H=3 少 30 根)
    用 1min bar ts 做 key 对齐, 取交集做公平评估
    """
    # 按 ts 对齐 (取所有 horizon 都有的 ts)
    ts_sets = {h: set(d['ts']) for h, d in all_test_preds.items()}
    common_ts = set.intersection(*ts_sets.values())
    print(f"\n  Union: 共同 ts 数 = {len(common_ts):,}", flush=True)
    
    union_idx = []  # (horizon, pred, y, ret, ts)
    for h in horizons:
        d = all_test_preds[h]
        # 只保留 common_ts
        keep = np.array([t in common_ts for t in d['ts']])
        sub_ts = d['ts'][keep]
        sub_pred = d['pred'][keep]
        sub_y = d['y'][keep]
        sub_ret = d['ret'][keep]
        
        n_per_h = max(int(len(sub_pred) * k_per_horizon), 10)
        top_pos = np.argsort(sub_pred)[-n_per_h:]
        
        for i in top_pos:
            union_idx.append((h, sub_pred[i], sub_y[i], sub_ret[i], sub_ts[i]))
    
    # 用 ts 去重 (同一根 bar 被多个 horizon 同时选中只算一次)
    # 保留 pred 最高的那个
    best_by_ts = {}
    for h, p, y, r, t in union_idx:
        if t not in best_by_ts or p > best_by_ts[t][1]:
            best_by_ts[t] = (h, p, y, r, t)
    
    all_y = np.array([v[2] for v in best_by_ts.values()])
    all_ret = np.array([v[3] for v in best_by_ts.values()])
    n_test_per_day = 24 * 60  # 1min bar
    days_in_test = len(common_ts) / n_test_per_day
    tpd = len(all_y) / days_in_test
    acc = (all_y == 1).mean()
    ret_bps = all_ret.mean() * 10000
    
    return {
        'k_per_horizon': k_per_horizon,
        'n_signals': len(all_y),
        'tpd': float(tpd),
        'acc': float(acc),
        'ret_bps': float(ret_bps),
        'days_in_test': float(days_in_test),
    }

# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    print("=" * 90, flush=True)
    print("  Multi-Horizon Union — ETH, 全局 LGB", flush=True)
    print("  Horizons: H=3, 5, 15, 30, 60", flush=True)
    print("  目标: tpd >= 15, acc >= 65%", flush=True)
    print("=" * 90, flush=True)
    
    HORIZONS = [3, 5, 15, 30, 60]
    
    # 1. 构建所有 horizon 数据集
    datasets = build_dataset_all_horizons(HORIZONS)
    
    # 2. 训练每个 horizon
    all_preds = {}
    horizon_summary = []
    
    print("\n" + "=" * 90, flush=True)
    print("  训练各 horizon 全局模型", flush=True)
    print("=" * 90, flush=True)
    
    for h in HORIZONS:
        t0 = time.time()
        print(f"\n  [H={h}]", flush=True)
        
        preds, bi, es_auc, te_auc = train_single_horizon(datasets[h], h)
        all_preds[h] = preds['test']
        
        # 汇总各 topK
        test_stats = topk_stats(preds['test']['pred'], preds['test']['y'], preds['test']['ret'])
        bpd = 24 * 60 // h
        row = {
            'h': h, 'bi': int(bi), 'es_auc': float(es_auc), 'te_auc': float(te_auc),
            'bars_per_day': bpd, 'label_ratio': float(preds['test']['y'].mean()),
        }
        for k, v in test_stats.items():
            row[f'top{k}_acc'] = v['acc']
            row[f'top{k}_ret_bps'] = v['ret_bps']
            row[f'top{k}_tpd'] = bpd * k
        horizon_summary.append(row)
        
        print(f"    bi={bi} | es={es_auc:.4f} te={te_auc:.4f} | bars/day={bpd}", flush=True)
        for k, v in test_stats.items():
            print(f"    top{k*100:.0f}%: acc={v['acc']:.3f} ret={v['ret_bps']:+.1f}bps tpd={bpd*k:.2f}", flush=True)
        print(f"    [耗时 {time.time()-t0:.0f}s]", flush=True)
    
    # 3. Union 评估: 每个 horizon 取不同的 k
    print("\n" + "=" * 90, flush=True)
    print("  Union 评估: 每个 horizon 取 top K%, 并集", flush=True)
    print("=" * 90, flush=True)
    
    union_results = []
    for k in [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20]:
        r = evaluate_union(all_preds, HORIZONS, k)
        r['tpd_target_met'] = r['tpd'] >= 15
        r['acc_target_met'] = r['acc'] >= 0.65
        r['both_met'] = r['tpd_target_met'] and r['acc_target_met']
        union_results.append(r)
        ok = "🎯 达标!" if r['both_met'] else f"{'✅' if r['tpd_target_met'] else ''}{'✅' if r['acc_target_met'] else ''}"
        print(f"  k={k*100:.0f}%: tpd={r['tpd']:.1f} acc={r['acc']:.3f} ret={r['ret_bps']:+.1f}bps | {ok}", flush=True)
    
    # 4. 保存
    results = {
        'horizon_summary': horizon_summary,
        'union_results': union_results,
        'horizons': HORIZONS,
    }
    with open(OUT_DIR / 'mh_union_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # 保存每个 horizon 的 meta_val + test preds 供后续 per-hour / union 使用
    for h in HORIZONS:
        preds, bi, es_auc, te_auc = train_single_horizon(datasets[h], h)
        pickle.dump(preds, open(OUT_DIR / f'preds_h{h}.pkl', 'wb'))
    
    print(f"\n[保存] {OUT_DIR}/", flush=True)
    print("\n[MH_UNION] 完成", flush=True)

if __name__ == '__main__':
    main()
