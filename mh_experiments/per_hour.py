"""
per_hour.py — Per-Hour 独立模型

思路: 24 个 UTC 小时, 每个小时有完全不同的 label 分布和市场状态:
  - UTC 06 (欧洲早市横盘): top1% acc 只有 46.2%
  - UTC 18 (美盘开盘): top1% acc 79.9%

一个统一的全局模型只能学到折中。
Per-Hour 模型 = 每个 UTC 小时单独训练一个 LGB, 各自优化。

这个脚本:
  1. 对指定 horizon, 训练 24 个 per-hour LGB
  2. 每个小时用自己的早停验证集校准
  3. 评估各小时 AUC + topK acc
  4. 保存各小时 meta_val + test preds
  5. 输出整体 union 后的 tpd & acc (好小时 union)
"""
import os, sys, time, gc, datetime, json, pickle
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from pathlib import Path

sys.path.insert(0, '/workspace')
from mh_experiments.mh_union import (
    build_dataset_all_horizons, ts_mask, topk_stats, subsample
)

OUT_DIR = Path('/workspace/mh_experiments')

def train_per_hour(h_dataset, horizon, max_train_per_hour=300_000):
    """训练 24 个 per-hour LGB, 返回各 split 的 preds + 汇总"""
    ts = h_dataset['ts']
    hour = h_dataset['hour']
    X = h_dataset['feats']
    y = h_dataset['label']
    ret = h_dataset['ret_future']
    
    # 全局 mask
    m_tr = ts_mask(ts, (2020, 1, 1), (2024, 6, 30))
    m_es = ts_mask(ts, (2024, 6, 30), (2024, 9, 30))
    m_mv = ts_mask(ts, (2024, 9, 30), (2025, 9, 30))
    m_te = ts_mask(ts, (2025, 9, 30), (2026, 8, 29))
    
    per_hour_preds = {}
    summary = []
    
    for h in range(24):
        t0 = time.time()
        h_tr = m_tr & (hour == h)
        h_es = m_es & (hour == h)
        h_mv = m_mv & (hour == h)
        h_te = m_te & (hour == h)
        
        if h_tr.sum() < 1000 or h_es.sum() < 100:
            print(f"  UTC {h:02d}: train={h_tr.sum()} es={h_es.sum()} SKIP (太少)", flush=True)
            summary.append({'hour': h, 'bi': None, 'es_auc': None, 'te_auc': None, 'n_tr': int(h_tr.sum())})
            continue
        
        Xtr = X[h_tr]; ytr = y[h_tr]
        Xes = X[h_es]; yes = y[h_es]
        
        # subsample
        if len(Xtr) > max_train_per_hour:
            rng = np.random.RandomState(42 + h)
            idx = rng.choice(len(Xtr), max_train_per_hour, replace=False)
            Xtr, ytr = Xtr[idx], ytr[idx]
        
        # per-hour 用更简单的 params (数据量小)
        p = {
            'num_leaves': 31, 'learning_rate': 0.03, 'min_data_in_leaf': 50,
            'feature_fraction': 1.0, 'bagging_fraction': 1.0, 'lambda_l2': 1.0,
            'verbose': -1, 'num_threads': 2, 'objective': 'binary', 'metric': 'auc',
            'seed': 42 + h,
        }
        dtr = lgb.Dataset(Xtr, label=ytr)
        des = lgb.Dataset(Xes, label=yes, reference=dtr)
        model = lgb.train(p, dtr, num_boost_round=1000, valid_sets=[des],
                         callbacks=[lgb.early_stopping(100, verbose=False)])
        
        bi = model.best_iteration
        es_auc = roc_auc_score(yes, model.predict(Xes))
        del Xtr, ytr, Xes, yes, dtr, des; gc.collect()
        
        # 预测 meta_val + test (这个小时内)
        mv_preds = model.predict(X[h_mv])
        te_preds = model.predict(X[h_te])
        te_auc = roc_auc_score(y[h_te], te_preds) if h_te.sum() > 50 else None
        
        per_hour_preds[h] = {
            'hour': h,
            'bi': bi,
            'es_auc': float(es_auc),
            'te_auc': float(te_auc) if te_auc else None,
            'meta_val': {'pred': mv_preds, 'y': y[h_mv], 'ret': ret[h_mv], 'ts': ts[h_mv]},
            'test': {'pred': te_preds, 'y': y[h_te], 'ret': ret[h_te], 'ts': ts[h_te]},
        }
        
        # test top1% 快速评估
        if h_te.sum() > 50:
            t1 = topk_stats(te_preds, y[h_te], ret[h_te], [0.01, 0.05])
            summary.append({
                'hour': h, 'bi': int(bi), 'es_auc': float(es_auc), 'te_auc': float(te_auc),
                'n_tr': int(h_tr.sum()), 'n_te': int(h_te.sum()),
                'top1_acc': t1[0.01]['acc'], 'top1_ret': t1[0.01]['ret_bps'],
                'top5_acc': t1[0.05]['acc'], 'top5_ret': t1[0.05]['ret_bps'],
            })
            print(f"  UTC {h:02d}: bi={bi} es={es_auc:.4f} te={te_auc:.4f} | top1%={t1[0.01]['acc']:.3f}({t1[0.01]['ret_bps']:+.1f}bps) top5%={t1[0.05]['acc']:.3f}({t1[0.05]['ret_bps']:+.1f}bps) | {time.time()-t0:.0f}s", flush=True)
        else:
            summary.append({'hour': h, 'bi': int(bi), 'es_auc': float(es_auc), 'te_auc': None,
                          'n_tr': int(h_tr.sum()), 'n_te': 0})
    
    return per_hour_preds, summary


def evaluate_per_hour_union(per_hour_preds, good_hours=None, k_per_hour=0.05):
    """
    在指定的好小时里, 每个小时取 top K%, 合并评估
    如果 good_hours=None, 用 ES 上 AUC > 0.53 的小时
    """
    if good_hours is None:
        good_hours = [h for h, d in per_hour_preds.items() 
                      if d.get('es_auc') and d['es_auc'] > 0.53]
    
    all_y, all_ret, all_ts = [], [], []
    
    for h in good_hours:
        d = per_hour_preds[h]['test']
        if len(d['pred']) == 0: continue
        n = max(int(len(d['pred']) * k_per_hour), 1)
        top_pos = np.argsort(d['pred'])[-n:]
        all_y.extend(d['y'][top_pos].tolist())
        all_ret.extend(d['ret'][top_pos].tolist())
        all_ts.extend(d['ts'][top_pos].tolist())
    
    all_y = np.array(all_y); all_ret = np.array(all_ret)
    # 按 ts 去重 (同一根 bar 可能被多个小时预测 → 不会, 因为每个小时的 bar 不重叠)
    days_in_test = len(set(all_ts)) / (24 * 60) if len(all_ts) > 0 else 1
    tpd = len(all_y) / max(days_in_test, 1)
    acc = (all_y == 1).mean() if len(all_y) > 0 else 0
    ret_bps = all_ret.mean() * 10000 if len(all_ret) > 0 else 0
    
    return {'good_hours': good_hours, 'tpd': float(tpd), 'acc': float(acc), 
            'ret_bps': float(ret_bps), 'n_signals': len(all_y)}


def main():
    print("=" * 90, flush=True)
    print("  Per-Hour 独立模型 — ETH H=15 (重点 horizon), 再加 H=5, H=30", flush=True)
    print("=" * 90, flush=True)
    
    HORIZONS = [15, 5, 30]
    
    # 先复用 mh_union 已有的 datasets (如果有), 否则重建
    datasets_path = OUT_DIR / 'ph_datasets.pkl'
    if datasets_path.exists():
        print("[load] 复用已缓存的 datasets", flush=True)
        datasets = pickle.load(open(datasets_path, 'rb'))
    else:
        datasets = build_dataset_all_horizons(HORIZONS)
        pickle.dump(datasets, open(datasets_path, 'wb'))
    
    all_ph_results = {}
    
    for horizon in HORIZONS:
        print(f"\n{'='*90}", flush=True)
        print(f"  H={horizon} — 24 个 per-hour 模型", flush=True)
        print(f"{'='*90}", flush=True)
        
        preds, summary = train_per_hour(datasets[horizon], horizon)
        all_ph_results[horizon] = {'preds': preds, 'summary': summary}
        
        # 汇总这个 horizon 的 per-hour 效果
        aucs = [s['te_auc'] for s in summary if s.get('te_auc') is not None]
        top1_accs = [s['top1_acc'] for s in summary if 'top1_acc' in s]
        print(f"\n  H={horizon} per-hour 汇总:", flush=True)
        print(f"    te_auc: mean={np.mean(aucs):.4f} std={np.std(aucs):.4f}", flush=True)
        print(f"    top1% acc (有足够数据的小时): mean={np.mean(top1_accs):.3f} max={max(top1_accs):.3f} min={min(top1_accs):.3f}", flush=True)
        
        # 在 test 上, 用 ES 选 good hours → union 评估
        good_hours_es = sorted([s['hour'] for s in summary 
                                if s.get('es_auc') and s['es_auc'] > 0.53])
        print(f"    ES AUC > 0.53 的 UTC 小时: {good_hours_es}", flush=True)
        
        for k in [0.01, 0.03, 0.05, 0.10]:
            r = evaluate_per_hour_union(preds, good_hours_es, k)
            ok = "🎯" if (r['tpd']>=15 and r['acc']>=0.65) else ""
            print(f"    top{k*100:.0f}%/hour → tpd={r['tpd']:.1f} acc={r['acc']:.3f} ret={r['ret_bps']:+.1f}bps {ok}", flush=True)
    
    # 保存
    for horizon, data in all_ph_results.items():
        pickle.dump(data['preds'], open(OUT_DIR / f'per_hour_preds_h{horizon}.pkl', 'wb'))
        with open(OUT_DIR / f'per_hour_summary_h{horizon}.json', 'w') as f:
            json.dump(data['summary'], f, indent=2)
    
    print("\n[PER_HOUR] 完成", flush=True)

if __name__ == '__main__':
    main()
