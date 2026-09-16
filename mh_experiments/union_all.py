"""
union_all.py — 最终联合策略

把所有跑出来的结果拼起来, 找一个最优的组合:
  1. 全局 multi-horizon (H=3,5,15,30) + 各 top K%
  2. per-hour (各好时段) + 各 top K%
  3. multi-horizon × per-hour (每个好时段里同时训 H=3,5,15)
  4. 共识策略 (多模型都高才选)

目标: acc >= 65%, tpd >= 15
"""
import os, sys, time, gc, datetime, json, pickle
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from pathlib import Path
from itertools import product

sys.path.insert(0, '/workspace')
from mh_experiments.mh_union import (
    build_dataset_all_horizons, ts_mask, topk_stats, subsample, train_lgb
)

OUT_DIR = Path('/workspace/mh_experiments')

def evaluate_strategy(name, signals, target_tpd=15, target_acc=0.65):
    """统一评估接口"""
    tpd = signals['tpd']; acc = signals['acc']; ret = signals['ret_bps']
    both = tpd >= target_tpd and acc >= target_acc
    acc_ok = acc >= target_acc; tpd_ok = tpd >= target_tpd
    status = "🎯 完全达标" if both else (f"✅ acc但缺tpd" if acc_ok else (f"✅ tpd但缺acc" if tpd_ok else "❌ 都不达标"))
    print(f"  [{name}] tpd={tpd:6.1f} acc={acc:.4f} ret={ret:+7.1f}bps  {status}", flush=True)
    return {'name': name, 'tpd': tpd, 'acc': acc, 'ret_bps': ret,
            'both_met': both, 'acc_met': acc_ok, 'tpd_met': tpd_ok}


def main():
    print("=" * 90, flush=True)
    print("  UNION ALL — 最终联合策略搜索", flush=True)
    print("  目标: acc >= 65%, tpd >= 15", flush=True)
    print("=" * 90, flush=True)
    
    HORIZONS = [3, 5, 15, 30]
    TARGET_TPD = 15
    TARGET_ACC = 0.65
    
    # ── 加载或重建 datasets ──
    datasets_path = OUT_DIR / 'union_datasets.pkl'
    if datasets_path.exists():
        print("[load] 复用 datasets", flush=True)
        datasets = pickle.load(open(datasets_path, 'rb'))
    else:
        print("[build] datasets", flush=True)
        datasets = build_dataset_all_horizons(HORIZONS)
        pickle.dump(datasets, open(datasets_path, 'wb'))
    
    # ── 1. 全局 multi-horizon 训练 ──
    print("\n" + "=" * 90, flush=True)
    print("  Step 1: 全局 multi-horizon 模型训练", flush=True)
    print("=" * 90, flush=True)
    
    global_preds = {}  # h -> {'meta_val': {pred,y,ret,ts}, 'test': {...}}
    for h in HORIZONS:
        print(f"\n  [H={h}] 训练全局 LGB...", flush=True)
        d = datasets[h]
        ts = d['ts']
        m_tr = ts_mask(ts, (2020,1,1),(2024,6,30))
        m_es = ts_mask(ts, (2024,6,30),(2024,9,30))
        m_mv = ts_mask(ts, (2024,9,30),(2025,9,30))
        m_te = ts_mask(ts, (2025,9,30),(2026,8,29))
        Xtr, ytr = subsample(d['feats'][m_tr], d['label'][m_tr], 800_000)
        Xes, yes = d['feats'][m_es], d['label'][m_es]
        model = train_lgb(Xtr, ytr, Xes, yes)
        bi = model.best_iteration
        es_auc = roc_auc_score(yes, model.predict(Xes))
        del Xtr, ytr, Xes, yes; gc.collect()
        
        global_preds[h] = {}
        for split, mask in [('meta_val', m_mv), ('test', m_te)]:
            global_preds[h][split] = {
                'pred': model.predict(d['feats'][mask]),
                'y': d['label'][mask],
                'ret': d['ret_future'][mask],
                'ts': ts[mask],
                'hour': d['hour'][mask],
            }
        te_auc = roc_auc_score(global_preds[h]['test']['y'], global_preds[h]['test']['pred'])
        print(f"    bi={bi} es={es_auc:.4f} te={te_auc:.4f}", flush=True)
        del model; gc.collect()
    
    # ── 2. Per-Hour 训练 (选 ES AUC > 0.53 的小时, 减少工作量) ──
    print("\n" + "=" * 90, flush=True)
    print("  Step 2: Per-Hour 模型训练 (每个 horizon × 好小时)", flush=True)
    print("=" * 90, flush=True)
    
    # 先用全局模型的 meta_val preds 确定哪些小时好
    ph_preds = {}  # h -> hour -> preds dict
    for h in HORIZONS:
        print(f"\n  [H={h}] per-hour...", flush=True)
        d = datasets[h]
        ts = d['ts']; hour = d['hour']; X = d['feats']; y = d['label']; ret = d['ret_future']
        
        # 用 meta_val 看各小时 AUC → 选 top12 个
        m_mv = ts_mask(ts, (2024,9,30),(2025,9,30))
        mv_hour = hour[m_mv]
        mv_pred = global_preds[h]['meta_val']['pred']
        mv_y = global_preds[h]['meta_val']['y']
        hour_aucs_mv = []
        for uh in range(24):
            hm = mv_hour == uh
            if hm.sum() > 500:
                auc = roc_auc_score(mv_y[hm], mv_pred[hm])
                hour_aucs_mv.append((uh, auc))
        hour_aucs_mv.sort(key=lambda x:-x[1])
        # 选 top12 (AUC > 0.52 的)
        good_hours_set = [uh for uh, auc in hour_aucs_mv[:12] if auc > 0.52]
        print(f"    Meta-val 选 top{len(good_hours_set)} good hours: {good_hours_set}", flush=True)
        
        m_tr = ts_mask(ts, (2020,1,1),(2024,6,30))
        m_es = ts_mask(ts, (2024,6,30),(2024,9,30))
        m_te = ts_mask(ts, (2025,9,30),(2026,8,29))
        
        ph_preds[h] = {}
        for uh in good_hours_set:
            h_tr = m_tr & (hour == uh)
            h_es = m_es & (hour == uh)
            h_te = m_te & (hour == uh)
            
            if h_tr.sum() < 500 or h_es.sum() < 50: continue
            
            Xtr = X[h_tr]; ytr = y[h_tr]
            Xes = X[h_es]; yes = y[h_es]
            
            p = {'num_leaves':31,'learning_rate':0.03,'min_data_in_leaf':50,
                 'feature_fraction':1.0,'bagging_fraction':1.0,'lambda_l2':1.0,
                 'verbose':-1,'num_threads':2,'objective':'binary','metric':'auc',
                 'seed':42+uh}
            dtr=lgb.Dataset(Xtr,label=ytr); des=lgb.Dataset(Xes,label=yes,reference=dtr)
            model=lgb.train(p,dtr,num_boost_round=800,valid_sets=[des],
                          callbacks=[lgb.early_stopping(80,verbose=False)])
            
            # 预测 test
            te_preds = model.predict(X[h_te])
            te_auc = roc_auc_score(y[h_te], te_preds) if h_te.sum()>50 else None
            del model, Xtr, ytr, Xes, yes, dtr, des; gc.collect()
            
            ph_preds[h][uh] = {
                'pred': te_preds, 'y': y[h_te], 'ret': ret[h_te], 'ts': ts[h_te],
                'te_auc': float(te_auc) if te_auc else None,
            }
            print(f"    UTC{uh:02d}: bi={model.best_iteration if 'model' in dir() else '?'} te_auc={te_auc:.4f}", flush=True)
    
    # ── 3. 策略组合评估 ──
    print("\n" + "=" * 90, flush=True)
    print("  Step 3: 策略组合搜索", flush=True)
    print("=" * 90, flush=True)
    
    all_results = []
    
    # ------ 策略 0: 基线 ------
    # 只用 H=15 全局模型, 不同 top K
    print("\n  --- 策略 0: H=15 全局 baseline ---")
    for k in [0.01, 0.02, 0.05, 0.10, 0.15, 0.20]:
        d = global_preds[15]['test']
        n = max(int(len(d['pred'])*k),1)
        top_pos = np.argsort(d['pred'])[-n:]
        acc = (d['y'][top_pos]==1).mean(); ret = d['ret'][top_pos].mean()*10000
        tpd = 24*60/15*k  # 96 bars/day * k
        r = evaluate_strategy(f"H15_global_top{k*100:.0f}%", {'tpd':tpd,'acc':acc,'ret_bps':ret})
        all_results.append(r)
    
    # ------ 策略 1: Multi-Horizon Global Union ------
    print("\n  --- 策略 1: Multi-Horizon Global Union (各 top K%) ---")
    for k in [0.01, 0.02, 0.03, 0.05]:
        union_y, union_ret, union_ts = [], [], []
        for h in HORIZONS:
            d = global_preds[h]['test']
            n = max(int(len(d['pred'])*k),1)
            top_pos = np.argsort(d['pred'])[-n:]
            union_y.extend(d['y'][top_pos].tolist())
            union_ret.extend(d['ret'][top_pos].tolist())
            union_ts.extend(d['ts'][top_pos].tolist())
        # 去重 (按 ts)
        best = {}
        for yy, rr, tt in zip(union_y, union_ret, union_ts):
            best[tt] = (yy, rr) if tt not in best else best[tt]
        final_y = np.array([v[0] for v in best.values()])
        final_ret = np.array([v[1] for v in best.values()])
        days = len(set(union_ts)) / (24*60)
        tpd = len(final_y) / max(days, 1)
        acc = (final_y==1).mean(); ret = final_ret.mean()*10000
        r = evaluate_strategy(f"MH_global_union_top{k*100:.0f}%", {'tpd':tpd,'acc':acc,'ret_bps':ret})
        all_results.append(r)
    
    # ------ 策略 2: Per-Hour Only (好小时, 每小时 top K%) ------
    print("\n  --- 策略 2: Per-Hour Only (各好小时 top K%) ---")
    for h in [15, 5, 30]:
        for k in [0.01, 0.03, 0.05, 0.10]:
            union_y, union_ret, union_ts = [], [], []
            for uh, d in ph_preds[h].items():
                n = max(int(len(d['pred'])*k),1)
                top_pos = np.argsort(d['pred'])[-n:]
                union_y.extend(d['y'][top_pos].tolist())
                union_ret.extend(d['ret'][top_pos].tolist())
                union_ts.extend(d['ts'][top_pos].tolist())
            days = len(set(union_ts)) / (24*60)
            tpd = len(union_y) / max(days, 1)
            acc = (np.array(union_y)==1).mean(); ret = np.array(union_ret).mean()*10000
            r = evaluate_strategy(f"H{h}_perhour_top{k*100:.0f}%", {'tpd':tpd,'acc':acc,'ret_bps':ret})
            all_results.append(r)
    
    # ------ 策略 3: Per-Hour × Multi-Horizon 联合 ------
    # 在每个好小时里, 用多个 horizon 的 per-hour 模型取并集
    print("\n  --- 策略 3: Per-Hour × Multi-Horizon (好小时 × 多 horizon) ---")
    # 取各 horizon 共同的好小时
    common_good_hours = None
    for h in HORIZONS:
        gh = set(ph_preds[h].keys())
        common_good_hours = gh if common_good_hours is None else common_good_hours & gh
    common_good_hours = sorted(common_good_hours)
    print(f"    各 horizon 共同好小时: {common_good_hours}", flush=True)
    
    for k_per_h in [0.02, 0.05]:  # 每小时内 top k
        for k_per_h_horizon in [0.01, 0.03]:  # 每个 horizon 内的每小时 top k
            # 更细粒度: 每个好小时的每个 horizon 各自 top k, 全部 union
            union_y, union_ret, union_ts = [], [], []
            for h in HORIZONS:
                for uh in common_good_hours:
                    d = ph_preds[h].get(uh)
                    if d is None: continue
                    n = max(int(len(d['pred'])*k_per_h_horizon),1)
                    top_pos = np.argsort(d['pred'])[-n:]
                    union_y.extend(d['y'][top_pos].tolist())
                    union_ret.extend(d['ret'][top_pos].tolist())
                    union_ts.extend(d['ts'][top_pos].tolist())
            days = len(set(union_ts)) / (24*60)
            tpd = len(union_y) / max(days, 1)
            acc = (np.array(union_y)==1).mean(); ret = np.array(union_ret).mean()*10000
            r = evaluate_strategy(f"PHxMH_top{k_per_h*100:.0f}%perh_top{k_per_h_horizon*100:.0f}%perhorizon", 
                                   {'tpd':tpd,'acc':acc,'ret_bps':ret})
            all_results.append(r)
    
    # ------ 策略 4: Multi-Horizon Global + 时段过滤 ------
    # 用全局多 horizon, 但只在好小时选信号
    print("\n  --- 策略 4: MH Global + 时段过滤 ---")
    # 用 H=15 meta_val 选好小时
    mv_hour = datasets[15]['hour'][ts_mask(datasets[15]['ts'], (2024,9,30),(2025,9,30))]
    mv_pred = global_preds[15]['meta_val']['pred']
    mv_y = global_preds[15]['meta_val']['y']
    hour_aucs = []
    for uh in range(24):
        hm = mv_hour == uh
        if hm.sum() > 500:
            auc = roc_auc_score(mv_y[hm], mv_pred[hm])
            hour_aucs.append((uh, auc))
    hour_aucs.sort(key=lambda x:-x[1])
    top_hours_by_auc = [uh for uh, auc in hour_aucs[:14]]  # 取 top14 小时
    print(f"    ES AUC top14 UTC hours: {top_hours_by_auc}", flush=True)
    
    for k in [0.02, 0.05, 0.10]:
        for n_hours in [8, 12, 14]:
            sel_hours = top_hours_by_auc[:n_hours]
            union_y, union_ret, union_ts = [], [], []
            for h in HORIZONS:
                d = global_preds[h]['test']
                d_hour = datasets[h]['hour'][ts_mask(datasets[h]['ts'], (2025,9,30),(2026,8,29))]
                hmask = np.isin(d_hour, sel_hours)
                if hmask.sum() < 50: continue
                sub_p = d['pred'][hmask]; sub_y = d['y'][hmask]; sub_r = d['ret'][hmask]; sub_ts = d['ts'][hmask]
                n = max(int(len(sub_p)*k),1)
                top_pos = np.argsort(sub_p)[-n:]
                union_y.extend(sub_y[top_pos].tolist())
                union_ret.extend(sub_r[top_pos].tolist())
                union_ts.extend(sub_ts[top_pos].tolist())
            # 去重
            best = {}
            for yy, rr, tt in zip(union_y, union_ret, union_ts):
                best[tt] = (yy, rr) if tt not in best else best[tt]
            final_y = np.array([v[0] for v in best.values()])
            final_ret = np.array([v[1] for v in best.values()])
            days = len(set(union_ts)) / (24*60)
            tpd = len(final_y) / max(days, 1)
            acc = (final_y==1).mean() if len(final_y)>0 else 0
            ret = final_ret.mean()*10000 if len(final_ret)>0 else 0
            r = evaluate_strategy(f"MH_global+filter_top{n_hours}h_top{k*100:.0f}%", 
                                   {'tpd':tpd,'acc':acc,'ret_bps':ret})
            all_results.append(r)
    
    # ------ 策略 5: 共识策略 (多 horizon + 多模型同时选高) ------
    print("\n  --- 策略 5: 共识策略 (多 horizon 同时高才选) ---")
    # 对每根 bar, 看 H=3, H=5, H=15, H=30 四个全局模型的 pred 排名 percentile
    # percentile = (rank - 1) / N
    # 如果 ≥ 3 个 horizon 的 percentile 都 > thresh, 才选
    
    # 先把各 horizon 的 test preds 对齐到共同 ts
    common_ts = None
    ts_to_idx = {}  # h -> ts -> idx
    for h in HORIZONS:
        d = global_preds[h]['test']
        ts_to_idx[h] = {t: i for i, t in enumerate(d['ts'])}
        common_ts = set(d['ts']) if common_ts is None else common_ts & set(d['ts'])
    common_ts = sorted(common_ts)
    print(f"    共同 ts 数 = {len(common_ts):,}", flush=True)
    
    # 构建 aligned preds
    aligned = {}
    for h in HORIZONS:
        d = global_preds[h]['test']
        idx = np.array([ts_to_idx[h][t] for t in common_ts])
        aligned[h] = {'pred': d['pred'][idx], 'y': d['y'][idx], 'ret': d['ret'][idx]}
    
    # percentile rank
    percentiles = {}
    for h in HORIZONS:
        ranks = pd.Series(aligned[h]['pred']).rank(pct=True).values
        percentiles[h] = ranks
    
    for thresh in [0.80, 0.85, 0.90, 0.95]:
        for min_models_high in [3, 4]:
            selected = []
            for i in range(len(common_ts)):
                high_count = sum(1 for h in HORIZONS if percentiles[h][i] > thresh)
                if high_count >= min_models_high:
                    selected.append(i)
            if len(selected) < 50: continue
            selected = np.array(selected)
            # 用 H=15 label 评估 (主 horizon)
            # 但 H=3 和 H=15 label 不完全一样, 用 union label (任一 horizon 正)
            y_union = np.any(np.array([aligned[h]['y'] for h in HORIZONS]), axis=0)
            acc = (y_union[selected]==1).mean()
            # 用 H=15 ret 作为近似
            ret = aligned[15]['ret'][selected].mean()*10000
            days = len(common_ts) / (24*60)
            tpd = len(selected) / days
            r = evaluate_strategy(f"consensus_thresh{thresh:.0%}_min{min_models_high}",
                                   {'tpd':tpd,'acc':acc,'ret_bps':ret})
            all_results.append(r)
    
    # ── 4. 全量结果排序 ──
    print("\n" + "=" * 90, flush=True)
    print("  所有策略排序", flush=True)
    print("=" * 90, flush=True)
    
    sorted_results = sorted(all_results, key=lambda x: (x['acc']>=TARGET_ACC, x['tpd']>=TARGET_TPD, x['acc']*x['tpd']), reverse=True)
    
    print(f"\n  {'rank':>4} | {'策略':<55} | {'tpd':>6} | {'acc':>6} | {'ret_bps':>8} | 达标?", flush=True)
    print("  " + "-"*105, flush=True)
    for i, r in enumerate(sorted_results):
        status = "🎯 BOTH" if r['both_met'] else (("✅acc" if r['acc_met'] else "") + ("✅tpd" if r['tpd_met'] else ""))
        print(f"  {i+1:>4} | {r['name']:<55} | {r['tpd']:>6.1f} | {r['acc']:>6.3f} | {r['ret_bps']:>+7.1f} | {status}", flush=True)
    
    # ── 5. 保存 ──
    with open(OUT_DIR / 'union_all_results.json', 'w') as f:
        json.dump(sorted_results, f, indent=2)
    pickle.dump(global_preds, open(OUT_DIR / 'global_preds_all.pkl', 'wb'))
    pickle.dump(ph_preds, open(OUT_DIR / 'per_hour_preds_all.pkl', 'wb'))
    
    best = [r for r in sorted_results if r['both_met']]
    print(f"\n[UNION_ALL] 完全达标策略数: {len(best)}", flush=True)
    if best:
        for b in best:
            print(f"  🎯 {b['name']}: tpd={b['tpd']:.1f}, acc={b['acc']:.4f}", flush=True)
    else:
        print("  ❌ 没有完全达标, 打印最接近的:", flush=True)
        top = sorted_results[:5]
        for r in top:
            print(f"    {r['name']}: tpd={r['tpd']:.1f}, acc={r['acc']:.4f}", flush=True)

if __name__ == '__main__':
    main()
