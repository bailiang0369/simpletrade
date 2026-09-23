"""三个决定性实验回答用户质疑:
A. 去噪是真的提升了排序能力, 还是仅仅因为 q 阈值缩小覆盖率才导致 acc 增加?
B. 去噪在 BTC / 其他 horizon 上是否也有效?
C. ε=0.0005 是 sweet spot 还是偶然?
"""
import sys; sys.path.insert(0,'/workspace')
import os, gc, time
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import config
from data_store import AssetContext

SEEDS = [42, 49, 56, 63, 70, 77, 84, 91, 98, 105]
T0 = time.time()
def now(): return f"{(time.time()-T0)/60:.1f}m"

def rank_ens(arrays):
    P = np.stack(arrays, axis=0)
    R = np.zeros_like(P)
    for i in range(P.shape[0]):
        R[i] = np.argsort(np.argsort(P[i])).astype(np.float64) / (P.shape[1] - 1)
    return R.mean(axis=0)

def nolook(q, conf, pred, y, ts, win=30, cold=30):
    day_of = ts // 86400; days = np.unique(day_of)
    sel = np.zeros(len(y), dtype=bool)
    for di, d in enumerate(days.astype(int).tolist()):
        prior = days.astype(int).tolist()[max(0, di-win):di]
        if len(prior) < cold: continue
        today = day_of == d
        for side in [1, 0]:
            m = today & (pred == side)
            hist = np.isin(day_of, prior) & (pred == side)
            if hist.sum() == 0 or m.sum() == 0: continue
            tau = float(np.percentile(conf[hist], q))
            sel[m & (conf >= tau)] = True
    acc = float((pred[sel]==y[sel]).mean()) if sel.sum() else 0.0
    tpd = sel.sum() / max(len(days), 1)
    return acc*100, tpd, int(sel.sum())

def train_one(ctx, Xtr, ytr, Xes, yes, wtr, params):
    mm = ctx.split_rows["meta_val"]; mt = ctx.split_rows["test"]
    Pmv, Pte = [], []
    sau_mv, sau_te = [], []
    for s in SEEDS:
        p = {**params, "seed": s}
        tr = lgb.Dataset(Xtr, label=ytr, weight=wtr) if wtr is not None else lgb.Dataset(Xtr, label=ytr)
        es = lgb.Dataset(Xes, label=yes, reference=tr)
        m = lgb.train(p, tr, num_boost_round=3000, valid_sets=[es],
                      callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
        pmv = m.predict(ctx.Xall[mm]).astype(np.float64)
        pte = m.predict(ctx.Xall[mt]).astype(np.float64)
        Pmv.append(pmv); Pte.append(pte)
        sau_mv.append(roc_auc_score(ctx.label[mm], pmv))
        sau_te.append(roc_auc_score(ctx.label[mt], pte))
        gc.collect()
    return rank_ens(Pmv), rank_ens(Pte), np.array(sau_mv), np.array(sau_te)

def show_for(ctx, Pmv, Pte, sau_mv, sau_te, title):
    for split, P, sau in [("mv", Pmv, sau_mv), ("te", Pte, sau_te)]:
        mask = ctx.split_rows["meta_val"] if split=="mv" else ctx.split_rows["test"]
        y = ctx.label[mask]
        ts = np.asarray(ctx.times("meta_val" if split=="mv" else "test")).astype("datetime64[s]").astype(np.int64)
        conf = np.abs(P - 0.5) * 2; pred = (P >= 0.5).astype(np.int8)
        auc = roc_auc_score(y, P)
        print(f"  [{title}] {split} AUC_ens={auc:.4f}  seed_AUC={sau.mean():.4f}±{sau.std():.4f}", flush=True)
        for q in [98.0, 99.0, 99.2, 99.4, 99.5]:
            a, t, n = nolook(q, conf, pred, y, ts)
            print(f"      q={q} -> {a:.2f}% @ {t:.1f}t ({n})", flush=True)

base_params = {"objective": "binary", "metric": "auc", "verbose": -1,
               "num_leaves": 127, "min_data_in_leaf": 200,
               "learning_rate": 0.05, "feature_fraction": 0.8, "bagging_fraction": 0.8}

results = {}

# ====== 实验 A+C: ETH h15 ε 精细扫描 ======
print("\n" + "="*70, flush=True)
print("实验 A+C: ETH h15 ε 精细扫描 8 值", flush=True)
print("="*70, flush=True)

ctx = AssetContext("ETH", horizon=15)
Xtr_f = ctx.Xall[ctx.split_rows["train"]]
ytr_f = ctx.label[ctx.split_rows["train"]].astype(np.float64)
Xes = ctx.Xall[ctx.split_rows["early_stop"]]
yes = ctx.label[ctx.split_rows["early_stop"]].astype(np.float64)
ret_tr = ctx.retf("train")
wtr_f = np.clip(np.abs(ret_tr) * 50, 0.5, 5.0)

EPS_LIST = [0, 0.0001, 0.0002, 0.0003, 0.0005, 0.0008, 0.001, 0.0015]

for eps in EPS_LIST:
    print(f"\n[{now()}] eps={eps}  ({'ALL' if eps==0 else f'|ret|>{eps}'})", flush=True)
    if eps == 0:
        Xd, yd, wd = Xtr_f, ytr_f, wtr_f
    else:
        m = np.abs(ret_tr) > eps
        Xd, yd, wd = Xtr_f[m], ytr_f[m], wtr_f[m]
    print(f"  keep={len(Xd)}/{len(Xtr_f)} ({len(Xd)/len(Xtr_f)*100:.1f}%)", flush=True)

    Pmv, Pte, sau_mv, sau_te = train_one(ctx, Xd, yd, Xes, yes, wd, base_params)
    show_for(ctx, Pmv, Pte, sau_mv, sau_te, f"eps{eps}")

    results[("ETH", 15, eps)] = {
        "auc_te": roc_auc_score(ctx.label[ctx.split_rows["test"]], Pte),
        "auc_mv": roc_auc_score(ctx.label[ctx.split_rows["meta_val"]], Pmv),
        "seed_te_mean": sau_te.mean(), "seed_te_std": sau_te.std(),
    }
    gc.collect()

# ====== 实验 B: BTC h15 / ETH h30 / BTC h30 ======
print("\n" + "="*70, flush=True)
print("实验 B: BTC h15, ETH h30, BTC h30 — ε=0 vs 0.0005", flush=True)
print("="*70, flush=True)

for sym, h in [("BTC", 15), ("ETH", 30), ("BTC", 30)]:
    print(f"\n--- {sym} h{h} ---", flush=True)
    ctx2 = AssetContext(sym, horizon=h)
    Xtr_f2 = ctx2.Xall[ctx2.split_rows["train"]]
    ytr_f2 = ctx2.label[ctx2.split_rows["train"]].astype(np.float64)
    Xes2 = ctx2.Xall[ctx2.split_rows["early_stop"]]
    yes2 = ctx2.label[ctx2.split_rows["early_stop"]].astype(np.float64)
    ret_tr2 = ctx2.retf("train")
    wtr_f2 = np.clip(np.abs(ret_tr2) * 50, 0.5, 5.0)

    for eps in [0, 0.0005]:
        print(f"\n  eps={eps}:", flush=True)
        if eps == 0:
            Xd, yd, wd = Xtr_f2, ytr_f2, wtr_f2
        else:
            m = np.abs(ret_tr2) > eps
            Xd, yd, wd = Xtr_f2[m], ytr_f2[m], wtr_f2[m]
        print(f"    keep={len(Xd)}/{len(Xtr_f2)} ({len(Xd)/len(Xtr_f2)*100:.1f}%)", flush=True)

        Pmv, Pte, sau_mv, sau_te = train_one(ctx2, Xd, yd, Xes2, yes2, wd, base_params)
        show_for(ctx2, Pmv, Pte, sau_mv, sau_te, f"{sym}h{h}_eps{eps}")

        results[(sym, h, eps)] = {
            "auc_te": roc_auc_score(ctx2.label[ctx2.split_rows["test"]], Pte),
            "seed_te_mean": sau_te.mean(), "seed_te_std": sau_te.std(),
        }
        gc.collect()
    del ctx2, Xtr_f2, ytr_f2, Xes2, yes2, ret_tr2, wtr_f2

# ====== 最终判定 ======
print("\n" + "="*70, flush=True)
print("判定: 去噪是真提升还是 q-threshold artifact?", flush=True)
print("="*70, flush=True)
print("""
判定标准:
  ✅ 真提升: AUC_ens 或 seed_AUC_mean 显著提升 (>0.0003)
  ❌ artifact: AUC 不变, 仅 acc 在高 q 提升 (纯 coverage trade-off)
  ⚠ 混合: AUC 微升 (<0.0003) 但 seed_AUC_mean 稳定
""", flush=True)

for sym, h in [("ETH", 15), ("BTC", 15), ("ETH", 30), ("BTC", 30)]:
    base = results.get((sym, h, 0))
    dn = results.get((sym, h, 0.0005))
    if not base or not dn: continue
    d_ens = dn["auc_te"] - base["auc_te"]
    d_seed = dn["seed_te_mean"] - base["seed_te_mean"]
    pct = d_seed / base["seed_te_mean"] * 100
    if d_ens > 0.0005: v = "✅ AUC 显著提升"
    elif d_ens > 0: v = "⚠ AUC 微升"
    elif d_ens > -0.0002: v = "➖ AUC 无变化"
    else: v = "❌ AUC 下降"
    print(f"  {sym} h{h}: ens_AUC {base['auc_te']:.4f}->{dn['auc_te']:.4f} (Δ{d_ens:+.4f})  "
          f"seed_AUC {base['seed_te_mean']:.4f}->{dn['seed_te_mean']:.4f} (Δ{d_seed:+.4f}, {pct:+.2f}%)  {v}", flush=True)

# ETH h15 ε 扫描表
print(f"\nETH h15 ε 精细扫描:", flush=True)
print(f"{'ε':>8} {'保留%':>7} {'te_AUC':>8} {'seed_AUC':>10} {'±':>6}", flush=True)
for eps in EPS_LIST:
    r = results[("ETH", 15, eps)]
    keep = (1 - eps*4) * 100 if eps else 100  # 估算
    print(f"{eps:8.4f} {keep:7.1f}% {r['auc_te']:8.4f} {r['seed_te_mean']:10.4f} ±{r['seed_te_std']:.4f}", flush=True)

print(f"\n[{now()}] ✅ 全部完成", flush=True)
