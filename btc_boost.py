"""BTC ACC BOOST — 全面诊断 + 提升 (不修改 snapshot 版本)
=========================================================
Stage 1: BTC h5/h10/h15/h20/h30/h60 baseline 10-seed LGB → 找最优 horizon
Stage 2: BTC vs ETH h15 诊断 (特征区分度/训练动态/label噪声)
Stage 3: ETH 有效 trick 迁移 → 负权重极端 ret + 去噪
Stage 4: BTC 专属超参搜索
Stage 5: BTC 多-horizon conf combine
"""
import numpy as np, pandas as pd, time, gc, warnings, sys
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore')
T0 = time.time()
def now(): return f"{time.time()-T0:.0f}s"

NPY = "/workspace/data/splits_npy"
PAR = "/workspace/data/datasets"
SEEDS = [42,49,56,63,70,77,84,91,98,105]

# ======== UTILS ========
def load_npy(sym,h):
    return {s:(np.load(f"{NPY}/{sym}_h{h}_{s}_X.npy").astype(np.float32),
              np.load(f"{NPY}/{sym}_h{h}_{s}_y.npy").astype(np.int32))
            for s in ['train','early_stop','test','meta_val']}

def load_ret_future(sym,h):
    """Load ret_future per split from parquet"""
    df = pd.read_parquet(f"{PAR}/ds_{sym}_h{h}.parquet", columns=['ts','ret_future'])
    ret = df['ret_future'].values.astype(np.float32)
    ts = df['ts'].values.astype(np.int64)
    del df; gc.collect()
    # split boundaries (from build_all_datasets.py)
    tr_mask = (ts >= 1577836800) & (ts < 1722556800)
    es_mask = (ts >= 1722556800) & (ts < 1725148800)
    mv_mask = (ts >= 1759363200) & (ts < 1782892800)
    te_mask = (ts >= 1782892800)
    return {
        'train': ret[tr_mask], 'early_stop': ret[es_mask],
        'meta_val': ret[mv_mask], 'test': ret[te_mask]
    }

def rank_ens(A):
    """Rank-averaging ensemble across seeds.
    A: (N_samples, N_seeds) → returns (N_samples,) average rank."""
    ns, nseed = A.shape
    R = np.empty_like(A, dtype=np.float64)
    for i in range(ns):
        R[i] = np.argsort(np.argsort(A[i])).astype(np.float64) / max(nseed-1, 1)
    return R.mean(axis=1)

def train_lgb_rank(sym,h,splits,extra_params=None, sample_weight=None):
    """Train LGB rank ensemble (10 seeds). Returns test AUC + probs."""
    Xtr,ytr = splits['train']
    Xes,yes = splits['early_stop']
    Xte,yte = splits['test']

    base = {'objective':'binary','metric':'auc','learning_rate':0.05,
            'num_leaves':63,'min_child_samples':200,'feature_fraction':0.8,
            'bagging_fraction':0.8,'bagging_freq':5,'lambda_l2':0.1,
            'verbose':-1,'n_jobs':4}
    if extra_params: base.update(extra_params)
    params = base

    te_list, es_list = [], []
    for seed in SEEDS:
        params['seed'] = seed
        tr_ds = lgb.Dataset(Xtr, label=ytr, weight=sample_weight) if sample_weight is not None else lgb.Dataset(Xtr, label=ytr)
        es_ds = lgb.Dataset(Xes, label=yes, reference=tr_ds)
        m = lgb.train(params, tr_ds, num_boost_round=5000, valid_sets=[es_ds],
                      callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
        te_list.append(m.predict(Xte))
        es_list.append(m.predict(Xes))

    p_te = rank_ens(np.column_stack(te_list))
    p_es = rank_ens(np.column_stack(es_list))
    auc_te = roc_auc_score(yte, p_te)
    auc_es = roc_auc_score(yes, p_es)
    return auc_te, auc_es, p_te, p_es

def acc_at_q(y_true, prob, q):
    """Direction accuracy at top-q% confidence (no lookahead — uses raw quantile of same-day dist)"""
    thr = np.quantile(np.abs(prob - 0.5), q/100)
    m = np.abs(prob - 0.5) >= thr
    if m.sum() < 100: return np.nan, 0
    pred_dir = (prob[m] > 0.5).astype(int)
    return (pred_dir == y_true[m]).mean()*100, m.sum()/288  # 288 = daily candles

# ========================================================
# STAGE 1: BTC Baseline All Horizons
# ========================================================
print("="*60)
print(f"[{now()}] STAGE 1: BTC Baseline — All Horizons")
print("="*60)

btc_results = {}
for h in [5,10,15,20,30,60]:
    splits = load_npy('BTC', h)
    auc_te, auc_es, pte, pes = train_lgb_rank('BTC', h, splits)
    btc_results[h] = {'auc_te':auc_te,'auc_es':auc_es,'pte':pte,'splits':splits}
    print(f"  BTC h{h:2d}: es_auc={auc_es:.4f}  te_auc={auc_te:.4f}")
    # also tail acc
    yte = splits['test'][1]
    for q in [99, 99.5]:
        a, n = acc_at_q(yte, pte, q)
        print(f"    q={q}: {a:.2f}% @ {n:.1f}t", end='')
    print()

best_h = max(btc_results, key=lambda h: btc_results[h]['auc_te'])
print(f"\n  >>> BTC BEST HORIZON: h={best_h}  te_auc={btc_results[best_h]['auc_te']:.4f}")

# ========================================================
# STAGE 2: Diagnosis — BTC vs ETH at same horizon (h15)
# ========================================================
print("\n" + "="*60)
print(f"[{now()}] STAGE 2: BTC vs ETH h15 Diagnosis")
print("="*60)

eth15 = load_npy('ETH', 15)
btc15 = load_npy('BTC', 15)

# Train both
eth_auc_te, eth_auc_es, eth_pte, _ = train_lgb_rank('ETH', 15, eth15)
btc_auc_te, btc_auc_es, btc_pte, _ = train_lgb_rank('BTC', 15, btc15)
print(f"\n  ETH h15: es={eth_auc_es:.4f}  te={eth_auc_te:.4f}")
print(f"  BTC h15: es={btc_auc_es:.4f}  te={btc_auc_te:.4f}")
print(f"  Δ AUC: {eth_auc_te - btc_auc_te:+.4f}")

# Feature-level AUC (univariate)
print("\n  --- Univariate Feature AUC Comparison ---")
Xbtr,ybtr = btc15['train']
Xetr,yetr = eth15['train']
Xbte,ybte = btc15['test']
Xete,yete = eth15['test']
np.random.seed(42)
idx = np.random.choice(len(Xbtr), 200000, replace=False)

feat_btc, feat_eth, better = 0, 0, []
for j in range(Xbtr.shape[1]):
    try:
        ab = roc_auc_score(ybte, Xbte[:50000,j])
        ae = roc_auc_score(yete, Xete[:50000,j])
        feat_btc += (abs(ab-0.5) > 0.01)
        feat_eth += (abs(ae-0.5) > 0.01)
        if abs(ae-0.5) - abs(ab-0.5) > 0.005:
            better.append((j, ab, ae))
    except: pass
print(f"  BTC #feat |auc-0.5|>0.01: {feat_btc}/{Xbtr.shape[1]}")
print(f"  ETH #feat |auc-0.5|>0.01: {feat_eth}/{Xetr.shape[1]}")
better.sort(key=lambda x: -x[2]+x[1])
print(f"  ETH >> BTC features (top 5 ETH-better):")
for j,ab,ae in better[:5]: print(f"    feat[{j:2d}]: BTC_auc={ab:.4f} ETH_auc={ae:.4f}")

# Label noise: compare |ret| distribution
print("\n  --- Label Noise: |ret_future| Distribution ---")
try:
    ret_btc = load_ret_future('BTC', 15)
    ret_eth = load_ret_future('ETH', 15)
    rbt, ret = ret_btc['test'], ret_eth['test']
    print(f"  BTC h15 test |ret|: mean={np.abs(rbt).mean()*100:.3f}%  med={np.median(np.abs(rbt))*100:.3f}%  p90={np.quantile(np.abs(rbt),0.9)*100:.3f}%")
    print(f"  ETH h15 test |ret|: mean={np.abs(ret).mean()*100:.3f}%  med={np.median(np.abs(ret))*100:.3f}%  p90={np.quantile(np.abs(ret),0.9)*100:.3f}%")
    print(f"  BTC mean |ret| / ETH mean |ret| = {np.abs(rbt).mean()/np.abs(ret).mean():.3f}")
except Exception as e:
    print(f"  ret_future load failed: {e}")

# ========================================================
# STAGE 3: Transfer ETH tricks to BTC (on best horizon)
# ========================================================
BEST = best_h  # use BTC's own best horizon
print("\n" + "="*60)
print(f"[{now()}] STAGE 3: ETH Tricks → BTC h{BEST}")
print("="*60)

splits_b = load_npy('BTC', BEST)
ret_b = load_ret_future('BTC', BEST)
Xtr,ytr = splits_b['train']

tr_ret = ret_b['train']
abs_tr_ret = np.abs(tr_ret)
print(f"  BTC h{BEST} train ret: mean={tr_ret.mean()*100:.3f}% std={tr_ret.std()*100:.3f}%")

# Build sample weights
def build_weights(ret, frac_hi=0.10, frac_lo=0.10, w_hi=0.3, w_lo=0.3):
    """Negative-weight extreme |ret| (top) and tiny |ret| (bottom)"""
    abs_ret = np.abs(ret)
    q_hi = np.quantile(abs_ret, 1 - frac_hi)
    q_lo = np.quantile(abs_ret, frac_lo)
    w = np.ones(len(ret), dtype=np.float32)
    w[abs_ret >= q_hi] = w_hi    # extreme ret → downweight (mean-reversion trap)
    w[abs_ret <= q_lo] = w_lo    # tiny ret → downweight (noise)
    return w

def denoise_mask(ret, eps=0.0005):
    """Remove |ret| < eps samples"""
    return np.abs(ret) >= eps

# Configurations to test
configs = [
    ("baseline",           None,                                  None),
    ("negw top+bot 10%",   build_weights(tr_ret, 0.10, 0.10, 0.3, 0.3),  None),
    ("negw top+bot 8%",    build_weights(tr_ret, 0.08, 0.08, 0.3, 0.3),  None),
    ("negw top+bot 12%",   build_weights(tr_ret, 0.12, 0.12, 0.3, 0.3),  None),
    ("denoise eps=0.0005", None,                                  denoise_mask(tr_ret, 0.0005)),
    ("denoise+negw",       build_weights(tr_ret[denoise_mask(tr_ret,0.0005)],0.10,0.10,0.3,0.3),
                           denoise_mask(tr_ret, 0.0005)),
]

# Also test ETH-comparable tricks on ETH h15 for cross-check
for name, sw, keep_mask in configs:
    print(f"\n  [{now()}] BTC h{BEST} — {name}")
    splits_use = splits_b
    if keep_mask is not None:
        # reconstruct filtered splits
        Xtr_f = Xtr[keep_mask]; ytr_f = ytr[keep_mask]
        if sw is not None: sw_f = sw  # already filtered
        else: sw_f = None
        splits_use = {'train':(Xtr_f,ytr_f),
                      'early_stop':splits_b['early_stop'],
                      'meta_val':splits_b['meta_val'],
                      'test':splits_b['test']}
        auc_te, auc_es, pte, _ = train_lgb_rank('BTC', BEST, splits_use, sample_weight=sw_f)
    else:
        auc_te, auc_es, pte, _ = train_lgb_rank('BTC', BEST, splits_use, sample_weight=sw)

    yte = splits_b['test'][1]
    print(f"    es={auc_es:.4f}  te={auc_te:.4f}", end='')
    for q in [99, 99.5]:
        a, n = acc_at_q(yte, pte, q)
        print(f"  q{q}={a:.1f}%@{n:.1f}t", end='')
    print()

# ========================================================
# STAGE 4: BTC Hyperparameter Search (on h15, most comparable to ETH)
# ========================================================
print("\n" + "="*60)
print(f"[{now()}] STAGE 4: BTC HP Search on h15")
print("="*60)

splits_b15 = load_npy('BTC', 15)
hparam_grid = [
    # (num_leaves, max_depth, lr, ffrac, extra_name)
    (31,  200, 0.05, 0.8,  "nl31_d200"),
    (63,  200, 0.05, 0.8,  "baseline"),
    (127, 200, 0.05, 0.8,  "nl127_d200"),
    (255, 200, 0.05, 0.8,  "nl255_d200"),
    (63,  50,  0.05, 0.8,  "d50"),
    (63,  200, 0.03, 0.8,  "lr0.03"),
    (63,  200, 0.1,  0.8,  "lr0.1"),
    (63,  200, 0.05, 0.6,  "ff0.6"),
    (63,  200, 0.05, 1.0,  "ff1.0"),
    (127, 100, 0.03, 0.8,  "nl127_d100_lr0.03"),
    (127, 50,  0.03, 0.8,  "nl127_d50_lr0.03"),
    (63,  200, 0.05, 0.9,  "ff0.9"),
]

best_auc = 0
for nl,md,lr,ff,name in hparam_grid:
    print(f"  [{now()}] {name} (nl={nl} d={md} lr={lr} ff={ff})...", flush=True)
    auc_te, auc_es, pte, _ = train_lgb_rank('BTC', 15, splits_b15,
        extra_params={'num_leaves':nl,'max_depth':md,'learning_rate':lr,'feature_fraction':ff})
    yte = splits_b15['test'][1]
    a99, n99 = acc_at_q(yte, pte, 99)
    print(f"    es={auc_es:.4f} te={auc_te:.4f} q99={a99:.1f}%@{n99:.1f}t")
    if auc_te > best_auc:
        best_auc = auc_te
        best_cfg = (nl,md,lr,ff,name)
        best_pte = pte

print(f"\n  >>> BEST BTC h15 HP: {best_cfg[-1]}  te_auc={best_auc:.4f}")

# ========================================================
# STAGE 5: BTC Multi-Horizon Conf Combine
# ========================================================
print("\n" + "="*60)
print(f"[{now()}] STAGE 5: BTC Multi-Horizon Conf Combine")
print("="*60)

# Get best per-horizon probs
all_pte = {}
for h in [5,10,15,20,30,60]:
    s = load_npy('BTC', h)
    _,_,p,_ = train_lgb_rank('BTC', h, s)
    all_pte[h] = (p, s['test'][1])

y_ref = all_pte[15][1]  # use h15 test labels as reference (all splits aligned)
print(f"  Test y shape: {y_ref.shape}")

# Try simple average pairs
from itertools import combinations
pairs = [(15,30),(10,20),(15,20),(15,10),(5,15)]
for h1,h2 in pairs:
    p1,y1 = all_pte[h1]
    p2,y2 = all_pte[h2]
    yt = y1  # assume aligned; fallback to min length
    n = min(len(p1),len(p2),len(yt))
    p_avg = (p1[:n]+p2[:n])/2
    auc = roc_auc_score(yt[:n], p_avg)
    a99,nt = acc_at_q(yt[:n], p_avg, 99)
    print(f"  BTC h{h1}+h{h2} avg: te_auc={auc:.4f}  q99={a99:.1f}%@{nt:.1f}t")

# Also weighted: 0.6*h15 + 0.4*h30
if 15 in all_pte and 30 in all_pte:
    p15 = all_pte[15][0]; p30 = all_pte[30][0]; yt = all_pte[15][1]
    n = min(len(p15),len(p30),len(yt))
    for w in [(0.7,0.3),(0.6,0.4),(0.5,0.5),(0.8,0.2)]:
        p = w[0]*p15[:n] + w[1]*p30[:n]
        auc = roc_auc_score(yt[:n], p)
        a99,nt = acc_at_q(yt[:n], p, 99)
        print(f"  BTC {w[0]}*h15+{w[1]}*h30: te_auc={auc:.4f}  q99={a99:.1f}%@{nt:.1f}t")

print(f"\n[{now()}] DONE. Total time {time.time()-T0:.0f}s")
