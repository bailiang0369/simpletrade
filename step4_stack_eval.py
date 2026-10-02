"""Step 4: Load all pv → 3-way stacking → rolling quantile no-lookahead eval."""
import sys; sys.stdout.reconfigure(line_buffering=True)
import time, os, numpy as np
from sklearn.metrics import roc_auc_score
import config

t0 = time.time()

print("Loading saved predictions...", flush=True)
tree = np.load(f'{config.PROJECT_DIR}/results/eth_tree.npz')
eth_nn = np.load(f'{config.PROJECT_DIR}/results/eth_nn.npz')
btc_nn = np.load(f'{config.PROJECT_DIR}/results/btc_nn.npz')

pv_lgb_te = tree['pv_lgb_te']
pv_cat_te = tree['pv_cat_te']
pv_tree_te = tree['pv_tree_te']
pv_tree_mv = tree['pv_tree_mv']
y_te = tree['y_te']; y_mv = tree['y_mv']
ts_te = tree['ts_te']

pv_eth_nn_te = eth_nn['pv_eth_nn_te']
pv_eth_nn_mv = eth_nn['pv_eth_nn_mv']

pv_btc_nn_te = btc_nn['pv_btc_nn_te']
pv_btc_nn_mv = btc_nn['pv_btc_nn_mv']

print("  All loaded.", flush=True)

# ============================================================
# Individual AUCs
# ============================================================
print(f"\n{'='*60}"); print("INDIVIDUAL AUCs"); print("="*60, flush=True)
for name, pv_mv, pv_te in [
    ('LGBM', pv_lgb_te, pv_lgb_te),  # lgb_te is the saved pv, but let's use correct ones
]:
    pass

# Actually let's print all
print(f"  {'Model':<20} {'MV AUC':>8} {'TE AUC':>8}", flush=True)
print(f"  {'-'*40}", flush=True)
for name, pv_mv, pv_te in [
    ('LGBM', tree['pv_lgb_mv'] if 'pv_lgb_mv' in tree else pv_tree_mv, pv_lgb_te),
    ('CatBoost', tree['pv_cat_mv'] if 'pv_cat_mv' in tree else pv_tree_mv, pv_cat_te),
    ('TREE-ENS', pv_tree_mv, pv_tree_te),
    ('ETH NN-ENS', pv_eth_nn_mv, pv_eth_nn_te),
    ('BTC NN-ENS', pv_btc_nn_mv, pv_btc_nn_te),
]:
    auc_mv = roc_auc_score(y_mv, pv_mv)
    auc_te = roc_auc_score(y_te, pv_te)
    print(f"  {name:<20} {auc_mv:>8.4f} {auc_te:>8.4f}", flush=True)

# ============================================================
# Rank transform
# ============================================================
def to_rank(pv):
    return np.argsort(np.argsort(pv)).astype(np.float64) / (len(pv) - 1)

r_tree_mv = to_rank(pv_tree_mv); r_tree_te = to_rank(pv_tree_te)
r_eth_nn_mv = to_rank(pv_eth_nn_mv); r_eth_nn_te = to_rank(pv_eth_nn_te)
r_btc_nn_mv = to_rank(pv_btc_nn_mv); r_btc_nn_te = to_rank(pv_btc_nn_te)

# ============================================================
# Correlation analysis
# ============================================================
print(f"\n{'='*60}"); print("CORRELATION ANALYSIS (MV)"); print("="*60, flush=True)
def corr(a, b):
    return np.corrcoef(a, b)[0, 1]
def tail_corr(a, b, q_low=99):
    m = np.argsort(-a)[:int(len(a) * (100 - q_low) / 100)]
    return np.corrcoef(a[m], b[m])[0, 1]

print(f"  {'Pair':<25} {'Overall':>8} {'Top1%':>8} {'Bot1%':>8}", flush=True)
print(f"  {'-'*52}", flush=True)
for a_name, a, b_name, b in [
    ('Tree', r_tree_mv, 'ETH_NN', r_eth_nn_mv),
    ('Tree', r_tree_mv, 'BTC_NN', r_btc_nn_mv),
    ('ETH_NN', r_eth_nn_mv, 'BTC_NN', r_btc_nn_mv),
]:
    oc = corr(a, b)
    t1 = tail_corr(a, b, 99)
    b1 = tail_corr(-a, -b, 99)   # bottom 1%
    print(f"  {a_name+' vs '+b_name:<25} {oc:>8.4f} {t1:>8.4f} {b1:>8.4f}", flush=True)

# ============================================================
# 3-way blend search
# ============================================================
print(f"\n{'='*60}"); print("3-WAY BLEND SEARCH"); print("="*60, flush=True)

# Coarse grid
best_auc = 0.0; best_w = (0.4, 0.4, 0.2)
for wt in np.arange(0.0, 1.01, 0.1):
    for we in np.arange(0.0, 1.01 - wt + 1e-9, 0.1):
        wb = 1.0 - wt - we
        if wb < -1e-6: continue
        pv = wt * r_tree_mv + we * r_eth_nn_mv + max(wb, 0) * r_btc_nn_mv
        auc = roc_auc_score(y_mv, pv)
        if auc > best_auc:
            best_auc = auc; best_w = (wt, we, max(wb, 0))

# Fine around best
wt_c, we_c, wb_c = best_w
for wt in np.arange(max(0, wt_c - 0.12), min(1, wt_c + 0.121), 0.03):
    for we in np.arange(max(0, we_c - 0.12), min(1 - wt + 0.01, we_c + 0.121), 0.03):
        wb = 1.0 - wt - we
        if wb < -1e-6: continue
        pv = wt * r_tree_mv + we * r_eth_nn_mv + max(wb, 0) * r_btc_nn_mv
        auc = roc_auc_score(y_mv, pv)
        if auc > best_auc:
            best_auc = auc; best_w = (round(wt,2), round(we,2), round(max(wb,0),2))

wt, we, wb = best_w
print(f"  Coarse best: Tree={best_w[0]:.2f} ETH_NN={best_w[1]:.2f} BTC_NN={best_w[2]:.2f}  MV AUC={best_auc:.4f}", flush=True)

pv_stack_te = wt * r_tree_te + we * r_eth_nn_te + wb * r_btc_nn_te
pv_stack_mv = wt * r_tree_mv + we * r_eth_nn_mv + wb * r_btc_nn_mv
print(f"  ★ STACK: MV AUC={roc_auc_score(y_mv, pv_stack_mv):.4f} TE AUC={roc_auc_score(y_te, pv_stack_te):.4f}", flush=True)

# Also: 2-way Tree+ETH_NN blend (compare with 3-way)
best2_auc = 0.0; best2_w = 0.5
for w in np.arange(0.0, 1.01, 0.02):
    pv = w * r_tree_mv + (1 - w) * r_eth_nn_mv
    auc = roc_auc_score(y_mv, pv)
    if auc > best2_auc:
        best2_auc = auc; best2_w = w
pv_2way_te = best2_w * r_tree_te + (1 - best2_w) * r_eth_nn_te
print(f"  2-way (Tree+ETH_NN): wT={best2_w:.2f} wE={1-best2_w:.2f}  MV AUC={best2_auc:.4f}  TE AUC={roc_auc_score(y_te, pv_2way_te):.4f}", flush=True)

# ============================================================
# ROLLING QUANTILE NO-LOOKAHEAD EVAL
# ============================================================
print(f"\n{'='*60}"); print("ROLLING QUANTILE EVAL (30-day window, no lookahead)"); print("="*60, flush=True)

DAYS = (ts_te[-1] - ts_te[0]) / 86400.0
q_list = [97, 98, 98.5, 99, 99.2, 99.5, 99.8]

def rolling_eval(name, pv, y, ts, days, q_list):
    ts_arr = np.array(ts); pv_arr = np.array(pv); y_arr = np.array(y)
    day_sec = 86400
    day_start = ts_arr.min()
    all_ts = np.arange(day_start, ts_arr.max() + day_sec, day_sec)
    n_days = len(all_ts) - 1
    window = 30

    auc = roc_auc_score(y, pv)
    print(f"\n  [{name}] AUC={auc:.4f}", flush=True)
    print(f"    {'q':>6} {'acc':>8} {'tpd':>6} {'n':>7}  flag", flush=True)
    print(f"    {'-'*42}", flush=True)
    for q in q_list:
        trades = []
        for d in range(window, n_days):
            day_lo = all_ts[d]; day_hi = all_ts[d + 1]
            hist_lo = all_ts[d - window]; hist_hi = day_lo
            hist_mask = (ts_arr >= hist_lo) & (ts_arr < hist_hi)
            hist_pv = pv_arr[hist_mask]
            if len(hist_pv) < 100: continue
            today_mask = (ts_arr >= day_lo) & (ts_arr < day_hi)
            thr = np.percentile(hist_pv, q)
            pick = pv_arr[today_mask] >= thr
            if pick.sum() > 0:
                trades.extend(y_arr[today_mask][pick].tolist())
        if len(trades) > 0:
            acc = np.mean(trades) * 100; tpd = len(trades) / days
            flag = "★★★" if (tpd >= 14 and acc >= 65) else ("★★" if (tpd >= 14 and acc >= 60) else ("★" if (tpd >= 14 and acc >= 55) else ""))
            print(f"    {q:>6.1f} {acc:>7.1f}% {tpd:>6.1f} {len(trades):>7}  {flag}", flush=True)

for name, pv in [('LGBM', pv_lgb_te), ('CatBoost', pv_cat_te),
                 ('TREE-ENS', pv_tree_te), ('ETH NN-ENS', pv_eth_nn_te),
                 ('BTC NN-ENS', pv_btc_nn_te),
                 ('2-way Stack (T+E)', pv_2way_te),
                 ('★ 3-way Stack', pv_stack_te)]:
    rolling_eval(name, pv, y_te, ts_te, DAYS, q_list)

# ============================================================
# SUMMARY
# ============================================================
print(f"\n{'='*60}"); print("FINAL SUMMARY"); print("="*60, flush=True)
print(f"  3-way Stack weights: Tree={wt:.2f} + ETH_NN={we:.2f} + BTC_NN={wb:.2f}", flush=True)
print(f"  Test period: {DAYS:.0f} days", flush=True)
print(f"  Target: top-1% acc >= 65% AND tpd >= 14", flush=True)
print(f"{'='*60}", flush=True)

# Save final
os.makedirs(f'{config.PROJECT_DIR}/results', exist_ok=True)
np.savez(f'{config.PROJECT_DIR}/results/final_stack.npz',
         pv_stack_te=pv_stack_te, pv_stack_mv=pv_stack_mv,
         pv_2way_te=pv_2way_te,
         weights_3way=np.array([wt, we, wb]),
         weights_2way=np.array([best2_w, 1-best2_w]),
         y_te=y_te, y_mv=y_mv, ts_te=ts_te)
print(f"\nSaved final_stack.npz [{time.time()-t0:.0f}s]", flush=True)
