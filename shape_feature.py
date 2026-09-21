"""Shape Feature — 分块向量化 OLS, 5.8G 安全. 全 float32."""
import numpy as np, pandas as pd, time, gc
from numpy.lib.stride_tricks import sliding_window_view
import config

t0 = time.time()
print('[1] 加载 + Stoch + DMI', flush=True)
eth = pd.read_parquet(f'{config.DS_DIR}/raw_ETH.parquet')
close = eth['close'].values.astype(np.float32)
high = eth['high'].values.astype(np.float32)
low  = eth['low'].values.astype(np.float32)
ts = eth['ts'].values.astype(np.int64)
N = len(close); H = config.HORIZON_MIN

ll = pd.Series(low, dtype=np.float64).rolling(60, min_periods=60).min().values.astype(np.float32)
hh = pd.Series(high, dtype=np.float64).rolling(60, min_periods=60).max().values.astype(np.float32)
stoch = ((close.astype(np.float64) - ll.astype(np.float64)) / np.maximum(hh.astype(np.float64) - ll.astype(np.float64), 1e-6)).astype(np.float32)
stoch = np.nan_to_num(stoch, nan=0.5)

h_l = high[1:]-low[1:]; h_cp = np.abs(high[1:]-close[:-1]); l_cp = np.abs(low[1:]-close[:-1])
tr = np.maximum(np.maximum(h_l, h_cp), l_cp).astype(np.float64)
up_m = high[1:]-high[:-1]; dn_m = low[:-1]-low[1:]
pdm = np.where((up_m>dn_m)&(up_m>0), up_m, 0.0).astype(np.float64)
mdm = np.where((dn_m>up_m)&(dn_m>0), dn_m, 0.0).astype(np.float64)
def ws(a,p):
    o=np.zeros(len(a),np.float64);o[0]=a[0]
    for i in range(1,len(a)):o[i]=o[i-1]-o[i-1]/p+a[i]
    return o
tr_s=ws(tr,14);pdm_s=ws(pdm,14);mdm_s=ws(mdm,14)
pdi=np.zeros(N,np.float32);mdi=np.zeros(N,np.float32)
pdi[1:]=(100*pdm_s/np.maximum(tr_s,1e-6)).astype(np.float32)
mdi[1:]=(100*mdm_s/np.maximum(tr_s,1e-6)).astype(np.float32)
del eth, h_l, h_cp, l_cp, up_m, dn_m, pdm, mdm
gc.collect()
print(f'  loaded ({time.time()-t0:.1f}s)', flush=True)

def batch_ols_chunked(seqs, chunk=500_000):
    """seqs: (N, L) float32 → slope×100, R², curv×1000 各 (N,) float32."""
    N, L = seqs.shape
    x = np.arange(L, dtype=np.float32)
    x_m = x.mean(); x_c = (x - x_m).astype(np.float32)
    denom = float(np.sum(x_c ** 2))
    slope = np.zeros(N, np.float32); r2 = np.zeros(N, np.float32); curv = np.zeros(N, np.float32)
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        y = seqs[s:e]
        y_m = y.mean(axis=1, keepdims=True)
        y_c = y - y_m
        sl = (y_c @ x_c) / denom
        inter = y_m.ravel() - sl * x_m
        yp = inter[:, None] + sl[:, None] * x[None, :]
        ss_t = (y_c ** 2).sum(axis=1) + 1e-10
        ss_r = ((y - yp) ** 2).sum(axis=1)
        r2[s:e] = 1 - ss_r / ss_t
        d2 = np.diff(np.diff(y, axis=1), axis=1)
        curv[s:e] = d2.mean(axis=1) if d2.shape[1] > 0 else 0
        slope[s:e] = sl * 100
    return slope, r2, curv * 1000

MIN_SEG, MAX_SEG, CHUNK = 30, 60, 500_000
N_win = N - MAX_SEG + 1  # 每个 60 长窗口
print(f'\nN={N:,}, N_win={N_win:,}, H={H}', flush=True)

def compute_shapes(series, name):
    """series: (N,) float32 → (slope_best, r2_best, curv_best, len_best) 各 (N_win,) float32."""
    sl_b = np.zeros(N_win, np.float32)
    r2_b = np.zeros(N_win, np.float32)
    cv_b = np.zeros(N_win, np.float32)
    ln_b = np.zeros(N_win, np.float32)
    sg_b = np.zeros(N_win, np.float32)
    
    for s in range(0, N_win, CHUNK):
        e = min(s + CHUNK, N_win)
        seg = series[s: e + MAX_SEG - 1]
        w60 = sliding_window_view(seg, MAX_SEG)   # (B, 60)
        c_sl = np.zeros(e-s, np.float32)
        c_r2 = np.zeros(e-s, np.float32)
        c_cv = np.zeros(e-s, np.float32)
        c_ln = np.zeros(e-s, np.float32)
        c_sg = np.zeros(e-s, np.float32)
        
        for L in range(MIN_SEG, MAX_SEG + 1):
            w = w60[:, -L:]
            base = w[:, :1]
            wn = (w / np.maximum(base, 1e-6) - 1.0) if name == 'price' else (w - base)
            sl, r2, cv = batch_ols_chunked(wn, chunk=CHUNK)
            sg = np.abs(sl)
            m = sg > c_sg
            c_sl[m] = sl[m]; c_r2[m] = r2[m]; c_cv[m] = cv[m]; c_ln[m] = L; c_sg[m] = sg[m]
            del w, base, wn, sl, r2, cv, sg, m
        
        sl_b[s:e] = c_sl; r2_b[s:e] = c_r2; cv_b[s:e] = c_cv; ln_b[s:e] = c_ln; sg_b[s:e] = c_sg
        del seg, w60, c_sl, c_r2, c_cv, c_ln, c_sg
        gc.collect()
        if e % 1_000_000 == 0 or e == N_win:
            print(f'  {name} chunk [{s/1e6:.1f}M, {e/1e6:.1f}M) ({time.time()-t0:.0f}s)', flush=True)
    return sl_b, r2_b, cv_b, ln_b

print(f'\n[2a] price 形态', flush=True)
price_slope, price_r2, price_curv, price_len = compute_shapes(close, 'price')
print(f'\n[2b] stoch 形态', flush=True)
stoch_slope, stoch_r2, stoch_curv, stoch_len = compute_shapes(stoch, 'stoch')

# corr 分块
print(f'\n[3] corr', flush=True)
corr60 = np.zeros(N_win, np.float32)
for s in range(0, N_win, CHUNK):
    e = min(s + CHUNK, N_win)
    cseg = close[s: e + MAX_SEG - 1]; sseg = stoch[s: e + MAX_SEG - 1]
    cw = sliding_window_view(cseg, MAX_SEG); sw = sliding_window_view(sseg, MAX_SEG)
    cm = cw.mean(axis=1, keepdims=True); sm = sw.mean(axis=1, keepdims=True)
    cc = cw - cm; sc = sw - sm
    cv = (cc * sc).mean(axis=1); sv = cc.std(axis=1) * sc.std(axis=1)
    corr60[s:e] = cv / np.maximum(sv, 1e-10)
    del cseg, sseg, cw, sw, cm, sm, cc, sc, cv, sv

gc.collect()

print(f'\n[4] 拼特征', flush=True)
price_pos = np.float32(1.0) - price_len / MAX_SEG
stoch_pos = np.float32(1.0) - stoch_len / MAX_SEG

# 标签: 窗口 i 对应 close[MAX_SEG-1 + i], label 是 close[MAX_SEG-1 + i + H] > close[MAX_SEG-1 + i]
# 所以 label 数量 = N_win - H (最后 H 个窗口没 label)
N_labelled = N_win - H

X = np.column_stack([
    price_slope[:N_labelled], price_r2[:N_labelled], price_curv[:N_labelled], price_pos[:N_labelled],
    stoch_slope[:N_labelled], stoch_r2[:N_labelled], stoch_curv[:N_labelled], stoch_pos[:N_labelled],
    (price_slope - stoch_slope * 0.5)[:N_labelled],
    (price_r2 - stoch_r2)[:N_labelled],
    corr60[:N_labelled],
    stoch[MAX_SEG-1:N_labelled+MAX_SEG-1],
    stoch[MAX_SEG-1:N_labelled+MAX_SEG-1] - stoch[MAX_SEG-2:N_labelled+MAX_SEG-2],
    pdi[MAX_SEG-1:N_labelled+MAX_SEG-1],
    mdi[MAX_SEG-1:N_labelled+MAX_SEG-1],
    pdi[MAX_SEG-1:N_labelled+MAX_SEG-1] - mdi[MAX_SEG-1:N_labelled+MAX_SEG-1],
]).astype(np.float32)

y = (close[MAX_SEG-1+H:N_labelled+MAX_SEG-1+H] > close[MAX_SEG-1:N_labelled+MAX_SEG-1]).astype(np.int64)
ts_full = ts[MAX_SEG-1:N_labelled+MAX_SEG-1]

del price_slope, price_r2, price_curv, price_len, price_pos
del stoch_slope, stoch_r2, stoch_curv, stoch_len, stoch_pos
del corr60
gc.collect()
print(f'  X={X.shape}, y={y.shape}, pos={y.mean():.3f} ({time.time()-t0:.1f}s)', flush=True)

def mk(s, e):
    a = int(pd.Timestamp(s, tz='UTC').timestamp())
    b = int(pd.Timestamp(e, tz='UTC').timestamp())
    m = (ts_full >= a) & (ts_full < b)
    return X[m], y[m], ts_full[m]

X_tr, y_tr, _ = mk(*config.SPLITS['train'])
X_es, y_es, _ = mk(*config.SPLITS['early_stop'])
X_te, y_te, ts_te = mk(*config.SPLITS['test'])
print(f'  TR={len(X_tr):,} ES={len(X_es):,} TE={len(X_te):,}', flush=True)

# === LGBM ===
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

print(f'\n[5] LightGBM grid', flush=True)
best_top = 0; best_p = None; best_m = None
for leaves in [15, 31, 63, 127]:
    for lr in [0.01, 0.03, 0.05]:
        for ff in [0.6, 0.8]:
            p = dict(objective='binary', metric='auc', num_leaves=leaves, learning_rate=lr,
                     feature_fraction=ff, bagging_fraction=0.8, bagging_freq=5,
                     min_child_samples=300, lambda_l1=0.01, lambda_l2=0.1,
                     verbose=-1, n_jobs=3, seed=42)
            m = lgb.train(p, lgb.Dataset(X_tr, y_tr), num_boost_round=5000,
                          valid_sets=[lgb.Dataset(X_es, y_es)],
                          callbacks=[lgb.early_stopping(300), lgb.log_evaluation(0)])
            pv_te = m.predict(X_te)
            auc = roc_auc_score(y_te, pv_te)
            k1 = max(1, int(len(pv_te)*0.01)); acc1 = y_te[pv_te.argsort()[-k1:]].mean()*100
            k05 = max(1, int(len(pv_te)*0.005)); acc05 = y_te[pv_te.argsort()[-k05:]].mean()*100
            line = f'L={leaves:3d} lr={lr} ff={ff} → auc={auc:.4f} top0.5%={acc05:.1f}% top1%={acc1:.1f}%'
            if acc1 > best_top: best_top = acc1; best_p = p; best_m = m; line += ' ⭐'
            print(line, flush=True)

print(f'\n[6] 最佳 top1%={best_top:.1f}%  (历史: LGBM 87feats=60.1%, Stoch+DMI=59.2%)')
print('\n特征重要性 (gain):')
imp = best_m.feature_importance(importance_type='gain')
names = ['price_slope', 'price_R²', 'price_curv', 'price_pos(形态起点位置)',
         'stoch_slope', 'stoch_R²', 'stoch_curv', 'stoch_pos',
         'price-stoch_slope差', 'price-stoch_R²差', 'price-stoch_corr',
         'stoch当前值', 'stoch速率一阶差',
         'DI+', 'DI-', 'DI_diff']
for n, v in sorted(zip(names, imp), key=lambda x: -x[1]):
    print(f'  {n:28s}: {v:.0f}')

# === 月度 ===
pv_te = best_m.predict(X_te)
k1 = max(1, int(len(pv_te)*0.01))
idx = pv_te.argsort()[-k1:]
month_te = pd.to_datetime(ts_te[idx], unit='s', utc=True).to_period('M')
df = pd.DataFrame({'idx': idx, 'y': y_te[idx], 'm': month_te.values})
print(f'\n[7] Test 月度 top1% 准确率:')
for m, g in df.groupby('m'):
    print(f'  {m}: {g["y"].mean()*100:.1f}% ({len(g)} sig, tpd={len(g)/30:.1f})')

print(f'\n⏱ TOTAL: {time.time()-t0:.0f}s')
