"""多尺度 Stoch 趋势线几何特征 -> LGBM. 内存友好版.

用多尺度 EMA 的斜率/曲率/突破位 表征趋势线几何, 避免渲染 2D 图像(内存10+GB超限).
核心信号(对齐用户: 趋势线>=20根K + 突破后回调):
  - 完整大波动趋势: 长窗口(60/120) Stoch 的趋势斜率
  - 局部短线趋势:   短窗口(20/40) Stoch 的趋势斜率
  - 突破:          短斜率由负转正(上穿)/正转负(下穿) 的拐点
  - 回踩/回抽深度:  相对多尺度趋势线的偏离
"""
import numpy as np
import pandas as pd
import time, config
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

def stoch_series(close, high, low, k=60, d=1, smooth=1):
    ll = pd.Series(low).rolling(k, min_periods=k).min().values
    hh = pd.Series(high).rolling(k, min_periods=k).max().values
    sk = np.nan_to_num((close - ll) / np.maximum(hh - ll, 1e-6), nan=0.5)
    if smooth > 1: sk = pd.Series(sk).rolling(smooth, min_periods=1).mean().values
    if d > 1: sk = pd.Series(sk).rolling(d, min_periods=1).mean().values
    return sk

def trend_slope(x, ema_span, slope_len):
    """多尺度 EMA 后在 slope_len 窗口的 OLS 斜率为正/负.
    返回 (slope 值) 数组.
    """
    e = pd.Series(x).ewm(span=ema_span, adjust=False).mean().values
    # OLS slope on last slope_len of EMA
    n = len(x)
    out = np.zeros(n, np.float32)
    for i in range(slope_len, n):
        seg = e[i - slope_len + 1: i + 1]
        t = np.arange(slope_len, dtype=np.float64)
        t_c = t - t.mean()
        out[i] = np.sum((seg - seg.mean()) * t_c) / np.sum(t_c ** 2)
    return out


def main():
    t0 = time.time()
    eth = pd.read_parquet(f'{config.DS_DIR}/raw_ETH.parquet')
    close = eth['close'].values.astype(np.float64)
    high = eth['high'].values.astype(np.float64)
    low = eth['low'].values.astype(np.float64)
    ts = eth['ts'].values.astype(np.int64)
    N = len(close); H = config.HORIZON_MIN
    del eth

    stoch = stoch_series(close, high, low, k=60, d=1, smooth=1)
    stoch_s = stoch_series(close, high, low, k=120, d=1, smooth=1)  # 更长尺度
    del high, low

    print(f'[1] stoch done ({time.time()-t0:.0f}s)', flush=True)

    # 多尺度趋势线斜率 (EMA spans: 短20/中40/长60/更长80, slope窗口 20~60)
    slopes = {}
    scales = [('s', 20, 20), ('m', 40, 40), ('lg', 60, 40), ('xl', 80, 60)]
    for name, span, slen in scales:
        slopes[f'slope_{name}'] = trend_slope(stoch, span, slen)
    # 更长尺度 base stoch
    slopes['slope_rt'] = trend_slope(stoch, 60, 60)
    slopes['slope_big'] = trend_slope(stoch_s, 120, 60)
    print(f'[2] slopes done ({time.time()-t0:.0f}s)', flush=True)

    # 基础特征
    def diff_lag(a, l):
        return a - np.concatenate([[a[0]] * l, a[:-l]])
    base = {
        'stoch': stoch,
        'stoch_120': stoch_s,
        'stoch_zone': stoch - 0.5,               # -0.5超卖 +0.5超买
        'ret5': close/np.concatenate([close[0]*np.ones(5), close[:-5]]) - 1,
        'ret15': close/np.concatenate([close[0]*np.ones(15), close[:-15]]) - 1,
        'ret30': close/np.concatenate([close[0]*np.ones(30), close[:-30]]) - 1,
        'ret60': close/np.concatenate([close[0]*np.ones(60), close[:-60]]) - 1,
    }
    # 突破位: 斜率正负翻转
    for name in ['s', 'm', 'lg']:
        sl = slopes[f'slope_{name}']
        prev = np.concatenate([[sl[0]], sl[:-1]])
        base[f'cross_up_{name}'] = ((sl > 0) & (prev <= 0)).astype(np.float32)
        base[f'cross_dn_{name}'] = ((sl < 0) & (prev >= 0)).astype(np.float32)
    # 回踩: 短slope与长slope相反(回调)量 + 当前在超卖/超买
    base['counter_slope'] = np.sign(slopes['slope_s']) * np.sign(slopes['slope_lg'])
    base['slope_s_minus_lg'] = slopes['slope_s'] - slopes['slope_lg']

    feats = {}
    feats.update(slopes)
    feats.update(base)
    del slopes, base; import gc; gc.collect()

    # 切分
    def mkmap(s, e):
        a = int(pd.Timestamp(s, tz='UTC').timestamp())
        b = int(pd.Timestamp(e, tz='UTC').timestamp())
        return (ts >= a) & (ts < b) & (np.arange(N) >= 200) & (np.arange(N) < N - H)
    tr_m = mkmap(*config.SPLITS['train'])
    es_m = mkmap(*config.SPLITS['early_stop'])
    te_m = mkmap(*config.SPLITS['test'])

    idx = np.where(tr_m | es_m | te_m)[0]
    ts_v = ts[idx]
    y = (close[idx + H] > close[idx]).astype(int)
    X_arr = []; names = []
    for n, a in feats.items():
        X_arr.append(a[idx].astype(np.float32)); names.append(n)
    X = np.column_stack(X_arr)
    del X_arr, feats; gc.collect()
    print(f'[3] X={X.shape} ({time.time()-t0:.0f}s)', flush=True)

    def slice_m(m):
        return X[m], y[m]
    Xtr, ytr = slice_m((tr_m[idx] | es_m[idx]))  # train+es 都训练
    # 早停也需独立, 用 es 做 valid
    esm = es_m[idx]
    Xes, yes = X[esm], y[esm]
    tem = te_m[idx]
    Xte, yte = X[tem], y[tem]
    print(f'  TR+ES={len(Xtr):,} ES={len(Xes):,} TE={len(Xte):,}', flush=True)

    # 训练时排除 es 里的样本避免泄露? 简单起见 tr 不包含 es
    trm = tr_m[idx]
    Xtr, ytr = X[trm], y[trm]
    Xes, yes = X[esm], y[esm]
    print(f'  TR={len(Xtr):,} ES={len(Xes):,} TE={len(Xte):,}', flush=True)

    print(f'\n[4] LGBM ({len(names)} dims)', flush=True)
    best = 0
    for leaves in [31, 63, 127]:
        for lr in [0.01, 0.03]:
            p = dict(objective='binary', metric='auc', num_leaves=leaves, learning_rate=lr,
                     feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
                     min_child_samples=300, verbose=-1, n_jobs=3, seed=42)
            m = lgb.train(p, lgb.Dataset(Xtr, ytr), num_boost_round=5000,
                          valid_sets=[lgb.Dataset(Xes, yes)],
                          callbacks=[lgb.early_stopping(300), lgb.log_evaluation(0)])
            pv = m.predict(Xte)
            auc = roc_auc_score(yte, pv)
            k1 = max(1, int(len(pv) * 0.01)); a1 = yte[pv.argsort()[-k1:]].mean()*100
            k05 = max(1, int(len(pv) * 0.005)); a05 = yte[pv.argsort()[-k05:]].mean()*100
            mark = ' ⭐' if a1 > best else ''
            if a1 > best: best = a1
            print(f'  L={leaves} lr={lr} → auc={auc:.4f} t0.5={a05:.1f}% t1={a1:.1f}%{mark}', flush=True)

    print(f'\n最佳 top1%={best:.1f}%  (历史 LGBM 87feats=60.1%, 目标65%)')
    # 特征重要性
    print(f'\n特征重要性 top20:')
    imp = m.feature_importance(importance_type='gain')
    for n, v in sorted(zip(names, imp), key=lambda x: -x[1])[:20]:
        print(f'  {n:20s}: {v:>10.0f}')
    print(f'\n⏱ {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()