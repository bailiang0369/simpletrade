"""把 Stoch 渲染成多尺度图像, CNN 识别趋势线几何结构.

设计思路 (对齐用户手动画趋势线):
- 用户画的是"完整的单向大波动 / 局部短线趋势线", 两者都要.
- 所以每个样本渲染成 3 个尺度的 Stoch 轨迹叠加图:
    原始 Stoch(短) / 长窗口 EMA(中) / 更长效(长) —— 一条图像里 3 条纵向轨迹.
- 纵轴 = Stoch [0,1] 映射到像素高度; 横轴 = 窗口内 K 线索引.
- 线条呈直线段, 卷积核天然能感知"斜率 / 趋势 / 突破拐点".
- 额外通道: 价格归一化轨迹(辅助), 突破点标记.

先用小规模验证(200k train)确认趋势线图像信息有效性.
"""
import os
import numpy as np
import pandas as pd
import time, config
from sklearn.metrics import roc_auc_score

SEG = 80          # 窗口长度(K 线)
HEIGHT = 56       # 图像高度(像素, Stoch 分辨力)
LINE_W = 2        # 线条半宽(像素)

def stoch_series(close, high, low, k=60, d=1, smooth=1):
    ll = pd.Series(low).rolling(k, min_periods=k).min().values
    hh = pd.Series(high).rolling(k, min_periods=k).max().values
    sk = np.nan_to_num((close - ll) / np.maximum(hh - ll, 1e-6), nan=0.5)
    if smooth > 1:
        sk = pd.Series(sk).rolling(smooth, min_periods=1).mean().values
    if d > 1:
        sk = pd.Series(sk).rolling(d, min_periods=1).mean().values
    return sk

def render_window(stoch_win, price_win, H, W):
    """把一段窗口渲染成 (4, H, W) 图像.
    stoch_win: (W,) in [0,1]; price_win: (W,) normalized.
    通道: 0=原始Stoch tan曲线, 1=EMA10 Stoch, 2=EMA30 Stoch, 3=price轨迹.
    用直线段+bresenham式加粗画线.
    """
    img = np.zeros((4, H, W), np.float32)
    # 映射: stoch [0,1] -> y 像素 [H-1 .. 0]
    def line_to_img(chan, values):
        xs = np.linspace(0, W - 1, W).astype(int)
        ys = np.clip(((1 - values) * (H - 1)).round().astype(int), 0, H - 1)
        for i in range(W - 1):
            x0, x1 = xs[i], xs[i + 1]
            y0, y1 = ys[i], ys[i + 1]
            n = max(abs(x1 - x0), abs(y1 - y0), 1)
            for t in np.linspace(0, 1, int(n) + 1):
                xx, yy = int(round(x0 + t * (x1 - x0))), int(round(y0 + t * (y1 - y0)))
                for dx in range(-LINE_W, LINE_W + 1):
                    for dy in range(-LINE_W, LINE_W + 1):
                        if 0 <= xx + dx < W and 0 <= yy + dy < H:
                            img[chan, yy + dy, xx + dx] = 1.0
        return img[chan]

    line_to_img(0, stoch_win)
    # EMA
    e10 = pd.Series(stoch_win).ewm(span=10, adjust=False).mean().values
    e30 = pd.Series(stoch_win).ewm(span=30, adjust=False).mean().values
    line_to_img(1, e10)
    line_to_img(2, e30)
    # price normalized to [0,1] within window
    pw = price_win
    pmax, pmin = pw.max(), pw.min()
    pn = (pw - pmin) / (pmax - pmin + 1e-9)
    line_to_img(3, pn)
    return img


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
    price_rolling = pd.Series(close).rolling(SEG, min_periods=1).mean().values
    price_norm = close / np.maximum(price_rolling, 1e-6)
    del high, low

    # 构建时间掩码
    def mkmap(s, e):
        a = int(pd.Timestamp(s, tz='UTC').timestamp())
        b = int(pd.Timestamp(e, tz='UTC').timestamp())
        return (ts >= a) & (ts < b) & (np.arange(N) >= SEG) & (np.arange(N) < N - H)

    tr_m = mkmap(*config.SPLITS['train'])
    es_m = mkmap(*config.SPLITS['early_stop'])
    te_m = mkmap(*config.SPLITS['test'])

    # 抽样子集: 训练 200k, val(es) 20k, test 全量评估
    rng = np.random.RandomState(config.SEED)
    tr_idx = np.where(tr_m)[0]
    if len(tr_idx) > 200_000:
        tr_idx = rng.choice(tr_idx, 200_000, replace=False)
    es_idx = np.where(es_m)[0]
    if len(es_idx) > 20_000:
        es_idx = es_idx[::max(1, len(es_idx) // 20000)]
    te_idx = np.where(te_m)[0]
    print(f'[data] TR={len(tr_idx):,} ES={len(es_idx):,} TE={len(te_idx):,} ({time.time()-t0:.0f}s)', flush=True)

    # 渲染图像(分块避免 OOM)
    def render_batch(idx_arr):
        X = np.zeros((len(idx_arr), 4, HEIGHT, SEG), np.float32)
        y = np.zeros(len(idx_arr), np.int64)
        for j, i in enumerate(idx_arr):
            X[j] = render_window(stoch[i - SEG + 1: i + 1], price_norm[i - SEG + 1: i + 1], HEIGHT, SEG)
            y[j] = 1 if close[i + H] > close[i] else 0
        return X, y

    # 保存为 npy 分批(内存控制)
    os.makedirs(f'{config.MODEL_DIR}', exist_ok=True)
    for name, idx_arr in [('tr', tr_idx), ('es', es_idx), ('te', te_idx)]:
        X, y = render_batch(idx_arr)
        np.savez_compressed(f'{config.MODEL_DIR}/trend_img_{name}.npz', X=X.astype(np.float16), y=y)
        print(f'  saved {name}: X{X.shape} ({time.time()-t0:.0f}s)', flush=True)
    print(f'\n⏱ render done {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()