#!/usr/bin/env python3
"""
纯序列 / 图模型 自动迭代优化器。

目标: ETH H=15, 无前视, top-1% accuracy >= 60% 首先达成, 检查泄露, 然后冲 65%。
- 只用序列/图模型 (LSTM / TCN / Transformer / GAT)
- 不与树模型做投票或堆叠
- 自动迭代: 没达标就换架构 / 加增强 / 加正则 / 加目标函数
- 全 CPU, 严格时间切分, 绝对无 lookahead
"""
import os, sys, time, gc, json, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
import datetime as dtm

torch.manual_seed(42); np.random.seed(42)
DEVICE = 'cpu'
print(f"torch={torch.__version__}, device={DEVICE}", flush=True)

import config

# ============ 0. 数据加载 ============
print(f"\n{'='*60}", flush=True)
print("Loading data...", flush=True)
t0 = time.time()

eth = pd.read_parquet(f'{config.DS_DIR}/raw_ETH.parquet').sort_values('ts').reset_index(drop=True)
btc = pd.read_parquet(f'{config.DS_DIR}/raw_BTC.parquet').sort_values('ts').reset_index(drop=True)
print(f"  ETH={eth.shape}, BTC={btc.shape}  ({time.time()-t0:.0f}s)", flush=True)

# BTC → ETH 的向后对齐
ei = pd.Index(eth['ts'].values)
br = np.clip(btc['ts'].values.searchsorted(ei.values, side='right') - 1, 0, len(btc)-1)

ts = eth['ts'].values.astype(np.int64)
C = eth['close'].values.astype(np.float64)
O = eth['open'].values.astype(np.float64)
H = eth['high'].values.astype(np.float64)
L = eth['low'].values.astype(np.float64)
BV = eth['buy_vol'].values.astype(np.float64)
SV = eth['sell_vol'].values.astype(np.float64)
FUND = eth['funding'].values.astype(np.float64)
BTC_C = btc['close'].values.astype(np.float64)[br]

del eth, btc; gc.collect()
N = len(C)
print(f"  N={N:,}", flush=True)

# ============ 1. 切分 mask ============
def ts_mask(s, e):
    a = int(dtm.datetime.strptime(s,'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())
    b = int(dtm.datetime.strptime(e,'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())
    return (ts>=a)&(ts<b)

TR_m = ts_mask(*config.SPLITS['train'])
ES_m = ts_mask(*config.SPLITS['early_stop'])
TE_m = ts_mask(*config.SPLITS['test'])
print(f"  train={TR_m.sum():,}  es={ES_m.sum():,}  test={TE_m.sum():,}", flush=True)

# ============ 2. 序列特征: 多尺度收益 + 主动买卖 + 波动率 + 资金费 + BTC cross + 形态 ============
print(f"\n{'='*60}", flush=True)
print("Building sequence features...", flush=True)

def rolling_zscore_pd(x, w=2880):
    s = pd.Series(x)
    mu = s.rolling(w, min_periods=w//4).mean().values
    sd = s.rolling(w, min_periods=w//4).std().values + 1e-8
    return ((x - mu) / sd).astype(np.float32)

# --- base channels (全量构建一次, 供所有 H / WINDOW 复用) ---
# 1min log returns
lr1 = np.zeros(N, dtype=np.float32)
lr1[1:] = np.log(np.maximum(C[1:], 1e-8) / np.maximum(C[:-1], 1e-8))
# funding clip
_tre = int(dtm.datetime.strptime(config.TRAIN_END,'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())
_trm = ts < _tre
f_lo, f_hi = np.percentile(FUND[_trm], 0.5), np.percentile(FUND[_trm], 99.5)
FUND_c = np.clip(FUND, f_lo, f_hi)
# cvd / volume
totv = BV + SV
totv_s = np.where(totv > 0, totv, 1.0)
cvd = (BV - SV) / totv_s
bratio = np.log(np.where(BV > 0, BV / np.maximum(SV, 1e-8), 1e-8))
# btc lr1
btc_lr1 = np.zeros(N, dtype=np.float32)
btc_lr1[1:] = np.log(np.maximum(BTC_C[1:], 1e-8) / np.maximum(BTC_C[:-1], 1e-8))
# ohlc range
rng = np.where(H > L, H - L, 1.0)
body = np.abs(C - O) / rng

# row-level z-score normalize (2-day rolling)
def z(x):
    return rolling_zscore_pd(x.astype(np.float64), w=2880)

# 多尺度收益 (1/5/15/30/60min)
lr = np.zeros((6, N), dtype=np.float32)
ws = [1, 5, 15, 30, 60, 120]
for i, w in enumerate(ws):
    lr[i, w:] = np.log(np.maximum(C[w:], 1e-8) / np.maximum(C[:-w], 1e-8))

# 多尺度波动率 (rolling std of lr1)
def roll_std(x, w):
    s = pd.Series(x.astype(np.float64))
    return s.rolling(w, min_periods=w//4).std().values.astype(np.float32)

vol = np.zeros((3, N), dtype=np.float32)
for i, w in enumerate([15, 60, 240]):
    vol[i] = roll_std(lr1, w)

# 主动买卖量 z-scored
tb_act = BV - np.roll(BV, 1)
ts_act = SV - np.roll(SV, 1)

# 22 channels total: 6 log-returns + 3 vol + cvd z + bratio z + totvol z + funding z + btc lr1 z + body z + 2*tb_act z + 2*ts_act z + btc_vol_60 z + btc_ret_15 z + 120min lr z (extra)
channels = np.stack([
    z(lr[0]), z(lr[1]), z(lr[2]), z(lr[3]), z(lr[4]), z(lr[5]),
    z(vol[0]), z(vol[1]), z(vol[2]),
    z(cvd), z(bratio), z(np.log(np.maximum(totv, 1.0))),
    z(FUND_c),
    z(btc_lr1),
    z(body),
    z(tb_act), z(ts_act),
    z(roll_std(btc_lr1, 60)),
    z(np.concatenate([[0], np.log(np.maximum(BTC_C[1:],1e-8)/np.maximum(BTC_C[:-1],1e-8))])),
    z(C - np.roll(C, 1)),
    z(np.maximum(H - L, 0)),
    np.sin(2 * np.pi * (ts % 86400) / 86400).astype(np.float32),
    np.cos(2 * np.pi * (ts % 86400) / 86400).astype(np.float32),
], axis=0).astype(np.float32)

print(f"  channels shape: {channels.shape}", flush=True)
del BV, SV, FUND, BTC_C, O, H, L, totv, totv_s, cvd, bratio, btc_lr1, tb_act, ts_act, body, rng; gc.collect()

# ============ 3. 窗口构建 ============
def build_windows(H, WINDOW, STRIDE, label_filter_eps=0.0):
    """Build sequence windows. label_filter_eps: 过滤 |ret|<ε 的样本 (训练时过滤, 测试时保留)."""
    rows = []
    tss_out = []
    labels = []
    rets = []
    N_possible = 0
    for start in range(WINDOW, N - H, STRIDE):
        end = start  # 预测锚点: close[end+H] vs close[end]
        N_possible += 1
        ret = C[end + H] / C[end] - 1
        rows.append(channels[:, start-WINDOW:start].copy())
        tss_out.append(ts[end])
        labels.append(1 if ret > 0 else 0)
        rets.append(ret)
    X = np.array(rows, dtype=np.float32)
    y = np.array(labels, dtype=np.int64)
    r = np.array(rets, dtype=np.float32)
    T = np.array(tss_out, dtype=np.int64)
    del rows, tss_out, labels, rets; gc.collect()
    return X, y, r, T, N_possible

H = 15
WINDOW = 96   # 96根 ≈ 1.6h
STRIDE = 32   # 3x 下采样减少训练量

print(f"\nBuilding windows H={H} WINDOW={WINDOW} STRIDE={STRIDE}...", flush=True)
t_build = time.time()
X_all, y_all, r_all, ts_all, _ = build_windows(H, WINDOW, STRIDE, label_filter_eps=0.0)
print(f"  X={X_all.shape}, pos_rate={y_all.mean():.3f}, built in {time.time()-t_build:.0f}s", flush=True)

# ============ 4. 切分 + label filter 训练时 ============
tr_m = (ts_all >= int(dtm.datetime.strptime(config.SPLITS['train'][0],'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())) & \
       (ts_all < int(dtm.datetime.strptime(config.SPLITS['train'][1],'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp()))
es_m = (ts_all >= int(dtm.datetime.strptime(config.SPLITS['early_stop'][0],'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())) & \
       (ts_all < int(dtm.datetime.strptime(config.SPLITS['early_stop'][1],'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp()))
te_m = (ts_all >= int(dtm.datetime.strptime(config.SPLITS['test'][0],'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp()))

# 训练时 label filter: 去掉 |ret| < ε 的微幅振荡 (噪声)
TR_FILTER = 0.0003
tr_filt = tr_m & (np.abs(r_all) > TR_FILTER)
print(f"  train keep: {tr_filt.sum():,} / {tr_m.sum():,}  (filter |ret|>{TR_FILTER*100:.2f}%)", flush=True)

X_tr = X_all[tr_filt].copy(); y_tr = y_all[tr_filt].copy(); r_tr = r_all[tr_filt].copy()
X_es = X_all[es_m]; y_es = y_all[es_m].copy()
X_te = X_all[te_m]; y_te = y_all[te_m].copy(); ts_te = ts_all[te_m].copy()
del X_all, y_all, r_all, ts_all; gc.collect()

# 训练集大 → 抽样
MAX_TR = 600_000
if len(X_tr) > MAX_TR:
    rng = np.random.default_rng(42)
    keep = rng.choice(len(X_tr), MAX_TR, replace=False)
    X_tr = X_tr[keep]; y_tr = y_tr[keep]; r_tr = r_tr[keep]
print(f"  final: TR={len(X_tr):,}  ES={len(X_es):,}  TE={len(X_te):,}", flush=True)

# ============ 5. 模型定义 (多架构候选) ============
class LSTMAttn(nn.Module):
    """LSTM + time step attention + last-state conv head."""
    def __init__(self, C_in, WINDOW, hidden=128, heads=4):
        super().__init__()
        self.lstm = nn.LSTM(C_in, hidden, num_layers=2, batch_first=True, bidirectional=True,
                            dropout=0.15)
        self.attn = nn.MultiheadAttention(hidden*2, heads, batch_first=True, dropout=0.1)
        self.norm = nn.LayerNorm(hidden*2)
        self.conv = nn.Sequential(
            nn.Conv1d(hidden*2, hidden, 3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden//2), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(hidden//2, 1)
        )
    def forward(self, x):
        x = x.transpose(1, 2)  # (B, C, T) → (B, T, C)
        out, _ = self.lstm(x)    # (B, T, 2H)
        attn_out, _ = self.attn(out, out, out)
        out = self.norm(out + attn_out)
        out = self.conv(out.transpose(1, 2)).flatten(1)  # (B, H)
        return self.head(out).squeeze(-1)

class TCN(nn.Module):
    """Temporal Convolutional Network with dilated causal convolutions."""
    def __init__(self, C_in, WINDOW, chs=[64, 128, 128, 64], kernel=3):
        super().__init__()
        def block(c_in, c_out, d):
            return nn.Sequential(
                nn.Conv1d(c_in, c_out, kernel, padding=(kernel-1)*d, dilation=d),
                nn.GELU(),
                nn.Conv1d(c_out, c_out, kernel, padding=(kernel-1)*d, dilation=d),
                nn.GELU(),
                nn.Dropout(0.15),
                nn.Conv1d(c_out, c_out, 1),  # residual adapter
            )
        layers = []
        d = 1
        prev = C_in
        for c in chs:
            b = block(prev, c, d)
            layers.append(nn.ModuleList([b, d]))
            prev = c; d *= 2
        self.blocks = nn.ModuleList([bb[0] for bb in layers])
        self.skips = nn.ModuleList([nn.Conv1d(C_in if i==0 else chs[i-1], c, 1) for i,c in enumerate(chs)])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(chs[-1], chs[-1]//2), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(chs[-1]//2, 1)
        )
    def forward(self, x):
        # x: (B, C, T)
        out = x
        for blk, skip in zip(self.blocks, self.skips):
            x_blk = blk(out)
            skip_out = skip(out)
            out = x_blk[:, :, :skip_out.shape[2]] + skip_out  # causal
            out = F.pad(out, (0, skip_out.shape[2] - out.shape[2]))
        pooled = self.pool(out).flatten(1)
        return self.head(pooled).squeeze(-1)

class TransformerSeq(nn.Module):
    """Positional encoding + Transformer encoder."""
    def __init__(self, C_in, WINDOW, d_model=128, nhead=4, nlayer=3):
        super().__init__()
        self.proj = nn.Linear(C_in, d_model)
        self.pos = nn.Parameter(torch.zeros(1, WINDOW, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                    dim_feedforward=d_model*4, dropout=0.15,
                                                    batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(encoder_layer, num_layers=nlayer)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model//2), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(d_model//2, 1)
        )
    def forward(self, x):
        x = x.transpose(1, 2)     # (B, T, C)
        x = self.proj(x) + self.pos
        out = self.enc(x)          # (B, T, d_model)
        pooled = out.mean(dim=1) + out[:, -1, :]
        return self.head(pooled).squeeze(-1)

class GRUCNN(nn.Module):
    """GRU + CNN hybrid (之前失败过，但这次 label filter + 更好正则 + 更大数据量再试一次)."""
    def __init__(self, C_in, WINDOW, hidden=96):
        super().__init__()
        self.gru = nn.GRU(C_in, hidden, num_layers=2, batch_first=True, bidirectional=True,
                          dropout=0.15)
        self.conv = nn.Sequential(
            nn.Conv1d(hidden*2, hidden, 5, padding=2), nn.GELU(),
            nn.Conv1d(hidden, hidden, 3, padding=1), nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden//2), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(hidden//2, 1)
        )
    def forward(self, x):
        x = x.transpose(1, 2)   # (B, T, C)
        out, _ = self.gru(x)     # (B, T, 2H)
        out = self.conv(out.transpose(1, 2))
        pooled = self.pool(out).flatten(1)
        return self.head(pooled).squeeze(-1)

def make_model(name, C_in, WINDOW):
    if name == 'lstm_attn':
        return LSTMAttn(C_in, WINDOW, hidden=128, heads=4)
    elif name == 'lstm_attn_big':
        return LSTMAttn(C_in, WINDOW, hidden=192, heads=4)
    elif name == 'lstm_attn_small':
        return LSTMAttn(C_in, WINDOW, hidden=96, heads=4)
    elif name == 'tcn':
        return TCN(C_in, WINDOW, chs=[64, 128, 128, 64])
    elif name == 'tcn_deep':
        return TCN(C_in, WINDOW, chs=[64, 128, 256, 256, 128])
    elif name == 'transformer':
        return TransformerSeq(C_in, WINDOW, d_model=128, nhead=4, nlayer=3)
    elif name == 'transformer_small':
        return TransformerSeq(C_in, WINDOW, d_model=64, nhead=4, nlayer=2)
    elif name == 'gru_cnn':
        return GRUCNN(C_in, WINDOW, hidden=96)
    elif name == 'gru_cnn_big':
        return GRUCNN(C_in, WINDOW, hidden=144)
    else:
        raise ValueError(name)

# ============ 6. 训练 + 评估 ============
def evaluate(model, X, y, batch=2048):
    model.eval()
    pv, yv = [], []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(X[i:i+batch]).to(DEVICE)
            pv.append(torch.sigmoid(model(xb)).cpu().numpy())
            yv.append(y[i:i+batch])
    return np.concatenate(pv), np.concatenate(yv)

def train_one(name, X_tr, y_tr, X_es, y_es, X_te, y_te, ts_te, r_tr,
              C_in, WINDOW,
              epochs=20, lr=3e-3, wd=1e-4, batch=256, pat=6,
              use_weighted=True, use_ret_weight=True, label_smoothing=0.0):
    """训练一个模型并返回完整评估。"""
    t0 = time.time()
    m = make_model(name, C_in, WINDOW).to(DEVICE)
    n_params = sum(p.numel() for p in m.parameters())
    pos = y_tr.mean()
    pos_w = (1 - pos) / pos   # 正例权重
    neg_w = pos / (1 - pos)

    # 样本权重
    if use_ret_weight:
        r_w = np.clip(np.abs(r_tr) * 200, 0.2, 5.0).astype(np.float32)  # |ret| × scale, clip
    else:
        r_w = np.ones(len(y_tr), dtype=np.float32)

    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_auc = 0; best_st = None; no_imp = 0
    best_pv_te = None; best_pv_es = None

    for ep in range(epochs):
        m.train(); idx = np.random.permutation(len(X_tr)); tl = 0; n_b = 0
        for i in range(0, len(idx), batch):
            b = idx[i:i+batch]
            xb = torch.from_numpy(X_tr[b]).to(DEVICE)
            yb = torch.from_numpy(y_tr[b]).float().to(DEVICE)
            wb = torch.from_numpy(r_w[b]).float().to(DEVICE)
            if use_weighted:
                pw = torch.where(yb > 0.5, pos_w, neg_w).float().to(DEVICE)
                wb = wb * pw
            logits = m(xb)
            # BCE with weights
            loss = F.binary_cross_entropy_with_logits(logits, yb, weight=wb)
            # label smoothing
            if label_smoothing > 0:
                y_smooth = yb * (1 - label_smoothing) + 0.5 * label_smoothing
                loss = F.binary_cross_entropy_with_logits(logits, y_smooth, weight=wb)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            tl += loss.item(); n_b += 1
        sch.step()

        pv_es, _ = evaluate(m, X_es, y_es)
        auc_es = roc_auc_score(y_es, pv_es)
        ep_t = time.time() - t0
        pv_te, _ = evaluate(m, X_te, y_te)
        auc_te = roc_auc_score(y_te, pv_te)
        print(f"  ep{ep+1:2d} loss={tl/n_b:.4f} es_auc={auc_es:.4f} te_auc={auc_te:.4f} [{ep_t:.0f}s]", flush=True)

        if auc_es > best_auc + 1e-5:
            best_auc = auc_es
            best_st = {k: v.detach().clone() for k, v in m.state_dict().items()}
            best_pv_te = pv_te.copy()
            best_pv_es = pv_es.copy()
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= pat:
                print(f"  early stop @ ep{ep+1}", flush=True)
                break

    if best_st:
        m.load_state_dict(best_st)
        best_pv_te, _ = evaluate(m, X_te, y_te)
        best_pv_es, _ = evaluate(m, X_es, y_es)

    return {
        'model': m, 'best_es_auc': best_auc,
        'pv_te': best_pv_te, 'pv_es': best_pv_es,
        'n_params': n_params, 'time_s': time.time()-t0,
    }

# ============ 7. 完整评估: 无前视 top-k accuracy ============
def full_eval(pv_es, y_es, pv_te, y_te, ts_te, label, save_prefix=None):
    """输出 AUC + top-{0.5,1,2,3,5}% accuracy + monthly stability + 简单 gate 扫描。"""
    out = {}
    auc_te = roc_auc_score(y_te, pv_te)
    auc_es = roc_auc_score(y_es, pv_es)
    print(f"\n  ★ ES AUC={auc_es:.4f}  TE AUC={auc_te:.4f}", flush=True)
    out['auc_es'] = auc_es; out['auc_te'] = auc_te

    DAYS = (ts_te[-1] - ts_te[0]) / 86400.0
    for pct in [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0]:
        k = max(1, int(len(pv_te) * pct / 100))
        top_idx = np.argsort(-pv_te)[:k]
        acc = y_te[top_idx].mean() * 100
        tpd = k / DAYS
        flag = '🎯' if pct == 1.0 and acc >= 60 else ('★' if pct == 1.0 and acc >= 55 else '')
        print(f"  top-{pct:.1f}%: acc={acc:.1f}%  tpd≈{tpd:.1f}  {flag}", flush=True)
        out[f'top{pct}_acc'] = acc; out[f'top{pct}_tpd'] = tpd

    # Monthly stability
    dt = pd.to_datetime(ts_te, unit='s', utc=True)
    month = dt.to_period('M').values
    all_m = sorted(pd.PeriodIndex(np.unique(month)))
    bad = 0; accs = []
    for mm in all_m:
        hm = month == mm; n = hm.sum(); k = max(1, int(n*0.01))
        a = y_te[hm][np.argsort(-pv_te[hm])[:k]].mean() * 100
        accs.append(a)
        if a < 55: bad += 1
    mean_a = np.mean(accs); std_a = np.std(accs)
    print(f"  monthly mean={mean_a:.1f}%  std={std_a:.1f}pp  bad_months(<55%)={bad}/{len(all_m)}", flush=True)
    out['monthly_mean'] = mean_a; out['monthly_std'] = std_a
    out['monthly_bad'] = bad

    # ==== 无前视评估 (rolling 90d P99 threshold, 独立 long/short) ====
    print(f"\n  --- 无前视滚动阈值评估 ---", flush=True)
    conf = np.abs(pv_te - 0.5) * 2
    # 90天 ≈ 90*1440/STRIDE 个窗口
    win_len = int(90 * 1440 / STRIDE)
    n = len(conf)
    # 滚动 P99
    q = 99.0
    thresholds = np.full(n, np.nan)
    # 滑窗: thresholds[i] 用 conf[i-win_len : i] (严格过去)
    for i in range(win_len, n):
        thresholds[i] = np.percentile(conf[i-win_len:i], q)
    # 用 conf > threshold 做 LONG / 1-conf > threshold 做 SHORT
    long_sel = conf > thresholds
    short_sel = (1 - pv_te - 0.5) * 2 > thresholds  # 等价于 (pv_te < 0.5 - threshold/2)
    sel = long_sel | short_sel
    n_sel = sel.sum()
    if n_sel > 100:
        acc_no_lookahead = y_te[sel].mean() * 100
        tpd_no = n_sel / DAYS
        print(f"  无前视 q=99 conf gate: acc={acc_no_lookahead:.1f}%  tpd={tpd_no:.1f}  (n={n_sel:,})", flush=True)
        out['no_lookahead_acc'] = acc_no_lookahead; out['no_lookahead_tpd'] = tpd_no
    else:
        print(f"  无前视样本不足 (n={n_sel})", flush=True)

    out['label'] = label
    if save_prefix:
        np.savez(save_prefix, pv_te=pv_te, y_te=y_te, ts_te=ts_te,
                 pv_es=pv_es, y_es=y_es, meta=out)
    return out

# ============ 8. 迭代实验 ============
C_IN = channels.shape[0]
res = []
round_idx = 0

def run_round(name, kwargs, label):
    global round_idx
    round_idx += 1
    print(f"\n{'='*60}")
    print(f"ROUND {round_idx}: {label} ({name})", flush=True)
    print(f"  {kwargs}", flush=True)
    r = train_one(name, X_tr, y_tr, X_es, y_es, X_te, y_te, ts_te, r_tr,
                  C_IN, WINDOW, **kwargs)
    save_p = f'/workspace/models_saved/seq_round{round_idx:02d}'
    m_res = full_eval(r['pv_es'], y_es, r['pv_te'], y_te, ts_te, label, save_prefix=save_p)
    m_res['round'] = round_idx
    m_res['model_name'] = name
    m_res['kwargs'] = kwargs
    m_res['time_s'] = r['time_s']
    res.append(m_res)
    return m_res

# ===== Round 1: LSTM+Attn baseline =====
r1 = run_round('lstm_attn', dict(epochs=25, lr=2e-3, wd=1e-4, batch=256, pat=7),
               'LSTM+Attn H=15 W=96 baseline')

if r1['top1_acc'] >= 60.0:
    print(f"\n🎉 ROUND 1 已达标 top1={r1['top1_acc']:.1f}%", flush=True)

# ===== Round 2: TCN =====
r2 = run_round('tcn', dict(epochs=30, lr=3e-3, wd=1e-4, batch=256, pat=8),
               'TCN dilated convolutions')

# ===== Round 3: Transformer =====
r3 = run_round('transformer', dict(epochs=25, lr=3e-3, wd=1e-4, batch=256, pat=7),
               'Transformer encoder')

# ===== Round 4: GRU+CNN hybrid =====
r4 = run_round('gru_cnn', dict(epochs=25, lr=2e-3, wd=1e-4, batch=256, pat=7),
               'GRU+CNN hybrid')

# ===== Round 5: 训练更大数据 (1M) + 更强 label filter =====
print(f"\n{'='*60}", flush=True)
print("ROUND 5: 更大训练集 + 更强 label filter", flush=True)
MAX_TR = 1_000_000
TR_FILTER = 0.0005
tr_filt = tr_m & (np.abs(r_all) > TR_FILTER)
print(f"  train keep: {tr_filt.sum():,}", flush=True)
X_tr2 = X_all[tr_filt].copy(); y_tr2 = y_all[tr_filt].copy(); r_tr2 = r_all[tr_filt].copy()
if len(X_tr2) > MAX_TR:
    rng = np.random.default_rng(42)
    keep = rng.choice(len(X_tr2), MAX_TR, replace=False)
    X_tr2 = X_tr2[keep]; y_tr2 = y_tr2[keep]; r_tr2 = r_tr2[keep]
print(f"  sampled: TR={len(X_tr2):,}", flush=True)

r5 = run_round('lstm_attn_big', dict(epochs=30, lr=1.5e-3, wd=2e-4, batch=192, pat=10),
               'LSTM+Attn big, 1M train, ret filter 0.05%')

# ===== Round 6: 换更长窗口 (W=192) =====
print(f"\n{'='*60}", flush=True)
print("ROUND 6: WINDOW=192 (更长历史)", flush=True)
WINDOW_2 = 192
X_all2, y_all2, r_all2, ts_all2, _ = build_windows(H, WINDOW_2, STRIDE, label_filter_eps=0.0)
tr_filt2 = (ts_all2 >= int(dtm.datetime.strptime(config.SPLITS['train'][0],'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())) & \
           (ts_all2 < int(dtm.datetime.strptime(config.SPLITS['train'][1],'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())) & \
           (np.abs(r_all2) > 0.0003)
es_m2 = (ts_all2 >= int(dtm.datetime.strptime(config.SPLITS['early_stop'][0],'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())) & \
        (ts_all2 < int(dtm.datetime.strptime(config.SPLITS['early_stop'][1],'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp()))
te_m2 = (ts_all2 >= int(dtm.datetime.strptime(config.SPLITS['test'][0],'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp()))
X_tr3 = X_all2[tr_filt2].copy(); y_tr3 = y_all2[tr_filt2].copy(); r_tr3 = r_all2[tr_filt2].copy()
X_es3 = X_all2[es_m2]; y_es3 = y_all2[es_m2].copy()
X_te3 = X_all2[te_m2]; y_te3 = y_all2[te_m2].copy(); ts_te3 = ts_all2[te_m2].copy()
del X_all2, y_all2, r_all2, ts_all2; gc.collect()
if len(X_tr3) > MAX_TR:
    rng = np.random.default_rng(42)
    keep = rng.choice(len(X_tr3), MAX_TR, replace=False)
    X_tr3 = X_tr3[keep]; y_tr3 = y_tr3[keep]; r_tr3 = r_tr3[keep]
print(f"  TR={len(X_tr3):,}  ES={len(X_es3):,}  TE={len(X_te3):,}", flush=True)

def run_round2(name, kwargs, label):
    global round_idx
    round_idx += 1
    print(f"\nROUND {round_idx}: {label}", flush=True)
    r = train_one(name, X_tr3, y_tr3, X_es3, y_es3, X_te3, y_te3, ts_te3, r_tr3,
                  C_IN, WINDOW_2, **kwargs)
    full_eval(r['pv_es'], y_es3, r['pv_te'], y_te3, ts_te3, label)
    return r

r6 = run_round2('tcn_deep', dict(epochs=30, lr=2e-3, wd=2e-4, batch=192, pat=10),
               'TCN deep, W=192')

# ===== Round 7: 多 horizon 预训练 (H5→H15 知识迁移) =====
print(f"\n{'='*60}", flush=True)
print("ROUND 7: H=5 预训练 → H=15 微调 (多 horizon 迁移)", flush=True)
X_all5, y_all5, r_all5, ts_all5, _ = build_windows(5, WINDOW, STRIDE, label_filter_eps=0.0)
tr_f5 = (ts_all5 >= int(dtm.datetime.strptime(config.SPLITS['train'][0],'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())) & \
        (ts_all5 < int(dtm.datetime.strptime(config.SPLITS['train'][1],'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())) & \
        (np.abs(r_all5) > 0.0003)
es_m5 = (ts_all5 >= int(dtm.datetime.strptime(config.SPLITS['early_stop'][0],'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())) & \
        (ts_all5 < int(dtm.datetime.strptime(config.SPLITS['early_stop'][1],'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp()))
X_tr5 = X_all5[tr_f5].copy(); y_tr5 = y_all5[tr_f5].copy(); r_tr5 = r_all5[tr_f5].copy()
X_es5 = X_all5[es_m5]; y_es5 = y_all5[es_m5].copy()
del X_all5, y_all5, r_all5, ts_all5; gc.collect()
if len(X_tr5) > MAX_TR:
    rng = np.random.default_rng(42)
    keep = rng.choice(len(X_tr5), MAX_TR, replace=False)
    X_tr5 = X_tr5[keep]; y_tr5 = y_tr5[keep]; r_tr5 = r_tr5[keep]

# 预训练
m_pre = make_model('lstm_attn_big', C_IN, WINDOW).to(DEVICE)
opt_pre = torch.optim.AdamW(m_pre.parameters(), lr=3e-3, weight_decay=1e-4)
best_pre = 0; best_st_pre = None
for ep in range(15):
    m_pre.train(); idx = np.random.permutation(len(X_tr5))
    for i in range(0, len(idx), 256):
        b = idx[i:i+256]
        xb = torch.from_numpy(X_tr5[b]).to(DEVICE); yb = torch.from_numpy(y_tr5[b]).float().to(DEVICE)
        rw = torch.from_numpy(np.clip(np.abs(r_tr5[b])*200,0.2,5.0)).float().to(DEVICE)
        loss = F.binary_cross_entropy_with_logits(m_pre(xb), yb, weight=rw)
        opt_pre.zero_grad(); loss.backward(); opt_pre.step()
    pv5, _ = evaluate(m_pre, X_es5, y_es5)
    auc5 = roc_auc_score(y_es5, pv5)
    print(f"  pretrain H=5 ep{ep+1}: es_auc={auc5:.4f}", flush=True)
    if auc5 > best_pre:
        best_pre = auc5
        best_st_pre = {k: v.detach().clone() for k, v in m_pre.state_dict().items()}

# 加载到 H=15 训练
m = make_model('lstm_attn_big', C_IN, WINDOW).to(DEVICE)
if best_st_pre: m.load_state_dict(best_st_pre)
round_idx += 1
print(f"\nROUND {round_idx}: LSTM+Attn H5→H15 transfer", flush=True)
# 用更大 batch, 冻结部分层先 warmup
opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=2e-4)
best_auc = 0; best_st = None; no_imp = 0
for ep in range(15):
    m.train(); idx = np.random.permutation(len(X_tr))
    for i in range(0, len(idx), 256):
        b = idx[i:i+256]
        xb = torch.from_numpy(X_tr[b]).to(DEVICE); yb = torch.from_numpy(y_tr[b]).float().to(DEVICE)
        rw = torch.from_numpy(np.clip(np.abs(r_tr[b])*200,0.2,5.0)).float().to(DEVICE)
        loss = F.binary_cross_entropy_with_logits(m(xb), yb, weight=rw)
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    pv_es, _ = evaluate(m, X_es, y_es); pv_te, _ = evaluate(m, X_te, y_te)
    auc_es = roc_auc_score(y_es, pv_es); auc_te = roc_auc_score(y_te, pv_te)
    print(f"  ft ep{ep+1}: es={auc_es:.4f} te={auc_te:.4f}", flush=True)
    if auc_es > best_auc + 1e-5:
        best_auc = auc_es; best_st = {k: v.detach().clone() for k,v in m.state_dict().items()}
        no_imp = 0
    else:
        no_imp += 1
        if no_imp >= 5: break
if best_st: m.load_state_dict(best_st)
pv_te, _ = evaluate(m, X_te, y_te); pv_es, _ = evaluate(m, X_es, y_es)
r7 = full_eval(pv_es, y_es, pv_te, y_te, ts_te, 'LSTM H5→H15 transfer')

# ===== Round 8: Label smoothing + 更激进的 ret weight =====
r8 = run_round('lstm_attn_big', dict(epochs=25, lr=2e-3, wd=1e-4, batch=256, pat=8,
                                      label_smoothing=0.05, use_ret_weight=True),
               'LSTM big + label smoothing 0.05')

# ===== Round 9: 数据增强 (random jitter on channels during train) =====
class AugDataset:
    def __init__(self, X, y, r, jitter_std=0.01, dropout_p=0.1):
        self.X = X; self.y = y; self.r = r
        self.jitter = jitter_std; self.dropout = dropout_p
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        x = self.X[i].copy()
        if self.jitter > 0:
            x = x + np.random.randn(*x.shape).astype(np.float32) * self.jitter
        if self.dropout > 0:
            mask = np.random.rand(*x.shape) > self.dropout
            x = x * mask
        return x.astype(np.float32), self.y[i], self.r[i]

def train_with_aug(name, X_tr, y_tr, r_tr, X_es, y_es, X_te, y_te, ts_te,
                   C_in, WINDOW, epochs=25, lr=2e-3, wd=1e-4, batch=256, pat=7,
                   jitter=0.02, drop=0.1):
    global round_idx
    round_idx += 1
    print(f"\nROUND {round_idx}: {name} + aug (jitter={jitter}, drop={drop})", flush=True)
    m = make_model(name, C_in, WINDOW).to(DEVICE)
    n_params = sum(p.numel() for p in m.parameters())
    ds_tr = AugDataset(X_tr, y_tr, r_tr, jitter_std=jitter, dropout_p=drop)
    dl_tr = DataLoader(ds_tr, batch_size=batch, shuffle=True, num_workers=0)
    pos = y_tr.mean(); pos_w = (1-pos)/pos; neg_w = pos/(1-pos)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_auc = 0; best_st = None; no_imp = 0; t0 = time.time()
    for ep in range(epochs):
        m.train(); tl = 0; n_b = 0
        for xb, yb, rb in dl_tr:
            xb = xb.to(DEVICE); yb = yb.float().to(DEVICE); rb = rb.to(DEVICE)
            pw = torch.where(yb > 0.5, pos_w, neg_w).float().to(DEVICE)
            rw = torch.clip(torch.abs(rb) * 200, 0.2, 5.0).float().to(DEVICE)
            loss = F.binary_cross_entropy_with_logits(m(xb), yb, weight=rw * pw)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
            tl += loss.item(); n_b += 1
        sch.step()
        pv_es, _ = evaluate(m, X_es, y_es); pv_te, _ = evaluate(m, X_te, y_te)
        auc_es = roc_auc_score(y_es, pv_es); auc_te = roc_auc_score(y_te, pv_te)
        print(f"  ep{ep+1:2d} loss={tl/n_b:.4f} es={auc_es:.4f} te={auc_te:.4f}", flush=True)
        if auc_es > best_auc + 1e-5:
            best_auc = auc_es
            best_st = {k: v.detach().clone() for k,v in m.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= pat: break
    if best_st: m.load_state_dict(best_st)
    pv_te, _ = evaluate(m, X_te, y_te); pv_es, _ = evaluate(m, X_es, y_es)
    full_eval(pv_es, y_es, pv_te, y_te, ts_te, f'{name} + aug')
    return m

train_with_aug('lstm_attn_big', X_tr, y_tr, r_tr, X_es, y_es, X_te, y_te, ts_te,
               C_IN, WINDOW, epochs=30, lr=1.5e-3, wd=2e-4, batch=192, pat=10,
               jitter=0.03, drop=0.15)

# ===== Round 10: TCN + aug =====
train_with_aug('tcn_deep', X_tr, y_tr, r_tr, X_es, y_es, X_te, y_te, ts_te,
               C_IN, WINDOW, epochs=30, lr=2e-3, wd=2e-4, batch=256, pat=8,
               jitter=0.02, drop=0.1)

# ===== Round 11: 多头集成 (3 个不同 seed + 不同 arch) =====
print(f"\n{'='*60}", flush=True)
print("ROUND 11: 多头集成 (3 模型 bagging, 不同 seed + arch)", flush=True)
m1 = make_model('lstm_attn_big', C_IN, WINDOW); torch.manual_seed(42)
m2 = make_model('tcn_deep', C_IN, WINDOW); torch.manual_seed(49)
m3 = make_model('transformer', C_IN, WINDOW); torch.manual_seed(56)

def train_single(m, X_tr, y_tr, r_tr, X_es, y_es, ep=25, lr=2e-3, wd=1e-4, batch=256):
    ds = AugDataset(X_tr, y_tr, r_tr, jitter_std=0.02, dropout_p=0.1)
    dl = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=0)
    pos = y_tr.mean(); pos_w = (1-pos)/pos; neg_w = pos/(1-pos)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    best = 0; st = None; ni = 0
    for e in range(ep):
        m.train()
        for xb, yb, rb in dl:
            xb = xb.to(DEVICE); yb = yb.float().to(DEVICE); rb = rb.to(DEVICE)
            pw = torch.where(yb>0.5, pos_w, neg_w).float().to(DEVICE)
            rw = torch.clip(torch.abs(rb)*200, 0.2, 5.0).float().to(DEVICE)
            loss = F.binary_cross_entropy_with_logits(m(xb), yb, weight=rw*pw)
            opt.zero_grad(); loss.backward(); opt.step()
        pv_es, _ = evaluate(m, X_es, y_es)
        auc_es = roc_auc_score(y_es, pv_es)
        if auc_es > best + 1e-5: best = auc_es; st = {k:v.detach().clone() for k,v in m.state_dict().items()}; ni=0
        else: ni += 1
        if ni >= 6: break
    if st: m.load_state_dict(st)
    return m

m1 = train_single(m1, X_tr, y_tr, r_tr, X_es, y_es); print(f"  m1 done")
m2 = train_single(m2, X_tr, y_tr, r_tr, X_es, y_es); print(f"  m2 done")
m3 = train_single(m3, X_tr, y_tr, r_tr, X_es, y_es); print(f"  m3 done")

def rank_ensemble(models, X, y):
    pv_all = []
    for m in models:
        pv, _ = evaluate(m, X, y)
        pv_all.append(pv)
    # rank within sample, then mean
    R = np.zeros((len(models), len(X)), dtype=np.float64)
    for i, pv in enumerate(pv_all):
        R[i] = np.argsort(np.argsort(pv)).astype(np.float64) / (len(X)-1)
    return R.mean(axis=0), pv_all

pv_ens_te, pv_all_te = rank_ensemble([m1, m2, m3], X_te, y_te)
pv_ens_es, _ = rank_ensemble([m1, m2, m3], X_es, y_es)
round_idx += 1
full_eval(pv_ens_es, y_es, pv_ens_te, y_te, ts_te, '3-model rank ensemble')

# 单模型对比
for i, (pv, label) in enumerate(zip(pv_all_te, ['lstm_big', 'tcn_deep', 'transformer'])):
    pv_es_i, _ = evaluate([m1,m2,m3][i], X_es, y_es)
    round_idx += 1
    full_eval(pv_es_i, y_es, pv, y_te, ts_te, label)

# ===== 总结 =====
print(f"\n{'='*60}", flush=True)
print("ALL ROUNDS SUMMARY", flush=True)
print(f"{'='*60}", flush=True)
for r in res:
    flag = '🏆' if r['top1_acc'] >= 60 else ('⭐' if r['top1_acc'] >= 57 else '  ')
    print(f"  R{r['round']:02d} {flag} AUC={r['auc_te']:.4f}  top1%={r['top1_acc']:.1f}%  tpd={r.get('top1_tpd',0):.1f}  "
          f"monthly={r.get('monthly_mean',0):.1f}±{r.get('monthly_std',0):.1f}  [{r.get('model_name','?')}]", flush=True)
print(f"\nTOTAL TIME: {time.time()-t0:.0f}s", flush=True)
