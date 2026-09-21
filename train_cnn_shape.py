"""
Shape-CNN: 保留完整时间序列形态, CNN 直接学形态结构.

核心转变: 不做 flatten! 保留 Stoch/DMI/close 的原始时间序列形态.

通道 (per window, shape = (C, SEG)):
  [0] close_ret_60  — close / close_t0 (60 分钟价格形态)
  [1] stoch_k_60    — Stochastic(60,1,1) %K 原始序列 (60 分钟形态)
  [2] stoch_k_diff  — %K - SMA(%K, 3) (形态 derivative: 快慢线分离)
  [3] plus_di_14    — DMI +DI 原始序列
  [4] minus_di_14   — DMI -DI 原始序列
  [5] di_diff       — +DI - -DI (形态: DI 交叉/分离)
  [6] adx_14        — ADX 原始序列 (趋势强度形态)
  [7] cvd_norm      — normalized buy-sell cumsum 形态
  [8] btc_ret       — BTC close / BTC_t0 (cross-asset 形态)

模型: 1D Dilated CNN, 多尺度卷积核同时看 5min/15min/30min/60min 形态
"""
import os, time, gc, math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score
import config

# ============ 超参 ============
SEG = 60              # 窗口长度 (与 Stoch 周期对齐)
STRIDE = 8            # 滑窗步长
H = config.HORIZON_MIN
BATCH = 256
EPOCHS = 25
LR = 2e-3
WD = 5e-4
PATIENCE = 8
DEVICE = torch.device('cpu')

# ============ 1. 加载 ============
t0 = time.time()
print(f'[1] 加载数据 SEG={SEG} STRIDE={STRIDE} H={H}', flush=True)
eth = pd.read_parquet(f'{config.DS_DIR}/raw_ETH.parquet')
btc = pd.read_parquet(f'{config.DS_DIR}/raw_BTC.parquet')
bi = pd.Index(btc['ts'].values); ei = pd.Index(eth['ts'].values)
br = np.clip(btc['ts'].values.searchsorted(ei.values, side='right') - 1, 0, len(btc)-1)

close = eth['close'].values.astype(np.float32)
high = eth['high'].values.astype(np.float32)
low = eth['low'].values.astype(np.float32)
buy_v = eth['buy_vol'].values.astype(np.float32)
sell_v = eth['sell_vol'].values.astype(np.float32)
ts = eth['ts'].values.astype(np.int64)
btc_c = btc['close'].values.astype(np.float32)[br]
N = len(close)
del btc; gc.collect()

# ============ 2. 预计算 Stoch(60), DMI(14) 全局序列 ============
print(f'[2] 预计算 Stoch/DMI 全序列 ...', flush=True)

# Stochastic(60, 1, 1): %K = (C-LL)/(HH-LL) — rolling
# 用 rolling min/max (pandas 快)
close_s = pd.Series(close); high_s = pd.Series(high); low_s = pd.Series(low)
ll_60 = low_s.rolling(SEG, min_periods=SEG).min().values
hh_60 = high_s.rolling(SEG, min_periods=SEG).max().values
hk = hh_60 - ll_60; hk = np.maximum(hk, 1e-6)
stoch_k_all = (close - ll_60) / hk  # shape (N,), NaN for t < SEG
# stoch_d = SMA(stoch_k, 3)  — 快慢线差 = 形态 derivative
stoch_d_all = pd.Series(stoch_k_all).rolling(3, min_periods=1).mean().values
stoch_diff_all = stoch_k_all - stoch_d_all  # 形态: 金叉/死叉信号

# DMI(14) — Wilder 平滑
h_l = high[1:] - low[1:]; h_cp = np.abs(high[1:] - close[:-1]); l_cp = np.abs(low[1:] - close[:-1])
tr = np.maximum(np.maximum(h_l, h_cp), l_cp)
up_m = high[1:] - high[:-1]; dn_m = low[:-1] - low[1:]
pdm = np.where((up_m > dn_m) & (up_m > 0), up_m, 0.0)
mdm = np.where((dn_m > up_m) & (dn_m > 0), dn_m, 0.0)

def wilder_smooth(arr, p):
    out = np.zeros(len(arr), np.float64)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = out[i-1] - out[i-1]/p + arr[i]
    return out

tr_s = wilder_smooth(tr, 14)
pdm_s = wilder_smooth(pdm, 14)
mdm_s = wilder_smooth(mdm, 14)
pdi_all = np.zeros(N, np.float32); mdi_all = np.zeros(N, np.float32)
pdi_all[1:] = (100 * pdm_s / np.maximum(tr_s, 1e-6)).astype(np.float32)
mdi_all[1:] = (100 * mdm_s / np.maximum(tr_s, 1e-6)).astype(np.float32)
di_diff_all = pdi_all - mdi_all
adx_all = np.zeros(N, np.float32)
dx = 100 * np.abs(pdm_s - mdm_s) / np.maximum(pdm_s + mdm_s, 1e-6)
adx_all[1:] = wilder_smooth(dx, 14).astype(np.float32)

# CVD — normalized buy-sell cumsum over 60 bars
bs = buy_v - sell_v  # buy-sell per bar

print(f'  预计算完成 ({time.time()-t0:.0f}s)', flush=True)

# ============ 3. 构建形态窗口 (保留完整时间序列) ============
print(f'\n[3] 构建 {SEG} 分钟形态窗口 (STRIDE={STRIDE}) ...', flush=True)

W_list = []; T_list = []; Ts_list = []
cnt = 0
for start in range(SEG, N - SEG - H, STRIDE):
    end = start + SEG
    if end + H > N: break

    c0 = close[start]
    bc0 = btc_c[start]

    w = np.zeros((9, SEG), np.float32)

    # [0] close_ret 形态 (以 start 为锚, log return)
    w[0] = np.log(close[start:end] / max(c0, 1e-6))

    # [1] stoch_k 原始形态 (60 分钟 %K 走势)
    sk = stoch_k_all[start:end]
    if np.isnan(sk).any(): continue
    w[1] = sk

    # [2] stoch_diff 形态 (快慢线分离 — 金叉/死叉信号序列)
    w[2] = stoch_diff_all[start:end]

    # [3] +DI 原始形态
    w[3] = pdi_all[start:end]

    # [4] -DI 原始形态
    w[4] = mdi_all[start:end]

    # [5] DI diff 形态 (+DI 上穿 -DI 的位置 = 这里序列会过零)
    w[5] = di_diff_all[start:end]

    # [6] ADX 形态 (趋势强度走势)
    w[6] = adx_all[start:end]

    # [7] CVD 形态 (窗口内 buy-sell 累积走势)
    bs_win = bs[start:end]
    cum_bs = np.cumsum(bs_win)
    w[7] = cum_bs / max(np.abs(cum_bs[-1]), 1.0)  # 归一化

    # [8] btc_ret 形态
    w[8] = np.log(btc_c[start:end] / max(bc0, 1e-6))

    if np.isnan(w).any() or np.isinf(w).any(): continue

    label = 1 if close[end + H] > close[end] else 0
    W_list.append(w); T_list.append(label); Ts_list.append(ts[end])
    cnt += 1
    if cnt % 200_000 == 0:
        print(f'  ... {cnt:,} ({time.time()-t0:.0f}s)', flush=True)

X_all = np.array(W_list, np.float32)
y_all = np.array(T_list, np.int64)
ts_all = np.array(Ts_list, np.int64)
del W_list, T_list, Ts_list, stoch_k_all, stoch_d_all, pdi_all, mdi_all, di_diff_all, adx_all, bs; gc.collect()
print(f'  ✅ {X_all.shape}  pos_rate={y_all.mean():.3f} ({time.time()-t0:.0f}s)', flush=True)

# ============ 4. 时间切分 ============
def mk(s, e):
    a = int(pd.Timestamp(s, tz='UTC').timestamp())
    b = int(pd.Timestamp(e, tz='UTC').timestamp())
    return (ts_all >= a) & (ts_all < b)
tr_m = mk(*config.SPLITS['train'])
es_m = mk(*config.SPLITS['early_stop'])
te_m = mk(*config.SPLITS['test'])
X_tr, y_tr = X_all[tr_m], y_all[tr_m]
X_es, y_es = X_all[es_m], y_all[es_m]
X_te, y_te = X_all[te_m], y_all[te_m]
del X_all, y_all, ts_all; gc.collect()
print(f'  TR={len(X_tr):,}  ES={len(X_es):,}  TE={len(X_te):,}', flush=True)

# ============ 5. 1D Dilated CNN (多尺度形态卷积核) ============
class DilatedRes(nn.Module):
    def __init__(self, ch, dil):
        super().__init__()
        self.c1 = nn.Conv1d(ch, ch, 5, padding=2*dil, dilation=dil, bias=False)
        self.c2 = nn.Conv1d(ch, ch, 5, padding=2*dil, dilation=dil, bias=False)
        self.ln = nn.LayerNorm([ch, SEG])
    def forward(self, x):
        res = x
        x = self.c2(F.gelu(self.ln(self.c1(x))))
        return res + x

class ShapeCNN(nn.Module):
    """多尺度卷积核: 同时捕捉 5min / 15min / 30min / 60min 形态"""
    def __init__(self, in_ch=9):
        super().__init__()
        # Stem
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, 96, 3, padding=1, bias=False),
            nn.LayerNorm([96, SEG]),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        # 分支 A: 短周期形态 (5-15min)
        self.sh1 = DilatedRes(96, 1)   # RF=5 (5min 形态)
        self.sh2 = DilatedRes(96, 2)   # RF=9  (~10min)

        # 分支 B: 中周期形态 (15-30min)
        self.md1 = DilatedRes(96, 4)   # RF=17 (~15min)
        self.md2 = DilatedRes(96, 8)   # RF=33 (~30min)

        # 分支 C: 长周期形态 (30-60min)
        self.lg1 = DilatedRes(96, 16)  # RF=65 (~60min, 覆盖全窗口!)

        # 融合
        self.fuse = nn.Sequential(
            nn.Conv1d(96*5, 192, 1, bias=False),
            nn.LayerNorm([192, SEG]),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(192, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )
    def forward(self, x):
        x = self.stem(x)
        a = self.sh2(self.sh1(x))
        b = self.md2(self.md1(x))
        c = self.lg1(x)
        fused = self.fuse(torch.cat([a, b, c, x, (a+b+c)/3], dim=1))
        return self.head(self.pool(fused).squeeze(-1))

model = ShapeCNN(in_ch=9)
np_ = sum(p.numel() for p in model.parameters())
print(f'\n[5] ShapeCNN params={np_:,} ({np_/1e6:.2f}M)', flush=True)

# ============ 6. 训练 ============
def to_ds(X, y):
    return TensorDataset(torch.from_numpy(X), torch.from_numpy(y))

ds_tr = to_ds(X_tr, y_tr); ds_es = to_ds(X_es, y_es)
dl_tr = DataLoader(ds_tr, BATCH, shuffle=True, num_workers=0, drop_last=True)
dl_es = DataLoader(ds_es, BATCH, shuffle=False, num_workers=0)
del X_tr, y_tr, X_es, y_es; gc.collect()

optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=EPOCHS, eta_min=1e-5)
crit = nn.BCEWithLogitsLoss()

best_auc = 0; best_st = None; ni = 0
print(f'\n[6] 训练 EPOCHS={EPOCHS}', flush=True)

for ep in range(1, EPOCHS+1):
    model.train(); tl = []
    for xb, yb in dl_tr:
        optim.zero_grad(); log = model(xb).squeeze(-1); loss = crit(log, yb.float())
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optim.step()
        tl.append(loss.item())
    sched.step()
    model.eval(); pva, ya = [], []
    with torch.no_grad():
        for xb, yb in dl_es:
            pva.append(torch.sigmoid(model(xb).squeeze(-1)).numpy())
            ya.append(yb.numpy())
    pv = np.concatenate(pva); y = np.concatenate(ya)
    auc = roc_auc_score(y, pv)
    if auc > best_auc + 1e-5:
        best_auc = auc; best_st = {k: v.clone() for k,v in model.state_dict().items()}; ni = 0
    else:
        ni += 1
    print(f'  ep {ep:2d}  tr={np.mean(tl):.4f}  es_auc={auc:.4f}  best={best_auc:.4f}  '
          f'lr={sched.get_last_lr()[0]:.2e} {"★" if ni==0 else ""} ({time.time()-t0:.0f}s)', flush=True)
    if ni >= PATIENCE: print(f'  ⏹ early stop', flush=True); break

# ============ 7. Test ============
print(f'\n[7] Test 最终评估', flush=True)
if best_st: model.load_state_dict(best_st)
model.eval()
dste = to_ds(X_te, y_te)
dlte = DataLoader(dste, BATCH, shuffle=False, num_workers=0)
pva, ya = [], []
with torch.no_grad():
    for xb, yb in dlte:
        pva.append(torch.sigmoid(model(xb).squeeze(-1)).numpy())
        ya.append(yb.numpy())
pv = np.concatenate(pva); y = np.concatenate(ya)
auc = roc_auc_score(y, pv)
print(f'  📊 Test AUC={auc:.4f}')

for pct in [0.005, 0.01, 0.02, 0.03, 0.05]:
    k = max(1, int(len(pv)*pct)); ti = pv.argsort()[-k:]
    acc = y[ti].mean()*100; tpd = k / 356
    flag = '🎯' if acc >= 65 and tpd >= 14 else ('★' if acc >= 60 and tpd >= 14 else '')
    print(f'  {flag} top{pct*100:.1f}%: acc={acc:.1f}% tpd={tpd:.1f}')

print(f'\n⏱ {time.time()-t0:.0f}s')
