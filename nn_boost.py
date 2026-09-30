"""NN 提升实验: 系统测试多种 MLP/CNN 配置, 目标超越 Tree AUC=0.5406。
关键策略:
1. 更大数据 (800K) + 样本权重 (极端 ret 降权)
2. 更宽/更深 MLP + 合理正则
3. CNN1D 序列模型
4. 多 seed 集成
"""
import sys; sys.stdout.reconfigure(line_buffering=True)
import time, gc, os
import numpy as np
import polars as pl
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import config

torch.manual_seed(42); np.random.seed(42)

# ============ 数据加载 (float16 省内存) ============
t0 = time.time()
print("Loading data...", flush=True)
df = pl.read_parquet(f"{config.DS_DIR}/ds_ETH_h15.parquet").sort("ts")
feat_cols = [c for c in df.columns if c not in ["ts", "label", "soft_label", "ret_future"]]
ts_all = df["ts"].to_numpy()
label_all = df["label"].to_numpy().astype(np.float32)
ret_all = df["ret_future"].to_numpy().astype(np.float32)
X_all = np.nan_to_num(df.select(feat_cols).to_numpy().astype(np.float16), nan=0.0)
del df; gc.collect()

TRAIN_END = 1722556800; ES_END = 1729641600; META_END = 1751241600
np.random.seed(42)
tr_idx = np.where(ts_all <= TRAIN_END)[0]
tr_idx_sub = np.random.choice(tr_idx, 800_000, replace=False)
mask_tr = np.zeros(len(ts_all), dtype=bool); mask_tr[tr_idx_sub] = True
mask_es = (ts_all > TRAIN_END) & (ts_all <= ES_END)
mask_te = ts_all > META_END

X_tr = X_all[mask_tr].astype(np.float32)
X_es = X_all[mask_es].astype(np.float32)
X_te = X_all[mask_te].astype(np.float32)
y_tr = label_all[mask_tr]
y_es = label_all[mask_es]
y_te = label_all[mask_te]
r_tr = ret_all[mask_tr]
del X_all, ts_all, label_all, ret_all; gc.collect()
print(f"TR={X_tr.shape} ES={X_es.shape} TE={X_te.shape} [{time.time()-t0:.1f}s]", flush=True)

# 标准化
mu = X_tr.mean(0); sd = X_tr.std(0) + 1e-8
X_tr = ((X_tr - mu) / sd).astype(np.float32)
X_es = ((X_es - mu) / sd).astype(np.float32)
X_te = ((X_te - mu) / sd).astype(np.float32)
gc.collect()

# 样本权重: 极端 ret 降权 (和 Tree 一样的技巧)
rw = np.clip(np.abs(r_tr) * 200, 0.2, 5.0).astype(np.float32)
ret_abs = np.abs(r_tr)
lo_q = np.percentile(ret_abs, 10)
hi_q = np.percentile(ret_abs, 90)
ext_mask = (ret_abs <= lo_q) | (ret_abs >= hi_q)
sw = np.where(ext_mask, rw * 0.3, rw).astype(np.float32)
print(f"Sample weights: ext_mask={ext_mask.mean():.1%} sw range=[{sw.min():.2f}, {sw.max():.2f}]", flush=True)

# ============ 模型定义 ============
class MLP(nn.Module):
    def __init__(self, ft, hs, drop=0.5):
        super().__init__()
        prev = ft; layers = []
        for h in hs:
            layers.extend([nn.Linear(prev, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(drop)])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x).squeeze(-1)

class ResMLP(nn.Module):
    """带残差连接的 MLP"""
    def __init__(self, ft, hs, drop=0.5):
        super().__init__()
        self.input = nn.Linear(ft, hs[0])
        self.blocks = nn.ModuleList()
        for i in range(len(hs) - 1):
            self.blocks.append(nn.Sequential(
                nn.Linear(hs[i], hs[i+1]), nn.BatchNorm1d(hs[i+1]), nn.GELU(), nn.Dropout(drop)))
        self.head = nn.Linear(hs[-1], 1)
    def forward(self, x):
        x = F.gelu(self.input(x))
        for b in self.blocks:
            x = x + b(x) if x.shape == b(x).shape else b(x)
        return self.head(x).squeeze(-1)

class TCN1D(nn.Module):
    """1D TCN 用于特征序列 (把 67 维特征当 67 步 1 通道序列)"""
    def __init__(self, C=1, chs=[32,64,128,64], drop=0.3):
        super().__init__()
        prev = C; d = 1; blocks = []
        for c in chs:
            blocks.append(nn.Sequential(
                nn.Conv1d(prev, c, 3, padding=2*d, dilation=d), nn.BatchNorm1d(c), nn.GELU(),
                nn.Conv1d(c, c, 3, padding=2*d, dilation=d), nn.BatchNorm1d(c), nn.GELU(), nn.Dropout(drop)))
            prev = c; d *= 2
        self.blocks = nn.ModuleList(blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(nn.Linear(chs[-1], chs[-1]//2), nn.GELU(), nn.Dropout(0.3), nn.Linear(chs[-1]//2, 1))
    def forward(self, x):
        # x: (N, 67) -> (N, 1, 67)
        x = x.unsqueeze(1)
        for b in self.blocks: x = b(x)
        return self.head(self.pool(x).flatten(1)).squeeze(-1)

def evaluate(m, X, bs=4096):
    m.eval(); pv = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            pv.append(torch.sigmoid(m(torch.from_numpy(np.ascontiguousarray(X[i:i+bs])))).numpy())
    return np.concatenate(pv)

def train_nn(m, Xtr, ytr, Xes, yes, ep=30, bs=512, pat=8, smooth=0.15, lr=5e-4, wd=0.06, use_sw=True):
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    best = 0.0; bst = None; ni = 0; best_ep = 0
    for e in range(ep):
        m.train(); idx = np.random.permutation(len(Xtr))
        for i in range(0, len(idx), bs):
            bi = idx[i:i+bs]
            xb = torch.from_numpy(np.ascontiguousarray(Xtr[bi].copy()))
            yb = torch.from_numpy(ytr[bi]).float()
            if use_sw: wb = torch.from_numpy(sw[bi]).float()
            if smooth > 0: yb = yb * (1 - smooth) + 0.5 * smooth
            logits = m(xb)
            if use_sw:
                loss = F.binary_cross_entropy_with_logits(logits, yb, weight=wb)
            else:
                loss = F.binary_cross_entropy_with_logits(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
        pv_es = evaluate(m, Xes); auc_es = roc_auc_score(yes, pv_es)
        if auc_es > best + 1e-5:
            best = auc_es; bst = {k: v.detach().clone() for k, v in m.state_dict().items()}; ni = 0; best_ep = e + 1
        else:
            ni += 1
            if ni >= pat: break
    if bst: m.load_state_dict(bst)
    pv_te = evaluate(m, X_te); pv_es = evaluate(m, Xes)
    return pv_te, pv_es, roc_auc_score(y_te, pv_te), best_ep

def rank_agg(pvs):
    R = np.zeros((len(pvs), len(pvs[0])), dtype=np.float64)
    for i, p in enumerate(pvs):
        R[i] = np.argsort(np.argsort(np.nan_to_num(p, nan=0.5))).astype(np.float64) / (len(p) - 1)
    return R.mean(0).astype(np.float32)

def run_exp(label, model_fn, seeds, **train_kwargs):
    print(f"\n{'='*60}\n{label}\n{'='*60}", flush=True)
    pvs_t, pvs_e = [], []
    for s in seeds:
        torch.manual_seed(s); np.random.seed(s)
        m = model_fn()
        t0s = time.time()
        pv_t, pv_e, auc_t, bep = train_nn(m, X_tr, y_tr, X_es, y_es, **train_kwargs)
        pvs_t.append(pv_t); pvs_e.append(pv_e)
        np_ = sum(p.numel() for p in m.parameters())
        print(f"  s={s} ep={bep} TE={auc_t:.4f} params={np_:,} [{time.time()-t0s:.0f}s]", flush=True)
    pv_ens_t = rank_agg(pvs_t); pv_ens_e = rank_agg(pvs_e)
    auc_te = roc_auc_score(y_te, pv_ens_t); auc_es = roc_auc_score(y_es, pv_ens_e)
    print(f"  ★ ENSEMBLE ES={auc_es:.4f} TE={auc_te:.4f}", flush=True)
    return auc_te, pv_ens_t

FT = X_tr.shape[1]
SEEDS = [42, 49, 56]
results = []

# ===== 实验 1: 基线 MLP (不用样本权重) =====
auc1, _ = run_exp(
    'E1: Baseline MLP [512,256] no sw',
    lambda: MLP(FT, [512, 256], 0.6), SEEDS,
    lr=5e-4, wd=0.06, smooth=0.15, ep=25, bs=512, pat=7, use_sw=False)
results.append(('E1_baseline_no_sw', auc1))

# ===== 实验 2: 基线 MLP + 样本权重 =====
auc2, _ = run_exp(
    'E2: MLP [512,256] + sample weights',
    lambda: MLP(FT, [512, 256], 0.6), SEEDS,
    lr=5e-4, wd=0.06, smooth=0.15, ep=25, bs=512, pat=7, use_sw=True)
results.append(('E2_baseline_sw', auc2))

# ===== 实验 3: 更大 MLP =====
auc3, _ = run_exp(
    'E3: Bigger MLP [1024,512,256] drop=0.65 + sw',
    lambda: MLP(FT, [1024, 512, 256], 0.65), SEEDS,
    lr=4e-4, wd=0.07, smooth=0.15, ep=25, bs=512, pat=7, use_sw=True)
results.append(('E3_bigger_mlp', auc3))

# ===== 实验 4: 最宽 MLP =====
auc4, _ = run_exp(
    'E4: Widest MLP [2048,512] drop=0.7 + sw',
    lambda: MLP(FT, [2048, 512], 0.7), SEEDS,
    lr=3e-4, wd=0.08, smooth=0.15, ep=25, bs=512, pat=7, use_sw=True)
results.append(('E4_widest_mlp', auc4))

# ===== 实验 5: 残差 MLP =====
auc5, _ = run_exp(
    'E5: ResMLP [512,512,512] drop=0.5 + sw',
    lambda: ResMLP(FT, [512, 512, 512], 0.5), SEEDS,
    lr=5e-4, wd=0.05, smooth=0.12, ep=25, bs=512, pat=8, use_sw=True)
results.append(('E5_resmlp', auc5))

# ===== 实验 6: TCN1D =====
auc6, _ = run_exp(
    'E6: TCN1D chs=[32,64,128,64] drop=0.3 + sw',
    lambda: TCN1D(1, [32, 64, 128, 64], 0.3), SEEDS,
    lr=3e-3, wd=1e-3, smooth=0.12, ep=30, bs=512, pat=10, use_sw=True)
results.append(('E6_tcn1d', auc6))

# ===== 总结 =====
print(f"\n{'='*60}\nFINAL RANKING (Tree baseline = 0.5406)\n{'='*60}", flush=True)
results.sort(key=lambda x: x[1], reverse=True)
for i, (nm, auc) in enumerate(results):
    diff = auc - 0.5406
    flag = '🏆 BEAT TREE!' if diff > 0 else ''
    print(f"  #{i+1} {nm}: TE AUC={auc:.4f} (diff={diff:+.4f}) {flag}", flush=True)

best_nn = results[0]
print(f"\nBest NN: {best_nn[0]} TE AUC={best_nn[1]:.4f}", flush=True)
print(f"Tree baseline: 0.5406", flush=True)
print(f"NN gap from Tree: {0.5406 - best_nn[1]:.4f}", flush=True)
