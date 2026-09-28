"""Best NN architectures: 1D-CNN, BiLSTM, ResMLP on seq_data_v6 (14×64).
Goal: Beat baseline MLP (AUC=0.5360) and try to get CORR < 0.7 with Tree.
"""
import sys; sys.stdout.reconfigure(line_buffering=True)
import time, gc
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score

t0 = time.time()
torch.manual_seed(42); np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}", flush=True)

# ========== Load Data ==========
d = np.load('/workspace/models_saved/seq_data_v6.npz')
X_tr = d['X_tr'].astype(np.float32)  # (200000, 14, 64)
y_tr = d['y_tr']
X_es = d['X_es'].astype(np.float32)  # (16560, 14, 64)
y_es = d['y_es']
X_te = d['X_te'].astype(np.float32)  # (60118, 14, 64)
y_te = d['y_te']
ts_te = d['ts_te']
C, T = X_tr.shape[1], X_tr.shape[2]  # C=14 channels, T=64 time steps
print(f"Data: C={C} T={T} TR={X_tr.shape} ES={X_es.shape} TE={X_te.shape}", flush=True)
print(f"Labels: TR pos={y_tr.mean():.4f} ES pos={y_es.mean():.4f} TE pos={y_te.mean():.4f}", flush=True)

# ========== Architectures ==========

class ResMLP(nn.Module):
    """MLP with residual connections and batch norm."""
    def __init__(self, c, t, hidden, drop=0.6):
        super().__init__()
        ft = c * t
        self.input_norm = nn.LayerNorm(ft)
        prev = ft
        layers = []
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(drop))
            prev = h
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(prev, 1)
        
    def forward(self, x):
        x = x.flatten(1)
        x = self.input_norm(x)
        x = self.body(x)
        return self.head(x).squeeze(-1)


class CNN1D(nn.Module):
    """1D Convolutional net with multiple kernel sizes for pattern detection."""
    def __init__(self, c, t, channels=[64, 128, 256], kernels=[3,5,7], drop=0.5):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(c)
        
        # Multi-scale convolutions
        self.convs = nn.ModuleList()
        prev_c = c
        for out_c, ks in zip(channels, kernels):
            self.convs.append(nn.Sequential(
                nn.Conv1d(prev_c, out_c, kernel_size=ks, padding=ks//2),
                nn.BatchNorm1d(out_c),
                nn.GELU(),
                nn.MaxPool1d(2),
                nn.Dropout(drop)
            ))
            prev_c = out_c
            t = t // 2
        
        # Global average pooling + classify
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(channels[-1], 128)
        self.fc_drop = nn.Dropout(drop)
        self.fc2 = nn.Linear(128, 1)
        
    def forward(self, x):
        x = self.input_bn(x)
        for conv in self.convs:
            x = conv(x)
        x = self.gap(x).squeeze(-1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc_drop(x)
        return self.fc2(x).squeeze(-1)


class InceptionTime(nn.Module):
    """Inception-style 1D conv for time series (multiple parallel kernels)."""
    def __init__(self, c, t, n_filters=32, drop=0.5):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(c)
        
        def inception_block(in_c, out_c):
            return nn.ModuleList([
                nn.Sequential(nn.Conv1d(in_c, out_c, 1, padding=0), nn.BatchNorm1d(out_c), nn.GELU()),
                nn.Sequential(nn.Conv1d(in_c, out_c, 3, padding=1), nn.BatchNorm1d(out_c), nn.GELU()),
                nn.Sequential(nn.Conv1d(in_c, out_c, 5, padding=2), nn.BatchNorm1d(out_c), nn.GELU()),
                nn.Sequential(nn.MaxPool1d(3, stride=1, padding=1), nn.Conv1d(in_c, out_c, 1), nn.BatchNorm1d(out_c), nn.GELU()),
            ])
        
        self.blocks1 = inception_block(c, n_filters)
        self.pool1 = nn.MaxPool1d(2)
        self.blocks2 = inception_block(n_filters*4, n_filters*2)
        self.pool2 = nn.MaxPool1d(2)
        self.blocks3 = inception_block(n_filters*8, n_filters*4)
        
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.before = None
        self.fc = nn.Sequential(
            nn.Linear(n_filters*16, 256),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(256, 1)
        )
        
    def forward(self, x):
        x = self.input_bn(x)
        for blk in self.blocks1:
            parts = [b(x) for b in self.blocks1]
        x = torch.cat(parts, dim=1)
        x = self.pool1(x)
        
        parts = [b(x) for b in self.blocks2]
        x = torch.cat(parts, dim=1)
        x = self.pool2(x)
        
        parts = [b(x) for b in self.blocks3]
        x = torch.cat(parts, dim=1)
        
        x = self.gap(x).squeeze(-1)
        return self.fc(x).squeeze(-1)


class BiLSTM(nn.Module):
    """Bidirectional LSTM with attention pooling."""
    def __init__(self, c, t, hidden=128, num_layers=2, drop=0.5):
        super().__init__()
        self.input_proj = nn.Conv1d(c, hidden, 1)  # 1x1 conv to project channels
        self.lstm = nn.LSTM(hidden, hidden, num_layers=num_layers, 
                           bidirectional=True, batch_first=True,
                           dropout=drop if num_layers>1 else 0)
        self.attention = nn.Sequential(
            nn.Linear(hidden*2, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1)
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden*2, 128),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(128, 1)
        )
        
    def forward(self, x):
        # x: (B, C, T) -> project channels -> (B, H, T) -> LSTM wants (B, T, H)
        x = self.input_proj(x)  # (B, H, T)
        x = x.permute(0, 2, 1)  # (B, T, H)
        out, _ = self.lstm(x)   # (B, T, 2H)
        
        # Attention pooling
        attn = self.attention(out).squeeze(-1)  # (B, T)
        attn = F.softmax(attn, dim=1).unsqueeze(-1)  # (B, T, 1)
        context = (out * attn).sum(dim=1)  # (B, 2H)
        
        return self.fc(context).squeeze(-1)


class TCN(nn.Module):
    """Temporal Convolutional Network (dilated causal conv)."""
    def __init__(self, c, t, channels=[64, 128, 256], drop=0.5):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(c)
        
        layers = []
        prev = c
        for ch in channels:
            dilation = 2 ** (channels.index(ch) % 4)
            padding = (3-1) * dilation  # kernel=3
            layers.extend([
                nn.Conv1d(prev, ch, 3, padding=padding, dilation=dilation),
                nn.BatchNorm1d(ch),
                nn.GELU(),
                nn.Dropout(drop),
                nn.Conv1d(ch, ch, 3, padding=padding, dilation=dilation),
                nn.BatchNorm1d(ch),
                nn.GELU(),
                nn.Dropout(drop),
            ])
            prev = ch
        self.net = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels[-1], 128),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(128, 1)
        )
        
    def forward(self, x):
        x = self.input_bn(x)
        x = self.net(x)
        x = self.gap(x).squeeze(-1)
        return self.fc(x).squeeze(-1)


# ========== Training / Evaluation ==========

def eval_model(m, X, bs=4096):
    m.eval()
    pv = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = torch.from_numpy(np.ascontiguousarray(X[i:i+bs]))
            pv.append(torch.sigmoid(m(xb.to(device))).cpu().numpy())
    return np.concatenate(pv)

def train_one(m, Xtr, ytr, Xes, yes, epochs=25, lr=5e-4, wd=0.06, 
              bs=512, pat=7, smooth=0.15, jitter=0.0, grad_clip=1.0):
    m = m.to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    
    best = 0.0; best_state = None; ni = 0; best_ep = 0
    for ep in range(epochs):
        m.train()
        idx = np.random.permutation(len(Xtr))
        tl = 0.0; nb = 0
        
        for i in range(0, len(idx), bs):
            bi = idx[i:i+bs]
            xb = torch.from_numpy(np.ascontiguousarray(Xtr[bi].copy()))
            yb = torch.from_numpy(ytr[bi]).float().to(device)
            
            # Label smoothing
            if smooth > 0:
                yb = yb * (1 - smooth) + 0.5 * smooth
            
            xb = xb.to(device)
            loss = F.binary_cross_entropy_with_logits(m(xb), yb)
            
            opt.zero_grad()
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(m.parameters(), grad_clip)
            opt.step()
            tl += loss.item(); nb += 1
        
        sched.step()
        
        pv_es = eval_model(m, Xes)
        auc_es = roc_auc_score(yes, pv_es)
        
        if auc_es > best + 1e-5:
            best = auc_es
            best_state = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
            ni = 0; best_ep = ep + 1
        else:
            ni += 1
            if ni >= pat:
                break
    
    if best_state:
        m.load_state_dict(best_state)
        m = m.to(device)
    
    return eval_model(m, X_te), pv_es, best, best_ep


def rank_agg(pvs):
    R = np.zeros((len(pvs), len(pvs[0])), dtype=np.float64)
    for i, p in enumerate(pvs):
        p = np.nan_to_num(p, nan=0.5)
        R[i] = np.argsort(np.argsort(p)).astype(np.float64) / (len(p) - 1)
    return R.mean(0).astype(np.float32)


# ========== Experiment Configs ==========

architectures = [
    # (name, build_fn, kwargs)
    ('ResMLP_big', lambda: ResMLP(C, T, [1024, 512, 256], drop=0.6), {}),
    ('ResMLP_huge', lambda: ResMLP(C, T, [2048, 1024, 512, 256], drop=0.65), {}),
    ('CNN1D_multi', lambda: CNN1D(C, T, channels=[64, 128, 256], kernels=[3,5,7], drop=0.5), {}),
    ('InceptionTime', lambda: InceptionTime(C, T, n_filters=32, drop=0.5), {}),
    ('BiLSTM_attn', lambda: BiLSTM(C, T, hidden=128, num_layers=2, drop=0.5), {}),
    ('TCN_dilated', lambda: TCN(C, T, channels=[64, 128, 256], drop=0.5), {}),
]

training_configs = [
    dict(lr=5e-4, wd=0.06, bs=512, epochs=30, pat=8, smooth=0.15),
    dict(lr=3e-4, wd=0.08, bs=256, epochs=30, pat=8, smooth=0.15),
]

all_results = {}
all_te_pvs = {}  # For later correlation

print(f"\n{'='*70}", flush=True)
print(f"SYSTEMATIC NN SEARCH: {len(architectures)} archs × {len(training_configs)} train_cfgs × 5 seeds", flush=True)
print(f"{'='*70}", flush=True)

for arch_name, build_fn, _ in architectures:
    for tcfg_idx, tcfg in enumerate(training_configs):
        exp_name = f"{arch_name}_t{tcfg_idx}"
        print(f"\n{'='*70}", flush=True)
        print(f"Experiment: {exp_name}", flush=True)
        print(f"{'='*70}", flush=True)
        
        pvs_te = []; pvs_es = []
        for seed_idx, seed in enumerate([42, 49, 56, 63, 70]):
            torch.manual_seed(seed); np.random.seed(seed)
            
            m = build_fn()
            total_params = sum(p.numel() for p in m.parameters())
            if seed_idx == 0:
                print(f"  Params: {total_params:,}", flush=True)
            
            try:
                pv_t, pv_e, best_es, best_ep = train_one(
                    m, X_tr, y_tr, X_es, y_es, 
                    epochs=tcfg['epochs'], lr=tcfg['lr'], wd=tcfg['wd'],
                    bs=tcfg['bs'], pat=tcfg['pat'], smooth=tcfg['smooth']
                )
                pvs_te.append(pv_t); pvs_es.append(pv_e)
                auc_t = roc_auc_score(y_te, pv_t)
                if seed_idx == 0:
                    print(f"  Seed 0: best_ep={best_ep} ES={best_es:.4f} TE={auc_t:.4f}", flush=True)
            except RuntimeError as e:
                print(f"  Seed {seed}: OOM or error: {e}", flush=True)
                break
        
        if len(pvs_te) >= 3:
            pv_agg = rank_agg(pvs_te)
            pv_es_agg = rank_agg(pvs_es)
            auc_te = roc_auc_score(y_te, pv_agg)
            auc_es = roc_auc_score(y_es, pv_es_agg)
            
            DAYS = (ts_te[-1] - ts_te[0]) / 86400.0
            top1k = max(1, int(len(pv_agg) * 0.01))
            acc1 = y_te[np.argsort(-pv_agg)[:top1k]].mean() * 100
            tpd = top1k / DAYS
            
            print(f"\n  ★ {exp_name}: ES={auc_es:.4f} TE={auc_te:.4f}", flush=True)
            print(f"    top-1%: acc={acc1:.1f}% tpd≈{tpd:.1f}", flush=True)
            
            all_results[exp_name] = {'auc_te': auc_te, 'auc_es': auc_es, 'acc1': acc1, 'tpd': tpd}
            all_te_pvs[exp_name] = pv_agg
        else:
            print(f"  FAILED: only {len(pvs_te)} seeds completed", flush=True)
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# ========== Summary ==========
print(f"\n{'='*70}", flush=True)
print(f"📊 FINAL RANKING (by TE AUC)", flush=True)
print(f"{'='*70}", flush=True)

sorted_results = sorted(all_results.items(), key=lambda x: -x[1]['auc_te'])
print(f"  {'Rank':>4s} {'Experiment':<30s} {'AUC':>8s} {'top1%':>8s} {'tpd':>6s}", flush=True)
print(f"  {'-'*56}", flush=True)
for i, (name, r) in enumerate(sorted_results):
    print(f"  {i+1:>4d} {name:<30s} {r['auc_te']:.4f}   {r['acc1']:.1f}%   {r['tpd']:.1f}", flush=True)

# Save all predictions for later correlation analysis
best_name = sorted_results[0][0]
print(f"\n  🏆 Best: {best_name} AUC={sorted_results[0][1]['auc_te']:.4f}", flush=True)

np.savez('/workspace/models_saved/nn_all_preds.npz', 
         **{k: v for k, v in all_te_pvs.items()},
         y_te=y_te, ts_te=ts_te)
print(f"\nSaved {len(all_te_pvs)} NN predictions for stacking analysis", flush=True)
print(f"TOTAL TIME: {time.time()-t0:.0f}s ({(time.time()-t0)/60:.1f} min)", flush=True)
