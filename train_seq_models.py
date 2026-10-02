"""Sequence model training v2: TCN + LSTM, aligned with MLP best practices.

Key differences from v1:
  - Full 1.18M train (stride=2, no |ret| filter)
  - label_smooth=0.05 (MLP best), wd=1e-3 (MLP best), epochs=50, patience=10
  - batch=512 (faster CPU training)
  - data loaded as float16 → cast to float32 on dataloader creation
"""
import sys; sys.stdout.reconfigure(line_buffering=True)
import os, time, gc, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score

torch.set_float32_matmul_precision('high')
DEVICE = 'cpu'

# ============ ARCHITECTURES ============

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__(); self.chomp_size = chomp_size
    def forward(self, x):
        if self.chomp_size == 0: return x
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    """Dilated causal conv with LayerNorm (faster on CPU than BatchNorm)."""
    def __init__(self, n_in, n_out, kernel, dilation, dropout=0.3):
        super().__init__()
        padding = (kernel - 1) * dilation
        self.conv1 = nn.Conv1d(n_in, n_out, kernel, dilation=dilation, padding=padding)
        self.chomp1 = Chomp1d(padding)
        self.ln1 = nn.GroupNorm(1, n_out)  # GroupNorm(1, C) = LayerNorm per channel
        self.drop1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(n_out, n_out, kernel, dilation=dilation, padding=padding)
        self.chomp2 = Chomp1d(padding)
        self.ln2 = nn.GroupNorm(1, n_out)
        self.drop2 = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.chomp1, self.ln1, nn.GELU(), self.drop1,
                                 self.conv2, self.chomp2, self.ln2, nn.GELU(), self.drop2)
        self.downsample = nn.Conv1d(n_in, n_out, 1) if n_in != n_out else None
        self.out = nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.out(out + res)

class TCN(nn.Module):
    def __init__(self, cin, channels=[64, 128, 128, 64], kernel=3, dropout=0.3):
        super().__init__()
        layers = []
        in_ch = cin
        for i, out_ch in enumerate(channels):
            dil = 2 ** i
            layers.append(TemporalBlock(in_ch, out_ch, kernel, dil, dropout))
            in_ch = out_ch
        self.network = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels[-1], channels[-1]//2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels[-1]//2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.head(self.network(x)).squeeze(-1)

class LSTMAttn(nn.Module):
    """LSTM with time-step attention."""
    def __init__(self, cin, hidden=128, layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(cin, hidden, layers, batch_first=True,
                           dropout=dropout if layers>1 else 0, bidirectional=True)
        self.attn = nn.Sequential(
            nn.Linear(hidden*2, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1)
        )
        self.head = nn.Sequential(
            nn.Linear(hidden*2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_w = F.softmax(self.attn(lstm_out), dim=1)
        context = torch.sum(attn_w * lstm_out, dim=1)
        return self.head(context).squeeze(-1)

# ============ CUSTOM DATASET (float16 on disk → float32 on __getitem__) ============

class SeqDataset(torch.utils.data.Dataset):
    def __init__(self, X_f16, y):
        # X_f16: numpy float16 array (N, C, T)
        self.X = X_f16
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx].astype(np.float32), float(self.y[idx])

# ============ TRAINING ============

def train_one(model, tr_dl, es_dl, epochs=50, lr=5e-4, wd=1e-3, seed=42,
              label_smooth=0.05, patience_limit=10):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr*0.1)

    best_auc, best_state, patience = 0.0, None, 0
    best_epoch = -1

    for ep in range(epochs):
        model.train(); losses = []
        for xb, yb in tr_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            if label_smooth > 0:
                yb = yb * (1 - label_smooth) + 0.5 * label_smooth
            opt.zero_grad()
            pred = model(xb)
            loss = F.binary_cross_entropy(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); losses.append(loss.item())

        model.eval(); preds, tgts = [], []
        with torch.no_grad():
            for xb, yb in es_dl:
                pred = model(xb.to(DEVICE)).cpu().numpy()
                preds.append(pred); tgts.append(yb.numpy())
        preds, tgts = np.concatenate(preds), np.concatenate(tgts)
        auc = roc_auc_score(tgts, preds)
        sch.step()

        if auc > best_auc:
            best_auc, patience, best_epoch = auc, 0, ep
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1

        print(f"  ep{ep+1:2d}: loss={np.mean(losses):.4f} es_auc={auc:.4f} "
              f"best={best_auc:.4f}@{best_epoch+1}L {'✅' if auc==best_auc else ''}", flush=True)

        if patience >= patience_limit:
            break

    if best_state:
        model.load_state_dict(best_state)
    return best_auc

def predict(model, dl):
    model.eval(); preds = []
    with torch.no_grad():
        for xb, _ in dl:
            preds.append(model(xb.to(DEVICE)).cpu().numpy())
    return np.concatenate(preds)

def run_experiment(data_path, model_class, model_kwargs, name, n_seeds=5, bs=512):
    print(f"\n{'='*60}", flush=True)
    print(f"🚀 {name} on {data_path}", flush=True)
    print(f"{'='*60}", flush=True)
    t0 = time.time()

    d = np.load(data_path)
    # Keep ALL data in float16 — only cast to float32 inside __getitem__
    X_tr, y_tr = d['X_tr'], d['y_tr']
    X_es, y_es = d['X_es'], d['y_es']
    X_te, y_te = d['X_te'], d['y_te']
    ts_te = d['ts_te']

    N, C, T = X_tr.shape
    print(f"  Data: C={C} T={T} tr={len(X_tr):,} es={len(X_es):,} te={len(X_te):,}", flush=True)
    print(f"  Dtype: {X_tr.dtype} ({X_tr.nbytes/1e9:.2f}GB float16)", flush=True)

    # Model-specific input shape handling
    # TCN conv1d wants (N, C, T) — layout already correct
    # LSTM batch_first wants (N, T, C) — SeqDataset.__getitem__ will transpose each sample
    is_lstm = (model_class != TCN)

    class SeqModelDataset(torch.utils.data.Dataset):
        def __init__(self, X_f16, y, transpose_tc=False):
            self.X = X_f16  # (N, C, T) float16
            self.y = y
            self.transpose_tc = transpose_tc
        def __len__(self):
            return len(self.X)
        def __getitem__(self, idx):
            x = self.X[idx].astype(np.float32)  # (C, T)
            if self.transpose_tc:
                x = x.T  # (T, C) for LSTM
            return x, np.float32(self.y[idx])

    tr_ds = SeqModelDataset(X_tr, y_tr, transpose_tc=is_lstm)
    es_ds = SeqModelDataset(X_es, y_es, transpose_tc=is_lstm)
    te_ds = SeqModelDataset(X_te, y_te, transpose_tc=is_lstm)
    tr_dl = DataLoader(tr_ds, batch_size=bs, shuffle=True, num_workers=0)
    es_dl = DataLoader(es_ds, batch_size=bs, shuffle=False, num_workers=0)
    te_dl = DataLoader(te_ds, batch_size=bs, shuffle=False, num_workers=0)

    # Free original data arrays (still in d)
    del X_tr, X_es, X_te; gc.collect()
    print(f"  After del split arrays — peak mem OK (only float16 in d+indexed)", flush=True)

    seed_aucs, all_preds = [], []
    for seed in range(42, 42 + n_seeds):
        print(f"\n  --- seed={seed} ---", flush=True)
        model = model_class(cin=C, **model_kwargs).to(DEVICE)
        total = sum(p.numel() for p in model.parameters())
        print(f"  params: {total:,}", flush=True)
        best_auc = train_one(model, tr_dl, es_dl, epochs=50, lr=5e-4, wd=1e-3,
                            seed=seed, label_smooth=0.05, patience_limit=10)
        seed_aucs.append(best_auc)
        preds = predict(model, te_dl)
        all_preds.append(preds)
        te_auc = roc_auc_score(y_te, preds)
        print(f"  ES AUC={best_auc:.4f}  TE AUC={te_auc:.4f}", flush=True)
        del model, preds; gc.collect()

    # Rank aggregation
    def rank(a): return np.argsort(np.argsort(a)).astype(np.float64)/len(a)
    ranks = np.array([rank(p) for p in all_preds])
    rank_ens = ranks.mean(axis=0)
    ens_auc = roc_auc_score(y_te, rank_ens)

    print(f"\n  {'='*40}", flush=True)
    print(f"  🏆 {name} RESULTS (took {time.time()-t0:.0f}s):", flush=True)
    print(f"  Seed AUCs: {[f'{a:.4f}' for a in seed_aucs]}", flush=True)
    print(f"  Mean±std:  {np.mean(seed_aucs):.4f}±{np.std(seed_aucs):.4f}", flush=True)
    print(f"  Rank-Ens TE AUC: {ens_auc:.4f}", flush=True)

    out = f'/workspace/models_saved/seq_{name}.npz'
    np.savez(out, ts=ts_te, y=y_te, pv=rank_ens, seed_preds=np.stack(all_preds), seeds_auc=np.array(seed_aucs))
    print(f"  Saved → {out}", flush=True)

    return ens_auc, rank_ens, y_te, ts_te

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--model', required=True, choices=['tcn','lstm'])
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--bs', type=int, default=512)
    a = ap.parse_args()

    models = {
        'tcn':  (TCN,      dict(channels=[64, 128, 128, 64], kernel=3, dropout=0.3)),
        'lstm': (LSTMAttn, dict(hidden=128, layers=2, dropout=0.3)),
    }
    cls, kw = models[a.model]
    tag = f"{a.model}_v7"
    run_experiment(a.data, cls, kw, tag, n_seeds=a.seeds, bs=a.bs)
