"""Proper sequence model training: TCN, LSTM+Attention, on seq_manual (67ch) + seq_v6 (14ch).

Core design principle:
  Same features as Tree (67 hand-crafted) → fair comparison
  Sequence architectures (TCN/LSTM) → capture temporal patterns within features that Tree can't
  5 seeds each → robust ensemble
  Rank aggregation → matches Tree ensemble method
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

DEVICE = 'cpu'  # No GPU, CPU is fine with small batch

# ============ ARCHITECTURES ============

class Chomp1d(nn.Module):
    """Ensure causal (no lookahead): chop off future positions."""
    def __init__(self, chomp_size):
        super().__init__(); self.chomp_size = chomp_size
    def forward(self, x):
        if self.chomp_size == 0: return x
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    """Dilated causal conv block with residual."""
    def __init__(self, n_in, n_out, kernel, dilation, dropout=0.2):
        super().__init__()
        padding = (kernel - 1) * dilation
        self.conv1 = nn.Conv1d(n_in, n_out, kernel, dilation=dilation, padding=padding)
        self.chomp1 = Chomp1d(padding)
        self.bn1 = nn.BatchNorm1d(n_out)
        self.drop1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(n_out, n_out, kernel, dilation=dilation, padding=padding)
        self.chomp2 = Chomp1d(padding)
        self.bn2 = nn.BatchNorm1d(n_out)
        self.drop2 = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.chomp1, self.bn1, nn.GELU(), self.drop1,
                                 self.conv2, self.chomp2, self.bn2, nn.GELU(), self.drop2)
        self.downsample = nn.Conv1d(n_in, n_out, 1) if n_in != n_out else None
        self.out = nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.out(out + res)

class TCN(nn.Module):
    """Temporal Convolutional Network with dilated causal convs."""
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
            nn.AdaptiveAvgPool1d(1),  # Global avg pool over time
            nn.Flatten(),
            nn.Linear(channels[-1], channels[-1]//2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels[-1]//2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, C, T) - conv1d expects (batch, channels, time)
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
        # x: (B, T, C) for LSTM with batch_first=True
        lstm_out, _ = self.lstm(x)  # (B, T, hidden*2)
        attn_w = F.softmax(self.attn(lstm_out), dim=1)  # (B, T, 1)
        context = torch.sum(attn_w * lstm_out, dim=1)   # (B, hidden*2)
        return self.head(context).squeeze(-1)

class TransformerMini(nn.Module):
    """Tiny Transformer encoder - for thorough comparison."""
    def __init__(self, cin, dmodel=128, heads=4, layers=2, dropout=0.3):
        super().__init__()
        self.in_proj = nn.Linear(cin, dmodel)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dmodel, nhead=heads, dim_feedforward=dmodel*2,
            dropout=dropout, batch_first=True, activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.head = nn.Sequential(
            nn.Linear(dmodel, dmodel//2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dmodel//2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, T, C)
        enc = self.encoder(self.in_proj(x))  # (B, T, dmodel)
        pooled = enc.mean(dim=1)  # global avg pool
        return self.head(pooled).squeeze(-1)

# ============ TRAINING ============

def make_dataloaders(X_tr, y_tr, X_es, y_es, X_te, y_te, bs=256):
    """Convert numpy (N, T, C) to torch, make dataloaders."""
    # Determine input format per model
    tr_ds = TensorDataset(torch.from_numpy(X_tr.astype(np.float32)),
                          torch.from_numpy(y_tr.astype(np.float32)))
    es_ds = TensorDataset(torch.from_numpy(X_es.astype(np.float32)),
                          torch.from_numpy(y_es.astype(np.float32)))
    te_ds = TensorDataset(torch.from_numpy(X_te.astype(np.float32)),
                          torch.from_numpy(y_te.astype(np.float32)))
    tr_dl = DataLoader(tr_ds, batch_size=bs, shuffle=True, num_workers=0)
    es_dl = DataLoader(es_ds, batch_size=bs, shuffle=False, num_workers=0)
    te_dl = DataLoader(te_ds, batch_size=bs, shuffle=False, num_workers=0)
    return tr_dl, es_dl, te_dl

def train_one(model, tr_dl, es_dl, epochs=20, lr=3e-4, wd=1e-4, seed=42,
              label_smooth=0.1, pos_weight=None, verbose=True):
    """Train a binary classifier with BCE + optional pos_weight."""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr*0.1)

    best_auc, best_state, patience = 0.0, None, 0
    patience_limit = 7

    for ep in range(epochs):
        model.train(); losses = []
        for xb, yb in tr_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            # Label smoothing
            if label_smooth > 0:
                yb = yb * (1 - label_smooth) + 0.5 * label_smooth
            opt.zero_grad()
            pred = model(xb)
            loss = F.binary_cross_entropy(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); losses.append(loss.item())

        # Eval on ES
        model.eval(); preds, tgts = [], []
        with torch.no_grad():
            for xb, yb in es_dl:
                pred = model(xb.to(DEVICE)).cpu().numpy()
                preds.append(pred); tgts.append(yb.numpy())
        preds, tgts = np.concatenate(preds), np.concatenate(tgts)
        auc = roc_auc_score(tgts, preds)
        sch.step()

        if auc > best_auc:
            best_auc, patience = auc, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1

        if verbose:
            print(f"  ep{ep+1:2d}: loss={np.mean(losses):.4f} es_auc={auc:.4f} "
                  f"best={best_auc:.4f} {'✅' if auc==best_auc else ''}", flush=True)

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

# ============ MAIN ============

def run_experiment(data_path, model_class, model_kwargs, name, n_seeds=5):
    """Train n_seeds models, rank-aggregate test predictions."""
    print(f"\n{'='*60}", flush=True)
    print(f"🚀 {name} on {data_path}", flush=True)
    print(f"{'='*60}", flush=True)

    d = np.load(data_path)
    X_tr, y_tr = d['X_tr'], d['y_tr']
    X_es, y_es = d['X_es'], d['y_es']
    X_te, y_te = d['X_te'], d['y_te']
    ts_te = d['ts_te'] if 'ts_te' in d.files else np.zeros(len(y_te), dtype=np.int64)

    # seq files save as (N, C, T) — conv1d layout
    N, C, T = X_tr.shape
    print(f"  Data: C={C} T={T} tr={len(X_tr):,} es={len(X_es):,} te={len(X_te):,}", flush=True)

    # Model-specific input shape
    if model_class == TCN:
        # TCN expects (N, C, T) — conv1d layout, already correct!
        X_tr_m = X_tr.astype(np.float32)
        X_es_m = X_es.astype(np.float32)
        X_te_m = X_te.astype(np.float32)
    else:
        # LSTM / Transformer expect (N, T, C) — transpose
        X_tr_m = X_tr.transpose(0, 2, 1).astype(np.float32)
        X_es_m = X_es.transpose(0, 2, 1).astype(np.float32)
        X_te_m = X_te.transpose(0, 2, 1).astype(np.float32)

    tr_dl, es_dl, te_dl = make_dataloaders(X_tr_m, y_tr, X_es_m, y_es, X_te_m, y_te, bs=256)

    seed_aucs, all_preds = [], []
    for seed in range(42, 42 + n_seeds):
        print(f"\n  --- seed={seed} ---", flush=True)
        model = model_class(cin=C, **model_kwargs).to(DEVICE)
        total = sum(p.numel() for p in model.parameters())
        print(f"  params: {total:,}", flush=True)
        best_auc = train_one(model, tr_dl, es_dl, epochs=20, lr=3e-4, wd=1e-4,
                            seed=seed, label_smooth=0.08)
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
    print(f"  🏆 {name} RESULTS:", flush=True)
    print(f"  Seed AUCs: {[f'{a:.4f}' for a in seed_aucs]}", flush=True)
    print(f"  Mean±std:  {np.mean(seed_aucs):.4f}±{np.std(seed_aucs):.4f}", flush=True)
    print(f"  Rank-Ens TE AUC: {ens_auc:.4f}", flush=True)

    # Save
    out = f'/workspace/models_saved/nn_{name}.npz'
    np.savez(out, ts=ts_te, y=y_te, pv=rank_ens, seeds_auc=np.array(seed_aucs))
    print(f"  Saved → {out}", flush=True)

    return ens_auc, rank_ens, y_te, ts_te

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='path to seq .npz')
    ap.add_argument('--model', required=True, choices=['tcn','lstm','trans'])
    ap.add_argument('--seeds', type=int, default=5)
    a = ap.parse_args()

    models = {
        'tcn':       (TCN,         dict(channels=[64,128,128,64], kernel=3, dropout=0.3)),
        'lstm':      (LSTMAttn,    dict(hidden=128, layers=2, dropout=0.3)),
        'trans':     (TransformerMini, dict(dmodel=128, heads=4, layers=2, dropout=0.3)),
    }
    cls, kw = models[a.model]
    tag = f"{a.model}_{os.path.basename(a.data).replace('.npz','')}"
    run_experiment(a.data, cls, kw, tag, n_seeds=a.seeds)
