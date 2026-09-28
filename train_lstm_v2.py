"""Stronger LSTM+Attention for seq_manual (67 channels) — FAIR comparison with Tree.

Design improvements over train_seq_models.py:
  - 2-layer bidirectional LSTM with larger hidden (192 → 384)
  - Attention pooling (not just mean/last)
  - Label smoothing + weight decay tuned for small NN
  - 3 seeds (faster but still meaningful)
  - Comprehensive evaluation including top-1% acc comparison with Tree
"""
import sys; sys.stdout.reconfigure(line_buffering=True)
import os, time, gc, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, accuracy_score

torch.set_float32_matmul_precision('high')
DEVICE = 'cpu'

class LSTMAttnV2(nn.Module):
    """BiLSTM + multi-head time-step attention + confident head."""
    def __init__(self, cin, hidden=192, layers=2, dropout=0.35):
        super().__init__()
        self.lstm = nn.LSTM(
            cin, hidden, layers, batch_first=True,
            dropout=dropout if layers > 1 else 0, bidirectional=True
        )
        attn_dim = hidden * 2
        self.attn = nn.Sequential(
            nn.Linear(attn_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1)
        )
        self.gate = nn.Sequential(
            nn.Linear(attn_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, T, C)
        out, _ = self.lstm(x)           # (B, T, H*2)
        w = F.softmax(self.attn(out), dim=1)  # (B, T, 1)
        ctx = torch.sum(w * out, dim=1)       # (B, H*2)
        return self.gate(ctx).squeeze(-1)

class BiLSTMLast(nn.Module):
    """BiLSTM — use last time step only (like traditional LSTM trading)."""
    def __init__(self, cin, hidden=192, layers=2, dropout=0.35):
        super().__init__()
        self.lstm = nn.LSTM(
            cin, hidden, layers, batch_first=True,
            dropout=dropout if layers > 1 else 0, bidirectional=True
        )
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out, _ = self.lstm(x)           # (B, T, H*2)
        return self.head(out[:, -1, :]).squeeze(-1)

def train(model, tr_dl, es_dl, epochs=25, lr=3e-4, wd=1e-4, seed=42, label_smooth=0.1, verbose=True):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr*0.05)
    best_auc, best_state, patience = 0.0, None, 0
    PAT = 8
    for ep in range(epochs):
        model.train(); losses = []
        for xb, yb in tr_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            if label_smooth > 0:
                yb = yb * (1 - label_smooth) + 0.5 * label_smooth
            opt.zero_grad()
            loss = F.binary_cross_entropy(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); losses.append(loss.item())
        model.eval(); preds, tgts = [], []
        with torch.no_grad():
            for xb, yb in es_dl:
                preds.append(model(xb.to(DEVICE)).cpu().numpy())
                tgts.append(yb.numpy())
        auc = roc_auc_score(np.concatenate(tgts), np.concatenate(preds))
        sch.step()
        if auc > best_auc:
            best_auc, patience = auc, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
        if verbose:
            flag = '🏆' if auc == best_auc else ''
            print(f"  ep{ep+1:2d}: loss={np.mean(losses):.4f} es_auc={auc:.4f} best={best_auc:.4f} {flag}", flush=True)
        if patience >= PAT:
            break
    if best_state: model.load_state_dict(best_state)
    return best_auc

def predict(model, dl):
    model.eval(); ps = []
    with torch.no_grad():
        for xb, _ in dl: ps.append(model(xb.to(DEVICE)).cpu().numpy())
    return np.concatenate(ps)

def rank(a): return np.argsort(np.argsort(a)).astype(np.float64)/len(a)

def eval_topk(y, pv, kpcts=[0.5,1.0,1.5,2.0]):
    """Evaluate accuracy at top-k%."""
    out = []
    for p in kpcts:
        k = max(1, int(len(pv)*p/100))
        acc = y[np.argsort(-pv)[:k]].mean()*100
        out.append(f"top-{p:.1f}%={acc:.1f}%")
    return "  ".join(out)

def run(data_path, model_cls, model_kw, name, n_seeds=3, batch=128):
    print(f"\n{'='*60}")
    print(f"🚀 {name} on {data_path}")
    print(f"{'='*60}")
    d = np.load(data_path)
    X_tr, y_tr = d['X_tr'], d['y_tr']
    X_es, y_es = d['X_es'], d['y_es']
    X_te, y_te = d['X_te'], d['y_te']
    ts_te = d['ts_te'] if 'ts_te' in d.files else np.zeros(len(y_te))

    # Input: seq files are (N, C, T) → transpose to (N, T, C) for LSTM
    X_tr = X_tr.transpose(0, 2, 1).astype(np.float32)
    X_es = X_es.transpose(0, 2, 1).astype(np.float32)
    X_te = X_te.transpose(0, 2, 1).astype(np.float32)
    N, T, C = X_tr.shape
    print(f"  Data: C={C} T={T} tr={N:,} es={len(X_es):,} te={len(X_te):,}  pos={y_tr.mean():.3f}/{y_es.mean():.3f}/{y_te.mean():.3f}", flush=True)

    tr_dl = DataLoader(TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr.astype(np.float32))),
                       batch_size=batch, shuffle=True)
    es_dl = DataLoader(TensorDataset(torch.from_numpy(X_es), torch.from_numpy(y_es.astype(np.float32))),
                       batch_size=batch, shuffle=False)
    te_dl = DataLoader(TensorDataset(torch.from_numpy(X_te), torch.from_numpy(y_te.astype(np.float32))),
                       batch_size=batch, shuffle=False)

    seed_aucs, seed_te_aucs, all_preds = [], [], []
    for seed in range(42, 42 + n_seeds):
        print(f"\n  --- seed={seed} ---", flush=True)
        m = model_cls(cin=C, **model_kw).to(DEVICE)
        total = sum(p.numel() for p in m.parameters())
        print(f"  params={total:,}", flush=True)
        best_es_auc = train(m, tr_dl, es_dl, epochs=25, lr=3e-4, wd=1e-4, seed=seed, label_smooth=0.08)
        preds = predict(m, te_dl)
        te_auc = roc_auc_score(y_te, preds)
        seed_aucs.append(best_es_auc)
        seed_te_aucs.append(te_auc)
        all_preds.append(preds)
        print(f"  ES={best_es_auc:.4f} TE={te_auc:.4f}  {eval_topk(y_te, preds)}", flush=True)
        del m, preds; gc.collect()

    ranks = np.array([rank(p) for p in all_preds])
    rank_ens = ranks.mean(axis=0)
    ens_auc = roc_auc_score(y_te, rank_ens)
    print(f"\n  {'='*50}", flush=True)
    print(f"  🏆 {name} ENSEMBLE TE AUC: {ens_auc:.4f}", flush=True)
    print(f"  Individual TE: {[f'{a:.4f}' for a in seed_te_aucs]}", flush=True)
    print(f"  Mean±std: {np.mean(seed_te_aucs):.4f}±{np.std(seed_te_aucs):.4f}", flush=True)
    print(f"  Top-k: {eval_topk(y_te, rank_ens)}", flush=True)

    out = f'/workspace/models_saved/lstm_{name}.npz'
    np.savez(out, ts=ts_te, y=y_te, pv=rank_ens, seed_aucs=np.array(seed_aucs), seed_te_aucs=np.array(seed_te_aucs))
    print(f"  Saved → {out}", flush=True)
    return ens_auc, rank_ens, y_te, ts_te

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--model', default='attn', choices=['attn', 'last'])
    a = ap.parse_args()

    cls = LSTMAttnV2 if a.model == 'attn' else BiLSTMLast
    tag = f"{a.model}_{os.path.basename(a.data).replace('.npz','')}"
    run(a.data, cls, dict(hidden=192, layers=2, dropout=0.35), tag, n_seeds=a.seeds, batch=a.batch)
