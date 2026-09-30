"""Non-tree NN training: flatten-MLP (67D+BTC) + seq-CNN/Transformer (mmap safe).

Memory-efficient: uses np.load with mmap for seq data, subsamples.
Goal: push NN AUC above 0.54, top-1% above 62%.
"""
import sys, os, time, gc, random, math
sys.path.insert(0, '/workspace')
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import polars as pl, config
from datetime import datetime, timezone
from sklearn.metrics import roc_auc_score

DEVICE = torch.device('cpu')
BS = 1024; SEQ_LEN = 64; N_SEEDS = 5; H = 15
log = lambda m: print(m, flush=True)

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)

def ts_to_dt(ts): return datetime.fromtimestamp(ts, tz=timezone.utc)

def load_flatten_data():
    """Load ETH 67D + BTC cross-asset → 70+D features."""
    t0 = time.time()
    ds = pl.read_parquet(f'{config.DS_DIR}/ds_ETH_h{H}.parquet')
    btc = pl.read_parquet(f'{config.DS_DIR}/ds_BTC_h{H}.parquet')
    btc_feats = [c for c in btc.columns if c not in ['ts','label','soft_label','ret_future']]
    btc = btc.select(['ts'] + btc_feats).rename({c: f'btc_{c}' for c in btc_feats})
    ds = ds.join(btc, on='ts', how='inner').drop_nulls()
    ts = ds['ts'].to_numpy().astype(np.int64)
    y = ds['label'].to_numpy().astype(np.int64)
    retf = ds['ret_future'].to_numpy().astype(np.float32)
    feat_cols = [c for c in ds.columns if c not in ['ts','label','soft_label','ret_future']]
    X = ds.select(feat_cols).to_numpy().astype(np.float32)
    log(f'ETH+BTC X={X.shape}, t={time.time()-t0:.1f}s')
    return ts, X, y, retf, feat_cols

def split_idx(ts):
    s_tr = int(datetime.strptime('2020-01-01','%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp())
    e_tr = int(datetime.strptime('2024-06-30','%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp())
    e_es = int(datetime.strptime('2024-09-30','%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp())
    e_mv = int(datetime.strptime('2025-09-30','%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp())
    m_tr = (ts >= s_tr) & (ts < e_tr)
    m_es = (ts >= e_tr) & (ts < e_es)
    m_mv = (ts >= e_es) & (ts < e_mv)
    m_te = ts >= e_mv
    return m_tr, m_es, m_mv, m_te

def load_seq_mmap():
    """Load seq_raw.npz with mmap (memory safe)."""
    t0 = time.time()
    d = np.load('/workspace/data/seq_raw.npz', mmap_mode='r')
    anchor_ts = d['anchor_ts']
    seqs = np.stack([d['lr'], d['h'], d['l'], d['c'], d['bv']], axis=1)  # [M,5,64]
    log(f'seq mmap: {seqs.shape}, t={time.time()-t0:.1f}s')
    return seqs, anchor_ts

def build_seq_labels(seqs_anchor_ts, full_ts, full_y, full_retf):
    """Map seq anchors to labels (no anchor gaps in our data except 15 tiny ones)."""
    idx_map = {int(t): i for i,t in enumerate(full_ts)}
    labels = np.full(len(seqs_anchor_ts), -1, dtype=np.int64)
    retfs = np.zeros(len(seqs_anchor_ts), dtype=np.float32)
    valid = np.zeros(len(seqs_anchor_ts), dtype=bool)
    for i, t in enumerate(seqs_anchor_ts):
        key = int(t)
        if key in idx_map:
            labels[i] = full_y[idx_map[key]]
            retfs[i] = full_retf[idx_map[key]]
            valid[i] = True
    return labels, retfs, valid

# ===== Models =====
class MLP(nn.Module):
    def __init__(self, d_in, widths=(2048,1024,512,256), dropout=0.6):
        super().__init__()
        layers = []; prev = d_in
        for w in widths:
            layers += [nn.Linear(prev,w), nn.LayerNorm(w), nn.GELU(), nn.Dropout(dropout)]
            prev = w
        layers.append(nn.Linear(prev,1))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x).squeeze(-1)

class SeqCNN(nn.Module):
    def __init__(self, d_feat=5, d_model=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(d_feat, 64, 7, padding=3), nn.LayerNorm([64, SEQ_LEN]), nn.GELU(),
            nn.Conv1d(64, 128, 5, padding=2), nn.LayerNorm([128, SEQ_LEN]), nn.GELU(),
            nn.Conv1d(128, d_model, 3, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(d_model, 256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, 1),
        )
    def forward(self, x): return self.conv(x).squeeze(-1)

class SeqTransformer(nn.Module):
    def __init__(self, d_feat=5, d_model=128, nhead=4, nlayer=3):
        super().__init__()
        self.in_proj = nn.Conv1d(d_feat, d_model, 1)
        encoder = nn.TransformerEncoderLayer(d_model, nhead, 256, 0.2, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(encoder, nlayer)
        self.head = nn.Sequential(nn.Linear(d_model, 256), nn.GELU(), nn.Dropout(0.3), nn.Linear(256,1))
    def forward(self, x):
        x = self.in_proj(x).transpose(1,2)
        x = self.enc(x)
        x = x.mean(dim=1)
        return self.head(x).squeeze(-1)

# ===== Train =====
def train(model, X_tr, y_tr, X_es, y_es, epochs=40, lr=3e-4, wd=0.05, label_smooth=0.1):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_auc, best_pat, best_state = 0.0, 0, None
    N = len(y_tr); perm = np.random.permutation(N)
    X_t = torch.from_numpy(X_tr); y_t = torch.from_numpy(y_tr).float()
    X_e = torch.from_numpy(X_es); y_e = torch.from_numpy(y_es).float()
    for ep in range(epochs):
        model.train()
        for i in range(0, N, BS):
            idx = perm[i:i+BS]
            xb, yb = X_t[idx], y_t[idx]
            logits = model(xb)
            loss = F.binary_cross_entropy_with_logits(logits, yb*(1-label_smooth) + 0.5*label_smooth)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            pv = torch.sigmoid(model(X_e[:100000])); auc = roc_auc_score(y_e[:100000].numpy(), pv.numpy())
        if auc > best_auc:
            best_auc, best_pat = auc, 0
            best_state = {k:v.clone() for k,v in model.state_dict().items()}
        else:
            best_pat += 1
        if best_pat >= 8: break
    if best_state: model.load_state_dict(best_state)
    return best_auc

def predict(model, X, bs=4096):
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            out.append(torch.sigmoid(model(torch.from_numpy(X[i:i+bs]))).numpy())
    return np.concatenate(out)

def eval_rolling(ts, pv, y, w_days=30, ql=0.99, qs=0.01):
    t = ts.astype(np.float64)/86400.0
    s, e = t.min(), t.max()
    tw, tl, ta = [], [], []
    while e - s > w_days:
        m = (t >= s) & (t < s + w_days)
        if m.sum() < 100: s += w_days; continue
        pv_w, y_w = pv[m], y[m]
        lo, hi = np.quantile(pv_w, ql), np.quantile(pv_w, 1-qs)
        long_sig = pv_w > lo; short_sig = pv_w < hi
        longs = y_w[long_sig]; shorts = y_w[short_sig]
        n = len(longs) + len(shorts)
        if n > 0:
            acc = (longs.mean() + (1-shorts.mean()))/2*100
            tw.append(n/w_days); ta.append(acc)
        s += w_days
    return (np.mean(tw), np.mean(ta)) if tw else (0,0)

# ===== Main =====
def main():
    t0 = time.time()
    log("="*60)
    log("🔥 NON-TREE ULTIMATE: MLP + CNN + Transformer")
    log("="*60)

    # ---- FLATTEN DATA ----
    ts_f, X_f, y_f, retf_f, cols = load_flatten_data()
    m_tr, m_es, m_mv, m_te = split_idx(ts_f)
    log(f"  tr={m_tr.sum():,} es={m_es.sum():,} te={m_te.sum():,}")

    # Standardize
    mu = X_f[m_tr].mean(0); sd = X_f[m_tr].std(0)+1e-6
    Xfn = (X_f - mu)/sd
    del X_f; gc.collect()

    # ---- SEQ DATA (mmap) ----
    log("\nLoading seq mmap...")
    seqs_mm, anchor_ts = load_seq_mmap()
    log(f"  mapping seq anchors to labels...")
    idx_map = {int(t): i for i,t in enumerate(ts_f)}
    valid = np.array([int(t) in idx_map for t in anchor_ts])
    seq_y = np.array([y_f[idx_map[int(t)]] if int(t) in idx_map else -1 for t in anchor_ts])
    seq_retf = np.array([retf_f[idx_map[int(t)]] if int(t) in idx_map else 0.0 for t in anchor_ts])
    log(f"  valid seq anchors: {valid.sum():,}/{len(valid):,}")

    def seq_split(anchor_ts):
        s_tr = int(datetime.strptime('2020-01-01','%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp())
        e_tr = int(datetime.strptime('2024-06-30','%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp())
        e_es = int(datetime.strptime('2024-09-30','%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp())
        e_te = int(datetime.strptime('2025-09-30','%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp())
        mtr = (anchor_ts >= s_tr) & (anchor_ts < e_tr)
        mes = (anchor_ts >= e_tr) & (anchor_ts < e_es)
        mte = anchor_ts >= e_te
        return mtr, mes, mte
    sm_tr, sm_es, sm_te = seq_split(anchor_ts)

    # For seq: we need to physically load the seqs (not mmap) for training subsample
    # Load train seq chunk
    log(f"  loading train seq (chunk)...")
    mtr_ok = sm_tr & valid
    idx_tr = np.where(mtr_ok)[0]
    # Subsample 150K for speed
    np.random.seed(42)
    idx_tr_sub = np.random.choice(idx_tr, min(150_000, len(idx_tr)), replace=False)
    sq_tr = seqs_mm[idx_tr_sub].astype(np.float32)
    sq_tr_y = seq_y[idx_tr_sub]
    log(f"  sq_tr={sq_tr.shape}")

    # Load ES seq
    mes_ok = sm_es & valid
    sq_es = seqs_mm[mes_ok].astype(np.float32)
    sq_es_y = seq_y[mes_ok]
    log(f"  sq_es={sq_es.shape}")

    # Standardize seq using train stats
    sq_mu = sq_tr.reshape(-1, sq_tr.shape[1]).mean(0)
    sq_sd = sq_tr.reshape(-1, sq_tr.shape[1]).std(0)+1e-6
    sq_trn = (sq_tr - sq_mu.reshape(1,-1,1))/sq_sd.reshape(1,-1,1)
    sq_esn = (sq_es - sq_mu.reshape(1,-1,1))/sq_sd.reshape(1,-1,1)
    del sq_tr, sq_es; gc.collect()

    # FLATTEN: subsample 800K
    np.random.seed(42)
    flat_tr_idx = np.random.choice(np.where(m_tr)[0], min(800_000, m_tr.sum()), replace=False)

    all_te_preds = []

    # ===== TRAIN =====
    log(f"\n🚀 Training {N_SEEDS} seeds × 3 model types...")

    for seed in range(N_SEEDS):
        set_seed(42+seed)
        log(f"\n--- SEED {seed+1}/{N_SEEDS} ---")

        # 1) FLATTEN-MLP (70+D, subsample 800K)
        log(f"  MLP (70+D flatten)...", end=' ', flush=True)
        m_mlp = MLP(Xfn.shape[1], widths=(2048,1024,512), dropout=0.6)
        X_tr_sub = Xfn[flat_tr_idx]; y_tr_sub = y_f[flat_tr_idx]
        auc_tr = train(m_mlp, X_tr_sub, y_tr_sub, Xfn[m_es], y_f[m_es], epochs=40, lr=3e-4, wd=0.06)
        pv_mlp = predict(m_mlp, Xfn[m_te])
        auc_te_mlp = roc_auc_score(y_f[m_te], pv_mlp)
        log(f"ES={auc_tr:.4f} TE={auc_te_mlp:.4f}")
        all_te_preds.append(('MLP', pv_mlp))
        del m_mlp; gc.collect()

        # 2) SEQ-CNN
        log(f"  CNN (seq 5×64)...", end=' ', flush=True)
        set_seed(42+seed)
        m_cnn = SeqCNN(d_feat=5, d_model=128)
        auc_c = train(m_cnn, sq_trn, sq_tr_y, sq_esn, sq_es_y, epochs=35, lr=5e-4, wd=0.01)
        # Predict on ALL test seq anchors
        log(f"    loading test seq...", end='', flush=True)
        mte_ok = sm_te & valid
        sq_te = seqs_mm[mte_ok].astype(np.float32)
        sq_ten = (sq_te - sq_mu.reshape(1,-1,1))/sq_sd.reshape(1,-1,1)
        log(f" {sq_ten.shape}")
        pv_cnn = predict(m_cnn, sq_ten)
        # Map to full test: anchor_ts[mte_ok] → idx_map → full te positions
        pv_cnn_full = np.full(m_te.sum(), 0.5)
        te_positions = np.where(m_te)[0]
        te_ts_list = [int(ts_f[i]) for i in te_positions]
        pos_in_full = {int(t): i for i,t in enumerate(ts_f)}
        for k, t in enumerate(anchor_ts[mte_ok]):
            it = pos_in_full.get(int(t))
            if it is not None:
                # find position in test split
                te_local = pos_in_full[int(t)] - (ts_f[:pos_in_full[int(t]]+1][m_te[:pos_in_full[int(t]]+1]].sum() if False else 0)
                # simpler approach: build mapping from ts → local te index
                pass
        # Simpler: full map
        te_idx_map = {int(ts_f[i]): local_i for local_i, i in enumerate(te_positions)}
        for k, t in enumerate(anchor_ts[mte_ok]):
            lt = te_idx_map.get(int(t))
            if lt is not None: pv_cnn_full[lt] = pv_cnn[k]
        auc_te_cnn = roc_auc_score(y_f[m_te], pv_cnn_full)
        log(f"  ES={auc_c:.4f} TE={auc_te_cnn:.4f}")
        all_te_preds.append(('CNN', pv_cnn_full))
        del m_cnn, sq_te, sq_ten; gc.collect()

        # 3) SEQ-TRANSFORMER
        log(f"  Transformer (seq 5×64)...", end=' ', flush=True)
        set_seed(42+seed)
        m_trans = SeqTransformer(d_feat=5, d_model=128, nhead=4, nlayer=3)
        auc_t = train(m_trans, sq_trn, sq_tr_y, sq_esn, sq_es_y, epochs=30, lr=5e-4, wd=0.01)
        sq_te2 = seqs_mm[mte_ok].astype(np.float32)
        sq_ten2 = (sq_te2 - sq_mu.reshape(1,-1,1))/sq_sd.reshape(1,-1,1)
        pv_trans = predict(m_trans, sq_ten2)
        pv_trans_full = np.full(m_te.sum(), 0.5)
        for k, t in enumerate(anchor_ts[mte_ok]):
            lt = te_idx_map.get(int(t))
            if lt is not None: pv_trans_full[lt] = pv_trans[k]
        auc_te_trans = roc_auc_score(y_f[m_te], pv_trans_full)
        log(f"  ES={auc_t:.4f} TE={auc_te_trans:.4f}")
        all_te_preds.append(('TRANS', pv_trans_full))
        del m_trans, sq_te2, sq_ten2; gc.collect()

    # ===== EVALUATE =====
    log(f"\n{'='*60}")
    log(f"📊 TEST EVALUATION")
    log(f"{'='*60}")

    ts_te = ts_f[m_te]; y_te = y_f[m_te]
    te_days = (ts_te[-1] - ts_te[0])/86400.0
    def rank(a): return np.argsort(np.argsort(a)).astype(np.float64)/len(a)

    def te_report(pv, name):
        auc = roc_auc_score(y_te, pv)
        for pct in [0.5, 1.0, 1.5, 2.0]:
            k = max(1,int(len(pv)*pct/100))
            a = y_te[np.argsort(-pv)[:k]].mean()*100; t = k/te_days
            log(f"  {name:<25s} top-{pct:.1f}%: acc={a:.1f}% tpd={t:.1f}")
        tpd, acc = eval_rolling(ts_te, pv, y_te, 30, 0.99, 0.01)
        log(f"  {name:<25s} rolling_q: tpd={tpd:.1f} acc={acc:.1f}% AUC={auc:.4f}")
        return auc, tpd, acc

    # Individual
    for mtype in ['MLP', 'CNN', 'TRANS']:
        preds = [p for n,p in all_te_preds if n==mtype]
        if preds:
            r = np.mean([rank(p) for p in preds], axis=0)
            te_report(r, f"ENSEMBLE-{mtype}(5)")

    # All 15 models
    all_ranks = [rank(p) for _,p in all_te_preds]
    full = np.mean(all_ranks, axis=0)
    auc_full, tpd_full, acc_full = te_report(full, "FULL-ENSEMBLE(15)")

    # Save best
    np.savez('/workspace/models_saved/nn_ultimate.npz',
             pv=full, y=y_te, ts=ts_te, auc=auc_full, tpd=tpd_full, acc=acc_full)

    log(f"\n⏱ Total: {time.time()-t0:.0f}s")
    log(f"\n🎯 BEST: AUC={auc_full:.4f} rolling_acc={acc_full:.1f}% tpd={tpd_full:.1f}")

if __name__ == '__main__':
    main()
