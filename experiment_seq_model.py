#!/usr/bin/env python3
"""评估"时序模型(LSTM)读原始1minOHLCV窗口 → 30min方向"在当前协议下能否继续提升准确率。

对比口径与 JOINT pool20 基线完全一致:
- 目标: 本项目 label(未来30根1min收盘>锚价), data_store.y()
- 覆盖: 固定 Top1%, R2 逐日滚动阈值(盘前历史99分位), 与 show_monthly.py 同协议
- 切分: train/early_stop/meta_val/test 严格因果; 特征标准化均值/方差只在 train 拟合
- 融合: rank 等权融合(pool20 与 seq), 无参数选择泄露

三档对比:  纯 pool20  |  纯 LSTM  |  pool20+LSTM 融合
证伪或证实"序列模型是否带来正交增量"。

用法:
  python experiment_seq_model.py               # ETH + BTC
  python experiment_seq_model.py BTC
"""
import os, sys, gc, argparse, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from data_store import AssetContext

FAMS = ["lgb", "xgb", "cat"]
P99 = 99.0
WIN_DAYS = 90
SAMP = 1440

WINDOW = 60          # 前瞻窗口(1min 根)：读近 60 根
HIDDEN = 32
LAYERS = 1
# 通道数 = raw_features 通道数: 多尺度收益6 + z-score4 + 波动率2 + 量比1 + 基础形态6 = 19
BATCH = 512
EPOCHS = 30
LR = 1e-3
PATIENCE = 6
TRAIN_CAP = 240_000      # train 拟合子样本(控 CPU 与内存: 240k*60*19*4B≈1.1GB)
ES_CAP = 60_000
SEED = 42


# ---------- 序列特征: 与 GBDT 信息量相当的"丰富多尺度"因果通道 ----------
def _shift(a, k):
    out = np.empty_like(a, dtype=np.float64)
    out[:k] = a[0]
    out[k:] = a[:-k]
    return out


def raw_features(ctx):
    """(N_raw, K) 因果逐通道特征: 多尺度收益 + 价格z-score + 波动率 + 量比 + 基础形态。
    K 与 GBDT 的信息量相当, 给序列模型公平的输入。全为 rolling 回溯(无未来)。"""
    o, h, l, c = ctx.o, ctx.h, ctx.l, ctx.c
    tb, tv = ctx.tb, ctx.vol
    lc = np.log(np.maximum(c, 1e-12))
    cols = {}
    for k in (1, 5, 15, 30, 60, 120):
        cols[f"r{k}"] = lc - _shift(lc, k)
    s = pd.Series(lc)
    for w in (15, 30, 60, 120):
        mu = s.rolling(w).mean().to_numpy()
        sd = s.rolling(w).std().to_numpy()
        cols[f"z{w}"] = np.where(sd > 1e-9, (lc - mu) / np.where(sd > 1e-9, sd, 1.0), 0.0)
    r1 = cols["r1"]
    for w in (15, 60):
        cols[f"vol{w}"] = pd.Series(r1).rolling(w).std().to_numpy()
    tot = (tb.astype(np.float64) + tv.astype(np.float64))
    vma = pd.Series(tot).rolling(60).mean().to_numpy()
    cols["volrel"] = np.where(vma > 1e-9, tot / np.where(vma > 1e-9, vma, 1.0) - 1.0, 0.0)
    base = ctx.raw_channels()                 # lr, body, uw, lw, tbr, cvd
    X = np.concatenate([np.stack(list(cols.values()), 1).astype(np.float32), base], axis=1)
    X[np.isnan(X)] = 0.0
    return X.astype(np.float32)


def build_sequences(X, pos, L):
    """返回 (m, L, C) float32, 每通道在窗口内做 z-score(用末尾L根内的均值/方差)。"""
    m = len(pos); C = X.shape[1]
    out = np.zeros((m, L, C), dtype=np.float32)
    st = pos - (L - 1)
    allok = (st >= 0).all()
    if allok:
        for i in range(C):
            base = np.lib.stride_tricks.as_strided(
                X[:, i], shape=(len(X) - L + 1, L), strides=(X.strides[0], X.strides[0]))
            out[:, :, i] = base[st]
        # 窗口内 z-score(原地减均值/除方差, 避免生成超大临时数组触发OOM)
        mu = out.mean(axis=1, keepdims=True)
        sd = out.std(axis=1, keepdims=True) + 1e-6
        out -= mu
        out /= sd
        return out
    # 少见的不足窗口情形(逐位置, 保底正确性)
    idx = np.where(st >= 0)[0]
    for i in range(C):
        for j in idx:
            out[j, :, i] = X[st[j]:st[j] + L, i]
    mu = out.mean(axis=1, keepdims=True)
    sd = out.std(axis=1, keepdims=True) + 1e-6
    out -= mu
    out /= sd
    return out


# ---------- 模型: 因果时序卷积(TCN, Bai et al. 2018), CPU 上远快于 LSTM ----------
class CausalConv1d(nn.Module):
    def __init__(self, cin, cout, k=3, dil=1):
        super().__init__()
        self.pad = (k - 1) * dil
        self.conv = nn.Conv1d(cin, cout, k, dilation=dil)

    def forward(self, x):
        return self.conv(F.pad(x, (self.pad, 0)))


class SeqNet(nn.Module):
    def __init__(self, C, H):
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(C, H, 3, 1), nn.ReLU(),
            CausalConv1d(H, H, 3, 2), nn.ReLU(),
            CausalConv1d(H, H, 3, 4), nn.ReLU(),
            CausalConv1d(H, H, 3, 8), nn.ReLU(),
        )
        self.head = nn.Sequential(nn.Linear(H, 24), nn.ReLU(), nn.Dropout(0.1),
                                  nn.Linear(24, 1))

    def forward(self, x):
        x = x.transpose(1, 2)         # (B, L, C) -> (B, C, L)
        h = self.net(x)
        h = h.mean(dim=2)             # 全时间步全局平均池化 -> (B, H)
        return self.head(h).squeeze(-1)


def train_model(Xtr, ytr, Xes, yes, device):
    torch.manual_seed(SEED)
    C = Xtr.shape[2]
    model = SeqNet(C, HIDDEN).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    lossf = nn.BCEWithLogitsLoss()
    xt = torch.from_numpy(Xtr).to(device); yt = torch.from_numpy(ytr).float().to(device)
    xe = torch.from_numpy(Xes).to(device); ye = torch.from_numpy(yes).float().to(device)
    best_es, best_state, pat = 1e18, None, 0
    n = len(xt)
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(n, device=device)
        tot = 0.0; cnt = 0
        for b0 in range(0, n, BATCH):
            idx = perm[b0:b0 + BATCH]
            opt.zero_grad()
            loss = lossf(model(xt[idx]), yt[idx])
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx); cnt += len(idx)
        # early stop
        model.eval()
        with torch.no_grad():
            es_loss = 0.0
            for b0 in range(0, len(xe), 4096):
                es_loss += lossf(model(xe[b0:b0 + 4096]), ye[b0:b0 + 4096]).item() * min(4096, len(xe) - b0)
            es_loss /= len(xe)
        if es_loss < best_es:
            best_es = es_loss; best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}; pat = 0
        else:
            pat += 1
            if pat >= PATIENCE:
                break
        if (ep + 1) % 3 == 0:
            print(f"    ep{ep+1} train={tot/cnt:.4f} es={es_loss:.4f}", flush=True)
    model.load_state_dict(best_state)
    model.eval()
    return model


def predict(model, X, device, BLK=8192):
    model.eval()
    out = np.zeros(len(X), np.float32)
    xt = torch.from_numpy(X)
    with torch.no_grad():
        for b0 in range(0, len(X), BLK):
            logit = model(xt[b0:b0 + BLK].to(device))
            out[b0:b0 + BLK] = torch.sigmoid(logit.cpu()).numpy()
    return out


def chunk_predict(model, Xfeat, positions, L, device, CHK=40_000):
    """分块构建序列并推理, 避免一次性生成 (N, L, C) 超大数组触发OOM。"""
    model.eval()
    m = len(positions)
    out = np.zeros(m, np.float32)
    for s0 in range(0, m, CHK):
        pos = positions[s0:s0 + CHK]
        seq = build_sequences(Xfeat, pos, L)
        with torch.no_grad():
            logit = model(torch.from_numpy(seq).to(device))
        out[s0:s0 + CHK] = torch.sigmoid(logit.cpu()).numpy()
        del seq
    return out


# ---------- 基线融合 ----------
def load_fused_rank(symbol, split):
    Ps = [np.load(f"{config.DS_DIR}/JOINT_{symbol}_{f}_{split}_P.npy") for f in FAMS]
    P = np.concatenate(Ps, axis=0)
    n = P.shape[1]
    R = np.stack([np.argsort(np.argsort(P[i])).astype(np.float64) / (n - 1) for i in range(P.shape[0])], axis=0)
    return R.mean(axis=0)


def pctl01(p):
    """把概率压缩为 (0,1) 内的排名分(均匀化), 避免直接把 sigmoid 当 rank。"""
    n = len(p)
    return np.argsort(np.argsort(p)).astype(np.float64) / (n - 1)


def r2_eval(conf, pred, sec, seed_conf):
    conf = conf.ravel(); pred = pred.ravel(); sec = sec.ravel()
    day = sec // 86400
    days = np.unique(day)
    hist = list(seed_conf)
    keep = np.zeros(len(sec), bool)
    for dd in days:
        md = day == dd
        tau = np.percentile(np.asarray(hist), P99)
        keep[md & (conf >= tau)] = True
        hist.extend(conf[md])
        if len(hist) > WIN_DAYS * SAMP * 2:
            del hist[:len(hist) - WIN_DAYS * SAMP * 2]
    sel = np.where(keep)[0]
    return dict(sel=sel, mts=(sec[sel].astype("datetime64[s]").astype("datetime64[M]")),
                pred=pred[sel], y=None)


def monthly(ps, ys, mts):
    if len(ps) == 0:
        return 0.0, 0.0, 0
    rows = []
    for u in np.unique(mts):
        m = mts == u
        rows.append((str(u)[:7], int(m.sum()), float((ps == ys)[m].mean())))
    acc_all = float((ps == ys).mean())
    return acc_all, rows


def report(name, ps, ys, mts):
    acc, rows = monthly(ps, ys, mts)
    print(f"  {name:<22} 总acc={acc:.4f}  最差={min(r[2] for r in rows):.4f}  坏月={sum(1 for r in rows if r[2] < 0.55)}  信号={len(ps)}")
    return acc


def run_symbol(symbol):
    print(f"\n############ {symbol} ############", flush=True)
    ctx = AssetContext(symbol, horizon=30)
    device = "cpu"
    t0 = time.time()
    X = raw_features(ctx)
    print(f"  [seq特征] {X.shape} 因果单K通道", flush=True)

    # train / early_stop 抽样(仅控 CPU 时间, 仍在 train/es 段内)
    tr_pos = ctx.ds_to_raw[ctx.split_rows["train"]]
    Lm1 = WINDOW - 1
    ok_tr = tr_pos - Lm1 >= 0
    seed = np.random.default_rng(SEED)
    subtr = np.sort(seed.choice(np.where(ok_tr)[0], size=min(TRAIN_CAP, ok_tr.sum()), replace=False))
    subtr_pos = tr_pos[subtr]
    yes = ctx.y("train")[subtr]

    es_pos = ctx.ds_to_raw[ctx.split_rows["early_stop"]]
    ok_es = es_pos - Lm1 >= 0
    es_pos = es_pos[ok_es]
    es_sub = np.sort(seed.choice(len(es_pos), size=min(ES_CAP, len(es_pos)), replace=False))
    Xes = build_sequences(X, es_pos[es_sub], WINDOW)
    yes_es = ctx.y("early_stop")[ok_es][es_sub]

    Xtr = build_sequences(X, subtr_pos, WINDOW)
    print(f"  训练序列 {Xtr.shape[0]}x{WINDOW}x{Xtr.shape[2]}, 构建耗时{(time.time()-t0):.0f}s, 开始训练", flush=True)
    model = train_model(Xtr, yes.astype(np.float32), Xes, yes_es.astype(np.float32), device)
    del Xtr, Xes
    gc.collect()
    print(f"  TCN(序列模型) 训练完成 耗时{(time.time()-t0):.0f}s", flush=True)

    # predict mv / test (分块, 复用已建好的特征矩阵 X, 防止OOM)
    m_idx = ctx.split_rows["meta_val"]; t_idx = ctx.split_rows["test"]
    mp = ctx.ds_to_raw[m_idx]; tp = ctx.ds_to_raw[t_idx]
    pmv = chunk_predict(model, X, mp, WINDOW, device)
    pte = chunk_predict(model, X, tp, WINDOW, device)
    del X, model
    gc.collect()

    sec_mv = ctx.ds_ts[m_idx].astype(np.int64); sec_te = ctx.ds_ts[t_idx].astype(np.int64)
    y_mv = ctx.y("meta_val"); y_te = ctx.y("test")

    # pool20 基线
    pmv_bl = load_fused_rank(symbol, "meta_val"); pte_bl = load_fused_rank(symbol, "test")

    # pool20 基线 + LSTM + 融合(rank 等权)
    fmv = (pctl01(pmv_bl) + pctl01(pmv)) / 2.0
    fte = (pctl01(pte_bl) + pctl01(pte)) / 2.0
    variants = {
        "pool20 基线":       (np.abs(pctl01(pmv_bl) - 0.5) * 2, np.abs(pctl01(pte_bl) - 0.5) * 2,
                              (pte_bl >= 0.5).astype(np.int8)),
        "纯 TCN":            (np.abs(pmv - 0.5) * 2,             np.abs(pte - 0.5) * 2,
                              (pte >= 0.5).astype(np.int8)),
        "pool20+TCN 融合":   (np.abs(fmv - 0.5) * 2,             np.abs(fte - 0.5) * 2,
                              (fte >= 0.5).astype(np.int8)),
    }

    print(f"\n===== {symbol}  30min方向, R2每日滚动Top1% =====")
    for nm, (cmv, cte, pred) in variants.items():
        seed_conf = list(cmv[-WIN_DAYS * SAMP:])
        r = r2_eval(cte, pred, sec_te, seed_conf)
        ysel = y_te[r["sel"]]
        report(nm, r["pred"], ysel, r["mts"])
        gc.collect()

    del ctx
    gc.collect()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("symbol", nargs="?", default=None)
    ap.add_argument("--symbols", default="ETH,BTC")
    a = ap.parse_args()
    for s in ([a.symbol] if a.symbol else a.symbols.split(",")):
        run_symbol(s)


if __name__ == "__main__":
    main()