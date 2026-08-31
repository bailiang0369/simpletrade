"""时序序列模型 (Transformer): 自注意力全局时序建模, 与 LSTM 形成正交信号。

与 LSTM 的核心差异:
- LSTM: 逐步压缩到隐状态, 偏重近端时序依赖
- Transformer: 同时看全部 60 步, 自注意力学习哪些时间步最关键
- 两者归纳偏置完全不同, 预期相关性低(~0.3-0.5), 形成正交第二信号

模型结构: InputProj -> PosEnc -> TransformerEncoder(2层,4头) -> 末步 -> Dense(1)
数据管道复用 seq_lstm.py 的 build_sequence_feats / build_ds_matrix。
"""
import time
import numpy as np
import torch
import torch.nn as nn

import config
from .seq_lstm import build_sequence_feats, build_ds_matrix, LOOK_BACK, F_DIM

D_MODEL = 64
NHEAD = 4
NUM_LAYERS = 2
MAX_TRAIN = 300_000
BATCH = 256
EPOCHS = 14


class TransformerSeqModel(nn.Module):
    def __init__(self, fan=F_DIM, d_model=D_MODEL, nhead=NHEAD, num_layers=NUM_LAYERS):
        super().__init__()
        self.input_proj = nn.Linear(fan, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, LOOK_BACK, d_model) * 0.1)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=d_model*2,
            dropout=0.2, batch_first=True, activation='gelu')
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers)
        self.drop = nn.Dropout(0.2)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: [B, T, F]
        x = self.input_proj(x) + self.pos_embed
        x = self.transformer(x)            # [B, T, D]
        x = x[:, -1, :]                    # 末步输出
        return self.head(self.drop(x)).squeeze(-1)


class SeqTransformer:
    """Transformer 时序模型, 接口对齐 BaseModel。"""
    name = "transformer"
    NAME = "transformer"

    def __init__(self, seed=42, d_model=D_MODEL, nhead=NHEAD, num_layers=NUM_LAYERS):
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.seed = seed
        self.model = TransformerSeqModel(d_model=d_model, nhead=nhead, num_layers=num_layers)
        self.fitted = False
        self._F = None

    def _prep(self, ctx):
        if self._F is None:
            self._F = build_sequence_feats(ctx.o, ctx.h, ctx.l, ctx.c, ctx.tb, ctx.vol)
        return self._F

    def fit(self, ctx):
        F = self._prep(ctx)
        torch.set_num_threads(config.N_JOBS)
        tr_pos = ctx.ds_to_raw[ctx.split_rows["train"]]
        es_pos = ctx.ds_to_raw[ctx.split_rows["early_stop"]]
        # 子采样(控内存)
        rng = np.random.default_rng(self.seed)
        if len(tr_pos) > MAX_TRAIN:
            keep = rng.choice(len(tr_pos), MAX_TRAIN, replace=False)
            tr_pos = tr_pos[keep]
        # 标签对齐
        tr_ds = np.searchsorted(ctx.ds_to_raw, tr_pos); tr_ds = np.clip(tr_ds, 0, len(ctx.label)-1)
        ytr = ctx.label[tr_ds].astype(np.float32)
        es_ds = np.searchsorted(ctx.ds_to_raw, es_pos); es_ds = np.clip(es_ds, 0, len(ctx.label)-1)
        yes = ctx.label[es_ds].astype(np.float32)
        # 构造序列
        print("构造训练序列...", flush=True); t0 = time.time()
        Xtr = build_ds_matrix(F, tr_pos, LOOK_BACK)
        print(f"  Xtr {Xtr.shape} {time.time()-t0:.0f}s", flush=True)
        print("构造验证序列...", flush=True)
        Xes = build_ds_matrix(F, es_pos, LOOK_BACK)
        # 过滤无效行
        vtr = Xtr.sum(axis=(1,2)) != 0
        ves = Xes.sum(axis=(1,2)) != 0
        Xtr, ytr = Xtr[vtr], ytr[vtr]
        Xes, yes = Xes[ves], yes[ves]
        print(f"  train {Xtr.shape} es {Xes.shape} 正例率 {ytr.mean():.3f} {yes.mean():.3f}", flush=True)

        Xt = torch.from_numpy(Xtr); yt = torch.from_numpy(ytr)
        Xe = torch.from_numpy(Xes); ye = torch.from_numpy(yes)
        self.model.train()
        opt = torch.optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
        lossf = nn.BCEWithLogitsLoss()
        n = len(Xt); nB = (n + BATCH - 1)//BATCH
        best = 1e9; best_state = None; patience = 3; bad = 0
        for ep in range(EPOCHS):
            perm = torch.randperm(n)
            self.model.train(); tot = 0.0
            for i in range(nB):
                idx = perm[i*BATCH:(i+1)*BATCH]
                xb = Xt[idx]; yb = yt[idx]
                opt.zero_grad()
                out = self.model(xb)
                loss = lossf(out, yb)
                loss.backward(); opt.step(); tot += loss.item()*len(idx)
            # 验证(分块防OOM)
            self.model.eval()
            with torch.no_grad():
                el = 0.0; correct = 0; nacc = 0; nel = len(Xe)
                for b0 in range(0, nel, BATCH*4):
                    b1 = min(b0+BATCH*4, nel)
                    yeo = self.model(Xe[b0:b1])
                    el += lossf(yeo, ye[b0:b1]).item() * (b1-b0)
                    correct += ((torch.sigmoid(yeo)>=0.5).float()==ye[b0:b1]).float().sum().item()
                    nacc += (b1-b0)
                el /= nel; acc_es = correct / max(1, nacc)
            print(f"[transformer] ep{ep} tr_loss={tot/n:.4f} es_loss={el:.4f} es_acc={acc_es:.4f}", flush=True)
            if el < best:
                best = el; bad = 0
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                bad += 1
                if bad >= patience: break
        if best_state: self.model.load_state_dict(best_state)
        self.fitted = True
        print("[transformer] fit done", flush=True)

    def predict(self, ctx, split):
        if not self.fitted: raise RuntimeError("fit first")
        F = self._prep(ctx)
        pos = ctx.ds_to_raw[ctx.split_rows[split]]
        X = build_ds_matrix(F, pos, LOOK_BACK)
        self.model.eval()
        Xt = torch.from_numpy(X)
        with torch.no_grad():
            p = torch.sigmoid(self.model(Xt)).numpy()
        return p.astype(np.float32)

    def predict_chunked(self, ctx, split, BLK=8_000):
        """分块预测, 避免验证集过大 OOM。"""
        if not self.fitted: raise RuntimeError("fit first")
        F = self._prep(ctx)
        pos = ctx.ds_to_raw[ctx.split_rows[split]]
        n = len(pos); out = np.zeros(n, dtype=np.float32)
        self.model.eval()
        with torch.no_grad():
            for b0 in range(0, n, BLK):
                pb = pos[b0:b0+BLK]
                X = build_ds_matrix(F, pb, LOOK_BACK)
                out[b0:b0+len(pb)] = torch.sigmoid(self.model(torch.from_numpy(X))).numpy()
        return out