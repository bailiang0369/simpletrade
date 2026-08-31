"""替代时序模型: GRU / AttentionLSTM / TransformerEncoder.
与前序 LSTM 共享特征构建 (build_sequence_feats, build_ds_matrix), 接口对齐 fit/predict。
"""
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import config
from models.seq_lstm import build_sequence_feats, build_ds_matrix, LOOK_BACK, F_DIM, HIDDEN, MAX_TRAIN, BATCH, EPOCHS

# ============================================================
# 模型定义
# ============================================================

class GRUModel(nn.Module):
    """GRU: 比 LSTM 少一个门, 参数更少, 训练更快。"""
    def __init__(self, fan=F_DIM, hidden=HIDDEN, num_layers=2):
        super().__init__()
        self.gru = nn.GRU(fan, hidden, num_layers, batch_first=True, dropout=0.2 if num_layers > 1 else 0)
        self.drop = nn.Dropout(0.3)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        o, _ = self.gru(x)
        out = o[:, -1, :]
        return self.head(self.drop(out)).squeeze(-1)


class AttentionLSTM(nn.Module):
    """LSTM + 自注意力池化: 不只用最后步, 而是对所有时间步加权求和。"""
    def __init__(self, fan=F_DIM, hidden=HIDDEN):
        super().__init__()
        self.lstm = nn.LSTM(fan, hidden, batch_first=True, bidirectional=True)
        self.attn = nn.Linear(hidden * 2, 1)  # 双向所以 *2
        self.drop = nn.Dropout(0.3)
        self.head = nn.Linear(hidden * 2, 1)

    def forward(self, x):
        o, _ = self.lstm(x)                        # [B, T, H*2]
        # 自注意力权重
        scores = self.attn(o)                       # [B, T, 1]
        alpha = F.softmax(scores, dim=1)            # [B, T, 1]
        ctx = (o * alpha).sum(dim=1)                # [B, H*2]
        return self.head(self.drop(ctx)).squeeze(-1)


class TransformerEncoderModel(nn.Module):
    """Transformer Encoder: 纯自注意力, 与 RNN 系列原理完全不同。"""
    def __init__(self, fan=F_DIM, d_model=64, nhead=4, num_layers=2, dim_feedforward=128):
        super().__init__()
        self.proj = nn.Linear(fan, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=LOOK_BACK)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                dim_feedforward=dim_feedforward,
                                                dropout=0.2, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.drop = nn.Dropout(0.3)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.proj(x)                            # [B, T, d_model]
        x = self.pos_enc(x)
        x = self.encoder(x)                         # [B, T, d_model]
        # 取全局平均池化 (而非最后步, 更鲁棒)
        pooled = x.mean(dim=1)                      # [B, d_model]
        return self.head(self.drop(pooled)).squeeze(-1)


class PositionalEncoding(nn.Module):
    """正弦位置编码 (Transformer 无时序归纳偏置, 需要位置信息)。"""
    def __init__(self, d_model, max_len=120):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


# ============================================================
# 训练封装 (接口对齐 SeqGBDLSTM)
# ============================================================

_MODEL_MAP = {
    "gru": GRUModel,
    "attn_lstm": AttentionLSTM,
    "transformer": TransformerEncoderModel,
}


class SeqAltModel:
    """替代时序模型封装, 接口对齐 SeqGBDLSTM。"""
    def __init__(self, model_type="gru", seed=42, look_back=LOOK_BACK):
        assert model_type in _MODEL_MAP, f"Unknown model_type: {model_type}, options: {list(_MODEL_MAP.keys())}"
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.seed = seed
        self.model_type = model_type
        self.look_back = look_back
        self.model = _MODEL_MAP[model_type]()
        self.fitted = False
        self._F = None
        self.name = model_type
        self.NAME = model_type.upper()

    def _prep(self, ctx):
        if self._F is None:
            self._F = build_sequence_feats(ctx.o, ctx.h, ctx.l, ctx.c, ctx.tb, ctx.vol)
        return self._F

    def fit(self, ctx):
        F = build_sequence_feats(ctx.o, ctx.h, ctx.l, ctx.c, ctx.tb, ctx.vol)
        torch.set_num_threads(config.N_JOBS)
        tr_pos = ctx.ds_to_raw[ctx.split_rows["train"]]
        es_pos = ctx.ds_to_raw[ctx.split_rows["early_stop"]]

        rng = np.random.default_rng(self.seed)
        if len(tr_pos) > MAX_TRAIN:
            keep = rng.choice(len(tr_pos), MAX_TRAIN, replace=False)
            tr_pos = tr_pos[keep]

        ds_raw = ctx.ds_to_raw
        tr_ds = np.searchsorted(ds_raw, tr_pos); tr_ds = np.clip(tr_ds, 0, len(ctx.label)-1)
        ytr = ctx.label[tr_ds].astype(np.float32)
        es_ds = np.searchsorted(ds_raw, es_pos); es_ds = np.clip(es_ds, 0, len(ctx.label)-1)
        yes = ctx.label[es_ds].astype(np.float32)

        print(f"构造训练序列 [{self.model_type}]...", flush=True); t0 = time.time()
        Xtr = build_ds_matrix(F, tr_pos, self.look_back)
        print(f"  Xtr {Xtr.shape} {time.time()-t0:.0f}s", flush=True)
        print("构造验证序列...", flush=True)
        Xes = build_ds_matrix(F, es_pos, self.look_back)

        vtr = Xtr.sum(axis=(1, 2)) != 0
        ves = Xes.sum(axis=(1, 2)) != 0
        Xtr, ytr = Xtr[vtr], ytr[vtr]
        Xes, yes = Xes[ves], yes[ves]
        print(f"  train {Xtr.shape} es {Xes.shape} 正例率 {ytr.mean():.3f} {yes.mean():.3f}", flush=True)

        Xt = torch.from_numpy(Xtr); yt = torch.from_numpy(ytr)
        Xe = torch.from_numpy(Xes); ye = torch.from_numpy(yes)

        self.model.train()
        opt = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        lossf = nn.BCEWithLogitsLoss()
        n = len(Xt); nB = (n + BATCH - 1) // BATCH
        best = 1e9; best_state = None; patience = 3; bad = 0

        for ep in range(EPOCHS):
            perm = torch.randperm(n)
            self.model.train(); tot = 0.0
            for i in range(nB):
                idx = perm[i * BATCH:(i + 1) * BATCH]
                xb = Xt[idx]; yb = yt[idx]
                opt.zero_grad()
                out = self.model(xb)
                loss = lossf(out, yb)
                loss.backward(); opt.step(); tot += loss.item() * len(idx)
            # 验证
            self.model.eval()
            with torch.no_grad():
                el = 0.0; correct = 0; nacc = 0
                nel = len(Xe)
                for b0 in range(0, nel, BATCH * 4):
                    b1 = min(b0 + BATCH * 4, nel)
                    yeo = self.model(Xe[b0:b1])
                    el += lossf(yeo, ye[b0:b1]).item() * (b1 - b0)
                    correct += ((torch.sigmoid(yeo) >= 0.5).float() == ye[b0:b1]).float().sum().item()
                    nacc += (b1 - b0)
                el /= nel
                acc_es = correct / max(1, nacc)
            print(f"[{self.model_type}] ep{ep} tr_loss={tot / n:.4f} es_loss={el:.4f} es_acc={acc_es:.4f}", flush=True)
            if el < best:
                best = el; bad = 0
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break
        if best_state:
            self.model.load_state_dict(best_state)
        self.fitted = True
        # 释放训练时的 F 特征缓存 (节省 ~400MB)
        self._F = None
        import gc; gc.collect()
        print(f"[{self.model_type}] fit done", flush=True)

    def predict(self, ctx, split, chunked=True, BLK=10_000):
        if not self.fitted:
            raise RuntimeError("fit first")
        import gc
        # 每次预测重建 F, 用完释放
        F = build_sequence_feats(ctx.o, ctx.h, ctx.l, ctx.c, ctx.tb, ctx.vol)
        pos = ctx.ds_to_raw[ctx.split_rows[split]]
        n = len(pos)
        out = np.zeros(n, dtype=np.float32)
        self.model.eval()
        with torch.no_grad():
            for b0 in range(0, n, BLK):
                b1 = min(b0 + BLK, n)
                Xb = build_ds_matrix(F, pos[b0:b1], self.look_back)
                out[b0:b1] = torch.sigmoid(self.model(torch.from_numpy(Xb))).numpy().squeeze()
                # 每块之后释放 Xb
                del Xb
                if b0 % 100000 == 0 and b0 > 0:
                    gc.collect()
                    print(f"  pred {b0}/{n}", flush=True)
        return out.astype(np.float32)