"""GRU 时序深度学习模型: 输入最近 LOOKBACK 分钟的逐分钟通道(收益率/实体/上下影/主动买占比/主动净买),
逐窗口zscore归一化。捕捉序列动态结构。
"""
import time

import numpy as np
import torch
import torch.nn as nn

import config
from .base import BaseModel

torch.set_num_threads(config.N_JOBS)


class GRUNet(nn.Module):
    def __init__(self, C=6, hidden=96, layers=1, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(C, hidden, layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1))

    def forward(self, x):
        _, h = self.gru(x)
        z = h[-1]
        return self.head(z).squeeze(-1)


class GRUModel(BaseModel):
    name = "gru"

    def __init__(self, seed=None, hidden=config.GRU_HIDDEN, epochs=config.DL_EPOCHS,
                 n_samples=config.N_DL_SAMPLES, W=None):
        super().__init__(seed)
        self.hidden = hidden
        self.epochs = epochs
        self.n_samples = n_samples
        self.W = W or config.LOOKBACK_MIN
        self.net = None

    def _windows(self, ctx, positions, chunk=12000):
        """返回 (windows, labels)。windows: (N,W,C) float32。"""
        return ctx.window_batch(np.asarray(positions), W=self.W, chunk=chunk)

    def fit(self, ctx):
        t0 = time.time()
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        tr_pos = ctx.split_positions("train")
        tr_y = ctx.y("train")
        es_pos = ctx.split_positions("early_stop")
        es_y = ctx.y("early_stop")

        rng = np.random.default_rng(self.seed)
        n_tr = min(self.n_samples, len(tr_pos))
        idx_tr = rng.choice(len(tr_pos), n_tr, replace=False)
        idx_es = rng.choice(len(es_pos), min(40000, len(es_pos)), replace=False)
        Xtr = torch.from_numpy(self._windows(ctx, tr_pos[idx_tr]))
        ytr = torch.tensor(tr_y[idx_tr], dtype=torch.float32)
        Xes = torch.from_numpy(self._windows(ctx, es_pos[idx_es]))
        yes = torch.tensor(es_y[idx_es], dtype=torch.float32)

        self.net = GRUNet(C=Xtr.shape[2], hidden=self.hidden)
        opt = torch.optim.Adam(self.net.parameters(), lr=1.5e-3, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
        crit = nn.BCEWithLogitsLoss()
        ds = torch.utils.data.TensorDataset(Xtr, ytr)
        dl = torch.utils.data.DataLoader(ds, batch_size=config.BATCH_SIZE, shuffle=True,
                                         num_workers=0)
        best_acc, best_state = 0.0, None
        for ep in range(self.epochs):
            self.net.train()
            tot, corr = 0, 0
            for xb, yb in dl:
                opt.zero_grad()
                logit = self.net(xb)
                loss = crit(logit, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
                opt.step()
                pred = (torch.sigmoid(logit) > 0.5).float()
                corr += (pred == yb).sum().item()
                tot += len(yb)
            sched.step()
            # 验证
            self.net.eval()
            with torch.no_grad():
                val_acc, vn = 0.0, 0
                for s in range(0, len(Xes), config.BATCH_SIZE):
                    xb = Xes[s:s + config.BATCH_SIZE]
                    yb = yes[s:s + config.BATCH_SIZE]
                    pred = (torch.sigmoid(self.net(xb)) > 0.5).float()
                    val_acc += (pred == yb).sum().item()
                    vn += len(yb)
                val_acc /= vn
            print(f"[gru] ep{ep} train_acc={corr/tot:.4f} val_acc={val_acc:.4f}", flush=True)
            if val_acc > best_acc:
                best_acc = val_acc
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.fitted = True
        print(f"[gru] {ctx.symbol} fit done {time.time()-t0:.0f}s, best_val_acc={best_acc:.4f}", flush=True)
        return self

    def predict(self, ctx, split):
        pos = ctx.split_positions(split)
        self.net.eval()
        out = np.empty(len(pos), dtype=np.float32)
        with torch.no_grad():
            for s in range(0, len(pos), 8192):
                e = min(s + 8192, len(pos))
                X = torch.from_numpy(self._windows(ctx, pos[s:e], chunk=8192))
                out[s:e] = torch.sigmoid(self.net(X)).numpy()
        return out

    def save(self, path):
        torch.save(self.net.state_dict(), path)

    def load(self, ctx, path, C=6):
        self.net = GRUNet(C=C, hidden=self.hidden)
        self.net.load_state_dict(torch.load(path, map_location="cpu"))
        self.net.eval()
        self.fitted = True
