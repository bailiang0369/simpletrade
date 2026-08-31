"""时序序列模型 (LSTM): 输入原始K线导出的分钟级紧凑特征序列, 预测未来30分钟涨跌。

针对用户需求「时序模型感受形态走势」:
- 输入: 每个样本 = 过去 LOOK_BACK 分钟的 (分钟级特征序列)  形状 [B, LOOK_BACK, F_DIM]
- 每步特征 (8维, 只看t时刻及之前, 无未来泄漏):
    lr1      : 本分钟对数收益
    lr5      : 本分钟相对前5分钟累计收益(跨步可用, 但严格可用t实时)
    rvol     : 本分钟振幅 (high-low)/close
    tbr      : 本分钟主动买占比 buy/(buy+sell)
    cvd      : 本分钟主动净买占比 (buy-sell)/tot
    fix_pos  : 本步在窗口内的位置索引 (时间结构)
    vol_rel  : 本分钟成交量强度/近期均值 (仅buy+sell)
    trend    : 本分钟相对窗口起点累计收益
- 模型: LSTM(64) -> Dropout -> Dense(1) sigmoid
- 内存: 流式构造序列(X按块生成并即时进模型, 不整段驻留), train 子采样 ~30万

使用方式: 与 BaseModel 接口对齐 fit/predict。
"""
import time
import numpy as np
import torch
import torch.nn as nn

import config

LOOK_BACK = 60        # 序列长度(分钟)
F_DIM = 12            # 每步特征维度 (完整 OHLC + 主动买卖微观)
HIDDEN = 64
MAX_TRAIN = 300_000   # 训练序列条数上限(控内存)
BATCH = 256
EPOCHS = 14

# 每步需原始量(相对当前raw位置): lr1需1、lr5需5、rvol需1、tbr/cvd需1、vol_rel需20
_NEED = 20            # 每步"本身就向前的原始索引"用到前 _NEED 根; 但序列窗口另需 LOOK_BACK

RZ_WINDOW = 240       # 滚动标准化窗口(分钟, 约4小时): 只用 t 及以前, 因果无泄漏


def _roll_zscore(x, w):
    """因果滚动 z-score: 每点用过去 w 个点(含当前)的均值/方差标准化。
    只用 x[i-w+1..i] 即 t 及以前, 无未来泄漏。O(n) cumsum。头部 warmup 回退为累计。"""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n == 0:
        return x
    if n < w:
        w = n
    cs = np.cumsum(x)
    # 均值: 用前缀和窗口 [i-w+1, i]
    mu = np.empty(n)
    mu[:w-1] = np.nan
    mu[w-1:] = (cs[w-1:] - np.concatenate([[0.0], cs[:n-w]])) / w
    # 累计均值(头部回退)
    for i in range(1, w-1):
        mu[i] = cs[i] / (i+1)
    mu[0] = x[0]
    # 方差(滚动, 直接累加窗口)
    cs2 = np.cumsum(x * x)
    sd = np.empty(n)
    sd[:w-1] = np.nan
    sd[w-1:] = np.sqrt(np.maximum((cs2[w-1:] - np.concatenate([[0.0], cs2[:n-w]])) / w - mu[w-1:]**2, 0.0))
    for i in range(1, w-1):
        v = max(cs2[i]/(i+1) - mu[i]**2, 0.0)
        sd[i] = np.sqrt(v)
    sd[0] = np.abs(x[0]) if x[0] != 0 else 1.0
    sd = np.where(sd < 1e-9, 1.0, sd)          # 防除零
    return (x - mu) / sd


def build_sequence_feats(raw_o, raw_h, raw_l, raw_c, raw_tb, raw_ts):
    """在 raw 数组上一次性计算分钟级特征向量序列(对齐 raw 行)。
    返回: F 为词典, 每个数组等长 raw; 每根1分钟一个12维向量(用roll回看)。
    完整 OHLC(开高低收) + 主动买卖微观。
    缩放策略:
      - 无界收益类 (lr1/lr5/rvol/trend): 因果滚动 z-score (只用过去窗口, 无泄漏)
      - 有界比率类 (tbr/cvd/body/wick/c_hl): 天然 0~1, 直接按比例不缩放
      - fix_pos: 位置索引 0~LB-1, 归一到 [0,1]
      - vol_rel: 量能强度比值, 取 log 压缩拖尾
    """
    n = len(raw_o)
    o = raw_o.astype(np.float64); h = raw_h.astype(np.float64)
    l = raw_l.astype(np.float64); c = raw_c.astype(np.float64)
    tb = raw_tb.astype(np.float64); ts = raw_ts.astype(np.float64)
    tot = tb + ts
    tot_safe = np.where(tot > 0, tot, 1.0)

    # 各原始序列(一维, 长n)
    # OHLC 相对量 (有界 0..1, 不缩放)
    rng = h - l
    rng_safe = np.where(rng > 0, rng, 1.0)
    body = (c - o) / rng_safe                 # 实体占比/方向(开收相对高低)
    wick_u = (h - np.maximum(o, c)) / rng_safe  # 上影线
    wick_l = (np.minimum(o, c) - l) / rng_safe  # 下影线
    c_hl = (c - l) / rng_safe                 # 收盘在高低区间位置(0..1)
    # 价格 (无界收益类, 滚动 z-score)
    lr1 = np.zeros(n); lr1[1:] = np.log(c[1:]/c[:-1])
    lr5 = np.zeros(n); lr5[5:] = np.log(c[5:]/c[:-5])
    rvol = np.log(c/ np.where(l>0,l,1.0))     # 用 (c/l) 作单分钟上行幅度
    lr1 = _roll_zscore(lr1, RZ_WINDOW)
    lr5 = _roll_zscore(lr5, RZ_WINDOW)
    rvol = _roll_zscore(rvol, RZ_WINDOW)
    # 有界比率 (0..1 / -1..1, 不缩放)
    tbr = tb / tot_safe
    cvd = (tb - ts) / tot_safe
    # 量能强度 (无界比值, log 压缩)
    amt = tb + ts
    vmu = np.convolve(amt, np.ones(20)/20, mode='same')
    vmu = np.where(vmu > 0, vmu, 1.0)
    vol_rel = np.log(np.maximum(amt / vmu, 1e-9))     # log 压缩拖尾
    # trend: 整段累计收益(相对固定起点)的因果滚动 z-score, 作为绝对涨幅的无界表示
    trend_c = np.zeros(n); trend_c[1:] = np.cumsum(np.log(c[1:]/np.where(c[:-1]>0,c[:-1],1.0)))
    trend_c = _roll_zscore(trend_c, RZ_WINDOW)
    return dict(c=c, o=o, h=h, l=l, tb=tb, ts=ts,
                lr1=lr1, lr5=lr5, rvol=rvol, tbr=tbr, cvd=cvd, vol_rel=vol_rel,
                body=body, wick_u=wick_u, wick_l=wick_l, c_hl=c_hl, trend_c=trend_c)


def build_ds_matrix(F, raw_positions, look_back=LOOK_BACK):
    """对目标 raw 位置构造序列矩阵 [len(pos), look_back, F_DIM]。
    F_DIM 列: lr1,lr5,rvol,tbr,cvd,body,wick_u,wick_l,c_hl,vol_rel,fix_pos,trend
    返回 float32。分块构造避免 OOM。
    """
    n = len(raw_positions)
    out = np.zeros((n, look_back, F_DIM), dtype=np.float32)
    c_idx = F["c"]
    f = {k: F[k] for k in ["lr1","lr5","rvol","tbr","cvd","body","wick_u","wick_l","c_hl","vol_rel"]}
    BLK = 60_000
    for b0 in range(0, n, BLK):
        b1 = min(b0+BLK, n)
        pos = raw_positions[b0:b1]
        valid = pos >= look_back-1
        sub = pos.astype(np.int64)
        # 序列轴: s = 0..look_back-1 对应 raw 位置 sub-(look_back-1)+ s
        seq_off = np.arange(look_back, dtype=np.int64)[None, :] - (look_back-1)   # [1, LB]
        row_pos = sub[:, None] + seq_off                        # [B, LB]
        # 各步特征(向量化 gather, 全部只取 row_pos 位置即 t 及以前)
        blk_list = [f[k][row_pos] for k in ["lr1","lr5","rvol","tbr","cvd",
                                            "body","wick_u","wick_l","c_hl","vol_rel"]]
        # fix_pos: 步索引 0..LB-1, 归一化到 [0,1] (避免与收益类特征量纲差几个数量级)
        st_fix = np.broadcast_to((np.arange(look_back, dtype=np.float32))[None, :] / float(look_back-1),
                                 blk_list[0].shape)
        # trend: 已由 feats 因果滚动标准化(绝对涨幅), 直接取历史步
        st_trend = F["trend_c"][row_pos]
        blk_list += [st_fix, st_trend]
        # 组装 [B, LB, 12]
        blk = np.stack(blk_list, axis=-1)
        blk = np.nan_to_num(blk, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        # 无效行(头部不足look_back)置零
        if not valid.all():
            blk[~valid] = 0.0
        out[b0:b1] = blk
    return out


class LSTMModel(nn.Module):
    def __init__(self, fan=F_DIM, hidden=HIDDEN):
        super().__init__()
        self.lstm = nn.LSTM(fan, hidden, batch_first=True)
        self.drop = nn.Dropout(0.3)
        self.head = nn.Linear(hidden, 1)
    def forward(self, x):
        o, _ = self.lstm(x)
        out = o[:, -1, :]               # 最后步
        return self.head(self.drop(out)).squeeze(-1)


class SeqGBDLSTM:
    """LSTM 时序模型, 接口对齐 BaseModel."""
    name = "lstm"
    NAME = "lstm"

    def __init__(self, seed=42, look_back=LOOK_BACK, hidden=HIDDEN):
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.seed = seed; self.look_back = look_back
        self.model = LSTMModel(hidden=hidden)
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
        # 训练子采样(控内存)
        rng = np.random.default_rng(self.seed)
        if len(tr_pos) > MAX_TRAIN:
            keep = rng.choice(len(tr_pos), MAX_TRAIN, replace=False)
            tr_pos = tr_pos[keep]
        ytr = ctx.label[ctx.split_rows["train"]][:0]  # 占位, 下面按 pos 重算
        # 标签对齐: 用 ds 行的 label, 需 pos->ds 反查
        ds_raw = ctx.ds_to_raw
        tr_ds = np.searchsorted(ds_raw, tr_pos); tr_ds = np.clip(tr_ds, 0, len(ctx.label)-1)
        ytr = ctx.label[tr_ds].astype(np.float32)
        es_ds = np.searchsorted(ds_raw, es_pos); es_ds = np.clip(es_ds, 0, len(ctx.label)-1)
        yes = ctx.label[es_ds].astype(np.float32)
        # 构造序列
        print("构造训练序列...", flush=True); t0=time.time()
        Xtr = build_ds_matrix(F, tr_pos, self.look_back)
        print(f"  Xtr {Xtr.shape} {time.time()-t0:.0f}s", flush=True)
        print("构造验证序列...", flush=True)
        Xes = build_ds_matrix(F, es_pos, self.look_back)
        # 过滤 warmup 无效行(全零序列)
        vtr = Xtr.sum(axis=(1,2)) != 0
        ves = Xes.sum(axis=(1,2)) != 0
        Xtr, ytr = Xtr[vtr], ytr[vtr]
        Xes, yes = Xes[ves], yes[ves]
        print(f"  train {Xtr.shape} es {Xes.shape} label正例率 {ytr.mean():.3f} {yes.mean():.3f}", flush=True)

        Xt = torch.from_numpy(Xtr); yt = torch.from_numpy(ytr)
        Xe = torch.from_numpy(Xes); ye = torch.from_numpy(yes)
        self.model.train()
        opt = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        lossf = nn.BCEWithLogitsLoss()
        n = len(Xt); nB = (n + BATCH - 1)//BATCH
        best = 1e9; best_state = None; patience=3; bad=0
        for ep in range(EPOCHS):
            perm = torch.randperm(n)
            self.model.train(); tot=0.0
            for i in range(nB):
                idx = perm[i*BATCH:(i+1)*BATCH]
                xb = Xt[idx]; yb = yt[idx]
                opt.zero_grad()
                out = self.model(xb)
                loss = lossf(out, yb)
                loss.backward(); opt.step(); tot += loss.item()*len(idx)
            # 验证 (需分块, 避免长序列整段驻留OOM)
            self.model.eval()
            with torch.no_grad():
                el = 0.0; cnt = 0; correct = 0; nacc = 0
                nel = len(Xe)
                for b0 in range(0, nel, BATCH*4):
                    b1 = min(b0+BATCH*4, nel)
                    yeo = self.model(Xe[b0:b1])
                    el += lossf(yeo, ye[b0:b1]).item() * (b1-b0)
                    correct += ((torch.sigmoid(yeo)>=0.5).float()==ye[b0:b1]).float().sum().item()
                    nacc += (b1-b0)
                el /= nel
                acc_es = correct / max(1, nacc)
            print(f"[lstm] ep{ep} tr_loss={tot/n:.4f} es_loss={el:.4f} es_acc={acc_es:.4f}", flush=True)
            if el < best:
                best = el; bad = 0
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                bad += 1
                if bad >= patience: break
        if best_state: self.model.load_state_dict(best_state)
        self.fitted = True
        print("[lstm] fit done", flush=True)

    def predict(self, ctx, split):
        if not self.fitted: raise RuntimeError("fit first")
        F = self._prep(ctx)
        pos = ctx.ds_to_raw[ctx.split_rows[split]]
        X = build_ds_matrix(F, pos, self.look_back)
        self.model.eval()
        Xt = torch.from_numpy(X)
        with torch.no_grad():
            p = torch.sigmoid(self.model(Xt)).numpy()
        return p.astype(np.float32)