"""时序序列模型 (LSTM): 输入原始K线导出的分钟级紧凑特征序列, 预测未来30分钟涨跌。

针对用户需求「时序模型感受形态走势」:
- 输入: 每个样本 = 过去 LOOK_BACK 分钟的 (分钟级特征序列)  形状 [B, LOOK_BACK, F_DIM]
- 每步特征 (22维, 只看t时刻及之前, 无未来泄漏):
    lr1       : 本分钟对数收益
    lr5       : 本分钟相对前5分钟累计收益(跨步可用, 但严格可用t实时)
    rvol      : 本分钟振幅 (high-low)/close
    tbr       : 本分钟主动买占比 buy/(buy+sell)
    cvd       : 本分钟主动净买占比 (buy-sell)/tot
    body      : 实体占比/方向(开收相对高低)
    wick_u    : 上影线占比
    wick_l    : 下影线占比
    c_hl      : 收盘在高低区间位置(0..1)
    vol_rel   : 本分钟成交量强度/近期均值 (仅buy+sell)
    stoch_k_50: 50分钟随机指标K线 (c-min50)/(max50-min50)
    di_plus   : DMI 正向动量 (+DI) / 归一化 [0,1]
    di_minus   : DMI 负向动量 (-DI) / 归一化 [0,1]
    lr_15     : 15分钟滚动收益 (z-score)
    lr_30     : 30分钟滚动收益 (z-score)
    rvol_5    : 5分钟滚动波幅 (z-score)
    rvol_15   : 15分钟滚动波幅 (z-score)
    tbr_15    : 15分钟滚动买占比 [0,1]
    pos_15    : 收盘在15分钟区间位置 [0,1]
    pos_30    : 收盘在30分钟区间位置 [0,1]
    fix_pos   : 本步在窗口内的位置索引 (时间结构)
    trend     : 本分钟相对窗口起点累计收益
- 模型: BiLSTM(128) -> LayerNorm -> Dropout -> Dense(1) sigmoid
- 内存: 流式构造序列(X按块生成并即时进模型, 不整段驻留), train 子采样 ~6万

使用方式: 与 BaseModel 接口对齐 fit/predict。
"""
import gc
import time
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import torch
import torch.nn as nn

import config

LOOK_BACK = 60        # 序列长度(分钟)
F_DIM = 22            # 每步特征维度 (OHLC + 主动买卖 + stoch + DMI + 多尺度聚合)
HIDDEN = 128
MAX_TRAIN = 50_000    # 训练序列条数上限(控内存, 22维特征, ETH尤其吃内存)
BATCH = 64
EPOCHS = 20
GRAD_ACCUM_STEPS = 4  # 梯度累积步数，等效 batch = 256，减少显存占用

# 每步需原始量(相对当前raw位置): lr1需1、lr5需5、rvol需1、tbr/cvd需1、vol_rel需20
_NEED = 20            # 每步"本身就向前的原始索引"用到前 _NEED 根; 但序列窗口另需 LOOK_BACK

RZ_WINDOW = 240       # 滚动标准化窗口(分钟, 约4小时): 只用 t 及以前, 因果无泄漏
USE_SOFT_LABEL = True # 训练用软标签，验证/评估仍用二元标签


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
    # 量能强度 (无界比值, log 压缩), 严格因果(只用过去20根, 不含未来)
    amt = tb + ts
    cs = np.cumsum(np.concatenate([[0.0], amt]))
    vmu = np.empty(n)
    vmu[:19] = amt[:19]  # warmup 直接取当前值
    if n > 20:
        vmu[19:] = (cs[20:] - cs[:-20]) / 20.0
    vmu = np.where(vmu > 0, vmu, 1.0)
    vol_rel = np.log(np.maximum(amt / vmu, 1e-6))     # log 压缩拖尾
    # Stochastic %K(50): 当前价格在过去50分钟区间位置, 天然有界0~1
    w = 50
    _tmp_min = np.empty(n); _tmp_min.fill(np.nan)
    _tmp_max = np.empty(n); _tmp_max.fill(np.nan)
    if n >= w:
        _tmp_min[w-1:] = sliding_window_view(l, w).min(axis=-1)
        _tmp_max[w-1:] = sliding_window_view(h, w).max(axis=-1)
    min50 = np.where(np.isfinite(_tmp_min), _tmp_min, l)
    max50 = np.where(np.isfinite(_tmp_max), _tmp_max, h)
    rng50 = max50 - min50; rng50 = np.where(rng50 > 0, rng50, 1.0)
    stoch_k_50 = (c - min50) / rng50
    # DMI (DI+ / DI-): 方向性动量指标, 天然有界0~1
    # 方向性运动
    up_move = np.zeros(n); up_move[1:] = h[1:] - h[:-1]
    up_move = np.maximum(up_move, 0).astype(np.float64)
    down_move = np.zeros(n); down_move[1:] = l[:-1] - l[1:]
    down_move = np.maximum(down_move, 0).astype(np.float64)
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    # 真实波幅
    c_prev = np.empty(n, dtype=np.float64); c_prev[0] = c[0]; c_prev[1:] = c[:-1]
    h_l = (h - l).astype(np.float64)
    h_cp = np.abs(h.astype(np.float64) - c_prev)
    l_cp = np.abs(l.astype(np.float64) - c_prev)
    tr = np.maximum(h_l, np.maximum(h_cp, l_cp))
    # Wilder's 平滑 (14周期)
    def wilder_smooth(series, period=14):
        out = np.empty(n, dtype=np.float64)
        out[:period-1] = series[:period-1]
        if n >= period:
            out[period-1] = np.mean(series[:period])
        alpha = 1.0 / period
        for i in range(period, n):
            out[i] = out[i-1] * (1-alpha) + series[i] * alpha
        return out.astype(np.float32)
    smooth_tr = wilder_smooth(tr)
    smooth_tr = np.where(smooth_tr > 0, smooth_tr, 1.0)
    di_plus = wilder_smooth(plus_dm) / smooth_tr
    di_minus = wilder_smooth(minus_dm) / smooth_tr
    # trend: 整段累计收益(相对固定起点)的因果滚动 z-score, 作为绝对涨幅的无界表示
    trend_c = np.zeros(n); trend_c[1:] = np.cumsum(np.log(c[1:]/np.where(c[:-1]>0,c[:-1],1.0)))
    trend_c = _roll_zscore(trend_c, RZ_WINDOW)
    # ---- 多尺度聚合特征 ---- 
    # 滚动收益率 (因果, 只用过去N根)
    def rolling_ret(series, w):
        out = np.zeros(n, dtype=np.float64)
        if n > w:
            out[w:] = np.log(series[w:] / series[:-w])
        return _roll_zscore(out, RZ_WINDOW)
    lr_15 = rolling_ret(c, 15)
    lr_30 = rolling_ret(c, 30)
    # 滚动波幅 (5/15分钟平均真实波幅, z-score)
    def rolling_vol(series, w):
        out = np.zeros(n, dtype=np.float64)
        if n > w:
            cs = np.cumsum(series)
            out[w-1:] = (cs[w-1:] - np.concatenate([[0.0], cs[:n-w]])) / w
        return _roll_zscore(out, RZ_WINDOW)
    # 使用单分钟振幅 abs(log(c/l)) 作为波动度量
    amp = np.abs(np.log(c / np.where(l>0, l, 1.0)))
    rvol_5 = rolling_vol(amp, 5)
    rvol_15 = rolling_vol(amp, 15)
    # 滚动买占比 (15分钟)
    def rolling_ratio(series, w):
        tot = tb + ts
        amt = series
        cs = np.cumsum(np.concatenate([[0.0], amt]))
        tcs = np.cumsum(np.concatenate([[0.0], tot]))
        out = np.zeros(n, dtype=np.float64)
        if n > w:
            amt_w = cs[w:] - cs[:-w]
            tot_w = tcs[w:] - tcs[:-w]
            out[w-1:] = amt_w / np.where(tot_w > 0, tot_w, 1.0)
        else:
            out = tb / tot_safe
        return out  # 天然0~1
    tbr_15 = rolling_ratio(tb, 15)
    # 滚动位置 (收盘在N分钟高低区间位置)
    def rolling_pos(series, w):
        min_w = np.empty(n, dtype=np.float64); min_w.fill(np.nan)
        max_w = np.empty(n, dtype=np.float64); max_w.fill(np.nan)
        if n >= w:
            min_w[w-1:] = sliding_window_view(l, w).min(axis=-1)
            max_w[w-1:] = sliding_window_view(h, w).max(axis=-1)
        mn = np.where(np.isfinite(min_w), min_w, l)
        mx = np.where(np.isfinite(max_w), max_w, h)
        rng = mx - mn; rng = np.where(rng > 0, rng, 1.0)
        return (c - mn) / rng  # 天然0~1
    pos_15 = rolling_pos(c, 15)
    pos_30 = rolling_pos(c, 30)
    return dict(c=c, o=o, h=h, l=l, tb=tb, ts=ts,
                lr1=lr1, lr5=lr5, rvol=rvol, tbr=tbr, cvd=cvd, vol_rel=vol_rel,
                body=body, wick_u=wick_u, wick_l=wick_l, c_hl=c_hl, trend_c=trend_c,
                stoch_k_50=stoch_k_50, di_plus=di_plus, di_minus=di_minus,
                lr_15=lr_15, lr_30=lr_30, rvol_5=rvol_5, rvol_15=rvol_15,
                tbr_15=tbr_15, pos_15=pos_15, pos_30=pos_30)


def build_ds_matrix(F, raw_positions, look_back=LOOK_BACK):
    """对目标 raw 位置构造序列矩阵 [len(pos), look_back, F_DIM]。
    F_DIM 列: lr1,lr5,rvol,tbr,cvd,body,wick_u,wick_l,c_hl,vol_rel,
              stoch_k_50,di_plus,di_minus,
              lr_15,lr_30,rvol_5,rvol_15,tbr_15,pos_15,pos_30,
              fix_pos,trend
    返回 float32。分块构造避免 OOM。
    """
    n = len(raw_positions)
    out = np.zeros((n, look_back, F_DIM), dtype=np.float32)
    c_idx = F["c"]
    f = {k: F[k] for k in ["lr1","lr5","rvol","tbr","cvd","body","wick_u","wick_l","c_hl","vol_rel",
                            "stoch_k_50","di_plus","di_minus",
                            "lr_15","lr_30","rvol_5","rvol_15","tbr_15","pos_15","pos_30"]}
    BLK = 10_000
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
                                            "body","wick_u","wick_l","c_hl","vol_rel",
                                            "stoch_k_50","di_plus","di_minus",
                                            "lr_15","lr_30","rvol_5","rvol_15","tbr_15","pos_15","pos_30"]]
        # fix_pos: 步索引 0..LB-1, 归一化到 [0,1] (避免与收益类特征量纲差几个数量级)
        st_fix = np.broadcast_to((np.arange(look_back, dtype=np.float32))[None, :] / float(look_back-1),
                                 blk_list[0].shape)
        # trend: 已由 feats 因果滚动标准化(绝对涨幅), 直接取历史步
        st_trend = F["trend_c"][row_pos]
        blk_list += [st_fix, st_trend]
        # 组装 [B, LB, 22]
        blk = np.stack(blk_list, axis=-1)
        blk = np.nan_to_num(blk, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        # 无效行(头部不足look_back)置零
        if not valid.all():
            blk[~valid] = 0.0
        out[b0:b1] = blk
    return out


class AttentionPool(nn.Module):
    """自注意力池化: 对序列所有时间步做加权平均，关注关键时间步。"""
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Linear(dim, 1, bias=False)
    def forward(self, x):
        # x: [B, T, D]
        w = self.attn(x)                          # [B, T, 1]
        w = torch.softmax(w, dim=1)               # [B, T, 1]
        return (w * x).sum(dim=1)                 # [B, D]


class LSTMModel(nn.Module):
    """BiLSTM(1层) + 自注意力 + LayerNorm + Dropout。
    
    改进(相对原版):
    - 自注意力池化替代最后步取法，关注关键时间步
    - 更好的权重初始化
    - Xavier/Orthogonal 初始化
    """
    def __init__(self, fan=F_DIM, hidden=HIDDEN):
        super().__init__()
        self.lstm = nn.LSTM(fan, hidden, num_layers=1, batch_first=True, bidirectional=True)
        self.attn = AttentionPool(hidden * 2)
        self.norm = nn.LayerNorm(hidden * 2)
        self.drop = nn.Dropout(0.25)
        self.head = nn.Linear(hidden * 2, 1)
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        for name, p in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(p.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(p.data)
            elif 'weight' in name and 'norm' not in name and 'attn' not in name:
                nn.init.xavier_uniform_(p.data)
            elif 'bias' in name:
                nn.init.zeros_(p.data)
    
    def forward(self, x):
        # x: [B, T, F]
        o, _ = self.lstm(x)                            # [B, T, 2*hidden]
        # 自注意力池化
        pooled = self.attn(o)                           # [B, 2*hidden]
        pooled = self.norm(pooled)
        return self.head(self.drop(pooled)).squeeze(-1)


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
        # 标签对齐: 用 ds 行的 label, 需 pos->ds 反查; 同时加载软标签
        ds_raw = ctx.ds_to_raw
        tr_ds = np.searchsorted(ds_raw, tr_pos); tr_ds = np.clip(tr_ds, 0, len(ctx.label)-1)
        ytr = ctx.label[tr_ds].astype(np.float32)
        ytr_soft = ctx.soft_label[tr_ds].astype(np.float32) if USE_SOFT_LABEL else ytr
        es_ds = np.searchsorted(ds_raw, es_pos); es_ds = np.clip(es_ds, 0, len(ctx.label)-1)
        yes = ctx.label[es_ds].astype(np.float32)
        # 构造序列
        print("构造训练序列...", flush=True); t0=time.time()
        Xtr = build_ds_matrix(F, tr_pos, self.look_back)
        print(f"  Xtr {Xtr.shape} {time.time()-t0:.0f}s", flush=True)
        vtr = Xtr.sum(axis=(1,2)) != 0
        Xtr, ytr = Xtr[vtr], ytr[vtr]
        ytr_soft = ytr_soft[vtr]
        del vtr
        gc.collect()
        print("构造验证序列...", flush=True)
        Xes = build_ds_matrix(F, es_pos, self.look_back)
        ves = Xes.sum(axis=(1,2)) != 0
        Xes, yes = Xes[ves], yes[ves]
        del ves
        gc.collect()
        print(f"  train {Xtr.shape} es {Xes.shape} label正例率 {ytr.mean():.3f} {yes.mean():.3f}", flush=True)

        Xt = torch.from_numpy(Xtr); yt = torch.from_numpy(ytr)
        del Xtr
        yt_soft = torch.from_numpy(ytr_soft)
        del ytr_soft
        Xe = torch.from_numpy(Xes); ye = torch.from_numpy(yes)
        del Xes
        gc.collect()
        self.model.train()
        opt = torch.optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-5)
        # 学习率调度: CosineAnnealing
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-5)
        lossf = nn.BCEWithLogitsLoss()
        n = len(Xt); nB = (n + BATCH - 1)//BATCH
        best = 1e9; best_state = None; patience=6; bad=0
        for ep in range(EPOCHS):
            perm = torch.randperm(n)
            self.model.train(); tot=0.0
            opt.zero_grad()
            for i in range(nB):
                idx = perm[i*BATCH:(i+1)*BATCH]
                xb = Xt[idx]; yb = yt_soft[idx] if USE_SOFT_LABEL else yt[idx]  # 训练用软标签
                out = self.model(xb)
                loss = lossf(out, yb) / GRAD_ACCUM_STEPS
                loss.backward()
                if (i + 1) % GRAD_ACCUM_STEPS == 0 or i == nB - 1:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    opt.step()
                    opt.zero_grad()
                tot += loss.item() * GRAD_ACCUM_STEPS * len(idx)
            scheduler.step()
            # 验证 (用二元标签评估)
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
            lr_now = scheduler.get_last_lr()[0]
            print(f"[lstm] ep{ep} tr_loss={tot/n:.4f} es_loss={el:.4f} es_acc={acc_es:.4f} lr={lr_now:.2e}", flush=True)
            if el < best:
                best = el; bad = 0
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                bad += 1
                if bad >= patience: break
        if best_state: self.model.load_state_dict(best_state)
        del Xt, Xe, ye, yt, yt_soft
        gc.collect()
        self.fitted = True
        print("[lstm] fit done", flush=True)

    def fit_from_data(self, Xtr, ytr, Xes, yes, ytr_soft=None):
        """直接从预构建的序列数据训练(用于重采样实验)。
        Args:
            ytr_soft: 可选的软标签(numpy), 若提供则用软标签训练
        """
        torch.set_num_threads(config.N_JOBS)
        ytr = ytr.astype(np.float32); yes = yes.astype(np.float32)
        if ytr_soft is None:
            ytr_soft = ytr
        ytr_soft = ytr_soft.astype(np.float32)
        Xt = torch.from_numpy(Xtr); yt = torch.from_numpy(ytr)
        yt_soft = torch.from_numpy(ytr_soft)
        Xe = torch.from_numpy(Xes); ye = torch.from_numpy(yes)
        self.model.train()
        opt = torch.optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-5)
        lossf = nn.BCEWithLogitsLoss()
        n = len(Xt); nB = (n + BATCH - 1)//BATCH
        best = 1e9; best_state = None; patience=6; bad=0
        for ep in range(EPOCHS):
            perm = torch.randperm(n)
            self.model.train(); tot=0.0
            opt.zero_grad()
            for i in range(nB):
                idx = perm[i*BATCH:(i+1)*BATCH]
                xb = Xt[idx]; yb = yt_soft[idx] if USE_SOFT_LABEL else yt[idx]
                out = self.model(xb)
                loss = lossf(out, yb) / GRAD_ACCUM_STEPS
                loss.backward()
                if (i + 1) % GRAD_ACCUM_STEPS == 0 or i == nB - 1:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    opt.step()
                    opt.zero_grad()
                tot += loss.item() * GRAD_ACCUM_STEPS * len(idx)
            scheduler.step()
            self.model.eval()
            with torch.no_grad():
                el = 0.0; correct = 0; nacc = 0
                nel = len(Xe)
                for b0 in range(0, nel, BATCH*4):
                    b1 = min(b0+BATCH*4, nel)
                    yeo = self.model(Xe[b0:b1])
                    el += lossf(yeo, ye[b0:b1]).item() * (b1-b0)
                    correct += ((torch.sigmoid(yeo)>=0.5).float()==ye[b0:b1]).float().sum().item()
                    nacc += (b1-b0)
                el /= nel
                acc_es = correct / max(1, nacc)
            lr_now = scheduler.get_last_lr()[0]
            print(f"[lstm] ep{ep} tr_loss={tot/n:.4f} es_loss={el:.4f} es_acc={acc_es:.4f} lr={lr_now:.2e}", flush=True)
            if el < best:
                best = el; bad = 0
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                bad += 1
                if bad >= patience: break
        if best_state: self.model.load_state_dict(best_state)
        self.fitted = True
        print("[lstm] fit_from_data done", flush=True)

    def predict(self, ctx, split):
        if not self.fitted: raise RuntimeError("fit first")
        F = self._prep(ctx)
        pos = ctx.ds_to_raw[ctx.split_rows[split]]
        n = len(pos)
        out = np.zeros(n, dtype=np.float32)
        BLK = 10000  # 分块预测避免OOM
        self.model.eval()
        with torch.no_grad():
            for b0 in range(0, n, BLK):
                b1 = min(b0 + BLK, n)
                Xb = build_ds_matrix(F, pos[b0:b1], self.look_back)
                Xt = torch.from_numpy(Xb)
                out[b0:b1] = torch.sigmoid(self.model(Xt)).numpy().squeeze()
                del Xb, Xt
        return out.astype(np.float32)