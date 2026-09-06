"""组装特征数据集 (polars + float32/int): 原始parquet -> 特征+标签 -> 每交易对一个 parquet。

内存受限环境(5GB)专用: 按行分块计算特征(滚动窗口需warmup重叠), 避免一次性全量矩阵。
null/NaN 处理:
- 原始行已在 prepare 阶段剔除 null/脏数据
- 滚动窗口不足产生的 NaN 由 valid_mask 统一剔除
- 输出数据集保证零 null, 且全部为 float32/int
"""
import os
import time

import numpy as np
import pandas as pd
import polars as pl

import config
from features import build_features

CHUNK_ROWS = 500_000   # 每个特征计算块的原始行数(第6轮特征增加后峰值内存上升, 减半分块避免OOM)
WARMUP = 400             # 保证滚动窗口在块首有足够历史
FEAT_DTYPES = {}         # {col: polars dtype} 由 probe 确定
SOFT_LABEL_SCALE = 0.005  # 软标签温度参数: sigmoid(ret/scale), 0.5%=~0.73 1%=~0.88

# ---- 跨资产特征 (源币 -> 目标币) ----
# 经验证(experiment_cross_asset.py / experiment_cross_extend.py / experiment_cross_lr17.py, R2无泄漏):
#   12列基线 ETH+BTC +0.0109, BTC+ETH +0.0277, 两币同向为正=真实正交信号(beta传导)。
#   增强: 更长回看(lr_480/960, z_240/480, rvol_240) 17列, ETH +0.0365(0.6093→0.6457, 坏月3→1),
#   BTC +0.0047(0.6372→0.6419), 两币同向为正, 纳入管线。
#   比值价差 spread_z(19列) 被证伪: 两币方向不一致(ETH +0.010 / BTC -0.018), 按 funding 教训不纳入。
# 特征全部只用源币 t 及以前信息, 按目标币每个 raw 行 ts 对齐到源币最近 <= t 的行, 无未来泄漏。
CROSS_LR_WINDOWS = (5, 15, 30, 60, 120, 240, 480, 960)
CROSS_Z_WINDOWS = (30, 60, 120, 240, 480)
CROSS_RVOL_WINDOWS = (60, 240)
CROSS_CVD_WINDOWS = (30, 60)


def build_cross_features(symbol, ts_sec):
    """为目标币 symbol 构建源币因果特征, 对齐到每个目标币 raw 行 -> (n, F), 列名带源币前缀。"""
    import pyarrow.parquet as pq
    other = "BTC" if symbol == "ETH" else "ETH"
    t = pq.read_table(f"{config.DS_DIR}/raw_{other}.parquet",
                      columns=["ts", "close", "buy_vol", "sell_vol"])
    ots = t["ts"].to_numpy().astype(np.int64)
    oc = t["close"].to_numpy().astype(np.float64)
    ob = t["buy_vol"].to_numpy().astype(np.float64)
    os_ = t["sell_vol"].to_numpy().astype(np.float64)
    del t
    gc = __import__("gc")
    n_o = len(oc)
    lc = np.log(np.maximum(oc, 1e-12))
    cols, names = [], []

    def add(nm, arr):
        cols.append(arr.astype(np.float32))
        names.append(f"{other}_{nm}")

    for k in CROSS_LR_WINDOWS:
        r = np.full(n_o, np.nan)
        r[k:] = lc[k:] - lc[:-k]
        add(f"lr_{k}", r)
    s = pd.Series(lc)
    for w in CROSS_Z_WINDOWS:
        mu = s.rolling(w).mean().to_numpy()
        sd = s.rolling(w).std().to_numpy()
        add(f"z_{w}", np.where(sd > 1e-9, (lc - mu) / sd, 0.0))
    lr1 = np.full(n_o, np.nan)
    lr1[1:] = lc[1:] - lc[:-1]
    for w in CROSS_RVOL_WINDOWS:
        add(f"rvol_{w}", pd.Series(lr1).rolling(w).std().to_numpy() * 100)
    d = pd.Series(ob - os_)
    tt = pd.Series(ob + os_)
    for w in CROSS_CVD_WINDOWS:
        add(f"cvd_{w}", (d.rolling(w).sum() / (tt.rolling(w).sum() + 1e-12)).to_numpy())
    del s, d, tt, lc, lr1, oc, ob, os_
    gc.collect()

    F = np.stack(cols, axis=1).astype(np.float32)            # (n_o, F)
    del cols
    gc.collect()
    idx = np.searchsorted(ots, ts_sec, side="right") - 1      # 源币最近 <= t 的行
    idx = np.clip(idx, 0, n_o - 1)
    out = F[idx]                                              # (n, F)
    del F, idx, ots
    gc.collect()
    return out, names


def build_symbol_dataset(symbol, horizon=None, overwrite=False):
    horizon = horizon or config.HORIZON_MIN
    out = os.path.join(config.DS_DIR, f"ds_{symbol}.parquet")
    if os.path.exists(out) and not overwrite:
        print(f"[dataset] {symbol} exists, skip.")
        return out

    import gc
    raw = pl.read_parquet(os.path.join(config.DS_DIR, f"raw_{symbol}.parquet"))
    raw = raw.sort("ts")
    n = raw.height
    ts_sec = raw["ts"].to_numpy().astype(np.int64)
    close = raw["close"].to_numpy().astype(np.float32)
    open_ = raw["open"].to_numpy().astype(np.float32)
    high = raw["high"].to_numpy().astype(np.float32)
    low = raw["low"].to_numpy().astype(np.float32)
    taker_buy = raw["buy_vol"].to_numpy().astype(np.float32)      # 主动买量
    sell = raw["sell_vol"].to_numpy().astype(np.float32)          # 主动卖量
    funding = raw["funding"].to_numpy().astype(np.float32)        # 资金费率
    # 立即释放 polars 原始表(占约300-500MB), 只用 numpy
    del raw
    gc.collect()
    t0 = time.time()

    # ---- 标签与未来收益 (全局, 内存占用小) ----
    label = np.zeros(n, dtype=np.int8)
    ret_future = np.full(n, np.nan, dtype=np.float32)
    soft_label = np.full(n, np.nan, dtype=np.float32)
    if horizon < n:
        fut = close[horizon:]
        cur = close[:-horizon]
        label[:-horizon] = (fut > cur).astype(np.int8)
        with np.errstate(divide="ignore", over="ignore"):
            ret = np.log(fut / cur).astype(np.float32)
            ret_future[:-horizon] = ret
            # 软标签: sigmoid(ret / scale), 保留涨幅大小信息作为训练辅助信号
            # 最终评估仍用二元标签, 软标签只在训练阶段使用
            ret_clipped = np.clip(ret / SOFT_LABEL_SCALE, -10, 10)
            soft_label[:-horizon] = (1.0 / (1.0 + np.exp(-ret_clipped))).astype(np.float32)

    # ---- 跨资产特征 (源币 -> 目标币, 按目标币每个 raw 行 ts 对齐, 无泄漏) ----
    X_cross, cross_names = build_cross_features(symbol, ts_sec)   # (n, 12)
    print(f"[dataset] {symbol}: 跨资产特征 {len(cross_names)} 列: {cross_names}", flush=True)

    # ---- 探测特征列数(用 numpy 重建一个 probe 表) ----
    probe = pl.from_dict({
        "ts": ts_sec[:300], "open": close[:300], "high": close[:300],
        "low": close[:300], "close": close[:300],
        "buy_vol": np.ones(300, np.float32) * 0.5, "sell_vol": np.ones(300, np.float32) * 0.5,
        "funding": np.ones(300, np.float32) * 0.0001,
    })
    probe = build_features(probe)
    feat_names = list(probe.columns) + cross_names
    nfeat = len(feat_names)
    del probe
    gc.collect()
    # ---- 流式写输出: 每块算完特征即挑选有效行并追加, 峰值内存=单块 ----
    import pyarrow as pa
    import pyarrow.parquet as pq
    # 注: ret_day 由 build_features 分块内计算并经 WARMUP 保证块首有当日首行,
    #     只在极少数跨 1M 行块边界的"当日"上有 <=400 分钟的基准偏差(仍只用当日及以前, 无未来泄漏)。
    #     此处不再重算 ret_day（旧全局重算为死代码，已移除）。

    out_cols = feat_names + ["label", "soft_label", "ret_future", "ts"]   # ret_day 已在特征列中
    # 先探测编译列类型
    pa_schema = pa.schema([(c, pa.float32()) for c in feat_names] + [
        ("label", pa.int8()), ("soft_label", pa.float32()),
        ("ret_future", pa.float32()), ("ts", pa.int64()),
    ])
    writer_vn = 0
    writer = None
    for start in range(0, n, CHUNK_ROWS):
        s = max(0, start - WARMUP)
        e = min(n, start + CHUNK_ROWS + WARMUP)
        off = start - s
        keep_n = min(CHUNK_ROWS, n - start)
        # 用 numpy 重造原始行(避免持有完整 polars 表)
        sub_df = pl.from_dict({
            "ts": ts_sec[s:e],
            "open": open_[s:e], "high": high[s:e],
            "low": low[s:e], "close": close[s:e],
            "buy_vol": taker_buy[s:e],
            "sell_vol": sell[s:e],
            "funding": funding[s:e],
        })
        F = build_features(sub_df).to_numpy()
        del sub_df
        F = F[off:off + keep_n]                       # (keep_n, nfeat_base)
        # 拼入跨资产特征 (按 raw 行号对齐, 已是 float32; NaN 由 row_valid 统一剔除)
        F = np.concatenate([F, X_cross[start:start + keep_n]], axis=1)
        row_valid = (np.isfinite(F).all(axis=1)
                     & np.isfinite(ret_future[start:start + keep_n])
                     & (ret_future[start:start + keep_n] != 0.0))
        ri = np.where(row_valid)[0]
        if len(ri):
            r_abs = start + ri
            # 特征列用 float32; label/ret_future/ts 直接从原始 typed 数组构造,
            # 严禁经 float32 中转(时间戳 ~1e9 超出 float32 24位精度会损坏)。
            arrays = [pa.array(F[ri, j], type=pa_schema.field(j).type) for j in range(nfeat)]
            arrays.append(pa.array(label[r_abs], type=pa.int8()))
            arrays.append(pa.array(soft_label[r_abs], type=pa.float32()))
            arrays.append(pa.array(ret_future[r_abs], type=pa.float32()))
            arrays.append(pa.array(ts_sec[r_abs], type=pa.int64()))
            ta = pa.Table.from_arrays(arrays, schema=pa_schema)
            if writer is None:
                writer = pq.ParquetWriter(out, ta.schema, compression="zstd")
            writer.write_table(ta)
            writer_vn += len(ri)
            del ta, arrays
        del F, row_valid
        gc.collect()
        print(f"[dataset] {symbol} chunk {start // CHUNK_ROWS + 1}: "
              f"{keep_n} rows, cum_valid={writer_vn}", flush=True)
    del X_cross
    gc.collect()
    if writer is not None:
        writer.close()
    vn_now = writer_vn

    print(f"[dataset] {symbol}: valid={vn_now}, "
          f"elapsed={time.time() - t0:.0f}s")
    # 校验写入结果零null + 类型(内存受限: 按列流式读, 峰值=单列, 不整表载入)
    import pyarrow.parquet as _pq
    pf = _pq.ParquetFile(out)
    assert pf.metadata.num_rows == vn_now, "row count mismatch"
    names = pf.schema_arrow.names
    dt = {nm: pf.schema_arrow.field(nm).type for nm in names}
    assert dt['label'] == pa.int8() and dt['soft_label'] == pa.float32() and dt['ts'] == pa.int64()
    tot_null = 0
    for _nm in names:
        _a = pf.read(columns=[_nm]).column(0)
        tot_null += _a.null_count
    assert tot_null == 0, "dataset still has nulls!"
    print(f"[dataset] {symbol}: rows={vn_now}, cols={len(names)}, "
          f"label_ratio={float(pf.read(columns=['label']).column(0).to_numpy().mean()):.4f}, "
          f"n_null=0[verified]")
    return out


def build_all():
    for s in config.SYMBOLS:
        build_symbol_dataset(s)


if __name__ == "__main__":
    build_all()
