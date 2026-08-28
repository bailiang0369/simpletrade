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
import polars as pl

import config
from features import build_features

CHUNK_ROWS = 1_000_000   # 每个特征计算块的原始行数
WARMUP = 400             # 保证滚动窗口在块首有足够历史
FEAT_DTYPES = {}         # {col: polars dtype} 由 probe 确定


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
    taker_buy = raw["taker_buy_base"].to_numpy().astype(np.float32)
    volume = raw["volume"].to_numpy().astype(np.float32)
    # 立即释放 polars 原始表(占约300-500MB), 只用 numpy
    del raw
    gc.collect()
    t0 = time.time()

    # ---- 标签与未来收益 (全局, 内存占用小) ----
    label = np.zeros(n, dtype=np.int8)
    ret_future = np.full(n, np.nan, dtype=np.float32)
    if horizon < n:
        fut = close[horizon:]
        cur = close[:-horizon]
        label[:-horizon] = (fut > cur).astype(np.int8)
        with np.errstate(divide="ignore"):
            ret_future[:-horizon] = np.log(fut / cur).astype(np.float32)

    # ---- 探测特征列数(用 numpy 重建一个 probe 表) ----
    probe = pl.from_dict({
        "ts": ts_sec[:300], "open": close[:300], "high": close[:300],
        "low": close[:300], "close": close[:300],
        "volume": np.ones(300, np.float32), "taker_buy_base": np.ones(300, np.float32) * 0.5,
    })
    probe = build_features(probe)
    feat_names = probe.columns
    nfeat = probe.width
    del probe
    gc.collect()
    # ---- 流式写输出: 每块算完特征即挑选有效行并追加, 峰值内存=单块 ----
    import pyarrow as pa
    import pyarrow.parquet as pq
    # ret_day 全局重算(依赖close)
    day_idx = ts_sec // 86400
    unique_days, first_idx = np.unique(day_idx, return_index=True)
    day_open_vals = close[first_idx]
    day_open_filled = day_open_vals[np.searchsorted(unique_days, day_idx)]
    ret_day = (close / np.where(day_open_filled > 0, day_open_filled, np.nan) - 1) * 100

    out_cols = feat_names + ["label", "ret_future", "ts"]   # ret_day 已在特征列中
    # 先探测编译列类型
    pa_schema = pa.schema([(c, pa.float32()) for c in feat_names] + [
        ("label", pa.int8()), ("ret_future", pa.float32()), ("ts", pa.int64()),
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
            "volume": volume[s:e],
            "taker_buy_base": taker_buy[s:e],
        })
        F = build_features(sub_df).to_numpy()
        del sub_df
        F = F[off:off + keep_n]                       # (keep_n, nfeat)
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
    if writer is not None:
        writer.close()
    vn_now = writer_vn

    print(f"[dataset] {symbol}: valid={vn_now}, "
          f"elapsed={time.time() - t0:.0f}s")
    # 校验写入结果零null + 类型
    ds = pl.read_parquet(out)
    assert ds.null_count().sum_horizontal().sum() == 0, "dataset still has nulls!"
    assert ds['label'].dtype == pl.Int8 and ds['ts'].dtype == pl.Int64
    print(f"[dataset] {symbol}: rows={ds.height}, cols={ds.width}, "
          f"label_ratio={float(ds['label'].mean()):.4f}, n_null=0[verified]")
    return out


def build_all():
    for s in config.SYMBOLS:
        build_symbol_dataset(s)


if __name__ == "__main__":
    build_all()
