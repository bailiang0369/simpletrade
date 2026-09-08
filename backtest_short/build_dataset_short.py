#!/usr/bin/env python3
"""短周期(3/5分钟)数据集构建: 复用 build_dataset.build_symbol_dataset 的完整分块特征管线,
仅把 horizon 换成 3/5 并把输出写成 ds_{SYM}_h{horizon}.parquet(隔离, 不覆盖 30min 数据集)。

特征与标签无耦合: build_features 纯滚动统计, horizon 只影响 label/soft_label/ret_future。
用法:
  python -m backtest_short.build_dataset_short --horizons 3 5 --symbols ETH BTC
"""
import os, sys, gc, argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from data_processing.build_dataset import (build_cross_features, CHUNK_ROWS, WARMUP,
                                           SOFT_LABEL_SCALE)

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from features import build_features


def build_short(symbol, horizon, overwrite=False):
    out = os.path.join(config.DS_DIR, f"ds_{symbol}_h{horizon}.parquet")
    if os.path.exists(out) and not overwrite:
        print(f"[short] {symbol}/h{horizon}: exists, skip.", flush=True)
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
    taker_buy = raw["buy_vol"].to_numpy().astype(np.float32)
    sell = raw["sell_vol"].to_numpy().astype(np.float32)
    funding = raw["funding"].to_numpy().astype(np.float32)
    del raw
    gc.collect()

    # ---- 标签与未来收益 (短周期 horizon) ----
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
            ret_clipped = np.clip(ret / SOFT_LABEL_SCALE, -10, 10)
            soft_label[:-horizon] = (1.0 / (1.0 + np.exp(-ret_clipped))).astype(np.float32)

    # ---- 跨资产特征 ----
    X_cross, cross_names = build_cross_features(symbol, ts_sec)
    print(f"[short] {symbol}/h{horizon}: 跨资产 {len(cross_names)} 列", flush=True)

    # ---- 探测特征列数 ----
    probe = pl.from_dict({
        "ts": ts_sec[:300], "open": close[:300], "high": close[:300],
        "low": close[:300], "close": close[:300],
        "buy_vol": np.ones(300, np.float32) * 0.5, "sell_vol": np.ones(300, np.float32) * 0.5,
        "funding": np.ones(300, np.float32) * 0.0001,
    })
    probe = build_features(probe)
    feat_names = list(probe.columns) + cross_names
    nfeat = len(feat_names)
    del probe, cross_names
    gc.collect()

    pa_schema = pa.schema([(c, pa.float32()) for c in feat_names] + [
        ("label", pa.int8()), ("soft_label", pa.float32()),
        ("ret_future", pa.float32()), ("ts", pa.int64()),
    ])
    writer = None
    writer_vn = 0
    for start in range(0, n, CHUNK_ROWS):
        s = max(0, start - WARMUP)
        e = min(n, start + CHUNK_ROWS + WARMUP)
        off = start - s
        keep_n = min(CHUNK_ROWS, n - start)
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
        F = F[off:off + keep_n]
        F = np.concatenate([F, X_cross[start:start + keep_n]], axis=1)
        row_valid = (np.isfinite(F).all(axis=1)
                     & np.isfinite(ret_future[start:start + keep_n])
                     & (ret_future[start:start + keep_n] != 0.0))
        ri = np.where(row_valid)[0]
        if len(ri):
            r_abs = start + ri
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
        print(f"[short] {symbol}/h{horizon} chunk "
              f"{start // CHUNK_ROWS + 1}: keep={writer_vn}", flush=True)
    del X_cross
    gc.collect()
    if writer is not None:
        writer.close()

    # 校验
    pf = pq.ParquetFile(out)
    names = pf.schema_arrow.names
    tot_null = 0
    for _nm in names:
        tot_null += pf.read(columns=[_nm]).column(0).null_count
    ratio = float(pf.read(columns=["label"]).column(0).to_numpy().mean())
    print(f"[short] {symbol}/h{horizon}: rows={writer_vn}, cols={len(names)}, "
          f"label_ratio={ratio:.4f}, null={tot_null}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", nargs="*", type=int, default=[3, 5])
    ap.add_argument("--symbols", nargs="*", default=list(config.SYMBOLS))
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    for h in a.horizons:
        for s in a.symbols:
            build_short(s, h, overwrite=a.overwrite)


if __name__ == "__main__":
    main()