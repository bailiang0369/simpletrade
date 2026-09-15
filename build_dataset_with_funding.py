#!/usr/bin/env python3
"""重建 ds_ETH_h15.parquet, 加入 funding 特征 + 恢复部分被移除特征。"""
import os, sys, gc
sys.path.insert(0, '/workspace')
import config
from data_processing.build_dataset import build_cross_features, CHUNK_ROWS, WARMUP, SOFT_LABEL_SCALE
import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

def build_features_with_funding(df: pl.DataFrame) -> pl.DataFrame:
    """在原 features.py 基础上增加 funding 特征, 并恢复部分低 gain 特征。"""
    from features import build_features as _orig_build
    # 先运行原特征得到 DataFrame (包含所有基础特征列)
    orig_df = _orig_build(df)
    
    C = df["close"].to_numpy(); TB = df["buy_vol"].to_numpy(); TS = df["sell_vol"].to_numpy()
    F = df["funding"].to_numpy(); ts = df["ts"].to_numpy()
    n = len(C)
    lr_1 = np.zeros(n, dtype=np.float64); lr_1[1:] = np.log(C[1:] / C[:-1])
    
    new_feats = {}
    
    # ---- 恢复部分被移除的动量特征 ----
    new_feats["lr_5"] = (pl.Series("lr_5", np.zeros(n, dtype=np.float32))).to_frame()
    # 用 numpy 计算更简单
    def _lr(w):
        col = np.zeros(n, dtype=np.float32)
        col[w:] = np.log(C[w:] / C[:-w]).astype(np.float32)
        return col
    new_feats["lr_5"] = _lr(5)
    new_feats["lr_30"] = _lr(30)
    new_feats["lr_720"] = _lr(720)
    new_feats["lr_1440"] = _lr(1440)
    new_feats["lr_2880"] = _lr(2880)
    
    # ---- z_10 (恢复) ----
    def _z(w):
        cs = np.cumsum(C); cs2 = np.cumsum(C**2)
        rm = np.zeros(n, dtype=np.float64); rv = np.zeros(n, dtype=np.float64)
        rm[w:] = (cs[w:] - cs[:-w]) / w
        rv[w:] = (cs2[w:] - cs2[:-w]) / w - rm[w:]**2
        rv[rv < 0] = 0; rs = np.sqrt(rv)
        z = (C - rm) / (rs + 1e-8)
        return z.astype(np.float32)
    new_feats["z_10"] = _z(10)
    new_feats["z_720"] = _z(720)
    new_feats["z_1440"] = _z(1440)
    
    # ---- rvol_ratio_60_5 ----
    def _rvol(w):
        cs = np.cumsum(lr_1); cs2 = np.cumsum(lr_1**2)
        rm = np.zeros(n, dtype=np.float64); rv = np.zeros(n, dtype=np.float64)
        rm[w:] = (cs[w:] - cs[:-w]) / w
        rv[w:] = (cs2[w:] - cs2[:-w]) / w - rm[w:]**2
        rv[rv < 0] = 0
        return (np.sqrt(rv * w / (w-1)) * 100).astype(np.float32)
    rvol_60 = _rvol(60); rvol_5 = _rvol(5)
    new_feats["rvol_ratio_60_5"] = (rvol_60 / (rvol_5 + 1e-8)).astype(np.float32)
    new_feats["rvol_720"] = _rvol(720)
    new_feats["rvol_1440"] = _rvol(1440)
    
    # ---- cvd_120 ----
    def _cvd(w):
        diff = TB - TS; total = TB + TS + 1e-8
        cs_diff = np.cumsum(diff); cs_total = np.cumsum(total)
        cvd = np.zeros(n, dtype=np.float64)
        cvd[w:] = (cs_diff[w:] - cs_diff[:-w]) / (cs_total[w:] - cs_total[:-w])
        return cvd.astype(np.float32)
    new_feats["cvd_120"] = _cvd(120)
    new_feats["cvd_1200"] = _cvd(1200)
    
    # ---- funding 特征 ----
    def _rolling_mean(arr, w):
        cs = np.cumsum(arr)
        rm = np.zeros(n, dtype=np.float64)
        rm[w:] = (cs[w:] - cs[:-w]) / w
        return rm
    def _rolling_std(arr, w):
        cs = np.cumsum(arr); cs2 = np.cumsum(arr**2)
        rm = np.zeros(n, dtype=np.float64); rv = np.zeros(n, dtype=np.float64)
        rm[w:] = (cs[w:] - cs[:-w]) / w
        rv[w:] = (cs2[w:] - cs2[:-w]) / w - rm[w:]**2
        rv[rv < 0] = 0; rs = np.sqrt(rv)
        return rs
    f_mean60 = _rolling_mean(F, 60); f_std60 = _rolling_std(F, 60)
    f_mean240 = _rolling_mean(F, 240); f_std240 = _rolling_std(F, 240)
    new_feats["funding_mean_60"] = f_mean60.astype(np.float32)
    new_feats["funding_std_60"] = f_std60.astype(np.float32)
    new_feats["funding_z_60"] = ((F - f_mean240) / (f_std240 + 1e-8)).astype(np.float32)
    new_feats["funding_diff_30"] = (F - np.roll(F, 30)).astype(np.float32)
    new_feats["funding_extreme_pos"] = (F > f_mean240 + 2 * f_std240).astype(np.float32)
    new_feats["funding_extreme_neg"] = (F < f_mean240 - 2 * f_std240).astype(np.float32)
    
    # 合并: orig_df (polars) + 新 numpy 数组转 polars Series
    new_cols = [pl.Series(name, arr) for name, arr in new_feats.items()]
    return orig_df.with_columns(new_cols)


def main():
    symbol = "ETH"; horizon = 15
    out = os.path.join(config.DS_DIR, f"ds_{symbol}_h{horizon}_v2.parquet")
    print(f"重建数据集: {out}", flush=True)
    
    raw_full = pl.read_parquet(os.path.join(config.DS_DIR, f"raw_{symbol}.parquet")).sort("ts")
    n = raw_full.height
    ts_sec = raw_full["ts"].to_numpy().astype(np.int64)
    close = raw_full["close"].to_numpy().astype(np.float32)
    
    # 标签
    label = np.zeros(n, dtype=np.int8)
    ret_future = np.full(n, np.nan, dtype=np.float32)
    soft_label = np.full(n, np.nan, dtype=np.float32)
    if horizon < n:
        fut = close[horizon:]; cur = close[:-horizon]
        label[:-horizon] = (fut > cur).astype(np.int8)
        with np.errstate(divide="ignore", over="ignore"):
            ret = np.log(fut / cur).astype(np.float32)
            ret_future[:-horizon] = ret
            ret_clipped = np.clip(ret / SOFT_LABEL_SCALE, -10, 10)
            soft_label[:-horizon] = (1.0 / (1.0 + np.exp(-ret_clipped))).astype(np.float32)
    
    # 跨资产特征
    X_cross, cross_names = build_cross_features(symbol, ts_sec)
    print(f"跨资产特征: {len(cross_names)} 列", flush=True)
    
    # 探测特征列数
    probe = raw_full.head(300)
    F_probe = build_features_with_funding(probe)
    feat_names = list(F_probe.columns) + cross_names
    nfeat = len(feat_names)
    print(f"基础特征: {len(F_probe.columns)}, 总特征: {nfeat}", flush=True)
    del probe, F_probe; gc.collect()
    
    pa_schema = pa.schema([(c, pa.float32()) for c in feat_names] + [
        ("label", pa.int8()), ("soft_label", pa.float32()),
        ("ret_future", pa.float32()), ("ts", pa.int64()),
    ])
    writer = None; writer_vn = 0
    
    for start in range(0, n, CHUNK_ROWS):
        s = max(0, start - WARMUP); e = min(n, start + CHUNK_ROWS + WARMUP)
        off = start - s; keep_n = min(CHUNK_ROWS, n - start)
        sub_df = raw_full.slice(s, e - s)
        F = build_features_with_funding(sub_df).to_numpy()
        del sub_df; gc.collect()
        F = F[off:off + keep_n]
        F = np.concatenate([F, X_cross[start:start + keep_n]], axis=1)
        row_valid = (np.isfinite(F).all(axis=1) & np.isfinite(ret_future[start:start + keep_n])
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
            writer.write_table(ta); writer_vn += len(ri); del ta, arrays
        del F, row_valid; gc.collect()
        if (start // CHUNK_ROWS + 1) % 50 == 0:
            print(f"  chunk {start // CHUNK_ROWS + 1}: keep={writer_vn}", flush=True)
    if writer: writer.close()
    print(f"完成! 总行数: {writer_vn}", flush=True)

if __name__ == "__main__":
    main()
