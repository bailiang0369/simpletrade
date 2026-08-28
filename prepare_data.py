"""把下载的月度zip解析、拼接、去重、排序，输出每交易对的原始parquet。

全部使用 polars；存储 dtype 统一为 float32(价格/量) + int64(时间戳)。
清洗规则(防"少量null/脏数据"):
- 核心列含 null 的行直接剔除
- 非正价格 / 高低价倒挂 的行剔除
- 负成交量 / 负主动买量 clip 到 0
保留列: open_time, open, high, low, close, volume, taker_buy_base
(volume 仅用于内部推导主动卖量，不会被用作模型特征)。
"""
import glob
import os
import time
import zipfile

import polars as pl

import config

COLUMNS = ["open_time", "open", "high", "low", "close", "volume",
           "close_time", "quote_asset_volume", "number_of_trades",
           "taker_buy_base", "taker_buy_quote", "ignore"]

KEEP = ["open_time", "open", "high", "low", "close", "volume", "taker_buy_base"]

PRICE_COLS = ["open", "high", "low", "close"]
VOL_COLS = ["volume", "taker_buy_base"]


def read_month_zip(path, symbol):
    """读取单个月度zip，返回清洗后的 polars DataFrame。"""
    with zipfile.ZipFile(path) as z:
        name = z.namelist()[0]
        df = pl.read_csv(z.open(name), has_header=False, new_columns=COLUMNS)
    # ---- null 处理: 核心列含 null 的行直接剔除 ----
    df = df.drop_nulls(subset=KEEP)
    # ---- 时间戳单位探测: data.binance.vision 历史文件曾在毫秒/微秒间切换
    #      2020年代约1.57e12(ms), 后期约1.73e15(us) ----
    unit_us = df["open_time"].max() > 1e14
    df = df.with_columns(
        ((pl.col("open_time") // (1_000_000 if unit_us else 1_000)).cast(pl.Int64)).alias("ts")
    )
    return df


def prepare_symbol(symbol, market=None, overwrite=False):
    market = market or config.MARKET
    out = os.path.join(config.DS_DIR, f"raw_{symbol}.parquet")
    if os.path.exists(out) and not overwrite:
        print(f"[prepare] {symbol} already exists: {out}")
        return out
    pat = os.path.join(config.RAW_DIR, market, symbol, "1m", f"{symbol}-1m-*.zip")
    files = sorted(glob.glob(pat))
    if not files:
        raise FileNotFoundError(f"no monthly zips for {symbol} in {pat}")

    parts = []
    for f in files:
        parts.append(read_month_zip(f, symbol))
    df = pl.concat(parts, how="vertical_relaxed")
    del parts
    # ---- 去重(按交易所原始 open_time) + 排序 ----
    df = df.unique(subset="open_time", keep="first").sort("open_time")
    # ---- 类型: 价格/量 -> float32, 时间 -> int64 ----
    df = df.with_columns(
        [pl.col(c).cast(pl.Float32) for c in PRICE_COLS + VOL_COLS]
    )
    # ---- 异常清洗 ----
    df = df.filter(
        (pl.col("open") > 0) & (pl.col("high") > 0) &
        (pl.col("low") > 0) & (pl.col("close") > 0) &
        (pl.col("high") >= pl.col("low")) &
        (pl.col("high") >= pl.col("open")) &
        (pl.col("low") <= pl.col("close"))
    )
    df = df.with_columns(
        [pl.col(c).clip(0.0).alias(c) for c in VOL_COLS]
    )
    # ---- 输出: 只保留需要列 ----
    df = df.select(["ts", "open", "high", "low", "close", "volume", "taker_buy_base"])
    df.write_parquet(out, compression="zstd")
    print(f"[prepare] {symbol}: {df.height} rows, dtypes={df.schema} -> {out}")
    return out


def prepare_all(symbols=None):
    symbols = symbols or config.SYMBOLS
    t0 = time.time()
    for s in symbols:
        prepare_symbol(s)
    print(f"[prepare] all done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    prepare_all()
