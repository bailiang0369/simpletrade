"""
验证 aggTrades 逐笔数据 vs raw 1min bar 的 buy_vol/sell_vol 一致性.
如果一致 → aggTrades 只是同一数据的更细粒度, 增量主要来自节奏特征
如果不一致 → aggTrades 可能有新信息 (比如不同聚合口径)
"""
import os, glob, zipfile
import numpy as np
import polars as pl
from datetime import datetime

AGG_DIR = "/tmp/aggtrades"
RAW_PATH = "data/datasets/raw_ETH.parquet"


def load_aggtrades_parquet():
    """加载所有 zip 里的 aggTrades, 返回一个 big polars df"""
    dfs = []
    for zippath in sorted(glob.glob(os.path.join(AGG_DIR, "*.zip"))):
        with zipfile.ZipFile(zippath) as zf:
            csvname = zf.namelist()[0]
            with zf.open(csvname) as f:
                df = pl.read_csv(f, dtypes={
                    "agg_trade_id": pl.Int64,
                    "price": pl.Float64,
                    "quantity": pl.Float64,
                    "first_trade_id": pl.Int64,
                    "last_trade_id": pl.Int64,
                    "transact_time": pl.Int64,
                    "is_buyer_maker": pl.Boolean,
                })
        print(f"  {os.path.basename(zippath)}: {df.height:,} rows")
        dfs.append(df)
    df = pl.concat(dfs)
    print(f"\nTotal aggTrades: {df.height:,}")
    return df


def main():
    print("=" * 60)
    print("Step 1: Load aggTrades zip files")
    print("=" * 60)
    agg = load_aggtrades_parquet()

    # agg_trade_time 是 ms 精度, 转成秒级 ts (对齐 raw bar 的秒精度)
    agg = agg.with_columns([
        (pl.col("transact_time") // 1000).alias("ts"),
        (pl.col("price") * pl.col("quantity")).alias("quote_amount"),
    ])

    print(f"\n  Time range: {agg['transact_time'].min()} → {agg['transact_time'].max()}")

    # 按 1min (60s) 聚合, 计算主动买卖 quote volume
    agg_1min = (
        agg.group_by("ts")
        .agg([
            pl.col("quote_amount").filter(pl.col("is_buyer_maker")).sum().alias("agg_buy_vol"),
            pl.col("quote_amount").filter(~pl.col("is_buyer_maker")).sum().alias("agg_sell_vol"),
            pl.len().alias("trade_count"),
        ])
        .sort("ts")
    )
    print(f"  1min agg bars: {agg_1min.height}")

    print("\n" + "=" * 60)
    print("Step 2: Load raw_ETH 1min bars")
    print("=" * 60)
    raw = pl.read_parquet(RAW_PATH)
    # 转 datetime 看看范围
    raw_first = raw["ts"][0]
    raw_last = raw["ts"][-1]
    print(f"  raw bar range: {raw_first} → {raw_last}")
    print(f"  total rows: {raw.height:,}")

    # 只取 aggTrades 覆盖的时间窗口
    agg_ts_min = agg["ts"].min()
    agg_ts_max = agg["ts"].max()
    raw_subset = raw.filter(
        (pl.col("ts") >= agg_ts_min) & (pl.col("ts") <= agg_ts_max)
    )
    print(f"  raw subset (aggTrades 范围): {raw_subset.height} rows")

    print("\n" + "=" * 60)
    print("Step 3: 对比 aggTrades 聚合 vs raw bar buy_vol/sell_vol")
    print("=" * 60)

    # inner join
    merged = raw_subset.join(agg_1min, on="ts", how="inner")
    print(f"  merged rows: {merged.height}")

    # 直接数值对比
    buy_diff = (merged["buy_vol"].to_numpy() - merged["agg_buy_vol"].to_numpy())
    sell_diff = (merged["sell_vol"].to_numpy() - merged["agg_sell_vol"].to_numpy())
    buy_rel = buy_diff / (merged["agg_buy_vol"].to_numpy() + 1e-9)
    sell_rel = sell_diff / (merged["agg_sell_vol"].to_numpy() + 1e-9)

    print(f"\n  buy_vol 对比:")
    print(f"    raw sum:     {merged['buy_vol'].sum():,.0f}")
    print(f"    agg sum:     {merged['agg_buy_vol'].sum():,.0f}")
    print(f"    sum rel diff: {abs(merged['buy_vol'].sum() - merged['agg_buy_vol'].sum()) / merged['agg_buy_vol'].sum() * 100:.6f}%")
    print(f"    单 bar 最大 rel diff:  {np.abs(buy_rel).max():.6f}")
    print(f"    单 bar median rel diff: {np.median(np.abs(buy_rel)):.6f}")

    print(f"\n  sell_vol 对比:")
    print(f"    raw sum:     {merged['sell_vol'].sum():,.0f}")
    print(f"    agg sum:     {merged['agg_sell_vol'].sum():,.0f}")
    print(f"    sum rel diff: {abs(merged['sell_vol'].sum() - merged['agg_sell_vol'].sum()) / merged['agg_sell_vol'].sum() * 100:.6f}%")
    print(f"    单 bar 最大 rel diff:  {np.abs(sell_rel).max():.6f}")
    print(f"    单 bar median rel diff: {np.median(np.abs(sell_rel)):.6f}")

    # Correlation
    buy_corr = np.corrcoef(merged["buy_vol"].to_numpy(), merged["agg_buy_vol"].to_numpy())[0, 1]
    sell_corr = np.corrcoef(merged["sell_vol"].to_numpy(), merged["agg_sell_vol"].to_numpy())[0, 1]
    print(f"\n  Correlation: buy={buy_corr:.6f}, sell={sell_corr:.6f}")

    print("\n" + "=" * 60)
    print("Step 4: aggTrades 独有的特征 → 大单检测 / 节奏")
    print("=" * 60)

    # 逐笔大单统计
    big_trade_thresh = 1000  # quote amount ≥ $1000 算大单
    big_trades = agg.filter(pl.col("quote_amount") >= big_trade_thresh)
    print(f"\n  大单阈值: ${big_trade_thresh:,}")
    print(f"  大单数量: {big_trades.height:,} ({big_trades.height/agg.height*100:.2f}%)")
    print(f"  大单占总成交量: {big_trades['quote_amount'].sum() / agg['quote_amount'].sum()*100:.1f}%")

    # 按 1min 聚合大单特征
    big_1min = (
        agg.with_columns([
            pl.when(pl.col("quote_amount") >= big_trade_thresh)
            .then(pl.col("quote_amount")).otherwise(0).alias("big_q"),
            pl.when((pl.col("quote_amount") >= big_trade_thresh) & pl.col("is_buyer_maker"))
            .then(pl.col("quote_amount")).otherwise(0).alias("big_buy_q"),
            pl.when((pl.col("quote_amount") >= big_trade_thresh) & ~pl.col("is_buyer_maker"))
            .then(pl.col("quote_amount")).otherwise(0).alias("big_sell_q"),
        ])
        .group_by("ts")
        .agg([
            pl.col("big_q").sum().alias("big_volume"),
            pl.col("big_buy_q").sum().alias("big_buy_volume"),
            pl.col("big_sell_q").sum().alias("big_sell_volume"),
            pl.len().alias("trades_per_min"),
            pl.col("quote_amount").sum().alias("total_volume"),
        ])
        .sort("ts")
    )
    print(f"\n  big_trades 1min 统计 (前10 bar):")
    print(big_1min.head(10))

    # 合并到 merged
    merged2 = merged.join(big_1min, on="ts", how="inner")

    # 大单比例 OFI 计算
    merged2 = merged2.with_columns([
        (pl.col("big_buy_volume") - pl.col("big_sell_volume")).alias("big_ofi"),
        (pl.col("big_buy_volume") + pl.col("big_sell_volume")).alias("big_total"),
        (pl.col("big_ofi") / (pl.col("big_total") + 1e-9)).alias("big_ofi_ratio"),
        (pl.col("trades_per_min") / pl.col("total_volume")).alias("trade_density"),
    ])

    print(f"\n  新增节奏特征:")
    for c in ["big_ofi_ratio", "big_volume", "trade_density", "trades_per_min"]:
        s = merged2[c].to_numpy()
        print(f"    {c:20s}: mean={np.mean(s):.4f}, std={np.std(s):.4f}, p50={np.median(s):.4f}")

    print("\n" + "=" * 60)
    print("结论")
    print("=" * 60)
    if np.abs(buy_corr) > 0.99 and np.abs(sell_corr) > 0.99:
        print("  ✅ aggTrades 聚合到 1min 与 raw bar buy_vol/sell_vol 几乎完全一致")
        print("  ⚠️  aggTrades 在 OFI 方向信号上增量 = 0")
        print("  💡 aggTrades 独有价值: 大单节奏、逐笔成交密度、连续买卖段长度")
    else:
        print(f"  ❓ 有差异: buy corr={buy_corr:.4f}, sell corr={sell_corr:.4f}")
        print("  → 需要检查数据源或聚合方式")


if __name__ == "__main__":
    main()
