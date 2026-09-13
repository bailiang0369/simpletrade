"""
验证 bookTicker (best bid/ask 实时推送) 的订单簿特征能否提升 ETH 15min 预测.

bookTicker 独有特征 (raw K 线 + aggTrades 都算不出来):
  1. spread = best_ask_price - best_bid_price               ← 市场宽度
  2. bid_size_ratio = best_bid_qty / (best_bid_qty + best_ask_qty)  ← 盘口失衡 (order book imbalance 最基础版)
  3. mid_price = (bid + ask) / 2                            ← 微观中间价
  4. spread_change = spread[t] - spread[t-1]                ← 宽度变化
  5. bid_size_change, ask_size_change                       ← 挂单量变化
  
这些都是微观结构特征, 直接反映做市商行为和流动性.
"""
import glob, zipfile, os
import numpy as np
import polars as pl

BOOKTICKER_DIR = "/tmp"
RAW_PATH = "data/datasets/raw_ETH.parquet"
RAW_BTC_PATH = "data/datasets/raw_BTC.parquet"


def load_bookticker(pattern):
    """加载所有 bookTicker zip, 返回 1min 聚合特征."""
    dfs = []
    for z in sorted(glob.glob(pattern)):
        with zipfile.ZipFile(z) as zf:
            with zf.open(zf.namelist()[0]) as f:
                df = pl.read_csv(f, has_header=True,
                    schema_overrides={
                        'best_bid_price': pl.Float64, 'best_bid_qty': pl.Float64,
                        'best_ask_price': pl.Float64, 'best_ask_qty': pl.Float64,
                        'transaction_time': pl.Int64, 'event_time': pl.Int64,
                    })
        dfs.append(df)
    df = pl.concat(dfs)
    print(f"  bookTicker rows: {df.height:,}")
    
    # ts = event_time/1000 秒级, K 线 open_time 向下取整到分钟
    # raw.ts = open_time + 60, 所以聚合到 floor(event_time/1e3/60)*60 + 60
    df = df.with_columns([
        (pl.col('event_time') // 60_000).alias('kline_open_min'),  # 分钟级戳
        (pl.col('best_ask_price') - pl.col('best_bid_price')).alias('spread'),
        ((pl.col('best_bid_price') + pl.col('best_ask_price')) / 2).alias('mid_price'),
        (pl.col('best_bid_qty') / (pl.col('best_bid_qty') + pl.col('best_ask_qty') + 1e-9)).alias('bid_size_ratio'),
        (pl.col('best_bid_qty').log() + 1e-9).alias('log_bid_qty'),
        (pl.col('best_ask_qty').log() + 1e-9).alias('log_ask_qty'),
    ])
    
    g = df.group_by('kline_open_min').agg([
        # spread 统计
        pl.col('spread').mean().alias('spread_mean'),
        pl.col('spread').std().alias('spread_std'),
        pl.col('spread').max().alias('spread_max'),
        # bid/ask size 统计
        pl.col('bid_size_ratio').mean().alias('bid_size_ratio_mean'),
        pl.col('bid_size_ratio').std().alias('bid_size_ratio_std'),
        pl.col('best_bid_qty').mean().alias('bid_qty_mean'),
        pl.col('best_ask_qty').mean().alias('ask_qty_mean'),
        # mid price 统计
        pl.col('mid_price').last().alias('mid_price_last'),
        pl.col('mid_price').first().alias('mid_price_first'),
        # 更新频率
        pl.len().alias('n_updates'),
    ]).sort('kline_open_min')
    
    # 对齐 raw.ts
    g = g.with_columns(
        (pl.col('kline_open_min') * 60 + 60).alias('ts')
    )
    
    # 滚动变化 (一阶差分, 1min → h15 时再滚动)
    g = g.with_columns([
        (pl.col('spread_mean').diff().fill_null(0)).alias('spread_delta'),
        (pl.col('bid_size_ratio_mean').diff().fill_null(0)).alias('bsr_delta'),
        (pl.col('mid_price_last') - pl.col('mid_price_first')).alias('mid_change'),
    ])
    
    return g


def compute_h15_features(bt_1min, close_1min):
    """1min bookTicker → 15min 对齐, 计算滚动特征."""
    # 1min close (raw.ts 对齐)
    n = len(close_1min)
    
    # 把 bookTicker 特征 reindex 到 raw 的所有 ts 上 (缺失前向填充)
    ts_to_row = {int(t): i for i, t in enumerate(close_1min['ts'].to_numpy())}
    
    bt_np = {c: bt_1min[c].to_numpy() for c in bt_1min.columns if c != 'kline_open_min'}
    bt_ts = bt_np['ts']
    
    # 构建对齐数组 (和 raw 等长)
    aligned = {}
    for c in bt_np:
        arr = np.full(n, np.nan)
        for i, t in enumerate(bt_ts):
            idx = ts_to_row.get(int(t))
            if idx is not None:
                arr[idx] = bt_np[c][i]
        # 前向填充
        mask = np.isnan(arr)
        last = np.nan
        for i in range(n):
            if not np.isnan(arr[i]):
                last = arr[i]
            elif not np.isnan(last):
                arr[i] = last
        aligned[c] = arr
    
    print(f"  对齐后 bookTicker 特征覆盖: {sum(~np.isnan(aligned['spread_mean']))} / {n} rows")
    
    return aligned


def test_group_diff(aligned_feats, close, horizon=15):
    """分组测试: 每个特征的极端组 vs next horizon return 差异."""
    future_ret = close[horizon:] / close[:-horizon] - 1  # 长度 n-horizon
    
    results = {}
    for fn, feat in aligned_feats.items():
        # 去掉前向填充产生的 NaN 前缀
        valid = ~np.isnan(feat) & (np.arange(len(feat)) < len(future_ret))
        f = feat[valid]
        r = future_ret[valid]
        
        if len(f) < 100:
            continue
        
        try:
            q25, q75 = np.percentile(f, [25, 75])
        except:
            continue
        
        lo = f <= q25
        hi = f > q75
        
        ret_lo = np.mean(r[lo]) if lo.sum() > 10 else np.nan
        ret_hi = np.mean(r[hi]) if hi.sum() > 10 else np.nan
        up_lo = np.mean(r[lo] > 0) if lo.sum() > 10 else np.nan
        up_hi = np.mean(r[hi] > 0) if hi.sum() > 10 else np.nan
        
        results[fn] = {
            'Δret': (ret_hi - ret_lo) * 100,
            'Δup': (up_hi - up_lo) * 100,
            'n_lo': lo.sum(), 'n_hi': hi.sum(),
            'ret_lo': ret_lo * 100, 'ret_hi': ret_hi * 100,
        }
    
    return results


def main():
    print("=" * 70)
    print("Step 1: 加载 bookTicker")
    print("=" * 70)
    
    eth_bt = load_bookticker("/tmp/ETHUSDT-bookTicker-2024-01-*.zip")
    btc_bt = load_bookticker("/tmp/BTCUSDT-bookTicker-2024-01-*.zip")
    
    print(f"\n  ETH bookTicker 1min bars: {eth_bt.height}, ts: {eth_bt['ts'].min()} → {eth_bt['ts'].max()}")
    print(f"  BTC bookTicker 1min bars: {btc_bt.height}, ts: {btc_bt['ts'].min()} → {btc_bt['ts'].max()}")
    
    print("\n" + "=" * 70)
    print("Step 2: 加载 raw 1min close, 对齐 bookTicker")
    print("=" * 70)
    
    raw_eth = pl.read_parquet(RAW_PATH)
    raw_btc = pl.read_parquet(RAW_BTC_PATH)
    
    eth_aligned = compute_h15_features(eth_bt, raw_eth)
    btc_aligned = compute_h15_features(btc_bt, raw_btc)
    
    print("\n" + "=" * 70)
    print("Step 3: ETH 15min 分组测试")
    print("=" * 70)
    
    close_eth = raw_eth['close'].to_numpy()
    results_eth = test_group_diff(eth_aligned, close_eth, horizon=15)
    
    print(f"\n{'Feature':<25s} {'Δret(%)':>10s} {'Δup(%)':>10s} {'Q1_ret':>10s} {'Q4_ret':>10s} {'sig':>6s}")
    print("-" * 75)
    for fn, r in sorted(results_eth.items(), key=lambda x: -abs(x[1]['Δret'])):
        sig = "★" if abs(r['Δret']) > 0.05 else ""
        print(f"{fn:<25s} {r['Δret']:>+10.4f} {r['Δup']:>+10.2f} {r['ret_lo']:>+10.4f} {r['ret_hi']:>+10.4f} {sig:>6s}")
    
    print("\n" + "=" * 70)
    print("Step 4: BTC 15min 分组测试")
    print("=" * 70)
    
    close_btc = raw_btc['close'].to_numpy()
    results_btc = test_group_diff(btc_aligned, close_btc, horizon=15)
    
    print(f"\n{'Feature':<25s} {'Δret(%)':>10s} {'Δup(%)':>10s} {'Q1_ret':>10s} {'Q4_ret':>10s} {'sig':>6s}")
    print("-" * 75)
    for fn, r in sorted(results_btc.items(), key=lambda x: -abs(x[1]['Δret'])):
        sig = "★" if abs(r['Δret']) > 0.05 else ""
        print(f"{fn:<25s} {r['Δret']:>+10.4f} {r['Δup']:>+10.2f} {r['ret_lo']:>+10.4f} {r['ret_hi']:>+10.4f} {sig:>6s}")
    
    print("\n" + "=" * 70)
    print("Step 5: 关键结论")
    print("=" * 70)
    
    significant_eth = [(fn, r) for fn, r in results_eth.items() if abs(r['Δret']) > 0.05]
    significant_btc = [(fn, r) for fn, r in results_btc.items() if abs(r['Δret']) > 0.05]
    
    print(f"\n  ETH: {len(significant_eth)} 个特征有显著分组差 (|Δret| > 0.05%)")
    for fn, r in significant_eth[:5]:
        print(f"    {fn:<25s}: Δret = {r['Δret']:+.4f}%, Δup = {r['Δup']:+.2f}%")
    
    print(f"\n  BTC: {len(significant_btc)} 个特征有显著分组差")
    for fn, r in significant_btc[:5]:
        print(f"    {fn:<25s}: Δret = {r['Δret']:+.4f}%, Δup = {r['Δup']:+.2f}%")
    
    # 对照组: 用 raw buy_vol/sell_vol 的分组差
    print(f"\n  --- 对照组: 现有特征的分组差 ---")
    for name, aligned_raw in [('ETH', None)]:
        # 用 raw 自身的 buy_vol/sell_vol 测试
        bv = raw_eth['buy_vol'].to_numpy()
        sv = raw_eth['sell_vol'].to_numpy()
        ratio = bv / (bv + sv + 1e-9)
        future_ret = close_eth[15:] / close_eth[:-15] - 1
        q25, q75 = np.percentile(ratio, [25, 75])
        lo = ratio <= q25
        hi = ratio > q75
        ret_lo = np.mean(future_ret[lo[:len(future_ret)]]) * 100
        ret_hi = np.mean(future_ret[hi[:len(future_ret)]]) * 100
        print(f"    ETH buy_vol_ratio: Δret = {ret_hi - ret_lo:+.4f}%")


if __name__ == "__main__":
    main()
