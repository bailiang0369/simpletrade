"""
最小实验: aggTrades 节奏特征 (不是方向信号, 是成交密度/大单/买卖节奏) 能否提升 ETH h15.
核心思路:
  - raw buy_vol/sell_vol = taker 方向 (aggTrades 聚合后 corr=1.0, 已被覆盖)
  - aggTrades 独有: n_trades, 大单节奏, 连续买卖段, 成交间隔
  - 3 天现货 aggTrades → 1min 聚合 → 对齐到 h15 ds_to_raw → 加特征跑 baseline + 加后特征
"""
import glob, zipfile
import numpy as np
import polars as pl
from datetime import datetime

AGG_PATTERN = "/tmp/aggtrades/ETHUSDT-spot-*.zip"
RAW_PATH = "data/datasets/raw_ETH.parquet"
DS_PATH = "data/datasets/ds_ETH_h15.parquet"

def load_spot_aggtrades(pattern):
    dfs = []
    for z in sorted(glob.glob(pattern)):
        with zipfile.ZipFile(z) as zf:
            with zf.open(zf.namelist()[0]) as f:
                df = pl.read_csv(f, has_header=False,
                    new_columns=['agg_trade_id','price','quantity','first_trade_id',
                                'last_trade_id','transact_time','is_buyer_maker','is_agg_trade'],
                    schema_overrides={'price': pl.Float64,'quantity': pl.Float64,
                                     'transact_time': pl.Int64,'is_buyer_maker': pl.Boolean})
        dfs.append(df)
    agg = pl.concat(dfs).with_columns([
        (pl.col('transact_time') // 60_000_000).alias('kline_open'),
        (pl.col('price') * pl.col('quantity')).alias('qa'),
    ])
    return agg

def build_rhythm_features(agg_df):
    """1min 聚合的节奏特征. 全 causal (只用 1min 内数据)."""
    # 1. 基础聚合
    g = agg_df.group_by('kline_open').agg([
        pl.col('quantity').filter(~pl.col('is_buyer_maker')).sum().alias('taker_buy_base'),
        pl.col('quantity').filter(pl.col('is_buyer_maker')).sum().alias('taker_sell_base'),
        pl.col('qa').filter(~pl.col('is_buyer_maker')).sum().alias('taker_buy_quote'),
        pl.col('qa').filter(pl.col('is_buyer_maker')).sum().alias('taker_sell_quote'),
        pl.len().alias('n_trades'),
        # 大单特征
        pl.col('qa').filter(pl.col('qa') >= 1000).sum().alias('big_vol'),
        pl.col('qa').filter((pl.col('qa') >= 1000) & ~pl.col('is_buyer_maker')).sum().alias('big_buy_vol'),
        pl.col('qa').filter((pl.col('qa') >= 1000) & pl.col('is_buyer_maker')).sum().alias('big_sell_vol'),
        # 超大单 ($10k+)
        pl.col('qa').filter(pl.col('qa') >= 10000).sum().alias('huge_vol'),
        pl.col('qa').filter((pl.col('qa') >= 10000) & ~pl.col('is_buyer_maker')).sum().alias('huge_buy_vol'),
        pl.col('qa').filter((pl.col('qa') >= 10000) & pl.col('is_buyer_maker')).sum().alias('huge_sell_vol'),
    ]).sort('kline_open')

    # 衍生
    g = g.with_columns([
        (pl.col('n_trades') / (pl.col('taker_buy_quote') + pl.col('taker_sell_quote') + 1e-9)).alias('trade_density'),
        (pl.col('big_vol') / (pl.col('taker_buy_quote') + pl.col('taker_sell_quote') + 1e-9)).alias('big_ratio'),
        (pl.col('huge_vol') / (pl.col('taker_buy_quote') + pl.col('taker_sell_quote') + 1e-9)).alias('huge_ratio'),
        (pl.col('big_buy_vol') - pl.col('big_sell_vol')).alias('big_ofi'),
        (pl.col('huge_buy_vol') - pl.col('huge_sell_vol')).alias('huge_ofi'),
        # ts 对齐: kline_open 是 μs 分钟戳, raw.ts = kline_open/1e6 + 60 (下一根K开始)
        ((pl.col('kline_open') // 1_000_000) + 60).alias('ts'),
    ])
    return g

def main():
    print("Step 1: Load aggTrades, build rhythm features")
    agg = load_spot_aggtrades(AGG_PATTERN)
    print(f"  aggTrades: {agg.height:,} rows, range: {agg['kline_open'].min()} → {agg['kline_open'].max()}")

    rhythm = build_rhythm_features(agg)
    print(f"  1min rhythm bars: {rhythm.height}")
    print(f"  新增特征 cols: {[c for c in rhythm.columns if c not in ('kline_open','ts')]}")

    print("\nStep 2: 检查 raw buy_vol vs agg 聚合一致性")
    raw = pl.read_parquet(RAW_PATH)
    # raw.ts = kline_open/1e6 + 60, 所以用 ts 直接 join
    raw_sub = raw.filter(pl.col('ts').is_in(rhythm['ts']))
    merged = raw_sub.join(rhythm, on='ts', how='inner')
    print(f"  merged with raw: {merged.height} bars")
    if merged.height > 0:
        rb = merged['buy_vol'].to_numpy()
        ab = merged['taker_buy_base'].to_numpy()
        print(f"  raw buy_vol vs agg taker_buy_base corr = {np.corrcoef(rb, ab)[0,1]:.8f}")

    print("\nStep 3: 对齐到 h15 ds_to_raw, 计算滚动节奏特征")
    try:
        ds = pl.read_parquet(DS_PATH)
        print(f"  ds_ETH_h15: {ds.height} rows, cols: {ds.columns[:10]}...")
    except:
        print("  ds_ETH_h15.parquet 不存在, 跳过对齐, 直接 1min 特征统计")
        return

    # 把 rhythm 转成 numpy, 按 raw ts 索引
    rhythm_np = {c: rhythm[c].to_numpy() for c in rhythm.columns}
    rhythm_ts = rhythm_np['ts']

    raw_ts = raw['ts'].to_numpy()
    # 构建 (ts → 行索引) 映射用于快速查找
    ts_to_idx = {int(t): i for i, t in enumerate(raw_ts)}
    print(f"  raw ts range: {raw_ts[0]} → {raw_ts[-1]}")
    print(f"  rhythm ts range: {rhythm_ts[0]} → {rhythm_ts[-1]}")

    # 检查 ds 有没有 ds_to_raw 映射
    ds_cols = ds.columns
    print(f"  ds columns: {ds_cols}")

    print("\nStep 4: 节奏特征统计")
    feat_cols = ['trade_density', 'big_ratio', 'huge_ratio', 'big_ofi', 'huge_ofi',
                 'n_trades', 'big_vol', 'huge_vol']
    for c in feat_cols:
        if c in rhythm.columns:
            s = rhythm[c].to_numpy()
            print(f"  {c:15s}: mean={np.mean(s):.4f}, std={np.std(s):.4f}, "
                  f"p10={np.percentile(s,10):.4f}, p90={np.percentile(s,90):.4f}")

    print("\nStep 5: 做 label = sign(close[t+15]/close[t]-1), 验证 n_trades 和 label 的关系")
    # 在 rhythm 覆盖的时间段内
    mask = (raw_ts >= rhythm_ts[0]) & (raw_ts <= rhythm_ts[-1])
    raw_idx = np.where(mask)[0]

    # rhythm 的每个 ts 对应 raw 的某一行
    rhythm_raw_idx = []
    for t in rhythm_ts:
        if t in ts_to_idx:
            rhythm_raw_idx.append(ts_to_idx[int(t)])
        else:
            rhythm_raw_idx.append(-1)

    valid = [i for i in rhythm_raw_idx if i >= 0 and i + 15 < len(raw_ts)]
    valid_idx = np.array(valid)
    print(f"  valid rows (有 rhythm 且 label 可算): {len(valid_idx)}")

    if len(valid_idx) > 100:
        close = raw['close'].to_numpy()
        y = np.sign(close[valid_idx + 15] / close[valid_idx] - 1)
        n_trades = rhythm['n_trades'].to_numpy()
        huge_ofi = rhythm['huge_ofi'].to_numpy()
        big_ofi = rhythm['big_ofi'].to_numpy()
        trade_density = rhythm['trade_density'].to_numpy()

        # 只能在 rhythm 覆盖的范围内取 valid_idx 对应的节奏值
        rhythm_idx_list = [rhythm_raw_idx.index(i) for i in valid_idx]
        rt_idx = np.array(rhythm_idx_list)

        print(f"\n  y (label) 分布: up={np.mean(y>0):.2%}, down={np.mean(y<0):.2%}, flat={np.mean(y==0):.2%}")
        print(f"\n  分组测试: n_trades 高低组 vs next 15min return")
        for name, feat in [('n_trades', n_trades), ('huge_ofi', huge_ofi),
                           ('big_ofi', big_ofi), ('trade_density', trade_density)]:
            f = feat[rt_idx]
            # 分 5 组
            try:
                quintiles = np.percentile(f, [20, 40, 60, 80])
                groups = np.digitize(f, quintiles)
                print(f"  {name}:")
                for g in range(5):
                    m = groups == g
                    if m.sum() > 10:
                        ret = np.mean(close[valid_idx[m] + 15] / close[valid_idx[m]] - 1) * 100
                        up_rate = np.mean(y[m] > 0)
                        print(f"    group {g}: n={m.sum()}, avg_ret={ret:+.4f}%, up_rate={up_rate:.2%}")
            except Exception as e:
                print(f"  {name}: skip ({e})")

if __name__ == "__main__":
    main()
