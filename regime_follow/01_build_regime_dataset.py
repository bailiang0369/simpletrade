"""01_build_regime_dataset.py

Resample 1min ETH → N-min candles, compute stoch(70,3,2), detect golden/death crosses,
label each cross with the forward segment up/down bar ratio.

This is the "regime-following" approach:
  - At golden cross (stochK ↑ cross stochD): predict if next segment ≥60% up bars → go long every bar until next death cross
  - At death cross (stochK ↓ cross stochD): predict if next segment ≥60% down bars → go short every bar until next golden cross

Design choices:
  - stoch(70,3,2): slow stochastic matching user's spec
  - Resample to BOTH 5min and 15min → compare
  - bar label within segment: close > prev close → upK, else → downK (no h15 ret prediction!)
  - Only segments with min_len ≥ 5 bars (avoid micro-segments) are kept
"""
import sys; sys.stdout.reconfigure(line_buffering=True)
import os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gc, time, numpy as np, pandas as pd
import config

t0 = time.time()
print(f"[{time.time()-t0:.1f}s] Loading raw ETH 1min...")
eth = pd.read_parquet(f'{config.DS_DIR}/raw_ETH.parquet').sort_values('ts').reset_index(drop=True)
eth['dt'] = pd.to_datetime(eth['ts'], unit='s', utc=True)
print(f"  {len(eth):,} rows, {eth['dt'].min()} → {eth['dt'].max()}")

def resample_candles(df, tf_min):
    """Resample 1min OHLCV → tf_min candles, preserving ts as close time."""
    df = df.set_index('dt')
    agg = df[['open','high','low','close','buy_vol','sell_vol','funding']].resample(f'{tf_min}min', closed='left', label='left').agg({
        'open':'first','high':'max','low':'min','close':'last',
        'buy_vol':'sum','sell_vol':'sum','funding':'mean'
    }).dropna()
    agg = agg.reset_index()
    agg['ts'] = agg['dt'].astype('int64').to_numpy()
    return agg

def stoch(df, n=70, m1=3, m2=3):
    """Compute Stochastic(70,3,2) → stoch_k, stoch_d."""
    low_n = df['low'].rolling(n, min_periods=n).min()
    high_n = df['high'].rolling(n, min_periods=n).max()
    denom = (high_n - low_n).replace(0, np.nan)
    stoch_k_raw = 100 * (df['close'] - low_n) / denom
    stoch_k = stoch_k_raw.rolling(m1, min_periods=1).mean()    # %K smoothed
    stoch_d = stoch_k.rolling(m2, min_periods=1).mean()        # %D
    return stoch_k, stoch_d

def find_crosses(df, k_col='stoch_k', d_col='stoch_d'):
    """Find golden crosses (k crosses above d) and death crosses (k crosses below d).

    Returns DataFrame with one row per cross:
      idx, ts, kind ('gold'|'death'), k_val, d_val, close
    """
    diff = df[k_col] - df[d_col]                    # k - d
    above = diff > 0                                  # is k above d?
    cross_gold  = (~above).shift(1).fillna(False) & above   # was below → now above
    cross_death = above.shift(1).fillna(False) & (~above)   # was above → now below

    crosses = []
    for i in range(len(df)):
        if cross_gold.iloc[i]:
            crosses.append({'idx': i, 'ts': df['ts'].iloc[i], 'kind': 'gold',
                            'stoch_k': df[k_col].iloc[i], 'stoch_d': df[d_col].iloc[i],
                            'close': df['close'].iloc[i],
                            'high': df['high'].iloc[i], 'low': df['low'].iloc[i],
                            'open': df['open'].iloc[i]})
        elif cross_death.iloc[i]:
            crosses.append({'idx': i, 'ts': df['ts'].iloc[i], 'kind': 'death',
                            'stoch_k': df[k_col].iloc[i], 'stoch_d': df[d_col].iloc[i],
                            'close': df['close'].iloc[i],
                            'high': df['high'].iloc[i], 'low': df['low'].iloc[i],
                            'open': df['open'].iloc[i]})
    return pd.DataFrame(crosses)

def attach_segment_labels(candles, crosses, min_len=5):
    """For each cross at index i, the NEXT segment is candles[i+1 : i+1_next_cross].

    For a golden cross → expect segment to be mostly up bars → up_ratio = upK/totalK
    For a death cross → expect segment to be mostly down bars → down_ratio = downK/totalK

    We store BOTH ratios on every cross (segment direction agnostic).
    """
    upK_all = (candles['close'].values[1:] > candles['close'].values[:-1]).astype(int)
    downK_all = 1 - upK_all

    # Index map: df.iloc[i] is cross; segment is [i, next_cross_idx)
    rows = []
    for ci in range(len(crosses)):
        i = crosses.iloc[ci]['idx']
        next_idx = crosses.iloc[ci+1]['idx'] if ci+1 < len(crosses) else len(candles)

        # Segment bars are candles[i+1 : next_idx] (the bars AFTER this cross up to next cross)
        seg_len = next_idx - i - 1
        if seg_len < min_len:
            continue

        upK = int(upK_all[i:next_idx-1].sum())   # i..next_idx-2 → length seg_len
        downK = seg_len - upK
        up_ratio = upK / seg_len
        down_ratio = downK / seg_len

        # PnL from close[i] → close[next_idx-1] (segment buy-hold return)
        start_close = candles['close'].iloc[i]
        end_close = candles['close'].iloc[next_idx-1] if next_idx-1 < len(candles) else candles['close'].iloc[-1]
        seg_ret = end_close / start_close - 1

        r = crosses.iloc[ci].to_dict()
        r['seg_len'] = seg_len
        r['upK'] = upK; r['downK'] = downK
        r['up_ratio'] = up_ratio; r['down_ratio'] = down_ratio
        r['seg_ret'] = seg_ret
        r['seg_ret_pct'] = seg_ret * 100
        rows.append(r)

    return pd.DataFrame(rows)

# ========== Build for multiple timeframes ==========
out_dir = f'{config.PROJECT_DIR}/regime_follow'
os.makedirs(out_dir, exist_ok=True)

for TF in [5, 15]:
    print(f"\n{'='*60}")
    print(f"TF = {TF}min")
    print(f"{'='*60}")

    cands = resample_candles(eth, TF)
    cands['stoch_k'], cands['stoch_d'] = stoch(cands, n=70, m1=3, m2=2)
    # Clean NaN from warmup
    cands = cands.dropna(subset=['stoch_k','stoch_d']).reset_index(drop=True)
    print(f"  Candles: {len(cands):,}  warmup={70} bars")

    crosses = find_crosses(cands)
    print(f"  Crosses: {len(crosses)}  "
          f"(gold={int((crosses['kind']=='gold').sum())}  death={int((crosses['kind']=='death').sum())})")

    labeled = attach_segment_labels(cands, crosses, min_len=5)
    print(f"  Labeled crosses (seg_len≥5): {len(labeled)}")

    # Save
    out_npz = f'{out_dir}/regime_TF{TF}.npz'
    np.savez(out_npz,
        ts=cands['ts'].values,
        close=cands['close'].values,
        stoch_k=cands['stoch_k'].values,
        stoch_d=cands['stoch_d'].values,
        crosses_ts=labeled['ts'].values,
        crosses_kind=labeled['kind'].values,
        crosses_idx=labeled['idx'].values,
        seg_len=labeled['seg_len'].values,
        up_ratio=labeled['up_ratio'].values,
        down_ratio=labeled['down_ratio'].values,
        seg_ret=labeled['seg_ret'].values)
    out_csv = f'{out_dir}/regime_TF{TF}_crosses.csv'
    labeled.to_csv(out_csv, index=False)
    print(f"  Saved → {out_npz} + {out_csv}")

    # Quick stats
    gold = labeled[labeled['kind']=='gold']
    death = labeled[labeled['kind']=='death']
    print(f"\n  GOLD crosses (expect up-dominant segments):")
    print(f"    n={len(gold)}  seg_len mean={gold['seg_len'].mean():.1f}  median={gold['seg_len'].median():.0f}")
    print(f"    up_ratio mean={gold['up_ratio'].mean():.3f}  median={gold['up_ratio'].median():.3f}")
    print(f"    up_ratio ≥0.60 = {(gold['up_ratio']>=0.60).mean():.3f}")
    print(f"    up_ratio ≥0.55 = {(gold['up_ratio']>=0.55).mean():.3f}")
    print(f"    seg_ret mean={gold['seg_ret'].mean()*100:.3f}%  median={gold['seg_ret'].median()*100:.3f}%")
    print(f"\n  DEATH crosses (expect down-dominant segments):")
    print(f"    n={len(death)}  seg_len mean={death['seg_len'].mean():.1f}  median={death['seg_len'].median():.0f}")
    print(f"    down_ratio mean={death['down_ratio'].mean():.3f}  median={death['down_ratio'].median():.3f}")
    print(f"    down_ratio ≥0.60 = {(death['down_ratio']>=0.60).mean():.3f}")
    print(f"    down_ratio ≥0.55 = {(death['down_ratio']>=0.55).mean():.3f}")
    print(f"    seg_ret mean={death['seg_ret'].mean()*100:.3f}%  median={death['seg_ret'].median()*100:.3f}%")

print(f"\n[{time.time()-t0:.1f}s] Done.")
