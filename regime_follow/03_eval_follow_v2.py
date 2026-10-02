"""03_eval_follow_v2.py

Corrected eval based on actual cross-frequency data:
  - stoch(70,3,2) on ETH 5min produces ~201K crosses, median gap=3, 74% gaps<5
  - But cross candle itself has 84% direction hit rate!
  - Previous v2 required min_len≥5 → filtered out ALL trades (buggy)

Strategies to test:
  A) cross-only: bet ONLY on the cross candle itself (gold→up, death→down) → expect ~84% acc
  B) long-seg-full: bet on cross candle + all following bars until next cross (but only if gap≥5)
  C) all-seg-full: bet on ALL segments regardless of gap (short gaps → quick exit)
  D) stoch_k filter: only trade high-confidence crosses

Metrics: bar-level acc (= 1{pred_match_bar_label}) / bars_traded, tpd
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(line_buffering=True)
import gc, time, numpy as np, pandas as pd, datetime as dtm
import config
from sklearn.metrics import roc_auc_score

t0 = time.time()

def load_candles(TF):
    eth = pd.read_parquet(f'{config.DS_DIR}/raw_ETH.parquet').sort_values('ts').reset_index(drop=True)
    eth['dt'] = pd.to_datetime(eth['ts'], unit='s', utc=True)
    df = eth.set_index('dt')
    agg = df[['open','high','low','close','buy_vol','sell_vol','funding']].resample(
        f'{TF}min', closed='left', label='left'
    ).agg({'open':'first','high':'max','low':'min','close':'last',
          'buy_vol':'sum','sell_vol':'sum','funding':'mean'}).dropna()
    agg = agg.reset_index()
    agg['ts'] = agg['dt'].astype('int64').to_numpy()
    return agg

def stoch(df, n=70, m1=3, m2=2):
    low_n = df['low'].rolling(n, min_periods=n).min()
    high_n = df['high'].rolling(n, min_periods=n).max()
    denom = (high_n - low_n).replace(0, np.nan)
    stoch_k_raw = 100 * (df['close'] - low_n) / denom
    stoch_k = stoch_k_raw.rolling(m1, min_periods=1).mean()
    stoch_d = stoch_k.rolling(m2, min_periods=1).mean()
    return stoch_k, stoch_d

def build_seg_df(candles, k_col, d_col):
    diff = candles[k_col] - candles[d_col]
    above = diff > 0
    cross_gold  = (~above).shift(1).fillna(False).astype(bool).values & above.values
    cross_death = above.shift(1).fillna(False).astype(bool).values & (~above.values)
    cross_any = cross_gold | cross_death
    cross_type = np.full(len(candles), '', dtype='U6')
    cross_type[cross_gold] = 'gold'
    cross_type[cross_death] = 'death'

    bar_label = np.zeros(len(candles), dtype=int)
    bar_label[1:] = (candles['close'].values[1:] > candles['close'].values[:-1])

    # gap[i] = distance from cross at i to the NEXT cross
    cross_idx = np.where(cross_any)[0]
    gap = np.full(len(candles), 0, dtype=np.int64)
    for i in range(len(cross_idx)-1):
        gap[cross_idx[i]] = cross_idx[i+1] - cross_idx[i]
    gap[cross_idx[-1]] = len(candles) - cross_idx[-1]

    seg_df = pd.DataFrame({
        'ts': candles['ts'].values,
        'close': candles['close'].values,
        'stoch_k': candles[k_col].values,
        'stoch_d': candles[d_col].values,
        'cross_type': cross_type,
        'cross_idx': np.arange(len(candles)),
        'gap': gap,
        'bar_label': bar_label,
        'is_cross': cross_any,
    })

    # For strategy B and C: fill seg_kind for ALL rows between crosses
    # Walk forward: every row inherits the kind of the most recent cross
    seg_kind = np.full(len(seg_df), '', dtype='U6')
    cur_kind = ''
    for i in range(len(seg_df)):
        if cross_gold[i]: cur_kind = 'gold'
        elif cross_death[i]: cur_kind = 'death'
        seg_kind[i] = cur_kind
    seg_df['seg_kind'] = seg_kind

    # For full-segment following, we also need to know whether this row is a "post-cross" bar
    # vs a "pre-cross" bar. All rows after a cross (inclusive) up to the next cross are post-cross.
    # So seg_kind just tells us which cross segment we're in.
    return seg_df

def tmask(ts, s, e):
    a = int(dtm.datetime.strptime(s,'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())
    b = int(dtm.datetime.strptime(e,'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())
    return (ts >= a) & (ts < b)

def pred_and_eval(seg_df, pred_fn, name):
    """Apply pred_fn(row) → 1(up)/0(down)/-1(none), then evaluate bar-level acc + tpd."""
    preds = seg_df.apply(pred_fn, axis=1).values
    seg_df = seg_df.assign(pred=preds)
    seg_df['correct'] = (seg_df['pred'] == seg_df['bar_label']).astype(int)

    splits = {
        'train':      config.SPLITS['train'],
        'early_stop': config.SPLITS['early_stop'],
        'meta_val':   config.SPLITS['meta_val'],
        'test':       config.SPLITS['test'],
    }

    print(f"\n  [{name}]")
    for sname, (s, e) in splits.items():
        ts_mask = tmask(seg_df['ts'].values, s, e)
        split = seg_df[ts_mask]
        days = max(len(np.unique(split['ts'].values // 86400)), 1)
        traded = split[split['pred'] != -1]

        if len(traded) == 0:
            print(f"    {sname:12s}: NO TRADES")
            continue

        acc = traded['correct'].mean() * 100
        tpd = len(traded) / days

        # dir ret proxy — vectorized on split's own 0-based index
        split_reset = split.reset_index(drop=True)
        close_arr = split_reset['close'].values
        bar_ret = np.zeros(len(split_reset))
        bar_ret[1:] = close_arr[1:] / close_arr[:-1] - 1
        traded_mask = split_reset['pred'] != -1
        preds_tr = split_reset.loc[traded_mask, 'pred'].values
        dir_ret = np.where(preds_tr == 1, bar_ret[traded_mask], -bar_ret[traded_mask])
        avg_bar = dir_ret.mean() * 100 if len(dir_ret) > 0 else 0

        # Also per-cross stats: how many crosses traded?
        n_crosses = traded[traded['is_cross']]['cross_idx'].nunique()
        seg_lens_traded = traded.groupby('cross_idx').size().values

        print(f"    {sname:12s}: acc={acc:.1f}%  tpd={tpd:.1f}  #crosses={n_crosses}  "
              f"avg_seg={seg_lens_traded.mean():.1f}  avg_bar_ret={avg_bar:+.3f}%")

# ========== MAIN ==========
for TF in [5, 15]:
    for (SN, M1, M2) in [(70,3,2), (14,3,3), (20,3,3)]:
        print(f"\n{'#'*70}")
        print(f"# TF={TF}min   STOCH({SN},{M1},{M2})")
        print(f"{'#'*70}")

        cands = load_candles(TF)
        cands['stoch_k'], cands['stoch_d'] = stoch(cands, n=SN, m1=M1, m2=M2)
        cands = cands.dropna(subset=['stoch_k','stoch_d']).reset_index(drop=True)
        seg = build_seg_df(cands, 'stoch_k', 'stoch_d')
        del cands; gc.collect()

        cross_rows = seg[seg['is_cross']]
        print(f"\n  Total candles={len(seg):,}  crosses={len(cross_rows):,}  "
              f"gaps<5={(cross_rows['gap']<5).sum()} ({(cross_rows['gap']<5).mean()*100:.1f}%)")

        # ====== STRATEGY A: Cross only (bet ONLY on cross candle itself) ======
        def pred_A(row):
            if not row['is_cross']: return -1
            if row['cross_type'] == 'gold': return 1   # predict up on gold cross candle
            if row['cross_type'] == 'death': return 0  # predict down on death cross candle
            return -1
        pred_and_eval(seg.copy(), pred_A, "A: cross-only")

        # ====== STRATEGY B: Cross + all post-cross bars (gap>=5 only) ======
        # Only follow segments where gap (cross distance to next cross) >= 5
        # pred on cross candle + all post-cross rows until next cross
        cross_gaps = cross_rows.set_index('cross_idx')['gap']
        big_gap_crosses = set(cross_gaps[cross_gaps >= 5].index)

        def pred_B(row):
            # If this is a big-gap cross → trade it + everything until next cross
            # Easier: just trade rows whose cross_idx (the initiating cross) is in big_gap_crosses
            pass  # implemented below

        # Walk: label each row with the cross_idx that initiated its segment
        row_seg_cross = np.full(len(seg), -1, dtype=np.int64)
        cur_seg_cross = -1
        for i in range(len(seg)):
            if seg['is_cross'].iloc[i]:
                cur_seg_cross = i
            row_seg_cross[i] = cur_seg_cross
        seg = seg.assign(row_seg_cross=row_seg_cross)

        big_gap_set = set(cross_gaps[cross_gaps >= 5].index)
        def pred_B(row):
            cidx = row['row_seg_cross']
            if cidx not in big_gap_set: return -1
            kind = seg.loc[cidx, 'cross_type']
            if kind == 'gold': return 1
            if kind == 'death': return 0
            return -1
        pred_and_eval(seg.copy(), pred_B, "B: big-gap segs (gap≥5) full")

        # ====== STRATEGY C: ALL segs full follow (no gap filter) ======
        def pred_C(row):
            cidx = row['row_seg_cross']
            if cidx == -1: return -1
            kind = seg.loc[cidx, 'cross_type']
            if kind == 'gold': return 1
            if kind == 'death': return 0
            return -1
        pred_and_eval(seg.copy(), pred_C, "C: ALL segs full follow")

        # ====== STRATEGY D: Cross-only + stoch_k confidence filter ======
        # Low stoch_k gold → stronger up signal; high stoch_k death → stronger down signal
        filter_pairs = [(40, 60), (30, 70), (25, 75), (20, 80)]
        for gi_dl in filter_pairs:
            g_hi_cur, d_lo_cur = gi_dl
            # only trade gold crosses with k ≤ g_hi_cur, death crosses with k ≥ d_lo_cur
            def pred_D(row, _gh=g_hi_cur, _dl=d_lo_cur):
                if not row['is_cross']: return -1
                if row['cross_type'] == 'gold' and row['stoch_k'] <= _gh: return 1
                if row['cross_type'] == 'death' and row['stoch_k'] >= _dl: return 0
                return -1
            tag = f"D: cross-only k≤{g_hi_cur}/k≥{d_lo_cur}"
            pred_and_eval(seg.copy(), pred_D, tag)

        # ====== STRATEGY E: Big-gap segs + stoch_k filter ======
        filter_pairs2 = [(40, 60), (30, 70)]
        for gi_dl in filter_pairs2:
            g_hi_cur, d_lo_cur = gi_dl
            conf_crosses = set()
            for cidx in big_gap_set:
                k = seg.loc[cidx, 'stoch_k']
                kind = seg.loc[cidx, 'cross_type']
                if (kind == 'gold' and k <= g_hi_cur) or (kind == 'death' and k >= d_lo_cur):
                    conf_crosses.add(cidx)

            def pred_E(row, cc=conf_crosses):
                if row['row_seg_cross'] not in cc: return -1
                kind = seg.loc[row['row_seg_cross'], 'cross_type']
                if kind == 'gold': return 1
                if kind == 'death': return 0
                return -1
            tag = f"E: big-gap+conf k≤{g_hi_cur}/k≥{d_lo_cur}"
            pred_and_eval(seg.copy(), pred_E, tag)

print(f"\n[{time.time()-t0:.1f}s] Done.")
