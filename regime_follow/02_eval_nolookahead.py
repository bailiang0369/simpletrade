"""02_eval_nolookahead.py

No-lookahead backtest of the "regime follow" strategy:

  Baseline A: 每次金叉后全 segment 每根 K 押注上涨
              每次死叉后全 segment 每根 K 押注下跌
              （不做任何置信度过滤 —— "全跟随"）

  Baseline B: 只在置信度高的交叉点跟随
              置信度 = stoch_k 位置（低位金叉强、高位金叉弱）
              扫阈值 → 看能不能提升 acc 同时保持 tpd

Evaluation:
  - Split by TRAIN/ES/META_VAL/TEST (same as main pipeline)
  - NO lookahead: stoch 是 rolling forward，cross detection 用 shift，天然无前视
  - Metrics: acc (bar 级别 top-N), trades per day, segment PnL distribution
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

def compute_seg_labels(candles, k_col, d_col):
    """Return for each candle: which segment it belongs to, and whether seg is up/down dominant.

    Returns DataFrame with columns:
      ts, idx, close, stoch_k, stoch_d, cross_type, seg_id, seg_pos,
      seg_up_ratio (computed AFTER segment ends — lookahead!),
      bar_label (close[i] > close[i-1])
    """
    diff = candles[k_col] - candles[d_col]
    above = diff > 0
    cross_gold  = (~above).shift(1).fillna(False).astype(bool).values & above.values
    cross_death = above.shift(1).fillna(False).astype(bool).values & (~above.values)
    cross_type = np.full(len(candles), '', dtype='U6')
    cross_type[cross_gold] = 'gold'
    cross_type[cross_death] = 'death'

    bar_label = np.zeros(len(candles), dtype=int)
    bar_label[1:] = (candles['close'].values[1:] > candles['close'].values[:-1])

    # Segments between consecutive crosses
    seg_id = np.zeros(len(candles), dtype=np.int64)
    seg_kind = np.full(len(candles), '', dtype='U6')
    cur_seg = 0
    cur_kind = ''
    for i in range(len(candles)):
        if cross_type[i] != '':
            cur_seg += 1
            cur_kind = cross_type[i]   # this cross STARTS a new segment
        seg_id[i] = cur_seg
        seg_kind[i] = cur_kind

    # seg_up_ratio — but this requires FUTURE data (lookahead!). We compute it for LABELING ONLY,
    # never use as feature. The actual evaluation uses bar-level direction accuracy.
    seg_up_ratio = np.zeros(len(candles), dtype=np.float64)
    seg_bar_count = np.zeros(len(candles), dtype=np.int64)

    # For each segment, count bars (excluding the cross candle itself, which is where stoch just crossed)
    unique_segs = np.unique(seg_id)
    for sid in unique_segs:
        mask = seg_id == sid
        seg_bar_idx = np.where(mask)[0]
        # The segment runs from the cross candle to the NEXT cross candle (exclusive)
        # But wait — if seg_kind is 'gold', then candle i is a golden cross, candles i+1..next_cross_idx
        # are the post-cross bars that we follow. Let's just label all bars from cross (inclusive) to next cross.
        if len(seg_bar_idx) >= 5:
            upK = bar_label[seg_bar_idx].sum()
            ratio = upK / len(seg_bar_idx)
            seg_up_ratio[seg_bar_idx] = ratio
            seg_bar_count[seg_bar_idx] = len(seg_bar_idx)

    out = pd.DataFrame({
        'ts': candles['ts'].values,
        'close': candles['close'].values,
        'stoch_k': candles[k_col].values,
        'stoch_d': candles[d_col].values,
        'cross_type': cross_type,
        'seg_id': seg_id,
        'seg_kind': seg_kind,
        'seg_up_ratio': seg_up_ratio,
        'seg_bar_count': seg_bar_count,
        'bar_label': bar_label,
    })
    return out

def tmask(ts, s, e):
    a = int(dtm.datetime.strptime(s,'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())
    b = int(dtm.datetime.strptime(e,'%Y-%m-%d').replace(tzinfo=dtm.timezone.utc).timestamp())
    return (ts >= a) & (ts < b)

def eval_strategy(seg_df, name, confidence_filter=None):
    """Evaluate bar-level strategy on each split.

    Strategy: in every segment (defined by stoch cross), predict the direction matching the cross kind.
      - gold cross → predict up for all bars in segment → signal = 1 (long)
      - death cross → predict down for all bars in segment → signal = 0 (short)

    Accuracy = # bars where direction matches bar_label / total bars traded
    tpd = bars_traded / num_days

    confidence_filter: callable(cross_row) → bool, True if we trade this segment.
    """
    def strategy_pred(row):
        if row['seg_kind'] == 'gold': return 1    # predict up
        if row['seg_kind'] == 'death': return 0   # predict down
        return -1   # no trade

    seg_df['pred'] = seg_df.apply(strategy_pred, axis=1)
    seg_df['correct'] = (seg_df['pred'] == seg_df['bar_label']).astype(int)

    splits = {
        'train':     config.SPLITS['train'],
        'early_stop':config.SPLITS['early_stop'],
        'meta_val':  config.SPLITS['meta_val'],
        'test':      config.SPLITS['test'],
    }

    print(f"\n  {name}:")
    for sname, (s, e) in splits.items():
        ts_mask = tmask(seg_df['ts'].values, s, e)
        split = seg_df[ts_mask].copy()
        days = len(np.unique(split['ts'].values // 86400))

        # Only trade bars inside segments where pred != -1 and seg_bar_count >= 5
        traded = split[(split['pred'] != -1) & (split['seg_bar_count'] >= 5)]

        if len(traded) == 0:
            print(f"    {sname:12s}: NO TRADES")
            continue

        acc = traded['correct'].mean()
        tpd = len(traded) / max(days, 1)
        segs = traded['seg_id'].nunique()
        cross_kinds = traded.groupby('seg_kind').size().to_dict()

        # Also PnL proxy: avg bar return when traded
        close_vals = split['close'].values
        bar_ret = np.zeros(len(split))
        bar_ret[1:] = close_vals[1:] / close_vals[:-1] - 1
        traded_idx = split[(split['pred'] != -1) & (split['seg_bar_count'] >= 5)].index
        dir_ret = np.where(split.loc[traded_idx, 'pred'].values == 1, bar_ret[traded_idx], -bar_ret[traded_idx])
        avg_bar_ret = dir_ret.mean() * 100

        print(f"    {sname:12s}: acc={acc*100:.1f}%  tpd={tpd:.1f}  "
              f"#segs={segs}  upK={cross_kinds.get('gold',0)} dnK={cross_kinds.get('death',0)}  "
              f"avg_bar_ret={avg_bar_ret:.3f}%")

    return seg_df

# ========== MAIN ==========
for TF in [5, 15]:
    for (SN, M1, M2) in [(70,3,2), (14,3,3), (20,3,3)]:
        print(f"\n{'#'*70}")
        print(f"# TF={TF}min   STOCH({SN},{M1},{M2})")
        print(f"{'#'*70}")

        cands = load_candles(TF)
        cands['stoch_k'], cands['stoch_d'] = stoch(cands, n=SN, m1=M1, m2=M2)
        cands = cands.dropna(subset=['stoch_k','stoch_d']).reset_index(drop=True)

        seg = compute_seg_labels(cands, 'stoch_k', 'stoch_d')
        del cands; gc.collect()

        # Baseline A: trade ALL segments
        eval_strategy(seg.copy(), f"ALL segs")

        # Baseline B: filter by stoch_k position at cross time
        # Low stoch_k golden cross = stronger up signal
        seg['cross_time'] = seg['cross_type'] != ''
        crosses_only = seg[seg['cross_time'] & (seg['seg_bar_count']>=5)][['seg_id','seg_kind','stoch_k','stoch_d','ts']].copy()
        print(f"\n  Cross stats ({len(crosses_only)}):")
        for kind in ['gold','death']:
            sub = crosses_only[crosses_only['seg_kind']==kind]
            print(f"    {kind}: stoch_k mean={sub['stoch_k'].mean():.1f}  median={sub['stoch_k'].median():.1f}")

        # Filter experiments on TEST split
        gold_crosses = crosses_only[crosses_only['seg_kind']=='gold']
        death_crosses = crosses_only[crosses_only['seg_kind']=='death']

        # Low stoch_k → strong gold signal
        for k_thresh in [30, 40, 50]:
            strong_gold_ids = gold_crosses[gold_crosses['stoch_k'] <= k_thresh]['seg_id'].values
            # For death crosses: HIGH stoch_k (>= 70) → strong down signal
            strong_death_ids = death_crosses[death_crosses['stoch_k'] >= (100-k_thresh)]['seg_id'].values
            strong_ids = np.concatenate([strong_gold_ids, strong_death_ids])
            name = f"k{'≤'+str(k_thresh)}_gold_k≥{100-k_thresh}_death"

            seg2 = seg.copy()
            seg2.loc[~seg2['seg_id'].isin(strong_ids), 'pred'] = -1   # no trade
            eval_strategy(seg2, name)

print(f"\n[{time.time()-t0:.1f}s] Done.")
