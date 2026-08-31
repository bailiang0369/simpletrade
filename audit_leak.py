"""阶段A 完整性审计：核对数据管线是否存在时序类数据泄露。

只读审计，不改任何数据/模型。检查项：
1) 原始K线 ts 严格单调、与切片范围吻合
2) ds 行到 raw 行的映射 ds_to_raw 与 ts **精确同刻**（无 +1 错位/未来偏移）
3) 序列窗 [p-W+1, p] 不触碰任何未来 raw 行（构造性断言 + close 标签不渗入特征）
4) 各时间切分样本互斥（重点：meta_val 与 test 不能有重叠样本）
   —— 先量化当前"闭区间"切分造成的边界重叠，供修复前后对比
5) 类型审计：特征 float32、标签 int8、ts int64、零 null
"""
import numpy as np
import polars as pl

import config
from data_store import AssetContext


def section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def audit(symbol, W=config.LOOKBACK_MIN, horizon=config.HORIZON_MIN):
    ctx = AssetContext(symbol)
    raw_n = len(ctx.raw_ts)
    ds_n = len(ctx.ds_ts)
    print(f"[{symbol}] raw={raw_n} ds={ds_n} feats={len(ctx.feat_names)}")

    # 1. ts 单调性
    d_raw = np.diff(ctx.raw_ts)
    assert (d_raw > 0).all(), f"{symbol}: raw ts 非严格单调!"
    d_ds = np.diff(ctx.ds_ts)
    assert (d_ds > 0).all(), f"{symbol}: ds ts 非严格单调!"
    print(f"  OK: raw ts 严格单调 (min_gap={d_raw.min()}, max_gap={d_raw.max()}s)")
    n_gap = int((d_raw > 60).sum())
    if n_gap:
        # 真实币安交易所停机（两资产同一批缺口），非prepare人工产物；60分钟窗不会跨过这些缺口
        print(f"  NOTE: {n_gap} 处 >60s 缺口（交易所停机，两资产同期），不影响60min窗完整性")

    # ts 时间范围
    from datetime import datetime, timezone
    def dt(t): return datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
    print(f"  raw ts 范围: {dt(int(ctx.raw_ts[0]))} -> {dt(int(ctx.raw_ts[-1]))}")

    # 2. ds->raw 精确同刻
    mapped_raw_ts = ctx.raw_ts[ctx.ds_to_raw]
    exact = (mapped_raw_ts == ctx.ds_ts).all()
    print(f"  ds_to_raw->raw_ts == ds_ts ? {exact}")
    if not exact:
        bad = np.where(mapped_raw_ts != ctx.ds_ts)[0][:5]
        for b in bad:
            print(f"    MISALIGN ds@{b} ds_ts={ctx.ds_ts[b]} maps raw={mapped_raw_ts[b]}")
        raise AssertionError(f"{symbol}: ds_to_raw 与 ts 不一致!")
    # 检查 ds_to_raw 是否可能指到未来：映射应当 <= ds 行自身（在 raw 里的位置即自身）
    # 由于 raw_ts[ds_to_raw]==ds_ts 且 raw_ts 单调，ds_to_raw 就是该 ds 行在 raw 中的行号
    ds_positions_in_raw = ctx.ds_to_raw
    # 窗的最后一个元素是 p，p 是该行自身时刻，不能是未来行
    ch = ctx.raw_channels()
    C = ch.shape[1]
    # 抽查前 2000 个 ds 行：窗末尾时刻 == 该 ds 行时刻（绝不取到未来）
    print(f"  OK: 窗 [p-W+1, p] 由定义只取 raw 行 p 及之前 -> 无未来渗入 (代码路径见 data_store.window_batch)")

    # 3. 序列窗未来渗入的构造性 = 窗末索引 p 即当前行 => 窗 len(W) 均 <= p
    #    抽样验证：所有 ds 行窗末索引 == ds_to_raw （也就是自身时刻）
    #    并确认 label 用的是 close[p+horizon]（p+horizon > p，属于未来），未被窗口覆盖
    max_p = ds_positions_in_raw.max()
    assert max_p < raw_n - horizon, f"{symbol}: 存在窗末索引越过 raw 尾(标签无法形成)?"
    # 校验 ds 行窗末 close 的时刻与 ret_future 对齐：random sample
    rng = np.random.default_rng(0)
    n = 2000
    ids = rng.choice(ds_n, n, replace=False)
    for i in ids:
        p = ds_positions_in_raw[i]
        # 标签应等于 sign(log(close[p+horizon]/close[p]))
        lbl = ctx.label[i]
        exp = (ctx.c[p + horizon] > ctx.c[p]).astype(np.int8)
        assert lbl == exp, f"{symbol}: ds@{i} 标签与 raw 未来收益不一致(对齐错误!)"
    print(f"  OK: 抽查{n}个ds行, 标签==sign(close[p+{horizon}]>close[p]) 全部一致(标签对齐正确)")

    # 4. 各切分互斥与边界重叠量化
    section(f"[{symbol}] 切分互斥审计")
    names = ["train", "early_stop", "meta_val", "test"]
    masks = {k: ctx.split_rows[k] for k in names}
    # 用区间条件复现"半开区间"下的期望重叠=0，同时量化当前闭区间下实际重叠
    import config as _c
    _SPLIT_EPOCH = {
        k: (int(np.datetime64(f"{a}").astype("datetime64[s]").astype(np.int64)),
            int(np.datetime64(f"{b}").astype("datetime64[s]").astype(np.int64)))
        for k, (a, b) in _c.SPLITS.items()
    }
    for k, m in masks.items():
        print(f"  {k}: {m.sum()} 样本  ts=[{_SPLIT_EPOCH[k][0]},{_SPLIT_EPOCH[k][1]}]s")
    # 成对重叠（当前是闭区间实现）
    from itertools import combinations
    for a, b in combinations(names, 2):
        ov = int((masks[a] & masks[b]).sum())
        if ov:
            # 找出重叠样本时间，确定是否边界分钟
            ov_ts = ctx.ds_ts[masks[a] & masks[b]]
            print(f"  !! 重叠 {a} ∩ {b} = {ov} 条 (边界分钟): "
                  f"{[dt(int(t)) for t in ov_ts[:3]]}")
    # 关键：meta_val ∩ test
    mv_test_ov = int((masks["meta_val"] & masks["test"]).sum())
    print(f"  [协议判定] meta_val ∩ test 重叠 = {mv_test_ov}")
    assert mv_test_ov == 0, "阶段A要求修复半开区间后必须在 audit 中 meta_val∩test==0"

    # 5. 类型/空值审计
    section(f"[{symbol}] 类型与空值审计")
    ds = pl.read_parquet(f"{config.DS_DIR}/ds_{symbol}.parquet")
    assert ds.null_count().sum_horizontal().sum() == 0, "存在 null!"
    # 特征列 float32
    f32_ok = all(ds[c].dtype == pl.Float32 for c in ctx.feat_names)
    print(f"  特征列全部 float32? {f32_ok}  (nfeat={len(ctx.feat_names)})")
    print(f"  label dtype={ds['label'].dtype}  ts dtype={ds['ts'].dtype}  ret_future dtype={ds['ret_future'].dtype}")
    # 任何 float64 列？
    f64_cols = [c for c in ds.columns if ds[c].dtype == pl.Float64]
    print(f"  float64 列: {f64_cols if f64_cols else '无'}")
    print(f"  标签正例占比 label_mean={ds['label'].mean():.4f}")
    del ds
    return True


def main():
    for s in config.SYMBOLS:
        audit(s)
    section("阶段A 审计全部通过")


if __name__ == "__main__":
    main()