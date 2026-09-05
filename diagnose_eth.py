#!/usr/bin/env python3
"""诊断 ETH 数据质量问题。"""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from data_store import AssetContext

# 加载两个数据集
ctx_btc = AssetContext('BTC', horizon=30)
ctx_eth = AssetContext('ETH', horizon=30)

print('=' * 65)
print('1. 数据基本信息')
print('=' * 65)
print(f'  BTC 总行数: {len(ctx_btc.ds_ts):,}')
print(f'  ETH 总行数: {len(ctx_eth.ds_ts):,}')
print(f'  BTC 训练集: {ctx_btc.split_rows["train"].sum():,}')
print(f'  ETH 训练集: {ctx_eth.split_rows["train"].sum():,}')

print(f'\n{"=" * 65}')
print('2. 标签分布 (y)')
print(f'{"=" * 65}')
for name, ctx in [('BTC', ctx_btc), ('ETH', ctx_eth)]:
    y = ctx.label
    up = (y > 0.5).sum()
    dn = (y <= 0.5).sum()
    print(f'  {name}: 涨={up:,} ({up/len(y)*100:.1f}%) 跌={dn:,} ({dn/len(y)*100:.1f}%)')

print(f'\n{"=" * 65}')
print('3. 收益率分布 (retf - 测试集)')
print(f'{"=" * 65}')
for name, ctx in [('BTC', ctx_btc), ('ETH', ctx_eth)]:
    retf = ctx.retf('test')
    fin = np.isfinite(retf)
    print(f'  {name}:')
    print(f'    均值: {np.nanmean(retf)*10000:.2f} bps')
    print(f'    标准差: {np.nanstd(retf)*10000:.2f} bps')
    print(f'    中位数: {np.nanmedian(retf)*10000:.2f} bps')
    print(f'    P99: {np.nanpercentile(retf, 99)*10000:.2f} bps')
    print(f'    P1:  {np.nanpercentile(retf, 1)*10000:.2f} bps')

print(f'\n{"=" * 65}')
print('4. 训练集收益率分布 (用于权重计算)')
print(f'{"=" * 65}')
for name, ctx in [('BTC', ctx_btc), ('ETH', ctx_eth)]:
    retf = ctx.retf('train')
    abs_retf = np.abs(retf)
    print(f'  {name}:')
    print(f'    均值: {np.nanmean(retf)*10000:.2f} bps')
    print(f'    标准差: {np.nanstd(retf)*10000:.2f} bps')
    print(f'    中位数|ret|: {np.nanmedian(abs_retf)*10000:.2f} bps')
    print(f'    P99|ret|: {np.nanpercentile(abs_retf, 99)*10000:.2f} bps')
    print(f'    有效样本: {np.isfinite(retf).sum():,}')

print(f'\n{"=" * 65}')
print('5. 检查特征列')
print(f'{"=" * 65}')
FEATURES = [
    'lr_5', 'lr_15', 'lr_30', 'lr_120', 'lr_240', 'mom_60',
    'z_10', 'z_30', 'z_60', 'z_120',
    'rvol_30', 'rvol_60', 'rvol_ratio_60_5', 'rvol_z_60', 'rvol_dir',
    'pos_30', 'pos_60', 'pos_120', 'pos_240',
    'dd_240', 'ru_240',
    'hh_dd_60', 'll_ru_60', 'body_pos_60',
    'body_ratio', 'up_wick', 'lo_wick', 'ngreen_10', 'gap', 'max_range_30',
    'tbr_z_30', 'cvd_30', 'cvd_60',
    'buyvol_strength_30', 'tb_act_60', 'ts_act_60', 'tb_acc_30',
    'cvd_dir_30', 'tbr_hi_60', 'lr_skew_60', 'up_body_ratio_30', 'mom_align_30_240',
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'is_us', 'is_eu', 'ret_day',
]
for name, ctx in [('BTC', ctx_btc), ('ETH', ctx_eth)]:
    cols = set(ctx.ds.columns)
    missing = [f for f in FEATURES if f not in cols]
    if missing:
        print(f'  {name} 缺失特征: {missing}')
    else:
        print(f'  {name}: 所有特征都存在')

print(f'\n{"=" * 65}')
print('6. 特征值统计 (训练集, 选几个关键特征)')
print(f'{"=" * 65}')
check_feats = ['lr_15', 'rvol_60', 'z_60', 'tbr_z_30', 'cvd_30']
for name, ctx in [('BTC', ctx_btc), ('ETH', ctx_eth)]:
    print(f'  {name}:')
    trm = ctx.split_rows['train']
    for f in check_feats:
        if f in ctx.ds_columns:
            v = ctx.ds[f][trm].values.astype(np.float64)
            fin = np.isfinite(v)
            print(f'    {f:<15}: mean={v[fin].mean():.4f} std={v[fin].std():.4f} '
                  f'p1={np.percentile(v[fin],1):.4f} p99={np.percentile(v[fin],99):.4f} '
                  f'nan={(~fin).sum():,}')

print(f'\n{"=" * 65}')
print('7. 检查数据切分时间范围')
print(f'{"=" * 65}')
for name, ctx in [('BTC', ctx_btc), ('ETH', ctx_eth)]:
    ts = ctx.ds_ts
    print(f'  {name}:')
    print(f'    总范围: {np.datetime64(ts[0], "s"):%Y-%m-%d} ~ {np.datetime64(ts[-1], "s"):%Y-%m-%d}')
    for split in ['train', 'early_stop', 'meta_val', 'test']:
        mask = ctx.split_rows[split]
        if mask.sum() > 0:
            split_ts = ts[mask]
            print(f'    {split:<12}: {np.datetime64(split_ts[0], "s"):%Y-%m-%d} ~ '
                  f'{np.datetime64(split_ts[-1], "s"):%Y-%m-%d} ({mask.sum():,} 行)')

print(f'\n{"=" * 65}')
print('8. 检查 retf 函数行为')
print(f'{"=" * 65}')
for name, ctx in [('BTC', ctx_btc), ('ETH', ctx_eth)]:
    tr_retf = ctx.retf('train')
    es_retf = ctx.retf('early_stop')
    mv_retf = ctx.retf('meta_val')
    te_retf = ctx.retf('test')
    print(f'  {name}:')
    print(f'    train retf: mean={np.nanmean(tr_retf)*10000:.2f}bps std={np.nanstd(tr_retf)*10000:.2f}bps n={np.isfinite(tr_retf).sum():,}')
    print(f'    early_stop retf: mean={np.nanmean(es_retf)*10000:.2f}bps std={np.nanstd(es_retf)*10000:.2f}bps n={np.isfinite(es_retf).sum():,}')
    print(f'    meta_val retf: mean={np.nanmean(mv_retf)*10000:.2f}bps std={np.nanstd(mv_retf)*10000:.2f}bps n={np.isfinite(mv_retf).sum():,}')
    print(f'    test retf: mean={np.nanmean(te_retf)*10000:.2f}bps std={np.nanstd(te_retf)*10000:.2f}bps n={np.isfinite(te_retf).sum():,}')

del ctx_btc, ctx_eth
print('\n完成.')