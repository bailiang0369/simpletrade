"""诊断 AUC 天花板 + 尝试真正的突破方法 (不是 coverage trade-off)。

核心诊断:
  1. 理论上限: 用 ret 本身预测自己 (cheating baseline) — 看 label 噪声天花板在哪
  2. 特征重要性 — 哪些真正有用, 哪些是噪声
  3. 噪声分桶 AUC — 在 top/bottom |ret| 样本上分别测 AUC, 看模型是否只在高噪声处失效

突破方向 (目标 AUC 0.55+):
  A. Cross-asset label stacking — BTC 模型的预测 P 作为 ETH 的新特征
  B. Soft-label: 用 ret_future 幅度加权 label (不是 0/1)
  C. Lambdarank 正确配置 — 大 group size 试试
  D. 更大模型 + 去噪联合: nL=512 + eps=0.0002
"""
import sys; sys.path.insert(0,'/workspace')
import os, gc, time, numpy as np, lightgbm as lgb
from sklearn.metrics import roc_auc_score
import config; from data_store import AssetContext

SEEDS = [42,49,56,63,70,77,84,91,98,105]
T0 = time.time()
def now(): return f"{(time.time()-T0)/60:.1f}m"

def rank_ens(a):
    P=np.stack(a,0); R=np.zeros_like(P)
    for i in range(P.shape[0]): R[i]=np.argsort(np.argsort(P[i])).astype(np.float64)/(P.shape[1]-1)
    return R.mean(0)

def train_ens(ctx, Xtr, ytr, Xes, yes, wtr, params):
    mm=ctx.split_rows["meta_val"]; mt=ctx.split_rows["test"]
    Pmv, Pte = [], []
    for s in SEEDS:
        tr=lgb.Dataset(Xtr,label=ytr,weight=wtr if wtr is not None else None)
        es=lgb.Dataset(Xes,label=yes,reference=tr)
        m=lgb.train({**params,"seed":s}, tr, 3000, [es],
                    callbacks=[lgb.early_stopping(100),lgb.log_evaluation(0)])
        Pmv.append(m.predict(ctx.Xall[mm]).astype(np.float64))
        Pte.append(m.predict(ctx.Xall[mt]).astype(np.float64))
        gc.collect()
    return rank_ens(Pmv), rank_ens(Pte)

def show(ctx, Pmv, Pte, title):
    for split,P in [("mv",Pmv),("te",Pte)]:
        m=ctx.split_rows["meta_val"] if split=="mv" else ctx.split_rows["test"]
        auc=roc_auc_score(ctx.label[m], P)
        print(f"  [{title}] {split} AUC={auc:.4f}", flush=True)

base_p = {"objective":"binary","metric":"auc","verbose":-1,"num_leaves":127,
          "min_data_in_leaf":200,"learning_rate":0.05,"feature_fraction":0.8,"bagging_fraction":0.8}

# ===== 诊断 =====
print("="*70, flush=True); print("诊断 AUC 天花板", flush=True); print("="*70, flush=True)
ctx=AssetContext("ETH",horizon=15)
ret=ctx.retf("train"); ret_es=ctx.retf("early_stop"); ret_mv=ctx.retf("meta_val"); ret_te=ctx.retf("test")
abs_ret=np.abs(ret); abs_ret_es=np.abs(ret_es); abs_ret_te=np.abs(ret_te)

# 1) Cheating baseline: ret 的符号 vs binary label
print("\n1) Cheating baseline (ret sign vs label):", flush=True)
for split_name, ret_s, y_s in [("train",ret,ctx.label[ctx.split_rows["train"]]),
                                ("early_stop",ret_es,ctx.label[ctx.split_rows["early_stop"]]),
                                ("test",ret_te,ctx.label[ctx.split_rows["test"]])]:
    p_cheat = (ret_s>0).astype(float)
    auc_cheat = roc_auc_score(y_s, p_cheat)
    print(f"  {split_name}: ret-sign AUC vs binary label = {auc_cheat:.4f}", flush=True)
    # 看看 ret 很小的样本 AUC 怎样
    for th in [0.0005, 0.001, 0.002]:
        mask = np.abs(ret_s) > th
        if mask.sum()<100: continue
        auc2 = roc_auc_score(y_s[mask], p_cheat[mask])
        print(f"    |ret|>{th}: n={mask.sum():,}, AUC={auc2:.4f}", flush=True)

# 2) 噪声分桶: 在不同噪声程度的样本上测 baseline 模型 AUC
print("\n2) Baseline 模型在不同噪声桶上的 AUC:", flush=True)
# 先训一个模型
m_base=lgb.train({**base_p,"seed":42},
                  lgb.Dataset(ctx.Xall[ctx.split_rows["train"]], label=ctx.label[ctx.split_rows["train"]].astype(int),
                              weight=np.clip(np.abs(ret)*50,0.5,5.0)),
                  3000, [lgb.Dataset(ctx.Xall[ctx.split_rows["early_stop"]], label=ctx.label[ctx.split_rows["early_stop"]].astype(int),
                                      reference=lgb.Dataset(ctx.Xall[ctx.split_rows["train"]], label=ctx.label[ctx.split_rows["train"]].astype(int)))],
                  callbacks=[lgb.early_stopping(100),lgb.log_evaluation(0)])
# 在 test 上按噪声分桶看 AUC
p_te = m_base.predict(ctx.Xall[ctx.split_rows["test"]])
y_te = ctx.label[ctx.split_rows["test"]]
noise_percs = [0, 10, 25, 50, 75, 90, 100]
print(f"  |ret| 分桶 (test, n={len(y_te):,}):", flush=True)
for i in range(len(noise_percs)-1):
    lo, hi = np.percentile(abs_ret_te, noise_percs[i]), np.percentile(abs_ret_te, noise_percs[i+1])
    mask = (abs_ret_te >= lo) & (abs_ret_te < hi)
    if mask.sum()<500: continue
    auc = roc_auc_score(y_te[mask], p_te[mask])
    print(f"    |ret| [{lo:.4f},{hi:.4f}]: n={mask.sum():,}, AUC={auc:.4f}", flush=True)

# ===== 突破方向 A: Cross-asset label stacking =====
print("\n" + "="*70, flush=True); print("突破 A: BTC h15 预测 P 作为 ETH h15 新特征", flush=True); print("="*70, flush=True)
# 先训 BTC h15 模型
print(f"\n[{now()}] 训 BTC h15 ens...", flush=True)
ctx_btc=AssetContext("BTC",horizon=15)
Xtr_b=ctx_btc.Xall[ctx_btc.split_rows["train"]]; Xes_b=ctx_btc.Xall[ctx_btc.split_rows["early_stop"]]
ytr_b=ctx_btc.label[ctx_btc.split_rows["train"]].astype(np.float64); yes_b=ctx_btc.label[ctx_btc.split_rows["early_stop"]].astype(np.float64)
ret_b=ctx_btc.retf("train"); wtr_b=np.clip(np.abs(ret_b)*50,0.5,5.0)

btc_Pmv, btc_Pte = [], []
for s in SEEDS:
    tr=lgb.Dataset(Xtr_b,label=ytr_b,weight=wtr_b); es=lgb.Dataset(Xes_b,label=yes_b,reference=tr)
    m=lgb.train({**base_p,"seed":s}, tr, 3000, [es],
                callbacks=[lgb.early_stopping(100),lgb.log_evaluation(0)])
    btc_Pmv.append(m.predict(ctx_btc.Xall[ctx_btc.split_rows["meta_val"]]).astype(np.float64))
    btc_Pte.append(m.predict(ctx_btc.Xall[ctx_btc.split_rows["test"]]).astype(np.float64))
btc_Pmv=rank_ens(btc_Pmv); btc_Pte=rank_ens(btc_Pte)
btc_auc = roc_auc_score(ctx_btc.label[ctx_btc.split_rows["test"]], btc_Pte)
print(f"  BTC h15 test AUC={btc_auc:.4f}", flush=True)

# 把 BTC h15 的预测 P 作为 ETH 的新特征
# 注意: ETH 的 train 区间 (2020-01→2024-06) 和 meta_val/test 区间 BTC 也有数据
# 所以 BTC 的预测 P 可以在 ETH 的所有 split 上算出来
print(f"\n[{now()}] 叠加 BTC P 作为 ETH 新特征...", flush=True)
# 需要把 BTC 模型应用到 ETH 的所有 split
ctx_eth=AssetContext("ETH",horizon=15)
# 用所有时间重训 BTC (ETH train 区间的 BTC 数据)
btc_all_ctx = AssetContext("BTC", horizon=15)
# BTC h15 在 ETH train 区间内的预测
tr_mask_eth = ctx_eth.split_rows["train"]
es_mask_eth = ctx_eth.split_rows["early_stop"]
mm_mask = ctx_eth.split_rows["meta_val"]
mt_mask = ctx_eth.split_rows["test"]

# 关键: BTC 和 ETH 时间戳对齐吗? 如果同源应该是对齐的
# 我们需要用 BTC 模型预测 ETH 时间戳对应的 BTC 值
# 但 AssetContext 默认返回的是自己的数据
# 简化: BTC 和 ETH 都是 1min 频率, 时间戳应该一致, 直接用 BTC 在对应 split 的预测

# 重训 BTC, 然后在 BTC 的每个 split 上预测 (BTC/ETH 时间戳应该一致因为同 bn_data)
# 实际上 AssetContext 是按 symbol 返回数据, ETH 和 BTC 的时间戳是独立从各自 parquet 读的
# 我们假设它们是同步的 (同交易所同频率)

# 一个更简单的方法: 用 BTC 在 ETH 训练区间内的 ret 作为特征 (已有的 cross-asset 特征)
# 但我们要的是 BTC label stacking, 即 BTC 模型的预测作为特征
# 最简单的做法: 用 BTC 全量训练, 然后在 BTC/ETH 所有时间点预测

# 直接在 BTC 全时间范围用 BTC 模型预测 (BTC 数据覆盖 2020-2026 和 ETH 应该一样)
# 然后把 BTC 的预测时间戳对齐 ETH 的时间戳
# 但 ETH 只返回自己的 Xall, 我们需要 join BTC 的预测

# 换思路: 利用 AssetContext 里已经有的 cross-asset 特征 + 额外加 BTC 的 h15 预测
# 但 AssetContext 只给 ETH 自己的 Xall, 没有 BTC 的预测

# 最简方法: 直接在 ETH 数据上加 BTC 相关的 ret_future 作为特征
# 因为 cross-asset label stacking 本质是让模型看到 "BTC 未来会涨跌" 这个信息
# 但这在 train 时不可用 (label leakage)

# 哦对 — label stacking 应该是: 用 meta_train (meta_val) 训练 meta-learner
# 不是把 label 当训练特征

# 回到基础: 我们要找 ETH h15 还没用到的有用信号
# 让我先看看现有 cross-asset 特征有什么

print("\n现有 cross-asset 特征 (ETH):", flush=True)
cross_feats = [f for f in ctx_eth.feat_names if 'btc' in f.lower() or 'cross' in f.lower()]
print(f"  {cross_feats}", flush=True)
print(f"  共 {len(cross_feats)} 个 cross-asset 特征", flush=True)

# ===== 突破方向 B: Soft-label / 幅度加权 =====
print("\n" + "="*70, flush=True); print("突破 B: Soft-label (ret 幅度加权)", flush=True); print("="*70, flush=True)
ctx_eth2=AssetContext("ETH",horizon=15)
Xtr_e=ctx_eth2.Xall[ctx_eth2.split_rows["train"]]
ytr_b=ctx_eth2.label[ctx_eth2.split_rows["train"]].astype(np.float64)
ret_e=ctx_eth2.retf("train"); yes_e=ctx_eth2.label[ctx_eth2.split_rows["early_stop"]].astype(np.float64)
Xes_e=ctx_eth2.Xall[ctx_eth2.split_rows["early_stop"]]

# soft label: 把 ret 压缩到 [0,1] 区间
soft = 0.5 + 0.5 * np.clip(ret_e, -0.02, 0.02) / 0.02
print(f"  soft label range: [{soft.min():.3f}, {soft.max():.3f}], mean={soft.mean():.3f}", flush=True)
Pmv_s, Pte_s = train_ens(ctx_eth2, Xtr_e, soft, Xes_e, yes_e, None,
                         {**base_p,"metric":"l2"})
show(ctx_eth2, Pmv_s, Pte_s, "soft_label")

# ===== 突破方向 C: 大模型 + 去噪联合 =====
print("\n" + "="*70, flush=True); print("突破 C: nL=512 + 去噪 ε=0.0002 (联合最好两个)", flush=True); print("="*70, flush=True)
eps=0.0002
m_dn = np.abs(ret_e) > eps
Xtr_d=Xtr_e[m_d]; soft_d=soft[m_d] if 'soft' in dir() else ytr_b[m_d]
wtr_d = np.clip(np.abs(ret_e[m_d])*50,0.5,5.0)
print(f"  去噪后保留: {len(Xtr_d)}/{len(Xtr_e)} ({len(Xtr_d)/len(Xtr_e)*100:.1f}%)", flush=True)

Pmv_b, Pte_b = train_ens(ctx_eth2, Xtr_d, ytr_b[m_d], Xes_e, yes_e, wtr_d,
                         {**base_p, "num_leaves": 512, "min_data_in_leaf": 100})
show(ctx_eth2, Pmv_b, Pte_b, "nL512_eps0.0002")

# soft + 去噪 + 大模型
Pmv_bs, Pte_bs = train_ens(ctx_eth2, Xtr_d, soft_d, Xes_e, yes_e, wtr_d,
                           {**base_p, "num_leaves": 512, "min_data_in_leaf": 100, "metric":"l2"})
show(ctx_eth2, Pmv_bs, Pte_bs, "soft_nL512_eps0.0002")

# ===== 汇总 =====
print("\n" + "="*70, flush=True); print("汇总: AUC 天花板能否突破?", flush=True); print("="*70, flush=True)
print("""
判断:
  如果 cheatin AUC < 0.60 → label 噪声是根本瓶颈, 任何模型都到不了 0.60+
  如果 cheatin AUC > 0.80 → 特征/模型还有很大空间
  如果 soft-label / nL512+去噪 的 AUC > 0.545 → 真正突破
""", flush=True)

print(f"\n[{now()}] ✅ 全部完成", flush=True)