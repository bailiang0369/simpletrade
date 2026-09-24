"""真正把 funding 特征和 base 特征合并后训练 BTC h15
不是 ensemble 两个独立模型，而是让一个模型同时看到两类特征
内存优化: 在 split 内部 hstack, 避免全量大数组
"""
import numpy as np, time, gc, sys, warnings
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore'); sys.stdout.reconfigure(line_buffering=True)
T0=time.time(); NPY='data/splits_npy'; SEEDS=[42,49,56,63,70]
def log(*a): print(' '.join(str(x) for x in a), flush=True)

def get_combined(split_name):
    """Load base + funding splits for a given split, hstack, return (X, y)"""
    Xb = np.load(f'{NPY}/BTC_h15_{split_name}_X.npy').astype(np.float32)
    yb = np.load(f'{NPY}/BTC_h15_{split_name}_y.npy').astype(np.int32)
    Xf = np.load(f'{NPY}/BTC_h15_fund_{split_name}_X.npy').astype(np.float32)
    yf = np.load(f'{NPY}/BTC_h15_fund_{split_name}_y.npy').astype(np.int32)
    ret_b = np.load(f'{NPY}/BTC_h15_{split_name}_ret.npy')
    
    # base 和 funding 的行数可能略有差异（base ds warmup 更宽）
    # 用 y 对齐: 找共同 y 值最多的索引
    # 实际上 funding split 是从原始 raw 构建的, base 是从 ds 构建的
    # 先简单试: 如果 yb.shape == yf.shape 就直接 hstack
    if Xb.shape[0] == Xf.shape[0]:
        X = np.hstack([Xb, Xf])
        y = yb
        log(f"    {split_name}: base={Xb.shape} fund={Xf.shape} combined={X.shape}")
    else:
        log(f"    {split_name}: SHAPE MISMATCH base={Xb.shape[0]} fund={Xf.shape[0]}")
        # 用 funding 做参考（它的 y 是原始 raw 的 label, 而 base ds 可能有不同 warmup）
        # 简单起见：用 funding split 的行（用 base ds 里对应时间戳的行）
        # 其实直接用 funding split 训练就好——它的 y 是对的，只是特征少
        X = Xb  # fallback: 只用 base
        y = yb
    del Xf, Xb; gc.collect()
    return X, y, ret_b

def train(Xtr,ytr,Xes,yes,Xte,yte,sw=None,label=""):
    params={'objective':'binary','metric':'auc','learning_rate':0.05,'num_leaves':63,
            'min_child_samples':200,'feature_fraction':0.8,'bagging_fraction':0.8,
            'bagging_freq':5,'lambda_l2':0.1,'verbose':-1,'n_jobs':4}
    pte,pes=[],[]; t0=time.time()
    for sd in SEEDS:
        params['seed']=sd
        tr=lgb.Dataset(Xtr,label=ytr,weight=sw); es=lgb.Dataset(Xes,label=yes,reference=tr)
        m=lgb.train(params,tr,5000,[es],callbacks=[lgb.early_stopping(100),lgb.log_evaluation(0)])
        pte.append(m.predict(Xte)); pes.append(m.predict(Xes))
    aes=roc_auc_score(yes,np.mean(pes,0)); ate=roc_auc_score(yte,np.mean(pte,0))
    log(f"  {label:35s}: es={aes:.4f} te={ate:.4f} ({time.time()-t0:.0f}s)")
    return aes,ate,np.mean(pte,0)

log("[1] Check split alignment...")
# 检查 train 行数
nb = np.load(f'{NPY}/BTC_h15_train_y.npy').shape[0]
nf = np.load(f'{NPY}/BTC_h15_fund_train_y.npy').shape[0]
log(f"  base train y rows: {nb:,}")
log(f"  fund train y rows:  {nf:,}")
log(f"  diff: {nf-nb}")

# 如果 funding 行数 >= base 行数，用 base 作为 filter（更严格的 warmup）
# 或者直接用 funding split 当参考，从 base ds 里挑对应的时间戳
# 简化方案: 直接尝试 hstack，如果 shapes 一致就用，否则各训各的然后调 ensemble

# 先各自训练 baseline 5-seed, 看能不能复现之前的结果
log("\n[2] Train base LGB 5-seed (reference)...")
Xtr_b=np.load(f'{NPY}/BTC_h15_train_X.npy').astype(np.float32)
ytr_b=np.load(f'{NPY}/BTC_h15_train_y.npy').astype(np.int32)
Xes_b=np.load(f'{NPY}/BTC_h15_early_stop_X.npy').astype(np.float32)
yes_b=np.load(f'{NPY}/BTC_h15_early_stop_y.npy').astype(np.int32)
Xte_b=np.load(f'{NPY}/BTC_h15_test_X.npy').astype(np.float32)
yte_b=np.load(f'{NPY}/BTC_h15_test_y.npy').astype(np.int32)
_,_,pte_base = train(Xtr_b,ytr_b,Xes_b,yes_b,Xte_b,yte_b,label='BTC h15 base 56')

log("\n[3] Try base+funding in-memory hstack...")
# base: n_train=2.4M×56, funding: n_train=2.4M×12
# hstack 后 2.4M×68 × 4byte ≈ 650MB float32 — 可能 OOM
# 逐个 split 处理，做完即删
import gc
def hstack_splits():
    result = {}
    for split in ['train','early_stop','test']:
        Xb = np.load(f'{NPY}/BTC_h15_{split}_X.npy').astype(np.float32)
        Xf = np.load(f'{NPY}/BTC_h15_fund_{split}_X.npy').astype(np.float32)
        yb = np.load(f'{NPY}/BTC_h15_{split}_y.npy').astype(np.int32)
        if Xb.shape[0] == Xf.shape[0]:
            X = np.hstack([Xb, Xf])
        else:
            log(f"    {split}: shape mismatch base={Xb.shape[0]} fund={Xf.shape[0]}, using min")
            n = min(Xb.shape[0], Xf.shape[0])
            X = np.hstack([Xb[:n], Xf[:n]])
            yb = yb[:n]
        result[split] = (X, yb)
        log(f"    {split}: {X.shape}")
        del Xb, Xf; gc.collect()
    return result

try:
    cs = hstack_splits()
    Xtr_c,ytr_c = cs['train']; Xes_c,yes_c = cs['early_stop']; Xte_c,yte_c = cs['test']
    _,_,pte_comb = train(Xtr_c,ytr_c,Xes_c,yes_c,Xte_c,yte_c,label='BTC h15 base+fund 68')
    log(f"\n  base-only te:    {roc_auc_score(yte_b, pte_base):.4f}")
    log(f"  base+fund  te:    {roc_auc_score(yte_c, pte_comb):.4f}")
    log(f"  DIFF (comb - base): {roc_auc_score(yte_c, pte_comb) - roc_auc_score(yte_b, pte_base):+.4f}")
    
    # negw on combined
    ret_b = np.load(f'{NPY}/BTC_h15_train_ret.npy')
    sw_c = np.where(np.abs(ret_b)>=np.quantile(np.abs(ret_b),0.90),0.3,1.0).astype(np.float32)
    _,_,pte_cnw = train(Xtr_c,ytr_c,Xes_c,yes_c,Xte_c,yte_c,sw_c,'BTC h15 base+fund+negw')
    log(f"  DIFF (comb+negw - base): {roc_auc_score(yte_c, pte_cnw) - roc_auc_score(yte_b, pte_base):+.4f}")
    
    del cs, Xtr_c, ytr_c, Xes_c, yes_c, Xte_c, yte_c; gc.collect()
except MemoryError:
    log("  OOM during hstack! Skip combined training.")

log(f"\nTOTAL: {time.time()-T0:.0f}s")
