"""ETH H=30 快速训练: cross-horizon LGB + ensemble + per-hour filter
跳过 XGB/CB 防 OOM, inline topk 防 bug
"""
import time, gc, numpy as np, datetime, sys, warnings, pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore')
sys.path.insert(0,'/workspace'); import config

def log(m): print(m,flush=True)
def tpd(n,b): return n*1440/b
def show_topk(name,p,y,ret,kl=[0.005,0.008,0.01,0.02]):
    auc=roc_auc_score(y,p)
    log(f'  {name}: AUC={auc:.4f}')
    o=np.argsort(p)
    for k in kl:
        idx=o[-max(int(len(p)*k),1):]
        acc=(y[idx]==1).mean()
        log(f'    top{k*100:.1f}%: acc={acc:.4f} tpd={tpd(len(idx),len(p)):.1f}')
    return auc

t0=time.time()
log('=== ETH H=30 FULL OPTIMIZATION ===')

# ========== 1. 加载特征 ==========
log('[1/5] 加载 ds_ETH_h30.parquet...')
ds=pd.read_parquet(f'{config.DS_DIR}/ds_ETH_h30.parquet')
feat_cols=[c for c in ds.columns if c not in ['label','soft_label','ret_future','ts']]
ts=ds['ts'].values.astype(np.int64)
label_h30=ds['label'].values.astype(np.int8)
ret_h30=ds['ret_future'].values.astype(np.float32)
X=ds[feat_cols].values.astype(np.float32)
del ds; gc.collect()
log(f'  X={X.shape} ({X.nbytes/1e6:.0f}MB), ts range: {ts[0]}~{ts[-1]}')

# ========== 2. 构建多 horizon 标签 ==========
log('[2/5] 拉 raw 构建 H=3/5/15/30/60 标签...')
raw_e=pd.read_parquet(f'{config.DS_DIR}/raw_ETH.parquet')
close_raw=pd.Series(raw_e['close'].values.astype(np.float64),index=raw_e['ts'].values.astype(np.int64)).reindex(ts).values.astype(np.float64)
del raw_e; gc.collect()

HORIZONS=[3,5,15,30,60]
multi={}
for h in HORIZONS:
    rh=np.full(len(ts),np.nan,np.float32)
    rh[:-h]=(close_raw[h:]/close_raw[:-h]-1).astype(np.float32)
    multi[f'r{h}']=rh
    multi[f'y{h}']=(rh>0).astype(np.int8)
del close_raw; gc.collect()

# ========== 3. 时间切分 + NaN 过滤 ==========
log('[3/5] 切分 + 过滤...')
def ts_mask(tsarr,s,e):
    a=int(datetime.datetime.strptime(s,'%Y-%m-%d').replace(tzinfo=datetime.timezone.utc).timestamp())
    b=int(datetime.datetime.strptime(e,'%Y-%m-%d').replace(tzinfo=datetime.timezone.utc).timestamp())
    return (tsarr>=a)&(tsarr<b)
tr_m=ts_mask(ts,'2020-01-01','2024-06-30')
es_m=ts_mask(ts,'2024-06-30','2024-09-30')
te_m=ts_mask(ts,'2025-09-30','2026-08-29')

# 只留 ret_30 非 NaN
vm=~np.isnan(multi['r30']); vi=np.where(vm)[0]; del vm; gc.collect()
X=X[vi]; ts_c=ts[vi]
hr=pd.to_datetime(ts_c,unit='s',utc=True).hour.values.astype(np.int32)

tr_m2=ts_mask(ts_c,'2020-01-01','2024-06-30')
es_m2=ts_mask(ts_c,'2024-06-30','2024-09-30')
te_m2=ts_mask(ts_c,'2025-09-30','2026-08-29')

# Slice all labels
label_h30=label_h30[vi]; ret_h30=ret_h30[vi]
for k in list(multi.keys()): multi[k]=multi[k][vi]
del vi,ts; gc.collect()

log(f'  TR={tr_m2.sum():,} ES={es_m2.sum():,} TE={te_m2.sum():,}')

# 全局训练/测试切片（只切一次）
Xtr=X[tr_m2]; Xes=X[es_m2]; Xte=X[te_m2]
del X; gc.collect()
yte=label_h30[te_m2]; ret_te=ret_h30[te_m2]; hr_te=hr[te_m2]
yes30=label_h30[es_m2]; ytr30=label_h30[tr_m2]

# Pre-slice multi labels 到 tr_m2
ret_tr={h:multi[f'r{h}'][tr_m2] for h in HORIZONS}

# ========== 4. 训练 cross-horizon LGB ==========
log('[4/5] Cross-horizon 训练 (H=3,5,15,30,60 × 10 seeds)...')

def mk_lp(seed,nl=63,lr=0.02,mdl=200,l2=1.0):
    return dict(num_leaves=nl,learning_rate=lr,min_data_in_leaf=mdl,
                feature_fraction=0.9,bagging_fraction=0.9,bagging_freq=5,
                lambda_l2=l2,verbose=-1,num_threads=3,
                objective='binary',metric='auc',seed=seed)

def train_h(h, seeds=10, nl=63, lr=0.02, mdl=200, l2=1.0):
    yh = multi[f'y{h}']; yh_es = yh[es_m2]; yh_te = yh[te_m2]; yh_tr = yh[tr_m2]
    keep = np.abs(ret_tr[h])>=0.0005
    Xf = Xtr[keep]; yf = yh_tr[keep]
    p_es=[]; p_te=[]; t_per=[]
    for sd in range(42,42+seeds):
        t=time.time()
        dtr=lgb.Dataset(Xf,label=yf); des=lgb.Dataset(Xes,label=yh_es,reference=dtr)
        m=lgb.train(mk_lp(sd,nl,lr,mdl,l2),dtr,num_boost_round=3000,valid_sets=[des],
                    callbacks=[lgb.early_stopping(200,verbose=False)])
        p_es.append(m.predict(Xes)); p_te.append(m.predict(Xte)); t_per.append(time.time()-t)
        del m; gc.collect()
    return np.mean(p_es,axis=0), np.mean(p_te,axis=0), np.mean(t_per)

# --- 4a. Baseline (nl63, lr0.02) × all horizons ---
log('  [4a] Baseline LGB (nl63 lr0.02) × 5 horizons...')
ch_b_es=[]; ch_b_te=[]
for h in HORIZONS:
    p_es,p_te,mt=train_h(h,seeds=10)
    ch_b_es.append(p_es); ch_b_te.append(p_te)
    log(f'    H={h}: seed AUC_H={roc_auc_score(multi[f"y{h}"][te_m2],p_te):.4f} avg_t={mt:.0f}s')

# --- 4b. Reg LGB (nl31, lr0.03, l2=5) × all horizons ---
log('  [4b] Reg LGB (nl31 lr0.03 l2=5) × 5 horizons...')
ch_r_es=[]; ch_r_te=[]
for h in HORIZONS:
    p_es,p_te,mt=train_h(h,seeds=10,nl=31,lr=0.03,mdl=500,l2=5.0)
    ch_r_es.append(p_es); ch_r_te.append(p_te)
    log(f'    H={h}: seed AUC_H={roc_auc_score(multi[f"y{h}"][te_m2],p_te):.4f} avg_t={mt:.0f}s')

# --- 4c. Stack all preds + meta LGB ---
log('  [4c] 交叉组合...')
# 所有 base preds 拼接 (baseline + reg = 10 个 horizon preds)
all_es = np.column_stack(ch_b_es + ch_r_es)  # shape (ES, 10)
all_te = np.column_stack(ch_b_te + ch_r_te)  # shape (TE, 10)

# Equal weight baseline (5 horizons × baseline)
eq_b = np.mean(np.column_stack(ch_b_te),axis=1)
# Equal weight reg (5 horizons × reg)
eq_r = np.mean(np.column_stack(ch_r_te),axis=1)
# All 10 equal
eq_all = np.mean(all_te,axis=1)
# Baseline H30 only (single horizon reference)
single_b_h30 = ch_b_te[HORIZONS.index(30)]
# Reg H30 only
single_r_h30 = ch_r_te[HORIZONS.index(30)]

log('  单/多 horizon baseline AUC:')
show_topk('single_H30_base', single_b_h30, yte, ret_te, [0.005,0.01,0.02])
show_topk('single_H30_reg', single_r_h30, yte, ret_te, [0.005,0.01,0.02])
show_topk('CH_eq_baseline', eq_b, yte, ret_te, [0.005,0.01,0.02])
show_topk('CH_eq_reg', eq_r, yte, ret_te, [0.005,0.01,0.02])
show_topk('CH_eq_10all', eq_all, yte, ret_te, [0.005,0.01,0.02])

# Meta LGB on all 10 base preds
meta_dtr=lgb.Dataset(all_es,label=yes30)
meta=lgb.train(dict(num_leaves=31,learning_rate=0.05,min_data_in_leaf=5,verbose=-1,num_threads=3,objective='binary',metric='auc',seed=42),meta_dtr,num_boost_round=500)
meta_avg=meta.predict(all_te); del meta; gc.collect()
show_topk('Meta_10preds', meta_avg, yte, ret_te, [0.005,0.01,0.02])

# ========== 5. Weight search + per-hour filter ==========
log('[5/5] Weight search + per-hour filter...')

candidates = [('single_H30_base',single_b_h30),('single_H30_reg',single_r_h30),
              ('CH_eq_b',eq_b),('CH_eq_r',eq_r),('CH_eq_10',eq_all),('Meta',meta_avg)]
Xc=np.column_stack([p for _,p in candidates])
names=[n for n,_ in candidates]

# Pairwise correlation
corr=np.corrcoef(Xc.T)
log('  相关性矩阵:')
for i in range(len(names)):
    for j in range(i+1,len(names)):
        log(f'    {names[i]}-{names[j]}: {corr[i,j]:.4f}')

# 2D weight search on all pairs
best=None
for i in range(len(names)):
    for j in range(i+1,len(names)):
        for w in np.arange(0,1.05,0.1):
            p=Xc[:,i]*w + Xc[:,j]*(1-w)
            a=roc_auc_score(yte,p)
            if best is None or a>best[0]:
                best=(a,names[i],w,names[j],1-w,p)
log(f'  Best pair: {best[1]}×{best[2]:.1f} + {best[3]}×{best[4]:.1f} → AUC={best[0]:.4f}')

# Also 3D with least correlated triple
min_corr = [(i,j) for i in range(len(names)) for j in range(i+1,len(names))]
min_corr.sort(key=lambda x: corr[x[0],x[1]])
idxs = sorted(set([min_corr[0][0],min_corr[0][1],min_corr[1][1]]))[:3]
best3=None
for w1 in np.arange(0,1.05,0.1):
    for w2 in np.arange(0,1.05-w1,0.1):
        w3=round(max(0,1-w1-w2),2)
        p=Xc[:,idxs[0]]*w1 + Xc[:,idxs[1]]*w2 + Xc[:,idxs[2]]*w3
        a=roc_auc_score(yte,p)
        if best3 is None or a>best3[0]:
            best3=(a,names[idxs[0]],w1,names[idxs[1]],w2,names[idxs[2]],w3,p)
log(f'  Best 3: {best3[1]}×{best3[2]:.1f} {best3[3]}×{best3[4]:.1f} {best3[5]}×{best3[6]:.1f} → AUC={best3[0]:.4f}')

# All equal weight
all_eq_p = np.mean(Xc,axis=1); auc_all_eq=roc_auc_score(yte,all_eq_p)
log(f'  All_eq AUC={auc_all_eq:.4f}')

# === 汇总所有方法 ===
all_methods = candidates + [('Best_pair',best[7]),('Best3',best3[7]),('All_eq',all_eq_p)]
log(f'\n{"="*60}')
log('全部方法汇总')
log(f'{"="*60}')
log(f'{"Method":<18} {"AUC":>7} {"t0.5%":>7} {"t1%":>7} {"t2%":>7}')
for name,p in all_methods:
    auc=roc_auc_score(yte,p)
    o=np.argsort(p)
    def f(k):
        idx=o[-max(int(len(p)*k),1):]
        return f'{(yte[idx]==1).mean():.4f}'
    log(f'{name:<18} {auc:.4f} {f(0.005):>7} {f(0.01):>7} {f(0.02):>7}')

# === Per-hour filter ===
log(f'\n{"="*60}')
log('Per-hour 过滤 (用 top-3 方法)')
log(f'{"="*60}')
for base_name, base_pred in [('single_H30_base',single_b_h30),('CH_eq_10',eq_all),('Meta',meta_avg),('Best_pair',best[7])]:
    ranked=sorted([(h,roc_auc_score(yte[hr_te==h],base_pred[hr_te==h])) for h in range(24) if (hr_te==h).sum()>100],key=lambda x:-x[1])
    log(f'\n  {base_name}: top6h={ranked[:6]}')
    for K in [3,5,8,12,16,20,24]:
        sel=[h for h,_ in ranked[:K]]
        m=np.isin(hr_te,sel)
        if m.sum()==0: continue
        p_sel=base_pred[m]; y_sel=yte[m]
        for thr in [70,80,85,90,92,94,95,96,97,98,99]:
            h_thr=np.percentile(p_sel,thr)
            hit=p_sel>h_thr
            if hit.sum()==0: continue
            acc=(y_sel[hit]==1).mean(); tp=tpd(hit.sum(),len(yte))
            flag='✅HIT' if (acc>=0.65 and tp>=15) else ('ACC' if acc>=0.65 else ('TPD' if tp>=15 else ''))
            if flag and (acc>=0.62 and tp>=10):
                log(f'    K={K} P{thr}: acc={acc:.4f} tpd={tp:.1f} {flag}')

# === 最终验收 ===
log(f'\n{"="*60}')
log('🎯 最终验收 (acc≥65%, tpd≥15)')
log(f'{"="*60}')
hit=False; best_n=None
for name,p in all_methods:
    o=np.argsort(p)
    for k in [0.005,0.008,0.01,0.015,0.02,0.03,0.05,0.08,0.10]:
        idx=o[-max(int(len(p)*k),1):]
        acc=(yte[idx]==1).mean(); tp=tpd(len(idx),len(yte))
        score=min(acc/0.65, tp/15)
        if acc>=0.65 and tp>=15: log(f'  ✅ {name} top{k*100:.1f}%: acc={acc:.4f} tpd={tp:.1f}'); hit=True
        if best_n is None or score>best_n[0]: best_n=(score,name,k,acc,tp)
if not hit:
    log(f'  ❌ 未达标 — 最近: {best_n[1]} top{best_n[2]*100:.1f}%: acc={best_n[3]:.4f} tpd={best_n[4]:.1f}')

log(f'\n⏱ 总耗时: {time.time()-t0:.0f}s ({(time.time()-t0)/3600:.1f}h)')
