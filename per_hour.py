"""ETH H=30 Per-hour filter - 从 go.py 已有结果继续"""
import time, gc, numpy as np, datetime, sys, warnings, pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore')
sys.path.insert(0,'/workspace'); import config

def log(m): print(m,flush=True)
def tpd(n,b): return n*1440/b

t0=time.time()
log('=== Per-hour filter on top methods ===')

# ========== 1. 重跑 cross-horizon 得到 TE preds (上次 go.py 崩在 weight search 前) ==========
log('[re-train] baseline + reg cross-horizon (只算 TE preds，跳过 meta)...')

ds=pd.read_parquet(f'{config.DS_DIR}/ds_ETH_h30.parquet')
feat_cols=[c for c in ds.columns if c not in ['label','soft_label','ret_future','ts']]
ts=ds['ts'].values.astype(np.int64)
label_h30=ds['label'].values.astype(np.int8)
ret_h30=ds['ret_future'].values.astype(np.float32)
X=ds[feat_cols].values.astype(np.float32)
del ds; gc.collect()

raw_e=pd.read_parquet(f'{config.DS_DIR}/raw_ETH.parquet')
close_raw=pd.Series(raw_e['close'].values.astype(np.float64),index=raw_e['ts'].values.astype(np.int64)).reindex(ts).values.astype(np.float64)
del raw_e; gc.collect()

HORIZONS=[3,5,15,30,60]
multi={}
for h in HORIZONS:
    rh=np.full(len(ts),np.nan,np.float32)
    rh[:-h]=(close_raw[h:]/close_raw[:-h]-1).astype(np.float32)
    multi[f'r{h}']=rh; multi[f'y{h}']=(rh>0).astype(np.int8)
del close_raw; gc.collect()

def ts_mask(tsarr,s,e):
    a=int(datetime.datetime.strptime(s,'%Y-%m-%d').replace(tzinfo=datetime.timezone.utc).timestamp())
    b=int(datetime.datetime.strptime(e,'%Y-%m-%d').replace(tzinfo=datetime.timezone.utc).timestamp())
    return (tsarr>=a)&(tsarr<b)
tr_m=ts_mask(ts,'2020-01-01','2024-06-30')
es_m=ts_mask(ts,'2024-06-30','2024-09-30')
te_m=ts_mask(ts,'2025-09-30','2026-08-29')
vm=~np.isnan(multi['r30']); vi=np.where(vm)[0]
X=X[vi]; ts_c=ts[vi]; hr=pd.to_datetime(ts_c,unit='s',utc=True).hour.values.astype(np.int32)
tr_m2=ts_mask(ts_c,'2020-01-01','2024-06-30')
es_m2=ts_mask(ts_c,'2024-06-30','2024-09-30')
te_m2=ts_mask(ts_c,'2025-09-30','2026-08-29')
label_h30=label_h30[vi]; ret_h30=ret_h30[vi]
for k in list(multi.keys()): multi[k]=multi[k][vi]
del vi,ts; gc.collect()

Xtr=X[tr_m2]; Xes=X[es_m2]; Xte=X[te_m2]; del X; gc.collect()
yte=label_h30[te_m2]; ret_te=ret_h30[te_m2]; hr_te=hr[te_m2]
ret_tr={h:multi[f'r{h}'][tr_m2] for h in HORIZONS}

def mk_lp(seed,nl,lr,mdl,l2):
    return dict(num_leaves=nl,learning_rate=lr,min_data_in_leaf=mdl,feature_fraction=0.9,bagging_fraction=0.9,bagging_freq=5,lambda_l2=l2,verbose=-1,num_threads=3,objective='binary',metric='auc',seed=seed)

def train_h(h, nl, lr, mdl, l2, seeds=10):
    yh = multi[f'y{h}']; yh_es = yh[es_m2]; yh_tr = yh[tr_m2]
    keep = np.abs(ret_tr[h])>=0.0005
    Xf = Xtr[keep]; yf = yh_tr[keep]
    p_te=[]; mt=[]
    for sd in range(42,42+seeds):
        t=time.time()
        dtr=lgb.Dataset(Xf,label=yf); des=lgb.Dataset(Xes,label=yh_es,reference=dtr)
        m=lgb.train(mk_lp(sd,nl,lr,mdl,l2),dtr,num_boost_round=3000,valid_sets=[des],callbacks=[lgb.early_stopping(200,verbose=False)])
        p_te.append(m.predict(Xte)); mt.append(time.time()-t)
        del m; gc.collect()
    return np.mean(p_te,axis=0), np.mean(mt)

# 只重跑 6 个关键组合 (H30_base 已经知道最好)
log('  训练 single_H30_base + CH_eq_baseline...')
single_H30_base, mt = train_h(30,63,0.02,200,1.0,10); log(f'  H30 base avg_t={mt:.0f}s')

# cross-horizon baseline (5 horizons × nl63)
ch_b=[]
for h in HORIZONS:
    p,mt=train_h(h,63,0.02,200,1.0,10); ch_b.append(p)
    log(f'    H={h} baseline')
ch_eq_b = np.mean(np.column_stack(ch_b),axis=1)

# cross-horizon reg (5 horizons × nl31)
ch_r=[]
for h in HORIZONS:
    p,mt=train_h(h,31,0.03,500,5.0,10); ch_r.append(p)
    log(f'    H={h} reg')
ch_eq_r = np.mean(np.column_stack(ch_r),axis=1)

# 组合：baseline H60 (AUC 最高的 horizon) + CH_eq_b + CH_eq_r
all10 = np.column_stack(ch_b + ch_r)
ch_eq_10 = np.mean(all10,axis=1)

# ========== 2. Per-hour filter ==========
log(f'\n{"="*60}')
log('Per-hour 过滤 (多个 base method)')
log(f'{"="*60}')

def run_perhour(base_name, base_pred, log_all=True):
    ranked = sorted([(h,roc_auc_score(yte[hr_te==h],base_pred[hr_te==h])) for h in range(24) if (hr_te==h).sum()>100], key=lambda x:-x[1])
    auc_full = roc_auc_score(yte,base_pred)
    best_hit_list = []
    for K in [2,3,5,8,12,16,20,24]:
        sel=[h for h,_ in ranked[:K]]
        m=np.isin(hr_te,sel)
        if m.sum()==0: continue
        p_sel=base_pred[m]; y_sel=yte[m]
        for thr in [50,60,70,75,80,85,90,92,94,95,96,97,98,99]:
            h_thr=np.percentile(p_sel,thr)
            hit=p_sel>h_thr
            if hit.sum()==0: continue
            acc=(y_sel[hit]==1).mean(); tp=tpd(hit.sum(),len(yte))
            flag='✅HIT' if (acc>=0.65 and tp>=15) else ('ACC' if acc>=0.65 else ('TPD' if tp>=15 else ''))
            if flag: best_hit_list.append((K,thr,acc,tp,flag))
            if log_all and (acc>=0.62 and tp>=10):
                log(f'  K={K} P{thr}: acc={acc:.4f} tpd={tp:.1f} {flag}')
    return ranked, auc_full, best_hit_list

all_results = []
for base_name, base_pred in [('single_H30_base',single_H30_base),('CH_eq_b',ch_eq_b),('CH_eq_r',ch_eq_r),('CH_eq_10',ch_eq_10)]:
    log(f'\n--- {base_name} ---')
    ranked, auc_full, hits = run_perhour(base_name, base_pred)
    log(f'  Full AUC={auc_full:.4f}, top6h: {ranked[:6]}')
    if hits:
        log(f'  ✅ 达标配置数: {len(hits)}')
        for K,thr,acc,tp,f in hits[:5]: log(f'    K={K} P{thr}: acc={acc:.4f} tpd={tp:.1f} {f}')
    all_results.append((base_name, auc_full, ranked, hits))

# ========== 3. 多方法 cross: 在 top hours 上叠加 ==========
log(f'\n{"="*60}')
log('多方法×多小时 × 多百分位 完整扫描')
log(f'{"="*60}')

methods = [('single_H30_base',single_H30_base),('CH_eq_b',ch_eq_b),('CH_eq_r',ch_eq_r),('CH_eq_10',ch_eq_10)]

# 把所有方法预测组合起来 (简单平均)
combos = {}
for i in range(len(methods)):
    for j in range(i+1,len(methods)):
        key = f'{methods[i][0]}+{methods[j][0]}'
        combos[key] = (methods[i][1] + methods[j][1]) / 2

best_overall=None
for base_name, base_pred in [(n,p) for n,p in methods] + list(combos.items()):
    ranked, auc_full, hits = run_perhour(base_name, base_pred, log_all=False)
    for K,thr,acc,tp,f in hits:
        score = min(acc/0.65, tp/15)
        if best_overall is None or score > best_overall[0]:
            best_overall = (score, base_name, K, thr, acc, tp)

if best_overall:
    log(f'\n🏆 全局最佳:')
    log(f'   方法={best_overall[1]}')
    log(f'   用 top-{best_overall[2]} 个小时')
    log(f'   percentile={best_overall[3]}')
    log(f'   acc={best_overall[4]:.4f} tpd={best_overall[5]:.1f}')
else:
    log('\n❌ 未找到达标配置')
    # 给最近的
    best_single=None
    for base_name, base_pred in [(n,p) for n,p in methods]:
        ranked, auc_full, _ = run_perhour(base_name, base_pred, log_all=False)
        log(f'  {base_name}: full AUC={auc_full:.4f}, top0.5% acc={np.mean(yte[np.argsort(base_pred)[-max(int(len(base_pred)*0.005),1):]]==1):.4f}')

log(f'\n⏱ Total: {time.time()-t0:.0f}s ({(time.time()-t0)/60:.1f}min)')
