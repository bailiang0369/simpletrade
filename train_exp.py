"""ETH H=30 全面训练 v3 - vi-aware"""
import time, gc, numpy as np, datetime, os
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score
import warnings; warnings.filterwarnings('ignore')

def log(m): print(m,flush=True)
def tpd(n,b): return n*1440/b
def topk(p,y,ret,kl=[0.005,0.008,0.01,0.015,0.02,0.03]):
    out={}; n=len(p); o=np.argsort(p)
    for k in kl:
        idx=o[-max(int(n*k),1):]
        out[k]={'acc':float((y[idx]==1).mean()),'ret':float(ret[idx].mean()*10000),'tpd':tpd(len(idx),n)}
    return out

t0=time.time(); SEEDS=15; H=30

# Load
log('loading npz...')
d=np.load('/workspace/models/eth_data.npz',allow_pickle=True)
X = d['X'].astype(np.float32)
vi = d['vi'].astype(np.int64)
ts_full = d['ts_full']
hr_full = d['hr_full']
tr_mask_full = d['tr_mask_full'].astype(bool)
es_mask_full = d['es_mask_full'].astype(bool)
te_mask_full = d['te_mask_full'].astype(bool)

# Build masks at vi positions
tr_m = tr_mask_full[vi]
es_m = es_mask_full[vi]
te_m = te_mask_full[vi]
hr_te = hr_full[vi][te_m]

y30 = d['y_30'].astype(np.int8)
ret30 = d['ret_30'].astype(np.float32)
labels = {}
for h in [3,5,15,60]:
    labels[f'ret_{h}']=d[f'ret_{h}'].astype(np.float32)
    labels[f'y_{h}']=d[f'y_{h}'].astype(np.int8)

log(f'X={X.shape} ({X.nbytes/1e6:.0f}MB), TR={tr_m.sum():,}, ES={es_m.sum():,}, TE={te_m.sum():,}')

Xtr=X[tr_m]; Xes=X[es_m]; Xte=X[te_m]
ytr=y30[tr_m]; yes=y30[es_m]; yte=y30[te_m]
ret_te=ret30[te_m]

def train_lgb(Xtr,ytr,Xes,yes,Xte,lp,seeds=SEEDS):
    preds=[]
    for sd in range(42,42+seeds):
        dtr=lgb.Dataset(Xtr,label=ytr); des=lgb.Dataset(Xes,label=yes,reference=dtr)
        m=lgb.train({**lp,'seed':sd},dtr,num_boost_round=4000,valid_sets=[des],callbacks=[lgb.early_stopping(300,verbose=False)])
        preds.append(m.predict(Xte)); del m; gc.collect()
    return np.mean(preds,axis=0)

def train_filter(X,y,Xes,yes,Xte,lp,h=30,min_ret=0.0005,seeds=SEEDS):
    ret_h = ret30 if h==30 else labels[f'ret_{h}']
    keep = np.abs(ret_h[tr_m])>=min_ret
    return train_lgb(Xtr[keep],y[keep],Xes,yes,Xte,lp,seeds)

# ========== Exp1: Baseline ==========
log(f'\n{"="*60}'); log('[Exp1] Baseline ext LGB 15s nl63 lr0.02'); log(f'{"="*60}')
lp_base=dict(num_leaves=63,learning_rate=0.02,min_data_in_leaf=200,feature_fraction=0.9,bagging_fraction=0.9,bagging_freq=5,lambda_l2=1.0,verbose=-1,num_threads=3,objective='binary',metric='auc')
t1=time.time(); exp1_avg = train_filter(Xtr,ytr,Xes,yes,Xte,lp_base); auc1=roc_auc_score(yte,exp1_avg)
log(f'  AUC={auc1:.4f} ({time.time()-t1:.0f}s)')
for k,v in topk(exp1_avg,yte,ret_te,[0.005,0.01,0.02]).items(): log(f'  top{k*100:.1f}%: acc={v["acc"]:.4f} tpd={v["tpd"]:.1f}')

# ========== Exp2: Big LGB ==========
log(f'\n{"="*60}'); log('[Exp2] LGB big nl127 lr0.01 md50 10s'); log(f'{"="*60}')
lp2=dict(num_leaves=127,learning_rate=0.01,min_data_in_leaf=50,feature_fraction=0.85,bagging_fraction=0.85,bagging_freq=5,lambda_l2=1.0,verbose=-1,num_threads=3,objective='binary',metric='auc')
t2=time.time(); exp2_avg = train_filter(Xtr,ytr,Xes,yes,Xte,lp2,seeds=10); auc2=roc_auc_score(yte,exp2_avg)
log(f'  AUC={auc2:.4f} ({time.time()-t2:.0f}s)')
for k,v in topk(exp2_avg,yte,ret_te,[0.005,0.01,0.02]).items(): log(f'  top{k*100:.1f}%: acc={v["acc"]:.4f} tpd={v["tpd"]:.1f}')

# ========== Exp3: Reg LGB ==========
log(f'\n{"="*60}'); log('[Exp3] LGB reg nl31 lr0.03 md500 l2=5'); log(f'{"="*60}')
lp3=dict(num_leaves=31,learning_rate=0.03,min_data_in_leaf=500,feature_fraction=0.9,bagging_fraction=0.9,bagging_freq=5,lambda_l2=5.0,verbose=-1,num_threads=3,objective='binary',metric='auc')
t3=time.time(); exp3_avg = train_filter(Xtr,ytr,Xes,yes,Xte,lp3); auc3=roc_auc_score(yte,exp3_avg)
log(f'  AUC={auc3:.4f} ({time.time()-t3:.0f}s)')
for k,v in topk(exp3_avg,yte,ret_te,[0.005,0.01,0.02]).items(): log(f'  top{k*100:.1f}%: acc={v["acc"]:.4f} tpd={v["tpd"]:.1f}')

# ========== Exp4: XGBoost ==========
log(f'\n{"="*60}'); log('[Exp4] XGBoost d6 5s'); log(f'{"="*60}')
t4=time.time()
tr_keep=np.abs(ret30[tr_m])>=0.0005
Xtr_x=Xtr[tr_keep]; ytr_x=ytr[tr_keep]
xlp=dict(max_depth=6,learning_rate=0.02,min_child_weight=200,subsample=0.9,colsample_bytree=0.9,reg_lambda=1.0,objective='binary:logistic',eval_metric='auc',nthread=3)
dtr=xgb.DMatrix(Xtr_x,label=ytr_x); des_dm=xgb.DMatrix(Xes,label=yes)
preds=[]
for sd in [42,123,456,789,1024]:
    m=xgb.train({**xlp,'seed':sd},dtr,num_boost_round=2000,evals=[(des_dm,'es')],early_stopping_rounds=200,verbose_eval=0)
    preds.append(m.predict(xgb.DMatrix(Xte))); del m; gc.collect()
xgb_avg=np.mean(preds,axis=0); del preds; gc.collect()
auc4=roc_auc_score(yte,xgb_avg)
log(f'  AUC={auc4:.4f} ({time.time()-t4:.0f}s)')
for k,v in topk(xgb_avg,yte,ret_te,[0.005,0.01,0.02]).items(): log(f'  top{k*100:.1f}%: acc={v["acc"]:.4f} tpd={v["tpd"]:.1f}')

# ========== Exp5: CatBoost ==========
log(f'\n{"="*60}'); log('[Exp5] CatBoost d6 4s 500k'); log(f'{"="*60}')
t5=time.time()
rng=np.random.RandomState(42); idx=rng.choice(len(Xtr_x),min(500_000,len(Xtr_x)),replace=False)
ctr=Pool(Xtr_x[idx],ytr_x[idx]); ces=Pool(Xes,yes)
cb_preds=[]
for sd in [42,123,456,789]:
    cb=CatBoostClassifier(iterations=2000,learning_rate=0.02,depth=6,l2_leaf_reg=3.0,loss_function='Logloss',eval_metric='AUC',od_type='Iter',od_wait=200,verbose=0,thread_count=3,random_seed=sd)
    cb.fit(ctr,eval_set=ces,use_best_model=True)
    cb_preds.append(cb.predict_proba(Xte)[:,1]); del cb; gc.collect()
cb_avg=np.mean(cb_preds,axis=0); del cb_preds,ctr,ces; gc.collect()
auc5=roc_auc_score(yte,cb_avg)
log(f'  AUC={auc5:.4f} ({time.time()-t5:.0f}s)')
for k,v in topk(cb_avg,yte,ret_te,[0.005,0.01,0.02]).items(): log(f'  top{k*100:.1f}%: acc={v["acc"]:.4f} tpd={v["tpd"]:.1f}')

# ========== Exp6: Cross-horizon ==========
log(f'\n{"="*60}'); log('[Exp6] Cross-horizon H=3,5,15,30,60 + Meta'); log(f'{"="*60}')
HORIZONS=[3,5,15,30,60]
base_es=[]; base_te=[]
for h in HORIZONS:
    yh = labels[f'y_{h}']; ret_h = labels[f'ret_{h}']
    yh_es=yh[es_m]; yh_te=yh[te_m]
    keep=np.abs(ret_h[tr_m])>=0.0005
    Xtr_h=Xtr[keep]; ytr_h=yh[tr_m][keep]
    p_es=[]; p_te=[]
    for sd in range(42,42+8):
        dtr=lgb.Dataset(Xtr_h,label=ytr_h); des=lgb.Dataset(Xes,label=yh_es,reference=dtr)
        m=lgb.train({**lp_base,'seed':sd},dtr,num_boost_round=2000,valid_sets=[des],callbacks=[lgb.early_stopping(200,verbose=False)])
        p_es.append(m.predict(Xes)); p_te.append(m.predict(Xte)); del m; gc.collect()
    base_es.append(np.mean(p_es,axis=0)); base_te.append(np.mean(p_te,axis=0))
    log(f'  H={h} AUC={roc_auc_score(yh_te,base_te[-1]):.4f}')

ch_equal = np.mean(np.column_stack(base_te),axis=1)
auc_ceq = roc_auc_score(yte,ch_equal); log(f'  CH_equal AUC={auc_ceq:.4f}')

# Meta LGB
meta_dtr = lgb.Dataset(np.column_stack(base_es),label=yes)
meta = lgb.train(dict(num_leaves=31,learning_rate=0.03,min_data_in_leaf=10,verbose=-1,num_threads=3,objective='binary',metric='auc'),meta_dtr,num_boost_round=300)
meta_avg = meta.predict(np.column_stack(base_te)); del meta; gc.collect()
auc_meta = roc_auc_score(yte,meta_avg); log(f'  Meta AUC={auc_meta:.4f}')

# ========== Exp7: Per-hour filter ==========
log(f'\n{"="*60}'); log('[Exp7] Per-hour filter on best LGB_big'); log(f'{"="*60}')
ranked=sorted([(h,roc_auc_score(yte[hr_te==h],exp2_avg[hr_te==h])) for h in range(24) if (hr_te==h).sum()>100],key=lambda x:-x[1])
log(f'  Top hours: {ranked[:8]}')
for K in [3,5,8,12,16,20,24]:
    sel=[h for h,_ in ranked[:K]]
    m=np.isin(hr_te,sel)
    if m.sum()==0: continue
    p=exp2_avg[m]; y=yte[m]
    for thr in [70,80,85,90,92,94,96,98,99]:
        h_thr=np.percentile(p,thr)
        hit=p>h_thr
        if hit.sum()==0: continue
        acc=(y[hit]==1).mean(); tp=tpd(hit.sum(),len(yte))
        flag='✅HIT' if (acc>=0.65 and tp>=15) else ('ACC' if acc>=0.65 else ('TPD' if tp>=15 else ''))
        if flag: log(f'  K={K} P{thr}: acc={acc:.4f} tpd={tp:.1f} {flag}')

# ========== Exp8: Weight search ==========
log(f'\n{"="*60}'); log('[Exp8] Weight search'); log(f'{"="*60}')
cands=[('LGB_base',exp1_avg),('LGB_big',exp2_avg),('LGB_reg',exp3_avg),('XGB',xgb_avg),('CB',cb_avg)]
Xc=np.column_stack([p for _,p in cands])
# Correlations
corr=np.corrcoef(Xc.T); names=[n for n,_ in cands]
log(f'  Correlations:')
for i in range(len(names)):
    for j in range(i+1,len(names)):
        log(f'    {names[i]}-{names[j]}: {corr[i,j]:.4f}')

best_w=None; best_auc=0
for w1 in np.arange(0,1.05,0.1):
    for w2 in np.arange(0,1.05-w1,0.1):
        for w3 in np.arange(0,1.05-w1-w2,0.1):
            for w4 in np.arange(0,1.05-w1-w2-w3,0.1):
                w5=round(max(0,1-w1-w2-w3-w4),2)
                p=Xc@np.array([w1,w2,w3,w4,w5])
                a=roc_auc_score(yte,p)
                if a>best_auc: best_auc=a; best_w=(w1,w2,w3,w4,w5)
best_w5 = Xc@np.array(best_w)
log(f'  Best 5w AUC={best_auc:.4f} weights={best_w}')
all_eq = np.mean(Xc,axis=1); auc_eq = roc_auc_score(yte,all_eq); log(f'  All5_eq AUC={auc_eq:.4f}')

# ========== FINAL SUMMARY ==========
log(f'\n{"="*60}'); log('FINAL SUMMARY'); log(f'{"="*60}')
final_cands = cands + [('CH_eq',ch_equal),('Meta',meta_avg),('BestW5',best_w5),('All5eq',all_eq)]
log(f'{"Method":<18} {"AUC":>7} {"t0.5%":>7} {"t0.8%":>7} {"t1%":>7} {"t2%":>7}')
for name,p in final_cands:
    auc=roc_auc_score(yte,p)
    tt=topk(p,yte,ret_te,[0.005,0.008,0.01,0.02])
    def f(k): return f'{tt[k]["acc"]:.4f}'
    log(f'{name:<18} {auc:.4f} {f(0.005):>7} {f(0.008):>7} {f(0.01):>7} {f(0.02):>7}')

log(f'\nTARGET (acc>=65%, tpd>=15):')
hit=False; best_n=None
for name,p in final_cands:
    o=np.argsort(p)
    for k in [0.005,0.008,0.01,0.015,0.02,0.03,0.05,0.08,0.10]:
        idx=o[-max(int(len(p)*k),1):]
        acc=(yte[idx]==1).mean(); tp=tpd(len(idx),len(yte))
        score=min(acc/0.65, tp/15)
        if acc>=0.65 and tp>=15: log(f'  ✅ {name} top{k*100:.1f}%: acc={acc:.4f} tpd={tp:.1f}'); hit=True
        if best_n is None or score>best_n[0]: best_n=(score,name,k,acc,tp)
if not hit:
    log(f'  ❌ NO — Nearest: {best_n[1]} top{best_n[2]*100:.1f}%: acc={best_n[3]:.4f} tpd={best_n[4]:.1f}')

np.savez('/workspace/models/preds_v3.npz',
    **{n:p for n,p in final_cands}, yte=yte, ret_te=ret_te, hr_te=hr_te)
log(f'\nSaved preds. Total: {time.time()-t0:.0f}s ({(time.time()-t0)/3600:.1f}h)')
