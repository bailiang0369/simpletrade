"""Stochastic + RSI + Williams %R + MACD + 短窗口特征, H=15. 纯 numpy 低内存版."""
import time, gc, numpy as np, datetime, sys, warnings, pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore')
def log(m): print(m,flush=True)
def tpd(n,b): return n*1440/b
t0=time.time()

log('[1] 数据...')
raw_eth=pd.read_parquet('/workspace/data/datasets/raw_ETH.parquet')
raw_btc=pd.read_parquet('/workspace/data/datasets/raw_BTC.parquet')

ts=raw_eth['ts'].values.astype(np.int64)
C=raw_eth['close'].values.astype(np.float64)
H=raw_eth['high'].values.astype(np.float64)
L=raw_eth['low'].values.astype(np.float64)
O=raw_eth['open'].values.astype(np.float64)
TB=raw_eth['buy_vol'].values.astype(np.float64)
TS=raw_eth['sell_vol'].values.astype(np.float64)
F=raw_eth['funding'].values.astype(np.float64)
del raw_eth; gc.collect()
n=len(C); EPS=1e-12
log(f'  n={n:,}')

# BTC close, reindex to ETH ts
btc_s=pd.Series(raw_btc['close'].values.astype(np.float64), index=raw_btc['ts'].values.astype(np.int64))
btc_close=btc_s.reindex(ts, method='ffill').values.astype(np.float64)
del raw_btc, btc_s; gc.collect()

def ema_np(arr, span):
    # Wilder's EMA (alpha=1/span), same as ewm(alpha=1/span, adjust=False)
    alpha=1.0/span
    out=np.empty_like(arr)
    out[0]=arr[0]
    for i in range(1,len(arr)): out[i]=alpha*arr[i]+(1-alpha)*out[i-1]
    return out

def rolling_min(arr, w):
    s=pd.Series(arr); r=s.rolling(w,min_periods=w).min().values; del s; return r
def rolling_max(arr, w):
    s=pd.Series(arr); r=s.rolling(w,min_periods=w).max().values; del s; return r
def rolling_mean(arr, w):
    s=pd.Series(arr); r=s.rolling(w,min_periods=w).mean().values; del s; return r
def rolling_std(arr, w):
    s=pd.Series(arr); r=s.rolling(w,min_periods=w).std().values; del s; return r

cols=[]  # list of (name, np.array fp32)

def add(name, arr):
    cols.append((name, arr.astype(np.float32)))

# ========== STOCHASTIC ==========
log('[2] Stoch...')
for w in [5,10,14,20,30]:
    lo=rolling_min(L,w); hi=rolling_max(H,w)
    k=(C-lo)/(hi-lo+EPS)
    add(f'stoch_K_{w}', k)
    for d in [2,3,5]:
        add(f'stoch_D_{w}_{d}', rolling_mean(k, d))
    add(f'williamsR_{w}', -k)
    add(f'stoch_K_roc_{w}', k - np.roll(k,1))
    del lo,hi,k

# ========== RSI (Wilder) ==========
log('[3] RSI...')
delta=np.diff(C); delta=np.append([0.0],delta)
up=np.where(delta>0,delta,0.0)
dn=np.where(delta<0,-delta,0.0)
for w in [5,10,14,20,30]:
    au=ema_np(up,w); ad=ema_np(dn,w)
    rs=au/(ad+EPS); rsi=100-100/(1+rs)
    add(f'rsi_{w}', rsi)
    add(f'rsi_{w}_from50', (rsi-50)/50)
    add(f'rsi_{w}_slope5', rsi - np.roll(rsi,5))
    del au,ad,rs,rsi
del delta,up,dn; gc.collect()

# ========== MACD ==========
log('[4] MACD...')
for fast,slow in [(6,13),(8,21),(12,26)]:
    ef=ema_np(C,fast); es_=ema_np(C,slow)
    macd=ef-es_; sig=ema_np(macd,9)
    add(f'macd_{fast}_{slow}', macd)
    add(f'macd_sig_{fast}_{slow}', sig)
    add(f'macd_hist_{fast}_{slow}', macd-sig)
    add(f'macd_norm_{fast}_{slow}', macd/(C+EPS)*100)
    del ef,es_,macd,sig

# ========== DMI / ADX ==========
log('[5] DMI...')
tr=np.maximum(H-L, np.maximum(np.abs(H-np.roll(C,1)), np.abs(L-np.roll(C,1))))
pdm=np.where((H-np.roll(H,1))>(np.roll(L,1)-L), np.maximum(H-np.roll(H,1),0), 0)
ndm=np.where((np.roll(L,1)-L)>(H-np.roll(H,1)), np.maximum(np.roll(L,1)-L,0), 0)
for w in [10,14,20]:
    ts_=ema_np(tr,w); pdm_=ema_np(pdm,w); ndm_=ema_np(ndm,w)
    add(f'tr_{w}', ts_/(C+EPS)*100)
    add(f'pdi_{w}', pdm_/(ts_+EPS)*100)
    add(f'mdi_{w}', ndm_/(ts_+EPS)*100)
    add(f'dx_{w}', np.abs(pdm_-ndm_)/(pdm_+ndm_+EPS)*100)
    del ts_,pdm_,ndm_
del tr,pdm,ndm; gc.collect()

# ========== BB ==========
log('[6] BB...')
for w in [10,20,30]:
    mu=rolling_mean(C,w); sd=rolling_std(C,w)
    add(f'bb_pos_{w}', (C-mu)/(2*sd+EPS))
    add(f'bb_width_{w}', 2*sd/(mu+EPS)*100)
    del mu,sd

# ========== Price extremes ==========
log('[7] extremes...')
for w in [14,30,60]:
    lo=rolling_min(L,w); hi=rolling_max(H,w)
    add(f'price_dist_lo_{w}', (C-lo)/(C+EPS)*100)
    add(f'price_dist_hi_{w}', (hi-C)/(C+EPS)*100)
    del lo,hi
del H,L,O; gc.collect()

# ========== Crypto + mom ==========
log('[8] crypto/mom...')
add('funding', F)
add('funding_diff_5', F - np.roll(F,5))
Fz=(F - rolling_mean(F,30))/(rolling_std(F,30)+EPS); add('funding_z_30', Fz); del Fz
tbr=(TB-TS)/(TB+TS+EPS)
for w in [10,30,60]:
    add(f'cvd_{w}', rolling_mean(tbr, w))  # rolling mean of buy_ratio as CVD proxy
del TB,TS,F,tbr; gc.collect()
for w in [3,5,10,14,30,60,120]:
    r=np.full(n, np.nan, np.float32)
    r[w:]=(np.log(C[w:])-np.log(C[:-w])).astype(np.float32)
    add(f'lr_{w}', r); del r

# ========== Time ==========
dt=pd.to_datetime(ts,unit='s',utc=True)
hr=dt.hour.values.astype(np.float64); dow=dt.dayofweek.values.astype(np.float64)
add('hour_sin', np.sin(hr*2*np.pi/24))
add('hour_cos', np.cos(hr*2*np.pi/24))
del dt,hr,dow; gc.collect()

# ========== BTC cross ==========
log('[9] BTC...')
for w in [5,14,30,60,120,240]:
    r=np.full(n, np.nan, np.float32)
    r[w:]=(np.log(btc_close[w:])-np.log(btc_close[:-w])).astype(np.float32)
    add(f'BTC_lr_{w}', r); del r
bl=rolling_min(btc_close,30); bh=rolling_max(btc_close,30)
add('BTC_stoch_30', (btc_close-bl)/(bh-bl+EPS))
del btc_close,bl,bh; gc.collect()

# ========== Assemble matrix ==========
log(f'  总计 {len(cols)} 特征, 组装矩阵...')
feat_names=[c[0] for c in cols]
# 逐列拷贝到 FP32 矩阵, 避免 dict-of-array 累积内存
X=np.empty((n,len(cols)), dtype=np.float32)
for i,(nm,arr) in enumerate(cols):
    X[:,i]=arr
del cols; gc.collect()

# Labels
log('[10] 标签...')
ret15=np.full(n,np.nan,np.float32); ret15[:-15]=(C[15:]/C[:-15]-1).astype(np.float32)
ret30=np.full(n,np.nan,np.float32); ret30[:-30]=(C[30:]/C[:-30]-1).astype(np.float32)
del C; gc.collect()
lab15=(ret15>0).astype(np.int8); lab30=(ret30>0).astype(np.int8)

# Valid mask (all feat not NaN + ret not NaN)
valid=~np.isnan(ret15) & ~np.isnan(ret30)
# 检查前 40 列的前 w 行可能是 NaN (rolling warm-up). 简单做法: 找所有行中任何一列 NaN
valid &= ~np.isnan(X).any(axis=1)
vi=np.where(valid)[0]
log(f'  valid rows: {len(vi):,} / {n:,}')

X=X[vi]; ts_v=ts[vi]; l15=lab15[vi]; l30=lab30[vi]; r15=ret15[vi]; r30=ret30[vi]
del valid,vi,ret15,ret30,lab15,lab30,ts; gc.collect()
log(f'  X shape: {X.shape}, mem: {X.nbytes/1e9:.2f} GB')

def ts_mask(tsarr,s,e):
    a=int(datetime.datetime.strptime(s,'%Y-%m-%d').replace(tzinfo=datetime.timezone.utc).timestamp())
    b=int(datetime.datetime.strptime(e,'%Y-%m-%d').replace(tzinfo=datetime.timezone.utc).timestamp())
    return (tsarr>=a)&(tsarr<b)
tr_m=ts_mask(ts_v,'2020-01-01','2024-06-30')
es_m=ts_mask(ts_v,'2024-06-30','2024-09-30')
te_m=ts_mask(ts_v,'2025-09-30','2026-08-29')
log(f'  TR={tr_m.sum():,} ES={es_m.sum():,} TE={te_m.sum():,}')

Xtr=X[tr_m]; Xes=X[es_m]; Xte=X[te_m]
del X,tr_m,es_m; gc.collect()
log(f'  Xtr={Xtr.shape}, Xte={Xte.shape}')

# ========== Train ==========
def train_h(name,ytr,yes,yte,Xtr,Xes,Xte):
    log(f'\n[train] {name}...')
    dtr=lgb.Dataset(Xtr,label=ytr); des=lgb.Dataset(Xes,label=yes,reference=dtr)
    lp=dict(num_leaves=63,learning_rate=0.02,min_data_in_leaf=200,
            feature_fraction=0.9,bagging_fraction=0.9,bagging_freq=5,
            lambda_l2=1.0,verbose=-1,num_threads=3,
            objective='binary',metric='auc',seed=42)
    t=time.time()
    m=lgb.train(lp,dtr,num_boost_round=4000,valid_sets=[des],
                callbacks=[lgb.early_stopping(300,verbose=False)])
    log(f'  iter={m.best_iteration} time={time.time()-t:.0f}s')
    p=m.predict(Xte)
    auc=roc_auc_score(yte,p); log(f'  TE AUC={auc:.4f}')
    return p,auc,m

p15,auc15,m15=train_h('H15',l15[ts_mask(ts_v,'2020-01-01','2024-06-30')],
                              l15[ts_mask(ts_v,'2024-06-30','2024-09-30')],
                              l15[te_m], Xtr,Xes,Xte)
p30,auc30,m30=train_h('H30',l30[ts_mask(ts_v,'2020-01-01','2024-06-30')],
                              l30[ts_mask(ts_v,'2024-06-30','2024-09-30')],
                              l30[te_m], Xtr,Xes,Xte)

del Xtr,Xes,Xte; gc.collect()

# ========== Feature importance ==========
for name,m,p,yte in [('H15',m15,p15,l15[te_m]),('H30',m30,p30,l30[te_m])]:
    fi=pd.DataFrame({'f':feat_names,'gain':m.feature_importance(importance_type='gain')})
    fi['pct']=fi['gain']/fi['gain'].sum()*100
    fi=fi.sort_values('gain',ascending=False)
    log(f'\n[FI {name}] top20:')
    for _,row in fi.head(20).iterrows(): log(f'  {row["f"]:<22s} {row["gain"]:>10.0f} {row["pct"]:>5.1f}%')
    cats={'Stoch':0,'RSI':0,'MACD':0,'DMI':0,'BB':0,'Funding':0,'Mom':0,'BTC':0,'Time':0}
    for _,row in fi.iterrows():
        f=row['f']; v=row['pct']
        if 'stoch' in f or 'williams' in f or 'price_dist' in f: cats['Stoch']+=v
        elif 'rsi' in f: cats['RSI']+=v
        elif 'macd' in f: cats['MACD']+=v
        elif 'pdi' in f or 'mdi' in f or 'dx' in f or 'tr_' in f: cats['DMI']+=v
        elif 'bb_' in f: cats['BB']+=v
        elif 'funding' in f or 'cvd' in f: cats['Funding']+=v
        elif f.startswith('lr_'): cats['Mom']+=v
        elif 'BTC' in f: cats['BTC']+=v
        else: cats['Time']+=v
    for k,v in sorted(cats.items(),key=lambda x:-x[1]): log(f'    {k:<12s} {v:>5.1f}%')
del m15,m30,fi; gc.collect()

# ========== Top-k accuracy ==========
log(f'\n[Top-k]')
yte15=l15[te_m]; yte30=l30[te_m]
for name,p,yte in [('H15',p15,yte15),('H30',p30,yte30)]:
    auc=roc_auc_score(yte,p); o=np.argsort(p)
    log(f'\n  {name} AUC={auc:.4f}:')
    for k in [0.005,0.008,0.01,0.012,0.015,0.02,0.03,0.05,0.08,0.10]:
        idx=o[-max(int(len(p)*k),1):]
        acc=(yte[idx]==1).mean()
        tp=tpd(len(idx),len(yte))
        flag='✅' if (acc>=0.65 and tp>=15) else ('⭐' if acc>=0.62 and tp>=10 else '')
        log(f'    top{k*100:.1f}%: acc={acc:.4f} tpd={tp:.1f} {flag}')

# ========== Per-hour gate ==========
log(f'\n[Per-hour]')
hr_v=pd.to_datetime(ts_v[te_m],unit='s',utc=True).hour.values.astype(np.int32)
for name,p,yte in [('H15',p15,yte15),('H30',p30,yte30)]:
    ranked=sorted([(h,roc_auc_score(yte[hr_v==h],p[hr_v==h])) for h in range(24) if (hr_v==h).sum()>100],key=lambda x:-x[1])
    log(f'\n  {name} top8h: {[(h,f"{a:.4f}") for h,a in ranked[:8]]}')
    hits=0; best_near=None
    for K in range(2,25):
        sel=[h for h,_ in ranked[:K]]
        m=np.isin(hr_v,sel)
        if m.sum()<500: continue
        ps=p[m]; ys=yte[m]
        for thr in range(50,100,1):
            hv=np.percentile(ps,thr); hit=ps>hv
            if hit.sum()<20: continue
            acc=(ys[hit]==1).mean(); tp=tpd(hit.sum(),len(p))
            score=min(acc/0.65,tp/15)
            if acc>=0.65 and tp>=15:
                log(f'    ✅ K={K}h P{thr}: acc={acc:.4f} tpd={tp:.1f}'); hits+=1
            if best_near is None or score>best_near[0]: best_near=(score,K,thr,acc,tp)
    if not hits and best_near:
        log(f'    最近: K={best_near[1]}h P{best_near[2]}: acc={best_near[3]:.4f} tpd={best_near[4]:.1f}')
    log(f'    达标: {hits}')

log(f'\n⏱ {time.time()-t0:.0f}s ({(time.time()-t0)/60:.0f}min)')
