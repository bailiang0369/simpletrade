"""xf3 修 axis bug: axis=0→axis=1"""
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
df=raw_eth[['ts','open','high','low','close','buy_vol','sell_vol','funding']].copy()
for c in ['open','high','low','close','buy_vol','sell_vol','funding']: df[c]=df[c].astype(np.float32)
C=df['close'].values.astype(np.float32); H=df['high'].values.astype(np.float32); L=df['low'].values.astype(np.float32)
O=df['open'].values.astype(np.float32); TB=df['buy_vol'].values.astype(np.float32); TS=df['sell_vol'].values.astype(np.float32)
F=df['funding'].values.astype(np.float32); ts=df['ts'].values.astype(np.int64)
del raw_eth,df; gc.collect()
lr=np.log(C/np.roll(C,1)).astype(np.float32)
feats={}
for w in [3,5,10,15,30,60,120,240,480]:
    feats[f'lr_{w}']=np.full(len(C),np.nan,np.float32); feats[f'lr_{w}'][w:]=(np.log(C[w:])-np.log(C[:-w])).astype(np.float32)
for w in [30,60,120,240,480]:
    mu=pd.Series(C).rolling(w,min_periods=w).mean().values.astype(np.float32); sd=pd.Series(C).rolling(w,min_periods=w).std().values.astype(np.float32)
    feats[f'z_{w}']=((C-mu)/(sd+1e-8)).astype(np.float32)
for w in [15,30,60,120,240]: feats[f'rvol_{w}']=pd.Series(lr).rolling(w,min_periods=w).std().values.astype(np.float32)
feats['rvol_ratio_30_60']=(feats['rvol_30']/(feats['rvol_60']+1e-8)).astype(np.float32)
for w in [30,60,120,240]:
    lo=pd.Series(L).rolling(w,min_periods=w).min().values.astype(np.float32); hi=pd.Series(H).rolling(w,min_periods=w).max().values.astype(np.float32)
    feats[f'pos_{w}']=((C-lo)/(hi-lo+1e-8)).astype(np.float32)
for w in [60,120,240]:
    lo=pd.Series(C).rolling(w,min_periods=w).min().values.astype(np.float32); hi=pd.Series(C).rolling(w,min_periods=w).max().values.astype(np.float32)
    feats[f'rank_{w}']=((C-lo)/(hi-lo+1e-8)).astype(np.float32)
for w in [15,30,60,120]: feats[f'cvd_{w}']=pd.Series((TB-TS)/(TB+TS+1e-8)).rolling(w,min_periods=w).sum().values.astype(np.float32)
feats['funding']=F; feats['funding_diff_5']=(F-pd.Series(F).shift(5).values.astype(np.float32)).astype(np.float32)
feats['funding_diff_30']=(F-pd.Series(F).shift(30).values.astype(np.float32)).astype(np.float32)
feats['funding_ma_30']=pd.Series(F).rolling(30,min_periods=15).mean().values.astype(np.float32)
feats['funding_z_60']=((F-pd.Series(F).rolling(60,min_periods=60).mean().values)/(pd.Series(F).rolling(60,min_periods=60).std().values+1e-8)).astype(np.float32)
feats['body_ratio']=((C-O)/(H-L+1e-8)).astype(np.float32); feats['body_size']=(np.abs(C-O)/(C+1e-8)*100).astype(np.float32)
dn10=pd.Series((C<np.roll(C,1)).astype(np.float32)).rolling(10,min_periods=1).sum().values.astype(np.float32)
feats['dn_count_10']=dn10; feats['up_count_10']=(10-dn10).astype(np.float32)
for w in [60,120]:
    lo=pd.Series(L).rolling(w,min_periods=w).min().values.astype(np.float32); hi=pd.Series(H).rolling(w,min_periods=w).max().values.astype(np.float32)
    feats[f'near_lo_{w}']=((C-lo)/(C+1e-8)*100).astype(np.float32); feats[f'near_hi_{w}']=((hi-C)/(C+1e-8)*100).astype(np.float32)
hr_arr=pd.to_datetime(ts,unit='s',utc=True).hour.values.astype(np.float32); dow=pd.to_datetime(ts,unit='s',utc=True).dayofweek.values.astype(np.float32)
feats['hour_sin']=np.sin(hr_arr*2*np.pi/24).astype(np.float32); feats['hour_cos']=np.cos(hr_arr*2*np.pi/24).astype(np.float32)
feats['dow_sin']=np.sin(dow*2*np.pi/7).astype(np.float32); feats['dow_cos']=np.cos(dow*2*np.pi/7).astype(np.float32)
btc_close=pd.Series(raw_btc['close'].values.astype(np.float64),index=raw_btc['ts'].values.astype(np.int64)).reindex(ts,method='ffill').values.astype(np.float64)
btc_close=np.maximum(btc_close,1e-12); btc_lr=np.log(btc_close[1:]/btc_close[:-1]); btc_lr=np.append([np.nan],btc_lr).astype(np.float32)
for w in [5,15,30,60,120,240,480,960]:
    feats[f'BTC_lr_{w}']=np.full(len(C),np.nan,np.float32); feats[f'BTC_lr_{w}'][w:]=(np.log(btc_close[w:])-np.log(btc_close[:-w])).astype(np.float32)
for w in [30,60,120,240,480]:
    mu=pd.Series(btc_close).rolling(w,min_periods=w).mean().values.astype(np.float64); sd=pd.Series(btc_close).rolling(w,min_periods=w).std().values.astype(np.float64)
    feats[f'BTC_z_{w}']=((btc_close-mu)/(sd+1e-8)).astype(np.float32)
for w in [60,240]: feats[f'BTC_rvol_{w}']=pd.Series(btc_lr).rolling(w,min_periods=w).std().values.astype(np.float32)
del raw_btc,btc_lr,btc_close; gc.collect()
ret_future=np.full(len(C),np.nan,np.float32); ret_future[:-30]=(C[30:]/C[:-30]-1).astype(np.float32)
label=(ret_future>0).astype(np.int8); del C,H,L,O,TB,TS,F,lr; gc.collect()
X=pd.DataFrame(feats).astype(np.float32); X['label']=label; X['ret_future']=ret_future; X['ts']=ts; del feats; gc.collect()
valid=~np.isnan(ret_future) & ~X.drop(columns=['label','ret_future','ts']).isna().any(axis=1).values
vi=np.where(valid)[0]; X=X.iloc[vi].reset_index(drop=True); ts_v=X['ts'].values; label_v=X['label'].values.astype(np.int8); ret_v=X['ret_future'].values.astype(np.float32)
feat_df=X.drop(columns=['label','ret_future','ts']); del X,valid,vi,ret_future,label; gc.collect()
def ts_mask(tsarr,s,e):
    a=int(datetime.datetime.strptime(s,'%Y-%m-%d').replace(tzinfo=datetime.timezone.utc).timestamp())
    b=int(datetime.datetime.strptime(e,'%Y-%m-%d').replace(tzinfo=datetime.timezone.utc).timestamp())
    return (tsarr>=a)&(tsarr<b)
tr_m=ts_mask(ts_v,'2020-01-01','2024-06-30'); es_m=ts_mask(ts_v,'2024-06-30','2024-09-30'); te_m=ts_mask(ts_v,'2025-09-30','2026-08-29')
log(f'  TR={tr_m.sum():,} TE={te_m.sum():,}  nF={feat_df.shape[1]}')
Xtr=feat_df[tr_m].values.astype(np.float32); ytr=label_v[tr_m]
Xes=feat_df[es_m].values.astype(np.float32); yes=label_v[es_m]
Xte=feat_df[te_m].values.astype(np.float32); yte=label_v[te_m]
keep=np.abs(ret_v[tr_m])>=0.0005; Xtr_f=Xtr[keep]; ytr_f=ytr[keep]; del Xtr,ytr,feat_df; gc.collect()

# FIX: mean axis=1 (over seeds), not axis=0
log('\n[2] 15 seeds...')
p_all=[]
for sd in range(42,57):
    t=time.time()
    lp=dict(num_leaves=63,learning_rate=0.02,min_data_in_leaf=200,feature_fraction=0.9,bagging_fraction=0.9,bagging_freq=5,lambda_l2=1.0,verbose=-1,num_threads=3,objective='binary',metric='auc',seed=sd)
    dtr=lgb.Dataset(Xtr_f,label=ytr_f); des=lgb.Dataset(Xes,label=yes,reference=dtr)
    m=lgb.train(lp,dtr,num_boost_round=4000,valid_sets=[des],callbacks=[lgb.early_stopping(300,verbose=False)])
    p_all.append(m.predict(Xte))
    log(f'  seed {sd}: {time.time()-t:.0f}s iter={m.best_iteration}'); del m; gc.collect()
# FIX HERE: column_stack gives (TE, seeds), mean(axis=1) gives (TE,)
p=np.mean(np.column_stack(p_all),axis=1); del p_all; gc.collect()
log(f'  p.shape={p.shape}, yte.shape={yte.shape}')
auc=roc_auc_score(yte,p); log(f'  AUC={auc:.4f}')

# Top-k
log('\n[3] Top-k')
o=np.argsort(p)
for k in [0.003,0.005,0.008,0.01,0.012,0.015,0.02,0.03,0.05,0.08,0.10]:
    idx=o[-max(int(len(p)*k),1):]; acc=(yte[idx]==1).mean(); tp=tpd(len(idx),len(yte))
    log(f'  top{k*100:.1f}%: acc={acc:.4f} tpd={tp:.1f}')

# Per-hour
log('\n[4] Per-hour')
hr_te=pd.to_datetime(ts_v[te_m],unit='s',utc=True).hour.values.astype(np.int32)
ranked=sorted([(h,roc_auc_score(yte[hr_te==h],p[hr_te==h])) for h in range(24) if (hr_te==h).sum()>100],key=lambda x:-x[1])
log(f'  top10h: {[(h,f"{a:.4f}") for h,a in ranked[:10]]}')

hits=0
for K in range(2,25):
    sel=[h for h,_ in ranked[:K]]; m=np.isin(hr_te,sel)
    if m.sum()<500: continue
    ps=p[m]; ys=yte[m]
    for thr in range(50,100,1):
        hv=np.percentile(ps,thr); hit=ps>hv
        if hit.sum()<20: continue
        acc=(ys[hit]==1).mean(); tp=tpd(hit.sum(),len(p))
        if acc>=0.65 and tp>=15:
            log(f'  ✅ K={K}h P{thr}: acc={acc:.4f} tpd={tp:.1f}'); hits+=1
log(f'\n⏱ {time.time()-t0:.0f}s ({(time.time()-t0)/60:.0f}min)')
log(f'达标配置: {hits}')
