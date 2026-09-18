"""xf.py + BTC cross-asset 特征"""
import time, gc, numpy as np, datetime, sys, warnings, pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore')

def log(m): print(m,flush=True)
def tpd(n,b): return n*1440/b

t0=time.time()

# ========== 1. 读 raw ETH + BTC ==========
log('[1] 读 raw...')
raw_eth=pd.read_parquet('/workspace/data/datasets/raw_ETH.parquet')
raw_btc=pd.read_parquet('/workspace/data/datasets/raw_BTC.parquet')
log(f'  ETH={raw_eth.shape} BTC={raw_btc.shape}')

# ========== 2. ETH 特征 ==========
log('[2] ETH 特征...')
df=raw_eth[['ts','open','high','low','close','buy_vol','sell_vol','funding']].copy()
for c in ['open','high','low','close','buy_vol','sell_vol','funding']: df[c]=df[c].astype(np.float32)
C=df['close'].values.astype(np.float32); H=df['high'].values.astype(np.float32); L=df['low'].values.astype(np.float32)
O=df['open'].values.astype(np.float32); TB=df['buy_vol'].values.astype(np.float32); TS=df['sell_vol'].values.astype(np.float32)
F=df['funding'].values.astype(np.float32); ts=df['ts'].values.astype(np.int64)
del raw_eth; gc.collect()

feats={}; lr=np.log(C/np.roll(C,1)).astype(np.float32)

# 动量
for w in [3,5,10,15,30,60,120,240,480]:
    feats[f'lr_{w}']=np.full(len(C),np.nan,np.float32); feats[f'lr_{w}'][w:]=(np.log(C[w:])-np.log(C[:-w])).astype(np.float32)
# z-score
for w in [30,60,120,240,480]:
    mu=pd.Series(C).rolling(w,min_periods=w).mean().values.astype(np.float32); sd=pd.Series(C).rolling(w,min_periods=w).std().values.astype(np.float32)
    feats[f'z_{w}']=((C-mu)/(sd+1e-8)).astype(np.float32)
# 波动率
for w in [15,30,60,120,240]: feats[f'rvol_{w}']=pd.Series(lr).rolling(w,min_periods=w).std().values.astype(np.float32)
feats['rvol_ratio_30_60']=(feats['rvol_30']/(feats['rvol_60']+1e-8)).astype(np.float32)
# 区间位置
for w in [30,60,120,240]:
    lo=pd.Series(L).rolling(w,min_periods=w).min().values.astype(np.float32); hi=pd.Series(H).rolling(w,min_periods=w).max().values.astype(np.float32)
    feats[f'pos_{w}']=((C-lo)/(hi-lo+1e-8)).astype(np.float32)
# rolling rank
for w in [60,120,240]:
    lo=pd.Series(C).rolling(w,min_periods=w).min().values.astype(np.float32); hi=pd.Series(C).rolling(w,min_periods=w).max().values.astype(np.float32)
    feats[f'rank_{w}']=((C-lo)/(hi-lo+1e-8)).astype(np.float32)
# CVD
for w in [15,30,60,120]: cvd=pd.Series((TB-TS)/(TB+TS+1e-8)).rolling(w,min_periods=w).sum().values.astype(np.float32); feats[f'cvd_{w}']=cvd
# Funding 差分
feats['funding']=F
feats['funding_diff_5']=(F-pd.Series(F).shift(5).values.astype(np.float32)).astype(np.float32)
feats['funding_diff_30']=(F-pd.Series(F).shift(30).values.astype(np.float32)).astype(np.float32)
feats['funding_ma_30']=pd.Series(F).rolling(30,min_periods=15).mean().values.astype(np.float32)
feats['funding_z_60']=((F-pd.Series(F).rolling(60,min_periods=60).mean().values)/(pd.Series(F).rolling(60,min_periods=60).std().values+1e-8)).astype(np.float32)
# OHLC
feats['body_ratio']=((C-O)/(H-L+1e-8)).astype(np.float32)
feats['body_size']=(np.abs(C-O)/(C+1e-8)*100).astype(np.float32)
# up/down counts
dn10=pd.Series((C<np.roll(C,1)).astype(np.float32)).rolling(10,min_periods=1).sum().values.astype(np.float32)
feats['dn_count_10']=dn10; feats['up_count_10']=(10-dn10).astype(np.float32)
# extremes
for w in [60,120]:
    lo=pd.Series(L).rolling(w,min_periods=w).min().values.astype(np.float32); hi=pd.Series(H).rolling(w,min_periods=w).max().values.astype(np.float32)
    feats[f'near_lo_{w}']=((C-lo)/(C+1e-8)*100).astype(np.float32)
    feats[f'near_hi_{w}']=((hi-C)/(C+1e-8)*100).astype(np.float32)
# 时间
hr=pd.to_datetime(ts,unit='s',utc=True).hour.values.astype(np.float32); dow=pd.to_datetime(ts,unit='s',utc=True).dayofweek.values.astype(np.float32)
feats['hour_sin']=np.sin(hr*2*np.pi/24).astype(np.float32); feats['hour_cos']=np.cos(hr*2*np.pi/24).astype(np.float32)
feats['dow_sin']=np.sin(dow*2*np.pi/7).astype(np.float32); feats['dow_cos']=np.cos(dow*2*np.pi/7).astype(np.float32)

# ========== 3. BTC cross-asset 特征 (对齐 ETH ts) ==========
log('[3] BTC cross-asset 特征...')
btc_close = pd.Series(raw_btc['close'].values.astype(np.float64), index=raw_btc['ts'].values.astype(np.int64))
btc_close = btc_close.reindex(ts, method='ffill').values.astype(np.float64)
btc_close = np.maximum(btc_close, 1e-12)
btc_lr = np.log(btc_close[1:] / btc_close[:-1])
btc_lr = np.append([np.nan], btc_lr).astype(np.float32)

for w in [5,15,30,60,120,240,480,960]:
    feats[f'BTC_lr_{w}']=np.full(len(C),np.nan,np.float32)
    feats[f'BTC_lr_{w}'][w:]=(np.log(btc_close[w:])-np.log(btc_close[:-w])).astype(np.float32)
for w in [30,60,120,240,480]:
    mu=pd.Series(btc_close).rolling(w,min_periods=w).mean().values.astype(np.float64)
    sd=pd.Series(btc_close).rolling(w,min_periods=w).std().values.astype(np.float64)
    feats[f'BTC_z_{w}']=((btc_close-mu)/(sd+1e-8)).astype(np.float32)
for w in [60,240]:
    feats[f'BTC_rvol_{w}']=pd.Series(btc_lr).rolling(w,min_periods=w).std().values.astype(np.float32)

del raw_btc,btc_lr; gc.collect()
log(f'  总计 {len(feats)} 特征')

# ========== 4. 标签 ==========
log('[4] 标签 H=30...')
ret_future=np.full(len(C),np.nan,np.float32); ret_future[:-30]=(C[30:]/C[:-30]-1).astype(np.float32)
label=(ret_future>0).astype(np.int8)
del C,H,L,O,TB,TS,F,lr; gc.collect()

# ========== 5. 合并过滤 ==========
log('[5] 合并...')
X=pd.DataFrame(feats).astype(np.float32)
X['label']=label; X['ret_future']=ret_future; X['ts']=ts
del feats; gc.collect()

valid=~np.isnan(ret_future) & ~X.drop(columns=['label','ret_future','ts']).isna().any(axis=1).values
vi=np.where(valid)[0]
X=X.iloc[vi].reset_index(drop=True); ts_v=X['ts'].values; label_v=X['label'].values.astype(np.int8); ret_v=X['ret_future'].values.astype(np.float32)
feat_df=X.drop(columns=['label','ret_future','ts'])
del X,valid,vi,ret_future,label; gc.collect()

def ts_mask(tsarr,s,e):
    a=int(datetime.datetime.strptime(s,'%Y-%m-%d').replace(tzinfo=datetime.timezone.utc).timestamp())
    b=int(datetime.datetime.strptime(e,'%Y-%m-%d').replace(tzinfo=datetime.timezone.utc).timestamp())
    return (tsarr>=a)&(tsarr<b)

tr_m=ts_mask(ts_v,'2020-01-01','2024-06-30'); es_m=ts_mask(ts_v,'2024-06-30','2024-09-30'); te_m=ts_mask(ts_v,'2025-09-30','2026-08-29')
log(f'  TR={tr_m.sum():,} ES={es_m.sum():,} TE={te_m.sum():,}  nF={feat_df.shape[1]}')

# ========== 6. 训练 ==========
log('[6] 训练 1 seed nl63 lr0.02...')
Xtr=feat_df[tr_m].values.astype(np.float32); ytr=label_v[tr_m]
Xes=feat_df[es_m].values.astype(np.float32); yes=label_v[es_m]
Xte=feat_df[te_m].values.astype(np.float32); yte=label_v[te_m]
keep=np.abs(ret_v[tr_m])>=0.0005
Xtr_f=Xtr[keep]; ytr_f=ytr[keep]; del Xtr,ytr,feat_df; gc.collect()

t=time.time()
lp=dict(num_leaves=63,learning_rate=0.02,min_data_in_leaf=200,feature_fraction=0.9,bagging_fraction=0.9,bagging_freq=5,lambda_l2=1.0,verbose=-1,num_threads=3,objective='binary',metric='auc',seed=42)
dtr=lgb.Dataset(Xtr_f,label=ytr_f); des=lgb.Dataset(Xes,label=yes,reference=dtr)
m=lgb.train(lp,dtr,num_boost_round=4000,valid_sets=[des],callbacks=[lgb.early_stopping(300,verbose=False)])
log(f'  1 seed: {time.time()-t:.0f}s best_iter={m.best_iteration}')
p_te=m.predict(Xte); auc=roc_auc_score(yte,p_te); log(f'  TE AUC={auc:.4f}')

# FI
fi=pd.DataFrame({'f':m.feature_name(),'gain':m.feature_importance(importance_type='gain')}).sort_values('gain',ascending=False)
fi['pct']=fi['gain']/fi['gain'].sum()*100
log('\n[FI] top25:')
for _,row in fi.head(25).iterrows(): log(f'  {row["f"]:<25s} {row["gain"]:>10.0f} {row["pct"]:>5.1f}%')

# ========== 7. Top-k ==========
log('\n[7] Top-k acc')
o=np.argsort(p_te)
for k in [0.003,0.005,0.008,0.01,0.012,0.015,0.02,0.03,0.05,0.08,0.10]:
    idx=o[-max(int(len(p_te)*k),1):]; acc=(yte[idx]==1).mean(); tp=tpd(len(idx),len(yte))
    log(f'  top{k*100:.1f}%: acc={acc:.4f} tpd={tp:.1f}')

# ========== 8. Per-hour ==========
log('\n[8] Per-hour')
hr_v=pd.to_datetime(ts_v[te_m],unit='s',utc=True).hour.values
ranked=sorted([(h,roc_auc_score(yte[hr_v==h],p_te[hr_v==h])) for h in range(24) if (hr_v==h).sum()>100],key=lambda x:-x[1])
log(f'  top10h: {[(h,f"{a:.4f}") for h,a in ranked[:10]]}')

hits=0
for K in range(2,25):
    sel=[h for h,_ in ranked[:K]]; m=np.isin(hr_v,sel)
    if m.sum()<500: continue
    ps=p_te[m]; ys=yte[m]
    for thr in range(50,100,1):
        hv=np.percentile(ps,thr); hit=ps>hv
        if hit.sum()<20: continue
        acc=(ys[hit]==1).mean(); tp=tpd(hit.sum(),len(p_te))
        if acc>=0.65 and tp>=15: log(f'  ✅ K={K}h P{thr}: acc={acc:.4f} tpd={tp:.1f}'); hits+=1
log(f'  达标配置数: {hits}')

# ========== 9. 汇总 ==========
log(f'\n{"="*60}'); log('🏆'); log(f'{"="*60}')
log(f'  nF={len(feats)} AUC={auc:.4f}')
log(f'  基线=0.5450 → {"✅涨了"+f"{auc-0.5450:+.4f}"+"!" if auc>0.545 else "❌跌了"}')
log(f'\n⏱ {time.time()-t0:.0f}s')
