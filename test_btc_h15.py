"""快速补跑 BTC h15 eps=0 vs 0.0005 — 只跑 BTC h15。"""
import sys; sys.path.insert(0,'/workspace')
import os, gc, time, numpy as np, lightgbm as lgb
from sklearn.metrics import roc_auc_score
import config; from data_store import AssetContext

SEEDS = [42, 49, 56, 63, 70, 77, 84, 91, 98, 105]
def rank_ens(a):
    P=np.stack(a,0); R=np.zeros_like(P)
    for i in range(P.shape[0]): R[i]=np.argsort(np.argsort(P[i])).astype(np.float64)/(P.shape[1]-1)
    return R.mean(0)
def nolook(q,c,p,y,ts):
    do=ts//86400; days=np.unique(do); s=np.zeros(len(y),bool)
    for di,d in enumerate(days.astype(int).tolist()):
        prior=days.astype(int).tolist()[max(0,di-30):di]
        if len(prior)<30: continue
        t=do==d
        for side in [1,0]:
            m=t&(p==side); h=np.isin(do,prior)&(p==side)
            if h.sum()==0 or m.sum()==0: continue
            s[m&(c>=float(np.percentile(c[h],q)))]=True
    return float((p[s]==y[s]).mean())*100, s.sum()/len(days), int(s.sum())
base_p = {"objective":"binary","metric":"auc","verbose":-1,"num_leaves":127,
          "min_data_in_leaf":200,"learning_rate":0.05,"feature_fraction":0.8,"bagging_fraction":0.8}
ctx = AssetContext("BTC", horizon=15)
Xtr=ctx.Xall[ctx.split_rows["train"]]; Xes=ctx.Xall[ctx.split_rows["early_stop"]]
ytr=ctx.label[ctx.split_rows["train"]].astype(np.float64); yes=ctx.label[ctx.split_rows["early_stop"]].astype(np.float64)
ret=ctx.retf("train"); wtr=np.clip(np.abs(ret)*50,0.5,5.0)
mm=ctx.split_rows["meta_val"]; mt=ctx.split_rows["test"]

for eps in [0, 0.0005]:
    print(f"\nBTC h15 eps={eps}", flush=True)
    m = slice(None) if eps==0 else np.abs(ret)>eps
    Xd,yd,wd = Xtr[m], ytr[m], wtr[m]
    print(f"  keep={len(Xd)}/{len(Xtr)} ({len(Xd)/len(Xtr)*100:.1f}%)", flush=True)
    Pmv,Pte=[],[]; t0=time.time()
    for s in SEEDS:
        tr=lgb.Dataset(Xd,yd,weight=wd); es=lgb.Dataset(Xes,yes,reference=tr)
        m2=lgb.train({**base_p,"seed":s}, tr, 3000, [es],
                     callbacks=[lgb.early_stopping(100),lgb.log_evaluation(0)])
        Pmv.append(m2.predict(ctx.Xall[mm]).astype(np.float64))
        Pte.append(m2.predict(ctx.Xall[mt]).astype(np.float64)); gc.collect()
    Pmv,Pte = rank_ens(Pmv),rank_ens(Pte)
    for split,P in [("mv",Pmv),("te",Pte)]:
        m2 = mm if split=="mv" else mt
        y=ctx.label[m2]
        ts=np.asarray(ctx.times("meta_val" if split=="mv" else "test")).astype("datetime64[s]").astype(np.int64)
        c=np.abs(P-0.5)*2; p=(P>=0.5).astype(np.int8); auc=roc_auc_score(y,P)
        print(f"  {split} AUC={auc:.4f}", flush=True)
        for q in [99.0, 99.2, 99.4, 99.5]:
            a,t,n=nolook(q,c,p,y,ts)
            print(f"    q={q} -> {a:.2f}% @ {t:.1f}t ({n})", flush=True)
    print(f"  done ({time.time()-t0:.0f}s)", flush=True)
print("\n✅ BTC h15 done", flush=True)
