"""Shape-CNN v2: 保留 Stoch/DMI 形态序列, 小模型避免 OOM"""
import os, time, gc
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score
import config

SEG = 60; STRIDE = 8; H = config.HORIZON_MIN
BATCH = 512; EPOCHS = 20; LR = 2e-3; WD = 5e-4; PAT = 7

t0 = time.time()
print(f'[1] 加载 SEG={SEG} STRIDE={STRIDE}', flush=True)
eth = pd.read_parquet(f'{config.DS_DIR}/raw_ETH.parquet')
btc = pd.read_parquet(f'{config.DS_DIR}/raw_BTC.parquet')
bi = pd.Index(btc['ts'].values); ei = pd.Index(eth['ts'].values)
br = np.clip(btc['ts'].values.searchsorted(ei.values, side='right')-1, 0, len(btc)-1)

close = eth['close'].values.astype(np.float32)
high = eth['high'].values.astype(np.float32)
low = eth['low'].values.astype(np.float32)
buy_v = eth['buy_vol'].values.astype(np.float32)
sell_v = eth['sell_vol'].values.astype(np.float32)
ts = eth['ts'].values.astype(np.int64)
btc_c = btc['close'].values.astype(np.float32)[br]
N = len(close); del btc

# Stoch(60)
print(f'[2] Stoch/DMI 全序列 ...', flush=True)
close_s = pd.Series(close); high_s = pd.Series(high); low_s = pd.Series(low)
ll_60 = low_s.rolling(SEG, min_periods=SEG).min().values.astype(np.float32)
hh_60 = high_s.rolling(SEG, min_periods=SEG).max().values.astype(np.float32)
hk = np.maximum(hh_60 - ll_60, 1e-6)
stoch_k_all = (close - ll_60) / hk
# 填补 NaN (前 60 bar)
stoch_k_all = np.nan_to_num(stoch_k_all, nan=0.5)

# DMI(14)
h_l = high[1:]-low[1:]; h_cp = np.abs(high[1:]-close[:-1]); l_cp = np.abs(low[1:]-close[:-1])
tr = np.maximum(np.maximum(h_l, h_cp), l_cp)
up_m = high[1:]-high[:-1]; dn_m = low[:-1]-low[1:]
pdm = np.where((up_m>dn_m)&(up_m>0), up_m, 0.0)
mdm = np.where((dn_m>up_m)&(dn_m>0), dn_m, 0.0)
def ws(a,p):
    o=np.zeros(len(a),np.float64);o[0]=a[0]
    for i in range(1,len(a)):o[i]=o[i-1]-o[i-1]/p+a[i]
    return o
tr_s=ws(tr,14);pdm_s=ws(pdm,14);mdm_s=ws(mdm,14)
pdi_all=np.zeros(N,np.float32);mdi_all=np.zeros(N,np.float32)
pdi_all[1:]=(100*pdm_s/np.maximum(tr_s,1e-6)).astype(np.float32)
mdi_all[1:]=(100*mdm_s/np.maximum(tr_s,1e-6)).astype(np.float32)

bs = buy_v - sell_v; del buy_v, sell_v, high, low; gc.collect()

# 构建窗口
print(f'[3] 形态窗口 ...', flush=True)
W_list, T_list, Ts_list = [], [], []
cnt = 0
for start in range(SEG, N-SEG-H, STRIDE):
    end = start + SEG
    if end+H > N: break
    c0 = close[start]; bc0 = btc_c[start]
    w = np.zeros((7, SEG), np.float32)  # 6 channels
    w[0] = np.log(close[start:end]/max(c0,1e-6))        # close_ret 形态
    w[1] = stoch_k_all[start:end]                         # Stoch %K 形态
    w[2] = pdi_all[start:end]                             # +DI 形态
    w[3] = mdi_all[start:end]                             # -DI 形态
    w[4] = pdi_all[start:end] - mdi_all[start:end]        # DI_diff 形态
    bs_win = bs[start:end]; cum = np.cumsum(bs_win)
    w[5] = cum / max(np.abs(cum[-1]), 1.0)
    w[6] = np.log(btc_c[start:end]/max(btc_c[start],1e-6))  # BTC 价格形态
    label = 1 if close[end+H] > close[end] else 0
    W_list.append(w); T_list.append(label); Ts_list.append(ts[end])
    cnt += 1
    if cnt % 100_000 == 0:
        print(f'  ... {cnt:,} ({time.time()-t0:.0f}s)', flush=True)

X_all = np.array(W_list, np.float32); y_all = np.array(T_list, np.int64); ts_all = np.array(Ts_list, np.int64)
del W_list, T_list, Ts_list, stoch_k_all, pdi_all, mdi_all, bs, close, btc_c; gc.collect()
print(f'  ✅ {X_all.shape} pos={y_all.mean():.3f} ({time.time()-t0:.0f}s)', flush=True)

def mk(s,e):
    a=int(pd.Timestamp(s,tz='UTC').timestamp());b=int(pd.Timestamp(e,tz='UTC').timestamp())
    return (ts_all>=a)&(ts_all<b)
tr_m=mk(*config.SPLITS['train']);es_m=mk(*config.SPLITS['early_stop']);te_m=mk(*config.SPLITS['test'])
X_tr,y_tr=X_all[tr_m],y_all[tr_m];X_es,y_es=X_all[es_m],y_all[es_m];X_te,y_te=X_all[te_m],y_all[te_m]
del X_all,y_all,ts_all;gc.collect()
print(f'  TR={len(X_tr):,} ES={len(X_es):,} TE={len(X_te):,}', flush=True)

# 小 CNN
class ShapeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(7, 128, 3, padding=1, bias=False),
                                  nn.LayerNorm([128, SEG]), nn.GELU())
        # 短周期 (RF~5,~13,~29)
        self.b1 = nn.Conv1d(128, 128, 3, padding=2, dilation=2, bias=False)
        self.b2 = nn.Conv1d(128, 128, 3, padding=4, dilation=4, bias=False)
        self.b3 = nn.Conv1d(128, 128, 3, padding=8, dilation=8, bias=False)
        self.b4 = nn.Conv1d(128, 128, 3, padding=16, dilation=16, bias=False)
        self.ln = nn.LayerNorm([128, SEG])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(nn.Dropout(0.4), nn.Linear(128, 64),
                                  nn.GELU(), nn.Dropout(0.3), nn.Linear(64,1))
    def forward(self, x):
        x = self.stem(x)
        x = F.gelu(self.ln(self.b1(x) + self.b2(x) + self.b3(x) + self.b4(x) + x))
        return self.head(self.pool(x).squeeze(-1))

m = ShapeNet(); np_ = sum(p.numel() for p in m.parameters())
print(f'\n[4] ShapeCNN params={np_:,} ({np_/1e6:.2f}M)', flush=True)

ds_tr=TensorDataset(torch.from_numpy(X_tr),torch.from_numpy(y_tr))
ds_es=TensorDataset(torch.from_numpy(X_es),torch.from_numpy(y_es))
dl_tr=DataLoader(ds_tr,BATCH,shuffle=True,num_workers=0,drop_last=True)
dl_es=DataLoader(ds_es,BATCH,shuffle=False,num_workers=0)
del X_tr,y_tr,X_es,y_es;gc.collect()

opt=torch.optim.AdamW(m.parameters(),lr=LR,weight_decay=WD)
sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS,eta_min=1e-5)
crit=nn.BCEWithLogitsLoss();bauc=0;bst=None;ni=0

print(f'\n[5] 训练', flush=True)
for ep in range(1,EPOCHS+1):
    m.train();tl=[]
    for xb,yb in dl_tr:
        opt.zero_grad();log=m(xb).squeeze(-1);loss=crit(log,yb.float())
        loss.backward();opt.step();tl.append(loss.item())
    sch.step()
    m.eval();pva,ya=[],[]
    with torch.no_grad():
        for xb,yb in dl_es:
            pva.append(torch.sigmoid(m(xb).squeeze(-1)).numpy())
            ya.append(yb.numpy())
    pv=np.concatenate(pva);y=np.concatenate(ya)
    auc=roc_auc_score(y,pv)
    if auc>bauc+1e-5:bauc=auc;bst={k:v.clone() for k,v in m.state_dict().items()};ni=0
    else:ni+=1
    print(f'  ep{ep:2d} tr={np.mean(tl):.4f} es_auc={auc:.4f} best={bauc:.4f} '
          f'lr={sch.get_last_lr()[0]:.2e} {"★" if ni==0 else ""} ({time.time()-t0:.0f}s)',flush=True)
    if ni>=PAT:print(f'  ⏹ early stop');break

# Test
print(f'\n[6] Test',flush=True)
if bst:m.load_state_dict(bst)
m.eval();ds_te=TensorDataset(torch.from_numpy(X_te),torch.from_numpy(y_te))
dl_te=DataLoader(ds_te,BATCH,shuffle=False,num_workers=0)
pva,ya=[],[]
with torch.no_grad():
    for xb,yb in dl_te:
        pva.append(torch.sigmoid(m(xb).squeeze(-1)).numpy())
        ya.append(yb.numpy())
pv=np.concatenate(pva);y=np.concatenate(ya);auc=roc_auc_score(y,pv)
print(f'  📊 Test AUC={auc:.4f}')
for pct in [0.005,0.01,0.02,0.03,0.05]:
    k=max(1,int(len(pv)*pct));ti=pv.argsort()[-k:]
    acc=y[ti].mean()*100;tpd=k/356
    flag='🎯' if acc>=65 and tpd>=14 else ('★' if acc>=60 and tpd>=14 else '')
    print(f'  {flag} top{pct*100:.1f}%: acc={acc:.1f}% tpd={tpd:.1f}')
print(f'\n⏱ {time.time()-t0:.0f}s')
