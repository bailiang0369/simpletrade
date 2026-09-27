"""Cross-config ensemble + NN×LGBM stacking check."""
import sys; sys.stdout.reconfigure(line_buffering=True)
import time
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

d = np.load('/workspace/models_saved/nn_predictions_v6_final.npz')
print("NN preds keys:", list(d.keys()), flush=True)
ts_te = d['ts_te']

# Also need y_te
d2 = np.load('/workspace/models_saved/seq_data_v6.npz')
y_te = d2['y_te']
ts_te_v6 = d2['ts_te']
y_es = d2['y_es']

def rank_agg_multi(pvs_list):
    """pvs_list: list of arrays, each array is (n_samples,)"""
    n = len(pvs_list[0])
    R = np.zeros((len(pvs_list), n), dtype=np.float64)
    for i,p in enumerate(pvs_list):
        p = np.nan_to_num(p, nan=0.5)
        R[i] = np.argsort(np.argsort(p)).astype(np.float64)/(n-1)
    return R.mean(0).astype(np.float32)

def do_eval(label, pv_es, yes, pv_te, yte):
    auc_es=roc_auc_score(yes,np.nan_to_num(pv_es,nan=0.5))
    auc_te=roc_auc_score(yte,np.nan_to_num(pv_te,nan=0.5))
    DAYS=(ts_te[-1]-ts_te[0])/86400.0
    print(f"\n  ★ [{label}] ES={auc_es:.4f} TE={auc_te:.4f}", flush=True)
    for pct in [0.5,1.0,1.5,2.0,3.0,5.0]:
        k=max(1,int(len(pv_te)*pct/100))
        acc=yte[np.argsort(-pv_te)[:k]].mean()*100; tpd=k/DAYS
        flag='🏆' if pct==1.0 and acc>=60 else ''
        print(f"    top-{pct:.1f}%: acc={acc:.1f}% tpd≈{tpd:.1f} {flag}", flush=True)

# Load all top3
names = list(set([k.replace('_te','').replace('_es','') for k in d.keys() if 'pv_' in k]))
print(f"\nAvailable: {names}", flush=True)

# Simple rank agg of all available
pvs_te = []; pvs_es = []
for k in d.keys():
    if k.endswith('_te') and k.startswith('pv_'):
        pvs_te.append(d[k])
    elif k.endswith('_es') and k.startswith('pv_'):
        pvs_es.append(d[k])
print(f"\nFound {len(pvs_te)} TE preds, {len(pvs_es)} ES preds", flush=True)

if len(pvs_te) >= 2:
    pv_ens_te = rank_agg_multi(pvs_te)
    pv_ens_es = rank_agg_multi(pvs_es)
    do_eval('ALL_top3_cross_config_ens', pv_ens_es, y_es, pv_ens_te, y_te)

# Try different combinations
combo_configs = [
    ('#1_d07 + #2_j01', ['pv_big_1024512256_adamw_d07_te', 'pv_big_1024512256_adamw_j01_te'],
                        ['pv_big_1024512256_adamw_d07_es', 'pv_big_1024512256_adamw_j01_es']),
]

