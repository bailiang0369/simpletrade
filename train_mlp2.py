"""MLP on engineered feats — compare with LGB/CB."""
import numpy as np, time, gc
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score

t0 = time.time()
print('[1] load splits...', flush=True)
ys = np.load('/workspace/models/y_split.npz', allow_pickle=True)
X_tr = np.load('/workspace/models/X_tr.npy').astype(np.float32)
X_es = np.load('/workspace/models/X_es.npy').astype(np.float32)
X_te = np.load('/workspace/models/X_te.npy').astype(np.float32)
y_tr = ys['y30_tr']; y_es = ys['y30_es']; y_te = ys['y30_te']
print(f'  X_tr={X_tr.shape}')

# Scale
tr_mean = X_tr.mean(axis=0); tr_std = np.maximum(X_tr.std(axis=0), 1e-6)
X_tr_s = (X_tr - tr_mean) / tr_std
X_es_s = (X_es - tr_mean) / tr_std
X_te_s = (X_te - tr_mean) / tr_std
del X_tr, X_es, tr_mean, tr_std; gc.collect()

# Subset for speed
sub = np.random.RandomState(42).choice(len(X_tr_s), size=min(200000, len(X_tr_s)), replace=False)
print(f'  subset: {len(sub):,}')
X_tr_sub = X_tr_s[sub]; y_tr_sub = y_tr[sub]

print('\n[2] train MLP...', flush=True)
for hidden in [(128,64), (256,128), (512,256)]:
    print(f'  hidden={hidden}...', flush=True)
    mlp = MLPClassifier(hidden_layer_sizes=hidden, activation='relu',
                        alpha=0.01, max_iter=100, early_stopping=True,
                        validation_fraction=0.1, random_state=42,
                        n_iter_no_change=15, learning_rate_init=0.001)
    mlp.fit(X_tr_sub, y_tr_sub)
    pv_es = mlp.predict_proba(X_es_s)[:,1]
    pv_te = mlp.predict_proba(X_te_s)[:,1]
    auc_es = roc_auc_score(y_es, pv_es)
    auc_te = roc_auc_score(y_te, pv_te)
    top1 = pv_te.argsort()[-max(1,int(len(pv_te)*0.01)):]
    acc1 = y_te[top1].mean()*100
    print(f'    es={auc_es:.4f} te={auc_te:.4f} top1%={acc1:.1f}%  ({time.time()-t0:.0f}s)', flush=True)
    del mlp; gc.collect()

print(f'\n⏱ TOTAL: {time.time()-t0:.0f}s')
