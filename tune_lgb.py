"""Grid search LightGBM params on 87-feat full data."""
import numpy as np, time, gc, lightgbm as lgb
from sklearn.metrics import roc_auc_score

t0 = time.time()
print('[1] load splits...', flush=True)
ys = np.load('/workspace/models/y_split.npz', allow_pickle=True)
X_tr = np.load('/workspace/models/X_tr.npy').astype(np.float32)
X_es = np.load('/workspace/models/X_es.npy').astype(np.float32)
y_tr_30 = ys['y30_tr']; y_es_30 = ys['y30_es']

print(f'  X_tr={X_tr.shape}')

# Grid
GRID = [
    {'num_leaves':31, 'min_child_samples':100, 'feature_fraction':0.8, 'lambda_l2':0.1},
    {'num_leaves':63, 'min_child_samples':200, 'feature_fraction':0.8, 'lambda_l2':0.1},
    {'num_leaves':127, 'min_child_samples':500, 'feature_fraction':0.8, 'lambda_l2':0.1},
    {'num_leaves':127, 'min_child_samples':200, 'feature_fraction':0.6, 'lambda_l2':0.1},
    {'num_leaves':63, 'min_child_samples':100, 'feature_fraction':1.0, 'lambda_l2':0.5},
    {'num_leaves':127, 'min_child_samples':300, 'feature_fraction':0.7, 'lambda_l2':1.0},
    {'num_leaves':255, 'min_child_samples':500, 'feature_fraction':0.6, 'lambda_l2':1.0},
    {'num_leaves':127, 'min_child_samples':200, 'feature_fraction':0.9, 'lambda_l2':0.01},
]

print('\n[2] grid search H=30 on ES...', flush=True)
best_auc = 0; best_params = None
for i, g in enumerate(GRID):
    p = {'objective':'binary','learning_rate':0.05,'bagging_fraction':0.8,'bagging_freq':5,
         'verbose':-1,'n_jobs':3,'seed':42, **g}
    m = lgb.train(p, lgb.Dataset(X_tr, y_tr_30), num_boost_round=5000,
                  valid_sets=[lgb.Dataset(X_es, y_es_30)],
                  callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
    pv = m.predict(X_es)
    auc = roc_auc_score(y_es_30, pv)

    # Also predict on TE
    X_te = np.load('/workspace/models/X_te.npy').astype(np.float32)
    pv_te = m.predict(X_te)
    auc_te = roc_auc_score(ys['y30_te'], pv_te)
    del m, X_te; gc.collect()

    top1 = pv_te.argsort()[-max(1,int(len(pv_te)*0.01)):]
    acc1 = ys['y30_te'][top1].mean()*100

    tag = 'BEST' if auc > best_auc else '    '
    print(f'  [{i+1}] {g}  es={auc:.4f} te={auc_te:.4f} top1%={acc1:.1f}%  {tag}  ({time.time()-t0:.0f}s)', flush=True)

    if auc > best_auc:
        best_auc = auc; best_params = g

print(f'\n  Best params on ES: {best_params}')
print(f'  Best ES AUC: {best_auc:.4f}')

# Focal loss
print('\n[3] Try focal loss custom objective...', flush=True)
def focal_obj(y_true, y_pred):
    gamma, alpha = 2.0, 0.5
    p = 1.0 / (1.0 + np.exp(-np.clip(y_pred, -10, 10)))
    grad = alpha * y_true * (1 - p) ** gamma * p - (1 - alpha) * (1 - y_true) * p ** gamma * (1 - p)
    hess = np.clip(p * (1 - p) * (alpha * y_true * (1 - p) ** gamma * (1 - gamma * np.clip(p/(1-p), -10, 10)) +
                            (1 - alpha) * (1 - y_true) * p ** gamma * (1 - gamma * np.clip((1-p)/p, -10, 10))),
                  1e-3, None)
    return grad, hess

p = {'learning_rate':0.05,'num_leaves':63,'min_child_samples':200,'feature_fraction':0.8,
     'bagging_fraction':0.8,'bagging_freq':5,'lambda_l2':0.1,'verbose':-1,'n_jobs':3,'seed':42}
m = lgb.train(p, lgb.Dataset(X_tr, y_tr_30), num_boost_round=2000,
              fobj=focal_obj,
              valid_sets=[lgb.Dataset(X_es, y_es_30)],
              callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
pv_es = 1.0 / (1.0 + np.exp(-m.predict(X_es)))
auc_es = roc_auc_score(y_es_30, pv_es)
X_te = np.load('/workspace/models/X_te.npy').astype(np.float32)
pv_te = 1.0 / (1.0 + np.exp(-m.predict(X_te)))
auc_te = roc_auc_score(ys['y30_te'], pv_te)
del m, X_te; gc.collect()
top1 = pv_te.argsort()[-max(1,int(len(pv_te)*0.01)):]
acc1 = ys['y30_te'][top1].mean()*100
print(f'  focal: es={auc_es:.4f} te={auc_te:.4f} top1%={acc1:.1f}%  ({time.time()-t0:.0f}s)', flush=True)

print(f'\n⏱ TOTAL: {time.time()-t0:.0f}s')
