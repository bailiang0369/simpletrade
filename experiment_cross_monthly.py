#!/usr/bin/env python3
"""打印当前 JOINT pool20(17列跨资产) 在 R2 协议下的逐月明细, 与 12列旧版对比。"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from data_store import AssetContext
import experiment_seq_model as ESM

for s in ("ETH", "BTC"):
    ctx = AssetContext(s, horizon=30)
    Ps = [np.load(f"{config.DS_DIR}/JOINT_{s}_{f}_test_P.npy") for f in ("lgb", "xgb", "cat")]
    P = np.concatenate(Ps, axis=0); n = P.shape[1]
    R = np.zeros_like(P, dtype=np.float64)
    for i in range(P.shape[0]):
        R[i] = np.argsort(np.argsort(P[i])).astype(np.float64) / (n - 1)
    p = R.mean(axis=0)
    conf = np.abs(p - 0.5) * 2
    Pv = [np.load(f"{config.DS_DIR}/JOINT_{s}_{f}_meta_val_P.npy") for f in ("lgb", "xgb", "cat")]
    Pv = np.concatenate(Pv, axis=0); nv = Pv.shape[1]
    Rv = np.zeros_like(Pv, dtype=np.float64)
    for i in range(Pv.shape[0]):
        Rv[i] = np.argsort(np.argsort(Pv[i])).astype(np.float64) / (nv - 1)
    pv = Rv.mean(axis=0)
    seed_conf = list(np.abs(pv - 0.5) * 2)[-ESM.WIN_DAYS * 1440:]
    r = ESM.r2_eval(conf, (p >= 0.5).astype(np.int8),
                    ctx.ds_ts[ctx.split_rows["test"]].astype(np.int64), seed_conf)
    ysel = ctx.y("test")[r["sel"]]
    acc, rows = ESM.monthly(r["pred"], ysel, r["mts"])
    print(f"\n=== {s} test 总acc={acc:.4f} 最差={min(x[2] for x in rows):.4f} 坏月={sum(1 for x in rows if x[2]<0.55)} 信号={len(ysel)}")
    for u, cnt, a in rows:
        flag = "  <-- 坏月" if a < 0.55 else ""
        print(f"  {u}: n={cnt:5d} acc={a:.4f}{flag}")
