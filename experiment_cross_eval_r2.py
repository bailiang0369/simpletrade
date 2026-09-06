#!/usr/bin/env python3
"""用 R2 无泄漏协议(与 HANDOFF 基线同口径)评估 pool20_joint 最新预测(含跨资产特征)。"""
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
    # meta_val 用于种子置信度锚定
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
    ESM.report(f"JOINT+跨资产 {s} test", r["pred"], ysel, r["mts"])
