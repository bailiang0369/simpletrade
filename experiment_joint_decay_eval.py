#!/usr/bin/env python3
"""JOINT 时间衰减权重 A/B 的 R2 无泄漏评估 (mv 段置信度种子 -> test 一次性)。

对比各 tag (base / decay365 / decay730) 的 JOINT lgb×5 预测, 输出总acc/最差月/坏月。
用法: python experiment_joint_decay_eval.py [base,decay365,decay730]
"""
import os, sys, gc
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from data_store import AssetContext
import experiment_seq_model as ESM

FAMS = ["lgb"]


def load_rank(symbol, split, tag):
    Ps = [np.load(f"{config.DS_DIR}/JOINT{tag}_{symbol}_{f}_{split}_P.npy") for f in FAMS]
    P = np.concatenate(Ps, axis=0)
    n = P.shape[1]
    R = np.stack([np.argsort(np.argsort(P[i])).astype(np.float64) / (n - 1) for i in range(P.shape[0])], axis=0)
    return R.mean(axis=0)


def run_symbol(symbol, tag):
    ctx = AssetContext(symbol, horizon=30)
    pmv = load_rank(symbol, "meta_val", tag)
    pt = load_rank(symbol, "test", tag)
    seed_conf = list(np.abs(pmv - 0.5) * 2)[-ESM.WIN_DAYS * 1440:]
    r = ESM.r2_eval(np.abs(pt - 0.5) * 2, (pt >= 0.5).astype(np.int8),
                    ctx.ds_ts[ctx.split_rows["test"]].astype(np.int64), seed_conf)
    ysel = ctx.y("test")[r["sel"]]
    acc, rows = ESM.monthly(r["pred"], ysel, r["mts"])
    worst = min(x[2] for x in rows)
    nbad = sum(1 for x in rows if x[2] < 0.55)
    return acc, worst, nbad, len(ysel)


def main():
    tags = sys.argv[1].split(",") if len(sys.argv) > 1 else ["", "_decay365", "_decay730"]
    tags = ["" if t == "base" else (t if t.startswith("_") else "_" + t) for t in tags]
    for s in ("ETH", "BTC"):
        print(f"\n===== {s} R2 (JOINT lgb×5) =====", flush=True)
        res = {}
        for tag in tags:
            try:
                acc, worst, nbad, nsig = run_symbol(s, tag)
            except FileNotFoundError as e:
                print(f"  [{tag or 'base'}] 缺文件: {e}", flush=True)
                continue
            res[tag] = (acc, worst, nbad, nsig)
            print(f"  [{tag or 'base':<10}] 总acc={acc:.4f}  最差月={worst:.4f}  坏月={nbad}  信号={nsig}", flush=True)
        if "base" in res or "" in res:
            base = res.get("base") or res[""]
            b_acc = base[0]
            print("  --- 相对 base 增量 ---", flush=True)
            for tag, (acc, worst, nbad, nsig) in res.items():
                if tag in ("base", ""):
                    continue
                print(f"  [{tag:<10}] acc {acc-b_acc:+.4f}  最差月 {worst-base[1]:+.4f}  坏月 {nbad-base[2]:+d}", flush=True)


if __name__ == "__main__":
    main()
