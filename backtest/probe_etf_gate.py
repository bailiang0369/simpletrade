#!/usr/bin/env python3
"""ETF 决策层门控探针: 生产模型(R2协议)在 test 段选出的 top1% 交易,
按"前日 BTC ETF 流入/流出"分组, 对比准确率。

如果流出日交易准确率显著更低(如 >5pp), ETF 门控才有价值;
如果相当, 则日频 ETF 信号对 top1% 选择无增益, 证伪该方向。
用法: python probe_etf_gate.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os, sys, json
import numpy as np
from datetime import datetime, timezone
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from data_store import AssetContext
import experiment_seq_model as ESM

ETF_JSON = "/tmp/btc_etf.json"
FAMS = ("lgb", "xgb", "cat")


def load_flow():
    d = json.load(open(ETF_JSON))
    rows = []
    for x in d["days"]:
        ts = datetime.fromisoformat(x["date"]).replace(tzinfo=timezone.utc)
        ed = int(ts.timestamp()) // 86400
        f = x.get("netFlowUsd")
        if f is not None:
            rows.append((ed, float(f)))
    rows.sort()
    eds = np.array([a for a, _ in rows], dtype=np.int64)
    flows = np.array([b for _, b in rows], dtype=np.float64)
    return eds, flows


def load_pmean(symbol, split):
    Ps = [np.load(f"{config.DS_DIR}/JOINT_{symbol}_{f}_{split}_P.npy") for f in FAMS]
    P = np.concatenate(Ps, axis=0); n = P.shape[1]
    R = np.zeros_like(P, dtype=np.float64)
    for i in range(P.shape[0]):
        R[i] = np.argsort(np.argsort(P[i])).astype(np.float64) / (n - 1)
    return R.mean(axis=0)


def main():
    eds, flows = load_flow()
    for split in ("meta_val", "test"):
        for s in ("ETH", "BTC"):
            ctx = AssetContext(s, horizon=30)
            pv = load_pmean(s, split)
            pmv = load_pmean(s, "meta_val")
            ts_seg = ctx.ds_ts[ctx.split_rows[split]].astype(np.int64)
            # 阈值种子: meta_val 段评估时用其自身之前的 meta_val 历史; 此处简化用全 mv 段自身(与test一致口径差异)
            if split == "meta_val":
                seed_conf = list(np.abs(pmv - 0.5) * 2)[-ESM.WIN_DAYS * 1440:]
            else:
                seed_conf = list(np.abs(pmv - 0.5) * 2)[-ESM.WIN_DAYS * 1440:]
            conf = np.abs(pv - 0.5) * 2
            r = ESM.r2_eval(conf, (pv >= 0.5).astype(np.int8), ts_seg, seed_conf)
            sel = r["sel"]
            ysel = ctx.y(split)[sel]
            pred = r["pred"]
            day = ts_seg[sel] // 86400
            pday = np.searchsorted(eds, day, side="left") - 1
            ok = pday >= 0
            print(f"\n===== {s} {split} top1% 交易按前日ETF流分组 =====", flush=True)
            if not ok.all():
                print(f"  [warn] {int((~ok).sum())} 笔交易无前日ETF数据, 剔除", flush=True)
            day = day[ok]; pday = pday[ok]; ysel = ysel[ok]; pred = pred[ok]
            flow = flows[pday]
            pos = flow > 0; neg = flow < 0
            for name, m in (("前日流入", pos), ("前日流出", neg), ("|流|大(P90)", np.abs(flow) > np.quantile(np.abs(flows), 0.9))):
                if m.sum() < 30:
                    print(f"  [{name:<10}] 交易={m.sum():4d}  (样本不足)", flush=True)
                    continue
                acc = float((pred[m] == ysel[m]).mean())
                print(f"  [{name:<10}] 交易={m.sum():4d}  准确率={acc:.4f}", flush=True)
            if pos.sum() >= 30 and neg.sum() >= 30:
                d = (pred[pos] == ysel[pos]).mean() - (pred[neg] == ysel[neg]).mean()
                print(f"  >>> 流入日-流出日 准确率差 {d:+.4f}", flush=True)
                mts = ts_seg[sel][ok].astype("datetime64[s]").astype("datetime64[M]")
                print("  逐月 流入/流出 准确率差:")
                line = []
                for u in np.unique(mts):
                    m = (mts == u)
                    mp = m & pos; mn = m & neg
                    if mp.sum() >= 15 and mn.sum() >= 15:
                        line.append(f"{str(u)[:7]}:{(pred[mp]==ysel[mp]).mean()-(pred[mn]==ysel[mn]).mean():+.3f}({mp.sum()}/{mn.sum()})")
                print("   " + "  ".join(line), flush=True)


if __name__ == "__main__":
    main()
