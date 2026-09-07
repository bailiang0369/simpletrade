#!/usr/bin/env python3
"""无泄露评估协议: 阈值只用历史(当天之前)数据确定, 不使用当天任何未来信息。

选择规则 (严格因果):
  R1 固定阈值: τ = meta_val 置信度 99 分位数 (校准集定阈值, test 直接套用)
  R2 每日滚动阈值: τ_d = meta_val + d 之前所有已见天置信度的 99 分位 (盘前定当天阈值)
  R3 逐样本滚动阈值: τ_t = 截至 t 前 W 个样本置信度的 99 分位 (最实时, 合并序列上计算)
参照(仅展示偏差): LEAK-daily = 每天取当天 top-1% (有泄露, 之前错误使用)。
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os, sys, gc
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from data_store import AssetContext

FAMS = ["lgb", "xgb", "cat"]
P99 = 99.0
WIN_SAMPLES = 130_000   # 滚动窗口: 约 90 天 (每天 1440 样本)
R3_UPDATE_EVERY = 5000  # R3 每 5000 样本重算一次阈值(块内复用, 实盘近似)


def rank_mean(P_all, n):
    R = np.zeros_like(P_all, dtype=np.float64)
    for i in range(P_all.shape[0]):
        R[i] = np.argsort(np.argsort(P_all[i])).astype(np.float64) / (n - 1)
    return R.mean(axis=0)


def load_fused(symbol, tag, split):
    Ps = []
    for f in FAMS:
        path = f"{config.DS_DIR}/JOINT_{symbol}_{f}_{split}_P.npy" if tag == "JOINT" \
            else f"{config.DS_DIR}/{symbol}_{f}_{split}_P.npy"
        if not os.path.exists(path):
            return None
        Ps.append(np.load(path))
    P = np.concatenate(Ps, axis=0)
    return rank_mean(P, P.shape[1])


def summarize(sel, pred, y, mts_all, sec_all):
    n_days = np.unique(sec_all // 86400).size
    n_all = len(sec_all)
    if sel.size == 0:
        return (np.nan, np.nan, 0, 0, 0.0, 0.0, {})
    ps, ys, ms = pred[sel], y[sel], mts_all[sel]
    acc = float((ps == ys).mean())
    uniq = np.unique(ms)
    acc_m = {str(u)[:7]: float((ps == ys)[ms == u].mean()) for u in uniq}
    min_k = min(int((ms == u).sum()) for u in uniq)
    min_a = min(acc_m.values())
    nbad = sum(1 for a in acc_m.values() if a < 0.55)
    tpd = float(len(ps)) / n_days
    cov = float(len(ps)) / n_all
    return (acc, min_a, min_k, nbad, tpd, cov, acc_m)


def show(tag, s):
    if s is None:
        print(f"[{tag}] skipped")
        return
    acc, min_a, min_k, nbad, tpd, cov, acc_m = s
    print(f"[{tag}] acc={acc:.4f} min_month={min_a:.4f}(n={min_k}) "
          f"bad(<55)={nbad} tpd={tpd:.2f} cov={cov:.4f}")
    print("   逐月:", {k: round(v, 3) for k, v in acc_m.items()})


def main():
    configs = [("ETH", ""), ("BTC", ""), ("ETH", "JOINT"), ("BTC", "JOINT")]
    for symbol, tag in configs:
        label = tag or "SOLO"
        ctx = AssetContext(symbol, horizon=30)
        data = {}
        ok = True
        for split in ("meta_val", "test"):
            p = load_fused(symbol, tag, split)
            if p is None:
                print(f"===== {label} {symbol}: 预测P缺失, 跳过(旧模型产物被清) =====")
                ok = False
                del p
                gc.collect()
                break
            y = ctx.y(split)
            if len(p) != len(y):
                print(f"===== {label} {symbol}: 预测P({len(p)})与标签({len(y)})长度失配, 跳过(旧模型或旧ds产物) =====")
                ok = False
                del p
                gc.collect()
                break
            sec = np.asarray(ctx.times(split)).astype("datetime64[s]").astype(np.int64)
            pred = (p >= 0.5).astype(np.int8)
            conf = np.maximum(p, 1 - p)
            mts = sec.astype("datetime64[s]").astype("datetime64[M]")
            data[split] = dict(p=p, y=y, sec=sec, pred=pred, conf=conf, mts=mts)
            del p
            gc.collect()
        if not ok:
            del ctx
            gc.collect()
            continue

        for split in ("meta_val", "test"):
            d = data[split]
            n = len(d["sec"])
            print(f"\n===== {label} {symbol} {split} =====")

            # LEAK-daily 参照 (泄露版)
            day = d["sec"] // 86400
            days = np.unique(day)
            sm = np.zeros(n, bool)
            for dd in days:
                md = day == dd
                kd = max(1, int(np.ceil(int(md.sum()) * 0.01)))
                sub = np.where(md)[0]
                sm[sub[np.argsort(-d["conf"][sub])[:kd]]] = True
            show("LEAK-daily(ref)", summarize(np.where(sm)[0], d["pred"], d["y"], d["mts"], d["sec"]))

            if split == "meta_val":
                # meta_val 自滚动 R3 (用自身前段作历史)
                sm3 = np.zeros(n, bool)
                tau = np.nan
                for i in range(n):
                    if i % R3_UPDATE_EVERY == 0 and i >= 1000:
                        tau = np.percentile(d["conf"][max(0, i - WIN_SAMPLES):i], P99)
                    if i >= 1000:
                        sm3[i] = d["conf"][i] >= tau
                show("R3-per-sample", summarize(np.where(sm3)[0], d["pred"], d["y"], d["mts"], d["sec"]))
            else:
                # R1 固定阈值: τ = meta_val conf 99 分位
                mv_conf = data["meta_val"]["conf"]
                tau = np.percentile(mv_conf, P99)
                sm1 = d["conf"] >= tau
                show("R1-fixed(mv99)", summarize(np.where(sm1)[0], d["pred"], d["y"], d["mts"], d["sec"]))

                # R2 每日滚动: τ_d = 最近90天(含meta_val尾部+已见test天) 99 分位, 盘前定当天阈值
                mv_tail = mv_conf[-WIN_SAMPLES:]
                hist = list(mv_tail)
                sel_list = []
                for dd in days:
                    md = day == dd
                    tau_d = np.percentile(np.asarray(hist), P99)
                    s = np.where(md & (d["conf"] >= tau_d))[0]
                    sel_list.append(s)
                    hist.extend(d["conf"][md])
                    if len(hist) > WIN_SAMPLES * 2:
                        del hist[:len(hist) - WIN_SAMPLES * 2]
                sel2 = np.concatenate(sel_list) if sel_list else np.array([], dtype=np.int64)
                show("R2-daily-roll", summarize(sel2, d["pred"], d["y"], d["mts"], d["sec"]))

                # R3 逐样本滚动 (warmup = meta_val 尾部, 严格因果, 分块更新阈值)
                hist3 = list(data["meta_val"]["conf"][-WIN_SAMPLES:])
                sm3 = np.zeros(n, bool)
                tau = np.percentile(np.asarray(hist3), P99)
                for i in range(n):
                    if i % R3_UPDATE_EVERY == 0:
                        tau = np.percentile(np.asarray(hist3), P99)
                    sm3[i] = d["conf"][i] >= tau
                    hist3.append(d["conf"][i])
                    if len(hist3) > WIN_SAMPLES * 2:
                        del hist3[:len(hist3) - WIN_SAMPLES * 2]
                show("R3-per-sample", summarize(np.where(sm3)[0], d["pred"], d["y"], d["mts"], d["sec"]))
        del ctx
        gc.collect()


if __name__ == "__main__":
    main()
