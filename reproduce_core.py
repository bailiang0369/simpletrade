"""阶段B 复现核心模型（stat / gbdt / gru）。按约束⑥跳过 cnn/dtw/faiss/timenorm。

用法(分模型子进程运行, 释放内存):
  python3 reproduce_core.py --symbol BTCUSDT --model stat
  ... gbdt / gru
输出: {DS_DIR}/{symbol}_{model}_pv.npy (meta_val 概率), ..._pt.npy (test 概率)

再以 --report 汇总各模型+堆叠meta+投票在 test 上的 top1% 准确率。
约束②: 各基模型只用 train 训练、early_stop 早停/校准; meta 只在 meta_val 上拟合;
        中间验证(模型取舍)只看 meta_val; test 仅作最终离线评测输出一次。
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from data_store import AssetContext


def run_train(symbol, model_name):
    # 按需导入对应模型类(避免因缺其它未实现模型而无法运行单个模型)
    from models.stat_signal import StatSignal
    factory = {"stat": StatSignal}
    if model_name == "gbdt":
        from models.gbdt import GBDTModel
        factory["gbdt"] = GBDTModel
    elif model_name == "gru":
        from models.dl_seq import GRUModel
        factory["gru"] = GRUModel
    ctx = AssetContext(symbol)
    m = factory[model_name](seed=config.SEED)
    t0 = time.time()
    m.fit(ctx)
    pv = np.asarray(m.predict(ctx, "meta_val"), dtype=np.float32)
    pt = np.asarray(m.predict(ctx, "test"), dtype=np.float32)
    np.save(f"{config.DS_DIR}/{symbol}_{model_name}_pv.npy", pv)
    np.save(f"{config.DS_DIR}/{symbol}_{model_name}_pt.npy", pt)
    print(f"[{symbol}] {model_name} done {time.time()-t0:.0f}s "
          f"pv={len(pv)} pt={len(pt)}", flush=True)


def report(symbols):
    from evaluate import evaluate_topk, evaluate_topk_daily
    names = ["stat", "gbdt", "gru"]
    out = {}
    for s in symbols:
        ctx = AssetContext(s)
        # 各模型概率在对应切分上加载
        pv = np.column_stack([np.load(f"{config.DS_DIR}/{s}_{n}_pv.npy") for n in names])
        pt = np.column_stack([np.load(f"{config.DS_DIR}/{s}_{n}_pt.npy") for n in names])
        yv, yt = ctx.y("meta_val"), ctx.y("test")
        rv, rt = ctx.retf("meta_val"), ctx.retf("test")
        tv, tt = ctx.times("meta_val"), ctx.times("test")

        # 约束②: 模型取舍只在 meta_val 上看
        rep_base_v = {n: evaluate_topk(pv[:, j], yv, rv, tv) for j, n in enumerate(names)}
        # 最终离线评估: test 只输出一次(所有基模型+两种集成)
        rep_base_t = {n: evaluate_topk(pt[:, j], yt, rt, tt) for j, n in enumerate(names)}

        # 堆叠meta(在 meta_val 上训练) + 投票
        from sklearn.linear_model import LogisticRegression
        Xm = np.hstack([pv, np.abs(2 * pv - 1)])
        meta = LogisticRegression(C=1.0, max_iter=1000, n_jobs=1)
        meta.fit(Xm, yv)
        pm = meta.predict_proba(np.hstack([pt, np.abs(2 * pt - 1)]))[:, 1]
        pvote = pt.mean(axis=1)

        out[s] = {
            "base_meta_val_acc": {n: rep_base_v[n]["accuracy"] for n in names},
            "base_test_acc": {n: rep_base_t[n]["accuracy"] for n in names},
            "meta_test": {k: v for k, v in evaluate_topk(pm, yt, rt, tt).items() if k in ("accuracy","k","trades_per_day")},
            "vote_test": {k: v for k, v in evaluate_topk(pvote, yt, rt, tt).items() if k in ("accuracy","k","trades_per_day")},
            # meta_val 上的 meta 表现(用于判断 test 结果是否可信/稳定)
            "pm_meta_val_acc": evaluate_topk(
                meta.predict_proba(Xm)[:, 1], yv, rv, tv)["accuracy"],
        }
        print(json.dumps({s: out[s]}, ensure_ascii=False, indent=2))
    with open(f"{config.RESULT_DIR}/phaseB_report.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    print(f"[report] saved -> {config.RESULT_DIR}/phaseB_report.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--model", choices=["stat", "gbdt", "gru"])
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if args.report:
        report([args.symbol])
    else:
        run_train(args.symbol, args.model)


if __name__ == "__main__":
    main()