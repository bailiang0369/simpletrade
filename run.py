"""主流程编排。
阶段:
  data      - 从 GitHub Release 拉取 bn_data 合并数据 -> 特征数据集
  train     - 对每个交易对训练全部基模型并保存概率
  ensemble  - 堆叠meta + 评估(全局top1% / 按日top1%) + 跨资产
  full      - 上面全部
  report    - 从已保存概率重新出报告(不重训)
"""
import argparse
import gc
import json
import os
import sys
import time

import numpy as np
import polars as pl

import config
from build_dataset import build_all
from data_store import AssetContext
from evaluate import check_all, evaluate_topk, evaluate_topk_daily
from ensemble import Ensemble
from fetch_data import fetch as fetch_data
from models.gbdt import GBDTModel
from models.dl_seq import GRUModel
from models.pattern import CNNImageModel, DTWKNN, FAISSNN, TimeNormKNN
from models.stat_signal import StatSignal


def build_models(seed=None):
    seed = seed or config.SEED
    return [
        GBDTModel(seed=seed),
        GRUModel(seed=seed),
        CNNImageModel(seed=seed),
        FAISSNN(seed=seed),
        TimeNormKNN(seed=seed),
        DTWKNN(seed=seed),
        StatSignal(seed=seed),
    ]


def train_symbol(symbol, save_models=True):
    """每个模型在独立子进程中训练+预测(内存随进程退出完全释放, 规避4GB限制)。
    已产出 {symbol}_{model}_pv.npy/_pt.npy 的模型自动跳过(断点续训)。"""
    import subprocess
    models = build_models()
    log_path = os.path.join(config.DATA_DIR, f"train_{symbol}.log")
    for m in models:
        pv_p = f"{config.DS_DIR}/{symbol}_{m.name}_pv.npy"
        pt_p = f"{config.DS_DIR}/{symbol}_{m.name}_pt.npy"
        if os.path.exists(pv_p) and os.path.exists(pt_p) and \
           os.path.getsize(pv_p) > 0 and os.path.getsize(pt_p) > 0:
            print(f"[train] {symbol} {m.name} already done, skip", flush=True)
            continue
        cmd = [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_one.py"),
               "--symbol", symbol, "--model", m.name]
        if not save_models:
            cmd.append("--no-save-model")
        t0 = time.time()
        with open(log_path, "a") as lf:
            r = subprocess.run(cmd, env={**os.environ,
                                         "BN_DATA_DIR": os.environ.get("BN_DATA_DIR", config.DATA_DIR)},
                               stdout=lf, stderr=subprocess.STDOUT)
        if r.returncode != 0:
            raise RuntimeError(f"[train] {symbol} {m.name} subprocess failed rc={r.returncode}, see {log_path}")
        print(f"[train] {symbol} {m.name} subprocess done in {time.time()-t0:.0f}s", flush=True)
    # 汇总各模型概率
    Pv_cols, Pt_cols = [], []
    for m in models:
        Pv_cols.append(np.load(f"{config.DS_DIR}/{symbol}_{m.name}_pv.npy"))
        Pt_cols.append(np.load(f"{config.DS_DIR}/{symbol}_{m.name}_pt.npy"))
    ctx = AssetContext(symbol)
    Pv = np.column_stack(Pv_cols)
    Pt = np.column_stack(Pt_cols)
    yv, yt = ctx.y("meta_val"), ctx.y("test")
    rv, rt = ctx.retf("meta_val"), ctx.retf("test")
    tv, tt = ctx.times("meta_val"), ctx.times("test")
    np.savez(f"{config.DS_DIR}/{symbol}_probs.npz",
             Pv=Pv, Pt=Pt, yv=yv, yt=yt, rv=rv, rt=rt, tv=tv.astype("datetime64[s]"),
             tt=tt.astype("datetime64[s]"))
    print(f"[train] {symbol} probs saved -> {config.DS_DIR}/{symbol}_probs.npz", flush=True)
    return models


def _save_model(m, symbol):
    path = os.path.join(config.MODEL_DIR, f"{symbol}_{m.name}")
    try:
        m.save(path)
    except Exception as e:
        print(f"  (model save skipped {m.name}: {e})")


def load_probs(symbol):
    d = np.load(f"{config.DS_DIR}/{symbol}_probs.npz")
    return d


def single_model_report(symbol):
    """各基模型单独在test上的top1%准确率(用于了解模型多样性)。"""
    d = load_probs(symbol)
    Pt, yt, rt, tt = d["Pt"], d["yt"], d["rt"], d["tt"]
    names = ["gbdt", "gru", "cnn_img", "faiss", "timenorm", "dtw", "stat"]
    out = {}
    for j, nm in enumerate(names):
        r = evaluate_topk(Pt[:, j], yt, rt, tt)
        out[nm] = {"acc": r["accuracy"], "tpd": r["trades_per_day"]}
    return out


def ensemble_symbol(symbol, pmeta=None):
    d = load_probs(symbol)
    Pv, Pt = d["Pv"], d["Pt"]
    yv, yt, rv, rt = d["yv"], d["yt"], d["rv"], d["rt"]
    tv, tt = d["tv"], d["tt"]
    names = ["gbdt", "gru", "cnn_img", "faiss", "timenorm", "dtw", "stat"]

    # 堆叠meta
    edges = np.abs(2 * Pv - 1)
    Xm = np.hstack([Pv, edges])
    from sklearn.linear_model import LogisticRegression
    meta = LogisticRegression(C=1.0, max_iter=1000, n_jobs=1)
    meta.fit(Xm, yv)
    Pm = meta.predict_proba(np.hstack([Pt, np.abs(2 * Pt - 1)]))[:, 1]

    # 平均投票
    Pvote = Pt.mean(axis=1)

    rep = {}
    for label, p in [("meta", Pm), ("vote", Pvote)]:
        g = evaluate_topk(p, yt, rt, tt)
        dd = evaluate_topk_daily(p, yt, rt, tt)
        rep[label] = {"global": g, "daily": dd}
    # 各基模型单独
    rep["base"] = {}
    for j, nm in enumerate(names):
        rep["base"][nm] = evaluate_topk(Pt[:, j], yt, rt, tt)["accuracy"]
    return rep, Pm, Pt


def run_report(symbols=None):
    symbols = symbols or config.SYMBOLS
    result = {}
    for s in symbols:
        result[s] = {}
        result[s]["base"] = single_model_report(s)
        rep, Pm, Pt = ensemble_symbol(s)
        result[s]["meta_global"] = rep["meta"]["global"]
        result[s]["meta_daily"] = rep["meta"]["daily"]
        result[s]["vote_global"] = rep["vote"]["global"]
    # 跨资产
    b = result[symbols[0]]["meta_global"]
    e = result[symbols[1]]["meta_global"]
    result["cross_asset"] = {
        "acc_btc": b["accuracy"], "acc_eth": e["accuracy"],
        "delta_pp": abs(b["accuracy"] - e["accuracy"]) * 100,
        "pass": abs(b["accuracy"] - e["accuracy"]) <= config.CROSS_ASSET_MAX_DELTA,
    }
    # 交易明细(meta, BTC)供人工核验
    try:
        d = load_probs(symbols[0])
        _, Pm, _ = ensemble_symbol(symbols[0])
        conf = np.maximum(Pm, 1 - Pm)
        pred = (Pm >= 0.5).astype(int)
        tt = np.asarray(d["tt"]).astype("datetime64[s]")
        k = int(len(Pm) * config.COVERAGE)
        sel = np.argsort(-conf)[:k]
        trades = pl.DataFrame({
            "time": np.asarray(tt[sel]).astype("datetime64[ms]"),
            "symbol": symbols[0], "prob": Pm[sel],
            "direction": pred[sel], "label": d["yt"][sel], "ret15m": d["rt"][sel],
        })
        trades.write_csv(os.path.join(config.RESULT_DIR, "trades_meta_btc.csv"))
    except Exception as ex:
        print("  (trade csv skip)", ex)
    with open(os.path.join(config.RESULT_DIR, "report.json"), "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="full", choices=["data", "train", "ensemble", "full", "report"])
    ap.add_argument("--symbols", nargs="*", default=None)
    args = ap.parse_args()
    symbols = args.symbols or config.SYMBOLS

    if args.stage in ("data", "full"):
        # 从 GitHub Release 拉取 bn_data 合并数据(增量跳过) -> 组装特征数据集
        fetch_data(symbols=symbols)
        build_all()
    if args.stage in ("train", "full"):
        for s in symbols:
            train_symbol(s)
    if args.stage in ("ensemble", "full"):
        run_report(symbols)
    if args.stage == "report":
        run_report(symbols)


if __name__ == "__main__":
    main()
