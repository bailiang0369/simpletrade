"""simpletrade 主入口: 数据获取 -> 训练 -> 多周期评估

阶段:
  data      - 从 GitHub Release 拉取 bn_data 合并数据, 构建特征数据集
  train     - 训练单周期模型(默认H=30): GBDT / LSTM / StatSignal
  multi     - 多周期训练(H=15,30,60): 使用独立子进程训练各周期
  eval      - 评估: 单周期融合 + 多周期Union叠加 + 月度稳定性
  full      - 上面全部

用法:
  python run.py --stage data                          # 数据准备
  python run.py --stage train --symbols BTC           # 训练BTC单周期模型
  python run.py --stage multi                         # 多周期训练
  python run.py --stage eval                          # 评估报告
  python run.py --stage full                          # 全部流程
"""
import argparse
import gc
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

import config
from data_store import AssetContext
from evaluate import evaluate_topk, check_all

# ============================================================
# 阶段1: 数据
# ============================================================
def stage_data():
    """获取原始数据 + 构建特征数据集。"""
    from fetch_data import fetch as fetch_data
    from build_dataset import build_all
    fetch_data(symbols=config.SYMBOLS)
    build_all()
    # 构建多周期数据集(如有)
    for s in config.SYMBOLS:
        for h in (15, 60):
            src = f"{config.DS_DIR}/ds_{s}.parquet"
            dst = f"{config.DS_DIR}/ds_{s}_h{h}.parquet"
            if not os.path.exists(dst):
                print(f"[data] 复制 ds_{s}.parquet -> ds_{s}_h{h}.parquet", flush=True)
                shutil.copy2(src, dst)
        # 默认 H=30 作为主数据集
        src = f"{config.DS_DIR}/ds_{s}.parquet"
        dst = f"{config.DS_DIR}/ds_{s}_h30.parquet"
        if not os.path.exists(dst):
            shutil.copy2(src, dst)


# ============================================================
# 阶段2: 单周期训练
# ============================================================
def stage_train(symbols):
    """对每个币种训练 GBDT + LSTM + StatSignal (默认H=30)。"""
    for s in symbols:
        for model_name in ("gbdt", "lstm", "stat"):
            log_path = f"{config.DS_DIR}/train_{s}_{model_name}.log"
            pv_path = f"{config.DS_DIR}/{s}_{model_name}_pv.npy"
            pt_path = f"{config.DS_DIR}/{s}_{model_name}_pt.npy"
            if os.path.exists(pv_path) and os.path.exists(pt_path):
                print(f"[train] {s} {model_name} 已存在, 跳过", flush=True)
                continue
            t0 = time.time()
            cmd = [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "train_one.py"), "--symbol", s, "--model", model_name]
            with open(log_path, "w") as lf:
                r = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)
            if r.returncode != 0:
                print(f"[train] ❌ {s} {model_name} 失败, 详见 {log_path}", flush=True)
            else:
                print(f"[train] ✅ {s} {model_name} 完成 ({time.time()-t0:.0f}s)", flush=True)


# ============================================================
# 阶段3: 多周期训练
# ============================================================
def stage_multi():
    """多周期(H=15,30,60) GBDT + LSTM 训练。"""
    for s in config.SYMBOLS:
        for h in (15, 30, 60):
            # 恢复数据集
            shutil.copy2(f"{config.DS_DIR}/ds_{s}_h{h}.parquet",
                         f"{config.DS_DIR}/ds_{s}.parquet")

            # GBDT: 用独立子进程训练
            gbdt_pv = f"{config.DS_DIR}/{s}_gbdt_h{h}_pv.npy"
            gbdt_pt = f"{config.DS_DIR}/{s}_gbdt_h{h}_pt.npy"
            if not os.path.exists(gbdt_pv):
                t0 = time.time()
                cmd = [sys.executable, "-c", f"""
import sys; sys.path.insert(0, '{os.path.dirname(os.path.abspath(__file__))}')
import config, numpy as np
from data_store import AssetContext
from models.gbdt import GBDTModel
ctx = AssetContext('{s}')
m = GBDTModel(seed=config.SEED)
m.fit(ctx)
pv = np.asarray(m.predict(ctx, 'meta_val'), dtype=np.float32)
pt = np.asarray(m.predict(ctx, 'test'), dtype=np.float32)
np.save('{gbdt_pv}', pv); np.save('{gbdt_pt}', pt)
print(f'GBDT {s} H={h} done', flush=True)
"""]
                r = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
                print(f"[multi] GBDT {s} H={h} {'✅' if r.returncode==0 else '❌'} ({time.time()-t0:.0f}s)", flush=True)

            # LSTM: 用独立子进程训练
            lstm_mv = f"/data/user/work/{s}_lstm_h{h}_mv.npy"
            lstm_te = f"/data/user/work/{s}_lstm_h{h}_te.npy"
            if not os.path.exists(lstm_mv):
                t0 = time.time()
                cmd = [sys.executable, "-c", f"""
import sys; sys.path.insert(0, '{os.path.dirname(os.path.abspath(__file__))}')
import os, gc, numpy as np, torch, config
from data_store import AssetContext
from evaluate import evaluate_topk
from models.seq_lstm import SeqGBDLSTM, build_sequence_feats, build_ds_matrix

ctx = AssetContext('{s}')
import models.seq_lstm as sl; sl.MAX_TRAIN = 200_000
lm = SeqGBDLSTM(seed=42); lm.fit(ctx); gc.collect()
F = build_sequence_feats(ctx.o, ctx.h, ctx.l, ctx.c, ctx.tb, ctx.vol)

def predict_chunked(model, F, pos, look_back, BLK=20000):
    n = len(pos); out = np.zeros(n, dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for b0 in range(0, n, BLK):
            b1 = min(b0 + BLK, n)
            Xb = build_ds_matrix(F, pos[b0:b1], look_back)
            out[b0:b1] = torch.sigmoid(model(torch.from_numpy(Xb))).numpy().squeeze()
            del Xb
    return out

p_mv = predict_chunked(lm.model, F, ctx.ds_to_raw[ctx.split_rows['meta_val']], lm.look_back)
p_te = predict_chunked(lm.model, F, ctx.ds_to_raw[ctx.split_rows['test']], lm.look_back)
np.save('{lstm_mv}', p_mv); np.save('{lstm_te}', p_te)
print(f'LSTM {s} H={h} done', flush=True)
"""]
                r = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
                print(f"[multi] LSTM {s} H={h} {'✅' if r.returncode==0 else '❌'} ({time.time()-t0:.0f}s)", flush=True)
            gc.collect()

    # 恢复默认数据集
    shutil.copy2(f"{config.DS_DIR}/ds_{config.SYMBOLS[0]}_h30.parquet",
                 f"{config.DS_DIR}/ds_{config.SYMBOLS[0]}.parquet")


# ============================================================
# 阶段4: 多周期评估 (与 multi_period_stack.py 一致)
# ============================================================
HORIZONS = [15, 30, 60]


def _eval_symbol(s):
    """对单个币种做完整的多周期评估: 融合+Union+Score Stacking+月度稳定性。"""
    print(f"\n{'='*65}")
    print(f"  {s} 多周期评估 (top {config.COVERAGE*100:.0f}%)")
    print(f"{'='*65}")

    # ---- 1. 加载各周期预测 & 最优融合权重 ----
    from evaluate import _month_of_epoch
    results = {}
    for h in HORIZONS:
        shutil.copy2(f"{config.DS_DIR}/ds_{s}_h{h}.parquet",
                     f"{config.DS_DIR}/ds_{s}.parquet")

        gb_mv = np.load(f"{config.DS_DIR}/{s}_gbdt_h{h}_pv.npy").astype(np.float64)
        gb_te = np.load(f"{config.DS_DIR}/{s}_gbdt_h{h}_pt.npy").astype(np.float64)
        ls_mv = np.load(f"/data/user/work/{s}_lstm_h{h}_mv.npy").astype(np.float64)
        ls_te = np.load(f"/data/user/work/{s}_lstm_h{h}_te.npy").astype(np.float64)

        ctx = AssetContext(s, horizon=h)
        y_mv = ctx.y("meta_val"); retf_mv = ctx.retf("meta_val"); ts_mv = ctx.times("meta_val")
        y_te = ctx.y("test"); retf_te = ctx.retf("test"); ts_te = ctx.times("test")

        best_w, best_acc = 0.0, 0.0
        for w in np.arange(0.0, 1.01, 0.05):
            acc = evaluate_topk(w * gb_mv + (1 - w) * ls_mv, y_mv, retf_mv, ts_mv)["accuracy"]
            if acc > best_acc:
                best_acc, best_w = acc, w

        p_test = (best_w * gb_te + (1 - best_w) * ls_te).astype(np.float64)
        results[h] = dict(p=p_test, y=y_te, retf=retf_te, ts=ts_te, w=best_w,
                          gb=gb_te, ls=ls_te)
        del ctx, gb_mv, gb_te, ls_mv, ls_te; gc.collect()

    shutil.copy2(f"{config.DS_DIR}/ds_{s}_h30.parquet", f"{config.DS_DIR}/ds_{s}.parquet")

    # ---- 2. 各周期独立评估 ----
    single_metrics = {}
    total_trades = 0
    total_tpd = 0.0
    for h in HORIZONS:
        r = evaluate_topk(results[h]["p"], results[h]["y"],
                          results[h]["retf"], results[h]["ts"])
        single_metrics[h] = r
        total_trades += r["k"]
        total_tpd += r["trades_per_day"]
        print(f"  H={h:2d}  w={results[h]['w']:.2f}  "
              f"acc={r['accuracy']:.4f}  tpd={r['trades_per_day']:.1f}  "
              f"k={r['k']}  days={r['n_days_selected']}")

    wavg_acc = sum(r["k"] * r["accuracy"] for r in single_metrics.values()) / total_trades
    print(f"\n  --- Union 汇总 ---")
    print(f"  Σ交易量: {total_trades}  Σ交易/天: {total_tpd:.1f}")
    print(f"  加权平均准确率: {wavg_acc:.4f}")
    print(f"  天>=30 {'✅' if total_tpd >= 30 else '❌'}  "
          f"准确率>=65% {'✅' if wavg_acc >= 0.65 else '❌'}")

    # ---- 3. 时间对齐 Score Stacking ----
    ts_to_score = {}
    for h in HORIZONS:
        ts = results[h]["ts"]
        p = results[h]["p"]
        ts_sec = np.asarray(ts, dtype="datetime64[s]").astype(np.int64)
        for i in range(len(ts)):
            t = ts_sec[i]
            if t not in ts_to_score:
                ts_to_score[t] = {}
            ts_to_score[t][h] = p[i]

    common_ts = []
    common_scores = []
    for t, scores in ts_to_score.items():
        if len(scores) == 3:
            common_ts.append(t)
            common_scores.append(np.mean([scores[h] for h in HORIZONS]))

    if len(common_ts) > 0:
        common_scores = np.array(common_scores)
        shutil.copy2(f"{config.DS_DIR}/ds_{s}_h30.parquet", f"{config.DS_DIR}/ds_{s}.parquet")
        ctx0 = AssetContext(s, horizon=30)
        ts0_sec = np.asarray(ctx0.times("test"), dtype="datetime64[s]").astype(np.int64)
        ts_set = set(common_ts)
        mask = np.array([t in ts_set for t in ts0_sec])
        if len(common_scores) == mask.sum():
            r = evaluate_topk(common_scores, ctx0.y("test")[mask],
                              ctx0.retf("test")[mask], ctx0.times("test")[mask])
            print(f"\n  Score Stacking: acc={r['accuracy']:.4f}  "
                  f"tpd={r['trades_per_day']:.1f}  k={r['k']}")

    # ---- 4. 月度稳定性 ----
    print(f"\n  --- 月度稳定性 ---")
    for h in HORIZONS:
        r = single_metrics[h]
        months = sorted(r["acc_by_month"].keys())
        accs = [r["acc_by_month"][m] for m in months]
        min_acc = min(accs); min_m = months[accs.index(min_acc)]
        print(f"  H={h:2d} 整体={r['accuracy']:.4f} 最低月={min_acc:.4f}({min_m})  ", end="")
        for m in months:
            bar = "█" * int(r["acc_by_month"][m] * 40)
            print(f"\n    {m[-7:]} {r['acc_by_month'][m]:.4f} n={r['n_by_month'][m]:4d} {bar}", end="")
        print()

    # ---- 5. Union 月度稳定性 ----
    monthly_stats = {}
    for h in HORIZONS:
        r_full = evaluate_topk(results[h]["p"], results[h]["y"],
                               results[h]["retf"], results[h]["ts"], return_sel=True)
        sel = r_full["_sel"]
        ts_sel = np.asarray(results[h]["ts"][sel], dtype="datetime64[s]").astype(np.int64)
        months = _month_of_epoch(ts_sel.astype("datetime64[s]"))
        y_sel = results[h]["y"][sel]
        pred_sel = (results[h]["p"][sel] >= 0.5).astype(np.int8)
        for i in range(len(sel)):
            m = str(months[i])
            if m not in monthly_stats:
                monthly_stats[m] = {"k": 0, "correct": 0}
            monthly_stats[m]["k"] += 1
            monthly_stats[m]["correct"] += int(pred_sel[i] == y_sel[i])

    union_months = sorted(monthly_stats.keys())
    union_month_accs = []
    for m in union_months:
        st = monthly_stats[m]
        acc = st["correct"] / max(1, st["k"])
        union_month_accs.append(acc)
        bar = "█" * int(acc * 40)
        print(f"  Union {m[-7:]} acc={acc:.4f} k={st['k']:4d} {bar}")

    union_min = min(union_month_accs) if union_month_accs else 0
    union_min_m = union_months[union_month_accs.index(union_min)] if union_month_accs else "N/A"

    # ---- 6. 总结 ----
    print(f"\n  --- 总结 ---")
    print(f"  Union: {total_trades}单总, {total_tpd:.1f}单/天, 准确率{wavg_acc:.4f}")
    print(f"  Union 最低月: {union_min:.4f} ({union_min_m})")
    for h in HORIZONS:
        print(f"  H={h}: {single_metrics[h]['accuracy']:.4f}", end="")
    print()
    print(f"  {'✅ ALL PASS' if total_tpd>=30 and wavg_acc>=0.65 else '❌'}")


def stage_eval():
    """完整多周期评估: 对每个币种运行 _eval_symbol。"""
    for s in config.SYMBOLS:
        _eval_symbol(s)


# ============================================================
# 主入口
# ============================================================
def main():
    ap = argparse.ArgumentParser(description="simpletrade: 数据→训练→评估 全流程")
    ap.add_argument("--stage", default="full", choices=["data", "train", "multi", "eval", "full"])
    ap.add_argument("--symbols", nargs="*", default=None)
    args = ap.parse_args()
    symbols = args.symbols or config.SYMBOLS

    total_t0 = time.time()

    if args.stage in ("data", "full"):
        print(f"\n{'='*50}")
        print("  阶段1: 数据获取 + 构建数据集")
        print(f"{'='*50}")
        stage_data()

    if args.stage in ("train", "full"):
        print(f"\n{'='*50}")
        print("  阶段2: 单周期模型训练")
        print(f"{'='*50}")
        stage_train(symbols)

    if args.stage in ("multi", "full"):
        print(f"\n{'='*50}")
        print("  阶段3: 多周期训练")
        print(f"{'='*50}")
        stage_multi()

    if args.stage in ("eval", "full"):
        print(f"\n{'='*50}")
        print("  阶段4: 评估报告")
        print(f"{'='*50}")
        stage_eval()

    print(f"\n{'='*50}")
    print(f"  总耗时: {time.time() - total_t0:.0f}s")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()