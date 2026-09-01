"""单模型训练+预测子进程: 每个模型独立进程运行, 内存随进程退出完全释放,
避免 4GB cgroup 下多个模型状态+预测缓冲在单进程内叠加导致 OOM。

用法: BN_DATA_DIR=... python3 train_one.py --symbol BTCUSDT --model gbdt
输出:
  {DS_DIR}/{symbol}_{model}_pv.npy   meta_val 概率
  {DS_DIR}/{symbol}_{model}_pt.npy   test 概率
  模型文件保存到 MODEL_DIR
"""
import argparse
import gc
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from data_store import AssetContext
from models.gbdt import GBDTModel
from models.stat_signal import StatSignal
from models.seq_lstm import SeqGBDLSTM
from models.catboost_model import CatBoostModel
from models.faiss_shape import FaissShapeModel

MODEL_FACTORY = {
    "gbdt": GBDTModel,
    "stat": StatSignal,
    "lstm": SeqGBDLSTM,
    "catboost": CatBoostModel,
    "faiss": FaissShapeModel,
}

# 注: 只加载已实现的模型, 不加载已删除的 FAISS/CNN/DTW/TimeNorm 等废弃模块


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--model", required=True, choices=list(MODEL_FACTORY))
    ap.add_argument("--no-save-model", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    ctx = AssetContext(args.symbol)
    m = MODEL_FACTORY[args.model](seed=config.SEED)
    m.fit(ctx)
    pv = np.asarray(m.predict(ctx, "meta_val"), dtype=np.float32)
    pt = np.asarray(m.predict(ctx, "test"), dtype=np.float32)
    assert len(pv) == ctx.split_rows["meta_val"].sum()
    assert len(pt) == ctx.split_rows["test"].sum()
    np.save(f"{config.DS_DIR}/{args.symbol}_{args.model}_pv.npy", pv)
    np.save(f"{config.DS_DIR}/{args.symbol}_{args.model}_pt.npy", pt)
    if not args.no_save_model:
        try:
            m.save(os.path.join(config.MODEL_DIR, f"{args.symbol}_{args.model}"))
        except Exception as e:
            print(f"  (model save skipped {args.model}: {e})")
    del m, ctx
    gc.collect()
    print(f"[train_one] {args.symbol} {args.model} done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
