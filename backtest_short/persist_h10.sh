#!/bin/bash
# 训练 h10 三family(5-seed) -> 评估 -> 持久化关键产物到 git (防沙箱重置丢失)
PY=/root/.pyenv/versions/3.12.13/bin/python
cd /workspace

for F in lgb xgb cat; do
    echo "[TRAIN] h10 ${F} $(date -Is)"
    $PY -m backtest_short.train_pool_short train 10 $F 2>&1 | tail -8
    echo "[DONE] h10 ${F} $(date -Is)"
done

echo "[EVAL] $(date -Is)"
mkdir -p /workspace/results
$PY /workspace/backtest_short/diag3_why_drop.py 10 > /workspace/results/diag_final_h10.log 2>&1

echo "[PUSH] $(date -Is)"
# 允许跟踪: 模型 + P.npy + 评估日志 (parquet 太大不推, 恢复时 fetch+build 只需几分钟)
if grep -q "^data/datasets/" .gitignore 2>/dev/null; then :; else
    echo "data/datasets/raw_*.parquet" >> .gitignore
    echo "data/datasets/ds_*.parquet" >> .gitignore
fi
if ! grep -q "^!data/datasets/SHORT_.*_P.npy" .gitignore; then
    echo "!data/datasets/SHORT_*_P.npy" >> .gitignore
    echo "!models_saved/pool_short_h10/" >> .gitignore
    echo "!results/" >> .gitignore
fi
git add -f data/datasets/SHORT_*_P.npy models_saved/pool_short_h10/ results/diag_final_h10.log
git add .gitignore
git commit -m "p5: h10 final persisted (P.npy+models+eval) $(date -Is)" --allow-empty
timeout 120 git push origin HEAD 2>&1 | tail -2
echo "[ALL DONE] $(date -Is)"
