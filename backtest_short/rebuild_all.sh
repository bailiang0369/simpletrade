#!/bin/bash
# 一键恢复+训练+评估+push (防沙箱重置丢失). 用法: bash backtest_short/rebuild_all.sh
# 流程: 装依赖(如缺) -> fetch -> build 3/5/15 -> 训练(跳过已有) -> 评估 -> push git
PY=/root/.pyenv/versions/3.12.13/bin/python
cd /workspace
git config --local user.name "trae-agent"
git config --local user.email "trae-agent@users.noreply.github.com"
git config --local credential.helper "store --file=/run/shared/git/credentials"

log() { echo "[$(date -Is)] $1"; }

log "STEP fetch"
$PY data_processing/fetch_data.py 2>&1 | tail -2

log "STEP build datasets 3/5/15"
$PY -m backtest_short.build_dataset_short --horizons 3 5 15 --symbols ETH BTC 2>&1 | grep -E "rows=" | tail -6

for H in 3 5 15; do
    for F in lgb xgb cat; do
        # 已有5-seed P则跳过
        if [ -f "/workspace/data/datasets/SHORT_ETH_h${H}_${F}_test_P.npy" ]; then
            DIM=$($PY -c "import numpy as np; print(np.load('/workspace/data/datasets/SHORT_ETH_h${H}_${F}_test_P.npy').ndim)")
            [ "$DIM" = "2" ] && { log "skip h${H} ${F} (5-seed已存在)"; continue; }
        fi
        log "TRAIN h${H} ${F}"
        $PY -m backtest_short.train_pool_short train $H $F 2>&1 | tail -3
        log "DONE h${H} ${F}"
    done
done

log "STEP eval 3/5/15"
mkdir -p /workspace/results
for H in 3 5 15; do
    $PY /workspace/backtest_short/diag3_why_drop.py $H > /workspace/results/diag_final_h${H}.log 2>&1
    log "eval h${H} -> results/diag_final_h${H}.log"
done

log "STEP push 持久化"
for H in 3 5 10 15; do
    git add -f data/datasets/SHORT_*_h${H}_*_P.npy 2>/dev/null
    git add -f models_saved/pool_short_h${H}/ 2>/dev/null
done
git add -f results/diag_final_h*.log backtest_short/rebuild_all.sh backtest_short/persist_h10.sh .gitignore
git commit -m "p5: h3/h5/h15 final persisted $(date -Is)" --allow-empty 2>&1 | tail -1
timeout 180 git push origin HEAD 2>&1 | tail -2
log "ALL DONE"
