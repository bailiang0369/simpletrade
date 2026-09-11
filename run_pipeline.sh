#!/usr/bin/env bash
# 快版管线 - 只训 seed42 (1/5 训练时间)
set -e
PY=/root/.pyenv/versions/3.12.13/bin/python
LOG=/workspace/pipeline.log
STATUS=/workspace/results/pipeline_status.json
cd /workspace
mkdir -p /workspace/results /workspace/models_saved /workspace/data/datasets
git config credential.helper "store --file=/run/shared/git/credentials"

echo "===== FAST PIPELINE STARTED $(date -Is) =====" | tee -a "$LOG"
trap 'echo "INTERRUPTED"; _commit "interrupted"; exit 127' INT TERM

update_status() {
    cat > "$STATUS" <<ENDJSON
{"phase":"$1","progress":"$2","note":"$3","ts":"$(date -Is)"}
ENDJSON
}
_commit() {
    cd /workspace
    git add results/ pipeline.log 2>/dev/null || true
    git commit -m "pipeline: $1 $(date -Is)" --allow-empty 2>/dev/null || true
    timeout 30 git push origin HEAD 2>&1 | tail -1 || true
}

# 0. deps
update_status "deps" "0%" "装依赖"
if ! $PY -c "import numpy, polars, lightgbm, xgboost, catboost, sklearn, scipy" 2>/dev/null; then
    $PY -m pip install -q numpy polars pyarrow pandas scikit-learn lightgbm xgboost catboost requests joblib scipy 2>&1 | tail -3 | tee -a "$LOG"
fi
_commit "deps_ready"

# 1. fetch
update_status "fetch" "5%" "拉数据"
if [ ! -f /workspace/data/datasets/raw_ETH.parquet ]; then
    $PY data_processing/fetch_data.py 2>&1 | tail -3 | tee -a "$LOG"
fi
_commit "fetch_done"

# 2. build
update_status "build" "10%" "建数据集"
need=0
for H in 3 5 10 15; do for S in ETH BTC; do [ ! -f "/workspace/data/datasets/ds_${S}_h${H}.parquet" ] && need=1 && break 2; done; done
if [ $need -eq 1 ]; then
    $PY -m backtest_short.build_dataset_short --horizons 3 5 10 15 --symbols ETH BTC 2>&1 | tail -5 | tee -a "$LOG"
fi
_commit "build_done"

# 3. train (fast: seed42 only)
update_status "train" "15%" "训练 12 组 seed42-only"
DONE=0
for H in 3 5 10 15; do
    for F in lgb xgb cat; do
        mp="/workspace/models_saved/pool_short_h${H}/JOINT_${F}_seed42.{'txt' if [ "$F" = "lgb" ] else 'json' if [ "$F" = "xgb" ] else 'cbm'}"
        if [ -f "$mp" ]; then
            echo "  [SKIP] h${H} ${F}" | tee -a "$LOG"
        else
            update_status "h${H}_${F}" "$((15 + DONE * 7))%" "训练 h${H} ${F} ($((DONE+1))/12)"
            echo "  [RUN] h${H} ${F} $(date -Is)" | tee -a "$LOG"
            $PY backtest_short/train_fast.py $H $F 2>&1 | tail -5 | tee -a "$LOG"
            echo "  [DONE] h${H} ${F} $(date -Is)" | tee -a "$LOG"
            _commit "train_h${H}_${F}_done"
        fi
        DONE=$((DONE + 1))
    done
done

# 4. analyze
update_status "analyze" "95%" "跑相关性分析"
$PY backtest_short/run_full_analysis.py 2>&1 | tee -a "$LOG"

echo "===== COMPLETE $(date -Is) =====" | tee -a "$LOG"
update_status "DONE" "100%" "完成"
_commit "FINAL_RESULT"
echo "===== 所有结果已 push 到 GitHub =====" | tee -a "$LOG"
