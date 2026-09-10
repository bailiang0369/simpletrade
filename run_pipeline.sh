#!/usr/bin/env bash
# 四周期信号相关性完整管线 (自包含版, 自动装依赖, 断点续训)
# nohup bash run_pipeline.sh > /workspace/pipeline.log 2>&1 &
set -e

PY=/root/.pyenv/versions/3.12.13/bin/python
LOG=/workspace/pipeline.log
STATUS=/workspace/results/pipeline_status.json
cd /workspace

mkdir -p /workspace/results /workspace/models_saved /workspace/data/datasets

echo "===== PIPELINE STARTED $(date -Is) =====" | tee -a "$LOG"

update_status() {
    cat > "$STATUS" <<ENDJSON
{"phase":"$1","progress":"$2","note":"$3","started":"$(date -Is)","finished":"","exit_code":"0"}
ENDJSON
}

finish_status() {
    cat > "$STATUS" <<ENDJSON
{"phase":"DONE","progress":"100%","note":"Pipeline finished with exit code $1","started":"","finished":"$(date -Is)","exit_code":"$1"}
ENDJSON
}

trap 'echo "===== PIPELINE INTERRUPTED $(date -Is) =====" | tee -a "$LOG"; finish_status 127; exit 127' INT TERM

# ===== STEP 0: ENSURE DEPS =====
update_status "install_deps" "0%" "检查并安装依赖"
echo "===== STEP 0: DEPENDENCIES $(date -Is) =====" | tee -a "$LOG"
if ! $PY -c "import numpy, polars, pyarrow, pandas, lightgbm, xgboost, catboost, sklearn, scipy" 2>/dev/null; then
    echo "  [INSTALL] 安装 Python 依赖..." | tee -a "$LOG"
    $PY -m pip install -q numpy polars pyarrow pandas scikit-learn lightgbm xgboost catboost requests joblib scipy 2>&1 | tee -a "$LOG"
else
    echo "  [OK] 依赖已存在" | tee -a "$LOG"
fi

# ===== STEP 1: FETCH DATA =====
update_status "fetch_data" "5%" "拉取 ETH/BTC 原始K线"
echo "===== STEP 1: FETCH DATA $(date -Is) =====" | tee -a "$LOG"
if [ -f /workspace/data/datasets/raw_ETH.parquet ] && [ -f /workspace/data/datasets/raw_BTC.parquet ]; then
    echo "  [SKIP] raw_*.parquet already exist" | tee -a "$LOG"
else
    $PY data_processing/fetch_data.py 2>&1 | tee -a "$LOG"
fi

# ===== STEP 2: BUILD DATASETS =====
update_status "build_datasets" "10%" "构建 3/5/10/15min 数据集"
echo "===== STEP 2: BUILD DATASETS $(date -Is) =====" | tee -a "$LOG"
DATASETS_NEED=0
for H in 3 5 10 15; do
    for SYM in ETH BTC; do
        if [ ! -f "/workspace/data/datasets/ds_${SYM}_h${H}.parquet" ]; then
            DATASETS_NEED=1; break 2
        fi
    done
done
if [ $DATASETS_NEED -eq 1 ]; then
    $PY -m backtest_short.build_dataset_short --horizons 3 5 10 15 --symbols ETH BTC 2>&1 | tee -a "$LOG"
else
    echo "  [SKIP] all datasets exist" | tee -a "$LOG"
fi

# ===== STEP 3: TRAIN ALL MODELS =====
update_status "training" "15%" "训练 4 周期 × 3 family 共 12 组 (断点续训)"
echo "===== STEP 3: TRAINING $(date -Is) =====" | tee -a "$LOG"

DONE=0
for H in 3 5 10 15; do
    for F in lgb xgb cat; do
        PCT=$((15 + DONE * 7))
        update_status "training h${H} ${F}" "$PCT%" "训练 h${H} ${F} (${DONE}/12)"
        echo "  ---- h${H} ${F} START $(date -Is) ----" | tee -a "$LOG"
        $PY -m backtest_short.train_pool_short train $H $F 2>&1 | tee -a "$LOG"
        echo "  ---- h${H} ${F} DONE $(date -Is) ----" | tee -a "$LOG"
        DONE=$((DONE + 1))
    done
done

# ===== STEP 4: CORRECTED ANALYSIS =====
update_status "analysis" "95%" "运行修正版因果信号分析 + 相关性 + 准确率对比"
echo "===== STEP 4: CORRECTED ANALYSIS $(date -Is) =====" | tee -a "$LOG"
$PY backtest_short/run_full_analysis.py 2>&1 | tee -a "$LOG"

echo "===== PIPELINE COMPLETE $(date -Is) =====" | tee -a "$LOG"
echo "结果在 /workspace/results/full_analysis_result.json" | tee -a "$LOG"
echo "查看进度: tail -f /workspace/pipeline.log" | tee -a "$LOG"
echo "查看状态: cat /workspace/results/pipeline_status.json" | tee -a "$LOG"
finish_status 0
