#!/usr/bin/env bash
# 四周期信号相关性完整管线 - 断点续训, 每步写 status.json
set -e
PY=/root/.pyenv/versions/3.12.13/bin/python
LOG=/workspace/pipeline.log
STATUS=/workspace/results/pipeline_status.json
RESULTS=/workspace/results/full_analysis_result.json
cd /workspace
mkdir -p /workspace/results /workspace/models_saved /workspace/data/datasets
echo "===== PIPELINE STARTED $(date -Is) =====" | tee -a "$LOG"
trap 'echo "INTERRUPTED" | tee -a "$LOG"; exit 127' INT TERM

update_status() {
    cat > "$STATUS" <<ENDJSON
{"phase":"$1","progress":"$2","note":"$3","ts":"$(date -Is)"}
ENDJSON
}

# 0. 依赖
update_status "install_deps" "0%" "检查并安装依赖"
if ! $PY -c "import numpy, polars, pyarrow, pandas, lightgbm, xgboost, catboost, sklearn, scipy" 2>/dev/null; then
    echo "  [INSTALL] pip install" | tee -a "$LOG"
    $PY -m pip install -q numpy polars pyarrow pandas scikit-learn lightgbm xgboost catboost requests joblib scipy 2>&1 | tee -a "$LOG"
fi

# 1. fetch
update_status "fetch_data" "5%" "拉取 ETH/BTC 原始K线"
if [ ! -f /workspace/data/datasets/raw_ETH.parquet ] || [ ! -f /workspace/data/datasets/raw_BTC.parquet ]; then
    echo "  [RUN] fetch_data" | tee -a "$LOG"
    $PY data_processing/fetch_data.py 2>&1 | tee -a "$LOG"
fi

# 2. build
update_status "build_datasets" "10%" "构建 3/5/10/15min 数据集"
need_build=0
for H in 3 5 10 15; do for S in ETH BTC; do [ ! -f "/workspace/data/datasets/ds_${S}_h${H}.parquet" ] && need_build=1 && break 2; done; done
if [ $need_build -eq 1 ]; then
    $PY -m backtest_short.build_dataset_short --horizons 3 5 10 15 --symbols ETH BTC 2>&1 | tee -a "$LOG"
fi

# 3. train (断点续训: 已存在的模型文件自动跳过)
update_status "training" "15%" "训练 12 组模型"
DONE=0
for H in 3 5 10 15; do
    for F in lgb xgb cat; do
        ext={"lgb":"txt","xgb":"json","cat":"cbm"}[$F]
        mp="/workspace/models_saved/pool_short_h${H}/JOINT_${F}_seed42.${ext}"
        if [ -f "$mp" ]; then
            echo "  [SKIP] h${H} ${F} 已存在" | tee -a "$LOG"
        else
            update_status "train h${H} ${F}" "$((15 + DONE * 7))%" "训练 h${H} ${F} ${DONE}/12"
            echo "  [RUN] h${H} ${F} $(date -Is)" | tee -a "$LOG"
            $PY -m backtest_short.train_pool_short train $H $F 2>&1 | tee -a "$LOG"
            echo "  [DONE] h${H} ${F} $(date -Is)" | tee -a "$LOG"
        fi
        DONE=$((DONE + 1))
    done
done

# 4. analyze
update_status "analysis" "95%" "运行修正版因果信号分析"
$PY backtest_short/run_full_analysis.py 2>&1 | tee -a "$LOG"

echo "===== COMPLETE $(date -Is) =====" | tee -a "$LOG"
update_status "DONE" "100%" "完成, 结果在 $RESULTS"
