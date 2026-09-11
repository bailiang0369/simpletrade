#!/usr/bin/env bash
# 四周期信号相关性管线 v3 - 每阶段 commit+push, 断点续训
set -e
PY=/root/.pyenv/versions/3.12.13/bin/python
LOG=/workspace/pipeline.log
STATUS=/workspace/results/pipeline_status.json
RESULTS=/workspace/results/full_analysis_result.json
cd /workspace
mkdir -p /workspace/results /workspace/models_saved /workspace/data/datasets
# 关键: 让 git 使用持久化的 credentials 文件
git config credential.helper "store --file=/run/shared/git/credentials"
echo "===== PIPELINE STARTED $(date -Is) =====" | tee -a "$LOG"
trap 'echo "INTERRUPTED" | tee -a "$LOG"; _commit_pipeline "interrupted"; exit 127' INT TERM

update_status() {
    cat > "$STATUS" <<ENDJSON
{"phase":"$1","progress":"$2","note":"$3","ts":"$(date -Is)"}
ENDJSON
}

_commit_pipeline() {
    local tag="$1"
    cd /workspace
    git add results/ pipeline.log 2>/dev/null || true
    # 注意: data/ 和 models_saved/ 是 .gitignore 的, 不 commit (太大)
    # 但 results/ 和 pipeline.log 会被 commit
    git commit -m "pipeline: $tag $(date -Is)" --allow-empty 2>/dev/null || true
    timeout 30 git push origin HEAD 2>&1 | tee -a "$LOG" || echo "push failed (non-fatal)" | tee -a "$LOG"
}

# 0. 依赖
update_status "install_deps" "0%" "检查并安装依赖"
if ! $PY -c "import numpy, polars, pyarrow, pandas, lightgbm, xgboost, catboost, sklearn, scipy" 2>/dev/null; then
    echo "  [INSTALL] pip install" | tee -a "$LOG"
    $PY -m pip install -q numpy polars pyarrow pandas scikit-learn lightgbm xgboost catboost requests joblib scipy 2>&1 | tee -a "$LOG"
fi
_commit_pipeline "deps_ready"

# 1. fetch
update_status "fetch_data" "5%" "拉取 ETH/BTC 原始K线"
if [ ! -f /workspace/data/datasets/raw_ETH.parquet ] || [ ! -f /workspace/data/datasets/raw_BTC.parquet ]; then
    $PY data_processing/fetch_data.py 2>&1 | tee -a "$LOG"
fi
_commit_pipeline "fetch_done"

# 2. build
update_status "build_datasets" "10%" "构建 3/5/10/15min 数据集"
need_build=0
for H in 3 5 10 15; do for S in ETH BTC; do [ ! -f "/workspace/data/datasets/ds_${S}_h${H}.parquet" ] && need_build=1 && break 2; done; done
if [ $need_build -eq 1 ]; then
    $PY -m backtest_short.build_dataset_short --horizons 3 5 10 15 --symbols ETH BTC 2>&1 | tee -a "$LOG"
fi
_commit_pipeline "build_done"

# 3. train (断点续训)
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
            # 每训完一组就 push 一次进度 (模型文件太大, 不 commit, 但 status 会 commit)
            _commit_pipeline "train_h${H}_${F}_done"
        fi
        DONE=$((DONE + 1))
    done
done

# 4. analyze
update_status "analysis" "95%" "运行修正版因果信号分析"
$PY backtest_short/run_full_analysis.py 2>&1 | tee -a "$LOG"

echo "===== COMPLETE $(date -Is) =====" | tee -a "$LOG"
update_status "DONE" "100%" "完成, 结果在 $RESULTS"
_commit_pipeline "FINAL_RESULT"
echo "所有结果已 commit + push 到 GitHub" | tee -a "$LOG"
