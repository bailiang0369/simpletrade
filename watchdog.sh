#!/bin/bash
# 盯管线跑完后, 把结果 commit + push 到 git (持久化!)
RESULTS=/workspace/results/full_analysis_result.json
STATUS=/workspace/results/pipeline_status.json
LOG=/workspace/watchdog.log

echo "watchdog started $(date -Is)" > "$LOG"

# 等管线跑完
for i in $(seq 1 360); do  # 最多 1 小时
    if [ -f "$STATUS" ]; then
        PHASE=$(/root/.pyenv/versions/3.12.13/bin/python -c "import json; print(json.load(open('$STATUS'))['phase'])" 2>/dev/null)
        echo "  [$i] phase=$PHASE $(date -Is)" >> "$LOG"
        if [ "$PHASE" = "DONE" ] && [ -f "$RESULTS" ]; then
            echo "PIPELINE DONE! committing..." >> "$LOG"
            break
        fi
    fi
    sleep 60
done

# 如果管线进程不在了但结果也不在, 报告错误
if ! ps -p 2089 > /dev/null 2>&1 && [ ! -f "$RESULTS" ]; then
    echo "ERROR: pipeline died before producing results" >> "$LOG"
    exit 1
fi

# Commit + push 结果
cd /workspace
git add -A results/ pipeline.log results/pipeline_status.json 2>/dev/null
git commit -m "pipeline-results $(date -Is)" 2>/dev/null
git push origin HEAD 2>/dev/null
echo "push exit code: $?" >> "$LOG"
echo "watchdog done $(date -Is)" >> "$LOG"
