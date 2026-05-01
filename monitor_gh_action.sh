#!/bin/bash
# GitHub Action 训练实时监控脚本
# 自动拉取最新训练日志，失败立刻告警

REPO="fsz168/kline-yolo"
# 从环境变量读取GitHub Token，避免明文泄露
# 运行前执行：export GH_TOKEN="你的Token"
GH_TOKEN=${GH_TOKEN:-}

echo "🚀 开始监控GitHub Action训练状态..."
while true; do
    # 获取最新运行ID
    RUN_ID=$(curl -s -H "Authorization: token ${GH_TOKEN}" \
        "https://api.github.com/repos/${REPO}/actions/runs?status=in_progress" \
        | jq -r '.workflow_runs[0].id' 2>/dev/null)
    
    if [ "${RUN_ID}" != "null" ]; then
        echo "✅ 找到正在运行的训练任务，ID: ${RUN_ID}"
        echo "🔗 日志地址：https://github.com/${REPO}/actions/runs/${RUN_ID}"
        
        # 实时拉取日志
        while true; do
            STATUS=$(curl -s -H "Authorization: token ${GH_TOKEN}" \
                "https://api.github.com/repos/${REPO}/actions/runs/${RUN_ID}" \
                | jq -r '.status' 2>/dev/null)
            
            if [ "${STATUS}" == "completed" ]; then
                RESULT=$(curl -s -H "Authorization: token ${GH_TOKEN}" \
                    "https://api.github.com/repos/${REPO}/actions/runs/${RUN_ID}" \
                    | jq -r '.conclusion' 2>/dev/null)
                if [ "${RESULT}" == "success" ]; then
                    echo "🎉 训练成功！最优模型已上传到Release"
                else
                    echo "❌ 训练失败，请查看日志排查问题：https://github.com/${REPO}/actions/runs/${RUN_ID}"
                fi
                break
            fi
            # 每30秒刷新一次状态
            sleep 30
        done
    fi
    # 每2分钟检查一次是否有新训练任务启动
    sleep 120
done