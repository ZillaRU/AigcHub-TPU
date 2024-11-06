#!/bin/bash
source hub_venv/bin/activate
export NO_ALBUMENTATIONS_UPDATE=1

# 提取参数列表，移除 --gradio 参数
ARGS=()
START_GRADIO=false

for arg in "$@"; do
    if [ "$arg" == "--gradio" ]; then
        START_GRADIO=true
    else
        ARGS+=("$arg")  # 只有当不是 --gradio 时才添加到 ARGS 数组
    fi
done

# 定义清理函数
cleanup() {
    echo "Cleaning up..."
    # 杀死所有子进程
    pkill -P $$
    exit 0
}

# 捕捉中断信号
trap 'cleanup' INT

# 启动 FastAPI 应用
python main_hub.py --host 0.0.0.0 --port 8000 "${ARGS[@]}" &

# 如果指定了 --gradio 参数，则启动 Gradio 应用
if [ "$START_GRADIO" == true ]; then
    python apps_gradio/base64convert.py &
fi

# 等待所有后台进程结束
wait
