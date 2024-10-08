#!/bin/bash

# 检查 ~/ 目录下的可用空间是否超过 500MB
AVAILABLE_SPACE=$(df -m ~ | awk 'NR==2 {print $4}')
REQUIRED_SPACE=500

if [ "$AVAILABLE_SPACE" -lt "$REQUIRED_SPACE" ]; then
    echo "可用空间不足 500MB，请清理空间。"
    echo "建议执行以下命令清理空间："
    echo "sudo apt-get clean"
    echo "sudo apt-get autoclean"
    echo "sudo apt-get autoremove"
    echo "rm -rf ~/.cache/pip"
    exit 1
fi

if [ ! -d "hub_venv" ]; then
    # Linux
    sudo apt-get install python3-venv  # 如有不同版本的Python3,可指定具体版本venv：python3.5-venv
    python3 -m venv hub_venv
fi

mkdir -p /data/tmpdir
source hub_venv/bin/activate
export TMPDIR=/data/tmpdir # to avoid "no space left" error during pip package installation
pip3 install -r requirements.txt --cache-dir /data/tmpdir