#!/bin/bash

if [ ! -d "hub_venv" ]; then
    # Linux
    sudo apt-get install python3-venv  # 如有不同版本的Python3,可指定具体版本venv：python3.5-venv
    python3 -m venv hub_venv
fi

mkdir -p /data/tmpdir
source hub_venv/bin/activate
export TMPDIR=/data/tmpdir # to avoid "no space left" error during pip package installation
pip3 install -r requirements.txt