#!/bin/bash

# 确保提供了至少一个应用名称
if [ $# -eq 0 ]; then
    echo "Usage: \$0 app_name [app_name2 ...]"
    exit 1
fi

source hub_venv/bin/activate

# 定义存放仓库的目录
repo_dir="repo"
mkdir -p "$repo_dir"

mkdir -p /data/tmpdir

# 定义包含应用名称和GitHub链接的文件
apps_file="apps.txt"

# 遍历所有提供的应用名称
for app_name in "$@"; do
    echo "Processing $app_name..."

    # 从文件中查找与应用名称匹配的GitHub URL
    app_info=$(grep "^$app_name," "$apps_file")
    if [ -z "$app_info" ]; then
        echo "No URL found for $app_name in $apps_file"
        continue
    fi

    # 解析出GitHub URL
    IFS=',' read -r name url <<< "$app_info"
    url=$(echo $url | xargs) 
    echo "Found URL for $app_name: $url"

    # 如果已经有仓库， pull最新的仓库
    if [ -d "$repo_dir/$app_name/.git" ]; then
        cd "$repo_dir/$app_name"
        git pull
        cd - > /dev/null
    else
        #  clone 仓库的 aigchub 分支
        git clone -b aigchub --single-branch "$url" "$repo_dir/$app_name"
        if [ $? -ne 0 ]; then
            echo "Failed to clone $url"
            continue
        fi
    fi

    # 进入仓库目录
    cd "$repo_dir/$app_name"

    # 执行仓库中的脚本
    if [ -f "prepare.sh" ]; then
        echo "Running prepare.sh for $app_name..."
        chmod +x prepare.sh
        (export TMPDIR=/data/tmpdir; export PIP_CACHE_DIR=/data/tmpdir; ./prepare.sh)
    else
        echo "No prepare.sh found for $app_name"
    fi

    if [ -f "download.sh" ]; then
        echo "Running download.sh for $app_name..."
        chmod +x download.sh
        ./download.sh
    else
        echo "No download.sh found for $app_name"
    fi

    # 返回到原始目录
    cd - > /dev/null
done

echo "All specified apps processed."
