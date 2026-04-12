#!/bin/bash
# 首次部署初始化脚本
# 功能：把镜像内的初始数据复制到宿主机 persistent/ 目录
# 用法：bash deploy/init_persistent.sh
# 注意：只在第一次部署时运行，后续重建容器不需要重新运行

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

echo "=== 初始化 persistent 目录 ==="

# 创建宿主机目录
mkdir -p persistent/data/processed
mkdir -p persistent/output/runs
mkdir -p persistent/models/scalers
mkdir -p persistent/db

# 用临时容器把镜像内的初始数据导出到宿主机
# （镜像已 build 完成后执行）
IMAGE="fyp-jiuzhaigou:latest"

echo "正在从镜像导出初始数据..."

docker run --rm \
  -v "$(pwd)/persistent:/persistent" \
  "$IMAGE" \
  sh -c "
    # 训练数据 CSV
    cp -n /app/data/processed/*.csv /persistent/data/processed/ 2>/dev/null || true
    cp -n /app/data/processed/*.json /persistent/data/processed/ 2>/dev/null || true

    # 模型权重和 runs 目录
    cp -rn /app/output/runs/. /persistent/output/runs/ 2>/dev/null || true

    # Scaler
    cp -n /app/models/scalers/*.pkl /persistent/models/scalers/ 2>/dev/null || true

    echo 'Done.'
  "

# 创建空的 append_log.json（如果不存在）
if [ ! -f persistent/data/append_log.json ]; then
  echo "[]" > persistent/data/append_log.json
fi

echo ""
echo "=== 初始化完成 ==="
echo "persistent/ 目录内容："
ls -lh persistent/data/processed/
ls -lh persistent/models/scalers/
echo ""
echo "现在可以运行：docker compose up -d"
