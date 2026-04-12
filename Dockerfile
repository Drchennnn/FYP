# ── Stage 1: builder ──────────────────────────────────────────────────────────
# 用 slim 基础镜像安装依赖，生成 site-packages 后复制到最终镜像，减小层缓存失效范围
FROM python:3.10-slim AS builder

WORKDIR /install

# 系统依赖（lxml、h5py 需要编译工具）
RUN apt-get update && apt-get install -y --no-install-recommends --fix-missing \
    gcc g++ libhdf5-dev && \
    rm -rf /var/lib/apt/lists/*

# 只复制 requirements，利用 Docker 层缓存（依赖不变时跳过此步）
COPY requirements_docker.txt .

RUN pip install --no-cache-dir --prefix=/deps --timeout=300 --retries=5 -r requirements_docker.txt


# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.10-slim

WORKDIR /app

# 运行时系统依赖（libhdf5 运行时库）
RUN apt-get update && apt-get install -y --no-install-recommends \
    libhdf5-310 curl && \
    rm -rf /var/lib/apt/lists/*

# 从 builder 复制安装好的 Python 包
COPY --from=builder /deps /usr/local

# ── 复制项目代码 ──────────────────────────────────────────────────────────────
# CACHEBUST 用于强制让代码层缓存失效，不影响依赖层
ARG CACHEBUST=1
# 代码层（改动频繁，放最后以充分利用缓存）
COPY models/       ./models/
COPY realtime/     ./realtime/
COPY scripts/      ./scripts/
COPY web_app/      ./web_app/
COPY run_pipeline.py run_benchmark.py ./

# ── 复制运行时必要数据（小文件，直接打包进镜像）──────────────────────────────
# 训练数据 CSV（1.1MB）
COPY data/processed/jiuzhaigou_daily_features_2016-01-01_2026-04-02.csv \
     ./data/processed/
COPY data/processed/feature_metadata.json \
     ./data/processed/
# 节假日配置
COPY web_app/holidays.json ./web_app/

# 模型权重与推理结果（按需复制最新各一份）
# GRU 单步
COPY output/runs/gru_8features_20260406_172031/ \
     ./output/runs/gru_8features_20260406_172031/
# GRU MIMO
COPY output/runs/gru_mimo_8features_20260404_174122/ \
     ./output/runs/gru_mimo_8features_20260404_174122/
# LSTM 单步
COPY output/runs/lstm_8features_20260403_230759/ \
     ./output/runs/lstm_8features_20260403_230759/
# LSTM MIMO
COPY output/runs/lstm_mimo_8features_20260404_174313/ \
     ./output/runs/lstm_mimo_8features_20260404_174313/
# Seq2Seq+Attention
COPY output/runs/seq2seq_attention_8features_20260403_230801/ \
     ./output/runs/seq2seq_attention_8features_20260403_230801/

# MinMax scaler
COPY models/scalers/ ./models/scalers/

# ── 持久化目录（挂载外部 volume 使数据在容器重建后保留）────────────────────
# data/processed（append 会更新 CSV）
# data/append_log.json
# output/runs（月度重训产生新 run 目录）
# *.db（SQLite 数据库）
# 这些目录在 docker-compose.yml 里声明为 volume

# 创建挂载点目录（保证 volume 挂载时目录存在）
RUN mkdir -p /app/data/raw /app/data/processed \
             /app/output/runs \
             /app/models/scalers

# ── 环境变量 ──────────────────────────────────────────────────────────────────
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    TZ=Asia/Shanghai \
    # 关闭 TF 的 GPU 日志（纯 CPU 服务器）
    CUDA_VISIBLE_DEVICES="" \
    TF_CPP_MIN_LOG_LEVEL=2

WORKDIR /app

EXPOSE 5000

# 健康检查（每30秒一次，等30秒启动，失败3次标记不健康）
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:5000/api/scheduler/status || exit 1

CMD ["python", "web_app/app.py"]
