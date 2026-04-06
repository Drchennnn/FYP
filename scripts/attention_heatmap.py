#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Seq2Seq+Attention 热力图生成脚本

加载已训练的 Seq2Seq+Attention 模型权重，在测试集上运行推理，
提取 AttentionLayer 的权重矩阵并生成热力图。

输出（保存至 output/attention/<timestamp>/）：
  attention_heatmap_mean.png   — 测试集均值热力图 (7×30)
  attention_heatmap_sample.png — 最新测试样本热力图
  attention_heatmap_by_horizon.png — 每个预测步的注意力分布折线图
  attention_weights_mean.npy   — 均值权重矩阵，供论文进一步分析

用法：
  python scripts/attention_heatmap.py
  python scripts/attention_heatmap.py --weights output/runs/seq2seq_attention_8features_<ts>/...
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# ─────────────────────────────────────────────
# 常量（与训练脚本保持一致）
# ─────────────────────────────────────────────

ENCODER_STEPS = 30
DECODER_STEPS = 7

ENCODER_FEATURE_COLS = [
    "visitor_count_scaled",
    "month_norm",
    "day_of_week_norm",
    "is_holiday",
    "tourism_num_lag_7_scaled",
    "meteo_precip_sum_scaled",
    "temp_high_scaled",
    "temp_low_scaled",
]

DECODER_FEATURE_COLS = [
    "month_norm",
    "day_of_week_norm",
    "is_holiday",
    "tourism_num_lag_7_scaled",
    "meteo_precip_sum_scaled",
    "temp_high_scaled",
    "temp_low_scaled",
]


# ─────────────────────────────────────────────
# 数据加载
# ─────────────────────────────────────────────

def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)
    if "tourism_num" in df.columns:
        df["visitor_count"] = pd.to_numeric(df["tourism_num"], errors="coerce")
    df = df.dropna(subset=["visitor_count"]).reset_index(drop=True)
    # Recompute from raw integer columns (CSV stores NaN for most rows)
    if "month" in df.columns:
        df["month_norm"] = (df["month"] - 1) / 11.0
    if "day_of_week" in df.columns:
        df["day_of_week_norm"] = df["day_of_week"] / 6.0
    return df


def build_sequences(df: pd.DataFrame, scaler: MinMaxScaler,
                    test_ratio: float = 0.10):
    """Build encoder/decoder sequences and split into train/test."""
    n = len(df)
    train_size_raw = int(n * (1 - test_ratio))

    # Fit scaler on training portion only
    scaler.fit(df["visitor_count"].values[:train_size_raw].reshape(-1, 1))
    df["visitor_count_scaled"] = scaler.transform(
        df["visitor_count"].values.reshape(-1, 1)
    ).flatten()

    # Scaled lag-7 (recompute to match training)
    lag7 = df["visitor_count"].shift(7).fillna(0).values
    lag7_scaler = MinMaxScaler()
    lag7_scaler.fit(lag7[:train_size_raw].reshape(-1, 1))
    df["tourism_num_lag_7_scaled"] = lag7_scaler.transform(
        lag7.reshape(-1, 1)
    ).flatten()

    enc_vals = df[ENCODER_FEATURE_COLS].values.astype(np.float32)
    dec_vals = df[DECODER_FEATURE_COLS].values.astype(np.float32)
    target   = df["visitor_count_scaled"].values.astype(np.float32)
    dates    = df["date"].values

    total = ENCODER_STEPS + DECODER_STEPS
    X_enc, X_dec, y, d = [], [], [], []
    for i in range(total, len(df)):
        X_enc.append(enc_vals[i - total: i - DECODER_STEPS])
        X_dec.append(dec_vals[i - DECODER_STEPS: i])
        y.append(target[i - DECODER_STEPS: i])
        d.append(dates[i - DECODER_STEPS])

    X_enc = np.array(X_enc, dtype=np.float32)
    X_dec = np.array(X_dec, dtype=np.float32)
    y     = np.array(y,     dtype=np.float32)
    d     = np.array(d)

    n_seq    = len(X_enc)
    test_n   = int(n_seq * test_ratio)
    train_n  = n_seq - test_n

    return (X_enc[:train_n], X_dec[:train_n],
            X_enc[train_n:], X_dec[train_n:],
            y[train_n:], d[train_n:])


# ─────────────────────────────────────────────
# 模型加载
# ─────────────────────────────────────────────

def find_seq2seq_weights() -> Optional[Path]:
    runs_dir = PROJECT_ROOT / "output" / "runs"
    candidates = sorted(
        runs_dir.glob("seq2seq_attention_8features_*"),
        key=lambda p: p.stat().st_mtime, reverse=True,
    )
    for top_dir in candidates:
        for f in top_dir.rglob("weights/*.keras"):
            return f
        for f in top_dir.rglob("weights/*.h5"):
            return f
    return None


def load_model(weights_path: Path) -> tf.keras.Model:
    from models.lstm.train_seq2seq_attention_8features import (
        AttentionLayer, Seq2SeqWithAttention,
    )
    custom_objects = {
        "AttentionLayer": AttentionLayer,
        "Seq2SeqWithAttention": Seq2SeqWithAttention,
    }
    model = tf.keras.models.load_model(
        str(weights_path), custom_objects=custom_objects, compile=False
    )
    return model


# ─────────────────────────────────────────────
# 热力图生成
# ─────────────────────────────────────────────

def extract_attention(model, X_enc: np.ndarray, X_dec: np.ndarray,
                      batch_size: int = 64) -> np.ndarray:
    """Run inference and collect attention weights.
    Returns array of shape (n_test, DECODER_STEPS, ENCODER_STEPS).
    """
    all_attn = []
    for start in range(0, len(X_enc), batch_size):
        end = min(start + batch_size, len(X_enc))
        # Forward pass stores attention in model._last_attention
        model([X_enc[start:end], X_dec[start:end]], training=False)
        attn = model.get_attention_weights(
            [X_enc[start:end], X_dec[start:end]]
        )  # (batch, DECODER_STEPS, ENCODER_STEPS)
        all_attn.append(attn)
    return np.concatenate(all_attn, axis=0)


def plot_mean_heatmap(mean_attn: np.ndarray, out_dir: Path) -> Path:
    """Mean attention heatmap over the full test set (7 × 30)."""
    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(mean_attn, aspect="auto", cmap="YlOrRd",
                   interpolation="nearest", vmin=0)
    ax.set_xlabel("Encoder History (days before forecast start)", fontsize=10)
    ax.set_ylabel("Forecast Horizon", fontsize=10)
    ax.set_title(
        "Seq2Seq Attention Weight Heatmap\n"
        "Mean over Test Set — Jiuzhaigou Visitor Flow",
        fontsize=11,
    )
    # X-axis: t-30 … t-1
    ax.set_xticks(range(ENCODER_STEPS))
    ax.set_xticklabels(
        [f"t-{ENCODER_STEPS - i}" for i in range(ENCODER_STEPS)],
        fontsize=6, rotation=90,
    )
    # Y-axis: Day+1 … Day+7
    ax.set_yticks(range(DECODER_STEPS))
    ax.set_yticklabels([f"Day+{i + 1}" for i in range(DECODER_STEPS)], fontsize=9)
    plt.colorbar(im, ax=ax, label="Attention Weight")
    plt.tight_layout()
    save_path = out_dir / "attention_heatmap_mean.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path


def plot_sample_heatmap(sample_attn: np.ndarray, sample_date: str,
                        out_dir: Path) -> Path:
    """Single-sample attention heatmap (most recent test window)."""
    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(sample_attn, aspect="auto", cmap="YlOrRd",
                   interpolation="nearest", vmin=0)
    ax.set_xlabel("Encoder History (days before forecast start)", fontsize=10)
    ax.set_ylabel("Forecast Horizon", fontsize=10)
    ax.set_title(
        f"Seq2Seq Attention Weight Heatmap\n"
        f"Sample: forecast window starting {sample_date}",
        fontsize=11,
    )
    ax.set_xticks(range(ENCODER_STEPS))
    ax.set_xticklabels(
        [f"t-{ENCODER_STEPS - i}" for i in range(ENCODER_STEPS)],
        fontsize=6, rotation=90,
    )
    ax.set_yticks(range(DECODER_STEPS))
    ax.set_yticklabels([f"Day+{i + 1}" for i in range(DECODER_STEPS)], fontsize=9)
    plt.colorbar(im, ax=ax, label="Attention Weight")
    plt.tight_layout()
    save_path = out_dir / "attention_heatmap_sample.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path


def plot_by_horizon(mean_attn: np.ndarray, out_dir: Path) -> Path:
    """Line plot: attention distribution per forecast horizon."""
    fig, ax = plt.subplots(figsize=(12, 4))
    x = range(ENCODER_STEPS)
    colors = plt.cm.viridis(np.linspace(0, 1, DECODER_STEPS))
    for h in range(DECODER_STEPS):
        ax.plot(x, mean_attn[h], color=colors[h],
                label=f"Day+{h + 1}", linewidth=1.4, alpha=0.85)
    ax.set_xlabel("Encoder History Step (0 = t-30, 29 = t-1)", fontsize=10)
    ax.set_ylabel("Mean Attention Weight", fontsize=10)
    ax.set_title(
        "Attention Distribution by Forecast Horizon\n"
        "Seq2Seq+Attention — Jiuzhaigou Visitor Flow",
        fontsize=11,
    )
    ax.set_xticks(range(0, ENCODER_STEPS, 5))
    ax.set_xticklabels([f"t-{ENCODER_STEPS - i}" for i in range(0, ENCODER_STEPS, 5)],
                       fontsize=8)
    ax.legend(fontsize=8, ncol=4, loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path = out_dir / "attention_heatmap_by_horizon.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path


# ─────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Seq2Seq Attention Heatmap")
    parser.add_argument(
        "--input-csv",
        default="data/processed/jiuzhaigou_daily_features_2016-01-01_2026-04-02.csv",
    )
    parser.add_argument("--weights", default=None,
                        help="Seq2Seq .keras 权重路径（默认自动发现最新运行）")
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument("--output", default="output/attention")
    args = parser.parse_args()

    # ── 1. 权重 ──
    weights_path = Path(args.weights) if args.weights else find_seq2seq_weights()
    if weights_path is None or not weights_path.exists():
        print("错误：找不到 Seq2Seq 权重，请先运行训练或用 --weights 指定路径")
        sys.exit(1)
    print(f"加载模型权重: {weights_path}")
    model = load_model(weights_path)

    # ── 2. 数据 ──
    csv_path = PROJECT_ROOT / args.input_csv
    if not csv_path.exists():
        print(f"错误：找不到数据文件 {csv_path}")
        sys.exit(1)
    print(f"加载数据: {csv_path}")
    df = load_data(csv_path)
    print(f"数据范围: {df['date'].min().date()} ~ {df['date'].max().date()}，共 {len(df)} 行")

    scaler = MinMaxScaler()
    _, _, X_enc_test, X_dec_test, y_test, d_test = build_sequences(
        df, scaler, test_ratio=args.test_ratio
    )
    print(f"测试集: {len(X_enc_test)} 样本  "
          f"({pd.Timestamp(d_test[0]).date()} ~ {pd.Timestamp(d_test[-1]).date()})")

    # ── 3. 提取 Attention 权重 ──
    print("提取 Attention 权重...")
    attn_all = extract_attention(model, X_enc_test, X_dec_test)
    # attn_all: (n_test, DECODER_STEPS, ENCODER_STEPS)
    print(f"Attention 矩阵形状: {attn_all.shape}")

    mean_attn   = attn_all.mean(axis=0)   # (7, 30)
    sample_attn = attn_all[-1]            # most recent test window
    try:
        sample_date = str(pd.Timestamp(d_test[-1]).date())
    except Exception:
        sample_date = "latest"

    # ── 4. 保存 ──
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = PROJECT_ROOT / args.output / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(str(out_dir / "attention_weights_mean.npy"), mean_attn)
    np.save(str(out_dir / "attention_weights_all.npy"),  attn_all)

    p1 = plot_mean_heatmap(mean_attn, out_dir)
    p2 = plot_sample_heatmap(sample_attn, sample_date, out_dir)
    p3 = plot_by_horizon(mean_attn, out_dir)

    print(f"\n均值热力图:    {p1}")
    print(f"样本热力图:    {p2}")
    print(f"按预测步分布:  {p3}")
    print(f"\n结果已保存至: {out_dir}")

    # ── 5. 打印摘要 ──
    print(f"\n{'='*50}")
    print("Attention 权重摘要（均值热力图）")
    print(f"{'='*50}")
    print(f"{'Horizon':<10}", end="")
    for i in [0, 4, 9, 14, 19, 24, 29]:
        print(f"  t-{ENCODER_STEPS - i:<3}", end="")
    print()
    for h in range(DECODER_STEPS):
        print(f"Day+{h+1:<6}", end="")
        for i in [0, 4, 9, 14, 19, 24, 29]:
            print(f"  {mean_attn[h, i]:.4f}", end="")
        print()
    top_step = int(mean_attn.mean(axis=0).argmax())
    print(f"\n最受关注的历史时间步: t-{ENCODER_STEPS - top_step} "
          f"(index {top_step}, weight={mean_attn.mean(axis=0)[top_step]:.4f})")


if __name__ == "__main__":
    main()
