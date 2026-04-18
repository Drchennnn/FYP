#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Walk-forward Evaluation（滚动窗口评估）

方法论：
  使用 expanding window（扩展窗口）策略，模拟真实部署场景：
  - GRU / LSTM：每个 fold 独立从头训练，评估模型在不同时间段的泛化能力
  - Seq2Seq+Attention：加载已训练权重，在各 fold 测试集上直接推理（不重训）
    原因：Seq2Seq 训练时间长（>30分钟/fold），且其多步直接输出特性使其
    在不同时间段的泛化能力评估更适合用固定权重推理。

与固定切分的区别：
  固定切分只在一个时间段（最后10%）测试，无法反映模型在不同季节、
  不同年份的稳定性。Walk-forward 覆盖多个时间段，包括春节、五一、
  国庆等高峰期，评估结果更贴近真实部署。

用法：
  python scripts/walk_forward_eval.py --model gru --folds 4
  python scripts/walk_forward_eval.py --model lstm --folds 4
  python scripts/walk_forward_eval.py --model seq2seq_attention --folds 4
  python scripts/walk_forward_eval.py --model all --folds 4
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# CJK 字体配置
def _setup_cjk_font():
    candidates = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS',
                  'WenQuanYi Micro Hei', 'Noto Sans CJK SC']
    for name in candidates:
        if any(name.lower() in f.name.lower() for f in fm.fontManager.ttflist):
            plt.rcParams['font.family'] = name
            plt.rcParams['axes.unicode_minus'] = False
            return
    plt.rcParams['axes.unicode_minus'] = False

_setup_cjk_font()

import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

from models.common.core_evaluation import (
    DEFAULT_PEAK_THRESHOLD,
    compute_core_metrics,
    compute_weather_thresholds_quantile,
)


# ─────────────────────────────────────────────
# 数据加载
# ─────────────────────────────────────────────

def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)
    if "tourism_num" in df.columns:
        df["visitor_count"] = pd.to_numeric(df["tourism_num"], errors="coerce")
    elif "visitor_count" not in df.columns:
        raise ValueError("CSV 中未找到 tourism_num 或 visitor_count 列")
    df = df.dropna(subset=["visitor_count"]).reset_index(drop=True)
    # Recompute month_norm and day_of_week_norm from raw columns (same as training script).
    # The CSV stores these as NaN for most rows; the raw integer columns are always valid.
    if "month" in df.columns:
        df["month_norm"] = (df["month"] - 1) / 11.0
    if "day_of_week" in df.columns:
        df["day_of_week_norm"] = df["day_of_week"] / 6.0
    return df


FEATURE_COLS = [
    "visitor_count_scaled",
    "month_norm",
    "day_of_week_norm",
    "is_holiday",
    "tourism_num_lag_7_scaled",
    "meteo_precip_sum_scaled",
    "temp_high_scaled",
    "temp_low_scaled",
]

# Seq2Seq decoder features (7 external features, no visitor_count)
DECODER_FEATURE_COLS = [
    "month_norm",
    "day_of_week_norm",
    "is_holiday",
    "tourism_num_lag_7_scaled",
    "meteo_precip_sum_scaled",
    "temp_high_scaled",
    "temp_low_scaled",
]

ENCODER_STEPS = 30
DECODER_STEPS = 7  # Seq2Seq predicts 7 days at once

WEATHER_COLS_RAW = ["meteo_precip_sum", "meteo_temp_max", "meteo_temp_min"]


def _check_features(df: pd.DataFrame) -> None:
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"缺少特征列: {missing}")


# ─────────────────────────────────────────────
# 序列构建
# ─────────────────────────────────────────────

def build_sequences(
    df: pd.DataFrame,
    look_back: int = 30,
) -> tuple:
    """构建 (X, y, dates) 序列，X shape = (N, look_back, 8)"""
    _check_features(df)
    values = df[FEATURE_COLS].values.astype(np.float32)
    target = df["visitor_count_scaled"].values.astype(np.float32)
    dates = df["date"].values

    X, y, d = [], [], []
    for i in range(look_back, len(df)):
        X.append(values[i - look_back: i])
        y.append(target[i])
        d.append(dates[i])
    return np.array(X), np.array(y), np.array(d)


def build_seq2seq_sequences(df: pd.DataFrame) -> tuple:
    """构建 Seq2Seq 序列。
    Returns:
        X_enc: (N, 30, 8)
        X_dec: (N, 7, 7)
        y:     (N, 7)   — 未来7天 visitor_count_scaled
        d:     (N,)     — 每个样本对应的预测起始日期
    """
    _check_features(df)
    enc_vals = df[FEATURE_COLS].values.astype(np.float32)
    dec_vals = df[DECODER_FEATURE_COLS].values.astype(np.float32)
    target = df["visitor_count_scaled"].values.astype(np.float32)
    dates = df["date"].values

    total = ENCODER_STEPS + DECODER_STEPS
    X_enc, X_dec, y, d = [], [], [], []
    for i in range(total, len(df)):
        X_enc.append(enc_vals[i - total: i - DECODER_STEPS])   # (30, 8)
        X_dec.append(dec_vals[i - DECODER_STEPS: i])           # (7, 7)
        y.append(target[i - DECODER_STEPS: i])                 # (7,)
        d.append(dates[i - DECODER_STEPS])                     # start of forecast window
    return (np.array(X_enc), np.array(X_dec),
            np.array(y), np.array(d))


def find_seq2seq_weights() -> Optional[Path]:
    """找最新的 seq2seq_attention_8features 权重文件。"""
    runs_dir = PROJECT_ROOT / "output" / "runs"
    candidates = sorted(
        runs_dir.glob("seq2seq_attention_8features_*"),
        key=lambda p: p.stat().st_mtime, reverse=True
    )
    for top_dir in candidates:
        for f in top_dir.rglob("weights/*.keras"):
            return f
        for f in top_dir.rglob("weights/*.h5"):
            return f
    return None


def load_seq2seq_model(weights_path: Path) -> tf.keras.Model:
    """加载 Seq2Seq+Attention 模型（需要注册自定义层）。"""
    sys.path.insert(0, str(PROJECT_ROOT))
    from models.lstm.train_seq2seq_attention_8features import (
        AttentionLayer, Seq2SeqWithAttention
    )
    custom_objects = {
        "AttentionLayer": AttentionLayer,
        "Seq2SeqWithAttention": Seq2SeqWithAttention,
    }
    return tf.keras.models.load_model(str(weights_path),
                                      custom_objects=custom_objects,
                                      compile=False)


# ─────────────────────────────────────────────
# 模型构建
# ─────────────────────────────────────────────

def build_gru(look_back: int) -> tf.keras.Model:
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(look_back, 8)),
        tf.keras.layers.GRU(128, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.GRU(64),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=tf.keras.losses.Huber())
    return model


def build_lstm(look_back: int) -> tf.keras.Model:
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(look_back, 8)),
        tf.keras.layers.LSTM(128, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=tf.keras.losses.Huber())
    return model


# ─────────────────────────────────────────────
# 单 fold 训练 + 评估
# ─────────────────────────────────────────────

def run_fold(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    model_type: str,
    look_back: int,
    epochs: int,
    batch_size: int,
    fold_idx: int,
    peak_threshold: float,
) -> Dict:
    """训练一个 fold 并返回评估指标字典"""

    # 用训练集拟合 scaler（防止数据泄露）
    scaler = MinMaxScaler()
    train_counts = df_train["visitor_count"].values.reshape(-1, 1)
    scaler.fit(train_counts)

    # 将 visitor_count_scaled 写入 df（仅在本 fold 内有效）
    df_train = df_train.copy()
    df_test = df_test.copy()
    df_train["visitor_count_scaled"] = scaler.transform(
        df_train["visitor_count"].values.reshape(-1, 1)
    ).flatten()
    df_test["visitor_count_scaled"] = scaler.transform(
        df_test["visitor_count"].values.reshape(-1, 1)
    ).flatten()

    # 构建序列
    X_train, y_train, _ = build_sequences(df_train, look_back)
    X_test, y_test, d_test = build_sequences(df_test, look_back)

    if len(X_train) < 50 or len(X_test) < 10:
        print(f"  Fold {fold_idx}: 数据不足，跳过 (train={len(X_train)}, test={len(X_test)})")
        return {}

    # 验证集：训练集最后 12.5%
    val_size = max(10, int(len(X_train) * 0.125))
    X_val, y_val = X_train[-val_size:], y_train[-val_size:]
    X_train, y_train = X_train[:-val_size], y_train[:-val_size]

    # 构建模型
    tf.random.set_seed(42 + fold_idx)
    if model_type == "gru":
        model = build_gru(look_back)
    else:
        model = build_lstm(look_back)

    cb = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=15, restore_best_weights=True, verbose=0
        )
    ]
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=cb,
        verbose=0,
    )

    # 预测 + 反归一化
    y_pred_scaled = model.predict(X_test, verbose=0).flatten()
    y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_true = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    # 天气阈值（仅用训练集计算）
    weather_thresholds = None
    weather_precip = np.full(len(y_true), np.nan)
    weather_temp_high = np.full(len(y_true), np.nan)
    weather_temp_low = np.full(len(y_true), np.nan)

    if all(c in df_train.columns for c in WEATHER_COLS_RAW):
        weather_thresholds = compute_weather_thresholds_quantile(
            train_precip=df_train["meteo_precip_sum"].values,
            train_temp_high=df_train["meteo_temp_max"].values,
            train_temp_low=df_train["meteo_temp_min"].values,
        )
        # 对齐测试集天气（去掉 look_back 偏移）
        test_weather = df_test.iloc[look_back:].reset_index(drop=True)
        n = min(len(test_weather), len(y_true))
        weather_precip[:n] = test_weather["meteo_precip_sum"].values[:n]
        weather_temp_high[:n] = test_weather["meteo_temp_max"].values[:n]
        weather_temp_low[:n] = test_weather["meteo_temp_min"].values[:n]

    # 核心指标
    metrics, _by_h = compute_core_metrics(
        y_true=y_true,
        y_pred=y_pred,
        peak_threshold=peak_threshold,
        weather_precip=weather_precip,
        weather_temp_high=weather_temp_high,
        weather_temp_low=weather_temp_low,
        weather_thresholds=weather_thresholds,
    )

    result = {
        "fold": fold_idx,
        "train_size": len(X_train) + val_size,
        "test_size": len(X_test),
        "test_start": str(pd.Timestamp(d_test[0]).date()),
        "test_end": str(pd.Timestamp(d_test[-1]).date()),
        "mae": metrics["regression"]["mae"],
        "rmse": metrics["regression"]["rmse"],
        "smape": metrics["regression"]["smape"],
        "crowd_f1": metrics["crowd_alert"]["f1"],
        "crowd_recall": metrics["crowd_alert"]["recall"],
        "crowd_precision": metrics["crowd_alert"]["precision"],
        "suitability_f1": metrics["suitability_warning"]["f1"],
        "suitability_recall": metrics["suitability_warning"]["recall"],
        "suitability_precision": metrics["suitability_warning"]["precision"],
        "brier": metrics["suitability_warning"]["brier"],
        "ece": metrics["suitability_warning"]["ece"],
        "expected_cost": metrics["suitability_warning"]["expected_cost"],
    }

    print(
        f"  Fold {fold_idx} [{result['test_start']} ~ {result['test_end']}] "
        f"MAE={result['mae']:.0f}  SMAPE={result['smape']:.1f}%  "
        f"Suit-F1={result['suitability_f1']:.3f}  Suit-Recall={result['suitability_recall']:.3f}"
    )
    return result


def run_fold_seq2seq(
    df_test: pd.DataFrame,
    model: tf.keras.Model,
    scaler: MinMaxScaler,
    fold_idx: int,
    peak_threshold: float,
    weather_thresholds,
) -> Dict:
    """Seq2Seq fold 评估（固定权重推理，不重训）。
    df_test 包含 look_back 前缀行（用于构建 encoder 窗口）。
    """
    df_test = df_test.copy()
    df_test["visitor_count_scaled"] = scaler.transform(
        df_test["visitor_count"].values.reshape(-1, 1)
    ).flatten()

    X_enc, X_dec, y_seq, d_seq = build_seq2seq_sequences(df_test)
    if len(X_enc) == 0:
        print(f"  Fold {fold_idx}: Seq2Seq 数据不足，跳过")
        return {}

    # Predict: output shape (N, 7, 1) or (N, 7)
    raw_out = model.predict([X_enc, X_dec], verbose=0)
    if raw_out.ndim == 3:
        raw_out = raw_out[:, :, 0]   # (N, 7)

    # Flatten 7-step predictions and targets for single-step metric computation
    y_pred_flat = scaler.inverse_transform(
        raw_out.reshape(-1, 1)
    ).flatten()
    y_true_flat = scaler.inverse_transform(
        y_seq.reshape(-1, 1)
    ).flatten()

    # Dates: repeat each start date 7 times to align with flattened arrays
    d_test_start = str(pd.Timestamp(d_seq[0]).date()) if len(d_seq) > 0 else "?"
    d_test_end   = str(pd.Timestamp(d_seq[-1]).date()) if len(d_seq) > 0 else "?"

    # Weather (align to flattened test window)
    n_flat = len(y_true_flat)
    weather_precip   = np.full(n_flat, np.nan)
    weather_temp_high = np.full(n_flat, np.nan)
    weather_temp_low  = np.full(n_flat, np.nan)
    wx_cols = ["meteo_precip_sum", "meteo_temp_max", "meteo_temp_min"]
    if all(c in df_test.columns for c in wx_cols):
        # Skip the encoder prefix rows
        wx_df = df_test.iloc[ENCODER_STEPS:].reset_index(drop=True)
        n_w = min(len(wx_df), n_flat)
        weather_precip[:n_w]    = wx_df["meteo_precip_sum"].values[:n_w]
        weather_temp_high[:n_w] = wx_df["meteo_temp_max"].values[:n_w]
        weather_temp_low[:n_w]  = wx_df["meteo_temp_min"].values[:n_w]

    metrics, _ = compute_core_metrics(
        y_true=y_true_flat,
        y_pred=y_pred_flat,
        peak_threshold=peak_threshold,
        weather_precip=weather_precip,
        weather_temp_high=weather_temp_high,
        weather_temp_low=weather_temp_low,
        weather_thresholds=weather_thresholds,
    )

    result = {
        "fold": fold_idx,
        "train_size": 0,   # no retraining
        "test_size": len(y_true_flat),
        "test_start": d_test_start,
        "test_end": d_test_end,
        "mae": metrics["regression"]["mae"],
        "rmse": metrics["regression"]["rmse"],
        "smape": metrics["regression"]["smape"],
        "crowd_f1": metrics["crowd_alert"]["f1"],
        "crowd_recall": metrics["crowd_alert"]["recall"],
        "crowd_precision": metrics["crowd_alert"]["precision"],
        "suitability_f1": metrics["suitability_warning"]["f1"],
        "suitability_recall": metrics["suitability_warning"]["recall"],
        "suitability_precision": metrics["suitability_warning"]["precision"],
        "brier": metrics["suitability_warning"]["brier"],
        "ece": metrics["suitability_warning"]["ece"],
        "expected_cost": metrics["suitability_warning"]["expected_cost"],
    }
    print(
        f"  Fold {fold_idx} [{result['test_start']} ~ {result['test_end']}] "
        f"MAE={result['mae']:.0f}  SMAPE={result['smape']:.1f}%  "
        f"Suit-F1={result['suitability_f1']:.3f}  Suit-Recall={result['suitability_recall']:.3f}"
    )
    return result


# ─────────────────────────────────────────────
# Walk-forward 主逻辑
# ─────────────────────────────────────────────

def walk_forward_eval(
    df: pd.DataFrame,
    model_type: str,
    n_folds: int,
    test_window: int,
    look_back: int,
    epochs: int,
    batch_size: int,
    peak_threshold: float,
    out_dir: Path,
) -> pd.DataFrame:
    """
    Expanding-window walk-forward evaluation.

    数据切分示意（n_folds=4, test_window=90）：
      Fold 1: train=[0, N-4*90)  test=[N-4*90, N-3*90)
      Fold 2: train=[0, N-3*90)  test=[N-3*90, N-2*90)
      Fold 3: train=[0, N-2*90)  test=[N-2*90, N-1*90)
      Fold 4: train=[0, N-1*90)  test=[N-1*90, N)
    """
    n = len(df)
    min_train = look_back + 200  # 最少训练样本

    # Seq2Seq: load pre-trained weights once, reuse across all folds
    seq2seq_model = None
    seq2seq_scaler = None
    if model_type == "seq2seq_attention":
        weights_path = find_seq2seq_weights()
        if weights_path is None:
            print("错误：找不到 Seq2Seq 模型权重，请先运行训练")
            return pd.DataFrame()
        print(f"  加载 Seq2Seq 权重: {weights_path}")
        seq2seq_model = load_seq2seq_model(weights_path)
        # Fit scaler on full training portion (first 90% of data)
        train_end = n - n_folds * test_window
        seq2seq_scaler = MinMaxScaler()
        seq2seq_scaler.fit(df["visitor_count"].values[:train_end].reshape(-1, 1))
        # Compute weather thresholds from training portion
        seq2seq_weather_thr = None
        wx_cols = ["meteo_precip_sum", "meteo_temp_max", "meteo_temp_min"]
        if all(c in df.columns for c in wx_cols):
            from models.common.core_evaluation import compute_weather_thresholds_quantile
            seq2seq_weather_thr = compute_weather_thresholds_quantile(
                train_precip=df["meteo_precip_sum"].values[:train_end],
                train_temp_high=df["meteo_temp_max"].values[:train_end],
                train_temp_low=df["meteo_temp_min"].values[:train_end],
            )

    results = []
    for fold in range(n_folds):
        # 从后往前切割
        test_end = n - fold * test_window
        test_start = test_end - test_window
        if test_start < min_train:
            print(f"  Fold {n_folds - fold}: 训练集不足 {min_train} 行，停止")
            break

        df_train = df.iloc[:test_start].copy()
        # For Seq2Seq, include encoder prefix before test window
        prefix = ENCODER_STEPS + DECODER_STEPS
        df_test = df.iloc[max(0, test_start - prefix): test_end].copy()

        fold_idx = n_folds - fold  # 从小到大编号
        print(f"\nFold {fold_idx}/{n_folds}  train=[0, {test_start})  test=[{test_start}, {test_end})")

        if model_type == "seq2seq_attention":
            r = run_fold_seq2seq(
                df_test=df_test,
                model=seq2seq_model,
                scaler=seq2seq_scaler,
                fold_idx=fold_idx,
                peak_threshold=peak_threshold,
                weather_thresholds=seq2seq_weather_thr,
            )
        else:
            df_test_gru = df.iloc[test_start - look_back: test_end].copy()
            r = run_fold(
                df_train=df_train,
                df_test=df_test_gru,
                model_type=model_type,
                look_back=look_back,
                epochs=epochs,
                batch_size=batch_size,
                fold_idx=fold_idx,
                peak_threshold=peak_threshold,
            )
        if r:
            results.append(r)

    if not results:
        print("警告：没有有效的 fold 结果")
        return pd.DataFrame()

    # 按 fold 排序
    results = sorted(results, key=lambda x: x["fold"])
    df_res = pd.DataFrame(results)

    # 汇总统计
    metric_cols = ["mae", "rmse", "smape", "crowd_f1", "crowd_recall",
                   "suitability_f1", "suitability_recall", "brier", "ece", "expected_cost"]
    summary = {}
    for c in metric_cols:
        if c in df_res.columns:
            summary[f"{c}_mean"] = float(df_res[c].mean())
            summary[f"{c}_std"] = float(df_res[c].std())

    # 保存结果
    out_dir.mkdir(parents=True, exist_ok=True)
    df_res.to_csv(out_dir / "walk_forward_folds.csv", index=False)
    with open(out_dir / "walk_forward_summary.json", "w", encoding="utf-8") as f:
        json.dump({"model": model_type, "n_folds": len(results),
                   "test_window_days": test_window, **summary}, f, indent=2, ensure_ascii=False)

    # 可视化
    _plot_walk_forward(df_res, out_dir, model_type)

    print(f"\n{'='*60}")
    print(f"Walk-forward 汇总 ({model_type.upper()}, {len(results)} folds)")
    print(f"{'='*60}")
    for c in metric_cols:
        if f"{c}_mean" in summary:
            print(f"  {c:25s}  {summary[f'{c}_mean']:.4f} ± {summary[f'{c}_std']:.4f}")

    return df_res


def _plot_walk_forward(df_res: pd.DataFrame, out_dir: Path, model_type: str) -> None:
    """生成跨 fold 的指标趋势图"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Walk-forward Evaluation — {model_type.upper()}", fontsize=13)

    metrics_to_plot = [
        ("mae", "MAE (visitors)", axes[0, 0]),
        ("smape", "sMAPE (%)", axes[0, 1]),
        ("suitability_f1", "Suitability Warning F1", axes[1, 0]),
        ("suitability_recall", "Suitability Warning Recall", axes[1, 1]),
    ]

    folds = df_res["fold"].values
    for col, ylabel, ax in metrics_to_plot:
        if col not in df_res.columns:
            continue
        vals = df_res[col].values
        ax.bar(folds, vals, color="steelblue", alpha=0.75, edgecolor="white")
        mean_val = float(np.mean(vals))
        ax.axhline(mean_val, color="tomato", linestyle="--", linewidth=1.5,
                   label=f"mean={mean_val:.3f}")
        ax.set_xlabel("Fold")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.legend(fontsize=8)
        ax.set_xticks(folds)

    plt.tight_layout()
    plt.savefig(out_dir / "walk_forward_metrics.png", dpi=150)
    plt.close()
    print(f"  图表已保存: {out_dir / 'walk_forward_metrics.png'}")


# ─────────────────────────────────────────────
# CLI 入口
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Walk-forward Evaluation")
    parser.add_argument("--model", default="gru",
                        choices=["gru", "lstm", "seq2seq_attention", "all"],
                        help="模型类型（默认 gru）")
    parser.add_argument("--input-csv",
                        default="data/processed/jiuzhaigou_daily_features_2016-01-01_2026-04-02.csv",
                        help="输入特征 CSV")
    parser.add_argument("--folds", type=int, default=4, help="fold 数量（默认 4）")
    parser.add_argument("--test-window", type=int, default=90,
                        help="每个 fold 的测试窗口天数（默认 90）")
    parser.add_argument("--look-back", type=int, default=30, help="历史窗口（默认 30）")
    parser.add_argument("--epochs", type=int, default=80, help="每 fold 最大训练轮次（默认 80）")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--peak-threshold", type=float, default=DEFAULT_PEAK_THRESHOLD)
    parser.add_argument("--output-dir", default="output/walk_forward",
                        help="输出目录（默认 output/walk_forward）")
    args = parser.parse_args()

    np.random.seed(42)
    tf.random.set_seed(42)

    csv_path = PROJECT_ROOT / args.input_csv
    if not csv_path.exists():
        print(f"错误：找不到数据文件 {csv_path}")
        sys.exit(1)

    print(f"加载数据: {csv_path}")
    df = load_data(csv_path)
    print(f"数据范围: {df['date'].min().date()} ~ {df['date'].max().date()}，共 {len(df)} 行")

    models_to_run = ["gru", "lstm", "seq2seq_attention"] if args.model == "all" else [args.model]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for model_type in models_to_run:
        print(f"\n{'='*60}")
        print(f"开始 Walk-forward Evaluation: {model_type.upper()}")
        print(f"{'='*60}")
        out_dir = PROJECT_ROOT / args.output_dir / f"{model_type}_{timestamp}"
        walk_forward_eval(
            df=df,
            model_type=model_type,
            n_folds=args.folds,
            test_window=args.test_window,
            look_back=args.look_back,
            epochs=args.epochs,
            batch_size=args.batch_size,
            peak_threshold=args.peak_threshold,
            out_dir=out_dir,
        )
        print(f"\n结果已保存至: {out_dir}")


if __name__ == "__main__":
    main()
