"""
GRU MIMO 多步输出模型 (Multi-Output Direct Multi-step)

与单步 GRU 的区别：
- 输出层从 Dense(1) 改为 Dense(PRED_STEPS)，一次性预测未来7天
- 训练标签从单值变为7步序列
- 推理时无需滚动，零误差累积
- 不依赖未来 visitor_count，推理最简单

用于在线预测时替代单步滚动 GRU，解决误差累积导致预测值雪崩的问题。
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Tuple

import chinese_calendar as cncal
import matplotlib
import numpy as np
import pandas as pd
import tensorflow as tf

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.common.core_evaluation import evaluate_and_save_run, compute_dynamic_peak_threshold
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from models.common.evaluator import calculate_metrics, save_metrics_to_files

matplotlib.use("Agg")

PRED_STEPS = 7  # 直接预测未来7天


def mark_core_holiday(date_val: pd.Timestamp) -> int:
    m, d = int(date_val.month), int(date_val.day)
    if m == 10 and 1 <= d <= 7:
        return 1
    if m == 5 and 1 <= d <= 5:
        return 1
    try:
        return int(cncal.is_holiday(date_val.date()))
    except Exception:
        return 0


def load_and_engineer_features(input_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(input_csv, encoding="utf-8-sig")
    if "tourism_num" in df.columns:
        target_col = "tourism_num"
    elif "visitor_count" in df.columns:
        target_col = "visitor_count"
    else:
        raise ValueError("未找到目标列")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)
    df["visitor_count"] = pd.to_numeric(df[target_col], errors="coerce")
    df = df.dropna(subset=["visitor_count"]).reset_index(drop=True)

    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.weekday
    df["is_holiday"] = df["date"].apply(mark_core_holiday).astype(float)
    df["month_norm"] = (df["month"] - 1) / 11.0
    df["day_of_week_norm"] = df["day_of_week"] / 6.0

    required_features = [
        "tourism_num_lag_7_scaled", "meteo_precip_sum_scaled",
        "temp_high_scaled", "temp_low_scaled"
    ]
    for feature in required_features:
        if feature not in df.columns:
            print(f"警告: 特征 '{feature}' 不存在，将用0填充")
            df[feature] = 0.0

    return df


def build_sequences_mimo(
    df: pd.DataFrame,
    look_back: int,
    pred_steps: int = PRED_STEPS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """构建 MIMO 多步序列。

    X: (n, look_back, 8) — 历史窗口
    y: (n, pred_steps)   — 未来 pred_steps 天的 visitor_count_scaled
    d: (n,)              — 预测起始日期（窗口后第1天）
    """
    feature_cols = [
        "visitor_count_scaled", "month_norm", "day_of_week_norm", "is_holiday",
        "tourism_num_lag_7_scaled", "meteo_precip_sum_scaled",
        "temp_high_scaled", "temp_low_scaled"
    ]
    for col in feature_cols:
        if col not in df.columns:
            raise ValueError(f"缺失特征列: {col}")

    values = df[feature_cols].values.astype(np.float32)
    target = df["visitor_count_scaled"].values.astype(np.float32)
    dates = df["date"].values

    x_list, y_list, d_list = [], [], []
    for i in range(look_back, len(df) - pred_steps + 1):
        x_list.append(values[i - look_back: i, :])          # (look_back, 8)
        y_list.append(target[i: i + pred_steps])             # (pred_steps,)
        d_list.append(dates[i])

    print(f"MIMO序列构建完成: {len(x_list)} 个样本，预测步数: {pred_steps}")
    return np.array(x_list), np.array(y_list), np.array(d_list)


def split_by_time(x, y, d, test_ratio, val_ratio):
    n = len(x)
    test_size = int(n * test_ratio)
    trainval_size = n - test_size
    val_size = int(trainval_size * val_ratio)
    train_size = trainval_size - val_size

    return (
        x[:train_size], y[:train_size], d[:train_size],
        x[train_size:train_size + val_size], y[train_size:train_size + val_size], d[train_size:train_size + val_size],
        x[trainval_size:], y[trainval_size:], d[trainval_size:]
    )


def create_gru_mimo_model(look_back: int, pred_steps: int = PRED_STEPS) -> tf.keras.Model:
    """GRU MIMO 模型：输入 (look_back, 8)，输出 (pred_steps,)"""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(look_back, 8)),
        tf.keras.layers.GRU(128, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.GRU(64),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(pred_steps),  # 直接输出 pred_steps 个值
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.Huber(),
    )
    print(f"GRU MIMO 模型创建完成: 输入=({look_back}, 8), 输出={pred_steps}, 参数={model.count_params():,}")
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="GRU MIMO 多步输出训练脚本")
    parser.add_argument("--input-csv", default="data/processed/jiuzhaigou_daily_features_2016-01-01_2026-04-02.csv")
    parser.add_argument("--look-back", type=int, default=30)
    parser.add_argument("--pred-steps", type=int, default=PRED_STEPS)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--val-ratio", type=float, default=0.125)
    parser.add_argument("--peak-quantile", type=float, default=0.75)
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--model-dir", default="model")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--save-plots", action=argparse.BooleanOptionalAction, default=True)
    args, _ = parser.parse_known_args()

    np.random.seed(42)
    tf.random.set_seed(42)

    output_dir = Path(args.output_dir)
    runs_dir = output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    auto_run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_lb{args.look_back}_ep{args.epochs}_gru_mimo_8features"
    run_name = args.run_name or auto_run_name
    run_name_pattern = r"^run_\d{8}_\d{6}_lb\d+_ep\d+.*"
    if not re.fullmatch(run_name_pattern, run_name):
        raise ValueError("run_name 格式不符合要求")

    run_dir = runs_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = run_dir / "weights"
    fig_dir = run_dir / "figures"
    weights_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    model_path = weights_dir / "gru_mimo_jiuzhaigou.h5"
    pred_path = run_dir / "gru_mimo_test_predictions.csv"
    history_path = run_dir / "gru_mimo_history.csv"

    # 1. 加载数据
    df = load_and_engineer_features(Path(args.input_csv))

    # 2. 归一化
    scaler = MinMaxScaler()
    df["visitor_count_scaled"] = scaler.fit_transform(df[["visitor_count"]]).reshape(-1)

    # 3. 构建 MIMO 序列
    x, y, d = build_sequences_mimo(df, look_back=args.look_back, pred_steps=args.pred_steps)

    # 天气数组（对齐到每个样本的预测起始日）
    def _pick_col(candidates):
        for c in candidates:
            if c in df.columns:
                return c
        return None

    precip_col = _pick_col(["meteo_precip_sum", "meteo_rain_sum"])
    temp_high_col = _pick_col(["meteo_temp_max", "temp_high_c"])
    temp_low_col = _pick_col(["meteo_temp_min", "temp_low_c"])

    weather_precip_all = df[precip_col].values[args.look_back: len(df) - args.pred_steps + 1].astype(float)
    weather_temp_high_all = df[temp_high_col].values[args.look_back: len(df) - args.pred_steps + 1].astype(float)
    weather_temp_low_all = df[temp_low_col].values[args.look_back: len(df) - args.pred_steps + 1].astype(float)

    # 4. 划分数据集
    x_train, y_train, d_train, x_val, y_val, d_val, x_test, y_test, d_test = split_by_time(
        x, y, d, test_ratio=args.test_ratio, val_ratio=args.val_ratio
    )

    n_train, n_val = len(y_train), len(y_val)
    weather_train_precip = weather_precip_all[:n_train]
    weather_train_temp_high = weather_temp_high_all[:n_train]
    weather_train_temp_low = weather_temp_low_all[:n_train]
    weather_test_precip = weather_precip_all[n_train + n_val:]
    weather_test_temp_high = weather_temp_high_all[n_train + n_val:]
    weather_test_temp_low = weather_temp_low_all[n_train + n_val:]

    # 动态峰值阈值（仅用训练集第一步）
    train_visitor_counts = scaler.inverse_transform(y_train[:, 0].reshape(-1, 1)).reshape(-1)
    dynamic_peak_threshold = compute_dynamic_peak_threshold(train_visitor_counts, quantile=args.peak_quantile)
    print(f"动态峰值阈值: {dynamic_peak_threshold:.0f}")

    # 5. 训练
    model = create_gru_mimo_model(args.look_back, args.pred_steps)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=7, min_lr=1e-5),
    ]
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs, batch_size=args.batch_size,
        shuffle=False, callbacks=callbacks, verbose=1,
    )

    # 6. 预测（取第一步作为单日预测，与单步模型评估口径一致）
    y_pred_scaled = model.predict(x_test, verbose=0)  # (n, pred_steps)
    y_pred = scaler.inverse_transform(y_pred_scaled[:, 0].reshape(-1, 1)).reshape(-1)
    y_true = scaler.inverse_transform(y_test[:, 0].reshape(-1, 1)).reshape(-1)

    # 7. 保存预测 CSV（保存第一步，格式与单步版本兼容）
    pred_df = pd.DataFrame({
        "date": pd.to_datetime(d_test),
        "y_true": y_true,
        "y_pred": y_pred,
    }).sort_values("date")
    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")

    # 8. 评估
    metrics = calculate_metrics(y_true=y_true, y_pred=y_pred, scaler=scaler)
    try:
        save_metrics_to_files(metrics, str(run_dir), "gru_mimo_baseline")
    except TypeError:
        pass

    # 9. 训练历史
    history_df = pd.DataFrame(history.history)
    history_df.insert(0, "epoch", np.arange(1, len(history_df) + 1))
    history_df.to_csv(history_path, index=False, encoding="utf-8-sig")

    # 10. 统一评估产物
    evaluate_and_save_run(
        str(run_dir),
        model_name="gru_mimo",
        feature_count=8,
        y_true=np.asarray(y_true),
        y_pred=np.asarray(y_pred),
        dates=pd.to_datetime(pred_df["date"]).values,
        horizon=1,
        peak_threshold=dynamic_peak_threshold,
        warning_temperature=1000.0,
        fn_fp_cost_ratio=(5.0, 1.0),
        weather_precip=np.asarray(weather_test_precip),
        weather_temp_high=np.asarray(weather_test_temp_high),
        weather_temp_low=np.asarray(weather_test_temp_low),
        weather_train_precip=np.asarray(weather_train_precip),
        weather_train_temp_high=np.asarray(weather_train_temp_high),
        weather_train_temp_low=np.asarray(weather_train_temp_low),
        extra_meta={
            "look_back": int(args.look_back),
            "pred_steps": int(args.pred_steps),
            "epochs_requested": int(args.epochs),
            "epochs_trained": int(len(history.history.get("loss", []))),
            "model_architecture": f"GRU-MIMO-128-64-Dense{args.pred_steps}",
            "feature_version": "8_features_v1",
        },
    )

    # 11. 保存模型
    model.save(model_path)

    # 12. 可视化
    if args.save_plots:
        plt.figure(figsize=(8, 4))
        plt.plot(history.history["loss"], label="train_loss")
        plt.plot(history.history["val_loss"], label="val_loss")
        plt.xlabel("Epoch"); plt.ylabel("Loss")
        plt.title("GRU MIMO: Training vs Validation Loss")
        plt.legend(); plt.tight_layout()
        plt.savefig(fig_dir / "gru_mimo_loss.png", dpi=150); plt.close()

        plt.figure(figsize=(12, 5))
        plt.plot(pred_df["date"], pred_df["y_true"], label="True", linewidth=1.6)
        plt.plot(pred_df["date"], pred_df["y_pred"], label="Pred (step1)", linewidth=1.6)
        plt.xlabel("Date"); plt.ylabel("Visitor Count")
        plt.title("GRU MIMO: Test Set - True vs Pred (Step 1)")
        plt.xticks(rotation=30); plt.legend(); plt.tight_layout()
        plt.savefig(fig_dir / "gru_mimo_true_vs_pred.png", dpi=150); plt.close()

    print(f"\nGRU MIMO 训练完成！")
    print(f"运行目录: {run_dir}")
    print(f"模型保存: {model_path}")
    print(f"预测结果: {pred_path}")
    print("\n回归指标 (Step 1):")
    for key, value in metrics["regression"].items():
        print(f"  {key.upper()}: {value:.4f}")


if __name__ == "__main__":
    main()
