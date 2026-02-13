"""九寨沟客流预测 LSTM 训练脚本（改进版）。

主要设计：
1. 基于 date 的特征工程：month_norm、day_of_week_norm、is_holiday
2. 对 visitor_count 使用 MinMax 归一化
3. Look-back 默认 30 天
4. LSTM + Dropout(0.2) + EarlyStopping
5. 输出预测结果、指标和可视化（含混淆矩阵1/2）
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
from matplotlib import pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
)
from sklearn.preprocessing import MinMaxScaler

matplotlib.use("Agg")


def mark_core_holiday(date_val: pd.Timestamp) -> int:
    """核心节假日标记函数。

    显式标记：
    - 国庆：10/01 - 10/07
    - 劳动节：05/01 - 05/05
    - 春节及其他法定节假日：由 chinese_calendar 动态判断
    """
    m, d = int(date_val.month), int(date_val.day)

    # 国庆黄金周
    if m == 10 and 1 <= d <= 7:
        return 1
    # 劳动节窗口
    if m == 5 and 1 <= d <= 5:
        return 1

    # 动态法定节假日（覆盖春节等）
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
        raise ValueError("未找到目标列，请包含 'tourism_num' 或 'visitor_count'。")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)
    df["visitor_count"] = pd.to_numeric(df[target_col], errors="coerce")
    df = df.dropna(subset=["visitor_count"]).reset_index(drop=True)

    # 必要时间特征
    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.weekday
    df["is_holiday"] = df["date"].apply(mark_core_holiday).astype(float)

    # 时间特征归一化（按你的要求，使用归一化而非 one-hot）
    df["month_norm"] = (df["month"] - 1) / 11.0
    df["day_of_week_norm"] = df["day_of_week"] / 6.0

    return df


def build_sequences(
    df: pd.DataFrame,
    look_back: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    feature_cols = ["visitor_count_scaled", "month_norm", "day_of_week_norm", "is_holiday"]
    values = df[feature_cols].values.astype(np.float32)
    target = df["visitor_count_scaled"].values.astype(np.float32)
    dates = df["date"].values

    x_list, y_list, d_list = [], [], []
    for i in range(look_back, len(df)):
        x_list.append(values[i - look_back : i, :])  # shape: (look_back, 4)
        y_list.append(target[i])
        d_list.append(dates[i])
    return np.array(x_list), np.array(y_list), np.array(d_list)


def split_by_time(
    x: np.ndarray,
    y: np.ndarray,
    d: np.ndarray,
    test_ratio: float,
    val_ratio: float,
) -> Tuple[np.ndarray, ...]:
    n = len(x)
    test_size = int(n * test_ratio)
    trainval_size = n - test_size
    val_size = int(trainval_size * val_ratio)
    train_size = trainval_size - val_size

    x_train = x[:train_size]
    y_train = y[:train_size]
    d_train = d[:train_size]

    x_val = x[train_size : train_size + val_size]
    y_val = y[train_size : train_size + val_size]
    d_val = d[train_size : train_size + val_size]

    x_test = x[trainval_size:]
    y_test = y[trainval_size:]
    d_test = d[trainval_size:]
    return x_train, y_train, d_train, x_val, y_val, d_val, x_test, y_test, d_test


def create_model(look_back: int) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(look_back, 4)),
            tf.keras.layers.LSTM(64, return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.Huber(),
    )
    return model


def calc_smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denominator = np.where(denominator == 0, 1e-8, denominator)
    return float(np.mean(np.abs(y_true - y_pred) / denominator))


def save_plots(
    out_dir: Path,
    history: tf.keras.callbacks.History,
    pred_df: pd.DataFrame,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # 训练损失曲线
    plt.figure(figsize=(8, 4))
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "lstm_loss.png", dpi=150)
    plt.close()

    # 测试集真实值 vs 预测值
    plt.figure(figsize=(12, 5))
    plt.plot(pred_df["date"], pred_df["y_true"], label="True", linewidth=1.6)
    plt.plot(pred_df["date"], pred_df["y_pred"], label="Pred", linewidth=1.6)
    plt.xlabel("Date")
    plt.ylabel("Visitor Count")
    plt.title("Test Set: True vs Pred")
    plt.xticks(rotation=30)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "lstm_true_vs_pred.png", dpi=150)
    plt.close()

def save_confusion_matrices(out_dir: Path, y_true_cls: np.ndarray, y_pred_cls: np.ndarray) -> None:
    """保存两种混淆矩阵：
    1) 计数矩阵
    2) 归一化矩阵（混淆矩阵2）
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # 混淆矩阵1：原始计数
    disp1 = ConfusionMatrixDisplay.from_predictions(
        y_true_cls,
        y_pred_cls,
        display_labels=["non_peak", "peak"],
        cmap="Blues",
        normalize=None,
    )
    disp1.ax_.set_title("Confusion Matrix 1 (Count)")
    plt.tight_layout()
    plt.savefig(out_dir / "lstm_confusion_matrix_1.png", dpi=150)
    plt.close()

    # 混淆矩阵2：按真实类别归一化
    disp2 = ConfusionMatrixDisplay.from_predictions(
        y_true_cls,
        y_pred_cls,
        display_labels=["non_peak", "peak"],
        cmap="Blues",
        normalize="true",
    )
    disp2.ax_.set_title("Confusion Matrix 2 (Normalized)")
    plt.tight_layout()
    plt.savefig(out_dir / "lstm_confusion_matrix_2.png", dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="改进版 LSTM 客流预测训练脚本。")
    parser.add_argument(
        "--input-csv",
        default="data/raw/jiuzhaigou_tourism_weather_2024_2026_latest.csv",
        help="输入 CSV（需包含 date + 客流列）。",
    )
    parser.add_argument("--look-back", type=int, default=30, help="历史窗口长度。")
    parser.add_argument("--epochs", type=int, default=120, help="训练轮次（建议 >=100）。")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--peak-quantile", type=float, default=0.75)
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--model-dir", default="model")
    parser.add_argument(
        "--run-name",
        default=None,
        help="可选轮次目录名，不传则自动按时间戳生成。",
    )
    parser.add_argument(
        "--save-plots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否保存可视化图。默认 True。",
    )
    args = parser.parse_args()

    np.random.seed(42)
    tf.random.set_seed(42)

    output_dir = Path(args.output_dir)
    model_root_dir = Path(args.model_dir)
    runs_dir = output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    model_runs_dir = model_root_dir / "runs"
    model_runs_dir.mkdir(parents=True, exist_ok=True)
    auto_run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_lb{args.look_back}_ep{args.epochs}"
    run_name = args.run_name or auto_run_name
    run_name_pattern = r"^run_\d{8}_\d{6}_lb\d+_ep\d+$"
    if not re.fullmatch(run_name_pattern, run_name):
        raise ValueError(
            "run_name 格式不符合要求，应为：run_YYYYMMDD_HHMMSS_lb<lookback>_ep<epochs>"
        )
    run_dir = runs_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    model_run_dir = model_runs_dir / run_name
    model_run_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_run_dir / "lstm_jiuzhaigou.keras"
    metrics_json_path = run_dir / "lstm_metrics.json"
    metrics_csv_path = run_dir / "lstm_metrics.csv"
    pred_path = run_dir / "lstm_test_predictions.csv"
    history_path = run_dir / "lstm_history.csv"
    fig_dir = run_dir / "figures"

    df = load_and_engineer_features(Path(args.input_csv))

    # 对目标客流做 MinMax 归一化
    scaler = MinMaxScaler()
    df["visitor_count_scaled"] = scaler.fit_transform(df[["visitor_count"]]).reshape(-1)

    x, y, d = build_sequences(df, look_back=args.look_back)
    x_train, y_train, d_train, x_val, y_val, d_val, x_test, y_test, d_test = split_by_time(
        x, y, d, test_ratio=args.test_ratio, val_ratio=args.val_ratio
    )

    model = create_model(args.look_back)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=15,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=7,
            min_lr=1e-5,
        ),
    ]

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        shuffle=False,
        callbacks=callbacks,
        verbose=1,
    )

    # 预测并反归一化
    y_pred_scaled = model.predict(x_test, verbose=0).reshape(-1, 1)
    y_pred = scaler.inverse_transform(y_pred_scaled).reshape(-1)
    y_true = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(-1)

    pred_df = pd.DataFrame(
        {
            "date": pd.to_datetime(d_test),
            "y_true": y_true,
            "y_pred": y_pred,
        }
    ).sort_values("date")
    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")

    # 回归指标
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = float(mean_absolute_percentage_error(y_true, y_pred))
    smape = calc_smape(y_true, y_pred)
    r2 = float(r2_score(y_true, y_pred))

    # 由回归输出衍生的“高峰日”二分类指标
    threshold = float(np.quantile(scaler.inverse_transform(y_train.reshape(-1, 1)).reshape(-1), args.peak_quantile))
    y_true_cls = (y_true >= threshold).astype(int)
    y_pred_cls = (y_pred >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_cls, y_pred_cls, average="binary", zero_division=0
    )
    acc = accuracy_score(y_true_cls, y_pred_cls)

    metrics = {
        "run_name": run_name,
        "samples": int(len(df)),
        "look_back": int(args.look_back),
        "epochs_requested": int(args.epochs),
        "epochs_trained": int(len(history.history["loss"])),
        "train_samples": int(len(x_train)),
        "val_samples": int(len(x_val)),
        "test_samples": int(len(x_test)),
        "input_dim": 4,
        "features": ["visitor_count_scaled", "month_norm", "day_of_week_norm", "is_holiday"],
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "smape": smape,
        "r2": r2,
        "peak_threshold": threshold,
        "classification_accuracy": float(acc),
        "classification_precision": float(precision),
        "classification_recall": float(recall),
        "classification_f1": float(f1),
    }

    model.save(model_path)
    metrics_json_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame([metrics]).to_csv(metrics_csv_path, index=False, encoding="utf-8-sig")
    history_df = pd.DataFrame(history.history)
    history_df.insert(0, "epoch", np.arange(1, len(history_df) + 1))
    history_df.to_csv(history_path, index=False, encoding="utf-8-sig")

    if args.save_plots:
        save_plots(fig_dir, history, pred_df)
        save_confusion_matrices(fig_dir, y_true_cls, y_pred_cls)

    print(f"run_dir: {run_dir}")
    print(f"model_run_dir: {model_run_dir}")
    print(f"history_path: {history_path}")
    print(f"model_path: {model_path}")
    print(f"metrics_json_path: {metrics_json_path}")
    print(f"metrics_csv_path: {metrics_csv_path}")
    print(f"pred_path: {pred_path}")
    if args.save_plots:
        print(f"fig_dir: {fig_dir}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
