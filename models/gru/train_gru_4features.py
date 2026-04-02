#!/usr/bin/env python3
"""Jiuzhaigou visitor forecasting - GRU (4 features, single-step).

Unified artifacts are saved under:
  output/runs/<run>/metrics.json
  output/runs/<run>/metrics.csv
  output/runs/<run>/figures/*.png
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Tuple

import chinese_calendar as cncal
import matplotlib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# Ensure project root is on sys.path when running as a script
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.common.core_evaluation import evaluate_and_save_run

matplotlib.use("Agg")


def mark_core_holiday(date_val: pd.Timestamp) -> int:
    """Core holiday marker (binary)."""

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
        raise ValueError("Missing target column: expected tourism_num or visitor_count")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)
    df["visitor_count"] = pd.to_numeric(df[target_col], errors="coerce")
    df = df.dropna(subset=["visitor_count"]).reset_index(drop=True)

    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.weekday
    df["is_holiday"] = df["date"].apply(mark_core_holiday).astype(float)

    df["month_norm"] = (df["month"] - 1) / 11.0
    df["day_of_week_norm"] = df["day_of_week"] / 6.0
    return df


def build_sequences(df: pd.DataFrame, look_back: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    feature_cols = ["visitor_count_scaled", "month_norm", "day_of_week_norm", "is_holiday"]
    values = df[feature_cols].values.astype(np.float32)
    target = df["visitor_count_scaled"].values.astype(np.float32)
    dates = df["date"].values

    x_list, y_list, d_list = [], [], []
    for i in range(look_back, len(df)):
        x_list.append(values[i - look_back : i, :])
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
            tf.keras.layers.GRU(64, return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.GRU(32),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss=tf.keras.losses.Huber())
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="GRU training (4 features, single-step).")
    parser.add_argument(
        "--input-csv",
        default="data/raw/jiuzhaigou_tourism_weather_2024_2026_latest.csv",
        help="Input CSV (must include date + target column).",
    )
    parser.add_argument("--look-back", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--model-dir", default="model")
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--save-plots",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args, _ = parser.parse_known_args()

    np.random.seed(42)
    tf.random.set_seed(42)

    output_dir = Path(args.output_dir)
    runs_dir = output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    auto_run_name = (
        f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_lb{args.look_back}_ep{args.epochs}_gru_4features"
    )
    run_name = args.run_name or auto_run_name
    if not re.fullmatch(r"^run_\d{8}_\d{6}_.+$", run_name):
        raise ValueError("run_name must match: run_YYYYMMDD_HHMMSS_...")

    run_dir = runs_dir / run_name
    (run_dir / "weights").mkdir(parents=True, exist_ok=True)
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)

    model_path = run_dir / "weights" / "gru_jiuzhaigou.h5"
    pred_path = run_dir / "gru_test_predictions.csv"
    history_path = run_dir / "gru_history.csv"

    df = load_and_engineer_features(Path(args.input_csv))
    scaler = MinMaxScaler()
    df["visitor_count_scaled"] = scaler.fit_transform(df[["visitor_count"]]).reshape(-1)

    x, y, d = build_sequences(df, look_back=args.look_back)
    x_train, y_train, d_train, x_val, y_val, d_val, x_test, y_test, d_test = split_by_time(
        x, y, d, test_ratio=args.test_ratio, val_ratio=args.val_ratio
    )

    model = create_model(args.look_back)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=7, min_lr=1e-5),
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

    y_pred_scaled = model.predict(x_test, verbose=0).reshape(-1, 1)
    y_pred = scaler.inverse_transform(y_pred_scaled).reshape(-1)
    y_true = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(-1)

    pred_df = pd.DataFrame(
        {"date": pd.to_datetime(d_test), "y_true": y_true, "y_pred": y_pred}
    ).sort_values("date")
    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")

    model.save(model_path)
    pd.DataFrame(history.history).assign(epoch=np.arange(1, len(history.history["loss"]) + 1)).to_csv(
        history_path, index=False, encoding="utf-8-sig"
    )

    extra_meta = {
        "samples": int(len(df)),
        "look_back": int(args.look_back),
        "epochs_requested": int(args.epochs),
        "epochs_trained": int(len(history.history["loss"])),
        "train_samples": int(len(x_train)),
        "val_samples": int(len(x_val)),
        "test_samples": int(len(x_test)),
        "features": ["visitor_count_scaled", "month_norm", "day_of_week_norm", "is_holiday"],
    }
    evaluate_and_save_run(
        run_dir=str(run_dir),
        model_name="gru",
        feature_count=4,
        y_true=y_true,
        y_pred=y_pred,
        dates=pd.to_datetime(pred_df["date"]),
        horizon=1,
        extra_meta=extra_meta,
        save_figures=bool(args.save_plots),
    )

    print("GRU training complete")
    print("Run dir:", run_dir)
    print("Core metrics:", run_dir / "metrics.json")


if __name__ == "__main__":
    main()
