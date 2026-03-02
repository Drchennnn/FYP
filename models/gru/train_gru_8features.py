"""
GRU 基线模型 (Gated Recurrent Unit Baseline Model)

GRU 作为 LSTM 的轻量级替代，参数更少，训练更快。
保持与 LSTM 相同的输入特征和评估框架，便于对比。
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
from sklearn.preprocessing import MinMaxScaler

# 导入通用评估器
from models.common.evaluator import calculate_metrics, save_metrics_to_files

matplotlib.use("Agg")


def mark_core_holiday(date_val: pd.Timestamp) -> int:
    """核心节假日标记函数（与LSTM保持一致）
    
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
    """加载数据并进行特征工程（8特征版本）
    
    包含8个核心特征：
    1. visitor_count_scaled (目标值，归一化)
    2. month_norm
    3. day_of_week_norm
    4. is_holiday
    5. tourism_num_lag_7_scaled
    6. meteo_precip_sum_scaled
    7. temp_high_scaled
    8. temp_low_scaled
    
    Args:
        input_csv: 输入CSV文件路径
        
    Returns:
        pd.DataFrame: 包含特征工程后的数据
    """
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

    # 时间特征归一化
    df["month_norm"] = (df["month"] - 1) / 11.0
    df["day_of_week_norm"] = df["day_of_week"] / 6.0

    # 检查并确保所有8个特征都存在
    required_features = [
        "tourism_num_lag_7_scaled",
        "meteo_precip_sum_scaled", 
        "temp_high_scaled",
        "temp_low_scaled"
    ]
    
    for feature in required_features:
        if feature not in df.columns:
            print(f"警告: 特征 '{feature}' 不存在，将用0填充")
            df[feature] = 0.0
    
    return df


def build_sequences(
    df: pd.DataFrame,
    look_back: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """构建时间序列输入（8特征版本）
    
    使用8个核心特征：
    1. visitor_count_scaled (目标值，归一化)
    2. month_norm
    3. day_of_week_norm
    4. is_holiday
    5. tourism_num_lag_7_scaled
    6. meteo_precip_sum_scaled
    7. temp_high_scaled
    8. temp_low_scaled
    
    Args:
        df: 特征工程后的DataFrame
        look_back: 历史窗口长度
        
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: (特征序列, 目标值, 日期)
    """
    feature_cols = [
        "visitor_count_scaled",    # 目标客流（归一化）
        "month_norm",              # 月份归一化
        "day_of_week_norm",        # 星期归一化
        "is_holiday",              # 节假日标记
        "tourism_num_lag_7_scaled", # 滞后7天客流（归一化）
        "meteo_precip_sum_scaled",  # 降水量（归一化）
        "temp_high_scaled",         # 最高温度（归一化）
        "temp_low_scaled"           # 最低温度（归一化）
    ]
    
    # 确保所有特征列都存在
    for col in feature_cols:
        if col not in df.columns:
            raise ValueError(f"缺失特征列: {col}")
    
    values = df[feature_cols].values.astype(np.float32)
    target = df["visitor_count_scaled"].values.astype(np.float32)
    dates = df["date"].values

    x_list, y_list, d_list = [], [], []
    for i in range(look_back, len(df)):
        x_list.append(values[i - look_back : i, :])  # shape: (look_back, 8)
        y_list.append(target[i])
        d_list.append(dates[i])
    
    print(f"序列构建完成: {len(x_list)} 个样本，特征维度: {len(feature_cols)}")
    print(f"使用特征: {feature_cols}")
    
    return np.array(x_list), np.array(y_list), np.array(d_list)


def split_by_time(
    x: np.ndarray,
    y: np.ndarray,
    d: np.ndarray,
    test_ratio: float,
    val_ratio: float,
) -> Tuple[np.ndarray, ...]:
    """按时间顺序划分数据集（与LSTM保持一致）
    
    Args:
        x: 特征序列
        y: 目标值
        d: 日期
        test_ratio: 测试集比例
        val_ratio: 验证集比例（相对于训练+验证集）
        
    Returns:
        Tuple[np.ndarray, ...]: 划分后的数据集
    """
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


def create_gru_model(look_back: int) -> tf.keras.Model:
    """创建GRU模型（8特征版本）
    
    输入形状: (look_back, 8)
    架构:
    - Input Layer: (look_back, 8)
    - GRU Layer 1: 128单元，return_sequences=True
    - Dropout: 0.2
    - GRU Layer 2: 64单元
    - Dropout: 0.2
    - Dense Layer: 32单元，ReLU激活
    - Output Layer: 1单元，线性激活
    
    Args:
        look_back: 历史窗口长度
        
    Returns:
        tf.keras.Model: 编译好的GRU模型
    """
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(look_back, 8)),
            tf.keras.layers.GRU(128, return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.GRU(64),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.Huber(),
    )
    
    print(f"模型创建完成: 输入形状=({look_back}, 8), 参数={model.count_params():,}")
    return model


def save_gru_plots(
    out_dir: Path,
    history: tf.keras.callbacks.History,
    pred_df: pd.DataFrame,
) -> None:
    """保存GRU训练可视化图
    
    Args:
        out_dir: 输出目录
        history: 训练历史
        pred_df: 预测结果DataFrame
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # 训练损失曲线
    plt.figure(figsize=(8, 4))
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("GRU: Training vs Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "gru_loss.png", dpi=150)
    plt.close()

    # 测试集真实值 vs 预测值
    plt.figure(figsize=(12, 5))
    plt.plot(pred_df["date"], pred_df["y_true"], label="True", linewidth=1.6)
    plt.plot(pred_df["date"], pred_df["y_pred"], label="Pred", linewidth=1.6)
    plt.xlabel("Date")
    plt.ylabel("Visitor Count")
    plt.title("GRU: Test Set - True vs Pred")
    plt.xticks(rotation=30)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "gru_true_vs_pred.png", dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="GRU 客流预测训练脚本（基线模型）")
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
    args, _ = parser.parse_known_args()

    np.random.seed(42)
    tf.random.set_seed(42)

    output_dir = Path(args.output_dir)
    model_root_dir = Path(args.model_dir)
    runs_dir = output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    model_runs_dir = model_root_dir / "runs"
    model_runs_dir.mkdir(parents=True, exist_ok=True)
    
    auto_run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_lb{args.look_back}_ep{args.epochs}_gru_8features"
    run_name = args.run_name or auto_run_name
    run_name_pattern = r"^run_\d{8}_\d{6}_lb\d+_ep\d+.*"
    if not re.fullmatch(run_name_pattern, run_name):
        raise ValueError(
            "run_name 格式不符合要求，应为：run_YYYYMMDD_HHMMSS_lb<lookback>_ep<epochs>"
        )
    
    run_dir = runs_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建必要的子目录
    weights_dir = run_dir / "weights"
    fig_dir = run_dir / "figures"
    weights_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 模型权重保存在 output/runs/<run_name>/weights/ 目录中
    model_path = weights_dir / "gru_jiuzhaigou.h5"
    metrics_json_path = run_dir / "gru_metrics.json"
    metrics_csv_path = run_dir / "gru_metrics.csv"
    pred_path = run_dir / "gru_test_predictions.csv"
    history_path = run_dir / "gru_history.csv"

    # 1. 加载数据并特征工程
    df = load_and_engineer_features(Path(args.input_csv))

    # 2. 归一化
    scaler = MinMaxScaler()
    df["visitor_count_scaled"] = scaler.fit_transform(df[["visitor_count"]]).reshape(-1)

    # 3. 构建序列
    x, y, d = build_sequences(df, look_back=args.look_back)
    x_train, y_train, d_train, x_val, y_val, d_val, x_test, y_test, d_test = split_by_time(
        x, y, d, test_ratio=args.test_ratio, val_ratio=args.val_ratio
    )

    # 4. 创建并训练GRU模型
    model = create_gru_model(args.look_back)
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

    # 5. 预测并反归一化
    y_pred_scaled = model.predict(x_test, verbose=0).reshape(-1, 1)
    y_pred = scaler.inverse_transform(y_pred_scaled).reshape(-1)
    y_true = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(-1)

    # 6. 保存预测结果
    pred_df = pd.DataFrame(
        {
            "date": pd.to_datetime(d_test),
            "y_true": y_true,
            "y_pred": y_pred,
        }
    ).sort_values("date")
    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")

    # 7. 使用通用评估器计算指标
    metrics = calculate_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_train_scaled=y_train,
        scaler=scaler,
        peak_quantile=args.peak_quantile,
    )

    # 8. 保存模型和指标
    model.save(model_path)
    
    # 使用通用评估器的保存函数
    additional_info = {
        "samples": int(len(df)),
        "look_back": int(args.look_back),
        "epochs_requested": int(args.epochs),
        "epochs_trained": int(len(history.history["loss"])),
        "train_samples": int(len(x_train)),
        "val_samples": int(len(x_val)),
        "test_samples": int(len(x_test)),
        "input_dim": 8,
        "features": [
            "visitor_count_scaled",      # 目标客流（归一化）
            "month_norm",                # 月份归一化
            "day_of_week_norm",          # 星期归一化
            "is_holiday",                # 节假日标记
            "tourism_num_lag_7_scaled",  # 滞后7天客流（归一化）
            "meteo_precip_sum_scaled",   # 降水量（归一化）
            "temp_high_scaled",          # 最高温度（归一化）
            "temp_low_scaled"            # 最低温度（归一化）
        ],
        "model_architecture": "GRU-128-64-32",
        "feature_version": "8_features_v1",
    }
    
    save_metrics_to_files(
        metrics=metrics,
        run_dir=str(run_dir),
        run_name=run_name,
        model_name="gru",
        additional_info=additional_info,
    )

    # 9. 保存训练历史
    history_df = pd.DataFrame(history.history)
    history_df.insert(0, "epoch", np.arange(1, len(history_df) + 1))
    history_df.to_csv(history_path, index=False, encoding="utf-8-sig")

    # 10. 使用统一评估器计算指标和可视化
    from models.common.evaluator import Evaluator, load_global_threshold
    
    # 加载统一的峰值阈值
    peak_threshold = load_global_threshold()
    
    # 初始化评估器
    evaluator = Evaluator(
        peak_threshold=peak_threshold,
        scaler=scaler
    )
    
    # 计算指标
    metrics_eval = evaluator.evaluate(y_true, y_pred)
    
    # 保存可视化
    if args.save_plots:
        evaluator.generate_visualizations(
            out_dir=fig_dir,
            history=history.history,
            dates=pd.to_datetime(pred_df["date"]),
            y_true=y_true,
            y_pred=y_pred
        )

    # 11. 输出结果
    print(f"GRU训练完成！")
    print(f"运行目录: {run_dir}")
    print(f"模型保存: {model_path}")
    print(f"预测结果: {pred_path}")
    print(f"指标文件: {metrics_json_path}, {metrics_csv_path}")
    
    print("\n回归指标:")
    for key, value in metrics["regression"].items():
        print(f"  {key.upper()}: {value:.4f}")
    
    print("\n分类指标 (高峰日预测):")
    for key, value in metrics["classification"].items():
        if key != "peak_threshold":
            print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()