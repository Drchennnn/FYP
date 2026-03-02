from __future__ import annotations
def safe_json_serializer(obj):
    if isinstance(obj, np.float32) or isinstance(obj, np.float64):
        return float(obj)
    if isinstance(obj, np.int32) or isinstance(obj, np.int64):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
# 基线模型：8特征标准单步GRU (未加入Seq2Seq)

"""九寨沟客流预测 GRU 训练脚本（改进版）。

主要设计：
1. 基于 date 的特征工程：month_norm、day_of_week_norm、is_holiday
2. 对 visitor_count 使用 MinMax 归一化
3. Look-back 默认 30 天
4. GRU + Dropout(0.2) + EarlyStopping
5. 输出预测结果、指标和可视化（含混淆矩阵1/2）
"""



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
        input_csv: 输入CSV路径
        
    Returns:
        包含8个特征的DataFrame
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


def main() -> None:
    parser = argparse.ArgumentParser(description="GRU 客流预测训练脚本。")
    parser.add_argument(
        "--input-csv",
        default="data/processed/jiuzhaigou_8features_latest.csv",
        help="输入 CSV（需包含 date + 客流列）。",
    )
    parser.add_argument("--look-back", type=int, default=30, help="历史窗口长度。")
    parser.add_argument("--epochs", type=int, default=120, help="训练轮次（建议 >=100）。")
    parser.add_argument("--batch-size", type=int, default=64)
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
    auto_run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_lb{args.look_back}_ep{args.epochs}_gru_4features"
    run_name = args.run_name or auto_run_name
    run_name_pattern = r"^run_\d{8}_\d{6}_.+$"
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
    model_path = weights_dir / "gru_jiuzhaigou_baseline.h5"
    metrics_json_path = run_dir / "gru_baseline_metrics.json"
    metrics_csv_path = run_dir / "gru_baseline_metrics.csv"
    pred_path = run_dir / "gru_baseline_test_predictions.csv"
    history_path = run_dir / "gru_baseline_history.csv"

    # 1. 加载数据并特征工程
    df = load_and_engineer_features(Path(args.input_csv))

    # 2. 对目标客流做 MinMax 归一化
    scaler = MinMaxScaler()
    df["visitor_count_scaled"] = scaler.fit_transform(df[["visitor_count"]]).reshape(-1)

    # 3. 准备 GRU 训练数据（标准单步预测）
    feature_cols = [
        "visitor_count_scaled",      # 目标客流（归一化）
        "month_norm",                # 月份归一化
        "day_of_week_norm",          # 星期归一化
        "is_holiday",                # 节假日标记
        "tourism_num_lag_7_scaled",  # 滞后7天客流（归一化）
        "meteo_precip_sum_scaled",   # 降水量（归一化）
        "temp_high_scaled",          # 最高温度（归一化）
        "temp_low_scaled"            # 最低温度（归一化）
    ]
    
    values = df[feature_cols].values.astype(np.float32)
    dates = df["date"].values
    
    # 创建序列数据
    X, y = [], []
    for i in range(args.look_back, len(values)):
        X.append(values[i - args.look_back : i, :])
        y.append(values[i, 0])
    
    X = np.array(X)
    y = np.array(y)
    
    # 时间划分训练/测试集
    n = len(X)
    test_size = int(n * args.test_ratio)
    train_size = n - test_size
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    
    X_test = X[train_size:]
    y_test = y[train_size:]
    
    print(f"数据准备完成:")
    print(f"  总样本数: {n}")
    print(f"  训练样本: {train_size}, 测试样本: {test_size}")
    print(f"  输入形状: {X_train.shape}")
    print(f"  输出形状: {y_train.shape}")

    # 4. 创建 GRU 模型（标准单步预测）
    model = tf.keras.Sequential([
        tf.keras.layers.GRU(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2, input_shape=(X_train.shape[1], X_train.shape[2])),
        tf.keras.layers.GRU(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
        tf.keras.layers.GRU(32, dropout=0.2, recurrent_dropout=0.2),
        tf.keras.layers.Dense(1)
    ])

    # 5. 编译模型 - 使用自定义非对称损失
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="mse", metrics=["mae"])

    # 6. 训练模型
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
        tf.keras.callbacks.ModelCheckpoint(
            str(model_path),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,
        )
    ]

    history = model.fit(
        X_train, y_train,
        validation_split=args.val_ratio,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1,
        shuffle=True
    )

    # 7. 预测
    y_pred = model.predict(X_test, verbose=0)

    # 8. 反归一化
    y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1)).reshape(-1)
    y_true = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(-1)

    # 9. 计算指标
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print("\n回归指标:")
    print(f"  MAE: {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAPE: {mape:.4f}")
    print(f"  R2: {r2:.4f}")

    # 10. 保存模型和指标
    model.save(model_path)
    
    # 修复：直接转换 NumPy 类型为 Python 原生类型
    metrics = {
        "regression": {
            "mae": float(mae),
            "rmse": float(rmse),
            "mape": float(mape),
            "r2": float(r2)
        },
        "run_info": {
            "samples": int(len(df)),
            "look_back": int(args.look_back),
            "epochs_requested": int(args.epochs),
            "epochs_trained": int(len(history.history["loss"])),
            "train_samples": int(len(X_train)),
            "val_samples": int(len(X_train) * args.val_ratio),
            "test_samples": int(len(X_test)),
            "input_dim": 8,
            "features": feature_cols,
            "model_architecture": "GRU-128-64-32",
            "feature_version": "8_features_v1",
            "loss_function": "MSE"
        }
    }
    
    with open(metrics_json_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)    # 保存CSV格式指标
    metrics_df = pd.DataFrame({
        "Metric": ["MAE", "RMSE", "MAPE", "R2"],
        "Value": [mae, rmse, mape, r2]
    })
    metrics_df.to_csv(metrics_csv_path, index=False, encoding="utf-8-sig")

    # 11. 保存训练历史
    history_df = pd.DataFrame(history.history)
    history_df.insert(0, "epoch", np.arange(1, len(history_df) + 1))
    history_df.to_csv(history_path, index=False, encoding="utf-8-sig")

    # 12. 保存测试预测结果
    pred_df = []
    for i in range(len(X_test)):
        date_idx = len(df) - len(X_test) + i
        if date_idx >= len(df):
            continue
            
        pred_df.append({
            "date": str(df["date"].iloc[date_idx]),
            "y_true": y_true[i],
            "y_pred": y_pred[i]
        })
    
    pd.DataFrame(pred_df).to_csv(pred_path, index=False, encoding="utf-8-sig")

    # 13. 使用统一评估器计算指标和可视化
    from models.common.evaluator import Evaluator, load_global_threshold
    
    # 加载统一的峰值阈值
    peak_threshold = load_global_threshold()
    
    # 初始化评估器
    evaluator = Evaluator(
        peak_threshold=peak_threshold,
        scaler=scaler
    )
    
    # 计算指标
    metrics = evaluator.evaluate(y_true, y_pred)
    
    # 保存可视化
    if args.save_plots and fig_dir:
        dates = pd.to_datetime([df['date'].iloc[-len(y_true):]]).flatten()
        evaluator.generate_visualizations(
            out_dir=fig_dir,
            history=history.history,
            dates=pd.to_datetime(dates),
            y_true=y_true,
            y_pred=y_pred
        )

    # 14. 输出结果
    print(f"\n{'='*80}")
    print(f"训练完成！")
    print(f"{'='*80}")
    print(f"运行目录: {run_dir}")
    print(f"模型保存: {model_path}")
    print(f"预测结果: {pred_path}")
    print(f"训练历史: {history_path}")


if __name__ == "__main__":
    main()