# 基线模型：8特征标准单步Prophet (未加入Seq2Seq)

"""九寨沟客流预测 Prophet 训练脚本（改进版）。

主要设计：
1. 基于 date 的特征工程：month_norm、day_of_week_norm、is_holiday
2. 对 visitor_count 使用 MinMax 归一化
3. 添加额外回归器：滞后特征、气象特征
4. 输出预测结果、指标和可视化
5. 包含8个核心特征的支持
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
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

matplotlib.use("Agg")

# 尝试导入Prophet，如果失败则提供友好的错误信息
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("警告: Prophet 未安装。请运行: pip install prophet")
    print("Prophet模型将无法运行，但框架已搭建完成。")


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
    parser = argparse.ArgumentParser(description="Prophet 客流预测训练脚本。")
    parser.add_argument(
        "--input-csv",
        default="data/processed/jiuzhaigou_8features_latest.csv",
        help="输入 CSV（需包含 date + 客流列）。",
    )
    parser.add_argument("--epochs", type=int, default=100, help="训练轮次。")
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

    output_dir = Path(args.output_dir)
    model_root_dir = Path(args.model_dir)
    runs_dir = output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    model_runs_dir = model_root_dir / "runs"
    model_runs_dir.mkdir(parents=True, exist_ok=True)
    auto_run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_lb0_prophet_4features"
    run_name = args.run_name or auto_run_name
    run_name_pattern = r"^run_\d{8}_\d{6}_.+$"
    if not re.fullmatch(run_name_pattern, run_name):
        raise ValueError(
            "run_name 格式不符合要求，应为：run_YYYYMMDD_HHMMSS_ep<epochs>"
        )
    run_dir = runs_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    model_run_dir = model_runs_dir / run_name
    model_run_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_run_dir / "prophet_jiuzhaigou_baseline.json"
    metrics_json_path = run_dir / "prophet_baseline_metrics.json"
    metrics_csv_path = run_dir / "prophet_baseline_metrics.csv"
    pred_path = run_dir / "prophet_baseline_test_predictions.csv"
    history_path = run_dir / "prophet_baseline_history.csv"
    fig_dir = run_dir / "figures"

    # 1. 加载数据并特征工程
    df = load_and_engineer_features(Path(args.input_csv))

    # 2. 对目标客流做 MinMax 归一化
    scaler = MinMaxScaler()
    df["visitor_count_scaled"] = scaler.fit_transform(df[["visitor_count"]]).reshape(-1)

    # 3. 准备 Prophet 训练数据（包含8个特征）
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
    
    # 构建Prophet格式数据
    prophet_df = df[["date"] + feature_cols].copy()
    prophet_df.columns = ["ds", "y"] + [col for col in feature_cols if col != "visitor_count_scaled"]
    
    # 时间划分训练/测试集
    n = len(prophet_df)
    test_size = int(n * args.test_ratio)
    train_size = n - test_size
    
    X_train = prophet_df[:train_size]
    X_test = prophet_df[train_size:]
    
    y_train = X_train["y"].values
    y_test = X_test["y"].values
    
    print(f"数据准备完成:")
    print(f"  总样本数: {n}")
    print(f"  训练样本: {train_size}, 测试样本: {test_size}")
    print(f"  输入形状: {X_train.shape}")
    print(f"  输出形状: {y_train.shape}")

    # 4. 创建 Prophet 模型（8特征版本）
    if not PROPHET_AVAILABLE:
        print("错误: Prophet 未安装。请运行 pip install prophet")
        return
    
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode='additive',
        holidays_prior_scale=10.0,
        seasonality_prior_scale=10.0
    )
    
    # 添加额外回归器
    regressor_cols = [
        "month_norm",
        "day_of_week_norm",
        "is_holiday",
        "tourism_num_lag_7_scaled",
        "meteo_precip_sum_scaled",
        "temp_high_scaled",
        "temp_low_scaled"
    ]
    
    for regressor in regressor_cols:
        model.add_regressor(regressor)
    
    # 5. 训练模型
    model.fit(X_train)

    # 6. 预测
    forecast = model.predict(X_test)
    
    # 提取预测值
    y_pred = forecast["yhat"].values

    # 7. 反归一化
    y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1)).reshape(-1)
    y_true = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(-1)

    # 8. 计算指标
    from models.common.evaluator import calculate_metrics
    
    metrics = calculate_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_train_scaled=y_train,
        scaler=scaler,
        peak_quantile=args.peak_quantile
    )

    print("\n回归指标:")
    for key, value in metrics["regression"].items():
        print(f"  {key.upper()}: {value:.4f}")

    print("\n分类指标:")
    for key, value in metrics["classification"].items():
        if key != "threshold":
            print(f"  {key.upper()}: {value:.4f}")

    # 9. 保存模型和指标
    # Prophet 模型不能直接保存为 JSON，跳过保存模型
    print("警告: Prophet 模型不能直接保存为 JSON，跳过保存模型")
    
    # 保存指标
    with open(metrics_json_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    
    # 保存CSV格式指标
    metrics_df = pd.DataFrame(metrics["regression"].items(), columns=["Metric", "Value"])
    metrics_df.to_csv(metrics_csv_path, index=False, encoding="utf-8-sig")

    # 保存测试预测结果
    pred_df = pd.DataFrame({
        "date": X_test["ds"].dt.strftime("%Y-%m-%d"),
        "y_true": y_true,
        "y_pred": y_pred
    })
    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")

    # 10. 保存可视化图
    if args.save_plots and fig_dir:
        fig_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存预测图
        plt.figure(figsize=(12, 6))
        plt.plot(df["date"], scaler.inverse_transform(df["visitor_count_scaled"].values.reshape(-1, 1)), label="真实值")
        plt.plot(X_test["ds"], y_pred, label="预测值")
        plt.title("Prophet 客流预测")
        plt.xlabel("日期")
        plt.ylabel("游客数量")
        plt.legend()
        plt.savefig(fig_dir / "prophet_baseline_prediction.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # 保存趋势图
        fig = model.plot_components(forecast)
        fig.savefig(fig_dir / "prophet_baseline_components.png", dpi=150, bbox_inches='tight')
        plt.close(fig)

    # 11. 输出结果
    print(f"\n{'='*80}")
    print(f"训练完成！")
    print(f"{'='*80}")
    print(f"运行目录: {run_dir}")
    print(f"模型保存: {model_path}")
    print(f"预测结果: {pred_path}")


if __name__ == "__main__":
    main()
