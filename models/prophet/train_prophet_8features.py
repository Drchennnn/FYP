"""
Prophet 基线模型 (Facebook Prophet Baseline Model)

Prophet 是基于加性模型的时间序列预测算法，对节假日和季节性有很好的处理能力。
作为统计模型的基线，与深度学习方法对比。
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

# 导入通用评估器
from models.common.evaluator import calculate_metrics, save_metrics_to_files

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
    """核心节假日标记函数（与LSTM/GRU保持一致）
    
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
    """加载数据并进行特征工程（与LSTM/GRU保持一致）
    
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

    # 必要时间特征（Prophet主要使用日期和客流，但保留特征用于分析）
    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.weekday
    df["is_holiday"] = df["date"].apply(mark_core_holiday).astype(float)

    # 时间特征归一化
    df["month_norm"] = (df["month"] - 1) / 11.0
    df["day_of_week_norm"] = df["day_of_week"] / 6.0

    return df


def prepare_prophet_data(df: pd.DataFrame, test_ratio: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list]:
    """为Prophet准备数据（多变量版本）
    
    Prophet多变量要求：ds (日期), y (目标值), 以及额外的回归器特征
    包含8个核心特征：
    1. visitor_count_scaled (目标值)
    2. month_norm
    3. day_of_week_norm  
    4. is_holiday
    5. tourism_num_lag_7_scaled
    6. meteo_precip_sum_scaled
    7. temp_high_scaled
    8. temp_low_scaled
    
    按时间顺序划分训练集和测试集
    
    Args:
        df: 特征工程后的DataFrame
        test_ratio: 测试集比例
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list]: 
        (训练集, 测试集, 完整数据集, 特征列列表)
    """
    # 定义8个核心特征列
    feature_cols = [
        "visitor_count_scaled",  # 目标值（归一化后）
        "month_norm",
        "day_of_week_norm", 
        "is_holiday",
        "tourism_num_lag_7_scaled",
        "meteo_precip_sum_scaled",
        "temp_high_scaled",
        "temp_low_scaled"
    ]
    
    # 检查特征是否存在
    available_features = []
    for col in feature_cols:
        if col in df.columns:
            available_features.append(col)
        else:
            print(f"警告: 特征 '{col}' 不存在，将跳过")
    
    # 创建Prophet格式数据（包含所有特征）
    prophet_cols = ["date"] + available_features
    prophet_df = df[prophet_cols].copy()
    
    # 重命名列：date -> ds, visitor_count_scaled -> y
    prophet_df = prophet_df.rename(columns={
        "date": "ds",
        "visitor_count_scaled": "y"
    })
    
    # 确保所有特征列是数值类型
    for col in available_features:
        if col != "visitor_count_scaled":  # y列已经重命名
            prophet_df[col] = pd.to_numeric(prophet_df[col], errors="coerce").fillna(0)
    
    # 按时间划分
    n = len(prophet_df)
    test_size = int(n * test_ratio)
    train_size = n - test_size
    
    train_df = prophet_df.iloc[:train_size].copy()
    test_df = prophet_df.iloc[train_size:].copy()
    
    # 回归器特征列（不包括目标值y）
    regressor_cols = [col for col in available_features if col != "visitor_count_scaled"]
    
    return train_df, test_df, prophet_df, regressor_cols


def create_prophet_model(regressor_cols: list, holidays_df: pd.DataFrame | None = None) -> Prophet:
    """创建Prophet模型（多变量版本）
    
    Prophet配置：
    - 年季节性：True
    - 周季节性：True
    - 日季节性：False（日粒度数据）
    - 节假日效应：如果提供节假日数据
    - 乘法季节性：False（客流数据可能更适合加法模型）
    - 添加额外回归器：温度、降水、滞后特征等
    
    Args:
        regressor_cols: 回归器特征列名列表
        holidays_df: 节假日DataFrame（可选）
        
    Returns:
        Prophet: 配置好的Prophet模型
    """
    if not PROPHET_AVAILABLE:
        raise ImportError("Prophet 未安装。请运行: pip install prophet")
    
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode='additive',  # 加法季节性
        holidays=holidays_df,
        changepoint_prior_scale=0.05,  # 趋势变化灵活性
        seasonality_prior_scale=10.0,  # 季节性强度
        holidays_prior_scale=10.0,     # 节假日强度
    )
    
    # 添加额外回归器（按老大指示）
    for regressor in regressor_cols:
        try:
            model.add_regressor(regressor)
            print(f"  已添加回归器: {regressor}")
        except Exception as e:
            print(f"  警告: 添加回归器 {regressor} 失败: {e}")
    
    return model


def create_chinese_holidays_df(df: pd.DataFrame) -> pd.DataFrame:
    """创建中国节假日DataFrame供Prophet使用
    
    Prophet节假日格式：holiday, ds, lower_window, upper_window
    lower_window/upper_window: 节假日前后影响天数
    
    Args:
        df: 原始数据DataFrame（用于获取日期范围）
        
    Returns:
        pd.DataFrame: Prophet格式的节假日数据
    """
    holidays = []
    
    # 获取日期范围
    start_date = df["date"].min()
    end_date = df["date"].max()
    
    # 生成所有日期
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    for date_val in date_range:
        is_holiday = mark_core_holiday(date_val)
        if is_holiday:
            # 简单标记节假日，可以进一步细化不同类型节假日
            holidays.append({
                'holiday': 'chinese_holiday',
                'ds': date_val,
                'lower_window': 0,
                'upper_window': 0
            })
    
    return pd.DataFrame(holidays)


def save_prophet_plots(
    out_dir: Path,
    model: Prophet,
    train_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    """保存Prophet可视化图
    
    Args:
        out_dir: 输出目录
        model: Prophet模型
        train_df: 训练数据
        forecast_df: 预测结果
        test_df: 测试数据
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Prophet组件图（趋势、季节性、节假日）
    try:
        fig1 = model.plot_components(forecast_df)
        fig1.savefig(out_dir / "prophet_components.png", dpi=150, bbox_inches='tight')
        plt.close(fig1)
    except Exception as e:
        print(f"保存组件图失败: {e}")
    
    # 2. 预测图
    try:
        fig2 = model.plot(forecast_df)
        # 添加测试集真实值
        if not test_df.empty:
            plt.scatter(test_df['ds'], test_df['y'], color='red', s=10, label='Test Actual')
        plt.legend()
        plt.title("Prophet Forecast")
        plt.savefig(out_dir / "prophet_forecast.png", dpi=150, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"保存预测图失败: {e}")
    
    # 3. 测试集真实值 vs 预测值
    try:
        # 提取测试期的预测
        test_forecast = forecast_df[forecast_df['ds'].isin(test_df['ds'])]
        
        plt.figure(figsize=(12, 5))
        plt.plot(test_df['ds'], test_df['y'], label='True', linewidth=1.6)
        plt.plot(test_forecast['ds'], test_forecast['yhat'], label='Pred', linewidth=1.6)
        plt.fill_between(test_forecast['ds'], 
                        test_forecast['yhat_lower'], 
                        test_forecast['yhat_upper'], 
                        alpha=0.3, label='Uncertainty')
        plt.xlabel("Date")
        plt.ylabel("Visitor Count")
        plt.title("Prophet: Test Set - True vs Pred")
        plt.xticks(rotation=30)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "prophet_true_vs_pred.png", dpi=150)
        plt.close()
    except Exception as e:
        print(f"保存测试集对比图失败: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prophet 客流预测训练脚本（统计基线模型）")
    parser.add_argument(
        "--input-csv",
        default="data/raw/jiuzhaigou_tourism_weather_2024_2026_latest.csv",
        help="输入 CSV（需包含 date + 客流列）。",
    )
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--peak-quantile", type=float, default=0.75)
    parser.add_argument("--output-dir", default="output")
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
    parser.add_argument(
        "--use-holidays",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否使用节假日特征。默认 True。",
    )
    args, _ = parser.parse_known_args()

    if not PROPHET_AVAILABLE:
        print("错误: Prophet 未安装，无法运行。")
        print("安装命令: pip install prophet")
        return

    np.random.seed(42)

    output_dir = Path(args.output_dir)
    runs_dir = output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    
    auto_run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_lb0_ep0_prophet_8features"
    run_name = args.run_name or auto_run_name
    run_name_pattern = r"^run_\d{8}_\d{6}_.+$"
    if not re.fullmatch(run_name_pattern, run_name):
        print(f"警告: run_name '{run_name}' 不符合常规模式，但将继续使用。")
    
    run_dir = runs_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    metrics_json_path = run_dir / "prophet_metrics.json"
    metrics_csv_path = run_dir / "prophet_metrics.csv"
    pred_path = run_dir / "prophet_test_predictions.csv"
    forecast_path = run_dir / "prophet_full_forecast.csv"
    fig_dir = run_dir / "figures"

    # 1. 加载数据并特征工程
    df = load_and_engineer_features(Path(args.input_csv))

    # 2. 对目标客流做 MinMax 归一化（与LSTM/GRU保持一致）
    scaler = MinMaxScaler()
    df["visitor_count_scaled"] = scaler.fit_transform(df[["visitor_count"]]).reshape(-1)
    
    # 3. 准备Prophet格式数据（多变量版本）
    train_df, test_df, full_df, regressor_cols = prepare_prophet_data(df, test_ratio=args.test_ratio)
    print(f"使用 {len(regressor_cols)} 个回归器特征: {regressor_cols}")

    # 3. 创建节假日数据（如果需要）
    holidays_df = None
    if args.use_holidays:
        try:
            holidays_df = create_chinese_holidays_df(df)
            print(f"创建了 {len(holidays_df)} 个节假日记录")
        except Exception as e:
            print(f"创建节假日数据失败: {e}")
            holidays_df = None

    # 4. 创建并训练Prophet模型（多变量版本）
    print("训练Prophet模型（多变量）...")
    model = create_prophet_model(regressor_cols, holidays_df)
    
    # 训练模型
    model.fit(train_df)
    
    # 5. 预测（包括测试期）
    # 对于多变量Prophet，需要使用包含回归器特征的完整数据
    # 注意：Prophet只能预测训练数据的时间范围，不能外推回归器特征
    forecast_df = model.predict(full_df)
    
    # 6. 提取测试期预测结果
    test_forecast = forecast_df[forecast_df['ds'].isin(test_df['ds'])].copy()
    
    # 对齐测试集和预测结果
    test_merged = pd.merge(test_df, test_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], 
                          on='ds', how='left')
    
    # 反归一化：将归一化的预测值转换回原始客流
    y_true_scaled = test_merged['y'].values
    y_pred_scaled = test_merged['yhat'].values
    
    # 使用之前创建的scaler进行反归一化
    y_true = scaler.inverse_transform(y_true_scaled.reshape(-1, 1)).reshape(-1)
    y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).reshape(-1)
    
    # 7. 保存预测结果
    pred_df = pd.DataFrame({
        'date': test_merged['ds'],
        'y_true': y_true,
        'y_pred': y_pred,
        'y_pred_lower': scaler.inverse_transform(test_merged['yhat_lower'].values.reshape(-1, 1)).reshape(-1),
        'y_pred_upper': scaler.inverse_transform(test_merged['yhat_upper'].values.reshape(-1, 1)).reshape(-1)
    })
    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")
    
    # 保存完整预测（反归一化后）
    full_forecast = forecast_df.copy()
    full_forecast['yhat'] = scaler.inverse_transform(full_forecast['yhat'].values.reshape(-1, 1)).reshape(-1)
    full_forecast['yhat_lower'] = scaler.inverse_transform(full_forecast['yhat_lower'].values.reshape(-1, 1)).reshape(-1)
    full_forecast['yhat_upper'] = scaler.inverse_transform(full_forecast['yhat_upper'].values.reshape(-1, 1)).reshape(-1)
    full_forecast.to_csv(forecast_path, index=False, encoding="utf-8-sig")

    # 8. 使用通用评估器计算指标
    # 使用反归一化后的训练数据计算阈值
    train_scaled = train_df['y'].values if 'y' in train_df.columns else None
    
    metrics = calculate_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_train_scaled=train_scaled,
        scaler=scaler,
        peak_quantile=args.peak_quantile,
    )

    # 9. 保存指标
    additional_info = {
        "samples": int(len(df)),
        "train_samples": int(len(train_df)),
        "test_samples": int(len(test_df)),
        "use_holidays": args.use_holidays,
        "holiday_count": len(holidays_df) if holidays_df is not None else 0,
        "regressor_count": len(regressor_cols),
        "regressors": regressor_cols,
        "model_params": {
            "yearly_seasonality": True,
            "weekly_seasonality": True,
            "daily_seasonality": False,
            "seasonality_mode": "additive",
            "changepoint_prior_scale": 0.05,
            "seasonality_prior_scale": 10.0,
            "holidays_prior_scale": 10.0,
        }
    }
    
    save_metrics_to_files(
        metrics=metrics,
        run_dir=str(run_dir),
        run_name=run_name,
        model_name="prophet",
        additional_info=additional_info,
    )

    # 10. 保存可视化图
    if args.save_plots:
        save_prophet_plots(fig_dir, model, train_df, forecast_df, test_df)

    # 11. 输出结果
    print(f"Prophet训练完成！")
    print(f"运行目录: {run_dir}")
    print(f"预测结果: {pred_path}")
    print(f"完整预测: {forecast_path}")
    
    print("\n回归指标:")
    for key, value in metrics["regression"].items():
        print(f"  {key.upper()}: {value:.4f}")
    
    print("\n分类指标 (高峰日预测):")
    for key, value in metrics["classification"].items():
        if key != "peak_threshold":
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value:.2f}")
    
    # 12. 输出预测区间信息
    if not test_merged.empty and 'yhat_lower' in test_merged.columns and 'yhat_upper' in test_merged.columns:
        coverage = np.mean((y_true >= test_merged['yhat_lower'].values) & 
                          (y_true <= test_merged['yhat_upper'].values))
        avg_interval_width = np.mean(test_merged['yhat_upper'].values - test_merged['yhat_lower'].values)
        print(f"\n预测区间统计:")
        print(f"  真实值在预测区间内的比例: {coverage:.2%}")
        print(f"  平均预测区间宽度: {avg_interval_width:.2f}")


if __name__ == "__main__":
    main()