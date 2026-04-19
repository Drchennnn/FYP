"""
XGBoost 基线模型 (Gradient Boosting Baseline)

传统机器学习对比基线。时序特征展开为表格形式（lag + rolling + 外生变量），
与 GRU/LSTM/Seq2Seq 同一测试集、同一评估框架，保证对比公平性。

特征工程策略：
  - 目标滞后：lag_1, lag_7, lag_14, lag_28（人工构造时序依赖）
  - 滚动统计：rolling_mean_7, rolling_mean_14, rolling_std_7
  - 气象特征：precip, temp_high, temp_low（当天值）
  - 时间编码：month_norm, day_of_week_norm, is_holiday, is_peak_season（新增）
  - 新增特征：lag_14_scaled, rolling_mean_14_scaled（P0 扩展）

共约 14 维输入，无序列结构，单步预测。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import chinese_calendar as cncal
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
from xgboost import XGBRegressor

# ── 项目根路径 ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.common.core_evaluation import evaluate_and_save_run, compute_dynamic_peak_threshold
from models.common.evaluator import calculate_metrics, save_metrics_to_files

# ── 常量 ─────────────────────────────────────────────────────────────────────
DEFAULT_DATA_PATH = (
    PROJECT_ROOT
    / "data/processed/jiuzhaigou_daily_features_2016-01-01_2026-04-02.csv"
)
OUTPUT_BASE = PROJECT_ROOT / "output" / "runs"

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

# ── 节假日标记（与 GRU 保持一致）─────────────────────────────────────────────
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


def is_peak_season(date_val: pd.Timestamp) -> int:
    """旺季标记：4月1日 ~ 11月15日"""
    m, d = int(date_val.month), int(date_val.day)
    if m < 4 or m > 11:
        return 0
    if m == 11 and d > 15:
        return 0
    return 1


# ── 特征工程 ──────────────────────────────────────────────────────────────────
def load_and_engineer_features(input_csv: Path) -> tuple[pd.DataFrame, MinMaxScaler]:
    """
    加载原始 CSV，构造 XGBoost 所需的表格特征。

    返回:
        df:     包含所有特征列和目标列（visitor_count_scaled）的 DataFrame
        scaler: 已拟合的 visitor_count MinMaxScaler（用于反归一化预测值）

    TODO (Codex):
        1. 读取 CSV，解析 date 列为 DatetimeIndex
        2. 构造以下列（参考 GRU 脚本的 load_and_engineer_features）：
           - month_norm, day_of_week_norm, is_holiday, is_peak_season
           - tourism_num_lag_1_scaled, lag_7_scaled, lag_14_scaled, lag_28_scaled
           - rolling_mean_7_scaled, rolling_mean_14_scaled, rolling_std_7_scaled
           - meteo_precip_sum_scaled, temp_high_scaled, temp_low_scaled
           - visitor_count_scaled（目标列，MinMax 基于训练集拟合）
        3. 删除因 lag/rolling 产生的 NaN 行（dropna）
        4. 返回 (df, scaler)
    """
    df = pd.read_csv(input_csv, encoding="utf-8-sig")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)
    df = df[df["date"] >= "2023-06-01"].reset_index(drop=True)

    if "tourism_num" in df.columns:
        target_col = "tourism_num"
    elif "visitor_count" in df.columns:
        target_col = "visitor_count"
    else:
        raise ValueError("未找到目标列，请包含 'tourism_num' 或 'visitor_count'。")

    df["visitor_count"] = pd.to_numeric(df[target_col], errors="coerce")
    df = df.dropna(subset=["visitor_count"]).reset_index(drop=True)

    # 时间特征
    df["month_norm"] = (df["date"].dt.month - 1) / 11.0
    df["day_of_week_norm"] = df["date"].dt.weekday / 6.0
    df["is_holiday"] = df["date"].apply(mark_core_holiday).astype(float)
    df["is_peak_season"] = df["date"].apply(is_peak_season).astype(float)
    # 节假日距离特征
    def days_to_next_hol(date_val: pd.Timestamp) -> float:
        for delta in range(1, 15):
            try:
                if cncal.is_holiday((date_val + pd.Timedelta(days=delta)).date()):
                    return delta / 14.0
            except Exception:
                pass
        return 1.0

    def days_since_last_hol(date_val: pd.Timestamp) -> float:
        for delta in range(1, 15):
            try:
                if cncal.is_holiday((date_val - pd.Timedelta(days=delta)).date()):
                    return delta / 14.0
            except Exception:
                pass
        return 1.0

    df["days_to_next_holiday"] = df["date"].apply(days_to_next_hol).astype(float)
    df["days_since_last_holiday"] = df["date"].apply(days_since_last_hol).astype(float)

    # 目标序列特征（先构造原始值）
    df["tourism_num_lag_1"] = df["visitor_count"].shift(1)
    df["tourism_num_lag_7"] = df["visitor_count"].shift(7)
    df["tourism_num_lag_14"] = df["visitor_count"].shift(14)
    df["tourism_num_rolling_mean_7"] = df["visitor_count"].rolling(7).mean()
    df["tourism_num_rolling_std_7"] = df["visitor_count"].rolling(7).std()

    # 天气原始列（统一命名便于后续评估）
    def _pick_col(candidates: list[str]) -> str | None:
        for col in candidates:
            if col in df.columns:
                return col
        return None

    precip_src = _pick_col(["meteo_precip_sum", "meteo_rain_sum", "precip_sum"])
    temp_high_src = _pick_col(["meteo_temp_max", "temp_high_c", "temp_high"])
    temp_low_src = _pick_col(["meteo_temp_min", "temp_low_c", "temp_low"])

    if precip_src is None:
        df["precip_raw"] = 0.0
    else:
        df["precip_raw"] = pd.to_numeric(df[precip_src], errors="coerce")

    if temp_high_src is None:
        df["temp_high_raw"] = 0.0
    else:
        df["temp_high_raw"] = pd.to_numeric(df[temp_high_src], errors="coerce")

    if temp_low_src is None:
        df["temp_low_raw"] = 0.0
    else:
        df["temp_low_raw"] = pd.to_numeric(df[temp_low_src], errors="coerce")

    # 丢弃由 lag/rolling 或天气缺失导致的空值
    required_raw_cols = [
        "visitor_count",
        "tourism_num_lag_1",
        "tourism_num_lag_7",
        "tourism_num_lag_14",
        "tourism_num_rolling_mean_7",
        "tourism_num_rolling_std_7",
        "precip_raw",
        "temp_high_raw",
        "temp_low_raw",
    ]
    df = df.dropna(subset=required_raw_cols).reset_index(drop=True)

    # 仅使用训练集拟合 scaler，再转换全量数据
    train_end = int(len(df) * TRAIN_RATIO)
    if train_end <= 0:
        raise ValueError("数据量不足，无法完成训练集拟合。")

    def _fit_transform_col(raw_col: str, scaled_col: str) -> MinMaxScaler:
        scaler = MinMaxScaler()
        scaler.fit(df.iloc[:train_end][[raw_col]])
        df[scaled_col] = scaler.transform(df[[raw_col]]).reshape(-1)
        return scaler

    visitor_scaler = _fit_transform_col("visitor_count", "visitor_count_scaled")
    _fit_transform_col("tourism_num_lag_1", "tourism_num_lag_1_scaled")
    _fit_transform_col("tourism_num_lag_7", "tourism_num_lag_7_scaled")
    _fit_transform_col("tourism_num_lag_14", "tourism_num_lag_14_scaled")
    _fit_transform_col("tourism_num_rolling_mean_7", "rolling_mean_7_scaled")
    _fit_transform_col("tourism_num_rolling_std_7", "rolling_std_7_scaled")
    _fit_transform_col("precip_raw", "meteo_precip_sum_scaled")
    _fit_transform_col("temp_high_raw", "temp_high_scaled")
    _fit_transform_col("temp_low_raw", "temp_low_scaled")

    return df, visitor_scaler


# ── 训练/验证/测试切分 ────────────────────────────────────────────────────────
def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """按时间顺序切分：80/10/10"""
    n = len(df)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    return train_df, val_df, test_df


# ── 特征列定义 ────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "month_norm",
    "day_of_week_norm",
    "is_holiday",
    "is_peak_season",
    "days_to_next_holiday",
    "days_since_last_holiday",
    "tourism_num_lag_1_scaled",
    "tourism_num_lag_7_scaled",
    "tourism_num_lag_14_scaled",
    "rolling_mean_7_scaled",
    "meteo_precip_sum_scaled",
    "temp_high_scaled",
    "temp_low_scaled",
]
TARGET_COL = "visitor_count_scaled"


# ── 模型定义 ──────────────────────────────────────────────────────────────────
def build_model(args: argparse.Namespace):
    """
    构建 XGBRegressor。

    TODO (Codex):
        使用以下超参（基准值，供调优起点）：
            n_estimators=300
            max_depth=6
            learning_rate=0.05
            subsample=0.8
            colsample_bytree=0.8
            reg_alpha=0.5
            reg_lambda=1.0
            random_state=42
            n_jobs=-1
        如果 args.sample_weight 为 True，在 fit() 时传入样本权重向量
        （COVID/地震异常期 2020-01-01~2022-12-31 权重=0.3，其余=1.0）
    """
    return XGBRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.lr,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=3,
        reg_alpha=0.3,
        reg_lambda=1.5,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=40,
    )


def compute_sample_weights(train_df: pd.DataFrame) -> np.ndarray:
    """
    对 COVID/地震异常期赋低权重，减少其对模型的干扰。

    TODO (Codex):
        异常期定义：2020-01-01 ~ 2022-12-31
        权重：异常期=0.3，其余=1.0
        返回 shape=(len(train_df),) 的 np.ndarray
    """
    mask = (train_df["date"] >= "2020-01-01") & (train_df["date"] <= "2022-12-31")
    weights = np.where(mask, 0.3, 1.0)
    return weights.astype(np.float32)


# ── 主流程 ────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="XGBoost 客流预测基线")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--sample-weight", action="store_true", default=True,
                        help="对异常期赋低权重（默认开启）")
    parser.add_argument("--peak-quantile", type=float, default=0.75)
    args = parser.parse_args()

    np.random.seed(42)

    # ── 数据准备 ────────────────────────────────────────────────────────────
    df, scaler = load_and_engineer_features(args.data)
    train_df, val_df, test_df = split_data(df)

    X_train = train_df[FEATURE_COLS]
    y_train = train_df[TARGET_COL]
    X_val = val_df[FEATURE_COLS]
    y_val = val_df[TARGET_COL]
    X_test = test_df[FEATURE_COLS]
    y_test = test_df[TARGET_COL]

    # ── 输出目录（与 GRU/LSTM 相同结构）────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_{ts}_xgboost_8features"
    runs_dir = OUTPUT_BASE / f"xgboost_8features_{ts}" / "runs"
    run_dir = runs_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "weights").mkdir(exist_ok=True)
    (run_dir / "figures").mkdir(exist_ok=True)

    # ── 训练 ────────────────────────────────────────────────────────────────
    model = build_model(args)
    weights = compute_sample_weights(train_df) if args.sample_weight else None
    fit_kwargs = {
        "eval_set": [(X_val, y_val)],
        "verbose": False,
    }
    if weights is not None:
        fit_kwargs["sample_weight"] = weights
    model.fit(X_train, y_train, **fit_kwargs)
    model.save_model(run_dir / "weights" / "xgboost_model.json")

    # ── 预测与反归一化 ───────────────────────────────────────────────────────
    y_pred_scaled = model.predict(X_test)
    y_true = scaler.inverse_transform(y_test.values.reshape(-1, 1)).ravel()
    y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    y_pred = np.clip(y_pred, 0, None)

    pred_df = pd.DataFrame(
        {
            "date": pd.to_datetime(test_df["date"]),
            "y_true": y_true,
            "y_pred": y_pred,
        }
    ).sort_values("date")
    pred_df.to_csv(run_dir / "xgboost_test_predictions.csv", index=False, encoding="utf-8-sig")

    # ── 评估（复用统一框架）────────────────────────────────────────────────
    dynamic_peak_threshold = compute_dynamic_peak_threshold(
        scaler.inverse_transform(train_df[TARGET_COL].values.reshape(-1, 1)).ravel(),
        args.peak_quantile,
    )

    weather_train_precip = train_df["precip_raw"].values.astype(float)
    weather_train_temp_high = train_df["temp_high_raw"].values.astype(float)
    weather_train_temp_low = train_df["temp_low_raw"].values.astype(float)
    weather_test_precip = test_df["precip_raw"].values.astype(float)
    weather_test_temp_high = test_df["temp_high_raw"].values.astype(float)
    weather_test_temp_low = test_df["temp_low_raw"].values.astype(float)

    evaluate_and_save_run(
        str(run_dir),
        model_name="xgboost",
        feature_count=len(FEATURE_COLS),
        y_true=np.asarray(y_true),
        y_pred=np.asarray(y_pred),
        dates=pd.to_datetime(test_df["date"]).values,
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
            "n_estimators": int(args.n_estimators),
            "max_depth": int(args.max_depth),
            "learning_rate": float(args.lr),
            "sample_weight": bool(args.sample_weight),
            "xgboost_version": xgb.__version__,
        },
    )

    metrics = calculate_metrics(y_true=y_true, y_pred=y_pred, scaler=scaler)
    try:
        save_metrics_to_files(metrics, str(run_dir), "xgboost_baseline")
    except TypeError:
        pass

    print("XGBoost 训练完成！")
    print(f"运行目录: {run_dir}")
    print(f"特征维度: {len(FEATURE_COLS)}")
    print(f"MAE: {metrics['regression']['mae']:.4f}")


if __name__ == "__main__":
    main()
