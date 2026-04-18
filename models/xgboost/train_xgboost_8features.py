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

# ── 项目根路径 ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.common.core_evaluation import evaluate_and_save_run, compute_dynamic_peak_threshold
from models.common.evaluator import calculate_metrics, save_metrics_to_files

# ── TODO: Codex 需要安装并导入 xgboost ───────────────────────────────────────
# pip install xgboost
# import xgboost as xgb
# from xgboost import XGBRegressor

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
    # --- SKELETON ---
    df = pd.read_csv(input_csv)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # TODO: 构造 lag / rolling / 时间编码特征（见上方文档注释）
    raise NotImplementedError("TODO: implement feature engineering in load_and_engineer_features()")


# ── 训练/验证/测试切分 ────────────────────────────────────────────────────────
def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """按 80/10/10 时序切分，不打乱顺序"""
    n = len(df)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]


# ── 特征列定义 ────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "month_norm",
    "day_of_week_norm",
    "is_holiday",
    "is_peak_season",
    "tourism_num_lag_1_scaled",
    "tourism_num_lag_7_scaled",
    "tourism_num_lag_14_scaled",
    "tourism_num_lag_28_scaled",
    "rolling_mean_7_scaled",
    "rolling_mean_14_scaled",
    "rolling_std_7_scaled",
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
    # TODO: return XGBRegressor(...)
    raise NotImplementedError("TODO: implement build_model()")


def compute_sample_weights(train_df: pd.DataFrame) -> np.ndarray:
    """
    对 COVID/地震异常期赋低权重，减少其对模型的干扰。

    TODO (Codex):
        异常期定义：2020-01-01 ~ 2022-12-31
        权重：异常期=0.3，其余=1.0
        返回 shape=(len(train_df),) 的 np.ndarray
    """
    # TODO: implement
    raise NotImplementedError("TODO: implement compute_sample_weights()")


# ── 主流程 ────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="XGBoost 客流预测基线")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--sample-weight", action="store_true", default=True,
                        help="对异常期赋低权重（默认开启）")
    parser.add_argument("--peak-quantile", type=float, default=0.75)
    args = parser.parse_args()

    # ── 数据准备 ────────────────────────────────────────────────────────────
    # TODO (Codex): 调用 load_and_engineer_features()，拿到 df 和 scaler
    # train_df, val_df, test_df = split_data(df)
    # X_train, y_train = train_df[FEATURE_COLS], train_df[TARGET_COL]
    # X_val,   y_val   = val_df[FEATURE_COLS],   val_df[TARGET_COL]
    # X_test,  y_test  = test_df[FEATURE_COLS],  test_df[TARGET_COL]

    # ── 输出目录（与 GRU/LSTM 相同结构）────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_{ts}_xgboost_8features"
    runs_dir = OUTPUT_BASE / f"xgboost_8features_{ts}" / "runs"
    run_dir = runs_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "weights").mkdir(exist_ok=True)
    (run_dir / "figures").mkdir(exist_ok=True)

    # ── 训练 ────────────────────────────────────────────────────────────────
    # TODO (Codex):
    #   model = build_model(args)
    #   weights = compute_sample_weights(train_df) if args.sample_weight else None
    #   model.fit(X_train, y_train,
    #             sample_weight=weights,
    #             eval_set=[(X_val, y_val)],
    #             verbose=50)
    #   model.save_model(run_dir / "weights" / "xgboost_model.json")

    # ── 预测与反归一化 ───────────────────────────────────────────────────────
    # TODO (Codex):
    #   y_pred_scaled = model.predict(X_test)
    #   y_true = scaler.inverse_transform(y_test.values.reshape(-1,1)).ravel()
    #   y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1,1)).ravel()
    #   y_pred = np.clip(y_pred, 0, None)

    # ── 评估（复用统一框架）────────────────────────────────────────────────
    # TODO (Codex):
    #   dynamic_peak_threshold = compute_dynamic_peak_threshold(
    #       scaler.inverse_transform(train_df[TARGET_COL].values.reshape(-1,1)).ravel(),
    #       args.peak_quantile
    #   )
    #   ... 提取 weather_test_precip, temp_high, temp_low（同 GRU 脚本）...
    #   evaluate_and_save_run(
    #       str(run_dir),
    #       model_name="xgboost",
    #       feature_count=len(FEATURE_COLS),
    #       y_true=y_true,
    #       y_pred=y_pred,
    #       dates=pd.to_datetime(test_df["date"]).values,
    #       horizon=1,
    #       peak_threshold=dynamic_peak_threshold,
    #       warning_temperature=1000.0,
    #       fn_fp_cost_ratio=(5.0, 1.0),
    #       weather_precip=...,
    #       weather_temp_high=...,
    #       weather_temp_low=...,
    #       weather_train_precip=...,
    #       weather_train_temp_high=...,
    #       weather_train_temp_low=...,
    #       extra_meta={"n_estimators": args.n_estimators, "max_depth": args.max_depth},
    #   )

    print("XGBoost 框架骨架已就绪，等待 Codex 补全 TODO 部分。")


if __name__ == "__main__":
    main()
