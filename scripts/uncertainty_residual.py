#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Part 1：残差分位数区间（Residual Quantile Interval）

原理：
  用 GRU 单步模型在验证集上的预测残差分布，提取 alpha/2 和 1-alpha/2 分位数
  作为固定偏移量，叠加到测试集点预测值上，形成 (1-alpha) 置信区间。
  模型权重和推理过程不变，计算开销为零。

输出：
  output/uncertainty/residual_<timestamp>/
    residual_interval.csv   — 测试集日期、y_true、y_pred、lower、upper
    residual_metrics.json   — PICP、MPIW、Winkler Score、分位数偏移量
    residual_interval.png   — 可视化图

用法：
  python scripts/uncertainty_residual.py
  python scripts/uncertainty_residual.py --alpha 0.10 --model gru
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_DIR = PROJECT_ROOT / 'output' / 'uncertainty'


# ─────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────

def find_latest_run(model_key: str) -> Path | None:
    pattern = str(PROJECT_ROOT / 'output' / 'runs' / f'{model_key}_*' / 'runs' / 'run_*')
    dirs = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    return Path(dirs[0]) if dirs else None


def load_model_and_data(run_dir: Path, model_key: str):
    """加载模型权重和处理后的数据集，返回 (model, x_val, y_val, x_test, y_test, d_test, scaler)"""
    import tensorflow as tf
    from sklearn.preprocessing import MinMaxScaler

    # 加载模型
    weight_files = glob.glob(str(run_dir / 'weights' / '*.h5')) + \
                   glob.glob(str(run_dir / 'weights' / '*.keras'))
    if not weight_files:
        raise FileNotFoundError(f'No weight file found in {run_dir}/weights/')
    model = tf.keras.models.load_model(weight_files[0], compile=False)

    # 加载处理数据
    csv_files = sorted(glob.glob(str(PROJECT_ROOT / 'data' / 'processed' / '*.csv')))
    if not csv_files:
        raise FileNotFoundError('No processed CSV found.')
    df = pd.read_csv(csv_files[-1])
    df = df.sort_values('date').drop_duplicates(subset=['date']).reset_index(drop=True)
    # 过滤掉 append_future_weather 追加的未来行
    df = df[pd.to_numeric(df['tourism_num'], errors='coerce').notna()].reset_index(drop=True)
    # month_norm / day_of_week_norm 可能在 CSV 里为 NaN，直接从原始列计算
    df['date_dt'] = pd.to_datetime(df['date'])
    df['month_norm'] = (df['date_dt'].dt.month - 1) / 11.0
    df['day_of_week_norm'] = df['date_dt'].dt.weekday / 6.0

    feature_cols = [
        'visitor_count_scaled', 'month_norm', 'day_of_week_norm', 'is_holiday',
        'tourism_num_lag_7_scaled', 'meteo_precip_sum_scaled',
        'temp_high_scaled', 'temp_low_scaled',
    ]
    for col in feature_cols:
        if col not in df.columns:
            raise ValueError(f'Missing feature column: {col}')

    values = df[feature_cols].values.astype(np.float32)
    target = df['visitor_count_scaled'].values.astype(np.float32)
    dates = df['date'].values
    look_back = 30

    # 构建滑动窗口
    x_list, y_list, d_list = [], [], []
    for i in range(look_back, len(values)):
        x_list.append(values[i - look_back:i])
        y_list.append(target[i])
        d_list.append(dates[i])
    X = np.array(x_list, dtype=np.float32)
    Y = np.array(y_list, dtype=np.float32)
    D = np.array(d_list)

    # 与训练脚本相同的切分比例
    n = len(X)
    test_ratio = 0.10
    val_ratio_of_trainval = 0.111  # val ≈ 10% of total → 0.10 / 0.90 ≈ 0.111
    test_size = int(n * test_ratio)
    trainval_size = n - test_size
    val_size = int(trainval_size * val_ratio_of_trainval)
    train_size = trainval_size - val_size

    x_val = X[train_size:train_size + val_size]
    y_val = Y[train_size:train_size + val_size]
    x_test = X[trainval_size:]
    y_test = Y[trainval_size:]
    d_test = D[trainval_size:]

    # visitor scaler（用全量 tourism_num 拟合，与训练时一致）
    visitor_vals = pd.to_numeric(df['tourism_num'], errors='coerce').dropna().values
    scaler = MinMaxScaler()
    scaler.fit(visitor_vals.reshape(-1, 1))

    return model, x_val, y_val, x_test, y_test, d_test, scaler


# ─────────────────────────────────────────────
# 评估指标
# ─────────────────────────────────────────────

def compute_picp(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Prediction Interval Coverage Probability"""
    return float(np.mean((y_true >= lower) & (y_true <= upper)))


def compute_mpiw(lower: np.ndarray, upper: np.ndarray) -> float:
    """Mean Prediction Interval Width"""
    return float(np.mean(upper - lower))


def compute_winkler(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float) -> float:
    """Winkler Score（越小越好）"""
    width = upper - lower
    penalty_low = np.where(y_true < lower, 2 / alpha * (lower - y_true), 0.0)
    penalty_high = np.where(y_true > upper, 2 / alpha * (y_true - upper), 0.0)
    return float(np.mean(width + penalty_low + penalty_high))


# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────

def run(model_key: str = 'gru_8features', alpha: float = 0.10):
    print(f'\n{"="*60}')
    print(f'残差分位数区间  model={model_key}  alpha={alpha}  CI={(1-alpha)*100:.0f}%')
    print(f'{"="*60}')

    run_dir = find_latest_run(model_key)
    if run_dir is None:
        raise RuntimeError(f'No run found for model key: {model_key}')
    print(f'Run dir: {run_dir}')

    model, x_val, y_val, x_test, y_test, d_test, scaler = load_model_and_data(run_dir, model_key)
    print(f'Val size: {len(x_val)}, Test size: {len(x_test)}')

    # ── 1. 验证集推理，计算残差 ──
    print('在验证集上推理...')
    y_val_pred_scaled = model.predict(x_val, verbose=0).flatten()
    y_val_true_raw = scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()
    y_val_pred_raw = scaler.inverse_transform(y_val_pred_scaled.reshape(-1, 1)).flatten()
    residuals = y_val_true_raw - y_val_pred_raw  # 正值=低估，负值=高估

    q_low = float(np.quantile(residuals, alpha / 2))       # 负数（下界偏移）
    q_high = float(np.quantile(residuals, 1 - alpha / 2))  # 正数（上界偏移）
    print(f'残差分位数: Q{alpha/2*100:.1f}%={q_low:.1f}, Q{(1-alpha/2)*100:.1f}%={q_high:.1f}')

    # ── 2. 测试集推理，叠加区间 ──
    print('在测试集上推理...')
    y_test_pred_scaled = model.predict(x_test, verbose=0).flatten()
    y_test_true_raw = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_test_pred_raw = scaler.inverse_transform(y_test_pred_scaled.reshape(-1, 1)).flatten()

    lower = y_test_pred_raw + q_low
    upper = y_test_pred_raw + q_high

    # ── 3. 计算评估指标 ──
    picp = compute_picp(y_test_true_raw, lower, upper)
    mpiw = compute_mpiw(lower, upper)
    winkler = compute_winkler(y_test_true_raw, lower, upper, alpha)

    print(f'\n评估结果（测试集）:')
    print(f'  PICP  = {picp:.4f}  (目标 ≥ {1-alpha:.2f})')
    print(f'  MPIW  = {mpiw:.1f} 人')
    print(f'  Winkler Score = {winkler:.1f}')

    # ── 4. 保存结果 ──
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = OUTPUT_DIR / f'residual_{timestamp}'
    out_dir.mkdir(parents=True, exist_ok=True)

    # CSV
    df_out = pd.DataFrame({
        'date': d_test,
        'y_true': y_test_true_raw.round(1),
        'y_pred': y_test_pred_raw.round(1),
        'lower': lower.round(1),
        'upper': upper.round(1),
    })
    df_out.to_csv(out_dir / 'residual_interval.csv', index=False)

    # metrics JSON
    metrics = {
        'method': 'residual_quantile',
        'model': model_key,
        'alpha': alpha,
        'confidence_level': 1 - alpha,
        'val_size': int(len(x_val)),
        'test_size': int(len(x_test)),
        'q_low': round(q_low, 2),
        'q_high': round(q_high, 2),
        'interval_width_fixed': round(q_high - q_low, 2),
        'picp': round(picp, 4),
        'mpiw': round(mpiw, 1),
        'winkler_score': round(winkler, 1),
        'generated_at': datetime.now().isoformat(),
    }
    with open(out_dir / 'residual_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # ── 5. 可视化 ──
    _plot(df_out, metrics, out_dir, alpha)

    print(f'\n输出目录: {out_dir}')
    return metrics


def _plot(df: pd.DataFrame, metrics: dict, out_dir: Path, alpha: float):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        dates = pd.to_datetime(df['date'])
        fig, ax = plt.subplots(figsize=(14, 5))

        ax.fill_between(dates, df['lower'], df['upper'],
                        alpha=0.25, color='steelblue',
                        label=f'{(1-alpha)*100:.0f}% Residual Interval')
        ax.plot(dates, df['y_true'], color='black', linewidth=1.2, label='Actual')
        ax.plot(dates, df['y_pred'], color='steelblue', linewidth=1.0,
                linestyle='--', label='GRU Pred')

        ax.set_title(
            f'Residual Quantile Interval  |  '
            f'PICP={metrics["picp"]:.3f}  MPIW={metrics["mpiw"]:.0f}  '
            f'Winkler={metrics["winkler_score"]:.0f}'
        )
        ax.set_xlabel('Date')
        ax.set_ylabel('Visitor Count')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.xticks(rotation=30)
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / 'residual_interval.png', dpi=150)
        plt.close()
        print('图表已保存: residual_interval.png')
    except Exception as e:
        print(f'可视化失败（不影响结果）: {e}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='残差分位数置信区间')
    parser.add_argument('--model', default='gru_8features',
                        help='模型 key，默认 gru_8features')
    parser.add_argument('--alpha', type=float, default=0.10,
                        help='显著性水平，默认 0.10（90%% CI）')
    args = parser.parse_args()
    run(model_key=args.model, alpha=args.alpha)
