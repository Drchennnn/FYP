#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Part 2：MC Dropout 区间（Monte Carlo Dropout Interval）

原理：
  推理阶段保持 GRU/LSTM 的 Dropout 层开启（training=True），
  对同一输入重复采样 T 次，取均值为中心预测，alpha/2 和 1-alpha/2
  分位数为区间上下界。模型权重不变，仅推理方式改变。

输出：
  output/uncertainty/mc_dropout_<timestamp>/
    mc_dropout_interval.csv   — 测试集日期、y_true、y_pred_mean、lower、upper、std
    mc_dropout_metrics.json   — PICP、MPIW、Winkler Score、T、std 统计
    mc_dropout_interval.png   — 可视化图

用法：
  python scripts/uncertainty_mc_dropout.py
  python scripts/uncertainty_mc_dropout.py --T 100 --alpha 0.10 --model gru_8features
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
# 工具函数（与 uncertainty_residual.py 相同的数据加载逻辑）
# ─────────────────────────────────────────────

def find_latest_run(model_key: str) -> Path | None:
    pattern = str(PROJECT_ROOT / 'output' / 'runs' / f'{model_key}_*' / 'runs' / 'run_*')
    dirs = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    return Path(dirs[0]) if dirs else None


def load_model_and_data(run_dir: Path, model_key: str):
    """加载模型权重和数据集，返回 (model, x_test, y_test, d_test, scaler)"""
    import tensorflow as tf
    from sklearn.preprocessing import MinMaxScaler

    weight_files = glob.glob(str(run_dir / 'weights' / '*.h5')) + \
                   glob.glob(str(run_dir / 'weights' / '*.keras'))
    if not weight_files:
        raise FileNotFoundError(f'No weight file in {run_dir}/weights/')
    model = tf.keras.models.load_model(weight_files[0], compile=False)

    csv_files = sorted(glob.glob(str(PROJECT_ROOT / 'data' / 'processed' / '*.csv')))
    if not csv_files:
        raise FileNotFoundError('No processed CSV found.')
    df = pd.read_csv(csv_files[-1])
    df = df.sort_values('date').drop_duplicates(subset=['date']).reset_index(drop=True)
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

    x_list, y_list, d_list = [], [], []
    for i in range(look_back, len(values)):
        x_list.append(values[i - look_back:i])
        y_list.append(target[i])
        d_list.append(dates[i])
    X = np.array(x_list, dtype=np.float32)
    Y = np.array(y_list, dtype=np.float32)
    D = np.array(d_list)

    n = len(X)
    test_size = int(n * 0.10)
    trainval_size = n - test_size

    x_test = X[trainval_size:]
    y_test = Y[trainval_size:]
    d_test = D[trainval_size:]

    visitor_vals = pd.to_numeric(df['tourism_num'], errors='coerce').dropna().values
    scaler = MinMaxScaler()
    scaler.fit(visitor_vals.reshape(-1, 1))

    return model, x_test, y_test, d_test, scaler


# ─────────────────────────────────────────────
# MC Dropout 推理
# ─────────────────────────────────────────────

def mc_dropout_predict(model, X: np.ndarray, T: int, batch_size: int = 64) -> np.ndarray:
    """
    标准 MC Dropout 推理：training=True 保持 GRU 内置 dropout 激活。
    要求模型使用 GRU(dropout=0.2, implementation=1) 而非独立 Dropout 层。
    返回 shape (T, N) 的预测矩阵（scaled 空间）。
    """
    all_preds = []
    for t in range(T):
        preds = []
        for i in range(0, len(X), batch_size):
            batch = X[i:i + batch_size]
            out = model(batch, training=True).numpy().flatten()
            preds.append(out)
        all_preds.append(np.concatenate(preds))
        if (t + 1) % 10 == 0:
            print(f'  Sampling: {t+1}/{T}', end='\r')
    print()
    return np.array(all_preds)  # (T, N)


# ─────────────────────────────────────────────
# 评估指标（与 Part 1 相同）
# ─────────────────────────────────────────────

def compute_picp(y_true, lower, upper):
    return float(np.mean((y_true >= lower) & (y_true <= upper)))

def compute_mpiw(lower, upper):
    return float(np.mean(upper - lower))

def compute_winkler(y_true, lower, upper, alpha):
    width = upper - lower
    penalty_low = np.where(y_true < lower, 2 / alpha * (lower - y_true), 0.0)
    penalty_high = np.where(y_true > upper, 2 / alpha * (y_true - upper), 0.0)
    return float(np.mean(width + penalty_low + penalty_high))


# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────

def run(model_key: str = 'gru_8features', T: int = 50, alpha: float = 0.10):
    print(f'\n{"="*60}')
    print(f'MC Dropout 区间  model={model_key}  T={T}  alpha={alpha}  CI={(1-alpha)*100:.0f}%')
    print(f'{"="*60}')

    run_dir = find_latest_run(model_key)
    if run_dir is None:
        raise RuntimeError(f'No run found for model key: {model_key}')
    print(f'Run dir: {run_dir}')

    model, x_test, y_test, d_test, scaler = load_model_and_data(run_dir, model_key)
    print(f'Test size: {len(x_test)}, T={T} 次采样')

    # ── 1. MC Dropout 采样 ──
    print(f'开始 MC Dropout 采样（T={T}）...')
    preds_scaled = mc_dropout_predict(model, x_test, T)  # (T, N)

    # 反归一化到原始空间
    preds_raw = scaler.inverse_transform(
        preds_scaled.reshape(-1, 1)
    ).reshape(T, -1)  # (T, N)

    y_test_raw = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    # ── 2. 计算区间 ──
    y_pred_mean = preds_raw.mean(axis=0)
    y_pred_std = preds_raw.std(axis=0)
    lower = np.quantile(preds_raw, alpha / 2, axis=0)
    upper = np.quantile(preds_raw, 1 - alpha / 2, axis=0)

    # ── 3. 评估指标 ──
    picp = compute_picp(y_test_raw, lower, upper)
    mpiw = compute_mpiw(lower, upper)
    winkler = compute_winkler(y_test_raw, lower, upper, alpha)

    print(f'\n评估结果（测试集）:')
    print(f'  PICP  = {picp:.4f}  (目标 ≥ {1-alpha:.2f})')
    print(f'  MPIW  = {mpiw:.1f} 人')
    print(f'  Winkler Score = {winkler:.1f}')
    print(f'  预测标准差均值 = {y_pred_std.mean():.1f} 人')
    print(f'  预测标准差最大 = {y_pred_std.max():.1f} 人（高峰期）')

    # ── 4. 保存结果 ──
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = OUTPUT_DIR / f'mc_dropout_{timestamp}'
    out_dir.mkdir(parents=True, exist_ok=True)

    df_out = pd.DataFrame({
        'date': d_test,
        'y_true': y_test_raw.round(1),
        'y_pred_mean': y_pred_mean.round(1),
        'lower': lower.round(1),
        'upper': upper.round(1),
        'std': y_pred_std.round(1),
    })
    df_out.to_csv(out_dir / 'mc_dropout_interval.csv', index=False)

    metrics = {
        'method': 'mc_dropout',
        'model': model_key,
        'T': T,
        'alpha': alpha,
        'confidence_level': 1 - alpha,
        'test_size': int(len(x_test)),
        'picp': round(picp, 4),
        'mpiw': round(mpiw, 1),
        'winkler_score': round(winkler, 1),
        'std_mean': round(float(y_pred_std.mean()), 1),
        'std_max': round(float(y_pred_std.max()), 1),
        'std_min': round(float(y_pred_std.min()), 1),
        'generated_at': datetime.now().isoformat(),
    }
    with open(out_dir / 'mc_dropout_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

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
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

        # 上图：区间 + 实际值
        ax = axes[0]
        ax.fill_between(dates, df['lower'], df['upper'],
                        alpha=0.25, color='darkorange',
                        label=f'{(1-alpha)*100:.0f}% MC Dropout Interval')
        ax.plot(dates, df['y_true'], color='black', linewidth=1.2, label='Actual')
        ax.plot(dates, df['y_pred_mean'], color='darkorange', linewidth=1.0,
                linestyle='--', label='MC Mean Pred')
        ax.set_title(
            f'MC Dropout Interval  |  '
            f'PICP={metrics["picp"]:.3f}  MPIW={metrics["mpiw"]:.0f}  '
            f'Winkler={metrics["winkler_score"]:.0f}  T={metrics["T"]}'
        )
        ax.set_ylabel('Visitor Count')
        ax.legend()
        ax.grid(alpha=0.3)

        # 下图：预测标准差（动态不确定性）
        ax2 = axes[1]
        ax2.fill_between(dates, 0, df['std'], alpha=0.5, color='darkorange')
        ax2.set_ylabel('Prediction Std (visitors)')
        ax2.set_xlabel('Date')
        ax2.set_title('MC Dropout Prediction Std (wider during peak seasons)')
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.xticks(rotation=30)
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(out_dir / 'mc_dropout_interval.png', dpi=150)
        plt.close()
        print('图表已保存: mc_dropout_interval.png')
    except Exception as e:
        print(f'可视化失败（不影响结果）: {e}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MC Dropout 置信区间')
    parser.add_argument('--model', default='gru_8features',
                        help='模型 key，默认 gru_8features')
    parser.add_argument('--T', type=int, default=50,
                        help='MC 采样次数，默认 50')
    parser.add_argument('--alpha', type=float, default=0.10,
                        help='显著性水平，默认 0.10（90%% CI）')
    args = parser.parse_args()
    run(model_key=args.model, T=args.T, alpha=args.alpha)
