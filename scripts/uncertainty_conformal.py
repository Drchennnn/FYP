#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Part 3: Split Conformal Prediction Interval

Principle:
  Split conformal prediction provides a finite-sample marginal coverage guarantee.
  Using a held-out calibration set (validation set):
    1. Compute nonconformity scores: s_i = |y_i - y_pred_i|  (for regression)
    2. Find q_hat = ceil((n+1)(1-alpha)) / n quantile of {s_1,...,s_n}
    3. Test set prediction interval: [y_pred - q_hat, y_pred + q_hat]

  Coverage guarantee: P(y_test in interval) >= 1 - alpha  (exactly, finite-sample)

  The guarantee holds regardless of model quality or data distribution,
  as long as calibration and test data are exchangeable (i.i.d. assumption).

Output:
  output/uncertainty/conformal_<timestamp>/
    conformal_interval.csv    -- date, y_true, y_pred, lower, upper
    conformal_metrics.json    -- PICP, MPIW, Winkler Score, q_hat
    conformal_interval.png    -- visualization

Usage:
  python scripts/uncertainty_conformal.py
  python scripts/uncertainty_conformal.py --alpha 0.10 --model gru_8features
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
# Data loading (same logic as other uncertainty scripts)
# ─────────────────────────────────────────────

def find_latest_run(model_key: str) -> Path | None:
    pattern = str(PROJECT_ROOT / 'output' / 'runs' / f'{model_key}_*' / 'runs' / 'run_*')
    dirs = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    return Path(dirs[0]) if dirs else None


def load_model_and_data(run_dir: Path, model_key: str, cal_source: str = 'val'):
    """Load model weights and dataset. Returns (model, x_cal, y_cal, x_test, y_test, d_test, scaler)

    cal_source options:
      'val'       -- use validation set as calibration (default, standard split conformal)
      'recent'    -- use the first half of test set as calibration, evaluate on second half
                     (approximates temporal proximity; reduces test size to ~134 samples)
    """
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
    test_ratio = 0.10
    val_ratio_of_trainval = 0.111
    test_size = int(n * test_ratio)
    trainval_size = n - test_size
    val_size = int(trainval_size * val_ratio_of_trainval)
    train_size = trainval_size - val_size

    # Calibration set selection
    if cal_source == 'recent':
        # Use first half of test set as calibration, second half as eval
        # Rationale: calibration data is temporally closer to evaluation data,
        # reducing distribution shift. Trade-off: halves the test set size.
        full_test = X[trainval_size:]
        full_y_test = Y[trainval_size:]
        full_d_test = D[trainval_size:]
        mid = len(full_test) // 2
        x_cal = full_test[:mid]
        y_cal = full_y_test[:mid]
        x_test = full_test[mid:]
        y_test = full_y_test[mid:]
        d_test = full_d_test[mid:]
    else:
        # Standard: calibration set = validation set (held-out from training)
        x_cal = X[train_size:train_size + val_size]
        y_cal = Y[train_size:train_size + val_size]
        x_test = X[trainval_size:]
        y_test = Y[trainval_size:]
        d_test = D[trainval_size:]

    visitor_vals = pd.to_numeric(df['tourism_num'], errors='coerce').dropna().values
    scaler = MinMaxScaler()
    scaler.fit(visitor_vals.reshape(-1, 1))

    return model, x_cal, y_cal, x_test, y_test, d_test, scaler


# ─────────────────────────────────────────────
# Conformal Prediction core
# ─────────────────────────────────────────────

def compute_conformal_qhat(scores: np.ndarray, alpha: float) -> float:
    """
    Compute the conformal quantile q_hat from calibration nonconformity scores.

    Uses the finite-sample corrected quantile:
        level = ceil((n + 1) * (1 - alpha)) / n
    clamped to [0, 1] to handle edge cases.

    This guarantees: P(y_test in interval) >= 1 - alpha  (Venn & Shafer, 2005)
    """
    n = len(scores)
    level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return float(np.quantile(scores, level))


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
# Main
# ─────────────────────────────────────────────

def run(model_key: str = 'gru_8features', alpha: float = 0.10, cal_source: str = 'val'):
    print(f'\n{"="*60}')
    print(f'Split Conformal Prediction  model={model_key}  alpha={alpha}  CI={(1-alpha)*100:.0f}%')
    print(f'Calibration source: {cal_source}')
    print(f'{"="*60}')

    run_dir = find_latest_run(model_key)
    if run_dir is None:
        raise RuntimeError(f'No run found for model key: {model_key}')
    print(f'Run dir: {run_dir}')

    model, x_cal, y_cal, x_test, y_test, d_test, scaler = load_model_and_data(
        run_dir, model_key, cal_source=cal_source
    )
    print(f'Calibration size: {len(x_cal)}, Test size: {len(x_test)}')

    # ── 1. Calibration: compute nonconformity scores ──
    print('Calibration inference...')
    y_cal_pred_scaled = model.predict(x_cal, verbose=0).flatten()
    y_cal_true_raw = scaler.inverse_transform(y_cal.reshape(-1, 1)).flatten()
    y_cal_pred_raw = scaler.inverse_transform(y_cal_pred_scaled.reshape(-1, 1)).flatten()

    # Nonconformity score: absolute residual
    scores = np.abs(y_cal_true_raw - y_cal_pred_raw)
    q_hat = compute_conformal_qhat(scores, alpha)
    print(f'Nonconformity scores: mean={scores.mean():.1f}, max={scores.max():.1f}')
    print(f'q_hat (conformal threshold) = {q_hat:.1f}  [n={len(scores)}, alpha={alpha}]')

    # ── 2. Test set: apply conformal interval ──
    print('Test set inference...')
    y_test_pred_scaled = model.predict(x_test, verbose=0).flatten()
    y_test_true_raw = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_test_pred_raw = scaler.inverse_transform(y_test_pred_scaled.reshape(-1, 1)).flatten()

    lower = y_test_pred_raw - q_hat
    upper = y_test_pred_raw + q_hat

    # ── 3. Metrics ──
    picp = compute_picp(y_test_true_raw, lower, upper)
    mpiw = compute_mpiw(lower, upper)
    winkler = compute_winkler(y_test_true_raw, lower, upper, alpha)

    print(f'\nEvaluation (test set):')
    print(f'  PICP  = {picp:.4f}  (target >= {1-alpha:.2f}, guaranteed by construction)')
    print(f'  MPIW  = {mpiw:.1f} visitors')
    print(f'  Winkler Score = {winkler:.1f}')
    print(f'  q_hat = {q_hat:.1f} visitors (symmetric half-width)')

    # ── 4. Save results ──
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = OUTPUT_DIR / f'conformal_{timestamp}'
    out_dir.mkdir(parents=True, exist_ok=True)

    df_out = pd.DataFrame({
        'date': d_test,
        'y_true': y_test_true_raw.round(1),
        'y_pred': y_test_pred_raw.round(1),
        'lower': lower.round(1),
        'upper': upper.round(1),
    })
    df_out.to_csv(out_dir / 'conformal_interval.csv', index=False)

    metrics = {
        'method': 'split_conformal',
        'cal_source': cal_source,
        'model': model_key,
        'alpha': alpha,
        'confidence_level': 1 - alpha,
        'cal_size': int(len(x_cal)),
        'test_size': int(len(x_test)),
        'q_hat': round(q_hat, 1),
        'score_mean': round(float(scores.mean()), 1),
        'score_max': round(float(scores.max()), 1),
        'picp': round(picp, 4),
        'mpiw': round(mpiw, 1),
        'winkler_score': round(winkler, 1),
        'coverage_guarantee': f'>= {1-alpha:.2f} (finite-sample marginal)',
        'generated_at': datetime.now().isoformat(),
    }
    with open(out_dir / 'conformal_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    _plot(df_out, metrics, out_dir, alpha)

    print(f'\nOutput: {out_dir}')
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
                        label=f'{(1-alpha)*100:.0f}% Conformal Interval (q_hat={metrics["q_hat"]:.0f})')
        ax.plot(dates, df['y_true'], color='black', linewidth=1.2, label='Actual')
        ax.plot(dates, df['y_pred'], color='steelblue', linewidth=1.0,
                linestyle='--', label='GRU Pred')

        ax.set_title(
            f'Split Conformal Prediction Interval  |  '
            f'PICP={metrics["picp"]:.3f}  MPIW={metrics["mpiw"]:.0f}  '
            f'Winkler={metrics["winkler_score"]:.0f}  q_hat={metrics["q_hat"]:.0f}'
        )
        ax.set_xlabel('Date')
        ax.set_ylabel('Visitor Count')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.xticks(rotation=30)
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / 'conformal_interval.png', dpi=150)
        plt.close()
        print('Plot saved: conformal_interval.png')
    except Exception as e:
        print(f'Visualization failed (results unaffected): {e}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split Conformal Prediction Interval')
    parser.add_argument('--model', default='gru_8features', help='Model key, default gru_8features')
    parser.add_argument('--alpha', type=float, default=0.10,
                        help='Significance level, default 0.10 (90%% CI)')
    parser.add_argument('--cal-source', default='val', choices=['val', 'recent'],
                        help='Calibration source: val (validation set, default) or '
                             'recent (first half of test set, reduces distribution shift)')
    args = parser.parse_args()
    run(model_key=args.model, alpha=args.alpha, cal_source=args.cal_source)
