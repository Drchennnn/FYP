#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Part 4: Deep Ensemble + Conformal Calibration Interval

Principle:
  Combines two methods:
  1. Deep Ensemble: N independently trained GRU models capture epistemic
     uncertainty via member disagreement (std).
  2. Conformal Calibration: The ensemble std alone is too narrow to cover
     the actual prediction error (std << MAE). To fix this, we use the
     validation set to calibrate a scale factor q_hat such that:

       nonconformity score: s_i = |y_i - mean_i| / (std_i + eps)
       q_hat = ceil((n+1)(1-alpha))/n quantile of {s_1,...,s_n}
       test interval: [mean - q_hat * (std + eps), mean + q_hat * (std + eps)]

  This gives a coverage guarantee IF the val/test distributions are similar,
  AND produces ADAPTIVE intervals: high-uncertainty samples (large std) get
  wider intervals, low-uncertainty samples get narrower ones.

  If val/test have distribution shift, use --cal-source recent (splits test
  set in half: first half calibrates, second half evaluates).

Output:
  output/uncertainty/ensemble_<timestamp>/
    ensemble_interval.csv    -- date, y_true, y_pred_mean, lower, upper, std, q_hat_scaled
    ensemble_metrics.json    -- PICP, MPIW, Winkler Score, q_hat, per-member MAE
    ensemble_interval.png    -- visualization (top: interval, bottom: adaptive width)

Usage:
  python scripts/uncertainty_ensemble.py
  python scripts/uncertainty_ensemble.py --alpha 0.10
  python scripts/uncertainty_ensemble.py --cal-source recent
  python scripts/uncertainty_ensemble.py --ensemble-dir output/runs/gru_ensemble_XXXXXXXX
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
EPS = 1.0  # prevent division by zero in std normalisation (1 visitor floor)


# ─────────────────────────────────────────────
# Ensemble directory discovery
# ─────────────────────────────────────────────

def find_latest_ensemble() -> Path | None:
    pattern = str(PROJECT_ROOT / 'output' / 'runs' / 'gru_ensemble_*')
    dirs = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    return Path(dirs[0]) if dirs else None


def load_ensemble_members(ensemble_dir: Path):
    """Load all member models. Returns (list_of_models, ensemble_info_dict)."""
    import tensorflow as tf

    info_path = ensemble_dir / 'ensemble_info.json'
    if not info_path.exists():
        raise FileNotFoundError(f'ensemble_info.json not found in {ensemble_dir}')
    with open(info_path) as f:
        info = json.load(f)

    models = []
    for m in info['members']:
        weight_path = m['weight_path']
        if not Path(weight_path).exists():
            weight_path = str(
                ensemble_dir / f"member_{m['member_idx']}" / 'weights' / 'gru_jiuzhaigou.h5'
            )
        model = tf.keras.models.load_model(weight_path, compile=False)
        models.append(model)
        print(f"  member_{m['member_idx']} (seed={m['seed']}, val_loss={m['val_loss']:.6f})")

    return models, info


# ─────────────────────────────────────────────
# Data loading — returns cal + test sets
# ─────────────────────────────────────────────

def load_cal_and_test_data(cal_source: str = 'val'):
    """
    Load calibration and test data.

    cal_source:
      'val'    -- calibration = validation set (standard split conformal)
      'recent' -- calibration = first half of test set; evaluation = second half
                  (reduces temporal distribution shift at the cost of test size)

    Returns: (x_cal, y_cal, x_test, y_test, d_test, scaler)
    """
    from sklearn.preprocessing import MinMaxScaler

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

    visitor_vals = pd.to_numeric(df['tourism_num'], errors='coerce').dropna().values
    scaler = MinMaxScaler()
    scaler.fit(visitor_vals.reshape(-1, 1))

    df['visitor_count_scaled'] = scaler.transform(
        pd.to_numeric(df['tourism_num'], errors='coerce').values.reshape(-1, 1)
    ).flatten()

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
    val_size = int(trainval_size * 0.111)
    train_size = trainval_size - val_size

    if cal_source == 'recent':
        full_test = X[trainval_size:]
        full_y = Y[trainval_size:]
        full_d = D[trainval_size:]
        mid = len(full_test) // 2
        x_cal, y_cal = full_test[:mid], full_y[:mid]
        x_test, y_test, d_test = full_test[mid:], full_y[mid:], full_d[mid:]
    else:
        x_cal = X[train_size:train_size + val_size]
        y_cal = Y[train_size:train_size + val_size]
        x_test = X[trainval_size:]
        y_test = Y[trainval_size:]
        d_test = D[trainval_size:]

    return x_cal, y_cal, x_test, y_test, d_test, scaler


# ─────────────────────────────────────────────
# Ensemble inference helper
# ─────────────────────────────────────────────

def ensemble_predict(models, X: np.ndarray, scaler) -> tuple[np.ndarray, np.ndarray]:
    """
    Run all members on X. Returns (mean_raw, std_raw) in original visitor units.
    """
    n_members = len(models)
    preds_scaled = np.array([
        m.predict(X, verbose=0).flatten() for m in models
    ])  # (N_members, N)
    preds_raw = scaler.inverse_transform(
        preds_scaled.reshape(-1, 1)
    ).reshape(n_members, -1)
    return preds_raw.mean(axis=0), preds_raw.std(axis=0)


# ─────────────────────────────────────────────
# Metrics
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
# Main
# ─────────────────────────────────────────────

def run(ensemble_dir: Path | None = None, alpha: float = 0.10, cal_source: str = 'val'):
    if ensemble_dir is None:
        ensemble_dir = find_latest_ensemble()
    if ensemble_dir is None:
        raise RuntimeError(
            'No gru_ensemble_* run found. Run scripts/train_gru_ensemble.py first.'
        )

    print(f'\n{"="*60}')
    print(f'Deep Ensemble + Conformal Calibration  alpha={alpha}  CI={(1-alpha)*100:.0f}%')
    print(f'Calibration source: {cal_source}')
    print(f'Ensemble dir: {ensemble_dir}')
    print(f'{"="*60}')

    # ── 1. Load ensemble ──
    print('Loading ensemble members...')
    models, ensemble_info = load_ensemble_members(ensemble_dir)
    n_members = len(models)
    print(f'Loaded {n_members} members')

    # ── 2. Load data ──
    x_cal, y_cal, x_test, y_test, d_test, scaler = load_cal_and_test_data(cal_source)
    print(f'Calibration size: {len(x_cal)}, Test size: {len(x_test)}')

    y_cal_raw = scaler.inverse_transform(y_cal.reshape(-1, 1)).flatten()
    y_test_raw = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    # ── 3. Calibration: compute normalised nonconformity scores ──
    print('Calibration inference (all members)...')
    cal_mean, cal_std = ensemble_predict(models, x_cal, scaler)

    # Normalised nonconformity score: |residual| / (std + eps)
    # Captures "how many std-widths does the true error span?"
    scores = np.abs(y_cal_raw - cal_mean) / (cal_std + EPS)

    n_cal = len(scores)
    level = min(np.ceil((n_cal + 1) * (1 - alpha)) / n_cal, 1.0)
    q_hat = float(np.quantile(scores, level))

    print(f'Normalised scores: mean={scores.mean():.2f}, p90={np.quantile(scores, 0.9):.2f}, max={scores.max():.2f}')
    print(f'q_hat = {q_hat:.4f}  (conformal scale factor,  n={n_cal}, alpha={alpha})')

    # ── 4. Test set: adaptive conformal interval ──
    print('Test set inference (all members)...')
    test_mean, test_std = ensemble_predict(models, x_test, scaler)

    half_width = q_hat * (test_std + EPS)
    lower = test_mean - half_width
    upper = test_mean + half_width

    # ── 5. Metrics ──
    picp = compute_picp(y_test_raw, lower, upper)
    mpiw = compute_mpiw(lower, upper)
    winkler = compute_winkler(y_test_raw, lower, upper, alpha)
    ensemble_mae = float(np.mean(np.abs(test_mean - y_test_raw)))

    print(f'\nEvaluation (test set):')
    print(f'  PICP  = {picp:.4f}  (target >= {1-alpha:.2f})')
    print(f'  MPIW  = {mpiw:.1f} visitors')
    print(f'  Winkler Score = {winkler:.1f}')
    print(f'  Ensemble mean MAE = {ensemble_mae:.1f} visitors')
    print(f'  Member std mean = {test_std.mean():.1f},  max = {test_std.max():.1f}')
    print(f'  Half-width mean = {half_width.mean():.1f},  max = {half_width.max():.1f}')
    print(f'  q_hat = {q_hat:.4f}  (std multiplier from conformal calibration)')

    # ── 6. Save ──
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = OUTPUT_DIR / f'ensemble_{timestamp}'
    out_dir.mkdir(parents=True, exist_ok=True)

    df_out = pd.DataFrame({
        'date': d_test,
        'y_true': y_test_raw.round(1),
        'y_pred_mean': test_mean.round(1),
        'lower': lower.round(1),
        'upper': upper.round(1),
        'std': test_std.round(1),
        'half_width': half_width.round(1),
    })
    df_out.to_csv(out_dir / 'ensemble_interval.csv', index=False)

    metrics = {
        'method': 'deep_ensemble_conformal',
        'cal_source': cal_source,
        'n_members': n_members,
        'seeds': ensemble_info['seeds'],
        'alpha': alpha,
        'confidence_level': 1 - alpha,
        'cal_size': int(n_cal),
        'test_size': int(len(x_test)),
        'q_hat': round(q_hat, 4),
        'eps': EPS,
        'picp': round(picp, 4),
        'mpiw': round(mpiw, 1),
        'winkler_score': round(winkler, 1),
        'ensemble_mae': round(ensemble_mae, 1),
        'std_mean': round(float(test_std.mean()), 1),
        'std_max': round(float(test_std.max()), 1),
        'half_width_mean': round(float(half_width.mean()), 1),
        'half_width_max': round(float(half_width.max()), 1),
        'ensemble_dir': str(ensemble_dir),
        'generated_at': datetime.now().isoformat(),
    }
    with open(out_dir / 'ensemble_metrics.json', 'w', encoding='utf-8') as f:
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
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

        # Top: interval + actual
        ax = axes[0]
        ax.fill_between(dates, df['lower'], df['upper'],
                        alpha=0.25, color='seagreen',
                        label=f'{(1-alpha)*100:.0f}% Ensemble+Conformal Interval')
        ax.plot(dates, df['y_true'], color='black', linewidth=1.2, label='Actual')
        ax.plot(dates, df['y_pred_mean'], color='seagreen', linewidth=1.0,
                linestyle='--', label='Ensemble Mean Pred')
        ax.set_title(
            f'Deep Ensemble + Conformal ({metrics["n_members"]} members)  |  '
            f'PICP={metrics["picp"]:.3f}  MPIW={metrics["mpiw"]:.0f}  '
            f'Winkler={metrics["winkler_score"]:.0f}  q_hat={metrics["q_hat"]:.2f}'
        )
        ax.set_ylabel('Visitor Count')
        ax.legend()
        ax.grid(alpha=0.3)

        # Bottom: adaptive half-width (shows where model is more/less uncertain)
        ax2 = axes[1]
        ax2.fill_between(dates, 0, df['half_width'], alpha=0.5, color='seagreen',
                         label='Conformal half-width')
        ax2.plot(dates, df['std'], color='darkgreen', linewidth=0.8,
                 linestyle='--', label='Raw ensemble std')
        ax2.set_ylabel('Interval Half-Width (visitors)')
        ax2.set_xlabel('Date')
        ax2.set_title(
            f'Adaptive Interval Width  |  '
            f'half-width = q_hat({metrics["q_hat"]:.2f}) × (ensemble_std + {metrics["eps"]})'
        )
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.xticks(rotation=30)
        ax2.legend()
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(out_dir / 'ensemble_interval.png', dpi=150)
        plt.close()
        print('Plot saved: ensemble_interval.png')
    except Exception as e:
        print(f'Visualization failed (results unaffected): {e}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deep Ensemble + Conformal Calibration Interval')
    parser.add_argument('--ensemble-dir', default=None,
                        help='Path to gru_ensemble_* directory. Default: latest.')
    parser.add_argument('--alpha', type=float, default=0.10,
                        help='Significance level, default 0.10 (90%% CI)')
    parser.add_argument('--cal-source', default='val', choices=['val', 'recent'],
                        help='Calibration source: val (validation set) or '
                             'recent (first half of test set, reduces distribution shift)')
    args = parser.parse_args()
    ensemble_dir = Path(args.ensemble_dir) if args.ensemble_dir else None
    run(ensemble_dir=ensemble_dir, alpha=args.alpha, cal_source=args.cal_source)
