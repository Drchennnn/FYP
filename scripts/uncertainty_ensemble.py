#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Part 4: Deep Ensemble Uncertainty Interval

Principle:
  Load N independently trained GRU models (different random seeds).
  For each test sample, collect N predictions and compute:
    - mean   = ensemble mean prediction
    - std    = standard deviation across members (epistemic uncertainty)
    - lower  = alpha/2 quantile of member predictions
    - upper  = 1 - alpha/2 quantile of member predictions

  Deep Ensembles (Lakshminarayanan et al., 2017) consistently outperform
  MC Dropout for uncertainty quantification in practice.

Output:
  output/uncertainty/ensemble_<timestamp>/
    ensemble_interval.csv    -- date, y_true, y_pred_mean, lower, upper, std
    ensemble_metrics.json    -- PICP, MPIW, Winkler Score, per-member MAE
    ensemble_interval.png    -- visualization

Usage:
  python scripts/uncertainty_ensemble.py
  python scripts/uncertainty_ensemble.py --alpha 0.10
  python scripts/uncertainty_ensemble.py --ensemble-dir output/runs/gru_ensemble_20260405_XXXXXX
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
# Ensemble directory discovery
# ─────────────────────────────────────────────

def find_latest_ensemble() -> Path | None:
    """Find the most recently modified gru_ensemble_* run directory."""
    pattern = str(PROJECT_ROOT / 'output' / 'runs' / 'gru_ensemble_*')
    dirs = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    return Path(dirs[0]) if dirs else None


def load_ensemble_members(ensemble_dir: Path) -> list:
    """Load all member models from ensemble directory. Returns list of keras models."""
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
            # Try relative path fallback
            weight_path = str(ensemble_dir / f"member_{m['member_idx']}" / 'weights' / 'gru_jiuzhaigou.h5')
        model = tf.keras.models.load_model(weight_path, compile=False)
        models.append(model)
        print(f"  Loaded member_{m['member_idx']} (seed={m['seed']}, val_loss={m['val_loss']:.6f})")

    return models, info


# ─────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────

def load_test_data():
    """Load test set. Returns (x_test, y_test, d_test, scaler)"""
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

    return X[trainval_size:], Y[trainval_size:], D[trainval_size:], scaler


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

def run(ensemble_dir: Path | None = None, alpha: float = 0.10):
    if ensemble_dir is None:
        ensemble_dir = find_latest_ensemble()
    if ensemble_dir is None:
        raise RuntimeError(
            'No gru_ensemble_* run found. Run scripts/train_gru_ensemble.py first.'
        )

    print(f'\n{"="*60}')
    print(f'Deep Ensemble Interval  alpha={alpha}  CI={(1-alpha)*100:.0f}%')
    print(f'Ensemble dir: {ensemble_dir}')
    print(f'{"="*60}')

    # ── 1. Load ensemble ──
    print('Loading ensemble members...')
    models, ensemble_info = load_ensemble_members(ensemble_dir)
    n_members = len(models)
    print(f'Loaded {n_members} members')

    # ── 2. Load test data ──
    x_test, y_test, d_test, scaler = load_test_data()
    print(f'Test size: {len(x_test)}')

    # ── 3. Predict with each member (deterministic, training=False) ──
    print('Running inference for each member...')
    preds_scaled = []
    for i, model in enumerate(models):
        pred = model.predict(x_test, verbose=0).flatten()
        preds_scaled.append(pred)
        print(f'  Member {i} done')

    preds_scaled = np.array(preds_scaled)  # (N_members, N_test)

    # Inverse transform each member's predictions
    preds_raw = scaler.inverse_transform(
        preds_scaled.reshape(-1, 1)
    ).reshape(n_members, -1)  # (N_members, N_test)

    y_test_raw = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    # ── 4. Compute ensemble statistics ──
    y_pred_mean = preds_raw.mean(axis=0)
    y_pred_std = preds_raw.std(axis=0)
    lower = np.quantile(preds_raw, alpha / 2, axis=0)
    upper = np.quantile(preds_raw, 1 - alpha / 2, axis=0)

    # ── 5. Metrics ──
    picp = compute_picp(y_test_raw, lower, upper)
    mpiw = compute_mpiw(lower, upper)
    winkler = compute_winkler(y_test_raw, lower, upper, alpha)

    # Per-member MAE for diagnostics
    member_maes = [
        round(float(np.mean(np.abs(preds_raw[i] - y_test_raw))), 1)
        for i in range(n_members)
    ]
    ensemble_mae = float(np.mean(np.abs(y_pred_mean - y_test_raw)))

    print(f'\nEvaluation (test set):')
    print(f'  PICP  = {picp:.4f}  (target >= {1-alpha:.2f})')
    print(f'  MPIW  = {mpiw:.1f} visitors')
    print(f'  Winkler Score = {winkler:.1f}')
    print(f'  Ensemble mean MAE = {ensemble_mae:.1f} visitors')
    print(f'  Std mean = {y_pred_std.mean():.1f}, Std max = {y_pred_std.max():.1f}')
    print(f'  Per-member MAE: {member_maes}')

    # ── 6. Save results ──
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = OUTPUT_DIR / f'ensemble_{timestamp}'
    out_dir.mkdir(parents=True, exist_ok=True)

    df_out = pd.DataFrame({
        'date': d_test,
        'y_true': y_test_raw.round(1),
        'y_pred_mean': y_pred_mean.round(1),
        'lower': lower.round(1),
        'upper': upper.round(1),
        'std': y_pred_std.round(1),
    })
    df_out.to_csv(out_dir / 'ensemble_interval.csv', index=False)

    metrics = {
        'method': 'deep_ensemble',
        'n_members': n_members,
        'seeds': ensemble_info['seeds'],
        'alpha': alpha,
        'confidence_level': 1 - alpha,
        'test_size': int(len(x_test)),
        'picp': round(picp, 4),
        'mpiw': round(mpiw, 1),
        'winkler_score': round(winkler, 1),
        'ensemble_mae': round(ensemble_mae, 1),
        'std_mean': round(float(y_pred_std.mean()), 1),
        'std_max': round(float(y_pred_std.max()), 1),
        'std_min': round(float(y_pred_std.min()), 1),
        'member_maes': member_maes,
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
                        label=f'{(1-alpha)*100:.0f}% Deep Ensemble Interval')
        ax.plot(dates, df['y_true'], color='black', linewidth=1.2, label='Actual')
        ax.plot(dates, df['y_pred_mean'], color='seagreen', linewidth=1.0,
                linestyle='--', label='Ensemble Mean Pred')
        ax.set_title(
            f'Deep Ensemble ({metrics["n_members"]} members)  |  '
            f'PICP={metrics["picp"]:.3f}  MPIW={metrics["mpiw"]:.0f}  '
            f'Winkler={metrics["winkler_score"]:.0f}  MAE={metrics["ensemble_mae"]:.0f}'
        )
        ax.set_ylabel('Visitor Count')
        ax.legend()
        ax.grid(alpha=0.3)

        # Bottom: ensemble std (disagreement between members)
        ax2 = axes[1]
        ax2.fill_between(dates, 0, df['std'], alpha=0.5, color='seagreen')
        ax2.set_ylabel('Ensemble Std (visitors)')
        ax2.set_xlabel('Date')
        ax2.set_title('Ensemble Member Disagreement (epistemic uncertainty)')
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.xticks(rotation=30)
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(out_dir / 'ensemble_interval.png', dpi=150)
        plt.close()
        print('Plot saved: ensemble_interval.png')
    except Exception as e:
        print(f'Visualization failed (results unaffected): {e}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deep Ensemble Uncertainty Interval')
    parser.add_argument('--ensemble-dir', default=None,
                        help='Path to gru_ensemble_* directory. Default: latest.')
    parser.add_argument('--alpha', type=float, default=0.10,
                        help='Significance level, default 0.10 (90%% CI)')
    args = parser.parse_args()
    ensemble_dir = Path(args.ensemble_dir) if args.ensemble_dir else None
    run(ensemble_dir=ensemble_dir, alpha=args.alpha)
