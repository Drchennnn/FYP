#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train GRU Deep Ensemble (5 members, different random seeds)

Principle:
  Train N independent GRU models with the same architecture but different
  random initialization seeds. Each model learns a slightly different
  mapping due to stochastic weight initialization and mini-batch ordering.
  Their disagreement captures model (epistemic) uncertainty.

Output:
  output/runs/gru_ensemble_<timestamp>/
    member_0/weights/gru_jiuzhaigou.h5   (seed=42)
    member_1/weights/gru_jiuzhaigou.h5   (seed=7)
    member_2/weights/gru_jiuzhaigou.h5   (seed=13)
    member_3/weights/gru_jiuzhaigou.h5   (seed=99)
    member_4/weights/gru_jiuzhaigou.h5   (seed=2024)
    ensemble_info.json                   (seeds, val_losses, data split info)

Usage:
  python scripts/train_gru_ensemble.py
  python scripts/train_gru_ensemble.py --epochs 120 --members 5
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

ENSEMBLE_SEEDS = [42, 7, 13, 99, 2024]


# ─────────────────────────────────────────────
# Data loading (reuses same pipeline as train_gru_8features.py)
# ─────────────────────────────────────────────

def load_data():
    """Load and prepare data. Returns (x_train, y_train, x_val, y_val, x_test, y_test, d_test, scaler)"""
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

    # Build visitor scaler first (fit on full data to match training script)
    visitor_vals = pd.to_numeric(df['tourism_num'], errors='coerce').dropna().values
    scaler = MinMaxScaler()
    scaler.fit(visitor_vals.reshape(-1, 1))

    # Scale visitor_count using tourism_num scaler (same as training)
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
    test_ratio = 0.10
    val_ratio_of_trainval = 0.111
    test_size = int(n * test_ratio)
    trainval_size = n - test_size
    val_size = int(trainval_size * val_ratio_of_trainval)
    train_size = trainval_size - val_size

    x_train = X[:train_size]
    y_train = Y[:train_size]
    x_val = X[train_size:train_size + val_size]
    y_val = Y[train_size:train_size + val_size]
    x_test = X[trainval_size:]
    y_test = Y[trainval_size:]
    d_test = D[trainval_size:]

    print(f'Data split: train={train_size}, val={val_size}, test={len(x_test)}')
    return x_train, y_train, x_val, y_val, x_test, y_test, d_test, scaler


def create_gru_model(look_back: int = 30):
    """Same architecture as train_gru_8features.py (implementation=1 for MC Dropout compat)"""
    import tensorflow as tf
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(look_back, 8)),
        tf.keras.layers.GRU(128, return_sequences=True, dropout=0.2, implementation=1),
        tf.keras.layers.GRU(64, dropout=0.2, implementation=1),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.Huber(),
    )
    return model


def train_one_member(
    seed: int,
    member_idx: int,
    x_train, y_train, x_val, y_val,
    out_dir: Path,
    epochs: int,
    batch_size: int,
) -> dict:
    """Train one ensemble member with given seed. Returns val_loss and weight path."""
    import tensorflow as tf

    np.random.seed(seed)
    tf.random.set_seed(seed)

    print(f'\n{"─"*50}')
    print(f'Member {member_idx}  (seed={seed})')
    print(f'{"─"*50}')

    member_dir = out_dir / f'member_{member_idx}'
    weights_dir = member_dir / 'weights'
    weights_dir.mkdir(parents=True, exist_ok=True)
    weight_path = weights_dir / 'gru_jiuzhaigou.h5'

    model = create_gru_model()
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=15, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=7, min_lr=1e-5
        ),
    ]

    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=False,
        callbacks=callbacks,
        verbose=1,
    )

    model.save(str(weight_path))
    val_loss = float(min(history.history['val_loss']))
    epochs_trained = len(history.history['loss'])
    print(f'  Saved: {weight_path}  val_loss={val_loss:.6f}  epochs={epochs_trained}')

    return {
        'member_idx': member_idx,
        'seed': seed,
        'weight_path': str(weight_path),
        'val_loss': round(val_loss, 6),
        'epochs_trained': epochs_trained,
    }


def run(epochs: int = 120, members: int = 5, batch_size: int = 32):
    print(f'\n{"="*60}')
    print(f'GRU Deep Ensemble Training  members={members}  epochs={epochs}')
    print(f'{"="*60}')

    x_train, y_train, x_val, y_val, x_test, y_test, d_test, scaler = load_data()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = PROJECT_ROOT / 'output' / 'runs' / f'gru_ensemble_{timestamp}'
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = ENSEMBLE_SEEDS[:members]
    member_infos = []

    for idx, seed in enumerate(seeds):
        info = train_one_member(
            seed=seed,
            member_idx=idx,
            x_train=x_train, y_train=y_train,
            x_val=x_val, y_val=y_val,
            out_dir=out_dir,
            epochs=epochs,
            batch_size=batch_size,
        )
        member_infos.append(info)

    # Save ensemble metadata
    ensemble_info = {
        'ensemble_dir': str(out_dir),
        'n_members': members,
        'seeds': seeds,
        'epochs_requested': epochs,
        'look_back': 30,
        'n_features': 8,
        'data_split': {
            'train_size': int(len(x_train)),
            'val_size': int(len(x_val)),
            'test_size': int(len(x_test)),
        },
        'members': member_infos,
        'generated_at': datetime.now().isoformat(),
    }
    info_path = out_dir / 'ensemble_info.json'
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(ensemble_info, f, ensure_ascii=False, indent=2)

    print(f'\n{"="*60}')
    print(f'Training complete. Ensemble saved to: {out_dir}')
    print(f'Member val_losses:')
    for m in member_infos:
        print(f'  member_{m["member_idx"]} (seed={m["seed"]}): val_loss={m["val_loss"]:.6f}  epochs={m["epochs_trained"]}')
    print(f'{"="*60}')
    return str(out_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train GRU Deep Ensemble')
    parser.add_argument('--epochs', type=int, default=120, help='Max epochs per member (early stopping applies)')
    parser.add_argument('--members', type=int, default=5, help='Number of ensemble members (max 5)')
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()
    run(epochs=args.epochs, members=min(args.members, len(ENSEMBLE_SEEDS)), batch_size=args.batch_size)
