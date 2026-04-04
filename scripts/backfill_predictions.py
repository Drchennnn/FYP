"""
backfill_predictions.py
-----------------------
用已训练的 GRU / LSTM 模型对"测试集结束日之后"的历史数据做滚动推断，
将结果追加到各自的 *_test_predictions.csv，从而在 Dashboard 中补全曲线。

使用方法（在项目根目录运行）：
    python scripts/backfill_predictions.py

逻辑：
1. 读取 data/processed/jiuzhaigou_daily_features_2016-01-01_2026-04-02.csv
2. 从 output/runs 找最近一次 gru / lstm / seq2seq run 的模型权重和预测 CSV
3. 对测试集截止日期之后的每一天做滚动单步预测
4. 结果追加到对应 CSV（不覆盖已有行）
"""

import os
import sys
import glob
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_CSV = os.path.join(BASE_DIR, 'data', 'processed',
                              'jiuzhaigou_daily_features_2016-01-01_2026-04-02.csv')
SCALER_PATH   = os.path.join(BASE_DIR, 'models', 'scalers', 'feature_scalers.pkl')
RUNS_DIR      = os.path.join(BASE_DIR, 'output', 'runs')
LOOK_BACK     = 30

FEATURE_COLS = [
    "visitor_count_scaled",
    "month_norm",
    "day_of_week_norm",
    "is_holiday",
    "tourism_num_lag_7_scaled",
    "meteo_precip_sum_scaled",
    "temp_high_scaled",
    "temp_low_scaled",
]


# ── helpers ──────────────────────────────────────────────────────────────────

def find_latest_run(model_key: str):
    """Return (pred_csv_path, model_weights_path) for the most recently modified run."""
    best = None
    best_mtime = -1

    # Search recursively under RUNS_DIR
    csvs_all = glob.glob(os.path.join(RUNS_DIR, '**', f'*{model_key}*_test_predictions.csv'), recursive=True)
    for csv_path in csvs_all:
        mtime = os.path.getmtime(csv_path)
        if mtime <= best_mtime:
            continue

        run_dir = os.path.dirname(csv_path)
        h5s = (glob.glob(os.path.join(run_dir, 'weights', '*.h5')) +
               glob.glob(os.path.join(run_dir, 'weights', '*.keras')))
        if not h5s:
            continue

        best_mtime = mtime
        best = (csv_path, h5s[0])

    return best


def build_feature_df(df_raw: pd.DataFrame, scalers: dict) -> pd.DataFrame:
    """Add visitor_count_scaled, month_norm, day_of_week_norm to df_raw."""
    df = df_raw.copy()
    df['date'] = pd.to_datetime(df['date']).dt.date

    vc_scaler = scalers['visitor_count']
    lag7_scaler = scalers['tourism_num_lag_7']
    prec_scaler = scalers['precipitation']
    th_scaler   = scalers['temp_high']
    tl_scaler   = scalers['temp_low']

    df['visitor_count_scaled'] = vc_scaler.transform(
        df[['tourism_num']].fillna(0).values).flatten()

    df['month_norm']       = df['month'] / 12.0
    df['day_of_week_norm'] = df['day_of_week'] / 7.0

    # recalculate lag_7_scaled with scaler (CSV may already have it, but recalculate to be safe)
    lag7 = df['tourism_num_lag_7'].fillna(0).values.reshape(-1, 1)
    df['tourism_num_lag_7_scaled'] = lag7_scaler.transform(lag7).flatten()

    prec = df['meteo_precip_sum'].fillna(0).values.reshape(-1, 1)
    df['meteo_precip_sum_scaled'] = prec_scaler.transform(prec).flatten()

    th = df['temp_high_c'].fillna(df['temp_high_c'].mean()).values.reshape(-1, 1)
    df['temp_high_scaled'] = th_scaler.transform(th).flatten()

    tl = df['temp_low_c'].fillna(df['temp_low_c'].mean()).values.reshape(-1, 1)
    df['temp_low_scaled'] = tl_scaler.transform(tl).flatten()

    return df


def rolling_predict_gru_lstm(model, df_feat: pd.DataFrame,
                              start_date: date, end_date: date,
                              vc_scaler) -> pd.DataFrame:
    """
    Single-step rolling inference from start_date to end_date (inclusive).
    Returns DataFrame with columns: date, y_true, y_pred
    """
    df_feat = df_feat.sort_values('date').reset_index(drop=True)
    results = []

    cur = start_date
    while cur <= end_date:
        # index of 'cur' in df_feat
        idx_list = df_feat.index[df_feat['date'] == cur].tolist()
        if not idx_list:
            cur += timedelta(days=1)
            continue
        idx = idx_list[0]

        # need LOOK_BACK rows ending at idx-1
        window_end = idx - 1
        window_start = window_end - LOOK_BACK + 1
        if window_start < 0:
            cur += timedelta(days=1)
            continue

        window = df_feat.iloc[window_start:window_end + 1]
        if len(window) < LOOK_BACK:
            cur += timedelta(days=1)
            continue

        X = window[FEATURE_COLS].values.astype(np.float32)
        # 未来日期行 visitor_count_scaled 可能为 NaN，用前向填充
        for col_i in range(X.shape[1]):
            col_vals = X[:, col_i]
            for row_i in range(1, len(col_vals)):
                if np.isnan(col_vals[row_i]):
                    col_vals[row_i] = col_vals[row_i - 1]
            if np.isnan(col_vals[0]):
                col_vals[0] = 0.0
        X = X[np.newaxis, :, :]  # (1, 30, 8)

        y_scaled = model.predict(X, verbose=0)
        y_pred = float(vc_scaler.inverse_transform(y_scaled.reshape(-1, 1))[0, 0])
        y_pred = max(0.0, round(y_pred, 1))

        # y_true from processed data
        y_true_raw = df_feat.loc[idx, 'tourism_num']
        y_true = float(y_true_raw) if not pd.isna(y_true_raw) else np.nan

        results.append({'date': str(cur), 'y_true': y_true, 'y_pred': y_pred})
        y_true_str = f"{y_true:.0f}" if not np.isnan(y_true) else "N/A"
        print(f"  {cur}: y_pred={y_pred:.0f}  y_true={y_true_str}")
        cur += timedelta(days=1)

    return pd.DataFrame(results)


def rolling_predict_seq2seq(model, df_feat: pd.DataFrame,
                             start_date: date, end_date: date,
                             vc_scaler) -> pd.DataFrame:
    """
    Multi-step rolling inference for Seq2Seq+Attention.
    For each target date, build encoder input (30 steps) + decoder input (7 steps of external features).
    Take only the first output step as the prediction for that date.
    Returns DataFrame with columns: date, y_true, y_pred
    """
    df_feat = df_feat.sort_values('date').reset_index(drop=True)
    results = []

    cur = start_date
    while cur <= end_date:
        idx_list = df_feat.index[df_feat['date'] == cur].tolist()
        if not idx_list:
            cur += timedelta(days=1)
            continue
        idx = idx_list[0]

        # encoder: 30 steps ending at idx-1
        window_end = idx - 1
        window_start = window_end - LOOK_BACK + 1
        if window_start < 0:
            cur += timedelta(days=1)
            continue

        window = df_feat.iloc[window_start:window_end + 1]
        if len(window) < LOOK_BACK:
            cur += timedelta(days=1)
            continue

        enc_arr = window[FEATURE_COLS].values.astype(np.float32)
        # 前向填充 NaN（未来日期的 visitor_count_scaled 可能为 NaN）
        for col_i in range(enc_arr.shape[1]):
            col_vals = enc_arr[:, col_i]
            for row_i in range(1, len(col_vals)):
                if np.isnan(col_vals[row_i]):
                    col_vals[row_i] = col_vals[row_i - 1]
            if np.isnan(col_vals[0]):
                col_vals[0] = 0.0
        enc_input = enc_arr[np.newaxis, :, :]  # (1,30,8)

        # decoder: 7 steps of external features (no visitor_count), starting from cur
        dec_steps = 7
        dec_rows = []
        for step in range(dec_steps):
            d = cur + timedelta(days=step)
            d_idx_list = df_feat.index[df_feat['date'] == d].tolist()
            if d_idx_list:
                row = df_feat.iloc[d_idx_list[0]]
                dec_rows.append([
                    row['month_norm'], row['day_of_week_norm'], row['is_holiday'],
                    row['tourism_num_lag_7_scaled'], row['meteo_precip_sum_scaled'],
                    row['temp_high_scaled'], row['temp_low_scaled'],
                ])
            else:
                # fallback: use last known values
                last = df_feat.iloc[window_end]
                m_norm = (d.month - 1) / 11.0
                dow_norm = d.weekday() / 6.0
                dec_rows.append([m_norm, dow_norm, 0.0,
                                  last['tourism_num_lag_7_scaled'],
                                  last['meteo_precip_sum_scaled'],
                                  last['temp_high_scaled'], last['temp_low_scaled']])

        dec_input = np.array(dec_rows, dtype=np.float32)[np.newaxis, :, :]  # (1,7,7)

        y_scaled = model.predict([enc_input, dec_input], verbose=0)
        # take first step output
        y_pred = float(vc_scaler.inverse_transform(y_scaled[0, 0, 0].reshape(1, 1))[0, 0])
        y_pred = max(0.0, round(y_pred, 1))

        y_true_raw = df_feat.loc[idx, 'tourism_num']
        y_true = float(y_true_raw) if not pd.isna(y_true_raw) else np.nan

        results.append({'date': str(cur), 'y_true': y_true, 'y_pred': y_pred})
        y_true_str = f"{y_true:.0f}" if not np.isnan(y_true) else "N/A"
        print(f"  {cur}: y_pred={y_pred:.0f}  y_true={y_true_str}")
        cur += timedelta(days=1)

    return pd.DataFrame(results)


def append_predictions(csv_path: str, new_df: pd.DataFrame):
    """Append new rows to existing CSV, skipping dates already present."""
    existing = pd.read_csv(csv_path)
    existing['date'] = pd.to_datetime(existing['date']).dt.date.astype(str)

    existing_dates = set(existing['date'].tolist())
    new_df['date'] = new_df['date'].astype(str)
    to_add = new_df[~new_df['date'].isin(existing_dates)]

    if to_add.empty:
        print(f"  Nothing to append (all dates already present).")
        return 0

    combined = pd.concat([existing, to_add], ignore_index=True)
    combined.to_csv(csv_path, index=False)
    print(f"  Appended {len(to_add)} rows -> {csv_path}")
    return len(to_add)


def run_backfill_seq2seq():
    print(f"\n{'='*60}")
    print(f"  Backfilling: Seq2Seq+Attention (8 features)")
    print(f"{'='*60}")

    # find seq2seq run
    result = find_latest_run('seq2seq')
    if result is None:
        print("  ERROR: no run found for key='seq2seq'")
        return

    csv_path, weights_path = result
    print(f"  CSV    : {csv_path}")
    print(f"  Weights: {weights_path}")

    existing = pd.read_csv(csv_path)
    existing['date'] = pd.to_datetime(existing['date']).dt.date
    last_test_date = existing['date'].max()
    print(f"  Existing CSV ends: {last_test_date}")

    df_raw = pd.read_csv(PROCESSED_CSV)
    df_raw['date_col'] = pd.to_datetime(df_raw['date']).dt.date
    last_processed = df_raw['date_col'].max()
    print(f"  Processed data ends: {last_processed}")

    start_date = last_test_date + timedelta(days=1)
    end_date = last_processed

    if start_date > end_date:
        print("  Already up-to-date. Nothing to do.")
        return

    print(f"  Gap to fill: {start_date} ~ {end_date} ({(end_date - start_date).days + 1} days)")

    with open(SCALER_PATH, 'rb') as f:
        scalers = pickle.load(f)

    df_feat = build_feature_df(df_raw.drop(columns=['date_col']), scalers)

    print("  Loading Seq2Seq model...")
    import sys as _sys
    _sys.path.insert(0, BASE_DIR)
    import tensorflow as tf
    from models.lstm.train_seq2seq_attention_8features import (
        AttentionLayer, Seq2SeqWithAttention, create_custom_asymmetric_loss
    )
    _loss_fn = create_custom_asymmetric_loss()
    custom_objects = {
        'AttentionLayer': AttentionLayer,
        'Seq2SeqWithAttention': Seq2SeqWithAttention,
        'custom_asymmetric_loss': _loss_fn,
        'custom_loss': _loss_fn,
    }
    model = tf.keras.models.load_model(weights_path, custom_objects=custom_objects, compile=False)

    print("  Running rolling inference...")
    new_df = rolling_predict_seq2seq(model, df_feat, start_date, end_date, scalers['visitor_count'])

    if new_df.empty:
        print("  No predictions generated.")
        return

    n = append_predictions(csv_path, new_df)
    print(f"  Done. Added {n} new rows.")


    """Append new rows to existing CSV, skipping dates already present."""
    existing = pd.read_csv(csv_path)
    existing['date'] = pd.to_datetime(existing['date']).dt.date.astype(str)

    existing_dates = set(existing['date'].tolist())
    new_df['date'] = new_df['date'].astype(str)
    to_add = new_df[~new_df['date'].isin(existing_dates)]

    if to_add.empty:
        print(f"  Nothing to append (all dates already present).")
        return 0

    combined = pd.concat([existing, to_add], ignore_index=True)
    combined.to_csv(csv_path, index=False)
    print(f"  Appended {len(to_add)} rows -> {csv_path}")
    return len(to_add)


# ── main ─────────────────────────────────────────────────────────────────────

def run_backfill(model_key: str, model_label: str):
    print(f"\n{'='*60}")
    print(f"  Backfilling: {model_label}")
    print(f"{'='*60}")

    result = find_latest_run(model_key)
    if result is None:
        print(f"  ERROR: no run found for key='{model_key}'")
        return

    csv_path, weights_path = result
    print(f"  CSV    : {csv_path}")
    print(f"  Weights: {weights_path}")

    # Determine gap: last date in existing CSV + 1 day  ->  last date in processed CSV
    existing = pd.read_csv(csv_path)
    existing['date'] = pd.to_datetime(existing['date']).dt.date
    last_test_date = existing['date'].max()
    print(f"  Existing CSV ends: {last_test_date}")

    df_raw = pd.read_csv(PROCESSED_CSV)
    df_raw['date_col'] = pd.to_datetime(df_raw['date']).dt.date
    last_processed = df_raw['date_col'].max()
    print(f"  Processed data ends: {last_processed}")

    start_date = last_test_date + timedelta(days=1)
    end_date   = last_processed

    if start_date > end_date:
        print(f"  Already up-to-date. Nothing to do.")
        return

    print(f"  Gap to fill: {start_date} ~ {end_date} ({(end_date - start_date).days + 1} days)")

    # Load scaler
    with open(SCALER_PATH, 'rb') as f:
        scalers = pickle.load(f)

    # Build feature df
    df_feat = build_feature_df(df_raw.drop(columns=['date_col']), scalers)

    # Load model
    print("  Loading model...")
    from tensorflow import keras
    model = keras.models.load_model(weights_path)

    # Run rolling inference
    print("  Running rolling inference...")
    new_df = rolling_predict_gru_lstm(
        model, df_feat, start_date, end_date, scalers['visitor_count'])

    if new_df.empty:
        print("  No predictions generated.")
        return

    # Append to CSV
    n = append_predictions(csv_path, new_df)
    print(f"  Done. Added {n} new rows.")


def main():
    if not os.path.exists(PROCESSED_CSV):
        print(f"ERROR: processed CSV not found: {PROCESSED_CSV}")
        sys.exit(1)
    if not os.path.exists(SCALER_PATH):
        print(f"ERROR: scaler not found: {SCALER_PATH}")
        sys.exit(1)

    # GRU and LSTM (single-step rolling)
    run_backfill('gru', 'GRU (8 features)')
    run_backfill('lstm', 'LSTM (8 features)')
    # Seq2Seq (multi-step, take first output step)
    run_backfill_seq2seq()

    print("\nBackfill complete. Restart the Flask server to reload predictions.")


if __name__ == '__main__':
    main()
