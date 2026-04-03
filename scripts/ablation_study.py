#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ablation Study — 逐特征消融实验

目的：量化各特征组合对预警 F1 和回归误差的贡献，
     为论文提供"哪些特征最重要"的实验证据。

消融方案（共6组）：
  baseline_4feat  : 4特征基线（visitor, month, dow, holiday）
  no_weather      : 8特征去掉天气（去掉 precip, temp_high, temp_low）
  no_lag7         : 8特征去掉 lag_7
  no_holiday      : 8特征去掉 is_holiday
  no_weather_lag7 : 8特征去掉天气+lag_7（仅时间特征）
  full_8feat      : 完整8特征（对照组）

每组方案在 GRU 上训练（速度最快），用固定测试集评估，
输出对比表格和柱状图。

用法：
  python scripts/ablation_study.py
  python scripts/ablation_study.py --epochs 60 --output output/ablation
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

from models.common.core_evaluation import (
    DEFAULT_PEAK_THRESHOLD,
    compute_core_metrics,
    compute_dynamic_peak_threshold,
    compute_weather_thresholds_quantile,
)

# ─────────────────────────────────────────────
# 消融方案定义
# ─────────────────────────────────────────────

# 完整8特征列表（与训练脚本一致）
ALL_8_FEATURES = [
    'visitor_count_scaled',
    'month_norm',
    'day_of_week_norm',
    'is_holiday',
    'tourism_num_lag_7_scaled',
    'meteo_precip_sum_scaled',
    'temp_high_scaled',
    'temp_low_scaled',
]

ABLATION_CONFIGS: Dict[str, List[str]] = {
    'baseline_4feat': [
        'visitor_count_scaled', 'month_norm', 'day_of_week_norm', 'is_holiday',
    ],
    'no_weather': [
        'visitor_count_scaled', 'month_norm', 'day_of_week_norm', 'is_holiday',
        'tourism_num_lag_7_scaled',
    ],
    'no_lag7': [
        'visitor_count_scaled', 'month_norm', 'day_of_week_norm', 'is_holiday',
        'meteo_precip_sum_scaled', 'temp_high_scaled', 'temp_low_scaled',
    ],
    'no_holiday': [
        'visitor_count_scaled', 'month_norm', 'day_of_week_norm',
        'tourism_num_lag_7_scaled', 'meteo_precip_sum_scaled',
        'temp_high_scaled', 'temp_low_scaled',
    ],
    'no_weather_lag7': [
        'visitor_count_scaled', 'month_norm', 'day_of_week_norm', 'is_holiday',
    ],
    'full_8feat': ALL_8_FEATURES,
}

# 中文标签（用于图表）
LABELS_ZH = {
    'baseline_4feat':   '4特征基线',
    'no_weather':       '去掉天气特征',
    'no_lag7':          '去掉Lag-7',
    'no_holiday':       '去掉节假日',
    'no_weather_lag7':  '去掉天气+Lag-7',
    'full_8feat':       '完整8特征',
}


# ─────────────────────────────────────────────
# 数据加载
# ─────────────────────────────────────────────

def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').drop_duplicates(subset=['date']).reset_index(drop=True)
    if 'tourism_num' in df.columns:
        df['visitor_count'] = pd.to_numeric(df['tourism_num'], errors='coerce')
    df = df.dropna(subset=['visitor_count']).reset_index(drop=True)
    return df


def build_sequences(df: pd.DataFrame, feature_cols: List[str], look_back: int = 30):
    """构建 (X, y, dates) 序列。"""
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        # 缺失特征用0填充（消融时会出现）
        for c in missing:
            df[c] = 0.0

    values = df[feature_cols].values.astype(np.float32)
    target = df['visitor_count_scaled'].values.astype(np.float32)
    dates = df['date'].values

    X, y, d = [], [], []
    for i in range(look_back, len(df)):
        X.append(values[i - look_back: i])
        y.append(target[i])
        d.append(dates[i])
    return np.array(X), np.array(y), np.array(d)


# ─────────────────────────────────────────────
# 模型构建（GRU，输入维度可变）
# ─────────────────────────────────────────────

def build_gru(look_back: int, n_features: int) -> tf.keras.Model:
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(look_back, n_features)),
        tf.keras.layers.GRU(128, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.GRU(64),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=tf.keras.losses.Huber())
    return model


# ─────────────────────────────────────────────
# 单方案训练+评估
# ─────────────────────────────────────────────

def run_ablation_config(
    df: pd.DataFrame,
    config_name: str,
    feature_cols: List[str],
    look_back: int,
    epochs: int,
    batch_size: int,
    test_ratio: float,
    val_ratio: float,
) -> Dict:
    print(f"\n  [{config_name}] 特征数={len(feature_cols)}: {feature_cols}")

    # 用训练集拟合 scaler
    n = len(df)
    test_size = int(n * test_ratio)
    train_val_size = n - test_size
    val_size = int(train_val_size * val_ratio)
    train_size = train_val_size - val_size

    df_work = df.copy()
    scaler = MinMaxScaler()
    train_counts = df_work['visitor_count'].values[:train_size].reshape(-1, 1)
    scaler.fit(train_counts)
    df_work['visitor_count_scaled'] = scaler.transform(
        df_work['visitor_count'].values.reshape(-1, 1)
    ).flatten()

    X, y, d = build_sequences(df_work, feature_cols, look_back)

    # 重新切分（序列构建后 n 变小）
    n_seq = len(X)
    test_size_seq = int(n_seq * test_ratio)
    trainval_size_seq = n_seq - test_size_seq
    val_size_seq = int(trainval_size_seq * val_ratio)
    train_size_seq = trainval_size_seq - val_size_seq

    X_train = X[:train_size_seq]
    y_train = y[:train_size_seq]
    X_val = X[train_size_seq:trainval_size_seq]
    y_val = y[train_size_seq:trainval_size_seq]
    X_test = X[trainval_size_seq:]
    y_test = y[trainval_size_seq:]
    d_test = d[trainval_size_seq:]

    if len(X_train) < 50 or len(X_test) < 10:
        print(f"    数据不足，跳过")
        return {}

    # 训练
    tf.random.set_seed(42)
    model = build_gru(look_back, len(feature_cols))
    cb = [tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=15, restore_best_weights=True, verbose=0
    )]
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=epochs, batch_size=batch_size, callbacks=cb, verbose=0)

    # 预测+反归一化
    y_pred_s = model.predict(X_test, verbose=0).flatten()
    y_pred = scaler.inverse_transform(y_pred_s.reshape(-1, 1)).flatten()
    y_true = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    # 动态峰值阈值
    train_counts_inv = scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()
    peak_thr = compute_dynamic_peak_threshold(train_counts_inv)

    # 天气阈值（仅训练集）
    weather_thresholds = None
    weather_precip = np.full(len(y_true), np.nan)
    weather_temp_high = np.full(len(y_true), np.nan)
    weather_temp_low = np.full(len(y_true), np.nan)

    if all(c in df_work.columns for c in ['meteo_precip_sum', 'meteo_temp_max', 'meteo_temp_min']):
        weather_thresholds = compute_weather_thresholds_quantile(
            train_precip=df_work['meteo_precip_sum'].values[:train_size],
            train_temp_high=df_work['meteo_temp_max'].values[:train_size],
            train_temp_low=df_work['meteo_temp_min'].values[:train_size],
        )
        test_weather = df_work.iloc[trainval_size_seq + look_back:].reset_index(drop=True)
        n_w = min(len(test_weather), len(y_true))
        weather_precip[:n_w] = test_weather['meteo_precip_sum'].values[:n_w]
        weather_temp_high[:n_w] = test_weather['meteo_temp_max'].values[:n_w]
        weather_temp_low[:n_w] = test_weather['meteo_temp_min'].values[:n_w]

    metrics = compute_core_metrics(
        y_true=y_true, y_pred=y_pred, dates=d_test,
        peak_threshold=peak_thr,
        weather_precip=weather_precip,
        weather_temp_high=weather_temp_high,
        weather_temp_low=weather_temp_low,
        weather_thresholds=weather_thresholds,
    )

    result = {
        'config': config_name,
        'label': LABELS_ZH.get(config_name, config_name),
        'n_features': len(feature_cols),
        'features': feature_cols,
        'peak_threshold': peak_thr,
        'mae': metrics['regression']['mae'],
        'rmse': metrics['regression']['rmse'],
        'smape': metrics['regression']['smape'],
        'crowd_f1': metrics['crowd_alert']['f1'],
        'crowd_recall': metrics['crowd_alert']['recall'],
        'suitability_f1': metrics['suitability_warning']['f1'],
        'suitability_recall': metrics['suitability_warning']['recall'],
        'brier': metrics['suitability_warning']['brier'],
        'ece': metrics['suitability_warning']['ece'],
    }

    print(f"    MAE={result['mae']:.0f}  SMAPE={result['smape']:.1f}%  "
          f"Suit-F1={result['suitability_f1']:.3f}  Suit-Recall={result['suitability_recall']:.3f}")
    return result


# ─────────────────────────────────────────────
# 可视化
# ─────────────────────────────────────────────

def plot_ablation_results(df_res: pd.DataFrame, out_dir: Path):
    configs = df_res['config'].tolist()
    labels = df_res['label'].tolist()
    x = np.arange(len(configs))
    width = 0.35

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Ablation Study — GRU Model (Feature Contribution)', fontsize=13)

    # MAE
    ax = axes[0]
    bars = ax.bar(x, df_res['mae'], color='steelblue', alpha=0.8, edgecolor='white')
    # 高亮 full_8feat
    for i, c in enumerate(configs):
        if c == 'full_8feat':
            bars[i].set_color('tomato')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha='right', fontsize=8)
    ax.set_ylabel('MAE (visitors)')
    ax.set_title('Regression Error (MAE)')

    # Suitability F1
    ax = axes[1]
    bars = ax.bar(x, df_res['suitability_f1'], color='steelblue', alpha=0.8, edgecolor='white')
    for i, c in enumerate(configs):
        if c == 'full_8feat':
            bars[i].set_color('tomato')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha='right', fontsize=8)
    ax.set_ylabel('F1 Score')
    ax.set_title('Suitability Warning F1')
    ax.set_ylim(0, 1.05)

    # Suitability Recall
    ax = axes[2]
    bars = ax.bar(x, df_res['suitability_recall'], color='steelblue', alpha=0.8, edgecolor='white')
    for i, c in enumerate(configs):
        if c == 'full_8feat':
            bars[i].set_color('tomato')
    ax.axhline(0.8, color='red', linestyle='--', linewidth=1.2, label='Min Recall=0.80')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha='right', fontsize=8)
    ax.set_ylabel('Recall')
    ax.set_title('Suitability Warning Recall')
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)

    plt.tight_layout()
    save_path = out_dir / 'ablation_study.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  图表已保存: {save_path}")


# ─────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Ablation Study')
    parser.add_argument('--input-csv',
                        default='data/processed/jiuzhaigou_daily_features_2016-01-01_2026-04-02.csv')
    parser.add_argument('--epochs', type=int, default=80, help='每方案最大训练轮次（默认80）')
    parser.add_argument('--look-back', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--test-ratio', type=float, default=0.1)
    parser.add_argument('--val-ratio', type=float, default=0.125)
    parser.add_argument('--output', default='output/ablation',
                        help='输出目录（默认 output/ablation）')
    args = parser.parse_args()

    np.random.seed(42)
    tf.random.set_seed(42)

    csv_path = PROJECT_ROOT / args.input_csv
    if not csv_path.exists():
        print(f"错误：找不到数据文件 {csv_path}")
        sys.exit(1)

    print(f"加载数据: {csv_path}")
    df = load_data(csv_path)
    print(f"数据范围: {df['date'].min().date()} ~ {df['date'].max().date()}，共 {len(df)} 行")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = PROJECT_ROOT / args.output / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Ablation Study — {len(ABLATION_CONFIGS)} 个方案")
    print(f"{'='*60}")

    results = []
    for config_name, feature_cols in ABLATION_CONFIGS.items():
        r = run_ablation_config(
            df=df.copy(),
            config_name=config_name,
            feature_cols=feature_cols,
            look_back=args.look_back,
            epochs=args.epochs,
            batch_size=args.batch_size,
            test_ratio=args.test_ratio,
            val_ratio=args.val_ratio,
        )
        if r:
            results.append(r)

    if not results:
        print("没有有效结果")
        return

    df_res = pd.DataFrame(results)

    # 保存结果
    df_res.to_csv(out_dir / 'ablation_results.csv', index=False)
    with open(out_dir / 'ablation_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 打印汇总表
    print(f"\n{'='*60}")
    print("Ablation Study 汇总")
    print(f"{'='*60}")
    cols = ['label', 'n_features', 'mae', 'smape', 'suitability_f1', 'suitability_recall', 'brier']
    print(df_res[cols].to_string(index=False, float_format='{:.3f}'.format))

    # 可视化
    plot_ablation_results(df_res, out_dir)

    print(f"\n结果已保存至: {out_dir}")


if __name__ == '__main__':
    main()
