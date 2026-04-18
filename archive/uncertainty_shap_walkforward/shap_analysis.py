#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SHAP 特征重要性分析（GRU / LSTM）

方法：特征消融 SHAP（Feature Ablation，model-agnostic）
     对每个特征，将测试集输入中该特征的所有时间步替换为训练集均值，
     计算模型输出变化量作为该特征的贡献值。
     保留 GRU 的完整时序结构，避免 tile/flatten 技巧破坏时序动态。
     计算复杂度 O(n_explain × n_features)，约 1-3 分钟。

输出：
  - shap_summary_bar_<model>.png  : 全局特征重要性柱状图（均值绝对 SHAP 值）
  - shap_importance_<model>.csv   : 特征重要性数值
  - shap_values_<model>.npy       : 原始 SHAP 值矩阵（供后续分析）

用法：
  python scripts/shap_analysis.py --model gru
  python scripts/shap_analysis.py --model lstm
  python scripts/shap_analysis.py --model all

依赖：
  pip install shap
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

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

FEATURE_NAMES = [
    'visitor_count_scaled',
    'month_norm',
    'day_of_week_norm',
    'is_holiday',
    'tourism_num_lag_7_scaled',
    'meteo_precip_sum_scaled',
    'temp_high_scaled',
    'temp_low_scaled',
]

# 更易读的特征标签
FEATURE_LABELS = [
    'Visitor Count (t-1~t-30)',
    'Month',
    'Day of Week',
    'Is Holiday',
    'Lag-7 Visitor Count',
    'Precipitation',
    'Temp High',
    'Temp Low',
]


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').drop_duplicates(subset=['date']).reset_index(drop=True)
    if 'tourism_num' in df.columns:
        df['visitor_count'] = pd.to_numeric(df['tourism_num'], errors='coerce')
    df = df.dropna(subset=['visitor_count']).reset_index(drop=True)
    # Recompute month_norm and day_of_week_norm from raw columns (same as training script).
    # The CSV stores these as NaN for most rows; the raw integer columns are always valid.
    if 'month' in df.columns:
        df['month_norm'] = (df['month'] - 1) / 11.0
    if 'day_of_week' in df.columns:
        df['day_of_week_norm'] = df['day_of_week'] / 6.0
    return df


def build_sequences(df: pd.DataFrame, look_back: int = 30):
    missing = [c for c in FEATURE_NAMES if c not in df.columns]
    if missing:
        for c in missing:
            df[c] = 0.0
    values = df[FEATURE_NAMES].values.astype(np.float32)
    target = df['visitor_count_scaled'].values.astype(np.float32)
    X, y = [], []
    for i in range(look_back, len(df)):
        X.append(values[i - look_back: i])
        y.append(target[i])
    return np.array(X), np.array(y)


def find_latest_model(model_type: str) -> Optional[Path]:
    """在 output/runs/ 中找最新的对应模型权重文件。"""
    runs_dir = PROJECT_ROOT / 'output' / 'runs'
    pattern = f'{model_type}_8features_*'
    candidates = sorted(runs_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    for run_dir in candidates:
        for sub in run_dir.rglob('weights/*.h5'):
            return sub
        for sub in run_dir.rglob('weights/*.keras'):
            return sub
    return None


def run_shap_analysis(
    model_type: str,
    csv_path: Path,
    out_dir: Path,
    look_back: int = 30,
    test_ratio: float = 0.1,
    n_background: int = 100,
    n_explain: int = 200,
):
    """对指定模型运行 SHAP 分析。"""
    try:
        import shap
    except ImportError:
        print("错误：请先安装 shap：pip install shap")
        return

    print(f"\n{'='*60}")
    print(f"SHAP 分析: {model_type.upper()}")
    print(f"{'='*60}")

    # 1. 加载数据
    df = load_data(csv_path)
    scaler = MinMaxScaler()
    n = len(df)
    test_size = int(n * test_ratio)
    train_size = n - test_size
    scaler.fit(df['visitor_count'].values[:train_size].reshape(-1, 1))
    df['visitor_count_scaled'] = scaler.transform(
        df['visitor_count'].values.reshape(-1, 1)
    ).flatten()

    X, y = build_sequences(df, look_back)
    n_seq = len(X)
    test_size_seq = int(n_seq * test_ratio)
    X_train = X[:n_seq - test_size_seq]
    X_test = X[n_seq - test_size_seq:]

    print(f"  训练集: {len(X_train)} 样本，测试集: {len(X_test)} 样本")

    # 2. 加载模型
    model_path = find_latest_model(model_type)
    if model_path is None:
        print(f"  错误：找不到 {model_type} 模型权重，请先运行训练")
        return

    print(f"  加载模型: {model_path}")
    try:
        model = tf.keras.models.load_model(str(model_path), compile=False)
    except Exception as e:
        print(f"  模型加载失败: {e}")
        return

    # 3. 计算 SHAP 值
    # 方法：特征消融 SHAP（Feature Ablation）
    #
    # 对每个特征 f，将测试集中该特征的所有时间步替换为训练集均值，
    # 计算模型输出变化量作为该特征的 SHAP 贡献。
    # 这与 KernelExplainer 的 mean-baseline 等价，但直接在 3D 序列上操作，
    # 保留了 GRU 的时序动态，避免了 tile 技巧破坏时序结构的问题。
    # 计算复杂度：O(n_explain × n_features)，约 1-3 分钟。
    np.random.seed(42)
    exp_idx = np.random.choice(len(X_test), min(n_explain, len(X_test)), replace=False)
    X_explain = X_test[exp_idx]  # (n_explain, look_back, n_features)

    n_features = len(FEATURE_NAMES)

    # 训练集各特征的时间步均值 → (n_features,)，用作遮蔽基准
    train_feature_means = X_train.mean(axis=(0, 1))  # (n_features,)

    print(f"  计算 SHAP 值（特征消融法, explain={len(X_explain)}, features={n_features}）...")
    print("  提示：逐特征消融，预计 1-3 分钟...")

    # 基准预测（完整特征）
    baseline_preds = model.predict(X_explain, verbose=0).flatten()  # (n_explain,)

    shap_values = np.zeros((len(X_explain), n_features), dtype=np.float32)

    for feat_idx in range(n_features):
        X_masked = X_explain.copy()
        X_masked[:, :, feat_idx] = train_feature_means[feat_idx]
        masked_preds = model.predict(X_masked, verbose=0).flatten()
        # SHAP contribution = baseline - masked (positive = feature increases prediction)
        shap_values[:, feat_idx] = baseline_preds - masked_preds
        print(f"    特征 {feat_idx+1}/{n_features} ({FEATURE_NAMES[feat_idx]}): "
              f"mean|SHAP|={np.abs(shap_values[:, feat_idx]).mean():.5f}")

    print(f"  SHAP 值形状: {shap_values.shape}")  # (n_explain, n_features)

    # 4. 聚合：已是 (n_explain, n_features)，直接取绝对值
    shap_agg = np.abs(shap_values)  # (n_explain, n_features)

    # 5. 保存原始值
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(str(out_dir / f'shap_values_{model_type}.npy'), shap_values)

    # 6. 全局特征重要性柱状图
    mean_abs_shap = np.mean(shap_agg, axis=0)  # (8,)
    sorted_idx = np.argsort(mean_abs_shap)[::-1]

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ['tomato' if i == sorted_idx[0] else 'steelblue' for i in range(len(FEATURE_LABELS))]
    bars = ax.barh(
        [FEATURE_LABELS[i] for i in sorted_idx],
        mean_abs_shap[sorted_idx],
        color=[colors[i] for i in sorted_idx],
        alpha=0.85, edgecolor='white'
    )
    ax.set_xlabel('Mean |SHAP Value|', fontsize=10)
    ax.set_title(f'SHAP Feature Importance — {model_type.upper()} (8 features)', fontsize=11)
    ax.invert_yaxis()
    plt.tight_layout()
    save_path = out_dir / f'shap_summary_bar_{model_type}.png'
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  特征重要性图已保存: {save_path}")

    # 7. 打印排名
    print(f"\n  特征重要性排名 ({model_type.upper()}):")
    for rank, i in enumerate(sorted_idx):
        print(f"    {rank+1}. {FEATURE_LABELS[i]:35s}  {mean_abs_shap[i]:.5f}")

    # 8. 保存 CSV
    df_imp = pd.DataFrame({
        'feature': [FEATURE_LABELS[i] for i in sorted_idx],
        'feature_key': [FEATURE_NAMES[i] for i in sorted_idx],
        'mean_abs_shap': mean_abs_shap[sorted_idx],
    })
    df_imp.to_csv(out_dir / f'shap_importance_{model_type}.csv', index=False)


def main():
    parser = argparse.ArgumentParser(description='SHAP 特征重要性分析')
    parser.add_argument('--model', default='gru', choices=['gru', 'lstm', 'all'])
    parser.add_argument('--input-csv',
                        default='data/processed/jiuzhaigou_daily_features_2016-01-01_2026-04-02.csv')
    parser.add_argument('--look-back', type=int, default=30)
    parser.add_argument('--n-background', type=int, default=100,
                        help='SHAP 背景样本数（默认100）')
    parser.add_argument('--n-explain', type=int, default=200,
                        help='解释样本数（默认200）')
    parser.add_argument('--output', default='output/shap')
    args = parser.parse_args()

    csv_path = PROJECT_ROOT / args.input_csv
    if not csv_path.exists():
        print(f"错误：找不到数据文件 {csv_path}")
        sys.exit(1)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = PROJECT_ROOT / args.output / timestamp

    models = ['gru', 'lstm'] if args.model == 'all' else [args.model]
    for m in models:
        run_shap_analysis(
            model_type=m,
            csv_path=csv_path,
            out_dir=out_dir,
            look_back=args.look_back,
            n_background=args.n_background,
            n_explain=args.n_explain,
        )

    print(f"\n结果已保存至: {out_dir}")


if __name__ == '__main__':
    main()
