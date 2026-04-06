#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ablation Study — 特征遮蔽消融实验（Feature Masking）

设计原则：
  加载已训练好的 GRU 模型权重，在固定测试集上对每个消融配置
  将被移除的特征替换为该特征在训练集上的均值（mean-fill masking）。
  所有配置使用同一套模型权重，差异完全来自特征，而非训练随机性。

  这是论文标准的消融实验做法：
    - 控制变量：模型权重固定，只改变输入特征
    - 可解释性：MAE/F1 的差异直接反映特征贡献
    - 可重复性：无训练随机性，结果确定

消融方案（共6组）：
  baseline_4feat  : 仅保留 4 个基础特征（visitor, month, dow, holiday）
  no_weather      : 去掉 3 个天气特征（precip, temp_high, temp_low）
  no_lag7         : 去掉 lag_7 特征
  no_holiday      : 去掉 is_holiday 特征
  no_weather_lag7 : 去掉天气 + lag_7（仅时间特征）
  full_8feat      : 完整 8 特征（对照组，无遮蔽）

用法：
  python scripts/ablation_study.py
  python scripts/ablation_study.py --output output/ablation
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

from models.common.core_evaluation import (
    compute_core_metrics,
    compute_dynamic_peak_threshold,
    compute_weather_thresholds_quantile,
)

# ─────────────────────────────────────────────
# 字体配置（支持中文）
# ─────────────────────────────────────────────

def _setup_cjk_font():
    """尝试配置支持中文的字体，失败则回退到英文标签。"""
    candidates = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS',
                  'WenQuanYi Micro Hei', 'Noto Sans CJK SC']
    for name in candidates:
        if any(name.lower() in f.name.lower() for f in fm.fontManager.ttflist):
            plt.rcParams['font.family'] = name
            plt.rcParams['axes.unicode_minus'] = False
            return True
    # 找不到 CJK 字体，使用英文标签
    plt.rcParams['axes.unicode_minus'] = False
    return False

HAS_CJK = _setup_cjk_font()

# ─────────────────────────────────────────────
# 消融方案定义
# ─────────────────────────────────────────────

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

# 每个配置：保留的特征列表（其余特征用训练集均值填充）
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

LABELS_ZH = {
    'baseline_4feat':   '4-Feat Baseline',
    'no_weather':       'No Weather',
    'no_lag7':          'No Lag-7',
    'no_holiday':       'No Holiday',
    'no_weather_lag7':  'No Weather+Lag7',
    'full_8feat':       'Full 8 Features',
}

LABELS_ZH_FULL = {
    'baseline_4feat':   '4特征基线\n(visitor/month/dow/holiday)',
    'no_weather':       '去掉天气特征\n(precip/temp_h/temp_l)',
    'no_lag7':          '去掉Lag-7',
    'no_holiday':       '去掉节假日',
    'no_weather_lag7':  '去掉天气+Lag-7',
    'full_8feat':       '完整8特征\n(对照组)',
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
    # Recompute month_norm and day_of_week_norm from raw columns (same as training script).
    # The CSV stores these as NaN for most rows; the raw integer columns are always valid.
    if 'month' in df.columns:
        df['month_norm'] = (df['month'] - 1) / 11.0
    if 'day_of_week' in df.columns:
        df['day_of_week_norm'] = df['day_of_week'] / 6.0
    return df


def build_sequences(values: np.ndarray, target: np.ndarray, dates: np.ndarray,
                    look_back: int = 30):
    """构建 (X, y, d) 序列，X shape = (N, look_back, n_features)"""
    X, y, d = [], [], []
    for i in range(look_back, len(values)):
        X.append(values[i - look_back: i])
        y.append(target[i])
        d.append(dates[i])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), np.array(d)


# ─────────────────────────────────────────────
# 模型发现
# ─────────────────────────────────────────────

def find_latest_gru_weights() -> Optional[Path]:
    """在 output/runs/ 中找最新的 gru_8features 权重文件。"""
    runs_dir = PROJECT_ROOT / 'output' / 'runs'
    candidates = sorted(
        runs_dir.glob('gru_8features_*'),
        key=lambda p: p.stat().st_mtime, reverse=True
    )
    for top_dir in candidates:
        for h5 in top_dir.rglob('weights/*.h5'):
            return h5
        for keras in top_dir.rglob('weights/*.keras'):
            return keras
    return None


def load_gru_model(weights_path: Path) -> tf.keras.Model:
    """加载已训练的 GRU 模型（固定 8 特征输入）。"""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(30, 8)),
        tf.keras.layers.GRU(128, return_sequences=True, implementation=1),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.GRU(64, implementation=1),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1),
    ])
    model.load_weights(str(weights_path))
    return model


# ─────────────────────────────────────────────
# 特征遮蔽消融
# ─────────────────────────────────────────────

def run_ablation_config(
    model: tf.keras.Model,
    X_test_full: np.ndarray,       # (N, 30, 8) — 完整8特征测试集
    y_test: np.ndarray,            # (N,) — scaled targets
    train_feature_means: np.ndarray,  # (8,) — 训练集各特征均值（用于遮蔽）
    scaler: MinMaxScaler,
    config_name: str,
    keep_features: List[str],
    peak_thr: float,
    weather_precip: np.ndarray,
    weather_temp_high: np.ndarray,
    weather_temp_low: np.ndarray,
    weather_thresholds,
) -> Dict:
    """对单个消融配置做特征遮蔽推理。"""
    print(f"\n  [{config_name}] 保留特征: {keep_features}")

    # 构建遮蔽后的输入：被移除的特征替换为训练集均值
    X_masked = X_test_full.copy()
    for feat_idx, feat_name in enumerate(ALL_8_FEATURES):
        if feat_name not in keep_features:
            X_masked[:, :, feat_idx] = train_feature_means[feat_idx]

    # 推理
    y_pred_s = model.predict(X_masked, verbose=0).flatten()
    y_pred = scaler.inverse_transform(y_pred_s.reshape(-1, 1)).flatten()
    y_true = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    metrics, _ = compute_core_metrics(
        y_true=y_true, y_pred=y_pred,
        peak_threshold=peak_thr,
        weather_precip=weather_precip,
        weather_temp_high=weather_temp_high,
        weather_temp_low=weather_temp_low,
        weather_thresholds=weather_thresholds,
    )

    masked_features = [f for f in ALL_8_FEATURES if f not in keep_features]
    result = {
        'config': config_name,
        'label': LABELS_ZH.get(config_name, config_name),
        'label_full': LABELS_ZH_FULL.get(config_name, config_name),
        'n_features_kept': len(keep_features),
        'n_features_masked': len(masked_features),
        'masked_features': masked_features,
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

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Ablation Study — GRU Model (Feature Masking)', fontsize=13)

    full_idx = configs.index('full_8feat') if 'full_8feat' in configs else -1

    def _bar(ax, values, ylabel, title, ylim=None):
        colors = ['tomato' if i == full_idx else 'steelblue' for i in range(len(configs))]
        bars = ax.bar(x, values, color=colors, alpha=0.85, edgecolor='white')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha='right', fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if ylim:
            ax.set_ylim(*ylim)
        # 数值标注
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01 * (ylim[1] if ylim else max(values)),
                    f'{v:.0f}' if v > 10 else f'{v:.3f}',
                    ha='center', va='bottom', fontsize=7)

    _bar(axes[0], df_res['mae'].tolist(), 'MAE (visitors)', 'Regression Error (MAE)')
    _bar(axes[1], df_res['suitability_f1'].tolist(), 'F1 Score', 'Suitability Warning F1', ylim=(0, 1.1))

    # Recall with threshold line
    colors = ['tomato' if i == full_idx else 'steelblue' for i in range(len(configs))]
    bars = axes[2].bar(x, df_res['suitability_recall'].tolist(), color=colors, alpha=0.85, edgecolor='white')
    axes[2].axhline(0.8, color='red', linestyle='--', linewidth=1.2, label='Min Recall=0.80')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=25, ha='right', fontsize=8)
    axes[2].set_ylabel('Recall')
    axes[2].set_title('Suitability Warning Recall')
    axes[2].set_ylim(0, 1.1)
    axes[2].legend(fontsize=8)
    for bar, v in zip(bars, df_res['suitability_recall'].tolist()):
        axes[2].text(bar.get_x() + bar.get_width() / 2, v + 0.01,
                     f'{v:.3f}', ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    save_path = out_dir / 'ablation_study.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  图表已保存: {save_path}")


# ─────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Ablation Study (Feature Masking)')
    parser.add_argument('--input-csv',
                        default='data/processed/jiuzhaigou_daily_features_2016-01-01_2026-04-02.csv')
    parser.add_argument('--look-back', type=int, default=30)
    parser.add_argument('--test-ratio', type=float, default=0.1)
    parser.add_argument('--val-ratio', type=float, default=0.125)
    parser.add_argument('--output', default='output/ablation')
    parser.add_argument('--weights', default=None,
                        help='GRU 权重路径（默认自动发现最新 gru_8features 运行）')
    args = parser.parse_args()

    np.random.seed(42)

    csv_path = PROJECT_ROOT / args.input_csv
    if not csv_path.exists():
        print(f"错误：找不到数据文件 {csv_path}")
        sys.exit(1)

    # ── 1. 加载模型权重 ──
    weights_path = Path(args.weights) if args.weights else find_latest_gru_weights()
    if weights_path is None or not weights_path.exists():
        print("错误：找不到 GRU 模型权重，请先运行 python run_pipeline.py --model gru --features 8 --epochs 120")
        sys.exit(1)
    print(f"加载模型权重: {weights_path}")
    model = load_gru_model(weights_path)

    # ── 2. 加载数据 ──
    print(f"加载数据: {csv_path}")
    df = load_data(csv_path)
    print(f"数据范围: {df['date'].min().date()} ~ {df['date'].max().date()}，共 {len(df)} 行")

    # ── 3. 数据切分（与训练脚本一致：train 80%, val 10%, test 10%）──
    n = len(df)
    test_size = int(n * args.test_ratio)
    train_val_size = n - test_size
    val_size = int(train_val_size * args.val_ratio)
    train_size = train_val_size - val_size

    # ── 4. 拟合 scaler（仅训练集）──
    scaler = MinMaxScaler()
    scaler.fit(df['visitor_count'].values[:train_size].reshape(-1, 1))
    df['visitor_count_scaled'] = scaler.transform(
        df['visitor_count'].values.reshape(-1, 1)
    ).flatten()

    # ── 5. 构建完整 8 特征序列 ──
    missing = [c for c in ALL_8_FEATURES if c not in df.columns]
    if missing:
        print(f"警告：CSV 中缺少以下特征列，将用 0 填充: {missing}")
        for c in missing:
            df[c] = 0.0

    feature_matrix = df[ALL_8_FEATURES].values.astype(np.float32)
    target = df['visitor_count_scaled'].values.astype(np.float32)
    dates = df['date'].values

    X_all, y_all, d_all = build_sequences(feature_matrix, target, dates, args.look_back)

    # 序列切分
    n_seq = len(X_all)
    test_size_seq = int(n_seq * args.test_ratio)
    trainval_size_seq = n_seq - test_size_seq
    val_size_seq = int(trainval_size_seq * args.val_ratio)
    train_size_seq = trainval_size_seq - val_size_seq

    X_train_full = X_all[:train_size_seq]          # (N_train, 30, 8)
    X_test_full  = X_all[trainval_size_seq:]        # (N_test, 30, 8)
    y_test       = y_all[trainval_size_seq:]        # (N_test,)
    d_test       = d_all[trainval_size_seq:]

    print(f"训练集: {train_size_seq} 序列，测试集: {len(X_test_full)} 序列")

    # ── 6. 计算训练集各特征均值（用于遮蔽）──
    # X_train_full shape: (N_train, 30, 8)
    # 对 sample 和 time 两个维度取均值 → (8,)
    train_feature_means = X_train_full.mean(axis=(0, 1))  # (8,)
    print(f"训练集特征均值（遮蔽基准）: {dict(zip(ALL_8_FEATURES, train_feature_means.round(4)))}")

    # ── 7. 峰值阈值（训练集动态计算）──
    train_counts_inv = scaler.inverse_transform(
        y_all[:train_size_seq].reshape(-1, 1)
    ).flatten()
    peak_thr = compute_dynamic_peak_threshold(train_counts_inv)
    print(f"动态峰值阈值: {peak_thr:.0f}")

    # ── 8. 天气阈值（仅训练集）──
    weather_thresholds = None
    weather_precip = np.full(len(y_test), np.nan)
    weather_temp_high = np.full(len(y_test), np.nan)
    weather_temp_low = np.full(len(y_test), np.nan)

    wx_cols = ['meteo_precip_sum', 'meteo_temp_max', 'meteo_temp_min']
    if all(c in df.columns for c in wx_cols):
        weather_thresholds = compute_weather_thresholds_quantile(
            train_precip=df['meteo_precip_sum'].values[:train_size],
            train_temp_high=df['meteo_temp_max'].values[:train_size],
            train_temp_low=df['meteo_temp_min'].values[:train_size],
        )
        # 测试集天气（对齐到序列索引）
        test_df_start = trainval_size_seq + args.look_back
        test_wx = df.iloc[test_df_start:].reset_index(drop=True)
        n_w = min(len(test_wx), len(y_test))
        weather_precip[:n_w]   = test_wx['meteo_precip_sum'].values[:n_w]
        weather_temp_high[:n_w] = test_wx['meteo_temp_max'].values[:n_w]
        weather_temp_low[:n_w]  = test_wx['meteo_temp_min'].values[:n_w]

    # ── 9. 运行所有消融配置 ──
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = PROJECT_ROOT / args.output / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Ablation Study — {len(ABLATION_CONFIGS)} 个方案（特征遮蔽法）")
    print(f"模型权重: {weights_path.name}")
    print(f"{'='*60}")

    results = []
    for config_name, keep_features in ABLATION_CONFIGS.items():
        r = run_ablation_config(
            model=model,
            X_test_full=X_test_full,
            y_test=y_test,
            train_feature_means=train_feature_means,
            scaler=scaler,
            config_name=config_name,
            keep_features=keep_features,
            peak_thr=peak_thr,
            weather_precip=weather_precip,
            weather_temp_high=weather_temp_high,
            weather_temp_low=weather_temp_low,
            weather_thresholds=weather_thresholds,
        )
        if r:
            results.append(r)

    if not results:
        print("没有有效结果")
        return

    df_res = pd.DataFrame(results)

    # ── 10. 保存结果 ──
    df_res.to_csv(out_dir / 'ablation_results.csv', index=False)
    with open(out_dir / 'ablation_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2,
                  default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o))

    # ── 11. 打印汇总表 ──
    print(f"\n{'='*60}")
    print("Ablation Study 汇总（特征遮蔽法）")
    print(f"{'='*60}")
    cols = ['label', 'n_features_kept', 'mae', 'smape', 'suitability_f1', 'suitability_recall', 'brier']
    print(df_res[cols].to_string(index=False, float_format='{:.3f}'.format))

    # ── 12. 可视化 ──
    plot_ablation_results(df_res, out_dir)

    print(f"\n结果已保存至: {out_dir}")


if __name__ == '__main__':
    main()
