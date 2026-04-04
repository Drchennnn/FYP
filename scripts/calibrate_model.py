"""
模型校准脚本 - Platt Scaling & Isotonic Regression

用于校准深度学习模型的概率输出，使其更接近真实的置信度。
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt


def load_predictions(model_dir):
    """加载模型预测结果和真实标签"""
    metrics_path = Path(model_dir) / 'metrics.json'
    
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    # 从 suitability_warning_by_horizon 提取预测概率和真实标签
    # 这里我们需要重新计算，因为 JSON 中没有保存原始概率
    # 暂时返回 None，需要修改训练脚本保存原始预测概率
    return None, None


def platt_scaling(y_true, y_prob):
    """
    Platt Scaling 校准
    
    使用 Logistic Regression 将原始概率映射到校准后的概率
    """
    # 将概率转换为 logits
    y_prob = np.clip(y_prob, 1e-7, 1 - 1e-7)
    
    # 训练 Logistic Regression
    lr = LogisticRegression()
    lr.fit(y_prob.reshape(-1, 1), y_true)
    
    # 返回校准后的概率
    y_calibrated = lr.predict_proba(y_prob.reshape(-1, 1))[:, 1]
    
    return y_calibrated, lr


def isotonic_regression(y_true, y_prob):
    """
    Isotonic Regression 校准
    
    使用保序回归将原始概率映射到校准后的概率
    """
    ir = IsotonicRegression(out_of_bounds='clip')
    y_calibrated = ir.fit_transform(y_prob, y_true)
    
    return y_calibrated, ir


def calculate_ece(y_true, y_prob, n_bins=10):
    """计算 Expected Calibration Error"""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    ece = 0.0
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_acc = y_true[mask].mean()
            bin_conf = y_prob[mask].mean()
            bin_weight = mask.sum() / len(y_true)
            ece += bin_weight * abs(bin_acc - bin_conf)
    
    return ece


def plot_reliability_diagram(y_true, y_prob_raw, y_prob_platt, y_prob_iso, output_path):
    """绘制校准前后的 Reliability Diagram"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    methods = [
        ('Raw (Uncalibrated)', y_prob_raw),
        ('Platt Scaling', y_prob_platt),
        ('Isotonic Regression', y_prob_iso)
    ]
    
    for ax, (title, y_prob) in zip(axes, methods):
        # 计算每个 bin 的统计
        bins = np.linspace(0, 1, 11)
        bin_indices = np.digitize(y_prob, bins) - 1
        bin_indices = np.clip(bin_indices, 0, 9)
        
        bin_accs = []
        bin_confs = []
        bin_counts = []
        
        for i in range(10):
            mask = bin_indices == i
            if mask.sum() > 0:
                bin_accs.append(y_true[mask].mean())
                bin_confs.append(y_prob[mask].mean())
                bin_counts.append(mask.sum())
            else:
                bin_accs.append(0)
                bin_confs.append((bins[i] + bins[i+1]) / 2)
                bin_counts.append(0)
        
        # 绘制
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=2)
        ax.scatter(bin_confs, bin_accs, s=[c*2 for c in bin_counts], 
                  alpha=0.6, label='Actual')
        
        # 计算 ECE
        ece = calculate_ece(y_true, y_prob)
        
        ax.set_xlabel('Mean Predicted Probability', fontsize=12)
        ax.set_ylabel('Fraction of Positives', fontsize=12)
        ax.set_title(f'{title}\nECE = {ece:.4f}', fontsize=14)
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f'Reliability diagram saved to: {output_path}')


def calibrate_model(model_dir, output_dir=None):
    """
    对模型进行校准
    
    Args:
        model_dir: 模型输出目录
        output_dir: 校准结果输出目录（默认为 model_dir/calibrated）
    """
    if output_dir is None:
        output_dir = Path(model_dir) / 'calibrated'
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f'Loading predictions from: {model_dir}')
    
    # TODO: 需要修改训练脚本保存原始预测概率
    # 目前 metrics.json 中没有保存原始概率，只有聚合后的指标
    
    print('ERROR: Original prediction probabilities not found in metrics.json')
    print('Please modify training script to save raw predictions.')
    print()
    print('Required format:')
    print('{')
    print('  "raw_predictions": {')
    print('    "y_true": [0, 1, 0, ...],')
    print('    "y_prob": [0.1, 0.9, 0.2, ...]')
    print('  }')
    print('}')
    
    return None


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Calibrate model predictions')
    parser.add_argument('--model-dir', required=True, help='Model output directory')
    parser.add_argument('--output-dir', default=None, help='Calibration output directory')
    
    args = parser.parse_args()
    
    calibrate_model(args.model_dir, args.output_dir)
