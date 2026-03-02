#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一可视化模块 - 九寨沟景区客流动态预测系统

严格遵循英文输出规范，防止编码问题
强制使用蓝色配色方案
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for batch processing
import matplotlib.pyplot as plt

# ==================== 常量定义 ====================

BLUE_CMAP = plt.cm.Blues  # 强制蓝色配色
PEAK_LABELS = ['non_peak', 'peak']  # 严格英文标签
OUTPUT_DIR = 'figures'

def setup_plot_style() -> None:
    """
    设置统一的绘图风格（英文）
    """
    plt.rcParams.update({
        'font.size': 10,
        'font.family': 'Arial',
        'axes.unicode_minus': False,
        'axes.labelsize': 10,
        'axes.titlesize': 12,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 9
    })

def plot_loss_curve(history, output_dir=OUTPUT_DIR) -> None:
    """
    绘制损失曲线（英文）
    
    Args:
        history: 训练历史
        output_dir: 输出目录
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    setup_plot_style()
    
    plt.figure(figsize=(10, 6))
    plt.plot(history['loss'], label='Training Loss', color='#1f77b4')
    plt.plot(history['val_loss'], label='Validation Loss', color='#ff7f0e')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'loss.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Loss curve saved to", os.path.join(output_dir, 'loss.png'))

def plot_true_vs_predicted(dates, y_true, y_pred, output_dir=OUTPUT_DIR) -> None:
    """
    绘制真实值与预测值对比图（英文）
    
    Args:
        dates: 日期序列
        y_true: 真实值数组
        y_pred: 预测值数组
        output_dir: 输出目录
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    setup_plot_style()
    
    plt.figure(figsize=(12, 6))
    plt.plot(dates, y_true, label='True Values', color='#1f77b4', linewidth=1.6)
    plt.plot(dates, y_pred, label='Predictions', color='#ff7f0e', linewidth=1.6)
    plt.xlabel('Date')
    plt.ylabel('Visitor Count')
    plt.title('Test Set: True Values vs Predictions')
    plt.xticks(rotation=30)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'true_vs_pred.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print("True vs Predicted plot saved to", os.path.join(output_dir, 'true_vs_pred.png'))

def plot_confusion_matrix(y_true, y_pred, output_dir=OUTPUT_DIR) -> None:
    """
    绘制混淆矩阵（英文，蓝色配色）
    
    Args:
        y_true: 真实值数组
        y_pred: 预测值数组
        output_dir: 输出目录
    """
    import os
    from sklearn.metrics import confusion_matrix
    
    os.makedirs(output_dir, exist_ok=True)
    
    PEAK_THRESHOLD = 18500
    y_true_cls = (y_true >= PEAK_THRESHOLD).astype(int)
    y_pred_cls = (y_pred >= PEAK_THRESHOLD).astype(int)
    
    setup_plot_style()
    
    disp = confusion_matrix(y_true_cls, y_pred_cls)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(disp, cmap=BLUE_CMAP)
    plt.title('Confusion Matrix (Count)')
    plt.colorbar()
    plt.xticks([0, 1], PEAK_LABELS)
    plt.yticks([0, 1], PEAK_LABELS)
    
    thresh = disp.max() / 2.
    for i in range(disp.shape[0]):
        for j in range(disp.shape[1]):
            plt.text(j, i, disp[i, j],
                    ha="center", va="center",
                    color="white" if disp[i, j] > thresh else "black")
    
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    disp_normalized = confusion_matrix(y_true_cls, y_pred_cls, normalize='true')
    
    plt.subplot(1, 2, 2)
    plt.imshow(disp_normalized, cmap=BLUE_CMAP)
    plt.title('Confusion Matrix (Normalized)')
    plt.colorbar()
    plt.xticks([0, 1], PEAK_LABELS)
    plt.yticks([0, 1], PEAK_LABELS)
    
    for i in range(disp_normalized.shape[0]):
        for j in range(disp_normalized.shape[1]):
            plt.text(j, i, f"{disp_normalized[i, j]:.2f}",
                    ha="center", va="center",
                    color="white" if disp_normalized[i, j] > 0.5 else "black")
    
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Confusion matrix saved to", os.path.join(output_dir, 'confusion_matrix.png'))

def generate_comprehensive_plots(history, dates, y_true, y_pred, output_dir=OUTPUT_DIR) -> None:
    """
    生成完整的可视化图表（英文）
    
    Args:
        history: 训练历史
        dates: 日期序列
        y_true: 真实值数组
        y_pred: 预测值数组
        output_dir: 输出目录
    """
    plot_loss_curve(history, output_dir)
    plot_true_vs_predicted(dates, y_true, y_pred, output_dir)
    plot_confusion_matrix(y_true, y_pred, output_dir)
    
    print("All comprehensive plots generated in", output_dir)

def plot_feature_importance(importances, feature_names, output_dir=OUTPUT_DIR) -> None:
    """
    绘制特征重要性（英文）
    
    Args:
        importances: 特征重要性数组
        feature_names: 特征名称列表
        output_dir: 输出目录
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    setup_plot_style()
    
    sorted_indices = np.argsort(importances)[::-1]
    sorted_importances = importances[sorted_indices]
    sorted_features = np.array(feature_names)[sorted_indices]
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(sorted_features)), sorted_importances, color='#1f77b4')
    plt.yticks(range(len(sorted_features)), sorted_features)
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature Name')
    plt.title('Feature Importance Analysis')
    plt.gca().invert_yaxis()
    plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Feature importance plot saved to", os.path.join(output_dir, 'feature_importance.png'))

def plot_actual_predicted_distribution(y_true, y_pred, output_dir=OUTPUT_DIR) -> None:
    """
    绘制真实值与预测值分布图（英文）
    
    Args:
        y_true: 真实值数组
        y_pred: 预测值数组
        output_dir: 输出目录
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    setup_plot_style()
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.6, color='#1f77b4')
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--', linewidth=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted Values')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'prediction_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Prediction distribution plot saved to", os.path.join(output_dir, 'prediction_distribution.png'))

def generate_comprehensive_report(
    history, dates, y_true, y_pred, feature_names=None,
    output_dir=OUTPUT_DIR
) -> None:
    """
    生成完整的可视化报告（英文）
    
    Args:
        history: 训练历史
        dates: 日期序列
        y_true: 真实值数组
        y_pred: 预测值数组
        feature_names: 特征名称列表（可选）
        output_dir: 输出目录
    """
    generate_comprehensive_plots(history, dates, y_true, y_pred, output_dir)
    
    if feature_names is not None:
        plot_feature_importance(np.ones(len(feature_names))/len(feature_names), 
                              feature_names, output_dir)
    
    plot_actual_predicted_distribution(y_true, y_pred, output_dir)
    
    print("Comprehensive report generated in", output_dir)
