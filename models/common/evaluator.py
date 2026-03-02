#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一评估模块 - 九寨沟景区客流动态预测系统

本模块提供完整的模型评估功能，支持回归和分类指标计算
严格遵循英文输出规范，防止编码问题
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for batch processing
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# ==================== 常量定义 ====================

PEAK_THRESHOLD = 18500  # 峰值阈值
CONFUSION_MATRIX_CMAP = plt.cm.Blues  # 强制蓝色配色
CONFUSION_LABELS = ['non_peak', 'peak']  # 严格英文标签
METRICS_FILENAME = "model_evaluation_report.json"

def save_metrics_to_files(metrics: Dict, run_dir: str, model_name: str) -> None:
    """
    将指标保存到JSON和CSV文件
    
    Args:
        metrics: 评估指标字典
        run_dir: 输出目录
        model_name: 模型名称
    """
    import json
    import os
    
    os.makedirs(run_dir, exist_ok=True)
    
    metrics_json_path = os.path.join(run_dir, f"{model_name}_metrics.json")
    metrics_csv_path = os.path.join(run_dir, f"{model_name}_metrics.csv")
    
    # 转换所有 NumPy 类型到 Python 原生类型
    def convert_numpy(obj):
        if isinstance(obj, np.generic):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        else:
            return obj
    
    metrics = convert_numpy(metrics)
    
    with open(metrics_json_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
        
    print(f"Metrics saved to {metrics_json_path}")

def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """
    计算回归指标
    
    Args:
        y_true: 真实值数组
        y_pred: 预测值数组
        
    Returns:
        回归指标字典
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    smape = 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))
    r2 = r2_score(y_true, y_pred)
    
    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "smape": smape,
        "r2": r2
    }

def calculate_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """
    计算分类指标（峰值检测）
    
    Args:
        y_true: 真实值数组
        y_pred: 预测值数组
        
    Returns:
        分类指标字典
    """
    y_true_cls = (y_true >= PEAK_THRESHOLD).astype(int)
    y_pred_cls = (y_pred >= PEAK_THRESHOLD).astype(int)
    
    accuracy = accuracy_score(y_true_cls, y_pred_cls)
    precision = precision_score(y_true_cls, y_pred_cls)
    recall = recall_score(y_true_cls, y_pred_cls)
    f1 = f1_score(y_true_cls, y_pred_cls)
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "y_true_cls": y_true_cls,
        "y_pred_cls": y_pred_cls
    }

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, scaler: MinMaxScaler) -> Dict:
    """
    完整评估流程
    
    Args:
        y_true: 真实值数组（原始尺度）
        y_pred: 预测值数组（原始尺度）
        scaler: 归一化器
        
    Returns:
        完整指标字典
    """
    regression_metrics = calculate_regression_metrics(y_true, y_pred)
    classification_metrics = calculate_classification_metrics(y_true, y_pred)
    
    return {
        "regression": regression_metrics,
        "classification": classification_metrics,
        "peak_threshold": PEAK_THRESHOLD
    }

def plot_training_loss(out_dir: str, history) -> None:
    """
    绘制训练损失曲线（严格英文）
    
    Args:
        out_dir: 输出目录
        history: 训练历史
    """
    import os
    os.makedirs(out_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(out_dir, 'loss.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Loss plot saved to", os.path.join(out_dir, 'loss.png'))

def plot_true_vs_pred(out_dir: str, dates: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """
    绘制真实值与预测值对比图（严格英文）
    
    Args:
        out_dir: 输出目录
        dates: 日期序列
        y_true: 真实值数组
        y_pred: 预测值数组
    """
    import os
    os.makedirs(out_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 6))
    plt.plot(dates, y_true, label='True Values', linewidth=1.6)
    plt.plot(dates, y_pred, label='Predictions', linewidth=1.6)
    plt.xlabel('Date')
    plt.ylabel('Visitor Count')
    plt.title('Test Set: True Values vs Predictions')
    plt.xticks(rotation=30)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(out_dir, 'true_vs_pred.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print("True vs Pred plot saved to", os.path.join(out_dir, 'true_vs_pred.png'))

def plot_confusion_matrices(out_dir: str, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """
    绘制混淆矩阵（严格英文，蓝色配色）
    
    Args:
        out_dir: 输出目录
        y_true: 真实值数组
        y_pred: 预测值数组
    """
    import os
    os.makedirs(out_dir, exist_ok=True)
    
    y_true_cls = (y_true >= PEAK_THRESHOLD).astype(int)
    y_pred_cls = (y_pred >= PEAK_THRESHOLD).astype(int)
    
    # Confusion matrix 1: Count
    disp1 = confusion_matrix(y_true_cls, y_pred_cls)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(disp1, cmap=CONFUSION_MATRIX_CMAP)
    plt.title('Confusion Matrix (Count)')
    plt.colorbar()
    plt.xticks([0, 1], CONFUSION_LABELS)
    plt.yticks([0, 1], CONFUSION_LABELS)
    
    # 添加数值标签
    thresh = disp1.max() / 2.
    for i in range(disp1.shape[0]):
        for j in range(disp1.shape[1]):
            plt.text(j, i, disp1[i, j],
                    ha="center", va="center",
                    color="white" if disp1[i, j] > thresh else "black")
    
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    # Confusion matrix 2: Normalized
    disp2 = confusion_matrix(y_true_cls, y_pred_cls, normalize='true')
    
    plt.subplot(1, 2, 2)
    plt.imshow(disp2, cmap=CONFUSION_MATRIX_CMAP)
    plt.title('Confusion Matrix (Normalized)')
    plt.colorbar()
    plt.xticks([0, 1], CONFUSION_LABELS)
    plt.yticks([0, 1], CONFUSION_LABELS)
    
    for i in range(disp2.shape[0]):
        for j in range(disp2.shape[1]):
            plt.text(j, i, f"{disp2[i, j]:.2f}",
                    ha="center", va="center",
                    color="white" if disp2[i, j] > 0.5 else "black")
    
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Confusion matrix saved to", os.path.join(out_dir, 'confusion_matrix.png'))

def generate_visualizations(out_dir: str, history=None, dates=None, y_true=None, y_pred=None) -> None:
    """
    生成所有可视化图表（严格英文）
    
    Args:
        out_dir: 输出目录
        history: 训练历史（可选）
        dates: 日期序列（可选）
        y_true: 真实值数组（可选）
        y_pred: 预测值数组（可选）
    """
    import os
    os.makedirs(out_dir, exist_ok=True)
    
    if history is not None:
        plot_training_loss(out_dir, history)
    
    if dates is not None and y_true is not None and y_pred is not None:
        plot_true_vs_pred(out_dir, dates, y_true, y_pred)
        plot_confusion_matrices(out_dir, y_true, y_pred)
    
    print("All visualizations generated in", out_dir)
