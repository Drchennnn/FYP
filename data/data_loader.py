#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据加载器 - 九寨沟景区客流动态预测系统

提供统一的数据加载、预处理和序列构建功能
支持4特征和8特征版本，以及Seq2Seq特殊处理
"""

import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.preprocessing import MinMaxScaler

def load_and_preprocess_data(file_path: str, features: int = 8) -> Tuple[pd.DataFrame, MinMaxScaler]:
    """
    加载和预处理数据
    
    Args:
        file_path: 数据文件路径
        features: 特征数量 [4, 8]
        
    Returns:
        预处理后的数据, 归一化器
    """
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # ==================== 特征工程 ====================
    
    # 时间特征
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.weekday
    
    # 日期归一化
    df['month_norm'] = (df['month'] - 1) / 11.0
    df['day_of_week_norm'] = df['day_of_week'] / 6.0
    
    # 节假日识别（简化）
    df['is_holiday'] = 0
    holiday_dates = [
        # 春节期间
        pd.to_datetime('2024-02-10'), pd.to_datetime('2024-02-11'),
        pd.to_datetime('2024-02-12'), pd.to_datetime('2024-02-13'),
        pd.to_datetime('2024-02-14'), pd.to_datetime('2024-02-15'),
        pd.to_datetime('2024-02-16'),
        
        # 劳动节
        pd.to_datetime('2024-05-01'), pd.to_datetime('2024-05-02'),
        pd.to_datetime('2024-05-03'),
        
        # 国庆节
        pd.to_datetime('2024-10-01'), pd.to_datetime('2024-10-02'),
        pd.to_datetime('2024-10-03'), pd.to_datetime('2024-10-04'),
        pd.to_datetime('2024-10-05'), pd.to_datetime('2024-10-06'),
        pd.to_datetime('2024-10-07'),
        
        # 其他重要假期
        pd.to_datetime('2024-01-01')  # 元旦
    ]
    
    for holiday in holiday_dates:
        df.loc[df['date'] == holiday, 'is_holiday'] = 1
    
    # ==================== 特征选择 ====================
    
    if features == 4:
        required_cols = [
            'date', 'visitor_count_scaled', 'month_norm', 
            'day_of_week_norm', 'is_holiday'
        ]
    elif features == 8:
        required_cols = [
            'date', 'visitor_count_scaled', 'month_norm', 
            'day_of_week_norm', 'is_holiday',
            'tourism_num_lag_7_scaled', 'meteo_precip_sum_scaled',
            'temp_high_scaled', 'temp_low_scaled'
        ]
    else:
        raise ValueError(f"Unsupported feature count: {features}")
    
    # 检查并填充缺失特征
    for col in required_cols[1:]:
        if col not in df.columns:
            df[col] = 0.0
    
    df = df[required_cols]
    
    # ==================== 归一化 ====================
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    visitor_scaler = MinMaxScaler(feature_range=(0, 1))
    
    # 单独归一化游客数量
    if 'visitor_count' in df.columns:
        df['visitor_count_scaled'] = visitor_scaler.fit_transform(
            df['visitor_count'].values.reshape(-1, 1)
        )
    else:
        # 如果列名不同，尝试其他可能的列名
        target_col = 'tourism_num' if 'tourism_num' in df.columns else 'visitor_count_scaled'
        df['visitor_count_scaled'] = visitor_scaler.fit_transform(
            df[target_col].values.reshape(-1, 1)
        )
    
    return df, visitor_scaler

def build_sequences(
    df: pd.DataFrame,
    look_back: int,
    model_type: str = "lstm",
    features: int = 8
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    构建时间序列滑动窗口
    
    Args:
        df: 包含特征和标签的DataFrame
        look_back: 历史窗口大小
        model_type: 模型类型 ['lstm', 'gru', 'seq2seq_attention']
        features: 特征数量 [4, 8]
        
    Returns:
        输入序列, 目标值, 日期序列
    """
    
    # ==================== 特征列选择 ====================
    
    base_4_features = [
        "visitor_count_scaled",
        "month_norm",
        "day_of_week_norm",
        "is_holiday"
    ]
    
    additional_4_features = [
        "tourism_num_lag_7_scaled",
        "meteo_precip_sum_scaled",
        "temp_high_scaled",
        "temp_low_scaled"
    ]
    
    if features == 4:
        feature_cols = base_4_features
    elif features == 8:
        feature_cols = base_4_features + additional_4_features
    else:
        raise ValueError(f"Unsupported feature count: {features}")
    
    # ==================== 数据验证 ====================
    
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        for col in missing_cols:
            df[col] = 0.0
    
    # ==================== Seq2Seq 适配器 ====================
    
    if model_type == "seq2seq_attention":
        if features != 8:
            features = 8
            feature_cols = base_4_features + additional_4_features
            
        return _build_seq2seq_sequences(df, look_back, feature_cols)
    
    # ==================== LSTM/GRU 通用模式 ====================
    
    return _build_standard_sequences(df, look_back, feature_cols)

def _build_standard_sequences(
    df: pd.DataFrame,
    look_back: int,
    feature_cols: list
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """构建标准LSTM/GRU序列"""
    
    values = df[feature_cols].values.astype(np.float32)
    target = df["visitor_count_scaled"].values.astype(np.float32)
    dates = df["date"].values
    
    x_list, y_list, d_list = [], [], []
    
    for i in range(look_back, len(df)):
        x_list.append(values[i - look_back : i, :])
        y_list.append(target[i])
        d_list.append(dates[i])
    
    x = np.array(x_list)
    y = np.array(y_list)
    d = np.array(d_list)
    
    return x, y, d

def _build_seq2seq_sequences(
    df: pd.DataFrame,
    look_back: int,
    feature_cols: list
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """构建Seq2Seq+Attention序列"""
    
    values = df[feature_cols].values.astype(np.float32)
    target = df["visitor_count_scaled"].values.astype(np.float32)
    dates = df["date"].values
    
    predict_horizon = 7
    
    encoder_inputs = []
    decoder_inputs = []
    decoder_targets = []
    date_list = []
    
    for i in range(look_back, len(df) - predict_horizon + 1):
        encoder_input = values[i - look_back : i, :]
        encoder_inputs.append(encoder_input)
        
        decoder_input = values[i : i + predict_horizon, :]
        decoder_inputs.append(decoder_input)
        
        decoder_target = target[i : i + predict_horizon]
        decoder_targets.append(decoder_target)
        
        date_list.append(dates[i])
    
    return (
        np.array(encoder_inputs),
        np.array(decoder_inputs),
        np.array(decoder_targets),
        np.array(date_list)
    )

def train_test_split(
    x: np.ndarray,
    y: np.ndarray,
    dates: np.ndarray,
    test_size: float = 0.2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    划分训练和测试集
    
    Args:
        x: 输入序列
        y: 目标值
        dates: 日期序列
        test_size: 测试集比例
        
    Returns:
        训练集和测试集
    """
    n_samples = len(x)
    test_samples = int(n_samples * test_size)
    
    x_train = x[:-test_samples]
    y_train = y[:-test_samples]
    d_train = dates[:-test_samples]
    
    x_test = x[-test_samples:]
    y_test = y[-test_samples:]
    d_test = dates[-test_samples:]
    
    return x_train, y_train, d_train, x_test, y_test, d_test

def inverse_transform_scaler(value: float, scaler: MinMaxScaler) -> float:
    """
    反归一化单个值
    
    Args:
        value: 归一化后的值
        scaler: 归一化器
        
    Returns:
        原始值
    """
    return scaler.inverse_transform(np.array([[value]]))[0][0]
