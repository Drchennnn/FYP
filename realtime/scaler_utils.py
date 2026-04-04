"""
保存和加载 MinMaxScaler 的工具脚本

用于确保预测时使用与训练时相同的归一化参数
"""

import pickle
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler


def save_scaler_from_training_data(
    data_path: str,
    output_path: str,
    feature_ranges: dict = None
):
    """
    从训练数据创建并保存 Scaler
    
    Args:
        data_path: 训练数据 CSV 路径
        output_path: Scaler 保存路径
        feature_ranges: 自定义特征范围（可选）
    """
    import pandas as pd
    
    df = pd.read_csv(data_path)
    
    # 默认特征范围（基于训练脚本）
    if feature_ranges is None:
        feature_ranges = {
            'visitor_count': (0, 50000),
            'tourism_num_lag_7': (0, 50000),
            'precipitation': (0, 50),
            'temp_high': (-10, 40),
            'temp_low': (-10, 40)
        }
    
    # 创建 Scaler 字典
    scalers = {}
    
    for feature, (min_val, max_val) in feature_ranges.items():
        scaler = MinMaxScaler(feature_range=(0, 1))
        
        # 使用预定义范围拟合
        dummy_data = np.array([[min_val], [max_val]])
        scaler.fit(dummy_data)
        
        scalers[feature] = scaler
    
    # 保存
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(scalers, f)
    
    print(f'Scalers saved to: {output_path}')
    print(f'Features: {list(scalers.keys())}')
    
    return scalers


def load_scalers(scaler_path: str) -> dict:
    """
    加载 Scaler
    
    Args:
        scaler_path: Scaler 文件路径
        
    Returns:
        dict of {feature_name: MinMaxScaler}
    """
    with open(scaler_path, 'rb') as f:
        scalers = pickle.load(f)
    
    print(f'Scalers loaded from: {scaler_path}')
    print(f'Features: {list(scalers.keys())}')
    
    return scalers


def transform_feature(scaler: MinMaxScaler, value: float) -> float:
    """归一化单个特征值"""
    return scaler.transform([[value]])[0][0]


def inverse_transform_feature(scaler: MinMaxScaler, scaled_value: float) -> float:
    """反归一化单个特征值"""
    return scaler.inverse_transform([[scaled_value]])[0][0]


if __name__ == '__main__':
    # 创建并保存 Scaler
    data_path = 'data/processed/jiuzhaigou_daily_features_2016-01-01_2026-04-02.csv'
    output_path = 'models/scalers/feature_scalers.pkl'
    
    if Path(data_path).exists():
        scalers = save_scaler_from_training_data(data_path, output_path)
        
        # 测试
        print()
        print('Testing scalers...')
        
        # 测试客流量归一化
        visitor_count = 20000
        scaled = transform_feature(scalers['visitor_count'], visitor_count)
        inversed = inverse_transform_feature(scalers['visitor_count'], scaled)
        
        print(f'Original: {visitor_count}')
        print(f'Scaled: {scaled:.4f}')
        print(f'Inversed: {inversed:.2f}')
        
        # 测试温度归一化
        temp = 18.5
        scaled = transform_feature(scalers['temp_high'], temp)
        inversed = inverse_transform_feature(scalers['temp_high'], scaled)
        
        print()
        print(f'Original temp: {temp}')
        print(f'Scaled: {scaled:.4f}')
        print(f'Inversed: {inversed:.2f}')
    else:
        print(f'Data file not found: {data_path}')
