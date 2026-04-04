"""
在线预测脚本 - 方案A（无需重新训练）

功能：
1. 加载已训练的模型
2. 获取最新历史数据（过去30天）
3. 获取未来7天的外部特征（天气预报、节假日）
4. 执行预测并保存结果
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import requests
import chinese_calendar
from tensorflow import keras

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def get_weather_forecast(latitude=33.252, longitude=103.918, days=7):
    """
    获取未来天气预报（Open-Meteo API）
    
    Args:
        latitude: 纬度
        longitude: 经度
        days: 预报天数
        
    Returns:
        DataFrame with weather forecast
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,windspeed_10m_max",
        "forecast_days": days,
        "timezone": "Asia/Shanghai"
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        df = pd.DataFrame({
            'date': data['daily']['time'],
            'temp_high': data['daily']['temperature_2m_max'],
            'temp_low': data['daily']['temperature_2m_min'],
            'precip_sum': data['daily']['precipitation_sum'],
            'wind_max': data['daily']['windspeed_10m_max']
        })
        
        return df
    except Exception as e:
        print(f"Error fetching weather forecast: {e}")
        return None


def get_future_holidays(days=7):
    """
    获取未来节假日信息
    
    Args:
        days: 未来天数
        
    Returns:
        List of (date_str, is_holiday)
    """
    holidays = []
    today = datetime.now().date()
    
    for i in range(days):
        future_date = today + timedelta(days=i+1)
        is_holiday = chinese_calendar.is_holiday(future_date)
        holidays.append((future_date.strftime('%Y-%m-%d'), int(is_holiday)))
    
    return holidays


def prepare_online_input(processed_csv_path, lookback=30, forecast_days=7):
    """
    准备在线预测的输入数据
    
    Args:
        processed_csv_path: 处理后的特征数据路径
        lookback: 历史窗口大小
        forecast_days: 预测天数
        
    Returns:
        (encoder_input, decoder_input, future_dates)
    """
    # 1. 加载历史数据
    df = pd.read_csv(processed_csv_path)
    df = df.sort_values('date').reset_index(drop=True)
    
    # 2. 获取最近 lookback 天的数据作为编码器输入
    encoder_data = df.tail(lookback).copy()
    
    # 3. 获取未来天气预报
    weather_forecast = get_weather_forecast(days=forecast_days)
    if weather_forecast is None:
        print("Warning: Failed to get weather forecast, using default values")
        # 使用默认值
        weather_forecast = pd.DataFrame({
            'date': [(datetime.now().date() + timedelta(days=i+1)).strftime('%Y-%m-%d') 
                     for i in range(forecast_days)],
            'temp_high': [20.0] * forecast_days,
            'temp_low': [10.0] * forecast_days,
            'precip_sum': [0.0] * forecast_days,
            'wind_max': [10.0] * forecast_days
        })
    
    # 4. 获取未来节假日
    future_holidays = get_future_holidays(days=forecast_days)
    
    # 5. 构建解码器输入（未来7天的外部特征）
    decoder_data = []
    for i, (date_str, is_holiday) in enumerate(future_holidays):
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        
        # 时间特征
        month = date_obj.month
        day_of_week = date_obj.weekday()
        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)
        dow_sin = np.sin(2 * np.pi * day_of_week / 7)
        dow_cos = np.cos(2 * np.pi * day_of_week / 7)
        
        # 天气特征（归一化）
        temp_high_scaled = (weather_forecast.iloc[i]['temp_high'] - df['temp_high_c'].min()) / \
                          (df['temp_high_c'].max() - df['temp_high_c'].min())
        temp_low_scaled = (weather_forecast.iloc[i]['temp_low'] - df['temp_low_c'].min()) / \
                         (df['temp_low_c'].max() - df['temp_low_c'].min())
        precip_scaled = weather_forecast.iloc[i]['precip_sum'] / df['meteo_precip_sum'].max()
        
        # 滞后特征（使用最新的历史数据）
        lag_7_scaled = encoder_data.iloc[-7]['tourism_num_lag_7_scaled'] if len(encoder_data) >= 7 else 0.5
        
        decoder_data.append([
            month_sin, month_cos, dow_sin, dow_cos,
            is_holiday, lag_7_scaled, precip_scaled,
            temp_high_scaled, temp_low_scaled
        ])
    
    # 6. 准备编码器输入（8特征）
    encoder_features = [
        'visitor_count_scaled', 'month_norm', 'day_of_week_norm', 'is_holiday',
        'tourism_num_lag_7_scaled', 'meteo_precip_sum_scaled',
        'temp_high_scaled', 'temp_low_scaled'
    ]
    
    # 归一化客流数据
    visitor_min = df['tourism_num'].min()
    visitor_max = df['tourism_num'].max()
    encoder_data['visitor_count_scaled'] = (encoder_data['tourism_num'] - visitor_min) / (visitor_max - visitor_min)
    encoder_data['month_norm'] = encoder_data['month'] / 12.0
    encoder_data['day_of_week_norm'] = encoder_data['day_of_week'] / 7.0
    
    encoder_input = encoder_data[encoder_features].values
    decoder_input = np.array(decoder_data)
    
    # 添加 batch 维度
    encoder_input = np.expand_dims(encoder_input, axis=0)
    decoder_input = np.expand_dims(decoder_input, axis=0)
    
    future_dates = [date_str for date_str, _ in future_holidays]
    
    return encoder_input, decoder_input, future_dates, visitor_min, visitor_max


def online_predict(model_path, processed_csv_path, output_csv_path=None):
    """
    执行在线预测
    
    Args:
        model_path: 模型权重路径
        processed_csv_path: 处理后的特征数据路径
        output_csv_path: 输出预测结果路径（可选）
        
    Returns:
        DataFrame with predictions
    """
    print(f"Loading model from: {model_path}")
    model = keras.models.load_model(model_path)
    
    print("Preparing input data...")
    encoder_input, decoder_input, future_dates, visitor_min, visitor_max = \
        prepare_online_input(processed_csv_path)
    
    print("Making predictions...")
    predictions_scaled = model.predict([encoder_input, decoder_input], verbose=0)
    
    # 反归一化
    predictions = predictions_scaled[0] * (visitor_max - visitor_min) + visitor_min
    predictions = np.maximum(predictions, 0)  # 确保非负
    
    # 构建结果 DataFrame
    results = pd.DataFrame({
        'date': future_dates,
        'predicted_visitors': predictions.astype(int)
    })
    
    print("\nPrediction Results:")
    print(results.to_string(index=False))
    
    if output_csv_path:
        results.to_csv(output_csv_path, index=False)
        print(f"\nResults saved to: {output_csv_path}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Online Prediction Script")
    parser.add_argument("--model", required=True, help="Path to model weights (.h5)")
    parser.add_argument("--data", required=True, help="Path to processed CSV")
    parser.add_argument("--output", default=None, help="Output CSV path")
    
    args = parser.parse_args()
    
    online_predict(args.model, args.data, args.output)
