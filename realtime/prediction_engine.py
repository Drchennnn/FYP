"""
实时预测引擎模块

功能：
1. 加载训练好的 GRU 模型
2. 准备实时特征（天气、节假日、滞后特征）
3. 滚动预测未来 7 天客流
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime, timedelta
from pathlib import Path
import chinese_calendar as cncal
from typing import List, Dict, Tuple
import pickle


class RealtimePredictionEngine:
    """实时预测引擎"""
    
    def __init__(self, model_path: str, scaler_path: str = None, lookback: int = 30):
        """
        Args:
            model_path: 训练好的模型路径
            scaler_path: MinMaxScaler 路径（如果有）
            lookback: 历史窗口长度
        """
        self.lookback = lookback
        self.model = tf.keras.models.load_model(model_path)
        
        # 加载 scaler（如果有）
        if scaler_path and Path(scaler_path).exists():
            with open(scaler_path, 'rb') as f:
                self.scalers = pickle.load(f)
            print(f'Scalers loaded from: {scaler_path}')
        else:
            self.scalers = None
            print('Warning: No scalers loaded, using default normalization')
        
        print(f'Model loaded from: {model_path}')
        print(f'Lookback: {lookback}')
    
    def predict_future(
        self, 
        historical_data: pd.DataFrame,
        weather_forecast: pd.DataFrame,
        forecast_days: int = 7
    ) -> pd.DataFrame:
        """
        预测未来客流
        
        Args:
            historical_data: 历史数据 (至少 lookback 天)
                columns: date, visitor_count, temp_high, temp_low, precipitation
            weather_forecast: 未来天气预报
                columns: date, temp_high, temp_low, precipitation
            forecast_days: 预测天数
            
        Returns:
            DataFrame with columns: date, predicted_visitor_count
        """
        # 1. 准备历史特征
        hist_features = self._prepare_features(historical_data)
        
        # 2. 滚动预测
        predictions = []
        dates = []
        
        # 使用最近 lookback 天作为初始窗口
        current_window = hist_features[-self.lookback:].copy()
        
        for i in range(forecast_days):
            forecast_date = pd.to_datetime(weather_forecast.iloc[i]['date'])
            
            # 准备当天特征
            day_features = self._prepare_forecast_features(
                forecast_date,
                weather_forecast.iloc[i],
                current_window
            )
            
            # 预测
            X = current_window.reshape(1, self.lookback, -1)
            pred_scaled = self.model.predict(X, verbose=0)[0][0]
            
            # 反归一化
            if self.scaler:
                pred = self._inverse_transform_visitor_count(pred_scaled)
            else:
                pred = pred_scaled
            
            predictions.append(max(0, pred))  # 确保非负
            dates.append(forecast_date.strftime('%Y-%m-%d'))
            
            # 更新滑动窗口
            day_features[0] = pred_scaled  # 使用归一化后的预测值作为下一步的输入
            current_window = np.vstack([current_window[1:], day_features])
        
        return pd.DataFrame({
            'date': dates,
            'predicted_visitor_count': predictions
        })
    
    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        准备历史特征
        
        Returns:
            shape: (n_samples, 8)
            features: [visitor_count_scaled, month_norm, day_of_week_norm, 
                      is_holiday, tourism_num_lag_7_scaled, 
                      meteo_precip_sum_scaled, temp_high_scaled, temp_low_scaled]
        """
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        
        # 时间特征
        df['month'] = df['date'].dt.month
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month_norm'] = (df['month'] - 1) / 11.0
        df['day_of_week_norm'] = df['day_of_week'] / 6.0
        
        # 节假日特征
        df['is_holiday'] = df['date'].apply(self._is_holiday)
        
        # 滞后特征
        df['tourism_num_lag_7'] = df['visitor_count'].shift(7)
        df['tourism_num_lag_7'].fillna(df['visitor_count'].mean(), inplace=True)
        
        # 归一化（使用真实 scaler）
        if self.scalers:
            df['visitor_count_scaled'] = df['visitor_count'].apply(
                lambda x: self.scalers['visitor_count'].transform([[x]])[0][0]
            )
            df['tourism_num_lag_7_scaled'] = df['tourism_num_lag_7'].apply(
                lambda x: self.scalers['tourism_num_lag_7'].transform([[x]])[0][0]
            )
            df['meteo_precip_sum_scaled'] = df['precipitation'].apply(
                lambda x: self.scalers['precipitation'].transform([[x]])[0][0]
            )
            df['temp_high_scaled'] = df['temp_high'].apply(
                lambda x: self.scalers['temp_high'].transform([[x]])[0][0]
            )
            df['temp_low_scaled'] = df['temp_low'].apply(
                lambda x: self.scalers['temp_low'].transform([[x]])[0][0]
            )
        else:
            # 降级方案：使用简单归一化
            df['visitor_count_scaled'] = df['visitor_count'] / 50000.0
            df['tourism_num_lag_7_scaled'] = df['tourism_num_lag_7'] / 50000.0
            df['meteo_precip_sum_scaled'] = df['precipitation'] / 50.0
            df['temp_high_scaled'] = (df['temp_high'] + 10) / 50.0
            df['temp_low_scaled'] = (df['temp_low'] + 10) / 50.0
        
        # 选择特征
        feature_cols = [
            'visitor_count_scaled',
            'month_norm',
            'day_of_week_norm',
            'is_holiday',
            'tourism_num_lag_7_scaled',
            'meteo_precip_sum_scaled',
            'temp_high_scaled',
            'temp_low_scaled'
        ]
        
        return df[feature_cols].values
    
    def _prepare_forecast_features(
        self, 
        forecast_date: pd.Timestamp,
        weather: pd.Series,
        current_window: np.ndarray
    ) -> np.ndarray:
        """
        准备预测日特征
        
        Returns:
            shape: (8,)
        """
        # 时间特征
        month_norm = (forecast_date.month - 1) / 11.0
        day_of_week_norm = forecast_date.dayofweek / 6.0
        
        # 节假日
        is_holiday = self._is_holiday(forecast_date)
        
        # 滞后特征（使用 7 天前的预测值）
        if len(current_window) >= 7:
            tourism_num_lag_7_scaled = current_window[-7][0]
        else:
            tourism_num_lag_7_scaled = current_window[-1][0]
        
        # 天气特征（归一化）
        meteo_precip_sum_scaled = weather['precipitation'] / 50.0
        temp_high_scaled = (weather['temp_high'] + 10) / 50.0
        temp_low_scaled = (weather['temp_low'] + 10) / 50.0
        
        # visitor_count_scaled 将由预测结果填充
        return np.array([
            0.0,  # visitor_count_scaled (placeholder)
            month_norm,
            day_of_week_norm,
            is_holiday,
            tourism_num_lag_7_scaled,
            meteo_precip_sum_scaled,
            temp_high_scaled,
            temp_low_scaled
        ])
    
    def _is_holiday(self, date) -> int:
        """判断是否为节假日"""
        if isinstance(date, str):
            date = pd.to_datetime(date)
        
        # 国庆黄金周
        if date.month == 10 and 1 <= date.day <= 7:
            return 1
        
        # 劳动节
        if date.month == 5 and 1 <= date.day <= 5:
            return 1
        
        # 使用 chinese_calendar 判断其他节假日
        try:
            return 1 if cncal.is_holiday(date) else 0
        except:
            return 0
    
    def _inverse_transform_visitor_count(self, scaled_value: float) -> float:
        """反归一化客流量"""
        if self.scalers and 'visitor_count' in self.scalers:
            return self.scalers['visitor_count'].inverse_transform([[scaled_value]])[0][0]
        else:
            # 降级方案
            return scaled_value * 50000.0


if __name__ == '__main__':
    # 测试
    print('Testing realtime prediction engine...')
    print()
    
    # 模拟历史数据
    dates = pd.date_range(end=datetime.now(), periods=40, freq='D')
    historical_data = pd.DataFrame({
        'date': dates,
        'visitor_count': np.random.randint(10000, 30000, size=40),
        'temp_high': np.random.uniform(15, 25, size=40),
        'temp_low': np.random.uniform(5, 15, size=40),
        'precipitation': np.random.uniform(0, 10, size=40)
    })
    
    # 模拟天气预报
    forecast_dates = pd.date_range(start=datetime.now() + timedelta(days=1), periods=7, freq='D')
    weather_forecast = pd.DataFrame({
        'date': forecast_dates,
        'temp_high': np.random.uniform(15, 25, size=7),
        'temp_low': np.random.uniform(5, 15, size=7),
        'precipitation': np.random.uniform(0, 10, size=7)
    })
    
    # 加载模型并预测
    model_path = 'output/runs/gru_8features_20260403_163039/runs/run_20260403_163039_lb30_ep120_gru_8features/weights/gru_jiuzhaigou.h5'
    
    if Path(model_path).exists():
        engine = RealtimePredictionEngine(model_path, lookback=30)
        predictions = engine.predict_future(historical_data, weather_forecast, forecast_days=7)
        print('Predictions:')
        print(predictions)
    else:
        print(f'Model not found: {model_path}')
