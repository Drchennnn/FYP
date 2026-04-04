"""
多模型预测引擎

功能：
1. 加载三个 8 特征模型（LSTM、GRU、Seq2Seq+Attention）
2. 并行预测
3. 返回对比结果
"""

import tensorflow as tf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import pickle
import sys

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import chinese_calendar


class MultiModelPredictionEngine:
    """多模型预测引擎"""
    
    def __init__(self, model_configs: dict, scaler_path: str = None, lookback: int = 30):
        """
        Args:
            model_configs: {
                'lstm': {'path': '...', 'name': 'LSTM'},
                'gru': {'path': '...', 'name': 'GRU'},
                'seq2seq': {'path': '...', 'name': 'Seq2Seq+Attention'}
            }
            scaler_path: MinMaxScaler 路径
            lookback: 历史窗口长度
        """
        self.lookback = lookback
        self.models = {}
        
        # 加载所有模型
        for model_key, config in model_configs.items():
            try:
                model = tf.keras.models.load_model(config['path'])
                self.models[model_key] = {
                    'model': model,
                    'name': config['name']
                }
                print(f'✅ {config["name"]} loaded from: {config["path"]}')
            except Exception as e:
                print(f'⚠️  Failed to load {config["name"]}: {e}')
        
        # 加载 scaler
        if scaler_path and Path(scaler_path).exists():
            with open(scaler_path, 'rb') as f:
                self.scalers = pickle.load(f)
            print(f'✅ Scalers loaded from: {scaler_path}')
        else:
            self.scalers = None
            print('⚠️  Warning: No scalers loaded, using default normalization')
        
        print(f'Lookback: {lookback}')
        print(f'Models loaded: {len(self.models)}')
    
    def predict_future_all_models(
        self,
        historical_data: pd.DataFrame,
        weather_forecast: pd.DataFrame,
        forecast_days: int = 7
    ) -> dict:
        """
        使用所有模型预测未来
        
        Args:
            historical_data: 历史数据（至少 lookback 天）
            weather_forecast: 天气预报数据
            forecast_days: 预测天数
            
        Returns:
            {
                'lstm': DataFrame with predictions,
                'gru': DataFrame with predictions,
                'seq2seq': DataFrame with predictions
            }
        """
        results = {}
        
        for model_key, model_info in self.models.items():
            try:
                predictions = self._predict_single_model(
                    model_info['model'],
                    historical_data.copy(),
                    weather_forecast.copy(),
                    forecast_days
                )
                results[model_key] = predictions
                print(f'✅ {model_info["name"]} prediction completed')
            except Exception as e:
                print(f'❌ {model_info["name"]} prediction failed: {e}')
                results[model_key] = None
        
        return results
    
    def _predict_single_model(
        self,
        model,
        historical_data: pd.DataFrame,
        weather_forecast: pd.DataFrame,
        forecast_days: int
    ) -> pd.DataFrame:
        """单个模型预测"""
        
        # 确保有足够的历史数据
        if len(historical_data) < self.lookback:
            raise ValueError(f'Need at least {self.lookback} days of historical data')
        
        # 准备特征
        df = self._prepare_features(historical_data, weather_forecast, forecast_days)
        
        # 归一化
        df = self._normalize_features(df)
        
        # 滚动预测
        predictions = []
        
        for i in range(forecast_days):
            # 获取最近 lookback 天的数据
            window_data = df.iloc[-(self.lookback + forecast_days - i):-(forecast_days - i)]
            
            # 构建输入序列
            X = window_data[[
                'visitor_count_scaled',
                'tourism_num_lag_7_scaled',
                'month_sin', 'month_cos',
                'weekday_sin', 'weekday_cos',
                'is_holiday',
                'meteo_precip_sum_scaled',
                'temp_high_scaled',
                'temp_low_scaled'
            ]].values
            
            X = X.reshape(1, self.lookback, 10)
            
            # 预测
            pred_scaled = model.predict(X, verbose=0)[0][0]
            pred_value = self._inverse_transform_visitor_count(pred_scaled)
            
            predictions.append(pred_value)
            
            # 更新下一步的输入
            next_idx = len(df) - forecast_days + i
            df.loc[df.index[next_idx], 'visitor_count_scaled'] = pred_scaled
            df.loc[df.index[next_idx], 'visitor_count'] = pred_value
            
            # 更新滞后特征
            if i >= 7:
                lag_7_value = predictions[i - 7]
                df.loc[df.index[next_idx], 'tourism_num_lag_7'] = lag_7_value
                if self.scalers:
                    df.loc[df.index[next_idx], 'tourism_num_lag_7_scaled'] = \
                        self.scalers['tourism_num_lag_7'].transform([[lag_7_value]])[0][0]
                else:
                    df.loc[df.index[next_idx], 'tourism_num_lag_7_scaled'] = lag_7_value / 50000.0
        
        # 构建结果
        result_df = df.tail(forecast_days)[['date', 'temp_high', 'temp_low', 'precipitation']].copy()
        result_df['predicted_visitor_count'] = predictions
        
        return result_df
    
    def _prepare_features(
        self,
        historical_data: pd.DataFrame,
        weather_forecast: pd.DataFrame,
        forecast_days: int
    ) -> pd.DataFrame:
        """准备特征"""
        
        # 合并历史数据和未来日期
        last_date = pd.to_datetime(historical_data['date'].iloc[-1])
        future_dates = [last_date + timedelta(days=i+1) for i in range(forecast_days)]
        
        future_df = pd.DataFrame({
            'date': [d.strftime('%Y-%m-%d') for d in future_dates]
        })
        
        # 合并天气预报
        future_df = future_df.merge(weather_forecast, on='date', how='left')
        
        # 合并历史和未来
        df = pd.concat([historical_data, future_df], ignore_index=True)
        
        # 时间特征
        df['date_dt'] = pd.to_datetime(df['date'])
        df['month'] = df['date_dt'].dt.month
        df['weekday'] = df['date_dt'].dt.weekday
        
        # 周期编码
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['weekday_sin'] = np.sin(2 * np.pi * df['weekday'] / 7)
        df['weekday_cos'] = np.cos(2 * np.pi * df['weekday'] / 7)
        
        # 节假日
        df['is_holiday'] = df['date_dt'].apply(
            lambda x: 1 if chinese_calendar.is_holiday(x) else 0
        )
        
        # 滞后特征（初始化为历史平均值）
        if 'tourism_num_lag_7' not in df.columns:
            avg_visitor = historical_data['visitor_count'].mean()
            df['tourism_num_lag_7'] = avg_visitor
            
            # 填充已知的滞后值
            for i in range(7, len(df)):
                if i < len(historical_data):
                    df.loc[i, 'tourism_num_lag_7'] = df.loc[i-7, 'visitor_count']
        
        return df
    
    def _normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """归一化特征"""
        
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
            # 降级方案
            df['visitor_count_scaled'] = df['visitor_count'] / 50000.0
            df['tourism_num_lag_7_scaled'] = df['tourism_num_lag_7'] / 50000.0
            df['meteo_precip_sum_scaled'] = df['precipitation'] / 50.0
            df['temp_high_scaled'] = (df['temp_high'] + 10) / 50.0
            df['temp_low_scaled'] = (df['temp_low'] + 10) / 50.0
        
        return df
    
    def _inverse_transform_visitor_count(self, scaled_value: float) -> float:
        """反归一化客流量"""
        if self.scalers and 'visitor_count' in self.scalers:
            return self.scalers['visitor_count'].inverse_transform([[scaled_value]])[0][0]
        else:
            return scaled_value * 50000.0
