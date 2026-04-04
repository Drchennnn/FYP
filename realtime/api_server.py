"""
Flask API 接口模块

提供实时预测 API：
- GET /api/realtime/current - 获取最新数据（昨天的）
- GET /api/realtime/forecast - 获取未来预测（三个模型对比）
- GET /api/realtime/weather - 获取实时天气
- GET /api/realtime/accuracy - 获取预测准确度
- GET /api/realtime/history - 获取预测历史
- GET /api/realtime/status - 获取系统状态
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import sys
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from realtime.data_fetcher import RealtimeDataFetcher
from realtime.multi_model_engine import MultiModelPredictionEngine
from realtime.jiuzhaigou_crawler import JiuzhaigouCrawler
import pandas as pd
from datetime import datetime, timedelta

app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 初始化模块
data_fetcher = RealtimeDataFetcher(cache_ttl=3600)  # 1 小时缓存
crawler = JiuzhaigouCrawler()

# 加载多模型预测引擎
MODEL_CONFIGS = {
    'lstm': {
        'path': 'output/runs/lstm_8features_20260403_164549/runs/run_20260403_164549_lb30_ep120_lstm_8features/weights/lstm_jiuzhaigou_baseline.h5',
        'name': 'LSTM'
    },
    'gru': {
        'path': 'output/runs/gru_8features_20260403_163039/runs/run_20260403_163039_lb30_ep120_gru_8features/weights/gru_jiuzhaigou.h5',
        'name': 'GRU'
    },
    'seq2seq': {
        'path': 'output/runs/seq2seq_attention_8features_20260403_164843/runs/run_20260403_164843_lb30_ep120_seq2seq_attention_8features/weights/seq2seq_jiuzhaigou.keras',
        'name': 'Seq2Seq+Attention'
    }
}
SCALER_PATH = 'models/scalers/feature_scalers.pkl'
prediction_engine = None

try:
    prediction_engine = MultiModelPredictionEngine(MODEL_CONFIGS, SCALER_PATH, lookback=30)
    print('✅ Multi-model prediction engine loaded successfully')
except Exception as e:
    print(f'⚠️  Warning: Failed to load prediction engine: {e}')


@app.route('/api/realtime/current', methods=['GET'])
def get_current_data():
    """
    获取最新数据（昨天的数据）
    
    Returns:
        {
            "success": true,
            "data": {
                "date": "2026-04-02",
                "visitor_count": 15000,
                "temperature": 18.5,
                "precipitation": 0.0,
                "temp_high": 22.0,
                "temp_low": 12.0,
                "timestamp": "2026-04-03 17:00:00",
                "note": "Data is from yesterday (official website updates daily)"
            }
        }
    """
    try:
        # 获取客流数据
        visitor_data = data_fetcher.get_current_visitor_count()
        
        # 获取天气数据
        weather_data = data_fetcher.get_current_weather()
        
        if not visitor_data or not weather_data:
            return jsonify({
                'success': False,
                'error': 'Failed to fetch realtime data'
            }), 500
        
        # 合并数据
        result = {
            'date': visitor_data['date'],
            'visitor_count': visitor_data['visitor_count'],
            'temperature': weather_data['temperature'],
            'precipitation': weather_data['precipitation'],
            'temp_high': weather_data['temp_high'],
            'temp_low': weather_data['temp_low'],
            'timestamp': visitor_data['timestamp'],
            'note': 'Data is from yesterday (official website updates daily)'
        }
        
        return jsonify({
            'success': True,
            'data': result
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/realtime/forecast', methods=['GET'])
def get_forecast():
    """
    获取未来预测（三个模型对比）
    
    Query params:
        days: 预测天数 (默认 7)
        
    Returns:
        {
            "success": true,
            "data": {
                "models": {
                    "lstm": [{"date": "2026-04-04", "predicted_visitor_count": 18000, ...}],
                    "gru": [...],
                    "seq2seq": [...]
                },
                "generated_at": "2026-04-03 17:30:00"
            }
        }
    """
    try:
        days = int(request.args.get('days', 7))
        
        if prediction_engine is None:
            return jsonify({
                'success': False,
                'error': 'Prediction engine not loaded'
            }), 503
        
        # 获取历史数据（最近 40 天）
        historical_data = crawler.get_historical_data(days=40)
        
        if len(historical_data) < 30:
            return jsonify({
                'success': False,
                'error': 'Not enough historical data (need at least 30 days)'
            }), 400
        
        historical_df = pd.DataFrame(historical_data)
        historical_df.columns = ['date', 'visitor_count']
        
        # 获取天气预报
        weather_forecast = data_fetcher.get_weather_forecast(days=days)
        
        if weather_forecast is None:
            return jsonify({
                'success': False,
                'error': 'Failed to fetch weather forecast'
            }), 503
        
        weather_df = pd.DataFrame(weather_forecast)
        
        # 使用所有模型预测
        all_predictions = prediction_engine.predict_future_all_models(
            historical_df,
            weather_df,
            forecast_days=days
        )
        
        # 保存预测到数据库（使用 GRU 作为主模型）
        prediction_date = datetime.now().strftime('%Y-%m-%d')
        if all_predictions.get('gru') is not None:
            for _, row in all_predictions['gru'].iterrows():
                crawler.save_prediction(
                    prediction_date=prediction_date,
                    target_date=row['date'],
                    predicted_value=int(row['predicted_visitor_count']),
                    model_version='GRU-v1'
                )
        
        # 转换为 JSON
        result = {}
        for model_key, predictions in all_predictions.items():
            if predictions is not None:
                result[model_key] = predictions.to_dict('records')
            else:
                result[model_key] = None
        
        return jsonify({
            'success': True,
            'data': {
                'models': result,
                'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/realtime/weather', methods=['GET'])
def get_weather():
    """
    获取实时天气和预报
    
    Query params:
        days: 预报天数 (默认 7)
    """
    try:
        days = int(request.args.get('days', 7))
        
        # 当前天气
        current_weather = data_fetcher.get_current_weather()
        
        # 天气预报
        weather_forecast = data_fetcher.get_weather_forecast(days=days)
        
        if not current_weather or weather_forecast is None:
            return jsonify({
                'success': False,
                'error': 'Failed to fetch weather data'
            }), 500
        
        return jsonify({
            'success': True,
            'data': {
                'current': current_weather,
                'forecast': weather_forecast.to_dict('records')
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/realtime/status', methods=['GET'])
def get_status():
    """获取系统状态"""
    models_loaded = {}
    if prediction_engine:
        for key, info in prediction_engine.models.items():
            models_loaded[key] = info['name']
    
    return jsonify({
        'success': True,
        'data': {
            'models_loaded': models_loaded,
            'scaler_loaded': prediction_engine.scalers is not None if prediction_engine else False,
            'cache_ttl': data_fetcher.cache_ttl,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    })


@app.route('/api/realtime/accuracy', methods=['GET'])
def get_accuracy():
    """获取预测准确度"""
    try:
        days = int(request.args.get('days', 30))
        
        accuracy = crawler.get_prediction_accuracy(days=days)
        
        return jsonify({
            'success': True,
            'data': {
                **accuracy,
                'period': f'last {days} days'
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/realtime/history', methods=['GET'])
def get_prediction_history():
    """获取预测历史记录"""
    try:
        days = int(request.args.get('days', 7))
        
        import sqlite3
        conn = sqlite3.connect(crawler.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT prediction_date, target_date, predicted_value, actual_value
            FROM prediction_history
            WHERE target_date >= date('now', '-' || ? || ' days')
            ORDER BY target_date DESC, prediction_date DESC
        ''', (days,))
        
        rows = cursor.fetchall()
        conn.close()
        
        predictions = []
        for row in rows:
            pred_date, target_date, pred_val, actual_val = row
            
            item = {
                'prediction_date': pred_date,
                'target_date': target_date,
                'predicted_value': pred_val,
                'actual_value': actual_val
            }
            
            if actual_val is not None:
                error = pred_val - actual_val
                error_pct = abs(error / actual_val * 100) if actual_val != 0 else 0
                item['error'] = error
                item['error_pct'] = round(error_pct, 2)
            else:
                item['error'] = None
                item['error_pct'] = None
            
            predictions.append(item)
        
        return jsonify({
            'success': True,
            'data': {
                'predictions': predictions
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


if __name__ == '__main__':
    print('='*80)
    print('Starting Jiuzhaigou Realtime Prediction API Server')
    print('='*80)
    print()
    app.run(host='127.0.0.1', port=5001, debug=True)
