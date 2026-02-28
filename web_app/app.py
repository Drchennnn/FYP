import os
import sys
import glob
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from flask import Flask, jsonify, request, render_template
from sklearn.preprocessing import MinMaxScaler

# --- Import Keras/TensorFlow (Compatibility Layer) ---
# 移除 TF_USE_LEGACY_KERAS 环境变量设置，因为我们使用的是 Keras 2.15.0，它应该原生工作
# 或者需要安装 tf_keras 包
# import os
# os.environ['TF_USE_LEGACY_KERAS'] = '1'

keras = None
tf = None

try:
    import tensorflow as tf
    # 尝试修复 trackable 缺失问题 (有时候是因为 lazy loading)
    import tensorflow.python.trackable 
    print(f"Successfully imported tensorflow version: {getattr(tf, '__version__', 'unknown')}")
except Exception as e:
    print(f"TensorFlow import failed: {e}")
    tf = None

# 如果 TF 失败，再尝试 Keras
if not tf:
    try:
        import keras
        print(f"Successfully imported keras version: {keras.__version__}")
    except Exception as e:
        print(f"Keras import failed: {e}")
        keras = None

if not keras and not tf:
    print("CRITICAL WARNING: Neither Keras nor TensorFlow could be imported. Prediction will fail.")

# --- Path Setup ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- App Imports ---
from web_app.config import Config
from web_app.models import db, TrafficRecord
from scripts.sync_to_cloud import sync_data

# --- Flask App Initialization ---
# Explicitly set template and static folders to avoid path issues
template_dir = os.path.join(current_dir, 'templates')
static_dir = os.path.join(current_dir, 'static')

app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)
app.config.from_object(Config)

# Initialize Database
db.init_app(app)

# --- Model Configuration ---
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_RUNS_DIR = os.path.join(base_dir, 'model', 'runs')
lstm_model = None

def get_latest_model_path():
    """Get the latest model file path (prefer .h5, then .keras)"""
    if not os.path.exists(MODEL_RUNS_DIR):
        return None
    
    run_dirs = glob.glob(os.path.join(MODEL_RUNS_DIR, 'run_*'))
    if not run_dirs:
        return None
    
    latest_run_dir = max(run_dirs, key=os.path.getmtime)
    
    # Priority 1: H5 format (Best compatibility)
    h5_path = os.path.join(latest_run_dir, 'lstm_jiuzhaigou.h5')
    if os.path.exists(h5_path):
        return h5_path
        
    # Priority 2: Keras format
    keras_path = os.path.join(latest_run_dir, 'lstm_jiuzhaigou.keras')
    return keras_path if os.path.exists(keras_path) else None

def load_model():
    """Load the LSTM model using available libraries"""
    global lstm_model
    path = get_latest_model_path()
    
    if not path:
        print("WARNING: No available model file found.")
        return

    print(f"Loading model from: {path}")
    try:
        # Strategy 1: TF Keras (Preferred when Keras is broken)
        if tf:
            try:
                print("Attempting load with tf.keras.models.load_model...")
                # 显式指定 compile=False
                lstm_model = tf.keras.models.load_model(path, compile=False)
                print("Model loaded successfully (tf.keras).")
                return
            except Exception as e:
                print(f"tf.keras load failed: {e}")

        # Strategy 2: Keras Standalone
        if keras:
            try:
                print(f"Attempting load with keras (v{keras.__version__})...")
                if hasattr(keras, 'saving') and hasattr(keras.saving, 'load_model'):
                    lstm_model = keras.saving.load_model(path, compile=False)
                elif hasattr(keras.models, 'load_model'):
                    lstm_model = keras.models.load_model(path, compile=False)
                else:
                    from keras.models import load_model as k_load_model
                    lstm_model = k_load_model(path, compile=False)
                
                if lstm_model:
                    print("Model loaded successfully (keras).")
                    return
            except Exception as e:
                print(f"Keras load failed: {e}")
        
        print("ERROR: All model loading methods failed.")
        
    except Exception as e:
        print(f"Unhandled exception during model loading: {e}")
        import traceback
        traceback.print_exc()

# --- Holiday Configuration ---
def load_holidays_config():
    try:
        config_path = os.path.join(current_dir, 'holidays.json')
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading holidays config: {e}")
    return []

HOLIDAYS_CONFIG = load_holidays_config()

def mark_core_holiday(date_val):
    """Check if a date is a holiday based on config"""
    date_str = date_val.strftime('%Y-%m-%d')
    for h in HOLIDAYS_CONFIG:
        if h['start'] <= date_str <= h['end']:
            return 1
    if date_val.weekday() >= 5:
        return 1
    return 0

# --- Routes ---

@app.route('/')
def index():
    print("Request received for root route /")
    return render_template('index.html')

@app.route('/api/data', methods=['GET'])
def get_data():
    """Get historical and prediction data"""
    try:
        records = TrafficRecord.query.order_by(TrafficRecord.record_date).all()
        
        # 找到真实数据的最后一天
        last_real_date = None
        for r in records:
            if r.actual_visitor is not None and r.actual_visitor > 0:
                last_real_date = r.record_date

        holiday_ranges = []
        for h in HOLIDAYS_CONFIG:
            holiday_ranges.append({
                "start": h['start'],
                "end": h['end'],
                "name": h['name'],
                "type": h.get('type', 'festival')
            })

        dates = []
        true_vals = []
        pred_vals = []

        for r in records:
            dates.append(r.record_date.strftime('%Y-%m-%d'))
            true_vals.append(r.actual_visitor)
            pred_vals.append(r.predicted_visitor)

        data = {
            "dates": dates,
            "true_vals": true_vals,
            "pred_vals": pred_vals,
            "holiday_ranges": holiday_ranges 
        }
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    """Rolling prediction endpoint"""
    if not lstm_model:
        return jsonify({"error": "Model not loaded"}), 500
        
    req_data = request.json
    target_date_str = req_data.get('future_date')
    days_to_predict = req_data.get('days')
    
    last_actual_record = TrafficRecord.query.filter(
        TrafficRecord.actual_visitor.isnot(None)
    ).order_by(TrafficRecord.record_date.desc()).first()
    
    if not last_actual_record:
        return jsonify({"error": "No history data available"}), 400
        
    # Reset future predictions logic
    try:
        TrafficRecord.query.filter(
            TrafficRecord.record_date > last_actual_record.record_date
        ).delete()
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        print(f"Warning: Failed to clear old predictions: {e}")

    start_date = last_actual_record.record_date + timedelta(days=1)
    
    if target_date_str:
        end_date = pd.to_datetime(target_date_str).date()
    elif days_to_predict:
        end_date = start_date + timedelta(days=int(days_to_predict) - 1)
    else:
        return jsonify({"error": "Missing future_date or days parameter"}), 400

    if end_date < start_date:
        return jsonify({"error": "Target date is in the past"}), 400

    # Prediction Logic
    try:
        # 1. Scaler
        all_actuals = db.session.query(TrafficRecord.actual_visitor).filter(
            TrafficRecord.actual_visitor.isnot(None)
        ).all()
        all_counts_arr = np.array([r[0] for r in all_actuals]).reshape(-1, 1)
        scaler = MinMaxScaler()
        scaler.fit(all_counts_arr)
        
        # 2. Initial Window (Lookback 30)
        look_back = 30
        history_records = TrafficRecord.query.filter(
            TrafficRecord.actual_visitor.isnot(None)
        ).order_by(TrafficRecord.record_date.desc()).limit(look_back).all()
        
        if len(history_records) < look_back:
            return jsonify({"error": "Not enough history"}), 400
            
        current_input_sequence = [scaler.transform([[r.actual_visitor]])[0][0] for r in reversed(history_records)]
        
        predictions_result = []
        current_date = last_actual_record.record_date + timedelta(days=1)
        
        while current_date <= end_date:
            model_input = []
            for i in range(look_back):
                val = current_input_sequence[i]
                feature_date = current_date - timedelta(days=(look_back - i))
                m_norm = (feature_date.month - 1) / 11.0
                d_norm = feature_date.weekday() / 6.0
                hol = mark_core_holiday(feature_date)
                model_input.append([val, m_norm, d_norm, hol])
            
            input_arr = np.array(model_input).reshape(1, look_back, 4)
            pred_scaled = lstm_model.predict(input_arr, verbose=0)[0][0]
            pred_val = int(round(scaler.inverse_transform([[pred_scaled]])[0][0]))
            
            if current_date >= start_date:
                predictions_result.append({
                    "date": current_date.strftime('%Y-%m-%d'),
                    "value": pred_val
                })
            
            # Save to DB
            existing = TrafficRecord.query.filter_by(record_date=current_date).first()
            if existing:
                existing.predicted_visitor = pred_val
                existing.is_forecast = True
            else:
                db.session.add(TrafficRecord(
                    record_date=current_date,
                    predicted_visitor=pred_val,
                    is_forecast=True
                ))
            
            current_input_sequence.pop(0)
            current_input_sequence.append(pred_scaled)
            current_date += timedelta(days=1)
            
        db.session.commit()
        
        return jsonify({
            "start_date": start_date.strftime('%Y-%m-%d'),
            "end_date": end_date.strftime('%Y-%m-%d'),
            "predictions": predictions_result
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# --- App Entry Point ---
if __name__ == '__main__':
    with app.app_context():
        load_model()
        db.create_all()
        
    print("Attempting to sync latest data...")
    try:
        sync_data()
    except Exception as e:
        print(f"Sync failed: {e}")

    # Use port 5000 as requested
    print("Starting Flask server on port 5000...")
    app.run(debug=True, port=5000)