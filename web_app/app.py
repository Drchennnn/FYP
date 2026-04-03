import os
import sys
import glob
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from flask import Flask, jsonify, request, render_template, redirect, url_for
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
OFFLINE_BACKUPS_DIR = os.path.join(base_dir, 'output', 'backups')
lstm_model = None


def _get_latest_backup_dir():
    """Return absolute path to the latest backup directory under output/backups."""
    if not os.path.isdir(OFFLINE_BACKUPS_DIR):
        return None
    cands = glob.glob(os.path.join(OFFLINE_BACKUPS_DIR, 'backup_*'))
    cands = [p for p in cands if os.path.isdir(p)]
    if not cands:
        return None
    return max(cands, key=os.path.getmtime)


def _load_compare_metrics(backup_dir: str):
    """Load compare_metrics.csv from latest backup (if exists)."""
    if not backup_dir:
        return None
    compare_dirs = glob.glob(os.path.join(backup_dir, 'run_compare_*'))
    compare_dirs = [p for p in compare_dirs if os.path.isdir(p)]
    if not compare_dirs:
        return None
    latest_compare = max(compare_dirs, key=os.path.getmtime)
    csv_path = os.path.join(latest_compare, 'compare_metrics.csv')
    if not os.path.exists(csv_path):
        return None
    try:
        return pd.read_csv(csv_path)
    except Exception as e:
        print(f"Failed to read compare_metrics.csv: {e}")
        return None


def _pick_champion_and_runner_up(df_cmp: pd.DataFrame):
    """Pick champion and runner-up based on weighted suitability warning metrics."""
    if df_cmp is None or df_cmp.empty:
        return None, None

    df = df_cmp.copy()
    needed = [
        'model',
        'run_dir',
        'suitability_warning_recall_weighted',
        'suitability_warning_f1_weighted',
        'suitability_warning_brier_weighted',
        'suitability_warning_ece_weighted'
    ]
    for c in needed:
        if c not in df.columns:
            return None, None

    # Champion policy: Recall>=0.8; maximize weighted F1; tie-breakers: lower Brier then lower ECE.
    df = df.sort_values(
        by=[
            'suitability_warning_recall_weighted',
            'suitability_warning_f1_weighted',
            'suitability_warning_brier_weighted',
            'suitability_warning_ece_weighted',
        ],
        ascending=[False, False, True, True]
    )
    top = df.head(2).to_dict(orient='records')
    if not top:
        return None, None
    champ = top[0]
    runner = top[1] if len(top) > 1 else None
    return champ, runner


def _resolve_backup_run_dir(backup_dir: str, run_dir_in_report: str):
    """Resolve run_dir reference (like output\\runs\\run_xxx) to the backup copy folder."""
    if not backup_dir or not run_dir_in_report:
        return None
    base = os.path.basename(str(run_dir_in_report).rstrip('\\/'))
    cand = os.path.join(backup_dir, base)
    if os.path.isdir(cand):
        return cand
    matches = glob.glob(os.path.join(backup_dir, f"{base}*"))
    matches = [p for p in matches if os.path.isdir(p)]
    return matches[0] if matches else None


def _safe_read_json(path: str):
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Failed to read json {path}: {e}")
        return None


def _load_predictions(run_dir: str):
    """Load test predictions csv from a run_dir.

    Returns DataFrame with columns: date (python date), y_true (float or NaN), y_pred (float)
    """
    if not run_dir or not os.path.isdir(run_dir):
        return None
    candidates = [
        os.path.join(run_dir, 'seq2seq_test_predictions.csv'),
        os.path.join(run_dir, 'gru_test_predictions.csv'),
        os.path.join(run_dir, 'lstm_test_predictions.csv'),
    ]
    pred_path = None
    for p in candidates:
        if os.path.exists(p):
            pred_path = p
            break
    if not pred_path:
        globbed = glob.glob(os.path.join(run_dir, '*_test_predictions.csv'))
        pred_path = globbed[0] if globbed else None

    if not pred_path or not os.path.exists(pred_path):
        return None

    try:
        df = pd.read_csv(pred_path)
        if 'date' not in df.columns:
            return None
        df['date'] = pd.to_datetime(df['date']).dt.date

        if 'y_pred' not in df.columns:
            for alt in ['pred', 'yhat', 'y_pred_mean']:
                if alt in df.columns:
                    df['y_pred'] = df[alt]
                    break
        if 'y_true' not in df.columns:
            for alt in ['true', 'y', 'actual', 'y_true_mean']:
                if alt in df.columns:
                    df['y_true'] = df[alt]
                    break
        if 'y_true' not in df.columns:
            df['y_true'] = np.nan

        df_agg = df.groupby('date', as_index=False).agg({'y_true': 'mean', 'y_pred': 'mean'})
        df_agg = df_agg.sort_values('date')
        return df_agg
    except Exception as e:
        print(f"Failed to load predictions from {pred_path}: {e}")
        return None


def _load_weather_by_date(dates: list):
    """Join weather information from processed dataset by date."""
    if not dates:
        return None
    processed_path = os.path.join(base_dir, 'data', 'processed', 'jiuzhaigou_8features_latest.csv')
    if not os.path.exists(processed_path):
        return None
    try:
        dfw = pd.read_csv(processed_path)
        if 'date' not in dfw.columns:
            return None
        dfw['date'] = pd.to_datetime(dfw['date']).dt.date
        dfw = dfw[dfw['date'].isin(dates)].copy()
        if dfw.empty:
            return None

        out = pd.DataFrame({'date': dfw['date']})

        # Core numeric fields
        out['precip_mm'] = dfw['meteo_precip_sum'] if 'meteo_precip_sum' in dfw.columns else np.nan
        if 'temp_high_c' in dfw.columns:
            out['temp_high_c'] = dfw['temp_high_c']
        elif 'meteo_temp_max' in dfw.columns:
            out['temp_high_c'] = dfw['meteo_temp_max']
        else:
            out['temp_high_c'] = np.nan

        if 'temp_low_c' in dfw.columns:
            out['temp_low_c'] = dfw['temp_low_c']
        elif 'meteo_temp_min' in dfw.columns:
            out['temp_low_c'] = dfw['meteo_temp_min']
        else:
            out['temp_low_c'] = np.nan

        # Extra fields for Weather Card (string fields kept as first() per date)
        if 'weather_code_en' in dfw.columns:
            out['weather_code_en'] = dfw['weather_code_en']
        if 'wind_level' in dfw.columns:
            out['wind_level'] = dfw['wind_level']
        if 'wind_dir_en' in dfw.columns:
            out['wind_dir_en'] = dfw['wind_dir_en']
        if 'aqi_value' in dfw.columns:
            out['aqi_value'] = dfw['aqi_value']
        if 'aqi_level_en' in dfw.columns:
            out['aqi_level_en'] = dfw['aqi_level_en']
        if 'meteo_wind_max' in dfw.columns:
            out['wind_max'] = dfw['meteo_wind_max']

        # Aggregate
        numeric_cols = [c for c in out.columns if c not in ['date', 'weather_code_en', 'wind_dir_en', 'aqi_level_en']]
        agg = {c: 'mean' for c in numeric_cols if c != 'date'}
        for c in ['weather_code_en', 'wind_dir_en', 'aqi_level_en']:
            if c in out.columns:
                agg[c] = 'first'

        out = out.groupby('date', as_index=False).agg(agg)
        out = out.sort_values('date')
        return out
    except Exception as e:
        print(f"Failed to load weather from processed data: {e}")
        return None

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


def _holiday_i18n_name(name_zh: str):
    """Return (zh, en) for a known holiday name.

    holidays.json may only include Chinese names. We keep a lightweight mapping
    here to support the dashboard language toggle without any external API.
    """
    if not name_zh:
        return None, None
    m = {
        '元旦': "New Year's Day",
        '春节': 'Spring Festival',
        '元宵节': 'Lantern Festival',
        '清明节': 'Qingming Festival',
        '劳动节': 'Labour Day',
        '端午节': 'Dragon Boat Festival',
        '中秋节': 'Mid-Autumn Festival',
        '国庆节': 'National Day Holiday',
        '暑假': 'Summer Vacation',
        '寒假': 'Winter Vacation',
    }
    return name_zh, m.get(name_zh, name_zh)


def _load_master_history_from_processed():
    """Load full historical timeline (2024-2026) from processed dataset.

    Returns DataFrame with columns:
      - date (python date)
      - actual (float)
      - precip_mm, temp_high_c, temp_low_c, weather_code_en, wind_level,
        wind_dir_en, wind_max, aqi_value, aqi_level_en
    """
    processed_path = os.path.join(base_dir, 'data', 'processed', 'jiuzhaigou_8features_latest.csv')
    if not os.path.exists(processed_path):
        return None


def _pretty_model_name(raw_key: str):
    """Convert internal model key to a professional UI label."""
    k = (raw_key or '').lower()
    if 'seq2seq' in k and ('att' in k or 'attention' in k):
        return 'Seq2Seq+Attention (8 features)'
    if 'seq2seq' in k:
        return 'Seq2Seq (8 features)'
    if 'gru' in k:
        return 'GRU (8 features)'
    if 'lstm' in k:
        return 'LSTM (8 features)'
    return raw_key or 'Model'
    try:
        df = pd.read_csv(processed_path)
        if 'date' not in df.columns:
            return None
        df['date'] = pd.to_datetime(df['date']).dt.date

        # Actual visitors
        if 'tourism_num' in df.columns:
            df['actual'] = pd.to_numeric(df['tourism_num'], errors='coerce')
        elif 'actual_visitor' in df.columns:
            df['actual'] = pd.to_numeric(df['actual_visitor'], errors='coerce')
        else:
            df['actual'] = np.nan

        # Weather fields
        out = pd.DataFrame({
            'date': df['date'],
            'actual': df['actual'],
            'precip_mm': pd.to_numeric(df['meteo_precip_sum'], errors='coerce') if 'meteo_precip_sum' in df.columns else np.nan,
            'temp_high_c': pd.to_numeric(df['temp_high_c'], errors='coerce') if 'temp_high_c' in df.columns else (pd.to_numeric(df['meteo_temp_max'], errors='coerce') if 'meteo_temp_max' in df.columns else np.nan),
            'temp_low_c': pd.to_numeric(df['temp_low_c'], errors='coerce') if 'temp_low_c' in df.columns else (pd.to_numeric(df['meteo_temp_min'], errors='coerce') if 'meteo_temp_min' in df.columns else np.nan),
            'weather_code_en': df['weather_code_en'] if 'weather_code_en' in df.columns else None,
            'wind_level': pd.to_numeric(df['wind_level'], errors='coerce') if 'wind_level' in df.columns else np.nan,
            'wind_dir_en': df['wind_dir_en'] if 'wind_dir_en' in df.columns else None,
            'wind_max': pd.to_numeric(df['meteo_wind_max'], errors='coerce') if 'meteo_wind_max' in df.columns else np.nan,
            'aqi_value': pd.to_numeric(df['aqi_value'], errors='coerce') if 'aqi_value' in df.columns else np.nan,
            'aqi_level_en': df['aqi_level_en'] if 'aqi_level_en' in df.columns else None,
        })

        out = out.sort_values('date')
        out = out.groupby('date', as_index=False).agg({
            'actual': 'mean',
            'precip_mm': 'mean',
            'temp_high_c': 'mean',
            'temp_low_c': 'mean',
            'weather_code_en': 'first',
            'wind_level': 'mean',
            'wind_dir_en': 'first',
            'wind_max': 'mean',
            'aqi_value': 'mean',
            'aqi_level_en': 'first',
        })
        return out
    except Exception as e:
        print(f"Failed to load master history from processed data: {e}")
        return None

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
    return redirect(url_for('dashboard'))


@app.route('/legacy')
def legacy_index():
    """Legacy UI kept for rollback."""
    return render_template('index.html')


@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')


@app.route('/compare')
def compare():
    return render_template('compare.html')


@app.route('/definitions')
def definitions():
    return render_template('definitions.html')


@app.route('/explain')
def explain():
    return render_template('explain.html')


@app.route('/api/models', methods=['GET'])
def api_models():
    """Offline artifact mode: return champion + runner-up from latest backup."""
    backup_dir = _get_latest_backup_dir()
    df_cmp = _load_compare_metrics(backup_dir)
    champ, runner = _pick_champion_and_runner_up(df_cmp)

    if not champ:
        return jsonify({
            'backup_dir': backup_dir,
            'models': [],
            'warning': 'No compare_metrics.csv found under latest backup.'
        })

    champ_run = _resolve_backup_run_dir(backup_dir, champ.get('run_dir'))
    runner_run = _resolve_backup_run_dir(backup_dir, runner.get('run_dir')) if runner else None

    models = [
        {
            'model_id': 'champion',
            'display_name': _pretty_model_name(champ.get('model')),
            'model_key': champ.get('model'),
            'run_dir': champ_run,
        }
    ]
    if runner and runner_run:
        models.append({
            'model_id': 'runner_up',
            'display_name': _pretty_model_name(runner.get('model')),
            'model_key': runner.get('model'),
            'run_dir': runner_run,
        })

    return jsonify({
        'backup_dir': backup_dir,
        'models': models
    })


@app.route('/api/metrics', methods=['GET'])
def api_metrics():
    """Return metrics.json for a given model_id (champion / runner_up)."""
    model_id = request.args.get('model_id', 'champion')
    models_resp = api_models().get_json() or {}
    models = {m['model_id']: m for m in (models_resp.get('models') or [])}
    if model_id not in models:
        return jsonify({'error': f'Unknown model_id: {model_id}'}), 400
    run_dir = models[model_id].get('run_dir')
    metrics_path = os.path.join(run_dir, 'metrics.json') if run_dir else None
    metrics = _safe_read_json(metrics_path)
    if not metrics:
        return jsonify({'error': 'metrics.json not found', 'run_dir': run_dir}), 404
    return jsonify({
        'model_id': model_id,
        'model_name': models[model_id].get('model_key'),
        'run_dir': run_dir,
        'metrics': metrics
    })


@app.route('/api/forecast', methods=['GET'])
def api_forecast():
    """Offline artifact forecast API.

    Dashboard uses this endpoint for the full time span, and only visualizes the
    latest *h* days as the forecast segment.

    Query:
      - h: int, [1, 14], window length
      - include_all: 1/0, if 1 returns champion + runner-up (if available)
      - mode: offline|online

    Modes:
      - offline (default): uses offline artifacts; "latest forecast" is the
        latest predicted window in artifacts (backtest-style) anchored to last
        predicted date.
      - online: generate a true future forecast (h days) via the loaded LSTM
        model and append to the end of the processed history timeline.
    """

    h = int(request.args.get('h', 7))
    h = max(1, min(h, 14))
    include_all = str(request.args.get('include_all', '0')).lower() in ['1', 'true', 'yes']
    mode = str(request.args.get('mode', 'offline')).strip().lower()
    if mode not in ['offline', 'online']:
        mode = 'offline'

    models_resp = api_models().get_json() or {}
    models = {m['model_id']: m for m in (models_resp.get('models') or [])}
    if 'champion' not in models:
        return jsonify({'error': 'Champion model not found in offline artifacts'}), 404

    def _load_one(mid: str):
        run_dir = models.get(mid, {}).get('run_dir')
        model_key = models.get(mid, {}).get('model_key')
        metrics_path = os.path.join(run_dir, 'metrics.json') if run_dir else None
        metrics = _safe_read_json(metrics_path) or {}
        df = _load_predictions(run_dir)
        return run_dir, model_key, metrics, df

    champ_run, champ_key, champ_metrics, df_c = _load_one('champion')
    if df_c is None or df_c.empty:
        return jsonify({'error': 'No prediction artifacts found', 'run_dir': champ_run}), 404

    runner_available = include_all and ('runner_up' in models)
    run_run, run_key, run_metrics, df_r = (None, None, {}, None)
    if runner_available:
        run_run, run_key, run_metrics, df_r = _load_one('runner_up')
        if df_r is None or df_r.empty:
            runner_available = False

    # Master time axis: prefer full processed history (expected 2024-2026)
    warning = None
    df_master = _load_master_history_from_processed()
    if df_master is None or df_master.empty:
        warning = 'Processed history not available; falling back to artifact date axis.'
        df_master = df_c[['date']].copy()
        df_master['actual'] = np.nan
        df_master['precip_mm'] = np.nan
        df_master['temp_high_c'] = np.nan
        df_master['temp_low_c'] = np.nan
        df_master['weather_code_en'] = None
        df_master['wind_level'] = np.nan
        df_master['wind_dir_en'] = None
        df_master['wind_max'] = np.nan
        df_master['aqi_value'] = np.nan
        df_master['aqi_level_en'] = None

    # Merge predictions into the master axis (null where absent)
    df_base = df_master[['date', 'actual', 'precip_mm', 'temp_high_c', 'temp_low_c', 'weather_code_en', 'wind_level', 'wind_dir_en', 'wind_max', 'aqi_value', 'aqi_level_en']].copy()
    df_base = pd.merge(df_base, df_c[['date', 'y_pred']], on='date', how='left')
    df_base = df_base.rename(columns={'y_pred': 'champion_pred'})

    if runner_available:
        df_base = pd.merge(df_base, df_r[['date', 'y_pred']], on='date', how='left', suffixes=('', '_runner'))
        if 'y_pred_runner' in df_base.columns:
            df_base = df_base.rename(columns={'y_pred_runner': 'runner_pred'})
        else:
            df_base['runner_pred'] = np.nan
    else:
        df_base['runner_pred'] = np.nan

    df_merge = df_base.sort_values('date')

    def _online_future_forecast_from_lstm(df_hist: pd.DataFrame, horizon: int):
        """Generate true future forecast for `horizon` days using loaded LSTM.

        Uses only the historical actual series for scaling and a minimal
        feature set (value_norm, month_norm, weekday_norm, holiday_flag),
        matching the existing /api/predict implementation.

        Returns DataFrame columns: date, champion_pred
        """
        if lstm_model is None:
            raise RuntimeError('Online forecast requested but LSTM model is not loaded.')
        if df_hist is None or df_hist.empty:
            raise RuntimeError('History not available for online forecast.')

        s = pd.to_numeric(df_hist['actual'], errors='coerce')
        s = s.dropna()
        if s.shape[0] < 30:
            raise RuntimeError('Not enough history (need >= 30 days) for online forecast.')

        look_back = 30
        scaler = MinMaxScaler()
        scaler.fit(s.values.reshape(-1, 1))

        last_date = df_hist['date'].max()
        hist_tail = s.values[-look_back:].reshape(-1, 1)
        current_seq = scaler.transform(hist_tail).flatten().tolist()

        out_rows = []
        for step in range(horizon):
            pred_date = last_date + timedelta(days=step + 1)
            model_input = []
            for i in range(look_back):
                feature_date = pred_date - timedelta(days=(look_back - i))
                m_norm = (feature_date.month - 1) / 11.0
                d_norm = feature_date.weekday() / 6.0
                hol = float(mark_core_holiday(feature_date))
                model_input.append([current_seq[i], m_norm, d_norm, hol])

            x_in = np.array(model_input, dtype=float).reshape(1, look_back, 4)
            y_norm = float(lstm_model.predict(x_in, verbose=0)[0][0])
            y_val = float(scaler.inverse_transform(np.array([[y_norm]], dtype=float))[0][0])
            out_rows.append({'date': pred_date, 'champion_pred': y_val})

            # roll window
            current_seq = current_seq[1:] + [y_norm]

        return pd.DataFrame(out_rows)

    # Online mode: append true future forecast to the end of timeline.
    online_used = False
    if mode == 'online':
        try:
            df_future = _online_future_forecast_from_lstm(df_master[['date', 'actual']].copy(), h)
            if df_future is not None and not df_future.empty:
                online_used = True
                df_future['runner_pred'] = np.nan
                # Add empty weather fields for future dates
                for c in ['actual', 'precip_mm', 'temp_high_c', 'temp_low_c', 'weather_code_en', 'wind_level', 'wind_dir_en', 'wind_max', 'aqi_value', 'aqi_level_en']:
                    if c not in df_future.columns:
                        df_future[c] = np.nan if c not in ['weather_code_en', 'wind_dir_en', 'aqi_level_en'] else None
                df_future['actual'] = np.nan

                # Keep the same column set and append
                df_merge = pd.concat([
                    df_merge,
                    df_future[['date', 'actual', 'precip_mm', 'temp_high_c', 'temp_low_c', 'weather_code_en', 'wind_level', 'wind_dir_en', 'wind_max', 'aqi_value', 'aqi_level_en', 'champion_pred', 'runner_pred']]
                ], ignore_index=True)
                df_merge = df_merge.sort_values('date')
        except Exception as e:
            warning = (warning + ' | ' if warning else '') + f'Online forecast failed; falling back to offline artifacts. ({e})'
            online_used = False

    # Thresholds come from champion metrics by default
    threshold_crowd = float((champ_metrics.get('meta') or {}).get('peak_threshold', 18500.0))
    wh = (champ_metrics.get('weather_hazard') or {})
    wh_thr = (wh.get('thresholds') or {})
    precip_high = float(wh_thr.get('precip_high', 8.0))
    temp_high = float(wh_thr.get('temp_high', 22.6))
    temp_low = float(wh_thr.get('temp_low', -10.61))
    quantiles = (wh_thr.get('quantiles') or {})

    def _compute_risk(pred_col: str):
        """Compute per-day risk fields required by dashboard.

        Returns lists aligned to df_merge rows:
          - crowd_alert: bool
          - weather_hazard: bool
          - suitability_warning_bin: int(0/1)
          - risk_level: int (0..3)
          - p_warn: float (0..1)
          - drivers: list[str]
          - risk_score: float (0..100)
        """

        y_pred = df_merge[pred_col].astype(float)
        crowd_alert = (y_pred >= threshold_crowd)
        weather_hazard = (
            (df_merge['precip_mm'] >= precip_high) |
            (df_merge['temp_high_c'] >= temp_high) |
            (df_merge['temp_low_c'] <= temp_low)
        )
        suitability = (crowd_alert | weather_hazard)

        crowd_alert_list = [bool(x) if not (x is None or (isinstance(x, float) and np.isnan(x))) else False for x in crowd_alert.tolist()]
        weather_hazard_list = [bool(x) if not (x is None or (isinstance(x, float) and np.isnan(x))) else False for x in weather_hazard.tolist()]
        suitability_bin = [1 if bool(x) else 0 for x in suitability.tolist()]

        risk_level = []
        drivers = []
        # Drivers are returned as stable codes; UI translates by language.
        for ca, whz, pr, th, tl in zip(
            crowd_alert_list,
            weather_hazard_list,
            df_merge['precip_mm'].tolist(),
            df_merge['temp_high_c'].tolist(),
            df_merge['temp_low_c'].tolist()
        ):
            lv = 0
            d = []
            if ca:
                lv += 2
                d.append('crowd_over_threshold')
            if whz:
                lv += 1
                if pr is not None and not (isinstance(pr, float) and np.isnan(pr)) and pr >= precip_high:
                    d.append('precip_high')
                if th is not None and not (isinstance(th, float) and np.isnan(th)) and th >= temp_high:
                    d.append('temp_high')
                if tl is not None and not (isinstance(tl, float) and np.isnan(tl)) and tl <= temp_low:
                    d.append('temp_low')
            risk_level.append(int(lv))
            drivers.append(d)

        # A lightweight calibrated probability proxy for UI (offline artifact mode)
        p_warn = [0.85 if s else 0.15 for s in suitability.tolist()]

        # Risk score (0..100) used by Thermometer UI
        risk_score = []
        for lv, pw in zip(risk_level, p_warn):
            v = max(0.0, min(1.0, (lv / 3.0) * 0.65 + float(pw) * 0.35))
            risk_score.append(round(v * 100.0, 1))

        return {
            'crowd_alert': crowd_alert_list,
            'weather_hazard': weather_hazard_list,
            'suitability_warning_bin': suitability_bin,
            'risk_level': risk_level,
            'p_warn': p_warn,
            'drivers': drivers,
            'risk_score': risk_score,
        }

    risk_champ = _compute_risk('champion_pred')
    risk_runner = _compute_risk('runner_pred') if runner_available else None

    # Holiday intervals (for markArea), with CN/EN names
    holiday_ranges = []
    for hh in HOLIDAYS_CONFIG:
        name_zh, name_en = _holiday_i18n_name(hh.get('name'))
        holiday_ranges.append({
            'start': hh['start'],
            'end': hh['end'],
            'name_zh': name_zh or hh.get('name', 'Holiday'),
            'name_en': name_en or 'Holiday',
            'type': hh.get('type', 'festival')
        })

    time_axis = [d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else str(d) for d in df_merge['date'].tolist()]

    if online_used:
        # True future window: last h days are the forecast segment.
        forecast_end_idx = len(time_axis) - 1
        forecast_start_idx = max(0, forecast_end_idx - h + 1)
        forecast_mode = 'online_future'
    else:
        # Offline artifacts: treat latest predicted window as the forecast segment.
        pred_series = df_merge['champion_pred'].astype(float)
        has_pred = ~(pred_series.isna())
        if has_pred.any():
            forecast_end_idx = int(np.where(has_pred.values)[0].max())
        else:
            forecast_end_idx = len(time_axis) - 1
        forecast_start_idx = max(0, forecast_end_idx - h + 1)
        forecast_mode = 'offline_latest_window_backtest'

    def _to_num_list(col):
        out = []
        for v in df_merge[col].tolist():
            if v is None or (isinstance(v, float) and np.isnan(v)):
                out.append(None)
            else:
                try:
                    out.append(float(v))
                except Exception:
                    out.append(None)
        return out

    return jsonify({
        'meta': {
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'champion': {
                'model_name': _pretty_model_name(champ_key),
            },
            'runner_up': ({
                'model_name': _pretty_model_name(run_key),
            } if runner_available else None)
            ,
            'forecast_mode': forecast_mode
        },
        'time_axis': time_axis,
        'forecast': {
            'h': h,
            'start_index': forecast_start_idx,
            'end_index': forecast_end_idx
        },
        'series': {
            'actual': _to_num_list('actual'),
            'champion_pred': _to_num_list('champion_pred'),
            'runner_pred': _to_num_list('runner_pred'),
        },
        'thresholds': {
            'crowd': threshold_crowd,
            'weather': {
                'precip_high': precip_high,
                'temp_high': temp_high,
                'temp_low': temp_low,
            },
            'weather_quantiles': {
                'precip_high': quantiles.get('precip_high'),
                'temp_high': quantiles.get('temp_high'),
                'temp_low': quantiles.get('temp_low'),
            }
        },
        'weather': {
            'precip_mm': _to_num_list('precip_mm'),
            'temp_high_c': _to_num_list('temp_high_c'),
            'temp_low_c': _to_num_list('temp_low_c'),
            'weather_code_en': [None if v is None or (isinstance(v, float) and np.isnan(v)) else str(v) for v in df_merge.get('weather_code_en', pd.Series([None] * len(df_merge))).tolist()],
            'wind_level': _to_num_list('wind_level') if 'wind_level' in df_merge.columns else [None] * len(df_merge),
            'wind_dir_en': [None if v is None or (isinstance(v, float) and np.isnan(v)) else str(v) for v in df_merge.get('wind_dir_en', pd.Series([None] * len(df_merge))).tolist()],
            'wind_max': _to_num_list('wind_max') if 'wind_max' in df_merge.columns else [None] * len(df_merge),
            'aqi_value': _to_num_list('aqi_value') if 'aqi_value' in df_merge.columns else [None] * len(df_merge),
            'aqi_level_en': [None if v is None or (isinstance(v, float) and np.isnan(v)) else str(v) for v in df_merge.get('aqi_level_en', pd.Series([None] * len(df_merge))).tolist()],
        },
        'holidays': holiday_ranges,
        'risk': {
            'champion': risk_champ,
            'runner_up': risk_runner
        },
        'warning': warning
    })

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
