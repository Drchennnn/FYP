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
            'display_name': f"Champion: {champ.get('model')}",
            'model_key': champ.get('model'),
            'run_dir': champ_run,
        }
    ]
    if runner and runner_run:
        models.append({
            'model_id': 'runner_up',
            'display_name': f"Runner-up: {runner.get('model')}",
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
      - h: int, [1, 14], latest window length
      - include_all: 1/0, if 1 returns champion + runner-up (if available)
    """

    h = int(request.args.get('h', 7))
    h = max(1, min(h, 14))
    include_all = str(request.args.get('include_all', '0')).lower() in ['1', 'true', 'yes']

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

    # Unified time axis (union by date)
    df_base = df_c.copy()
    df_base = df_base.rename(columns={'y_pred': 'champion_pred', 'y_true': 'actual'})

    if runner_available:
        df_rr = df_r.copy().rename(columns={'y_pred': 'runner_pred', 'y_true': 'actual_r'})
        df_base = pd.merge(df_base, df_rr[['date', 'runner_pred', 'actual_r']], on='date', how='outer')
        # Prefer champion actual; fall back to runner actual if champion missing
        df_base['actual'] = df_base['actual'].combine_first(df_base.get('actual_r'))
        if 'actual_r' in df_base.columns:
            df_base = df_base.drop(columns=['actual_r'])
    else:
        df_base['runner_pred'] = np.nan

    df_base = df_base.sort_values('date')

    dates_all = df_base['date'].tolist()
    warning = None
    dfw = _load_weather_by_date(dates_all)
    if dfw is None or dfw.empty:
        warning = 'Weather data not available; returning nulls for weather fields.'
        dfw = pd.DataFrame({
            'date': dates_all,
            'precip_mm': [np.nan] * len(dates_all),
            'temp_high_c': [np.nan] * len(dates_all),
            'temp_low_c': [np.nan] * len(dates_all),
            'weather_code_en': [None] * len(dates_all),
            'wind_level': [np.nan] * len(dates_all),
            'wind_dir_en': [None] * len(dates_all),
            'wind_max': [np.nan] * len(dates_all),
            'aqi_value': [np.nan] * len(dates_all),
            'aqi_level_en': [None] * len(dates_all),
        })

    df_merge = pd.merge(df_base, dfw, on='date', how='left')

    # Thresholds come from champion metrics by default
    threshold_crowd = float((champ_metrics.get('meta') or {}).get('peak_threshold', 18500.0))
    wh = (champ_metrics.get('weather_hazard') or {})
    wh_thr = (wh.get('thresholds') or {})
    precip_high = float(wh_thr.get('precip_high', 8.0))
    temp_high = float(wh_thr.get('temp_high', 22.6))
    temp_low = float(wh_thr.get('temp_low', -10.61))
    quantiles = (wh_thr.get('quantiles') or {})

    def _compute_risk(pred_col: str):
        y_pred = df_merge[pred_col].astype(float)
        crowd_alert = (y_pred >= threshold_crowd)
        weather_hazard = (
            (df_merge['precip_mm'] >= precip_high) |
            (df_merge['temp_high_c'] >= temp_high) |
            (df_merge['temp_low_c'] <= temp_low)
        )
        suitability = (crowd_alert | weather_hazard)

        risk_level = []
        drivers = []
        for ca, whz, pr, th, tl in zip(
            crowd_alert.tolist(),
            weather_hazard.tolist(),
            df_merge['precip_mm'].tolist(),
            df_merge['temp_high_c'].tolist(),
            df_merge['temp_low_c'].tolist()
        ):
            lv = 0
            d = []
            if bool(ca) and not (isinstance(ca, float) and np.isnan(ca)):
                lv += 2
                d.append('Crowd forecast exceeds threshold')
            if bool(whz) and not (isinstance(whz, float) and np.isnan(whz)):
                lv += 1
                if pr is not None and not (isinstance(pr, float) and np.isnan(pr)) and pr >= precip_high:
                    d.append('High precipitation')
                if th is not None and not (isinstance(th, float) and np.isnan(th)) and th >= temp_high:
                    d.append('High temperature')
                if tl is not None and not (isinstance(tl, float) and np.isnan(tl)) and tl <= temp_low:
                    d.append('Low temperature')
            risk_level.append(int(lv))
            drivers.append(d)

        p_warn = [0.85 if s else 0.15 for s in suitability.tolist()]

        return {
            'risk_level': risk_level,
            'drivers': drivers,
            'p_warn': p_warn,
        }

    risk_champ = _compute_risk('champion_pred')
    risk_runner = _compute_risk('runner_pred') if runner_available else None

    # Holiday intervals (for markArea)
    holiday_ranges = []
    for hh in HOLIDAYS_CONFIG:
        holiday_ranges.append({
            'start': hh['start'],
            'end': hh['end'],
            'name': hh.get('name', 'Holiday'),
            'type': hh.get('type', 'festival')
        })

    time_axis = [d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else str(d) for d in df_merge['date'].tolist()]
    forecast_start_idx = max(0, len(time_axis) - h)

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
                'model_name': champ_key,
                'run_dir': champ_run,
            },
            'runner_up': ({
                'model_name': run_key,
                'run_dir': run_run,
            } if runner_available else None)
        },
        'time_axis': time_axis,
        'forecast': {
            'h': h,
            'start_index': forecast_start_idx
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
