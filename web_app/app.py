import os
import sys
import glob
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone, date as _date_type
from zoneinfo import ZoneInfo
from flask import Flask, jsonify, request, render_template, redirect, url_for
from sklearn.preprocessing import MinMaxScaler

_TZ_CN = ZoneInfo('Asia/Shanghai')

def _today_cn() -> _date_type:
    """返回中国标准时间（CST, UTC+8）的今日日期，避免服务器时区差异。"""
    return datetime.now(_TZ_CN).date()

# --- Path Setup (must come before project-relative imports) ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
# 确保 project_root 在 sys.path 最前面（优先于 '' 空字符串和其他路径）
# 避免 web_app/models.py 遮蔽 project_root/models/ 包
if sys.path and sys.path[0] != project_root:
    sys.path.insert(0, project_root)

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

# --- Seq2Seq Custom Objects ---
_seq2seq_custom_objects = {}
try:
    from models.lstm.train_seq2seq_attention_8features import (
        AttentionLayer,
        Seq2SeqWithAttention,
        create_custom_asymmetric_loss,
    )
    _loss_fn = create_custom_asymmetric_loss()
    _seq2seq_custom_objects = {
        'AttentionLayer': AttentionLayer,
        'Seq2SeqWithAttention': Seq2SeqWithAttention,
        'custom_asymmetric_loss': _loss_fn,
        'custom_loss': _loss_fn,
        'CustomAsymmetricLoss': _loss_fn,
    }
    print("Seq2Seq custom objects registered successfully.")
except Exception as _e:
    print(f"WARNING: Could not import Seq2Seq custom objects: {_e}")

# --- App Imports ---
from web_app.config import Config
from web_app.models import db, TrafficRecord
from scripts.sync_to_cloud import sync_data

# --- Optional: chinese_calendar for holiday detection ---
try:
    import chinese_calendar as _cncal
    def _is_holiday(d) -> bool:
        try:
            return bool(_cncal.is_holiday(d))
        except Exception:
            return False
except ImportError:
    def _is_holiday(d) -> bool:
        return d.weekday() >= 5

# --- Flask App Initialization ---
# Explicitly set template and static folders to avoid path issues
template_dir = os.path.join(current_dir, 'templates')
static_dir = os.path.join(current_dir, 'static')

app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)
app.config.from_object(Config)

@app.after_request
def _no_cache_static(response):
    """开发模式：禁用 JS/CSS 静态文件缓存，确保每次都拿最新版本。"""
    if request.path.startswith('/static/'):
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        response.headers['Pragma'] = 'no-cache'
    return response

# Initialize Database
db.init_app(app)

# --- Model Configuration ---
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OFFLINE_BACKUPS_DIR = os.path.join(base_dir, 'output', 'backups')
OUTPUT_RUNS_DIR = os.path.join(base_dir, 'output', 'runs')


def _get_latest_backup_dir():
    """Return the best available artifact directory.

    Priority:
      1. output/backups/backup_* (legacy, kept for compatibility)
      2. output/runs/ itself (new default — models live here directly)
    """
    if os.path.isdir(OFFLINE_BACKUPS_DIR):
        cands = glob.glob(os.path.join(OFFLINE_BACKUPS_DIR, 'backup_*'))
        cands = [p for p in cands if os.path.isdir(p)]
        if cands:
            return max(cands, key=os.path.getmtime)
    # Fall back to output/runs/ directly
    if os.path.isdir(OUTPUT_RUNS_DIR):
        return OUTPUT_RUNS_DIR
    return None


def _load_compare_metrics(backup_dir: str):
    """항상 최신 metrics.json에서 직접 합성 (오래된 compare_metrics.csv 무시)."""
    return _synthesise_compare_metrics()


def _synthesise_compare_metrics():
    """Build a compare_metrics DataFrame on-the-fly from the latest run of each model type.

    Scans output/runs/ for the most recent run per model (gru_8features, lstm_8features,
    seq2seq_attention_8features) and reads their metrics.json.
    Returns a DataFrame compatible with _pick_champion_and_runner_up, or None.
    """
    if not os.path.isdir(OUTPUT_RUNS_DIR):
        return None

    model_patterns = {
        'gru_8features':         'gru_8features_*',
        'transformer_8features': 'transformer_8features_*',
        'xgboost_8features':     'xgboost_8features_*',
    }

    rows = []
    for model_key, pattern in model_patterns.items():
        top_dirs = sorted(
            glob.glob(os.path.join(OUTPUT_RUNS_DIR, pattern)),
            key=os.path.getmtime, reverse=True
        )
        for top_dir in top_dirs:
            # Each top_dir may contain a runs/ subdirectory with the actual run
            run_subdirs = glob.glob(os.path.join(top_dir, 'runs', 'run_*'))
            if not run_subdirs:
                run_subdirs = [top_dir]
            run_dir = max(run_subdirs, key=os.path.getmtime)
            metrics_path = os.path.join(run_dir, 'metrics.json')
            if not os.path.exists(metrics_path):
                continue
            try:
                with open(metrics_path, 'r', encoding='utf-8') as f:
                    m = json.load(f)
            except Exception:
                continue

            sw = m.get('suitability_warning') or {}
            sw_w = m.get('suitability_warning_weighted') or sw  # single-step models have no weighted

            rows.append({
                'model': model_key,
                'run_dir': run_dir,
                'suitability_warning_recall_weighted': sw_w.get('recall_weighted', sw.get('recall', 0.0)),
                'suitability_warning_f1_weighted': sw_w.get('f1_weighted', sw.get('f1', 0.0)),
                'suitability_warning_brier_weighted': sw_w.get('brier_weighted', sw.get('brier', 1.0)),
                'suitability_warning_ece_weighted': sw_w.get('ece_weighted', sw.get('ece', 1.0)),
                'mae': (m.get('regression') or {}).get('mae', float('inf')),
                'smape': (m.get('regression') or {}).get('smape', float('inf')),
            })
            break  # only latest run per model type

    if not rows:
        return None
    return pd.DataFrame(rows)


def _resolve_backup_run_dir(backup_dir: str, run_dir_in_report):
    """Resolve a run_dir reference to an absolute path.

    Handles three cases:
      1. run_dir_in_report is already an absolute path that exists → return as-is
      2. Basename lookup inside backup_dir
      3. Basename lookup inside output/runs/ (new default)
    """
    if not run_dir_in_report:
        return None
    rdir = str(run_dir_in_report).rstrip('\\/')

    # Case 1: already absolute and exists
    if os.path.isdir(rdir):
        return rdir

    base = os.path.basename(rdir)

    # Case 2: inside backup_dir
    if backup_dir and os.path.isdir(backup_dir):
        cand = os.path.join(backup_dir, base)
        if os.path.isdir(cand):
            return cand
        matches = [p for p in glob.glob(os.path.join(backup_dir, f'{base}*')) if os.path.isdir(p)]
        if matches:
            return matches[0]

    # Case 3: inside output/runs/ (direct or nested)
    if os.path.isdir(OUTPUT_RUNS_DIR):
        cand = os.path.join(OUTPUT_RUNS_DIR, base)
        if os.path.isdir(cand):
            return cand
        # nested: output/runs/<model_timestamp>/runs/<run_name>
        for nested in glob.glob(os.path.join(OUTPUT_RUNS_DIR, '*', 'runs', base)):
            if os.path.isdir(nested):
                return nested

    return None


def _pick_champion_and_runner_up(df_cmp: pd.DataFrame):
    """Pick champion, runner-up, and third based on weighted suitability warning metrics."""
    if df_cmp is None or df_cmp.empty:
        return None, None, None

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
            return None, None, None

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
    top = df.head(3).to_dict(orient='records')
    if not top:
        return None, None, None
    champ = top[0]
    runner = top[1] if len(top) > 1 else None
    third = top[2] if len(top) > 2 else None
    return champ, runner, third


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

        # Seq2Seq CSVs store 7 rows per date (one per decoder step).
        # De-duplicate: prefer rows with real y_true; among ties take first occurrence.
        if df.duplicated(subset='date').any():
            # There are duplicate dates → likely Seq2Seq multi-step output.
            # Split: rows with y_true (test-set) and rows without (backfill future).
            has_true = df['y_true'].notna()
            df_real = df[has_true].drop_duplicates(subset='date', keep='first')
            df_future = df[~has_true].drop_duplicates(subset='date', keep='first')
            # Exclude future dates already covered by real rows
            df_future = df_future[~df_future['date'].isin(set(df_real['date']))]
            df = pd.concat([df_real, df_future], ignore_index=True)
        # For non-duplicated CSVs (GRU/LSTM single-step + backfill): keep all rows,
        # including y_true=NaN backfill rows (they carry valid y_pred values).
        df_agg = df[['date', 'y_true', 'y_pred']].drop_duplicates(subset='date', keep='first').sort_values('date')

        # 截断：today 及以后的行由在线推理填充，CSV 里的不展示
        _today = _today_cn()
        df_agg = df_agg[df_agg['date'] < _today]

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
    """Load full historical timeline (2016-2026) from processed dataset.

    Returns DataFrame with columns:
      - date (python date)
      - actual (float)
      - precip_mm, temp_high_c, temp_low_c, weather_code_en, wind_level,
        wind_dir_en, wind_max, aqi_value, aqi_level_en
    """
    # 优先使用10年数据文件，回退到旧文件
    candidates = [
        os.path.join(base_dir, 'data', 'processed', 'jiuzhaigou_daily_features_2016-01-01_2026-04-02.csv'),
        os.path.join(base_dir, 'data', 'processed', 'jiuzhaigou_8features_latest.csv'),
    ]
    processed_path = next((p for p in candidates if os.path.exists(p)), None)
    if processed_path is None:
        return None

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

        # 过滤掉 append_future_weather 追加的未来天气行（actual=NaN 的行）
        # 只保留有真实 visitor 数据的历史行
        out = out[out['actual'].notna()]

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


def _pretty_model_name(raw_key: str):
    """Convert internal model key to a professional UI label."""
    k = (raw_key or '').lower()
    if 'transformer' in k:
        return 'Transformer (8 features)'
    if 'xgboost' in k:
        return 'XGBoost (8 features)'
    if 'gru' in k:
        return 'GRU (8 features)'
    return raw_key or 'Model'

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

@app.route('/legacy')
def index():
    return redirect(url_for('dashboard_v3'))


@app.route('/legacy')
def legacy_index():
    """Legacy UI kept for rollback."""
    return render_template('index.html')


@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')


@app.route('/dashboard/v2')
def dashboard_v2():
    """Dashboard v2 (legacy, kept for compatibility)."""
    return render_template('dashboard_v2.html')


@app.route('/dashboard/v3')
@app.route('/')
def dashboard_v3():
    """Dashboard v3 — Apple-inspired redesign (current default)."""
    return render_template('dashboard_v3.html')


@app.route('/compare')
def compare():
    return render_template('compare.html')


@app.route('/definitions')
def definitions():
    return render_template('definitions.html')


@app.route('/explain')
def explain():
    return render_template('explain.html')


# --- Interaction Test Pages (minimal, single-container pages) ---


@app.route('/test/chart')
def test_chart():
    """Minimal ECharts interaction test page."""
    return render_template('test_chart.html')


@app.route('/test/weather')
def test_weather():
    """Minimal weather card interaction test page."""
    return render_template('test_weather.html')


@app.route('/test/risk')
def test_risk():
    """Minimal risk + thermo interaction test page."""
    return render_template('test_risk.html')


@app.route('/api/weather', methods=['GET'])
def api_weather():
    """代理 Open-Meteo 天气请求，避免前端直连被防火墙阻断。
    返回近 14 天历史 + 未来 14 天预报，格式与前端 WMO_CODE_MAP 兼容。
    """
    try:
        import requests as _req
        params = {
            'latitude': 33.2, 'longitude': 103.9,
            'daily': 'temperature_2m_max,temperature_2m_min,precipitation_sum,weathercode,windspeed_10m_max',
            'timezone': 'Asia/Shanghai',
            'past_days': 14,
            'forecast_days': 14,
        }
        r = _req.get('https://api.open-meteo.com/v1/forecast', params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 502


@app.route('/api/models', methods=['GET'])
def api_models():
    """Offline artifact mode: return champion + runner-up + third from latest backup."""
    backup_dir = _get_latest_backup_dir()
    df_cmp = _load_compare_metrics(backup_dir)
    champ, runner, third = _pick_champion_and_runner_up(df_cmp)

    if not champ:
        return jsonify({
            'backup_dir': backup_dir,
            'models': [],
            'warning': 'No compare_metrics.csv found under latest backup.'
        })

    champ_run = _resolve_backup_run_dir(backup_dir, champ.get('run_dir'))
    runner_run = _resolve_backup_run_dir(backup_dir, runner.get('run_dir')) if runner else None
    third_run = _resolve_backup_run_dir(backup_dir, third.get('run_dir')) if third else None

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
    if third and third_run:
        models.append({
            'model_id': 'third',
            'display_name': _pretty_model_name(third.get('model')),
            'model_key': third.get('model'),
            'run_dir': third_run,
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
    """Forecast API — 固定返回5条模型曲线。

    设计逻辑：
      - GRU单步 / LSTM单步：只读离线 backfill CSV（历史回测段，到测试集结束日）
        测试集结束日到当天的空白段由官网数据滞后造成，属正常现象。
      - GRU多步(MIMO) / LSTM多步(MIMO) / Seq2Seq：始终触发在线实时推理，
        输入为最近30天真实历史数据，输出 last_real_date+1 ~ last_real_date+7。
        因不依赖滚动反馈，可连续覆盖数据空缺段，无误差累积。

    Query:
      - h: int, [1, 14], 预测窗口长度（影响图表 zoom 范围，不影响模型推理步数）
    """
    h = int(request.args.get('h', 7))
    h = max(1, min(h, 14))
    # mode 参数保留兼容性，但不再区分 online/offline 逻辑
    mode = str(request.args.get('mode', 'online')).strip().lower()

    # ── 3个模型 key，直接从 output/runs 加载 ──
    MODEL_KEYS = [
        'gru_8features',
        'transformer_8features',
        'xgboost_8features',
    ]

    def _find_latest_run_dir(model_key: str):
        """找到 output/runs 下最新的该模型 run_dir。"""
        pattern = model_key.replace('_8features', '_8features_*').replace('_attention_', '_attention_')
        # 直接用 model_key 作为前缀匹配
        top_dirs = sorted(
            glob.glob(os.path.join(OUTPUT_RUNS_DIR, f'{model_key}_*')),
            key=os.path.getmtime, reverse=True
        )
        for top_dir in top_dirs:
            run_subdirs = glob.glob(os.path.join(top_dir, 'runs', 'run_*'))
            if not run_subdirs:
                run_subdirs = [top_dir]
            run_dir = max(run_subdirs, key=os.path.getmtime)
            if os.path.exists(os.path.join(run_dir, 'metrics.json')):
                return run_dir
        return None

    # 加载各模型数据
    model_data = {}  # key -> {run_dir, df, metrics}
    for mk in MODEL_KEYS:
        run_dir = _find_latest_run_dir(mk)
        df = _load_predictions(run_dir) if run_dir else None
        metrics = _safe_read_json(os.path.join(run_dir, 'metrics.json')) if run_dir else {}
        model_data[mk] = {'run_dir': run_dir, 'df': df, 'metrics': metrics or {}}

    # 至少需要一个模型有数据
    if all(v['df'] is None or v['df'].empty for v in model_data.values()):
        return jsonify({'error': 'No prediction artifacts found'}), 404

    # Master time axis
    warning = None
    df_master = _load_master_history_from_processed()
    if df_master is None or df_master.empty:
        warning = 'Processed history not available.'
        for mk in MODEL_KEYS:
            if model_data[mk]['df'] is not None and not model_data[mk]['df'].empty:
                df_master = model_data[mk]['df'][['date']].copy()
                df_master['actual'] = np.nan
                for col in ['precip_mm','temp_high_c','temp_low_c','weather_code_en',
                            'wind_level','wind_dir_en','wind_max','aqi_value','aqi_level_en']:
                    df_master[col] = np.nan
                break

    # 仅展示2023-06-01以后的数据（与训练集对齐，避免2016-2022异常数据污染图表）
    from datetime import date as _date_cls
    _CHART_START = _date_cls(2023, 6, 1)
    if df_master is not None and not df_master.empty:
        df_master['date'] = pd.to_datetime(df_master['date']).dt.date
        df_master = df_master[df_master['date'] >= _CHART_START].reset_index(drop=True)

    df_base = df_master[['date','actual','precip_mm','temp_high_c','temp_low_c',
                          'weather_code_en','wind_level','wind_dir_en','wind_max',
                          'aqi_value','aqi_level_en']].copy()

    # 合并各模型预测
    col_map = {
        'gru_8features':         'gru_pred',
        'transformer_8features': 'transformer_pred',
        'xgboost_8features':     'xgboost_pred',
    }
    for mk, col_name in col_map.items():
        df_m = model_data[mk]['df']
        if df_m is not None and not df_m.empty:
            df_renamed = df_m[['date','y_pred']].rename(columns={'y_pred': col_name})
            # outer join：让预测CSV里超出历史范围的 backfill 日期也出现在时间轴上
            df_base = pd.merge(df_base, df_renamed, on='date', how='outer')
        else:
            df_base[col_name] = np.nan

    # 按日期排序，对 actual/weather 列中因 outer join 产生的新行填 NaN（已是默认行为）
    df_merge = df_base.sort_values('date').reset_index(drop=True)
    # 集成预测列（加权平均，缺任一模型则按可用模型归一化权重计算）
    # 权重由网格搜索得出（XGB MAE最低，权重最高）
    _ens_weights = {'gru_pred': 0.10, 'transformer_pred': 0.20, 'xgboost_pred': 0.70}
    _avail = {c: w for c, w in _ens_weights.items() if c in df_merge.columns}
    if _avail:
        _total_w = sum(_avail.values())
        df_merge['ensemble_pred'] = sum(
            pd.to_numeric(df_merge[c], errors='coerce') * (w / _total_w)
            for c, w in _avail.items()
        )
    else:
        df_merge['ensemble_pred'] = np.nan

    def _fetch_weather_forecast(horizon: int) -> pd.DataFrame:
        """从 Open-Meteo 获取天气数据（含最近7天历史 + 未来 horizon 天预报）。

        加入 past_days=7 以覆盖官网数据滞后导致的空缺段（如最新真实数据到4/2，
        今天4/5，则4/3、4/4的天气从过去数据里取）。
        返回 DataFrame，列：date(str), temp_high, temp_low, precip_sum,
                           weathercode, windspeed_max
        失败时返回全 NaN 的占位 DataFrame。
        """
        try:
            import requests as _req
            url = 'https://api.open-meteo.com/v1/forecast'
            params = {
                'latitude': 33.2, 'longitude': 103.9,
                'daily': 'temperature_2m_max,temperature_2m_min,precipitation_sum,weathercode,windspeed_10m_max',
                'timezone': 'Asia/Shanghai',
                'forecast_days': horizon,
                'past_days': 7,  # 覆盖数据滞后造成的空缺段
            }
            r = _req.get(url, params=params, timeout=10)
            r.raise_for_status()
            d = r.json()['daily']
            return pd.DataFrame({
                'date': d['time'],
                'temp_high': d['temperature_2m_max'],
                'temp_low': d['temperature_2m_min'],
                'precip_sum': d['precipitation_sum'],
                'weathercode': d.get('weathercode', [None] * len(d['time'])),
                'windspeed_max': d.get('windspeed_10m_max', [None] * len(d['time'])),
            })
        except Exception as e:
            print(f"Weather forecast fetch failed: {e}")
            # Use last date with actual visitor data (not future-appended weather rows)
            _actual_s = pd.to_numeric(df_master['actual'], errors='coerce')
            last_date = df_master.loc[_actual_s.notna(), 'date'].max()
            rows = []
            for i in range(horizon):
                rows.append({
                    'date': str(last_date + timedelta(days=i + 1)),
                    'temp_high': float('nan'), 'temp_low': float('nan'), 'precip_sum': float('nan'),
                    'weathercode': None, 'windspeed_max': float('nan'),
                })
            return pd.DataFrame(rows)

    def _compute_stepwise_qhat(alpha: float = 0.10, max_horizon: int = 7) -> dict:
        """
        Step-wise Conformal q̂_h: compute a separate conformal threshold for each
        prediction horizon h=1..max_horizon using the ensemble's recent error pool.

        Method:
          1. Load the latest gru_ensemble run's member weights.
          2. Run each member on the calibration set (val set, held out from training).
          3. For each horizon h, compute normalised nonconformity scores:
               s_i^h = |y_i - mean_i^h| / (std_i^h + 1)
             using a rolling-window simulation of the h-step-ahead prediction.
             Here we approximate h-step error by taking the h-th element of a
             sequential multi-step pass over the val set.
          4. Apply conformal quantile: q̂_h = quantile(scores^h, ceil((n+1)(1-α))/n)

        Returns dict {1: q1, 2: q2, ..., max_horizon: q_h} and ensemble metadata.
        Falls back to a single global q̂ if ensemble not available.
        """
        import glob as _glob
        OUTPUT_RUNS = os.path.join(project_root, 'output', 'runs')
        ensemble_dirs = sorted(
            _glob.glob(os.path.join(OUTPUT_RUNS, 'gru_ensemble_*')),
            key=os.path.getmtime, reverse=True
        )
        if not ensemble_dirs:
            return {'available': False, 'qhat_by_horizon': {h: None for h in range(1, max_horizon + 1)}}

        ensemble_dir = ensemble_dirs[0]
        info_path = os.path.join(ensemble_dir, 'ensemble_info.json')
        if not os.path.exists(info_path):
            return {'available': False, 'qhat_by_horizon': {h: None for h in range(1, max_horizon + 1)}}

        try:
            import json as _json
            with open(info_path) as _f:
                ens_info = _json.load(_f)

            # Load all member models
            members = []
            for m_info in ens_info['members']:
                wp = m_info['weight_path']
                if not os.path.exists(wp):
                    wp = os.path.join(ensemble_dir, f"member_{m_info['member_idx']}", 'weights', 'gru_jiuzhaigou.h5')
                if tf and os.path.exists(wp):
                    try:
                        members.append(tf.keras.models.load_model(wp, compile=False))
                    except Exception:
                        pass
            if not members:
                return {'available': False, 'qhat_by_horizon': {h: None for h in range(1, max_horizon + 1)}}

            # Load calibration data (val set, same split as training)
            from sklearn.preprocessing import MinMaxScaler as _MMS
            csv_files = sorted(_glob.glob(os.path.join(project_root, 'data', 'processed', '*.csv')))
            if not csv_files:
                return {'available': False, 'qhat_by_horizon': {h: None for h in range(1, max_horizon + 1)}}

            df_cal = pd.read_csv(csv_files[-1])
            df_cal = df_cal.sort_values('date').drop_duplicates(subset=['date']).reset_index(drop=True)
            df_cal = df_cal[pd.to_numeric(df_cal['tourism_num'], errors='coerce').notna()].reset_index(drop=True)
            df_cal['date_dt'] = pd.to_datetime(df_cal['date'])
            df_cal['month_norm'] = (df_cal['date_dt'].dt.month - 1) / 11.0
            df_cal['day_of_week_norm'] = df_cal['date_dt'].dt.weekday / 6.0

            feat_cols = ['visitor_count_scaled', 'month_norm', 'day_of_week_norm', 'is_holiday',
                         'tourism_num_lag_7_scaled', 'meteo_precip_sum_scaled',
                         'temp_high_scaled', 'temp_low_scaled']

            _scaler = _MMS()
            _visitor_vals = pd.to_numeric(df_cal['tourism_num'], errors='coerce').dropna().values
            _scaler.fit(_visitor_vals.reshape(-1, 1))
            df_cal['visitor_count_scaled'] = _scaler.transform(
                pd.to_numeric(df_cal['tourism_num'], errors='coerce').values.reshape(-1, 1)
            ).flatten()

            for _c in feat_cols:
                if _c not in df_cal.columns:
                    return {'available': False, 'qhat_by_horizon': {h: None for h in range(1, max_horizon + 1)}}

            vals = df_cal[feat_cols].values.astype(np.float32)
            target = df_cal['visitor_count_scaled'].values.astype(np.float32)
            look_back_e = 30
            X_all, Y_all = [], []
            for i in range(look_back_e, len(vals)):
                X_all.append(vals[i - look_back_e:i])
                Y_all.append(target[i])
            X_all = np.array(X_all, dtype=np.float32)
            Y_all = np.array(Y_all, dtype=np.float32)

            n_total = len(X_all)
            test_size = int(n_total * 0.10)
            trainval = n_total - test_size
            val_size = int(trainval * 0.111)
            train_size = trainval - val_size

            # Use the recent half of test set as calibration (same as --cal-source recent)
            # to reduce distribution shift: test[0:mid] calibrates, test[mid:] evaluates
            X_test = X_all[trainval:]
            Y_test = Y_all[trainval:]
            mid = len(X_test) // 2
            X_cal_e = X_test[:mid]
            Y_cal_e = Y_test[:mid]

            # Get ensemble predictions on calibration set
            n_cal = len(X_cal_e)
            preds_scaled = np.array([
                m.predict(X_cal_e, verbose=0).flatten() for m in members
            ])  # (n_members, n_cal)
            preds_raw = _scaler.inverse_transform(
                preds_scaled.reshape(-1, 1)
            ).reshape(len(members), n_cal)
            y_cal_raw = _scaler.inverse_transform(Y_cal_e.reshape(-1, 1)).flatten()

            cal_mean = preds_raw.mean(axis=0)

            # Step-wise q̂_h using DIRECT absolute residuals (not normalised by std).
            # Root cause of previous ±44k bug: normalising by std (mean=637) then
            # multiplying back by std amplifies extreme outlier scores (max=87) into
            # q̂×std = 26×637 = 16,651 — absurd for a ≤41k-max park.
            #
            # Fix: use raw |residual| as nonconformity score directly.
            # This gives q̂_h in visitor units, interpretable and bounded.
            # h-step scaling: q̂_h = q̂_1 × sqrt(h) — standard random walk growth.
            # Physical cap: half-width ≤ max_capacity (41,000 visitors).
            MAX_CAPACITY = float(_visitor_vals.max())  # physical upper bound

            abs_residuals = np.abs(y_cal_raw - cal_mean)  # in visitor units
            n_c = len(abs_residuals)

            # Winsorize at 95th percentile to remove extreme outlier influence
            # (e.g. park closure events with residual >> normal range)
            p95 = float(np.percentile(abs_residuals, 95))
            abs_residuals_w = np.clip(abs_residuals, 0, p95)

            level = min(np.ceil((n_c + 1) * (1 - alpha)) / n_c, 1.0)
            q_base = float(np.quantile(abs_residuals_w, level))  # h=1 half-width in visitors

            qhat_by_horizon = {}
            half_widths = {}
            for h_step in range(1, max_horizon + 1):
                # sqrt(h) growth models random-walk error accumulation over horizons
                hw = min(q_base * np.sqrt(h_step), MAX_CAPACITY)
                qhat_by_horizon[h_step] = round(hw, 1)   # directly in visitor units
                half_widths[h_step] = round(hw, 1)

            n_members_loaded = len(members)
            mean_std = float(preds_raw.std(axis=0).mean())

            return {
                'available': True,
                'n_members': n_members_loaded,
                'cal_size': n_cal,
                'alpha': alpha,
                'mean_ensemble_std': round(mean_std, 1),
                'qhat_by_horizon': qhat_by_horizon,
                'half_width_by_horizon': half_widths,
                'ensemble_dir': ensemble_dir,
            }

        except Exception as _e:
            print(f'[stepwise_qhat] failed: {_e}')
            return {'available': False, 'qhat_by_horizon': {h: None for h in range(1, max_horizon + 1)}}

    def _online_future_forecast_all_models(df_hist: pd.DataFrame, horizon: int):
        """五模型在线预测：用8特征 + Open-Meteo 天气预报生成未来 horizon 天预测。

        返回 DataFrame，列：date, gru_single_pred, gru_mimo_pred,
                              lstm_single_pred, lstm_mimo_pred, seq2seq_pred,
                              precip_mm, temp_high_c, temp_low_c
        """
        if df_hist is None or df_hist.empty:
            raise RuntimeError('History not available for online forecast.')

        # ── 1. 准备历史客流序列 ──
        s = pd.to_numeric(df_hist['actual'], errors='coerce').dropna()
        if len(s) < 30:
            raise RuntimeError('Not enough history (need >= 30 days).')

        look_back = 30
        visitor_scaler = MinMaxScaler()
        visitor_scaler.fit(s.values.reshape(-1, 1))

        # ── 2. 获取天气预报（多取2天以覆盖今天/昨天可能不在预报窗口的情况）──
        weather_df = _fetch_weather_forecast(horizon + 2)
        weather_df['date'] = pd.to_datetime(weather_df['date']).dt.date

        def _wmo_to_code_en(wmo):
            """Map WMO weathercode integer to weather_code_en string used by the dashboard."""
            if wmo is None:
                return None
            try:
                wmo = int(wmo)
            except Exception:
                return None
            if wmo == 0:
                return 'SUNNY'
            elif wmo in (1, 2):
                return 'PARTLY_CLOUDY'
            elif wmo == 3:
                return 'CLOUDY'
            elif wmo in (45, 48):
                return 'FOGGY'
            elif wmo in (51, 53, 55, 56, 57):
                return 'DRIZZLE'
            elif wmo in (61, 63, 65, 66, 67, 80, 81, 82):
                return 'RAINY'
            elif wmo in (71, 73, 75, 77, 85, 86):
                return 'SNOWY'
            elif wmo in (95, 96, 99):
                return 'THUNDERSTORM'
            elif wmo in (4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19):
                return 'OVERCAST'
            else:
                return 'CLOUDY'

        def _windspeed_to_level(kmh):
            """Convert km/h wind speed to wind level (Beaufort approximate)."""
            if kmh is None or (isinstance(kmh, float) and np.isnan(kmh)):
                return None
            kmh = float(kmh)
            if kmh < 1:   return 0
            elif kmh < 6:  return 1
            elif kmh < 12: return 2
            elif kmh < 20: return 3
            elif kmh < 29: return 4
            elif kmh < 39: return 5
            elif kmh < 50: return 6
            else:          return 7

        # ── 3. 从处理好的历史数据中获取天气 scaler 参数 ──
        # 用历史数据的 min/max 对天气特征归一化（与训练时一致）
        processed_path = os.path.join(base_dir, 'data', 'processed',
                                       'jiuzhaigou_daily_features_2016-01-01_2026-04-02.csv')
        precip_min, precip_max = 0.0, 50.0
        temp_high_min, temp_high_max = -10.0, 40.0
        temp_low_min, temp_low_max = -20.0, 30.0
        lag7_vals = None
        if os.path.exists(processed_path):
            try:
                _hist_df = pd.read_csv(processed_path, usecols=[
                    'meteo_precip_sum', 'temp_high_c', 'temp_low_c',
                    'tourism_num', 'tourism_num_lag_7_scaled', 'date'
                ])
                # 只用有真实访客数据的行（过滤 append_future_weather 追加的 NaN 行）
                _hist_df = _hist_df[_hist_df['tourism_num'].notna()].reset_index(drop=True)
                precip_min = float(_hist_df['meteo_precip_sum'].min())
                precip_max = float(_hist_df['meteo_precip_sum'].max()) or 50.0
                temp_high_min = float(_hist_df['temp_high_c'].min())
                temp_high_max = float(_hist_df['temp_high_c'].max())
                temp_low_min = float(_hist_df['temp_low_c'].min())
                temp_low_max = float(_hist_df['temp_low_c'].max())
                # 最近 look_back+7 天的 lag7 scaled 值（用于初始化滚动窗口，全为真实值）
                lag7_vals = _hist_df['tourism_num_lag_7_scaled'].values[-(look_back + 7):]
            except Exception:
                pass

        def _scale_precip(v):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return 0.0
            return float(np.clip((v - precip_min) / max(precip_max - precip_min, 1e-6), 0, 1))

        def _scale_temp_high(v):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return 0.5
            return float(np.clip((v - temp_high_min) / max(temp_high_max - temp_high_min, 1e-6), 0, 1))

        def _scale_temp_low(v):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return 0.5
            return float(np.clip((v - temp_low_min) / max(temp_low_max - temp_low_min, 1e-6), 0, 1))

        # ── 4. 构建初始 look_back 窗口（8特征） ──
        # last_date = 最后一条有真实 actual 的日期（过滤掉 append_future_weather 追加的 NaN 行）
        _hist_with_actual = df_hist[pd.to_numeric(df_hist['actual'], errors='coerce').notna()]
        if _hist_with_actual.empty:
            raise RuntimeError('No actual visitor data found in history.')
        last_date = _hist_with_actual['date'].max()
        hist_vals = s.values[-look_back:]
        hist_scaled = visitor_scaler.transform(hist_vals.reshape(-1, 1)).flatten()

        # lag_7 scaled：用历史数据中最近的值初始化
        if lag7_vals is not None and len(lag7_vals) >= look_back:
            lag7_window = list(lag7_vals[-look_back:].astype(float))
        else:
            lag7_window = list(hist_scaled)  # 降级：用 visitor_scaled 近似

        def _days_to_next_hol(d):
            try:
                import chinese_calendar as _cc
                for delta in range(1, 15):
                    if _cc.is_holiday(d + timedelta(days=delta)):
                        return delta / 14.0
            except Exception:
                pass
            return 1.0

        def _days_since_last_hol(d):
            try:
                import chinese_calendar as _cc
                for delta in range(1, 15):
                    if _cc.is_holiday(d - timedelta(days=delta)):
                        return delta / 14.0
            except Exception:
                pass
            return 1.0

        def _build_window_feat(visitor_window, lag7_window, dates, n_features=12):
            """构建 (look_back, n_features) 特征矩阵，默认12特征。"""
            rows = []
            for i, d in enumerate(dates):
                m_norm = (d.month - 1) / 11.0
                dow_norm = d.weekday() / 6.0
                hol = float(_is_holiday(d))
                m = d.month
                is_peak = float((4 <= m <= 10) or (m == 11 and d.day <= 15))
                d2n = _days_to_next_hol(d)
                d2s = _days_since_last_hol(d)
                if n_features >= 12:
                    row = [
                        visitor_window[i],   # visitor_count_scaled
                        m_norm,              # month_norm
                        dow_norm,            # day_of_week_norm
                        hol,                 # is_holiday
                        is_peak,             # is_peak_season
                        d2n,                 # days_to_next_holiday
                        d2s,                 # days_since_last_holiday
                        lag7_window[i],      # tourism_num_lag_7_scaled
                        visitor_window[max(0, i - 7)],  # tourism_num_lag_14_scaled (近似)
                        0.0,                 # meteo_precip_sum_scaled
                        0.5,                 # temp_high_scaled
                        0.5,                 # temp_low_scaled
                    ]
                elif n_features == 11:
                    lag14_s = visitor_window[max(0, i - 7)]  # 近似 lag14
                    win14 = visitor_window[max(0, i - 13):i + 1]
                    roll14 = float(np.mean(win14)) if len(win14) > 0 else visitor_window[i]
                    row = [
                        visitor_window[i],
                        m_norm,
                        dow_norm,
                        hol,
                        is_peak,
                        lag7_window[i],
                        lag14_s,
                        roll14,
                        0.0,
                        0.5,
                        0.5,
                    ]
                else:
                    row = [
                        visitor_window[i],
                        m_norm,
                        dow_norm,
                        hol,
                        lag7_window[i],
                        0.0,
                        0.5,
                        0.5,
                    ]
                rows.append(row)
            return np.array(rows, dtype=np.float32)

        # 历史窗口的日期
        hist_dates = [last_date - timedelta(days=look_back - 1 - i) for i in range(look_back)]

        def _predict_single_step_model(model, visitor_window, lag7_window, hist_dates, steps=None):
            """单步模型（LSTM/GRU/Transformer）滚动预测，自动检测输入形状（look_back, n_features）。"""
            if steps is None:
                steps = horizon
            # 自动检测模型输入维度
            try:
                _in_shape = model.input_shape  # (None, lb, n_feat)
                n_feat = _in_shape[-1]
                m_look_back = _in_shape[-2] if len(_in_shape) >= 3 and _in_shape[-2] else look_back
            except Exception:
                n_feat = 8
                m_look_back = look_back
            # 天气特征固定在最后3位
            precip_idx = n_feat - 3
            temp_h_idx = n_feat - 2
            temp_l_idx = n_feat - 1

            # 如果模型 look_back 与全局不同，重新切窗口
            if m_look_back != look_back:
                m_hist_vals = s.values[-m_look_back:]
                m_hist_scaled = visitor_scaler.transform(m_hist_vals.reshape(-1, 1)).flatten()
                if lag7_vals is not None and len(lag7_vals) >= m_look_back:
                    m_lag7 = list(lag7_vals[-m_look_back:].astype(float))
                else:
                    m_lag7 = list(m_hist_scaled)
                m_dates = [last_date - timedelta(days=m_look_back - 1 - i) for i in range(m_look_back)]
                v_win = list(m_hist_scaled)
                l7_win = m_lag7
                cur_dates = m_dates
            else:
                v_win = list(visitor_window)
                l7_win = list(lag7_window)
                cur_dates = list(hist_dates)

            preds = []
            for step in range(steps):
                pred_date = last_date + timedelta(days=step + 1)
                wrow = weather_df[weather_df['date'] == pred_date]
                p_s = _scale_precip(float(wrow['precip_sum'].iloc[0]) if len(wrow) else float('nan'))
                th_s = _scale_temp_high(float(wrow['temp_high'].iloc[0]) if len(wrow) else float('nan'))
                tl_s = _scale_temp_low(float(wrow['temp_low'].iloc[0]) if len(wrow) else float('nan'))

                X = _build_window_feat(v_win, l7_win, cur_dates, n_features=n_feat)
                X[-1, precip_idx] = p_s
                X[-1, temp_h_idx] = th_s
                X[-1, temp_l_idx] = tl_s

                x_in = X.reshape(1, m_look_back, n_feat)
                y_s = float(model.predict(x_in, verbose=0)[0][0])
                y_val = float(visitor_scaler.inverse_transform([[y_s]])[0][0])
                preds.append(y_val)

                v_win = v_win[1:] + [y_s]
                lag7_s = v_win[-7] if len(v_win) >= 7 else y_s
                l7_win = l7_win[1:] + [lag7_s]
                cur_dates = cur_dates[1:] + [pred_date]

            return preds

        # ── 5. 在线推理：GRU / Transformer 滚动单步，XGBoost 表格推理 ──
        ONLINE_MODELS = [
            ('gru_8features',         'gru_pred',         'single'),
            ('transformer_8features', 'transformer_pred',  'single'),
        ]

        def _load_model_by_key(model_key: str):
            """从 output/runs 找最新 run_dir 并加载模型。
            Seq2Seq 子类化模型用 _weights.h5 + load_weights 方式加载（跨平台兼容）。
            GRU/LSTM 用标准 load_model。
            """
            top_dirs = sorted(
                glob.glob(os.path.join(OUTPUT_RUNS_DIR, f'{model_key}_*')),
                key=os.path.getmtime, reverse=True
            )
            is_seq2seq = 'seq2seq' in model_key
            for top_dir in top_dirs:
                run_subdirs = glob.glob(os.path.join(top_dir, 'runs', 'run_*'))
                run_dir = max(run_subdirs, key=os.path.getmtime) if run_subdirs else top_dir
                # Seq2Seq：优先找 _weights.h5，用 load_weights 重建
                if is_seq2seq and tf and _seq2seq_custom_objects:
                    wfiles = glob.glob(os.path.join(run_dir, 'weights', '*_weights.h5'))
                    if wfiles:
                        try:
                            import numpy as np
                            from models.lstm.train_seq2seq_attention_8features import Seq2SeqWithAttention
                            m = Seq2SeqWithAttention(
                                encoder_units=128, decoder_units=256,
                                encoder_features=8, decoder_features=7
                            )
                            dummy_enc = np.zeros((1, 30, 8), dtype=np.float32)
                            dummy_dec = np.zeros((1, 7, 7), dtype=np.float32)
                            m([dummy_enc, dummy_dec], training=False)
                            m.load_weights(wfiles[0])
                            print(f"Seq2Seq loaded via load_weights: {wfiles[0]}")
                            return m
                        except Exception as e:
                            print(f"Seq2Seq load_weights failed ({wfiles[0]}): {e}")
                # GRU/LSTM（及 Seq2Seq fallback）：标准 load_model
                # 为 Transformer 注入 PositionalEncoding 自定义层
                _extra_custom_objs = {}
                if 'transformer' in model_key:
                    try:
                        from models.transformer.train_transformer_8features import (
                            PositionalEncoding, TransformerEncoderBlock
                        )
                        _extra_custom_objs['PositionalEncoding'] = PositionalEncoding
                        _extra_custom_objs['TransformerEncoderBlock'] = TransformerEncoderBlock
                    except Exception:
                        pass
                _all_custom_objs = {**(_seq2seq_custom_objects or {}), **_extra_custom_objs} or None
                for pattern in ['weights/*.keras', 'weights/*.h5']:
                    matches = [p for p in glob.glob(os.path.join(run_dir, pattern))
                               if '_weights.h5' not in p]  # 排除 _weights.h5
                    if matches:
                        try:
                            m = tf.keras.models.load_model(
                                matches[0],
                                custom_objects=_all_custom_objs,
                                compile=False
                            ) if tf else None
                            return m
                        except Exception as e:
                            print(f"Online load failed ({matches[0]}): {e}")
            return None

        results = {}  # col_name -> list of preds
        for mk, col_name, infer_type in ONLINE_MODELS:
            model_obj = _load_model_by_key(mk)
            if model_obj is None:
                results[col_name] = None
                continue
            try:
                # GRU / Transformer: rolling single-step inference
                # 需要推理 gap_days + horizon 步才能覆盖 today+horizon-1
                _gap = max(0, (_today_cn() - last_date).days - 1)
                _total_steps = _gap + horizon
                preds = _predict_single_step_model(
                    model_obj, list(hist_scaled), list(lag7_window), list(hist_dates),
                    steps=_total_steps
                )
                results[col_name] = preds
            except Exception as e:
                print(f"Online prediction failed for {col_name} ({mk}): {e}")
                results[col_name] = None

        # ── XGBoost 在线推理（表格模型，3步滚动）──
        xgb_preds = None
        try:
            import joblib as _joblib
            import xgboost as _xgb_lib
            xgb_top_dirs = sorted(
                glob.glob(os.path.join(OUTPUT_RUNS_DIR, 'xgboost_8features_*')),
                key=os.path.getmtime, reverse=True
            )
            for _xd in xgb_top_dirs:
                _run_subdirs = glob.glob(os.path.join(_xd, 'runs', 'run_*'))
                _run_dir = max(_run_subdirs, key=os.path.getmtime) if _run_subdirs else _xd
                # 优先 pkl/joblib，再找 xgboost 原生 .json
                _model_files = (glob.glob(os.path.join(_run_dir, 'weights', '*.pkl')) +
                                glob.glob(os.path.join(_run_dir, 'weights', '*.joblib')))
                _json_files  = [p for p in glob.glob(os.path.join(_run_dir, 'weights', '*.json'))
                                if 'metrics' not in os.path.basename(p)]
                if _model_files:
                    _xgb_model = _joblib.load(_model_files[0])
                elif _json_files:
                    _xgb_model = _xgb_lib.XGBRegressor()
                    _xgb_model.load_model(_json_files[0])
                else:
                    continue
                # Build feature row for last_date (same as training feature set)
                # Use last 45 rows of df_hist for lag features
                _hist_full = pd.read_csv(
                    os.path.join(base_dir, 'data', 'processed',
                                 'jiuzhaigou_daily_features_2016-01-01_2026-04-02.csv')
                )
                _hist_full['date'] = pd.to_datetime(_hist_full['date']).dt.date
                _hist_full = _hist_full[_hist_full['tourism_num'].notna()].sort_values('date')
                # XGBoost rolling inference — feature names must match training exactly
                _xgb_gap = max(0, (_today_cn() - last_date).days - 1)
                _xgb_total = _xgb_gap + horizon
                # 初始化滚动窗口（scaled visitor values），用来计算lag特征
                _xgb_win = list(hist_scaled.copy())  # scaled visitor history
                xgb_preds = []
                for _step in range(_xgb_total):
                    _pred_date = last_date + timedelta(days=_step + 1)
                    _m_norm = (_pred_date.month - 1) / 11.0
                    _dow_norm = _pred_date.weekday() / 6.0
                    _hol = float(_is_holiday(_pred_date))
                    _m = _pred_date.month
                    _is_peak = float((4 <= _m <= 10) or (_m == 11 and _pred_date.day <= 15))
                    _wrow = weather_df[weather_df['date'] == _pred_date]
                    _precip_raw = float(_wrow['precip_sum'].iloc[0]) if len(_wrow) else 0.0
                    _th_raw = float(_wrow['temp_high'].iloc[0]) if len(_wrow) else 15.0
                    _tl_raw = float(_wrow['temp_low'].iloc[0]) if len(_wrow) else 5.0

                    def _lag(win, k):
                        return float(win[-k]) if len(win) >= k else float(win[-1])

                    _win = _xgb_win
                    _feat = {
                        'month_norm':                _m_norm,
                        'day_of_week_norm':          _dow_norm,
                        'is_holiday':                _hol,
                        'is_peak_season':            _is_peak,
                        'days_to_next_holiday':      _days_to_next_hol(_pred_date),
                        'days_since_last_holiday':   _days_since_last_hol(_pred_date),
                        'tourism_num_lag_1_scaled':  _lag(_win, 1),
                        'tourism_num_lag_7_scaled':  _lag(_win, 7),
                        'tourism_num_lag_14_scaled': _lag(_win, 14),
                        'rolling_mean_7_scaled':     float(np.mean(_win[-7:])) if len(_win) >= 7 else float(np.mean(_win)),
                        'meteo_precip_sum_scaled':   _scale_precip(_precip_raw),
                        'temp_high_scaled':          _scale_temp_high(_th_raw),
                        'temp_low_scaled':           _scale_temp_low(_tl_raw),
                    }
                    try:
                        _feat_df = pd.DataFrame([_feat])
                        _y_s = float(_xgb_model.predict(_feat_df)[0])
                        _y_val = float(visitor_scaler.inverse_transform([[_y_s]])[0][0])
                    except Exception as _pe:
                        print(f"XGBoost step {_step} predict error: {_pe}")
                        _y_s = _xgb_win[-1]
                        _y_val = float(visitor_scaler.inverse_transform([[_y_s]])[0][0])
                    xgb_preds.append(_y_val)
                    _xgb_win = _xgb_win[1:] + [_y_s]  # 滚动更新 scaled 窗口
                break
        except Exception as _xe:
            print(f"XGBoost online inference failed: {_xe}")
            xgb_preds = None

        # ── 6. 组装结果 DataFrame ──
        today = _today_cn()
        gap_days = max(0, (today - last_date).days - 1)

        out_rows = []
        for step in range(horizon):
            pred_date = today + timedelta(days=step)
            wrow = weather_df[weather_df['date'] == pred_date]
            _precip = float(wrow['precip_sum'].iloc[0]) if len(wrow) else float('nan')
            _temp_h = float(wrow['temp_high'].iloc[0]) if len(wrow) else float('nan')
            _temp_l = float(wrow['temp_low'].iloc[0]) if len(wrow) else float('nan')
            _wmo = wrow['weathercode'].iloc[0] if len(wrow) and 'weathercode' in wrow.columns else None
            _wspd = float(wrow['windspeed_max'].iloc[0]) if len(wrow) and 'windspeed_max' in wrow.columns else float('nan')
            row = {
                'date': pred_date,
                'precip_mm': _precip,
                'temp_high_c': _temp_h,
                'temp_low_c': _temp_l,
                'weather_code_en': _wmo_to_code_en(_wmo),
                'wind_level': _windspeed_to_level(_wspd),
                'wind_max': _wspd if not (isinstance(_wspd, float) and np.isnan(_wspd)) else None,
            }
            # GRU / Transformer: step 0 = today = last_date + gap_days + 1
            for col_name in ['gru_pred', 'transformer_pred']:
                preds = results.get(col_name)
                # results starts from last_date+1; slice from gap_days to get today+step
                _idx = gap_days + step
                row[col_name] = preds[_idx] if preds and _idx < len(preds) else float('nan')
            # XGBoost: step 0 = last_date+1, need gap_days offset too
            row['xgboost_pred'] = xgb_preds[gap_days + step] if xgb_preds and (gap_days + step) < len(xgb_preds) else float('nan')
            _ens_vals = [row.get(c, float('nan')) for c in ['gru_pred', 'transformer_pred', 'xgboost_pred']]
            _ens_valid = [v for v in _ens_vals if not (isinstance(v, float) and np.isnan(v))]
            row['ensemble_pred'] = float(np.mean(_ens_valid)) if _ens_valid else float('nan')
            out_rows.append(row)
        return pd.DataFrame(out_rows)

    # ── 计算 zones 所需的边界日期（供 payload 和内部逻辑共用）──
    _outer_today = _today_cn()
    _actual_s_outer = pd.to_numeric(df_master['actual'], errors='coerce')
    _last_date_outer = df_master.loc[_actual_s_outer.notna(), 'date'].max()
    if hasattr(_last_date_outer, 'date'):
        _last_date_outer = _last_date_outer.date()
    _gap_days_outer = max(0, (_outer_today - _last_date_outer).days - 1) if _last_date_outer else 0

    # ── MIMO + Seq2Seq 始终触发在线推理，单步只读 CSV ──
    # 设计说明：
    #   单步（GRU/LSTM）：从 last_date+1 滚动推理到 today+horizon-1，只展示 today 起的部分
    #   MIMO/Seq2Seq：encoder 用 last_date 前30天真实数据，输出直接映射到 today~today+6
    online_used = False
    try:
        df_future = _online_future_forecast_all_models(df_master[['date', 'actual']].copy(), h)
        if df_future is not None and not df_future.empty:
            online_used = True
            df_future['actual'] = np.nan
            # 补齐其他可能缺失的列
            for _c in ['wind_dir_en', 'aqi_value', 'aqi_level_en']:
                if _c not in df_future.columns:
                    df_future[_c] = None
            for _c in ['weather_code_en', 'wind_level', 'wind_max']:
                if _c not in df_future.columns:
                    df_future[_c] = np.nan
            # 确保 df_merge 有所有预测列
            for _c in ['gru_pred', 'transformer_pred', 'xgboost_pred', 'ensemble_pred']:
                if _c not in df_merge.columns:
                    df_merge[_c] = np.nan
            _today = _today_cn()
            df_merge['date'] = pd.to_datetime(df_merge['date']).dt.date
            df_merge = df_merge[df_merge['date'] < _today]
            # 确保 df_future 也有所有预测列
            for _c in ['gru_pred', 'transformer_pred', 'xgboost_pred', 'ensemble_pred']:
                if _c not in df_future.columns:
                    df_future[_c] = np.nan
            df_merge = pd.concat([
                df_merge,
                df_future[['date', 'actual', 'precip_mm', 'temp_high_c', 'temp_low_c',
                            'weather_code_en', 'wind_level', 'wind_dir_en', 'wind_max',
                            'aqi_value', 'aqi_level_en',
                            'gru_pred', 'transformer_pred', 'xgboost_pred', 'ensemble_pred']]
            ], ignore_index=True)
            df_merge = df_merge.sort_values('date').reset_index(drop=True)
    except Exception as e:
        warning = (warning + ' | ' if warning else '') + f'MIMO/Seq2Seq online forecast failed: {e}'
        online_used = False

    # uncertainty removed (CI band not shown in simplified frontend)

    # Thresholds — 从任意有数据的模型 metrics 里取，优先 gru
    _ref_metrics = {}
    for _mk in ['gru_8features', 'transformer_8features', 'xgboost_8features']:
        if model_data.get(_mk, {}).get('metrics'):
            _ref_metrics = model_data[_mk]['metrics']
            break
    # 预警阈值按预测日期动态取（旺季32800/淡季18400），不再从 metrics.json 读固定值
    from models.common.core_evaluation import get_season_peak_threshold, PEAK_THRESHOLD_PEAK, PEAK_THRESHOLD_OFF
    _forecast_date = _today_cn()  # 预测起点为今天
    threshold_crowd = get_season_peak_threshold(_forecast_date)
    # 气象预警阈值：改用旅游舒适度绝对阈值，不再依赖统计分位数
    # - 高温(>28°C)：暑热，九寨沟夏季高温，游客体验明显下降
    # - 低温(<2°C)：接近冰点，山路结冰，徒步风险上升
    # - 强降水(>10mm/day)：中雨以上，影响景区观光和安全
    precip_high = 10.0   # mm/day，中雨下限
    temp_high   = 28.0   # °C，暑热阈值
    temp_low    = 2.0    # °C，冰点风险阈值
    quantiles = {}

    def _compute_risk(pred_col: str):
        """客流为主的风险计算：超阈值即预警，天气仅作辅助说明不单独触发预警。
        risk_level: 0=正常, 1=关注(天气异常但客流正常), 2=预警(客流超阈值)
        """
        actual_s = pd.to_numeric(df_merge['actual'], errors='coerce')
        if pred_col not in df_merge.columns:
            y_pred = actual_s.fillna(0)
        else:
            pred_s = pd.to_numeric(df_merge[pred_col], errors='coerce')
            y_pred = actual_s.where(actual_s.notna(), pred_s)
        dates_for_thr = pd.to_datetime(df_merge['date']).dt.date
        daily_thr = dates_for_thr.map(get_season_peak_threshold)
        crowd_alert = pd.Series(y_pred.values if hasattr(y_pred,'values') else y_pred) >= daily_thr.values
        weather_hazard = (
            (df_merge['precip_mm'].fillna(0) >= precip_high) |
            (df_merge['temp_high_c'].fillna(0) >= temp_high) |
            (df_merge['temp_low_c'].fillna(0) <= temp_low)
        )
        # suitability_warning 以客流为主：客流超阈值即触发，天气不单独触发
        suitability = crowd_alert.copy()
        ca_list = [bool(x) if not pd.isna(x) else False for x in crowd_alert]
        wh_list = [bool(x) if not pd.isna(x) else False for x in weather_hazard]
        sw_list = [1 if bool(x) else 0 for x in suitability]
        risk_level, drivers = [], []
        for ca, whz, pr, th, tl in zip(ca_list, wh_list,
                df_merge['precip_mm'].tolist(), df_merge['temp_high_c'].tolist(), df_merge['temp_low_c'].tolist()):
            if ca:
                lv = 2  # 客流超阈值 → 预警
                d = ['crowd_over_threshold']
                # 天气异常作为附加说明，不影响level
                if whz:
                    if pr is not None and not (isinstance(pr,float) and np.isnan(pr)) and pr >= precip_high: d.append('precip_high')
                    if th is not None and not (isinstance(th,float) and np.isnan(th)) and th >= temp_high: d.append('temp_high')
                    if tl is not None and not (isinstance(tl,float) and np.isnan(tl)) and tl <= temp_low: d.append('temp_low')
            elif whz:
                lv = 1  # 仅天气异常 → 关注（不触发预警）
                d = []
                if pr is not None and not (isinstance(pr,float) and np.isnan(pr)) and pr >= precip_high: d.append('precip_high')
                if th is not None and not (isinstance(th,float) and np.isnan(th)) and th >= temp_high: d.append('temp_high')
                if tl is not None and not (isinstance(tl,float) and np.isnan(tl)) and tl <= temp_low: d.append('temp_low')
            else:
                lv, d = 0, []
            risk_level.append(int(lv)); drivers.append(d)
        # risk_score 完全由客流决定（天气不参与评分）
        p_warn = [0.90 if ca else 0.10 for ca in ca_list]
        risk_score = [round(max(0.0, min(1.0, (lv / 2.0) * 0.80 + float(pw) * 0.20)) * 100.0, 1)
                      for lv, pw in zip(risk_level, p_warn)]
        return {'crowd_alert':ca_list,'weather_hazard':wh_list,'suitability_warning_bin':sw_list,
                'risk_level':risk_level,'p_warn':p_warn,'drivers':drivers,'risk_score':risk_score}

    # 用 GRU 作为主风险计算基准（优先级：gru > transformer > xgboost）
    _risk_col = next((c for c in ['gru_pred', 'transformer_pred', 'xgboost_pred']
                      if c in df_merge.columns and not df_merge[c].isna().all()), None)
    risk_main = _compute_risk(_risk_col) if _risk_col else _compute_risk('gru_pred')

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

    # forecast window：以今天为起点，今天+h-1 为终点
    # 与 _online_future_forecast_all_models 的输出日期对齐
    _today_str = _today_cn().isoformat()
    _today_end_str = (_today_cn() + timedelta(days=h - 1)).isoformat()
    forecast_start_idx = 0
    forecast_end_idx = len(time_axis) - 1
    if _today_str in time_axis:
        forecast_start_idx = time_axis.index(_today_str)
    if _today_end_str in time_axis:
        forecast_end_idx = time_axis.index(_today_end_str)
    elif forecast_start_idx > 0:
        forecast_end_idx = min(forecast_start_idx + h - 1, len(time_axis) - 1)
    forecast_mode = 'online_mimo_seq2seq' if online_used else 'offline_backtest'

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

    # 测试集开始日期（所有模型 CSV 最早日期的最小值）
    test_start_date = None
    for mk, mdata in model_data.items():
        _df = mdata['df']
        if _df is not None and not _df.empty and 'date' in _df.columns:
            _min = pd.to_datetime(_df['date']).min()
            if test_start_date is None or _min < test_start_date:
                test_start_date = _min
    test_start_str = test_start_date.strftime('%Y-%m-%d') if test_start_date is not None else None

    # 模型元信息列表
    models_meta = []
    display_names = {
        'gru_8features':         ('GRU',         'GRU'),
        'transformer_8features': ('Transformer',  'Transformer'),
        'xgboost_8features':     ('XGBoost',      'XGBoost'),
    }
    for mk in MODEL_KEYS:
        short, full = display_names.get(mk, (mk, mk))
        models_meta.append({
            'model_key': mk,
            'series_key': col_map[mk],
            'short_name': short,
            'display_name': full,
            'available': model_data[mk]['df'] is not None and not model_data[mk]['df'].empty,
        })

    return jsonify({
        'meta': {
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'forecast_mode': forecast_mode,
            'test_start_date': test_start_str,
            'models': models_meta,
        },
        'time_axis': time_axis,
        'forecast': {
            'h': h,
            'start_index': forecast_start_idx,
            'end_index': forecast_end_idx
        },
        'series': {
            'actual':           _to_num_list('actual'),
            'gru_pred':         _to_num_list('gru_pred'),
            'transformer_pred': _to_num_list('transformer_pred'),
            'xgboost_pred':     _to_num_list('xgboost_pred'),
            'ensemble_pred':    _to_num_list('ensemble_pred'),
        },
        'thresholds': {
            'crowd': threshold_crowd,
            'crowd_peak': float(PEAK_THRESHOLD_PEAK),    # 旺季阈值 32800（4/1~11/15）
            'crowd_off': float(PEAK_THRESHOLD_OFF),      # 淡季阈值 18400（11/16~3/31）
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
        'risk': risk_main,
        'zones': {
            'history_end': _last_date_outer.isoformat() if _last_date_outer else None,
            'gap_end': (_outer_today - timedelta(days=1)).isoformat() if _gap_days_outer > 0 else None,
            'forecast_start': _outer_today.isoformat(),
            'forecast_end': (_outer_today + timedelta(days=h - 1)).isoformat(),
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

# --- Scheduler Setup ---
def _start_scheduler():
    """Start APScheduler background scheduler for daily auto-crawl and monthly retrain."""
    try:
        from apscheduler.schedulers.background import BackgroundScheduler
        from scripts.append_and_retrain import start_data_pipeline_scheduler

        scheduler = BackgroundScheduler(timezone='Asia/Shanghai')
        # 每日 08:30 追加数据 + 每日 09:30 backfill 预测 + 每月1日 02:00 重训
        start_data_pipeline_scheduler(scheduler)
        scheduler.start()
        print("Scheduler started: daily append (08:30), daily backfill (09:30), monthly retrain (1st 02:00).")
        return scheduler
    except ImportError as e:
        print(f"WARNING: Scheduler failed to import dependency: {e}")
        return None
    except Exception as e:
        print(f"WARNING: Scheduler failed to start: {e}")
        return None


@app.route('/api/scheduler/status', methods=['GET'])
def api_scheduler_status():
    """Return scheduler status."""
    sched = app.config.get('_scheduler')
    if sched is None:
        return jsonify({'running': False, 'jobs': [], 'warning': 'Scheduler not started.'})
    jobs = [{'id': j.id, 'next_run': str(j.next_run_time)} for j in sched.get_jobs()]
    return jsonify({'running': sched.running, 'jobs': jobs})


# --- App Entry Point ---
if __name__ == '__main__':
    with app.app_context():
        db.create_all()

    print("Attempting to sync latest data...")
    try:
        sync_data()
    except Exception as e:
        print(f"Sync failed: {e}")

    _sched = _start_scheduler()
    if _sched:
        app.config['_scheduler'] = _sched

    # 绑定 0.0.0.0 使容器内可从外部访问（本地开发时等同于 localhost）
    print("Starting Flask server on port 5000...")
    app.run(host="0.0.0.0", debug=False, port=5000)
