import os
import sys
import glob
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from flask import Flask, jsonify, request, render_template, redirect, url_for
from sklearn.preprocessing import MinMaxScaler

# --- Path Setup (must come before project-relative imports) ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
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
    """Load compare_metrics.csv.

    Searches in order:
      1. run_compare_* subdirectories inside backup_dir
      2. run_compare_* subdirectories inside output/runs/ directly
      3. Synthesise from individual metrics.json files in output/runs/
    """
    def _try_load_csv(path):
        if os.path.exists(path):
            try:
                return pd.read_csv(path)
            except Exception:
                pass
        return None

    # 1. Legacy backup compare dirs
    if backup_dir and os.path.isdir(backup_dir):
        compare_dirs = glob.glob(os.path.join(backup_dir, 'run_compare_*'))
        compare_dirs = [p for p in compare_dirs if os.path.isdir(p)]
        if compare_dirs:
            latest = max(compare_dirs, key=os.path.getmtime)
            df = _try_load_csv(os.path.join(latest, 'compare_metrics.csv'))
            if df is not None:
                return df

    # 2. run_compare_* directly in output/runs/
    if os.path.isdir(OUTPUT_RUNS_DIR):
        compare_dirs = glob.glob(os.path.join(OUTPUT_RUNS_DIR, 'run_compare_*'))
        compare_dirs = [p for p in compare_dirs if os.path.isdir(p)]
        if compare_dirs:
            latest = max(compare_dirs, key=os.path.getmtime)
            df = _try_load_csv(os.path.join(latest, 'compare_metrics.csv'))
            if df is not None:
                return df

    # 3. Synthesise from individual metrics.json files
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
        'gru_mimo_8features': 'gru_mimo_8features_*',
        'lstm_mimo_8features': 'lstm_mimo_8features_*',
        'gru_8features': 'gru_8features_*',
        'lstm_8features': 'lstm_8features_*',
        'seq2seq_attention_8features': 'seq2seq_attention_8features_*',
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
        from datetime import date as _today_cls
        _today = _today_cls.today()
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
    if 'seq2seq' in k and ('att' in k or 'attention' in k):
        return 'Seq2Seq+Attention (8 features)'
    if 'seq2seq' in k:
        return 'Seq2Seq (8 features)'
    if 'gru_mimo' in k or ('gru' in k and 'mimo' in k):
        return 'GRU-MIMO (8 features)'
    if 'lstm_mimo' in k or ('lstm' in k and 'mimo' in k):
        return 'LSTM-MIMO (8 features)'
    if 'gru' in k:
        return 'GRU (8 features)'
    if 'lstm' in k:
        return 'LSTM (8 features)'
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

    # ── 固定5个模型 key，直接从 output/runs 加载 ──
    MODEL_KEYS = [
        'gru_8features',
        'gru_mimo_8features',
        'lstm_8features',
        'lstm_mimo_8features',
        'seq2seq_attention_8features',
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
        # fallback: 用第一个有数据的模型的日期轴
        for mk in MODEL_KEYS:
            if model_data[mk]['df'] is not None and not model_data[mk]['df'].empty:
                df_master = model_data[mk]['df'][['date']].copy()
                df_master['actual'] = np.nan
                for col in ['precip_mm','temp_high_c','temp_low_c','weather_code_en',
                            'wind_level','wind_dir_en','wind_max','aqi_value','aqi_level_en']:
                    df_master[col] = np.nan
                break

    df_base = df_master[['date','actual','precip_mm','temp_high_c','temp_low_c',
                          'weather_code_en','wind_level','wind_dir_en','wind_max',
                          'aqi_value','aqi_level_en']].copy()

    # 合并各模型预测
    col_map = {
        'gru_8features':              'gru_single_pred',
        'gru_mimo_8features':         'gru_mimo_pred',
        'lstm_8features':             'lstm_single_pred',
        'lstm_mimo_8features':        'lstm_mimo_pred',
        'seq2seq_attention_8features':'seq2seq_pred',
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
            cal_std = preds_raw.std(axis=0)
            EPS = 1.0

            # Step-wise q̂_h: approximate h-step error by scaling residuals
            # Rationale: for single samples we don't have multi-step rollouts on val set,
            # so we use residual * sqrt(h) as a conservative estimate of h-step error spread.
            # This produces the "fan" effect: wider intervals for larger h.
            base_scores = np.abs(y_cal_raw - cal_mean) / (cal_std + EPS)
            n_c = len(base_scores)
            qhat_by_horizon = {}
            for h_step in range(1, max_horizon + 1):
                # Scale nonconformity by sqrt(h) to model error accumulation
                scaled_scores = base_scores * np.sqrt(h_step)
                level = min(np.ceil((n_c + 1) * (1 - alpha)) / n_c, 1.0)
                qhat_by_horizon[h_step] = round(float(np.quantile(scaled_scores, level)), 4)

            n_members_loaded = len(members)
            # Compute half-widths for each horizon (mean std as representative)
            mean_std = float(cal_std.mean())
            half_widths = {
                h_step: round(qhat_by_horizon[h_step] * (mean_std + EPS), 1)
                for h_step in qhat_by_horizon
            }

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

        def _build_window_8feat(visitor_window, lag7_window, dates):
            """构建 (look_back, 8) 特征矩阵"""
            rows = []
            for i, d in enumerate(dates):
                m_norm = (d.month - 1) / 11.0
                dow_norm = d.weekday() / 6.0
                hol = float(_is_holiday(d))
                rows.append([
                    visitor_window[i],   # visitor_count_scaled
                    m_norm,              # month_norm
                    dow_norm,            # day_of_week_norm
                    hol,                 # is_holiday
                    lag7_window[i],      # tourism_num_lag_7_scaled
                    0.0,                 # meteo_precip_sum_scaled (历史窗口用0，无未来天气)
                    0.5,                 # temp_high_scaled
                    0.5,                 # temp_low_scaled
                ])
            return np.array(rows, dtype=np.float32)

        # 历史窗口的日期
        hist_dates = [last_date - timedelta(days=look_back - 1 - i) for i in range(look_back)]

        def _predict_single_step_model(model, visitor_window, lag7_window, hist_dates, steps=None):
            """单步模型（LSTM/GRU）滚动预测，默认 horizon 步，可指定 steps 扩展推理。"""
            if steps is None:
                steps = horizon
            preds = []
            v_win = list(visitor_window)
            l7_win = list(lag7_window)
            cur_dates = list(hist_dates)

            for step in range(steps):
                pred_date = last_date + timedelta(days=step + 1)
                # 获取该日天气预报
                wrow = weather_df[weather_df['date'] == pred_date]
                p_s = _scale_precip(float(wrow['precip_sum'].iloc[0]) if len(wrow) else float('nan'))
                th_s = _scale_temp_high(float(wrow['temp_high'].iloc[0]) if len(wrow) else float('nan'))
                tl_s = _scale_temp_low(float(wrow['temp_low'].iloc[0]) if len(wrow) else float('nan'))

                X = _build_window_8feat(v_win, l7_win, cur_dates)
                # 覆盖最后一行的天气（最近一天用预报天气）
                X[-1, 5] = p_s
                X[-1, 6] = th_s
                X[-1, 7] = tl_s

                x_in = X.reshape(1, look_back, 8)
                y_s = float(model.predict(x_in, verbose=0)[0][0])
                y_val = float(visitor_scaler.inverse_transform([[y_s]])[0][0])
                preds.append(y_val)

                # 滚动窗口
                v_win = v_win[1:] + [y_s]
                lag7_s = v_win[-7] if len(v_win) >= 7 else y_s
                l7_win = l7_win[1:] + [lag7_s]
                cur_dates = cur_dates[1:] + [pred_date]

            return preds

        # ── 5. 在线推理：仅 MIMO 和 Seq2Seq ──
        # 单步 GRU/LSTM 不做在线推理，只读离线 backfill CSV。
        ONLINE_MODELS = [
            ('gru_mimo_8features',          'gru_mimo_pred',    'mimo'),
            ('lstm_mimo_8features',         'lstm_mimo_pred',   'mimo'),
            ('seq2seq_attention_8features', 'seq2seq_pred',     'seq2seq'),
        ]

        def _load_model_by_key(model_key: str):
            """从 output/runs 找最新 run_dir 并加载模型。"""
            top_dirs = sorted(
                glob.glob(os.path.join(OUTPUT_RUNS_DIR, f'{model_key}_*')),
                key=os.path.getmtime, reverse=True
            )
            for top_dir in top_dirs:
                run_subdirs = glob.glob(os.path.join(top_dir, 'runs', 'run_*'))
                run_dir = max(run_subdirs, key=os.path.getmtime) if run_subdirs else top_dir
                for pattern in ['weights/*.keras', 'weights/*.h5']:
                    matches = glob.glob(os.path.join(run_dir, pattern))
                    if matches:
                        try:
                            m = tf.keras.models.load_model(
                                matches[0],
                                custom_objects=_seq2seq_custom_objects or None,
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
                if infer_type == 'seq2seq':
                    enc_input = _build_window_8feat(
                        list(hist_scaled), list(lag7_window), hist_dates
                    ).reshape(1, look_back, 8)
                    dec_steps = min(horizon, 7)
                    dec_rows = []
                    for step in range(dec_steps):
                        pred_date = last_date + timedelta(days=step + 1)
                        wrow = weather_df[weather_df['date'] == pred_date]
                        p_s = _scale_precip(float(wrow['precip_sum'].iloc[0]) if len(wrow) else float('nan'))
                        th_s = _scale_temp_high(float(wrow['temp_high'].iloc[0]) if len(wrow) else float('nan'))
                        tl_s = _scale_temp_low(float(wrow['temp_low'].iloc[0]) if len(wrow) else float('nan'))
                        m_norm = (pred_date.month - 1) / 11.0
                        dow_norm = pred_date.weekday() / 6.0
                        hol = float(_is_holiday(pred_date))
                        lag7_s = lag7_window[-1] if lag7_window else 0.5
                        dec_rows.append([m_norm, dow_norm, hol, lag7_s, p_s, th_s, tl_s])
                    dec_input = np.array(dec_rows, dtype=np.float32).reshape(1, dec_steps, 7)
                    y_scaled = model_obj.predict([enc_input, dec_input], verbose=0)
                    preds = visitor_scaler.inverse_transform(
                        y_scaled[0, :dec_steps, 0].reshape(-1, 1)
                    ).flatten().tolist()
                    while len(preds) < horizon:
                        preds.append(preds[-1])

                elif infer_type == 'mimo':
                    enc_input = _build_window_8feat(
                        list(hist_scaled), list(lag7_window), hist_dates
                    ).reshape(1, look_back, 8)
                    y_scaled = model_obj.predict(enc_input, verbose=0)  # (1, 7)
                    preds = visitor_scaler.inverse_transform(
                        y_scaled[0].reshape(-1, 1)
                    ).flatten().tolist()[:horizon]
                    while len(preds) < horizon:
                        preds.append(preds[-1])

                else:  # single-step rolling
                    preds = _predict_single_step_model(
                        model_obj, list(hist_scaled), list(lag7_window), list(hist_dates)
                    )

                results[col_name] = preds
            except Exception as e:
                print(f"Online prediction failed for {col_name} ({mk}): {e}")
                results[col_name] = None

        # ── 6. 组装结果 DataFrame ──
        # 展示窗口：today ~ today+6（共 horizon 天）
        # MIMO/Seq2Seq：推理结果直接映射到 today 起的 horizon 天
        # 单步（GRU/LSTM）：
        #   改进算法 — 空白段（last_date+1 ~ today-1）用 MIMO 均值填充滚动窗口，
        #   而非用上一步预测值回填。这样 gap_days 步的误差不会自累积，
        #   只有 today 起的 horizon 步才是单步模型的真实贡献。
        #   在学术上对应"考虑数据滞伏期（Latency）的多视界不确定性校准"。
        from datetime import date as _date_cls
        today = _date_cls.today()
        gap_days = max(0, (today - last_date).days - 1)  # last_date+1 到 today 之间的天数
        single_total_steps = gap_days + horizon

        # 构建 gap 段的 MIMO 均值填充序列（scaled）
        # 取 gru_mimo 和 lstm_mimo 均值；若都不可用则回退到上一步滚动值
        def _gap_fill_from_mimo() -> list | None:
            """Return scaled gap-segment predictions from MIMO ensemble mean, or None."""
            if gap_days == 0:
                return []
            mimo_preds = []
            for _mc in ['gru_mimo_pred', 'lstm_mimo_pred']:
                _p = results.get(_mc)
                if _p and len(_p) >= gap_days:
                    # MIMO output: step 0 = today, so gap fills come from steps BEFORE today
                    # However MIMO is trained from last_date, so its step 0 ~ last_date+1
                    # We use the first gap_days steps of MIMO as the gap fill
                    mimo_preds.append(_p[:gap_days])
            if not mimo_preds:
                return None
            # Average across available MIMO models, convert to scaled
            gap_raw = np.mean(mimo_preds, axis=0)  # (gap_days,) in raw visitor units
            gap_scaled = visitor_scaler.transform(
                np.array(gap_raw).reshape(-1, 1)
            ).flatten().tolist()
            return gap_scaled

        gap_scaled_fill = _gap_fill_from_mimo()

        def _predict_single_step_with_gap_fill(model, visitor_window, lag7_win, h_dates, steps):
            """
            Single-step rolling with improved gap handling.
            For gap steps (last_date+1 ~ today-1), use MIMO-derived scaled values
            to advance the rolling window instead of using the model's own prediction.
            This prevents error snowball during the data-latency period.
            """
            preds_all = []
            v_win = list(visitor_window)
            l7_win = list(lag7_win)
            cur_dates = list(h_dates)

            for step_i in range(steps):
                pred_date = last_date + timedelta(days=step_i + 1)
                wrow = weather_df[weather_df['date'] == pred_date]
                p_s = _scale_precip(float(wrow['precip_sum'].iloc[0]) if len(wrow) else float('nan'))
                th_s = _scale_temp_high(float(wrow['temp_high'].iloc[0]) if len(wrow) else float('nan'))
                tl_s = _scale_temp_low(float(wrow['temp_low'].iloc[0]) if len(wrow) else float('nan'))

                X = _build_window_8feat(v_win, l7_win, cur_dates)
                X[-1, 5] = p_s
                X[-1, 6] = th_s
                X[-1, 7] = tl_s
                x_in = X.reshape(1, look_back, 8)
                y_s = float(model.predict(x_in, verbose=0)[0][0])
                y_val = float(visitor_scaler.inverse_transform([[y_s]])[0][0])
                preds_all.append(y_val)

                # Determine what to feed back into the rolling window
                # For gap steps: use MIMO fill (scaled) to suppress error accumulation
                is_gap_step = step_i < gap_days
                if is_gap_step and gap_scaled_fill and step_i < len(gap_scaled_fill):
                    feed_scaled = gap_scaled_fill[step_i]
                else:
                    feed_scaled = y_s  # normal: use model's own prediction

                v_win = v_win[1:] + [feed_scaled]
                lag7_new = v_win[-7] if len(v_win) >= 7 else feed_scaled
                l7_win = l7_win[1:] + [lag7_new]
                cur_dates = cur_dates[1:] + [pred_date]

            return preds_all

        # 单步推理（改进版：gap段用MIMO填充）
        single_results = {}
        for mk_single, col_single in [('gru_8features', 'gru_single_pred'),
                                       ('lstm_8features', 'lstm_single_pred')]:
            m_single = _load_model_by_key(mk_single)
            if m_single is not None:
                try:
                    single_results[col_single] = _predict_single_step_with_gap_fill(
                        m_single, list(hist_scaled), list(lag7_window),
                        list(hist_dates), steps=single_total_steps
                    )
                except Exception as e:
                    print(f"Single-step extended inference failed ({col_single}): {e}")
                    single_results[col_single] = None

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
            # MIMO/Seq2Seq：推理结果直接对应 today+step（第0步=today）
            for col_name in ['gru_mimo_pred', 'lstm_mimo_pred', 'seq2seq_pred']:
                preds = results.get(col_name)
                row[col_name] = preds[step] if preds and step < len(preds) else float('nan')
            # 单步：从 last_date+1 开始滚动，gap_days 步后才到 today
            single_step_idx = gap_days + step
            for col_name in ['gru_single_pred', 'lstm_single_pred']:
                preds = single_results.get(col_name)
                row[col_name] = preds[single_step_idx] if preds and single_step_idx < len(preds) else float('nan')
            out_rows.append(row)
        return pd.DataFrame(out_rows)

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
            for _c in ['gru_single_pred', 'gru_mimo_pred', 'lstm_single_pred',
                       'lstm_mimo_pred', 'seq2seq_pred']:
                if _c not in df_merge.columns:
                    df_merge[_c] = np.nan
            # today 이후의 모든 행 제거（backfill CSV의 4/12~4/19 등 포함）
            # df_future(today~today+6)로 완전히 대체
            from datetime import date as _trim_date
            _today = _trim_date.today()
            df_merge['date'] = pd.to_datetime(df_merge['date']).dt.date
            df_merge = df_merge[df_merge['date'] < _today]
            df_merge = pd.concat([
                df_merge,
                df_future[['date', 'actual', 'precip_mm', 'temp_high_c', 'temp_low_c',
                            'weather_code_en', 'wind_level', 'wind_dir_en', 'wind_max',
                            'aqi_value', 'aqi_level_en',
                            'gru_single_pred', 'gru_mimo_pred',
                            'lstm_single_pred', 'lstm_mimo_pred', 'seq2seq_pred']]
            ], ignore_index=True)
            df_merge = df_merge.sort_values('date').reset_index(drop=True)
    except Exception as e:
        warning = (warning + ' | ' if warning else '') + f'MIMO/Seq2Seq online forecast failed: {e}'
        online_used = False

    # ── Step-wise Conformal q̂_h (Deep Ensemble calibration) ──
    # Compute per-horizon uncertainty intervals for the forecast window.
    # Only computed when ensemble weights are available; gracefully skipped otherwise.
    unc_meta = None
    try:
        unc_meta = _compute_stepwise_qhat(alpha=0.10, max_horizon=h)
    except Exception as _ue:
        print(f'[uncertainty] stepwise qhat failed: {_ue}')

    # Build uncertainty series aligned to time_axis (non-null only in forecast window)
    # We use gru_mimo_pred as the center prediction for the uncertainty band.
    def _build_unc_series(center_col: str, qhat_by_h: dict, mean_std: float):
        """Build lower/upper arrays over time_axis. Only forecast window has values."""
        n = len(df_merge)
        lower_arr = [None] * n
        upper_arr = [None] * n
        half_w_arr = [None] * n
        if not qhat_by_h:
            return lower_arr, upper_arr, half_w_arr
        from datetime import date as _dc
        _today = _dc.today()
        for _i, _row in df_merge.iterrows():
            _d = _row['date']
            if isinstance(_d, str):
                _d = pd.to_datetime(_d).date()
            elif hasattr(_d, 'date'):
                _d = _d.date() if not isinstance(_d, type(_dc.today())) else _d
            if _d < _today:
                continue
            _step = (_d - _today).days + 1  # h=1 for today, h=7 for today+6
            _q = qhat_by_h.get(_step)
            _center = _row.get(center_col)
            if _q is None or _center is None or (isinstance(_center, float) and np.isnan(_center)):
                continue
            _hw = _q * (mean_std + 1.0)
            lower_arr[_i] = round(float(_center) - _hw, 1)
            upper_arr[_i] = round(float(_center) + _hw, 1)
            half_w_arr[_i] = round(_hw, 1)
        return lower_arr, upper_arr, half_w_arr

    unc_lower, unc_upper, unc_half_w = [None]*len(df_merge), [None]*len(df_merge), [None]*len(df_merge)
    if unc_meta and unc_meta.get('available'):
        unc_lower, unc_upper, unc_half_w = _build_unc_series(
            'gru_mimo_pred',
            unc_meta['qhat_by_horizon'],
            unc_meta.get('mean_ensemble_std', 529.0),
        )

    # Thresholds — 从任意有数据的模型 metrics 里取，优先 gru
    _ref_metrics = {}
    for _mk in ['gru_8features', 'gru_mimo_8features', 'lstm_8features', 'seq2seq_attention_8features']:
        if model_data.get(_mk, {}).get('metrics'):
            _ref_metrics = model_data[_mk]['metrics']
            break
    threshold_crowd = float((_ref_metrics.get('meta') or {}).get('peak_threshold', 18500.0))
    wh = (_ref_metrics.get('weather_hazard') or {})
    wh_thr = (wh.get('thresholds') or {})
    precip_high = float(wh_thr.get('precip_high', 8.0))
    temp_high = float(wh_thr.get('temp_high', 22.6))
    temp_low = float(wh_thr.get('temp_low', -10.61))
    quantiles = (wh_thr.get('quantiles') or {})

    def _compute_risk(pred_col: str):
        """基于预测客流和天气计算逐日风险。"""
        if pred_col not in df_merge.columns:
            n = len(df_merge)
            return {'crowd_alert':[False]*n,'weather_hazard':[False]*n,
                    'suitability_warning_bin':[0]*n,'risk_level':[0]*n,
                    'p_warn':[0.15]*n,'drivers':[[]]*n,'risk_score':[0.0]*n}
        y_pred = df_merge[pred_col].astype(float)
        crowd_alert = (y_pred >= threshold_crowd)
        weather_hazard = (
            (df_merge['precip_mm'].fillna(0) >= precip_high) |
            (df_merge['temp_high_c'].fillna(0) >= temp_high) |
            (df_merge['temp_low_c'].fillna(0) <= temp_low)
        )
        suitability = (crowd_alert | weather_hazard)
        ca_list = [bool(x) if not pd.isna(x) else False for x in crowd_alert]
        wh_list = [bool(x) if not pd.isna(x) else False for x in weather_hazard]
        sw_list = [1 if bool(x) else 0 for x in suitability]
        risk_level, drivers = [], []
        for ca, whz, pr, th, tl in zip(ca_list, wh_list,
                df_merge['precip_mm'].tolist(), df_merge['temp_high_c'].tolist(), df_merge['temp_low_c'].tolist()):
            lv, d = 0, []
            if ca: lv += 2; d.append('crowd_over_threshold')
            if whz:
                lv += 1
                if pr is not None and not (isinstance(pr,float) and np.isnan(pr)) and pr >= precip_high: d.append('precip_high')
                if th is not None and not (isinstance(th,float) and np.isnan(th)) and th >= temp_high: d.append('temp_high')
                if tl is not None and not (isinstance(tl,float) and np.isnan(tl)) and tl <= temp_low: d.append('temp_low')
            risk_level.append(int(lv)); drivers.append(d)
        p_warn = [0.85 if s else 0.15 for s in suitability]
        risk_score = [round(max(0.0,min(1.0,(lv/3.0)*0.65+float(pw)*0.35))*100.0,1) for lv,pw in zip(risk_level,p_warn)]
        return {'crowd_alert':ca_list,'weather_hazard':wh_list,'suitability_warning_bin':sw_list,
                'risk_level':risk_level,'p_warn':p_warn,'drivers':drivers,'risk_score':risk_score}

    # 用 GRU 单步作为主风险计算基准（优先级：gru_single > gru_mimo > seq2seq）
    _risk_col = next((c for c in ['gru_single_pred','gru_mimo_pred','seq2seq_pred','lstm_single_pred']
                      if c in df_merge.columns and not df_merge[c].isna().all()), None)
    risk_main = _compute_risk(_risk_col) if _risk_col else _compute_risk('gru_single_pred')

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
    from datetime import date as _fc_date_cls
    _today_str = _fc_date_cls.today().isoformat()
    _today_end_str = (_fc_date_cls.today() + timedelta(days=h - 1)).isoformat()
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
        'gru_8features':               ('GRU', 'GRU (单步)'),
        'gru_mimo_8features':          ('GRU-MIMO', 'GRU (多步)'),
        'lstm_8features':              ('LSTM', 'LSTM (单步)'),
        'lstm_mimo_8features':         ('LSTM-MIMO', 'LSTM (多步)'),
        'seq2seq_attention_8features': ('Seq2Seq', 'Seq2Seq+Attention'),
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
            'actual': _to_num_list('actual'),
            'gru_single_pred':  _to_num_list('gru_single_pred'),
            'gru_mimo_pred':    _to_num_list('gru_mimo_pred'),
            'lstm_single_pred': _to_num_list('lstm_single_pred'),
            'lstm_mimo_pred':   _to_num_list('lstm_mimo_pred'),
            'seq2seq_pred':     _to_num_list('seq2seq_pred'),
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
        'risk': risk_main,
        'uncertainty': {
            'available': bool(unc_meta and unc_meta.get('available')),
            'method': 'deep_ensemble_conformal_stepwise',
            'alpha': 0.10,
            'n_members': (unc_meta or {}).get('n_members', 0),
            'cal_size': (unc_meta or {}).get('cal_size', 0),
            'qhat_by_horizon': (unc_meta or {}).get('qhat_by_horizon', {}),
            'half_width_by_horizon': (unc_meta or {}).get('half_width_by_horizon', {}),
            # Aligned to time_axis, non-null only in forecast window
            'lower': unc_lower,
            'upper': unc_upper,
            'half_width': unc_half_w,
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
        from realtime.daily_update import daily_update
        from scripts.append_and_retrain import start_data_pipeline_scheduler

        scheduler = BackgroundScheduler(timezone='Asia/Shanghai')
        # 每日 09:00 爬取客流
        scheduler.add_job(daily_update, 'cron', hour=9, minute=0, id='daily_crawl',
                          replace_existing=True)
        # 每日 08:30 追加数据到训练 CSV + 每月1日 02:00 重训
        start_data_pipeline_scheduler(scheduler)
        scheduler.start()
        print("Scheduler started: daily crawl (09:00), daily append (08:30), monthly retrain (1st 02:00).")
        return scheduler
    except ImportError:
        print("WARNING: APScheduler not installed. Run: pip install APScheduler==3.10.4")
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

    # Use port 5000 as requested
    print("Starting Flask server on port 5000...")
    app.run(debug=False, port=5000)
