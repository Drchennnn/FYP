"""生成 output/analysis_plots/ 下的补充可视化图表。

图表清单：
  07a_mae.png                  — MAE 三模型+集成对比
  07b_nrmse.png                — NRMSE 三模型+集成对比
  07c_f1.png                   — Crowd Alert F1 三模型+集成对比
  07d_recall.png               — Crowd Alert Recall 三模型+集成对比
  08_xgb_feature_importance.png — XGBoost 特征重要性
  09_gru_loss_curve.png        — GRU 训练 Loss 曲线
  10_transformer_loss_curve.png — Transformer 训练 Loss 曲线（需重训后有 history CSV）
"""

import os
import json
import glob

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.metrics import confusion_matrix

# ── 路径 ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS_DIR = os.path.join(BASE_DIR, 'output', 'runs')
OUT_DIR  = os.path.join(BASE_DIR, 'output', 'analysis_plots')
os.makedirs(OUT_DIR, exist_ok=True)

# ── 样式 ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'figure.dpi': 150,
})

MODEL_DISPLAY = {
    'gru_8features':         'GRU',
    'transformer_8features': 'Transformer',
    'xgboost_8features':     'XGBoost',
}
# 集成权重（与 app.py 保持一致）
ENS_WEIGHTS = {'gru': 0.10, 'transformer': 0.20, 'xgboost': 0.70}

COLORS = {
    'GRU':       '#0a84ff',
    'Transformer':'#30d158',
    'XGBoost':   '#ffd60a',
    'Ensemble':  '#bf5af2',
}

# 旺淡季动态阈值
def _seasonal_threshold(date_val):
    m, d = date_val.month, date_val.day
    if (m > 4 or (m == 4 and d >= 1)) and (m < 11 or (m == 11 and d <= 15)):
        return 32800
    return 18400

# ── 工具函数 ──────────────────────────────────────────────────────────────
def _find_latest_run_dir(model_key: str):
    top_dirs = sorted(
        glob.glob(os.path.join(RUNS_DIR, f'{model_key}_*')),
        key=os.path.getmtime, reverse=True
    )
    for top in top_dirs:
        subs = glob.glob(os.path.join(top, 'runs', 'run_*'))
        run_dir = max(subs, key=os.path.getmtime) if subs else top
        if os.path.exists(os.path.join(run_dir, 'metrics.json')):
            return run_dir
    return None


def _load_metrics(run_dir: str):
    path = os.path.join(run_dir, 'metrics.json')
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _load_pred_csv(run_dir: str):
    """Load test predictions CSV, return DataFrame with date/y_true/y_pred."""
    for pat in ['*_test_predictions.csv']:
        files = glob.glob(os.path.join(run_dir, pat))
        if files:
            df = pd.read_csv(files[0])
            df['date'] = pd.to_datetime(df['date'])
            return df
    return None


def _compute_ensemble_metrics():
    """计算集成预测在三模型共有测试集日期上的指标。"""
    dfs = {}
    for mk, label in MODEL_DISPLAY.items():
        run_dir = _find_latest_run_dir(mk)
        if not run_dir:
            continue
        df = _load_pred_csv(run_dir)
        if df is None:
            continue
        df = df[['date', 'y_true', 'y_pred']].rename(columns={'y_pred': label})
        dfs[label] = df

    if len(dfs) < 2:
        return None

    # 取三模型共有日期（内连接）
    merged = None
    for label, df in dfs.items():
        if merged is None:
            merged = df
        else:
            merged = merged.merge(df[['date', label]], on='date', how='inner')

    if merged is None or merged.empty:
        return None

    # 集成预测（加权）
    total_w = sum(ENS_WEIGHTS.values())
    merged['Ensemble'] = sum(
        merged[label] * (ENS_WEIGHTS[label.lower()] / total_w)
        for label in MODEL_DISPLAY.values()
        if label in merged.columns
    )

    y_true = merged['y_true'].values
    y_ens  = merged['Ensemble'].values
    dates  = merged['date']

    # NRMSE（用 y_true 的范围归一化）
    rmse = np.sqrt(np.mean((y_true - y_ens) ** 2))
    nrmse = rmse / (y_true.max() - y_true.min()) if (y_true.max() - y_true.min()) > 0 else np.nan

    # MAE
    mae = np.mean(np.abs(y_true - y_ens))

    # Crowd Alert（动态阈值）
    thresholds = dates.apply(_seasonal_threshold).values
    actual_pos = (y_true >= thresholds)
    pred_pos   = (y_ens  >= thresholds)
    tp = np.sum(actual_pos & pred_pos)
    fp = np.sum(~actual_pos & pred_pos)
    fn = np.sum(actual_pos & ~pred_pos)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {'MAE': mae, 'NRMSE': nrmse, 'F1': f1, 'Recall': recall,
            'n_samples': len(merged), 'tp': int(tp), 'fp': int(fp), 'fn': int(fn)}


def _collect_all_metrics():
    """收集三单模型 + 集成的指标，返回 dict {label: {MAE, NRMSE, F1, Recall}}。"""
    result = {}
    for mk, label in MODEL_DISPLAY.items():
        run_dir = _find_latest_run_dir(mk)
        if not run_dir:
            continue
        m = _load_metrics(run_dir)
        reg = m.get('regression', {})
        sw  = m.get('suitability_warning', {})
        result[label] = {
            'MAE':    reg.get('mae',   float('nan')),
            'NRMSE':  reg.get('nrmse', float('nan')),
            'F1':     sw.get('f1',     float('nan')),
            'Recall': sw.get('recall', float('nan')),
        }

    ens = _compute_ensemble_metrics()
    if ens:
        result['Ensemble'] = {
            'MAE':    ens['MAE'],
            'NRMSE':  ens['NRMSE'],
            'F1':     ens['F1'],
            'Recall': ens['Recall'],
        }
    return result


# ── 单指标图（通用）────────────────────────────────────────────────────────
def _plot_single_metric(all_metrics, metric_key, title, ylabel, is_pct,
                        filename, lower_better=True, subtitle=None):
    labels = list(all_metrics.keys())
    vals   = [all_metrics[l][metric_key] for l in labels]
    colors = [COLORS.get(l, '#888') for l in labels]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, vals, color=colors, alpha=0.88, width=0.5, edgecolor='none')

    ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
    if subtitle:
        ax.text(0.5, 1.01, subtitle, transform=ax.transAxes,
                ha='center', va='bottom', fontsize=8, color='#888')
    ax.set_ylabel(ylabel, fontsize=10)

    if is_pct:
        ax.set_ylim(0, 1.18)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    else:
        ax.set_ylim(0, max(v for v in vals if not np.isnan(v)) * 1.25)

    # 数值标注
    for bar, v in zip(bars, vals):
        if np.isnan(v):
            continue
        label_txt = f'{v:.1%}' if is_pct else f'{v:,.0f}' if v > 100 else f'{v:.4f}'
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (0.015 if is_pct else max(v for v in vals if not np.isnan(v)) * 0.02),
                label_txt, ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 高亮最优（黑边）
    clean = [(i, v) for i, v in enumerate(vals) if not np.isnan(v)]
    if clean:
        best_i = min(clean, key=lambda x: x[1])[0] if lower_better else max(clean, key=lambda x: x[1])[0]
        bars[best_i].set_edgecolor('black')
        bars[best_i].set_linewidth(2.0)

    fig.tight_layout()
    out = os.path.join(OUT_DIR, filename)
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'[OK] {out}')


# ── 图 07a~d：各指标独立图 ────────────────────────────────────────────────
def plot_individual_metrics():
    all_metrics = _collect_all_metrics()
    if not all_metrics:
        print('[SKIP] No metrics found')
        return

    _plot_single_metric(all_metrics, 'MAE',
        title='MAE by Model (Test Set)',
        ylabel='Visitors', is_pct=False,
        filename='07a_mae.png', lower_better=True)

    _plot_single_metric(all_metrics, 'NRMSE',
        title='NRMSE by Model (Test Set)',
        ylabel='NRMSE (range-normalized)',
        subtitle='RMSE / (y_max - y_min) on test set',
        is_pct=False, filename='07b_nrmse.png', lower_better=True)

    _plot_single_metric(all_metrics, 'F1',
        title='Crowd Alert F1 by Model',
        ylabel='F1 Score', is_pct=True,
        subtitle='Crowd Alert: seasonal threshold (32800 peak / 18400 off-peak)',
        filename='07c_f1.png', lower_better=False)

    _plot_single_metric(all_metrics, 'Recall',
        title='Crowd Alert Recall by Model',
        ylabel='Recall', is_pct=True,
        subtitle='Crowd Alert Recall — fraction of true overload days correctly flagged',
        filename='07d_recall.png', lower_better=False)


# ── 图 08：XGBoost 特征重要性 ─────────────────────────────────────────────
def plot_xgb_feature_importance():
    run_dir = _find_latest_run_dir('xgboost_8features')
    if not run_dir:
        print('[SKIP] XGBoost run_dir not found')
        return

    model_path = os.path.join(run_dir, 'weights', 'xgboost_model.json')
    if not os.path.exists(model_path):
        print(f'[SKIP] {model_path} not found')
        return

    try:
        import xgboost as xgb
    except ImportError:
        print('[SKIP] xgboost not installed')
        return

    model = xgb.XGBRegressor()
    model.load_model(model_path)

    importance = model.feature_importances_
    names = list(model.feature_names_in_) if hasattr(model, 'feature_names_in_') else [f'f{i}' for i in range(len(importance))]

    label_map = {
        'month_norm':                 'Month',
        'day_of_week_norm':           'Weekday',
        'is_holiday':                 'Holiday',
        'is_peak_season':             'Peak Season',
        'days_to_next_holiday':       'Days→Holiday',
        'days_since_last_holiday':    'Days←Holiday',
        'tourism_num_lag_1_scaled':   'Lag-1 Visitors',
        'tourism_num_lag_7_scaled':   'Lag-7 Visitors',
        'tourism_num_lag_14_scaled':  'Lag-14 Visitors',
        'rolling_mean_7_scaled':      'Roll-7 Mean',
        'meteo_precip_sum_scaled':    'Precipitation',
        'temp_high_scaled':           'Temp High',
        'temp_low_scaled':            'Temp Low',
    }
    labels = [label_map.get(n, n) for n in names]
    order  = np.argsort(importance)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh([labels[i] for i in order], importance[order],
                   color='#ffd60a', alpha=0.88, edgecolor='none', height=0.65)

    for bar, v in zip(bars, importance[order]):
        ax.text(v + 0.003, bar.get_y() + bar.get_height() / 2,
                f'{v:.3f}', va='center', fontsize=8)

    ax.set_xlabel('Feature Importance (gain)', fontsize=10)
    ax.set_title('XGBoost Feature Importance', fontsize=13, fontweight='bold')
    ax.set_xlim(0, importance.max() * 1.18)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, '08_xgb_feature_importance.png')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'[OK] {out}')


# ── Loss 曲线（通用）─────────────────────────────────────────────────────
def _plot_loss_curve(history_path, title, out_filename):
    if not os.path.exists(history_path):
        print(f'[SKIP] {history_path} not found')
        return

    df = pd.read_csv(history_path)
    df.columns = [c.strip().lstrip('\ufeff') for c in df.columns]
    if 'epoch' not in df.columns:
        df.insert(0, 'epoch', range(1, len(df) + 1))

    epochs    = df['epoch'].values
    train_loss = df['loss'].values
    val_loss   = df['val_loss'].values

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, train_loss, label='Train Loss', color='#0a84ff', linewidth=1.8)
    ax.plot(epochs, val_loss,   label='Val Loss',   color='#ff9f0a', linewidth=1.8, linestyle='--')

    best_epoch = int(df['val_loss'].idxmin()) + 1
    best_val   = df['val_loss'].min()
    ax.axvline(best_epoch, color='#ff453a', linewidth=1, linestyle=':', alpha=0.7)
    ax.scatter([best_epoch], [best_val], color='#ff453a', zorder=5, s=50)
    ax.text(best_epoch + 0.5, best_val * 1.05,
            f'Best ep {best_epoch}\n(val={best_val:.4f})',
            fontsize=8, color='#ff453a')

    ax.set_xlabel('Epoch', fontsize=10)
    ax.set_ylabel('Loss (MSE scaled)', fontsize=10)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_xlim(1, len(epochs))
    fig.tight_layout()
    out = os.path.join(OUT_DIR, out_filename)
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'[OK] {out}')


# ── 图 09：GRU Loss ───────────────────────────────────────────────────────
def plot_gru_loss_curve():
    run_dir = _find_latest_run_dir('gru_8features')
    if not run_dir:
        print('[SKIP] GRU run_dir not found')
        return
    _plot_loss_curve(
        os.path.join(run_dir, 'gru_history.csv'),
        'GRU Training Loss Curve',
        '09_gru_loss_curve.png'
    )


# ── 图 10：Transformer Loss ───────────────────────────────────────────────
def plot_transformer_loss_curve():
    run_dir = _find_latest_run_dir('transformer_8features')
    if not run_dir:
        print('[SKIP] Transformer run_dir not found')
        return
    _plot_loss_curve(
        os.path.join(run_dir, 'transformer_history.csv'),
        'Transformer Training Loss Curve',
        '10_transformer_loss_curve.png'
    )


# ── 主入口 ────────────────────────────────────────────────────────────────
def plot_dataset_split():
    """生成数据集划分扇形图（11_dataset_split.png）。
    宽幅横版，适合前端展示；透明背景；左饼右文字布局。
    """
    sizes   = [80.0, 10.0, 10.0]
    colors  = ['#0a84ff', '#30d158', '#ff9f0a']
    explode = (0.0, 0.04, 0.07)

    fig = plt.figure(figsize=(11, 4.8), facecolor='none')
    # 左：饼图  右：文字说明
    ax_pie  = fig.add_axes([0.03, 0.08, 0.42, 0.84])
    ax_text = fig.add_axes([0.50, 0.0,  0.50, 1.0])
    ax_pie.set_facecolor('none')
    ax_text.set_facecolor('none')
    ax_text.axis('off')

    ax_pie.pie(
        sizes,
        colors=colors,
        explode=explode,
        startangle=90,
        counterclock=False,
        wedgeprops=dict(linewidth=2, edgecolor='white'),
        shadow=False,
    )

    # 中心标注
    ax_pie.text(0, 0, '1,050\nrows\ntotal', ha='center', va='center',
                fontsize=10, color='#3a3a3c', linespacing=1.6, fontweight='medium')

    ax_pie.set_title('Dataset Split  ·  Jiuzhaigou (2023–2026)',
                     color='#1d1d1f', fontsize=11, pad=14, fontweight='semibold')

    # 右侧文字块
    lines = [
        ('TRAIN  ~80%', colors[0],
         '2023-07-16 ~ 2025-09-28',
         '~805 sequences  ·  model weights optimised here'),
        ('VAL     ~10%', colors[1],
         '2025-09-29 ~ 2026-01-06',
         '~100 sequences  ·  hyperparameter tuning & Early Stopping'),
        ('TEST    ~10%', colors[2],
         '2026-01-07 ~ 2026-04-16',
         '~100 sequences  ·  all reported metrics evaluated here'),
    ]

    y = 0.82
    for label, color, date_range, desc in lines:
        # colour swatch
        ax_text.add_patch(plt.Rectangle((0.0, y - 0.025), 0.03, 0.07,
                                         fc=color, ec='none', transform=ax_text.transAxes,
                                         clip_on=False))
        ax_text.text(0.055, y + 0.02, label,
                     transform=ax_text.transAxes,
                     fontsize=11, fontweight='bold', color='#1d1d1f', va='center')
        ax_text.text(0.055, y - 0.06, date_range,
                     transform=ax_text.transAxes,
                     fontsize=9.5, color='#3a3a3c', va='center')
        ax_text.text(0.055, y - 0.14, desc,
                     transform=ax_text.transAxes,
                     fontsize=8.5, color='#6e6e73', va='center', style='italic')
        y -= 0.30

    # 底部注释
    ax_text.text(0.0, 0.04,
                 'Data filtered to date ≥ 2023-06-01 · 2017 earthquake & COVID closure days excluded',
                 transform=ax_text.transAxes,
                 fontsize=7.5, color='#aeaeb2', va='bottom')

    out = os.path.join(OUT_DIR, '11_dataset_split.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', transparent=True)
    plt.close()
    print(f'  saved {out}')


def plot_historical_events():
    """生成历史客流时序 + 事件标注图（12_historical_events.png）。
    展示完整原始数据（2016–2026），叠加地震/疫情/建模起点标注。
    透明背景，适配前端浅色/深色主题。
    """
    # ── 读取数据 ──────────────────────────────────────────────────────────
    data_dir = os.path.join(BASE_DIR, 'data', 'processed')
    csv_files = sorted(glob.glob(os.path.join(data_dir, '*.csv')))
    if not csv_files:
        print('  [SKIP] no processed CSV found')
        return
    df = pd.read_csv(csv_files[-1], parse_dates=['date'])
    df = df[['date', 'tourism_num']].dropna().sort_values('date').reset_index(drop=True)

    # 7日滚动均值平滑曲线
    df['smooth'] = df['tourism_num'].rolling(7, center=True, min_periods=1).mean()

    # ── 事件区间定义 ──────────────────────────────────────────────────────
    events = [
        # (start, end, color, alpha, label, label_y_frac)
        ('2017-08-08', '2019-09-30', '#ff453a', 0.13,
         '2017 Earthquake\n& Closure', 0.88),
        ('2020-01-23', '2020-04-11', '#ff9f0a', 0.15,
         'COVID-19\nLockdown 1', 0.72),
        ('2021-08-01', '2021-09-30', '#ff9f0a', 0.15,
         'COVID\nLockdown 2', 0.72),
        ('2022-03-01', '2022-06-30', '#ff9f0a', 0.15,
         'COVID\nLockdown 3', 0.72),
    ]
    # 建模起点
    model_start = pd.Timestamp('2023-06-01')

    # ── 绘图 ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(16, 5.2), facecolor='none')
    ax.set_facecolor('none')

    # 原始散点（细，低透明度）
    ax.plot(df['date'], df['tourism_num'] / 10000,
            color='#3a3a3c', linewidth=0.5, alpha=0.25, zorder=1)
    # 平滑曲线
    ax.plot(df['date'], df['smooth'] / 10000,
            color='#0a84ff', linewidth=1.8, alpha=0.9, zorder=2, label='7-day rolling mean')

    # 事件阴影带
    label_used = {'COVID': False}
    for start, end, color, alpha, label, y_frac in events:
        ts, te = pd.Timestamp(start), pd.Timestamp(end)
        ax.axvspan(ts, te, color=color, alpha=alpha, zorder=0, linewidth=0)
        # 标注文字（只在带内居中）
        mid = ts + (te - ts) / 2
        ymax = ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 4.5
        # 用 transform blended 让 y 以 axes 坐标定位
        ax.text(mid, y_frac, label,
                transform=ax.get_xaxis_transform(),
                ha='center', va='top', fontsize=7, color=color,
                fontweight='semibold', linespacing=1.4,
                bbox=dict(fc='none', ec='none', pad=1))

    # 建模起点竖线
    ax.axvline(model_start, color='#30d158', linewidth=1.6,
               linestyle='--', zorder=3, alpha=0.9)
    ax.text(model_start, 0.97, '  Modelling\n  start\n  2023-06-01',
            transform=ax.get_xaxis_transform(),
            ha='left', va='top', fontsize=7.5, color='#30d158', fontweight='semibold',
            linespacing=1.4)

    # 灰色填充"已剔除"区间（地震+疫情）的背景提示
    ax.fill_between(df['date'],
                    df['smooth'] / 10000,
                    where=df['date'] < model_start,
                    alpha=0.06, color='#ff453a', zorder=0)

    # 轴格式
    import matplotlib.dates as mdates
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))
    ax.set_ylabel('Visitors (×10,000)', fontsize=9, color='#3a3a3c')
    ax.tick_params(axis='both', labelsize=8, colors='#3a3a3c')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.1f}w'))
    for spine in ax.spines.values():
        spine.set_edgecolor('#d1d1d6')
    ax.grid(axis='y', alpha=0.2, color='#3a3a3c')
    ax.set_xlim(df['date'].iloc[0], df['date'].iloc[-1])
    ax.set_ylim(bottom=0)

    ax.set_title(
        'Jiuzhaigou Daily Visitor Flow  ·  2016–2026  '
        '(raw data, closure days removed)',
        color='#1d1d1f', fontsize=10.5, pad=12, fontweight='semibold')

    # 图例
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [
        Line2D([0], [0], color='#0a84ff', linewidth=1.8, label='7-day rolling mean'),
        Patch(fc='#ff453a', alpha=0.35, label='Earthquake / Major closure'),
        Patch(fc='#ff9f0a', alpha=0.35, label='COVID lockdown'),
        Line2D([0], [0], color='#30d158', linewidth=1.6,
               linestyle='--', label='Modelling start (2023-06-01)'),
    ]
    ax.legend(handles=handles, loc='lower right', fontsize=8,
              framealpha=0.6, facecolor='white', edgecolor='none',
              labelcolor='#3a3a3c', ncol=1,
              handlelength=1.8, borderpad=0.7, labelspacing=0.6)

    plt.tight_layout(pad=1.4)
    out = os.path.join(OUT_DIR, '12_historical_events.png')
    plt.savefig(out, dpi=180, bbox_inches='tight', transparent=True)
    plt.close()
    print(f'  saved {out}')


if __name__ == '__main__':
    print('Generating analysis plots...')
    plot_individual_metrics()
    plot_xgb_feature_importance()
    plot_gru_loss_curve()
    plot_transformer_loss_curve()
    plot_dataset_split()
    plot_historical_events()
    print('Done.')
