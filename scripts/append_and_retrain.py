#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据追加与月度重训管道

功能：
1. append_new_data()：将爬取的新客流数据 + Open-Meteo 天气追加到10年训练 CSV
2. backfill_real_data()：自动检测最后真实数据日期，批量补追到昨天
3. monthly_retrain()：触发三个模型的重训练（通过 run_pipeline.py）
4. APScheduler 集成：每日追加，每月1日重训

用法（手动）：
  python scripts/append_and_retrain.py --append              # 追加昨日数据
  python scripts/append_and_retrain.py --append --date YYYY-MM-DD  # 追加指定日期
  python scripts/append_and_retrain.py --backfill            # 自动补追所有空挡
  python scripts/append_and_retrain.py --backfill --date YYYY-MM-DD  # 补追到指定日期
  python scripts/append_and_retrain.py --retrain             # 仅重训三个模型
  python scripts/append_and_retrain.py --append --retrain    # 追加+重训

APScheduler 集成（在 web_app/app.py 中调用 start_data_pipeline_scheduler()）
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 10年训练数据文件（追加目标）
TRAIN_CSV = PROJECT_ROOT / 'data' / 'processed' / 'jiuzhaigou_daily_features_2016-01-01_2026-04-02.csv'
# 追加日志
APPEND_LOG = PROJECT_ROOT / 'data' / 'append_log.json'

# Open-Meteo 九寨沟坐标
LAT, LON = 33.2, 103.9


# ─────────────────────────────────────────────
# 天气获取（历史存档）
# ─────────────────────────────────────────────

def _fetch_meteo_for_date(target_date: date) -> dict:
    """从 Open-Meteo 历史存档获取指定日期的天气数据。"""
    date_str = target_date.isoformat()
    try:
        url = 'https://archive-api.open-meteo.com/v1/archive'
        params = {
            'latitude': LAT, 'longitude': LON,
            'start_date': date_str, 'end_date': date_str,
            'daily': 'temperature_2m_max,temperature_2m_min,precipitation_sum,'
                     'weathercode,windspeed_10m_max',
            'timezone': 'Asia/Shanghai',
        }
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        d = r.json().get('daily', {})
        return {
            'meteo_weather_code': int(d.get('weathercode', [None])[0] or 0),
            'meteo_temp_max': float(d.get('temperature_2m_max', [None])[0] or float('nan')),
            'meteo_temp_min': float(d.get('temperature_2m_min', [None])[0] or float('nan')),
            'meteo_wind_max': float(d.get('windspeed_10m_max', [None])[0] or float('nan')),
            'meteo_precip_sum': float(d.get('precipitation_sum', [None])[0] or 0.0),
        }
    except Exception as e:
        print(f"  Open-Meteo 获取失败 ({date_str}): {e}")
        return {
            'meteo_weather_code': 0,
            'meteo_temp_max': float('nan'),
            'meteo_temp_min': float('nan'),
            'meteo_wind_max': float('nan'),
            'meteo_precip_sum': 0.0,
        }


# ─────────────────────────────────────────────
# 特征工程（与训练脚本保持一致）
# ─────────────────────────────────────────────

def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """对新追加的行补充特征工程列（与训练数据格式对齐）。"""
    try:
        import chinese_calendar as cncal
        def _is_holiday(d):
            try:
                return int(cncal.is_holiday(d))
            except Exception:
                return int(d.weekday() >= 5)
    except ImportError:
        def _is_holiday(d):
            return int(d.weekday() >= 5)

    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])

    # 时间特征
    df['day_of_week'] = df['date'].dt.weekday
    df['month'] = df['date'].dt.month
    df['day_of_month'] = df['date'].dt.day
    df['day_of_year'] = df['date'].dt.dayofyear
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_holiday'] = df['date'].apply(lambda x: _is_holiday(x.date()))
    df['is_workday'] = ((df['is_weekend'] == 0) & (df['is_holiday'] == 0)).astype(int)

    # 周期编码
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_norm'] = (df['month'] - 1) / 11.0
    df['day_of_week_norm'] = df['day_of_week'] / 6.0

    df['date'] = df['date'].dt.strftime('%Y-%m-%d')
    return df


def _compute_scaled_features(df_full: pd.DataFrame) -> pd.DataFrame:
    """用全量数据重新计算 scaled 特征（MinMax，基于全量数据）。"""
    from sklearn.preprocessing import MinMaxScaler

    df = df_full.copy()

    # visitor_count_scaled
    if 'tourism_num' in df.columns:
        vc = pd.to_numeric(df['tourism_num'], errors='coerce').fillna(0)
        scaler = MinMaxScaler()
        df['visitor_count_scaled'] = scaler.fit_transform(vc.values.reshape(-1, 1)).flatten()

    # lag_7 scaled
    if 'tourism_num' in df.columns:
        lag7 = pd.to_numeric(df['tourism_num'], errors='coerce').shift(7)
        scaler7 = MinMaxScaler()
        valid = lag7.dropna()
        if len(valid) > 0:
            scaler7.fit(valid.values.reshape(-1, 1))
            df['tourism_num_lag_7_scaled'] = scaler7.transform(
                lag7.fillna(lag7.mean()).values.reshape(-1, 1)
            ).flatten()
        df['tourism_num_lag_1'] = pd.to_numeric(df['tourism_num'], errors='coerce').shift(1)
        df['tourism_num_lag_7'] = lag7
        df['tourism_num_lag_14'] = pd.to_numeric(df['tourism_num'], errors='coerce').shift(14)
        df['tourism_num_lag_28'] = pd.to_numeric(df['tourism_num'], errors='coerce').shift(28)
        df['tourism_num_rolling_mean_7'] = pd.to_numeric(df['tourism_num'], errors='coerce').rolling(7).mean()
        df['tourism_num_rolling_std_7'] = pd.to_numeric(df['tourism_num'], errors='coerce').rolling(7).std()
        df['tourism_num_rolling_mean_14'] = pd.to_numeric(df['tourism_num'], errors='coerce').rolling(14).mean()

    # 天气 scaled
    for col, new_col in [
        ('meteo_precip_sum', 'meteo_precip_sum_scaled'),
        ('meteo_temp_max', 'temp_high_scaled'),
        ('meteo_temp_min', 'temp_low_scaled'),
    ]:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors='coerce').fillna(0)
            sc = MinMaxScaler()
            df[new_col] = sc.fit_transform(vals.values.reshape(-1, 1)).flatten()

    # temp_high_c / temp_low_c 别名
    if 'meteo_temp_max' in df.columns and 'temp_high_c' not in df.columns:
        df['temp_high_c'] = df['meteo_temp_max']
    if 'meteo_temp_min' in df.columns and 'temp_low_c' not in df.columns:
        df['temp_low_c'] = df['meteo_temp_min']

    return df


# ─────────────────────────────────────────────
# 核心：追加新数据
# ─────────────────────────────────────────────

def append_new_data(target_date: Optional[date] = None, dry_run: bool = False) -> bool:
    """将指定日期（默认昨天）的客流+天气追加到训练 CSV。

    Args:
        target_date: 要追加的日期，默认为昨天
        dry_run: 若 True，只打印不写入

    Returns:
        True 表示成功追加，False 表示跳过（已存在或数据不可用）
    """
    if target_date is None:
        target_date = date.today() - timedelta(days=1)

    date_str = target_date.isoformat()
    print(f"\n{'='*60}")
    print(f"数据追加任务: {date_str}")
    print(f"{'='*60}")

    # 1. 检查训练 CSV 是否存在
    if not TRAIN_CSV.exists():
        print(f"错误：训练 CSV 不存在: {TRAIN_CSV}")
        return False

    df_train = pd.read_csv(TRAIN_CSV, encoding='utf-8-sig')
    df_train['date'] = pd.to_datetime(df_train['date']).dt.strftime('%Y-%m-%d')

    # 2. 检查是否已存在
    if date_str in df_train['date'].values:
        existing_row = df_train[df_train['date'] == date_str]
        if pd.notna(existing_row['tourism_num'].values[0]):
            print(f"  跳过：{date_str} 已存在于训练数据中（含真实客流数据）")
            return False
        # 日期存在但 tourism_num 为 NaN（backfill 预写的天气占位行），继续覆盖
        print(f"  {date_str} 存在占位行（tourism_num=NaN），将用真实数据覆盖")
        df_train = df_train[df_train['date'] != date_str].reset_index(drop=True)

    # 3. 从爬虫获取客流数据
    visitor_count = None
    try:
        from realtime.jiuzhaigou_crawler import JiuzhaigouCrawler
        crawler = JiuzhaigouCrawler()
        data = crawler.fetch_latest_visitor_count()
        if data and str(data.get('date', '')) == date_str:
            visitor_count = int(data['visitor_count'])
            print(f"  爬虫获取客流: {visitor_count:,} 人")
        else:
            # fetch_latest 拿到的不是目标日期，改用历史查询
            rows = crawler.fetch_by_date_range(date_str, date_str)
            if rows:
                visitor_count = int(rows[0]['visitor_count'])
                print(f"  历史查询获取客流: {visitor_count:,} 人")
            else:
                # 最后兜底：数据库
                db_data = crawler.get_latest_from_database()
                if db_data and str(db_data.get('date', '')) == date_str:
                    visitor_count = int(db_data['visitor_count'])
                    print(f"  数据库获取客流: {visitor_count:,} 人")
    except Exception as e:
        print(f"  爬虫获取失败: {e}")

    if visitor_count is None:
        print(f"  警告：无法获取 {date_str} 的客流数据，跳过追加")
        return False

    # 4. 获取天气数据
    print(f"  获取 Open-Meteo 天气数据...")
    weather = _fetch_meteo_for_date(target_date)
    print(f"  天气: 最高{weather['meteo_temp_max']:.1f}°C, "
          f"最低{weather['meteo_temp_min']:.1f}°C, "
          f"降水{weather['meteo_precip_sum']:.1f}mm")

    # 5. 构建新行
    new_row = {
        'date': date_str,
        'tourism_num': visitor_count,
        'temp_high_c': weather['meteo_temp_max'],
        'temp_low_c': weather['meteo_temp_min'],
        'meteo_weather_code': weather['meteo_weather_code'],
        'meteo_temp_max': weather['meteo_temp_max'],
        'meteo_temp_min': weather['meteo_temp_min'],
        'meteo_wind_max': weather['meteo_wind_max'],
        'meteo_precip_sum': weather['meteo_precip_sum'],
        'meteo_rain_sum': 0.0,
        'meteo_snowfall_sum': 0.0,
        'meteo_precip_hours': 0.0,
        'weather_source': 'open-meteo',
    }

    # 6. 追加并重新计算特征
    df_new_row = pd.DataFrame([new_row])
    df_new_row = _engineer_features(df_new_row)

    # 合并到全量数据
    df_combined = pd.concat([df_train, df_new_row], ignore_index=True)
    df_combined = df_combined.sort_values('date').reset_index(drop=True)

    # 重新计算 scaled 特征（需要全量数据）
    df_combined = _compute_scaled_features(df_combined)

    if dry_run:
        print(f"  [DRY RUN] 将追加 {date_str}，visitor_count={visitor_count}")
        return True

    # 7. 写回 CSV
    df_combined.to_csv(TRAIN_CSV, index=False, encoding='utf-8-sig')
    print(f"  ✅ 已追加 {date_str} 到训练数据（共 {len(df_combined)} 行）")

    # 8. 更新追加日志
    _update_append_log(date_str, visitor_count, weather)

    return True


def _update_append_log(date_str: str, visitor_count: int, weather: dict):
    """记录追加历史。"""
    log = []
    if APPEND_LOG.exists():
        try:
            with open(APPEND_LOG, 'r', encoding='utf-8') as f:
                log = json.load(f)
        except Exception:
            log = []
    log.append({
        'date': date_str,
        'visitor_count': visitor_count,
        'weather': weather,
        'appended_at': datetime.now().isoformat(),
    })
    # 只保留最近 365 条
    log = log[-365:]
    with open(APPEND_LOG, 'w', encoding='utf-8') as f:
        json.dump(log, f, ensure_ascii=False, indent=2)


# ─────────────────────────────────────────────
# 未来天气追加（供 backfill 使用）
# ─────────────────────────────────────────────

def append_future_weather(days_ahead: int = 16) -> int:
    """从 Open-Meteo 预报 API 拉取未来 days_ahead 天天气，追加到 processed CSV。

    未来日期行的 tourism_num=NaN（无实际客流），其余特征正常填充。
    已存在的日期不重复追加。返回新增行数。
    """
    try:
        import chinese_calendar as cncal
        def _is_holiday(d):
            try:
                return int(cncal.is_holiday(d))
            except Exception:
                return int(d.weekday() >= 5)
    except ImportError:
        def _is_holiday(d):
            return int(d.weekday() >= 5)

    if not TRAIN_CSV.exists():
        print(f"  append_future_weather: processed CSV 不存在: {TRAIN_CSV}")
        return 0

    df_existing = pd.read_csv(TRAIN_CSV)
    df_existing['date'] = pd.to_datetime(df_existing['date']).dt.strftime('%Y-%m-%d')
    existing_dates = set(df_existing['date'].tolist())

    # 拉取未来天气预报
    try:
        r = requests.get(
            'https://api.open-meteo.com/v1/forecast',
            params={
                'latitude': LAT, 'longitude': LON,
                'daily': 'temperature_2m_max,temperature_2m_min,precipitation_sum,'
                         'weathercode,windspeed_10m_max',
                'timezone': 'Asia/Shanghai',
                'forecast_days': days_ahead,
            },
            timeout=15,
        )
        r.raise_for_status()
        d = r.json()['daily']
    except Exception as e:
        print(f"  append_future_weather: Open-Meteo 预报获取失败: {e}")
        return 0

    new_rows = []
    for i, date_str in enumerate(d['time']):
        if date_str in existing_dates:
            continue
        dt = pd.to_datetime(date_str).date()
        row = {
            'date': date_str,
            'tourism_num': float('nan'),
            'meteo_temp_max': float(d['temperature_2m_max'][i] or float('nan')),
            'meteo_temp_min': float(d['temperature_2m_min'][i] or float('nan')),
            'meteo_precip_sum': float(d['precipitation_sum'][i] or 0.0),
            'meteo_weather_code': int(d['weathercode'][i] or 0),
            'meteo_wind_max': float(d['windspeed_10m_max'][i] or float('nan')),
            'meteo_rain_sum': float(d['precipitation_sum'][i] or 0.0),
            'meteo_snowfall_sum': 0.0,
            'meteo_precip_hours': 0.0,
            'temp_high_c': float(d['temperature_2m_max'][i] or float('nan')),
            'temp_low_c': float(d['temperature_2m_min'][i] or float('nan')),
            'temp_range_c': float((d['temperature_2m_max'][i] or 0) - (d['temperature_2m_min'][i] or 0)),
            'weather_code_en': None,
            'weather_abbr': None,
            'weather_code_id': None,
            'wind_dir_en': None,
            'wind_level': None,
            'wind_dir_id': None,
            'aqi_value': float('nan'),
            'aqi_level_en': None,
            'weather_source': 'open-meteo-forecast',
            'day_of_week': dt.weekday(),
            'month': dt.month,
            'day_of_month': dt.day,
            'day_of_year': dt.timetuple().tm_yday,
            'week_of_year': dt.isocalendar()[1],
            'is_weekend': int(dt.weekday() >= 5),
            'is_holiday': _is_holiday(dt),
            'is_workday': int(dt.weekday() < 5 and not _is_holiday(dt)),
            'month_sin': float(np.sin(2 * np.pi * dt.month / 12)),
            'month_cos': float(np.cos(2 * np.pi * dt.month / 12)),
            'dow_sin': float(np.sin(2 * np.pi * dt.weekday() / 7)),
            'dow_cos': float(np.cos(2 * np.pi * dt.weekday() / 7)),
            'month_norm': (dt.month - 1) / 11.0,
            'day_of_week_norm': dt.weekday() / 6.0,
        }
        new_rows.append(row)

    if not new_rows:
        print(f"  append_future_weather: 无新日期需要追加（已有 {len(existing_dates)} 天）")
        return 0

    df_new = pd.DataFrame(new_rows)

    # 合并后重新计算 scaled 特征（lag_7 等需要全量数据）
    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    df_combined = _compute_scaled_features(df_combined)

    df_combined.to_csv(TRAIN_CSV, index=False)
    print(f"  append_future_weather: 追加 {len(new_rows)} 个未来日期 ({new_rows[0]['date']} ~ {new_rows[-1]['date']})")
    return len(new_rows)


def daily_backfill():
    """每日 backfill 任务：先追加未来天气，再滚动推理三模型预测。"""
    print(f"\n[daily_backfill] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    # 1. 追加未来14天天气到 processed CSV
    append_future_weather(days_ahead=16)
    # 2. 运行 backfill 推理
    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / 'scripts' / 'backfill_predictions.py')],
            cwd=str(PROJECT_ROOT),
            capture_output=True, text=True, timeout=600,
        )
        if result.returncode == 0:
            print(f"  backfill_predictions 完成")
            if result.stdout:
                print(result.stdout[-800:])
        else:
            print(f"  backfill_predictions 失败:\n{result.stderr[-500:]}")
    except Exception as e:
        print(f"  backfill_predictions 异常: {e}")




def backfill_real_data(end_date: Optional[date] = None, dry_run: bool = False) -> int:
    """
    自动检测 CSV 中最后一条真实数据日期，批量补追到 end_date（默认昨天）。

    - 跳过已有真实数据（tourism_num 非 NaN）的日期
    - 覆盖占位行（tourism_num=NaN）
    - 批量爬取后统一写一次 CSV，避免重复 fit scaler

    Returns:
        成功追加的天数
    """
    if end_date is None:
        end_date = date.today() - timedelta(days=1)

    if not TRAIN_CSV.exists():
        print(f'错误：训练 CSV 不存在: {TRAIN_CSV}')
        return 0

    df = pd.read_csv(TRAIN_CSV, encoding='utf-8-sig')
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')

    # 找最后一条真实数据日期
    real_mask = pd.to_numeric(df['tourism_num'], errors='coerce').notna()
    if not real_mask.any():
        print('错误：CSV 中没有任何真实客流数据')
        return 0
    last_real = date.fromisoformat(df.loc[real_mask, 'date'].max())

    if last_real >= end_date:
        print(f'数据已是最新（最后真实日期: {last_real}），无需补追')
        return 0

    gap_dates = [last_real + timedelta(days=i+1)
                 for i in range((end_date - last_real).days)]
    print(f'\n{"="*60}')
    print(f'批量补追: {gap_dates[0]} ~ {gap_dates[-1]}（共 {len(gap_dates)} 天）')
    print(f'{"="*60}')

    # 批量爬取客流数据
    from realtime.jiuzhaigou_crawler import JiuzhaigouCrawler
    crawler = JiuzhaigouCrawler()
    crawled = crawler.fetch_by_date_range(gap_dates[0].isoformat(), gap_dates[-1].isoformat())
    crawled_map = {r['date']: r['visitor_count'] for r in crawled}

    if not crawled_map:
        print('爬取失败，未获取到任何数据')
        return 0

    # 移除所有占位行（NaN 行）
    nan_mask = pd.to_numeric(df['tourism_num'], errors='coerce').isna()
    df = df[~nan_mask].reset_index(drop=True)

    # 逐日构建新行
    new_rows = []
    for d in gap_dates:
        d_str = d.isoformat()
        if d_str not in crawled_map:
            print(f'  跳过 {d_str}：未爬取到客流数据')
            continue
        visitor_count = crawled_map[d_str]
        weather = _fetch_meteo_for_date(d)
        new_row = {
            'date': d_str,
            'tourism_num': visitor_count,
            'temp_high_c': weather['meteo_temp_max'],
            'temp_low_c': weather['meteo_temp_min'],
            'meteo_weather_code': weather['meteo_weather_code'],
            'meteo_temp_max': weather['meteo_temp_max'],
            'meteo_temp_min': weather['meteo_temp_min'],
            'meteo_wind_max': weather['meteo_wind_max'],
            'meteo_precip_sum': weather['meteo_precip_sum'],
            'meteo_rain_sum': 0.0,
            'meteo_snowfall_sum': 0.0,
            'meteo_precip_hours': 0.0,
            'weather_source': 'open-meteo',
        }
        df_row = pd.DataFrame([new_row])
        df_row = _engineer_features(df_row)
        new_rows.append(df_row)
        print(f'  ✓ {d_str}: {visitor_count:,} 人，'
              f'{weather["meteo_temp_max"]:.1f}/{weather["meteo_temp_min"]:.1f}°C')

    if not new_rows:
        print('没有可追加的数据')
        return 0

    if dry_run:
        print(f'\n[DRY RUN] 将追加 {len(new_rows)} 天数据，不写入文件')
        return len(new_rows)

    # 合并并统一计算 scaled 特征（只 fit 一次）
    df_combined = pd.concat([df] + new_rows, ignore_index=True)
    df_combined = df_combined.sort_values('date').reset_index(drop=True)
    df_combined = _compute_scaled_features(df_combined)
    df_combined.to_csv(TRAIN_CSV, index=False, encoding='utf-8-sig')

    print(f'\n✅ 成功追加 {len(new_rows)} 天数据（CSV 共 {len(df_combined)} 行）')

    # 更新日志
    for row in new_rows:
        d_str = row['date'].values[0]
        _update_append_log(d_str, crawled_map[d_str], {})

    return len(new_rows)


def monthly_retrain(epochs: int = 120) -> bool:
    """触发三个模型的重训练。

    通过调用 run_pipeline.py 依次训练 GRU、LSTM、Seq2Seq。
    训练完成后，新的模型权重会自动进入 output/runs/，
    下次 Dashboard 加载时会使用最新的 backup 目录。

    Returns:
        True 表示全部成功，False 表示至少一个失败
    """
    print(f"\n{'='*60}")
    print(f"月度重训任务: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    models = ['gru', 'lstm', 'seq2seq_attention']
    success_all = True

    for model in models:
        print(f"\n  训练 {model.upper()}...")
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / 'run_pipeline.py'),
            '--model', model,
            '--features', '8',
            '--epochs', str(epochs),
        ]
        try:
            result = subprocess.run(
                cmd, cwd=str(PROJECT_ROOT),
                capture_output=True, text=True, timeout=3600
            )
            if result.returncode == 0:
                print(f"  ✅ {model.upper()} 训练完成")
            else:
                print(f"  ❌ {model.upper()} 训练失败:\n{result.stderr[-500:]}")
                success_all = False
        except subprocess.TimeoutExpired:
            print(f"  ❌ {model.upper()} 训练超时（1小时）")
            success_all = False
        except Exception as e:
            print(f"  ❌ {model.upper()} 训练异常: {e}")
            success_all = False

    if success_all:
        print(f"\n✅ 月度重训完成，所有模型已更新")
    else:
        print(f"\n⚠️  月度重训部分失败，请检查日志")

    return success_all


# ─────────────────────────────────────────────
# APScheduler 集成入口
# ─────────────────────────────────────────────

def start_data_pipeline_scheduler(scheduler):
    """向已有的 APScheduler 实例注册数据管道任务。"""
    # 每日 08:30 追加昨日实际数据
    scheduler.add_job(
        append_new_data, 'cron', hour=8, minute=30,
        id='daily_append', replace_existing=True,
        kwargs={'target_date': None}
    )
    print("数据管道调度器：每日08:30追加数据任务已注册")

    # 每日 09:30 追加未来天气 + backfill 三模型预测（爬虫09:00完成后）
    scheduler.add_job(
        daily_backfill, 'cron', hour=9, minute=30,
        id='daily_backfill', replace_existing=True,
    )
    print("数据管道调度器：每日09:30 backfill预测任务已注册")

    # 每月1日 02:00 重训（避开白天高峰）
    scheduler.add_job(
        monthly_retrain, 'cron', day=1, hour=2, minute=0,
        id='monthly_retrain', replace_existing=True,
        kwargs={'epochs': 120}
    )
    print("数据管道调度器：每月1日02:00重训任务已注册")


# ─────────────────────────────────────────────
# CLI 入口
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='数据追加与月度重训管道')
    parser.add_argument('--append', action='store_true', help='追加数据：自动检测空挡，1天或多天均可（默认追到昨天）')
    parser.add_argument('--retrain', action='store_true', help='重训三个模型')
    parser.add_argument('--date', default=None, help='指定截止日期 (YYYY-MM-DD)，默认昨天')
    parser.add_argument('--epochs', type=int, default=120, help='重训轮次（默认120）')
    parser.add_argument('--dry-run', action='store_true', help='仅打印，不写入')
    args = parser.parse_args()

    if not args.append and not args.retrain:
        parser.print_help()
        return

    if args.append:
        end = date.fromisoformat(args.date) if args.date else None
        backfill_real_data(end_date=end, dry_run=args.dry_run)

    if args.retrain:
        monthly_retrain(epochs=args.epochs)


if __name__ == '__main__':
    main()
