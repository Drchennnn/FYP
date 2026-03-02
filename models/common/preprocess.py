"""
Preprocess Jiuzhaigou tourism-weather data to English-only model features.

Input: Raw CSV from crawler (data/raw/jiuzhaigou_raw_*.csv)
Output: Processed CSV with features (data/processed/jiuzhaigou_daily_features.csv)
Features: Lag features, rolling stats, date encoding, holiday flags, etc.
"""

from __future__ import annotations

import os
import argparse
import json
import re
from pathlib import Path
from typing import Iterable, Optional

import chinese_calendar as cncal
import numpy as np
import pandas as pd
import requests

OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

WEATHER_CN_MAP = {
    "\u6674": ("SUNNY", "SUN"),
    "\u591a\u4e91": ("CLOUDY", "CLD"),
    "\u9634": ("OVERCAST", "OVC"),
    "\u9635\u96e8": ("SHOWER", "SHR"),
    "\u5c0f\u96e8": ("LIGHT_RAIN", "LRA"),
    "\u4e2d\u96e8": ("MODERATE_RAIN", "MRA"),
    "\u5927\u96e8": ("HEAVY_RAIN", "HRA"),
    "\u66b4\u96e8": ("STORM_RAIN", "STM"),
    "\u96f7\u9635\u96e8": ("THUNDER_SHOWER", "TSR"),
    "\u96e8\u5939\u96ea": ("SLEET", "SLT"),
    "\u5c0f\u96ea": ("LIGHT_SNOW", "LSN"),
    "\u4e2d\u96ea": ("MODERATE_SNOW", "MSN"),
    "\u5927\u96ea": ("HEAVY_SNOW", "HSN"),
    "\u9635\u96ea": ("SNOW_SHOWER", "SSR"),
    "\u96fe": ("FOG", "FOG"),
    "\u973e": ("HAZE", "HAZ"),
    "\u626c\u6c99": ("BLOWING_SAND", "BSD"),
    "\u6c99\u5c18\u66b4": ("SANDSTORM", "SST"),
}
WEATHER_EN_ABBR = {
    "SUNNY": "SUN",
    "CLOUDY": "CLD",
    "OVERCAST": "OVC",
    "SHOWER": "SHR",
    "LIGHT_RAIN": "LRA",
    "MODERATE_RAIN": "MRA",
    "HEAVY_RAIN": "HRA",
    "STORM_RAIN": "STM",
    "THUNDER_SHOWER": "TSR",
    "SLEET": "SLT",
    "LIGHT_SNOW": "LSN",
    "MODERATE_SNOW": "MSN",
    "HEAVY_SNOW": "HSN",
    "SNOW_SHOWER": "SSR",
    "FOG": "FOG",
    "HAZE": "HAZ",
    "BLOWING_SAND": "BSD",
    "SANDSTORM": "SST",
    "OTHER": "OTH",
}

WIND_CN_MAP = {
    "\u4e1c\u5317\u98ce": "NE",
    "\u4e1c\u5357\u98ce": "SE",
    "\u897f\u5317\u98ce": "NW",
    "\u897f\u5357\u98ce": "SW",
    "\u4e1c\u98ce": "E",
    "\u897f\u98ce": "W",
    "\u5357\u98ce": "S",
    "\u5317\u98ce": "N",
    "\u65e0\u6301\u7eed\u98ce\u5411": "VAR",
    "\u65cb\u8f6c\u98ce": "VAR",
    "\u5fae\u98ce": "BREEZE",
}

AQI_CN_MAP = {
    "\u4f18": "EXCELLENT",
    "\u826f": "GOOD",
    "\u8f7b\u5ea6\u6c61\u67d3": "LIGHT_POLLUTION",
    "\u4e2d\u5ea6\u6c61\u67d3": "MODERATE_POLLUTION",
    "\u91cd\u5ea6\u6c61\u67d3": "HEAVY_POLLUTION",
    "\u4e25\u91cd\u6c61\u67d3": "SEVERE_POLLUTION",
}


def parse_number(text: str) -> float:
    match = re.search(r"-?\d+(?:\.\d+)?", str(text))
    return float(match.group(0)) if match else np.nan


def parse_weather_code(value: str) -> tuple[str, str]:
    s = str(value).strip().replace("\u8f6c", "~").replace("/", "~")
    parts = [p for p in re.split(r"[~\s]+", s) if p]
    for part in parts:
        part_upper = part.upper()
        if part_upper in WEATHER_EN_ABBR:
            return part_upper, WEATHER_EN_ABBR[part_upper]
        if part in WEATHER_CN_MAP:
            return WEATHER_CN_MAP[part]
    return "OTHER", "OTH"


def parse_wind(value: str) -> tuple[str, float]:
    s = str(value).strip().upper()
    if re.fullmatch(r"(N|S|E|W|NE|NW|SE|SW|VAR|BREEZE)([_-]?\d+(?:\.\d+)?)?", s):
        m = re.search(r"\d+(?:\.\d+)?", s)
        return re.sub(r"[_-].*$", "", s), (float(m.group(0)) if m else np.nan)

    wind_dir = "UNK"
    for k in sorted(WIND_CN_MAP.keys(), key=len, reverse=True):
        if k in s:
            wind_dir = WIND_CN_MAP[k]
            break
    nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", s)]
    if "-" in s and len(nums) >= 2:
        wind_level = (nums[0] + nums[1]) / 2.0
    elif nums:
        wind_level = nums[0]
    else:
        wind_level = np.nan
    return wind_dir, wind_level


def parse_aqi(value: str) -> tuple[float, str]:
    s = str(value).strip()
    aqi_value = parse_number(s)
    m = re.search(r"[A-Za-z_]+$", s)
    if m:
        return aqi_value, m.group(0).upper()
    for zh, en in AQI_CN_MAP.items():
        if zh in s:
            return aqi_value, en
    return aqi_value, "UNKNOWN"


def fetch_open_meteo_features(latitude: float, longitude: float, start_date: str, end_date: str) -> pd.DataFrame:
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": (
            "weathercode,temperature_2m_max,temperature_2m_min,windspeed_10m_max,"
            "precipitation_sum,rain_sum,snowfall_sum,precipitation_hours"
        ),
        "timezone": "Asia/Shanghai",
    }
    response = requests.get(OPEN_METEO_ARCHIVE_URL, params=params, timeout=40)
    response.raise_for_status()
    d = response.json().get("daily", {})
    return pd.DataFrame(
        {
            "date": pd.to_datetime(d.get("time", [])),
            "meteo_weather_code": d.get("weathercode", []),
            "meteo_temp_max": d.get("temperature_2m_max", []),
            "meteo_temp_min": d.get("temperature_2m_min", []),
            "meteo_wind_max": d.get("windspeed_10m_max", []),
            "meteo_precip_sum": d.get("precipitation_sum", []),
            "meteo_rain_sum": d.get("rain_sum", []),
            "meteo_snowfall_sum": d.get("snowfall_sum", []),
            "meteo_precip_hours": d.get("precipitation_hours", []),
        }
    )


def build_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    ds = df["date"]
    df["day_of_week"] = ds.dt.weekday
    df["month"] = ds.dt.month
    df["day_of_month"] = ds.dt.day
    df["day_of_year"] = ds.dt.dayofyear
    df["week_of_year"] = ds.dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    
    # 【修改】增加容错处理，防止 chinese_calendar 不支持未来年份
    def get_holiday_safe(d):
        try:
            return int(cncal.is_holiday(d.date()))
        except NotImplementedError:
            # 如果年份超出支持范围（如2024+），降级为简单的周末判断
            # 这里简单假设非周末即工作日，或者默认非节假日
            # 更好的策略是手动补充关键节假日
            return 1 if d.weekday() >= 5 else 0
        except Exception:
            return 0

    def get_workday_safe(d):
        try:
            return int(cncal.is_workday(d.date()))
        except NotImplementedError:
            return 1 if d.weekday() < 5 else 0
        except Exception:
            return 1

    df["is_holiday"] = ds.apply(get_holiday_safe)
    df["is_workday"] = ds.apply(get_workday_safe)
    
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    return df


def add_lag_features(df: pd.DataFrame, target_col: str = "tourism_num") -> pd.DataFrame:
    """添加滞后特征和滚动统计特征
    
    包括：
    - 滞后特征：lag_1, lag_7, lag_14, lag_28
    - 滚动统计：7天均值、标准差，14天均值
    - 标准化滞后特征：tourism_num_lag_7_scaled（使用前向填充处理缺失值）
    
    Args:
        df: 输入DataFrame
        target_col: 目标列名
        
    Returns:
        添加了特征的DataFrame
    """
    # 原始滞后特征
    for lag in (1, 7, 14, 28):
        df[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)
    
    # 滚动统计特征
    df[f"{target_col}_rolling_mean_7"] = df[target_col].rolling(7).mean()
    df[f"{target_col}_rolling_std_7"] = df[target_col].rolling(7).std()
    df[f"{target_col}_rolling_mean_14"] = df[target_col].rolling(14).mean()
    
    # 新增：标准化滞后7天特征（处理缺失值）
    # 使用前向填充处理前7天的缺失值
    lag_7_col = f"{target_col}_lag_7"
    if lag_7_col in df.columns:
        # 先填充缺失值（前7天用第一个有效值填充）
        df[lag_7_col] = df[lag_7_col].fillna(method='ffill')
        # 如果开头还有NaN（比如整个序列开头），用0填充
        df[lag_7_col] = df[lag_7_col].fillna(0)
        
        # 计算标准化版本
        # 使用Min-Max标准化到[0, 1]范围
        lag_min = df[lag_7_col].min()
        lag_max = df[lag_7_col].max()
        if lag_max > lag_min:  # 避免除零
            df[f"{target_col}_lag_7_scaled"] = (df[lag_7_col] - lag_min) / (lag_max - lag_min)
        else:
            df[f"{target_col}_lag_7_scaled"] = 0.0
    
    return df


def add_meteorological_features(df: pd.DataFrame) -> pd.DataFrame:
    """添加标准化气象特征
    
    包括：
    - meteo_precip_sum_scaled: 标准化降水量
    - temp_high_scaled: 标准化最高温度
    - temp_low_scaled: 标准化最低温度
    
    使用Min-Max标准化到[0, 1]范围
    
    Args:
        df: 输入DataFrame
        
    Returns:
        添加了标准化气象特征的DataFrame
    """
    # 检查并准备气象数据列
    meteo_cols = {}
    
    # 降水量：优先使用meteo_precip_sum，否则使用0
    if "meteo_precip_sum" in df.columns:
        meteo_cols["precip"] = "meteo_precip_sum"
    else:
        # 如果没有降水量数据，创建全0列
        df["meteo_precip_sum"] = 0.0
        meteo_cols["precip"] = "meteo_precip_sum"
    
    # 最高温度：优先使用meteo_temp_max，否则使用temp_high_c
    if "meteo_temp_max" in df.columns:
        meteo_cols["temp_high"] = "meteo_temp_max"
    elif "temp_high_c" in df.columns:
        meteo_cols["temp_high"] = "temp_high_c"
    else:
        # 如果没有温度数据，创建默认值（九寨沟平均高温约15°C）
        df["temp_high_default"] = 15.0
        meteo_cols["temp_high"] = "temp_high_default"
    
    # 最低温度：优先使用meteo_temp_min，否则使用temp_low_c
    if "meteo_temp_min" in df.columns:
        meteo_cols["temp_low"] = "meteo_temp_min"
    elif "temp_low_c" in df.columns:
        meteo_cols["temp_low"] = "temp_low_c"
    else:
        # 如果没有温度数据，创建默认值（九寨沟平均低温约5°C）
        df["temp_low_default"] = 5.0
        meteo_cols["temp_low"] = "temp_low_default"
    
    # 标准化处理
    for feature_type, col_name in meteo_cols.items():
        if col_name in df.columns:
            # 确保是数值类型
            df[col_name] = pd.to_numeric(df[col_name], errors="coerce").fillna(0)
            
            # Min-Max标准化
            col_min = df[col_name].min()
            col_max = df[col_name].max()
            
            if col_max > col_min:  # 避免除零
                if feature_type == "precip":
                    df["meteo_precip_sum_scaled"] = (df[col_name] - col_min) / (col_max - col_min)
                elif feature_type == "temp_high":
                    df["temp_high_scaled"] = (df[col_name] - col_min) / (col_max - col_min)
                elif feature_type == "temp_low":
                    df["temp_low_scaled"] = (df[col_name] - col_min) / (col_max - col_min)
            else:
                # 如果所有值都相同，标准化为0.5
                if feature_type == "precip":
                    df["meteo_precip_sum_scaled"] = 0.5
                elif feature_type == "temp_high":
                    df["temp_high_scaled"] = 0.5
                elif feature_type == "temp_low":
                    df["temp_low_scaled"] = 0.5
    
    # 确保所有标准化列都存在
    if "meteo_precip_sum_scaled" not in df.columns:
        df["meteo_precip_sum_scaled"] = 0.5
    if "temp_high_scaled" not in df.columns:
        df["temp_high_scaled"] = 0.5
    if "temp_low_scaled" not in df.columns:
        df["temp_low_scaled"] = 0.5
    
    return df


def preprocess(input_csv: Path, raw_output_csv: Path, output_csv: Optional[Path], metadata_json: Path, latitude: float, longitude: float) -> None:
    df = pd.read_csv(input_csv, encoding="utf-8-sig")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)

    # Automatically generate output filename if not provided
    if output_csv is None:
        start_date = df["date"].min().strftime("%Y-%m-%d")
        end_date = df["date"].max().strftime("%Y-%m-%d")
        # Default to data/processed/jiuzhaigou_daily_features_{start}_{end}.csv
        # We assume input_csv is somewhere in the project, we try to locate 'data/processed' relative to project root
        # If input_csv is absolute or relative, we try to use its parent structure or fallback to cwd
        
        # Strategy: Use input_csv's parent's sibling 'processed' if it exists, else 'data/processed'
        # simpler: just use data/processed relative to where script is run (cwd) or relative to input file?
        # The user's pattern is data/raw -> data/processed.
        
        # Let's try to find data/processed relative to project root (2 levels up from this script)
        # This script is in models/common/preprocess.py (root/models/common)
        # So project root is parents[2]
        project_root = Path(__file__).resolve().parents[2]
        processed_dir = project_root / "data" / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)
        output_csv = processed_dir / f"jiuzhaigou_daily_features_{start_date}_{end_date}.csv"
        print(f"Auto-generated output filename: {output_csv}")

    df["tourism_num"] = pd.to_numeric(df["tourism_num"], errors="coerce")

    high_col = "temperature_high" if "temperature_high" in df.columns else "temp_high_c"
    low_col = "temperature_low" if "temperature_low" in df.columns else "temp_low_c"
    weather_col = "weather" if "weather" in df.columns else "weather_code_en"
    wind_col = "wind" if "wind" in df.columns else "wind_dir_en"
    air_col = "air_quality" if "air_quality" in df.columns else "aqi_value"

    df["temp_high_c"] = df[high_col].apply(parse_number)
    df["temp_low_c"] = df[low_col].apply(parse_number)
    df["temp_range_c"] = df["temp_high_c"] - df["temp_low_c"]

    weather_codes = df[weather_col].apply(parse_weather_code)
    df["weather_code_en"] = weather_codes.apply(lambda x: x[0])
    df["weather_abbr"] = weather_codes.apply(lambda x: x[1])
    weather_vocab = sorted(df["weather_abbr"].dropna().unique().tolist())
    weather_id_map = {w: i for i, w in enumerate(weather_vocab)}
    df["weather_code_id"] = df["weather_abbr"].map(weather_id_map).astype(int)

    wind_info = df[wind_col].apply(parse_wind)
    df["wind_dir_en"] = wind_info.apply(lambda x: x[0])
    df["wind_level"] = wind_info.apply(lambda x: x[1])
    wind_vocab = sorted(df["wind_dir_en"].dropna().unique().tolist())
    wind_id_map = {w: i for i, w in enumerate(wind_vocab)}
    df["wind_dir_id"] = df["wind_dir_en"].map(wind_id_map).astype(int)

    aqi_info = df[air_col].fillna("").apply(parse_aqi)
    df["aqi_value"] = aqi_info.apply(lambda x: x[0])
    df["aqi_level_en"] = aqi_info.apply(lambda x: x[1])

    raw_en = df[
        [
            "date",
            "tourism_num",
            "temp_high_c",
            "temp_low_c",
            "temp_range_c",
            "weather_code_en",
            "weather_abbr",
            "wind_dir_en",
            "wind_level",
            "aqi_value",
            "aqi_level_en",
            "weather_source",
        ]
    ].copy()
    raw_en["date"] = raw_en["date"].dt.strftime("%Y-%m-%d")
    raw_output_csv.parent.mkdir(parents=True, exist_ok=True)
    raw_en.to_csv(raw_output_csv, index=False, encoding="utf-8-sig")

    meteo = fetch_open_meteo_features(
        latitude=latitude,
        longitude=longitude,
        start_date=df["date"].min().strftime("%Y-%m-%d"),
        end_date=df["date"].max().strftime("%Y-%m-%d"),
    )
    df = df.merge(meteo, on="date", how="left")

    # Fill missing Open-Meteo data using local raw data or fallback strategies
    # This prevents dropping recent rows due to API delays
    if "meteo_temp_max" in df.columns and "temp_high_c" in df.columns:
        df["meteo_temp_max"] = df["meteo_temp_max"].fillna(df["temp_high_c"])
    if "meteo_temp_min" in df.columns and "temp_low_c" in df.columns:
        df["meteo_temp_min"] = df["meteo_temp_min"].fillna(df["temp_low_c"])
    
    # For other meteo columns, use forward fill then 0
    cols_to_ffill = ["meteo_wind_max", "meteo_weather_code"]
    for col in cols_to_ffill:
        if col in df.columns:
            df[col] = df[col].ffill().fillna(0)

    numeric_fill_cols: Iterable[str] = [
        "aqi_value",
        "wind_level",
        "meteo_precip_sum",
        "meteo_rain_sum",
        "meteo_snowfall_sum",
        "meteo_precip_hours",
    ]
    for col in numeric_fill_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df = build_calendar_features(df)
    df = add_lag_features(df)
    df = add_meteorological_features(df)
    
    # DEBUG: Print columns with NaN values in the last 15 rows before dropna
    last_rows = df.tail(15)
    nan_cols = last_rows.columns[last_rows.isna().any()].tolist()
    if nan_cols:
        print("\n[DEBUG] Columns with NaN in last 15 rows:", nan_cols)
        print("[DEBUG] NaN counts in last 15 rows:\n", last_rows[nan_cols].isna().sum())
        print("[DEBUG] First row of last 15 rows:\n", last_rows.iloc[0])

    df = df.dropna().reset_index(drop=True)

    # English-only processed dataset.
    processed_cols = [
        "date",
        "tourism_num",
        "temp_high_c",
        "temp_low_c",
        "temp_range_c",
        "weather_code_en",
        "weather_abbr",
        "weather_code_id",
        "wind_dir_en",
        "wind_level",
        "wind_dir_id",
        "aqi_value",
        "aqi_level_en",
        "weather_source",
        "meteo_weather_code",
        "meteo_temp_max",
        "meteo_temp_min",
        "meteo_wind_max",
        "meteo_precip_sum",
        "meteo_rain_sum",
        "meteo_snowfall_sum",
        "meteo_precip_hours",
        "day_of_week",
        "month",
        "day_of_month",
        "day_of_year",
        "week_of_year",
        "is_weekend",
        "is_holiday",
        "is_workday",
        "month_sin",
        "month_cos",
        "dow_sin",
        "dow_cos",
        "tourism_num_lag_1",
        "tourism_num_lag_7",
        "tourism_num_lag_14",
        "tourism_num_lag_28",
        "tourism_num_rolling_mean_7",
        "tourism_num_rolling_std_7",
        "tourism_num_rolling_mean_14",
        # 新增标准化特征（共8个新特征）
        "tourism_num_lag_7_scaled",    # 标准化滞后7天客流
        "meteo_precip_sum_scaled",     # 标准化降水量
        "temp_high_scaled",            # 标准化最高温度
        "temp_low_scaled",             # 标准化最低温度
    ]
    
    # Ensure all columns exist before selecting
    for col in processed_cols:
        if col not in df.columns:
            df[col] = 0

    out_df = df[processed_cols].copy()
    out_df["date"] = out_df["date"].dt.strftime("%Y-%m-%d")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    metadata = {
        "input_csv": str(input_csv),
        "raw_output_csv": str(raw_output_csv),
        "output_csv": str(output_csv),
        "rows": int(len(out_df)),
        "columns": out_df.columns.tolist(),
        "weather_id_map": weather_id_map,
        "wind_id_map": wind_id_map,
    }
    metadata_json.parent.mkdir(parents=True, exist_ok=True)
    metadata_json.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"preprocess done: {output_csv} ({len(out_df)} rows)")


def get_latest_raw_csv(raw_dir: Path) -> Path:
    """Helper to find the latest raw CSV file, preferring those starting with 'jiuzhaigou_raw_'."""
    if not raw_dir.exists():
        raise FileNotFoundError(f"Directory not found: {raw_dir}")
        
    files = list(raw_dir.glob("*.csv"))
    # Prefer files starting with jiuzhaigou_raw_
    raw_files = [f for f in files if f.name.startswith("jiuzhaigou_raw_")]
    
    if raw_files:
        return max(raw_files, key=os.path.getmtime)
    
    # Fallback to any csv except known intermediate files if possible
    valid_files = [f for f in files if "latest" not in f.name]
    if valid_files:
        return max(valid_files, key=os.path.getmtime)
        
    if files:
        return max(files, key=os.path.getmtime)
        
    raise FileNotFoundError(f"No CSV files found in {raw_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess Jiuzhaigou dataset for LSTM.")
    parser.add_argument(
        "--input-csv",
        default=None,
        help="Path to merged raw csv. If not provided, auto-detects latest in data/raw.",
    )
    parser.add_argument(
        "--raw-output-csv",
        default="data/raw/jiuzhaigou_tourism_weather_2024_2026_latest.csv",
        help="Path for standardized English raw csv.",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Path to processed feature csv. If not provided, will be auto-generated based on date range.",
    )
    parser.add_argument(
        "--metadata-json",
        default="data/processed/feature_metadata.json",
        help="Path to metadata output json.",
    )
    parser.add_argument("--latitude", type=float, default=33.252, help="Jiuzhaigou latitude.")
    parser.add_argument("--longitude", type=float, default=103.918, help="Jiuzhaigou longitude.")
    args, _ = parser.parse_known_args()

    # Handle input_csv logic
    if args.input_csv:
        input_csv_path = Path(args.input_csv)
    else:
        # Auto-detect latest raw csv
        project_root = Path(__file__).resolve().parents[2]
        raw_dir = project_root / "data" / "raw"
        try:
            input_csv_path = get_latest_raw_csv(raw_dir)
            print(f"Auto-detected latest input CSV: {input_csv_path}")
        except FileNotFoundError:
            # Fallback default
            input_csv_path = Path("data/raw/jiuzhaigou_tourism_weather_2024_2026_latest.csv")
            print(f"Warning: Could not auto-detect raw csv, using default: {input_csv_path}")

    preprocess(
        input_csv=input_csv_path,
        raw_output_csv=Path(args.raw_output_csv),
        output_csv=Path(args.output_csv) if args.output_csv else None,
        metadata_json=Path(args.metadata_json),
        latitude=args.latitude,
        longitude=args.longitude,
    )


if __name__ == "__main__":
    main()
