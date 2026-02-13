"""Preprocess Jiuzhaigou tourism-weather data to English-only model features."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

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
    df["is_holiday"] = ds.apply(lambda d: int(cncal.is_holiday(d.date())))
    df["is_workday"] = ds.apply(lambda d: int(cncal.is_workday(d.date())))
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    return df


def add_lag_features(df: pd.DataFrame, target_col: str = "tourism_num") -> pd.DataFrame:
    for lag in (1, 7, 14, 28):
        df[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)
    df[f"{target_col}_rolling_mean_7"] = df[target_col].rolling(7).mean()
    df[f"{target_col}_rolling_std_7"] = df[target_col].rolling(7).std()
    df[f"{target_col}_rolling_mean_14"] = df[target_col].rolling(14).mean()
    return df


def preprocess(input_csv: Path, raw_output_csv: Path, output_csv: Path, metadata_json: Path, latitude: float, longitude: float) -> None:
    df = pd.read_csv(input_csv, encoding="utf-8-sig")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)
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
    ]
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess Jiuzhaigou dataset for LSTM.")
    parser.add_argument(
        "--input-csv",
        default="data/raw/jiuzhaigou_tourism_weather_2024_2026_latest.csv",
        help="Path to merged raw csv.",
    )
    parser.add_argument(
        "--raw-output-csv",
        default="data/raw/jiuzhaigou_tourism_weather_2024_2026_latest.csv",
        help="Path for standardized English raw csv.",
    )
    parser.add_argument(
        "--output-csv",
        default="data/processed/jiuzhaigou_daily_features.csv",
        help="Path to processed feature csv.",
    )
    parser.add_argument(
        "--metadata-json",
        default="data/processed/feature_metadata.json",
        help="Path to metadata output json.",
    )
    parser.add_argument("--latitude", type=float, default=33.252, help="Jiuzhaigou latitude.")
    parser.add_argument("--longitude", type=float, default=103.918, help="Jiuzhaigou longitude.")
    args = parser.parse_args()

    preprocess(
        input_csv=Path(args.input_csv),
        raw_output_csv=Path(args.raw_output_csv),
        output_csv=Path(args.output_csv),
        metadata_json=Path(args.metadata_json),
        latitude=args.latitude,
        longitude=args.longitude,
    )


if __name__ == "__main__":
    main()
