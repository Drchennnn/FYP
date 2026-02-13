"""Merge tourism and weather CSV files, with optional Open-Meteo fallback for missing weather."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
from typing import Dict, List, Optional

import requests

OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


def parse_iso_date(value: Optional[str]) -> Optional[dt.date]:
    if not value:
        return None
    return dt.datetime.strptime(value, "%Y-%m-%d").date()


def read_csv_by_date(path: str) -> Dict[str, Dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return {row["date"]: row for row in reader if row.get("date")}


def in_date_range(date_text: str, start_date: Optional[dt.date], end_date: Optional[dt.date]) -> bool:
    current = dt.datetime.strptime(date_text, "%Y-%m-%d").date()
    if start_date and current < start_date:
        return False
    if end_date and current > end_date:
        return False
    return True


def fetch_open_meteo_daily(
    latitude: float, longitude: float, start_date: dt.date, end_date: dt.date
) -> Dict[str, Dict[str, str]]:
    archive_params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "daily": "weathercode,temperature_2m_max,temperature_2m_min,windspeed_10m_max",
        "timezone": "Asia/Shanghai",
    }

    data_by_date: Dict[str, Dict[str, str]] = {}
    resp = requests.get(OPEN_METEO_ARCHIVE_URL, params=archive_params, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    daily = payload.get("daily", {})
    dates = daily.get("time", [])
    highs = daily.get("temperature_2m_max", [])
    lows = daily.get("temperature_2m_min", [])
    codes = daily.get("weathercode", [])
    winds = daily.get("windspeed_10m_max", [])
    for i, date_text in enumerate(dates):
        high = highs[i] if i < len(highs) else None
        low = lows[i] if i < len(lows) else None
        code = codes[i] if i < len(codes) else None
        wind = winds[i] if i < len(winds) else None
        data_by_date[date_text] = {
            "date": date_text,
            "temperature_high": "" if high is None else f"{high}C",
            "temperature_low": "" if low is None else f"{low}C",
            "weather": "" if code is None else f"weather_code={code}",
            "wind": "" if wind is None else f"{wind}km/h",
            "air_quality": "",
            "weather_source": "open-meteo",
        }

    today = dt.date.today()
    forecast_start = max(start_date, today)
    if forecast_start <= end_date:
        forecast_params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": forecast_start.isoformat(),
            "end_date": end_date.isoformat(),
            "daily": "weathercode,temperature_2m_max,temperature_2m_min,windspeed_10m_max",
            "timezone": "Asia/Shanghai",
        }
        resp = requests.get(OPEN_METEO_FORECAST_URL, params=forecast_params, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
        daily = payload.get("daily", {})
        dates = daily.get("time", [])
        highs = daily.get("temperature_2m_max", [])
        lows = daily.get("temperature_2m_min", [])
        codes = daily.get("weathercode", [])
        winds = daily.get("windspeed_10m_max", [])
        for i, date_text in enumerate(dates):
            high = highs[i] if i < len(highs) else None
            low = lows[i] if i < len(lows) else None
            code = codes[i] if i < len(codes) else None
            wind = winds[i] if i < len(winds) else None
            data_by_date[date_text] = {
                "date": date_text,
                "temperature_high": "" if high is None else f"{high}C",
                "temperature_low": "" if low is None else f"{low}C",
                "weather": "" if code is None else f"weather_code={code}",
                "wind": "" if wind is None else f"{wind}km/h",
                "air_quality": "",
                "weather_source": "open-meteo",
            }
    return data_by_date


def merge_rows(
    tourism_rows: Dict[str, Dict[str, str]],
    weather_rows: Dict[str, Dict[str, str]],
    start_date: Optional[dt.date],
    end_date: Optional[dt.date],
    fill_missing_weather: bool,
    latitude: float,
    longitude: float,
) -> List[Dict[str, str]]:
    selected_dates = sorted(d for d in tourism_rows if in_date_range(d, start_date, end_date))
    missing_weather_dates = [d for d in selected_dates if d not in weather_rows]

    fallback_rows: Dict[str, Dict[str, str]] = {}
    if fill_missing_weather and missing_weather_dates:
        fallback_start = dt.datetime.strptime(min(missing_weather_dates), "%Y-%m-%d").date()
        fallback_end = dt.datetime.strptime(max(missing_weather_dates), "%Y-%m-%d").date()
        fallback_rows = fetch_open_meteo_daily(latitude, longitude, fallback_start, fallback_end)

    merged: List[Dict[str, str]] = []
    for date_text in selected_dates:
        tourism = tourism_rows[date_text]
        weather = weather_rows.get(date_text)
        if weather:
            merged.append(
                {
                    "date": date_text,
                    "tourism_num": tourism.get("tourism_num", ""),
                    "temperature_high": weather.get("temperature_high", ""),
                    "temperature_low": weather.get("temperature_low", ""),
                    "weather": weather.get("weather", ""),
                    "wind": weather.get("wind", ""),
                    "air_quality": weather.get("air_quality", ""),
                    "weather_source": "2345",
                }
            )
            continue

        fallback = fallback_rows.get(date_text, {})
        merged.append(
            {
                "date": date_text,
                "tourism_num": tourism.get("tourism_num", ""),
                "temperature_high": fallback.get("temperature_high", ""),
                "temperature_low": fallback.get("temperature_low", ""),
                "weather": fallback.get("weather", ""),
                "wind": fallback.get("wind", ""),
                "air_quality": fallback.get("air_quality", ""),
                "weather_source": fallback.get("weather_source", ""),
            }
        )
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge tourism + weather CSV into one file.")
    parser.add_argument("--tourism-csv", default="tourism_num_jiuzhaigou.csv")
    parser.add_argument("--weather-csv", default="weather_jiuzhaigou.csv")
    parser.add_argument("--output", default="jiuzhaigou_tourism_weather_merged.csv")
    parser.add_argument("--start-date", default=None, help="YYYY-MM-DD")
    parser.add_argument("--end-date", default=None, help="YYYY-MM-DD")
    parser.add_argument("--fill-missing-weather", action="store_true")
    parser.add_argument("--latitude", type=float, default=33.252)
    parser.add_argument("--longitude", type=float, default=103.918)
    args = parser.parse_args()

    start_date = parse_iso_date(args.start_date)
    end_date = parse_iso_date(args.end_date)
    if start_date and end_date and start_date > end_date:
        raise ValueError("start-date cannot be later than end-date")

    tourism_rows = read_csv_by_date(args.tourism_csv)
    weather_rows = read_csv_by_date(args.weather_csv)

    merged_rows = merge_rows(
        tourism_rows=tourism_rows,
        weather_rows=weather_rows,
        start_date=start_date,
        end_date=end_date,
        fill_missing_weather=args.fill_missing_weather,
        latitude=args.latitude,
        longitude=args.longitude,
    )

    with open(args.output, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "date",
                "tourism_num",
                "temperature_high",
                "temperature_low",
                "weather",
                "wind",
                "air_quality",
                "weather_source",
            ],
        )
        writer.writeheader()
        writer.writerows(merged_rows)
    print(f"done, total rows: {len(merged_rows)}, output: {args.output}")


if __name__ == "__main__":
    main()
