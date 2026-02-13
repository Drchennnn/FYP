"""Scrape historical weather data for Jiuzhaigou/Siguniangshan from 2345 weather."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import re
import time
from typing import Dict, Iterable, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup

CITY_MAP = {
    "jiuzhaigou": 60925,
    "siguniangshan": 70752,
}

BASE_HISTORY_URL = "http://tianqi.2345.com/wea_history/{area_id}.htm"
AJAX_HISTORY_URL = "http://tianqi.2345.com/Pc/GetHistory"
HEADERS = {
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Accept-Language": "zh-CN,zh;q=0.9",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/121.0.0.0 Safari/537.36"
    ),
    "X-Requested-With": "XMLHttpRequest",
}


def month_iter(start_year: int, start_month: int, months_back: int) -> Iterable[Tuple[int, int]]:
    year, month = start_year, start_month
    for _ in range(max(months_back, 0)):
        yield year, month
        if month == 1:
            year -= 1
            month = 12
        else:
            month -= 1


def month_diff_inclusive(start_date: dt.date, end_date: dt.date) -> int:
    return (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month) + 1


def parse_iso_date(value: Optional[str]) -> Optional[dt.date]:
    if not value:
        return None
    return dt.datetime.strptime(value, "%Y-%m-%d").date()


def parse_rows(html_fragment: str) -> List[Dict[str, str]]:
    soup = BeautifulSoup(html_fragment, "lxml")
    rows = []
    for tr in soup.select("table.history-table tr"):
        tds = tr.find_all("td")
        if len(tds) < 6:
            continue
        date_text = tds[0].get_text(" ", strip=True)
        date_match = re.search(r"\d{4}-\d{2}-\d{2}", date_text)
        if not date_match:
            continue
        date_value = date_match.group(0)
        rows.append(
            {
                "date": date_value,
                "temperature_high": tds[1].get_text(" ", strip=True),
                "temperature_low": tds[2].get_text(" ", strip=True),
                "weather": tds[3].get_text(" ", strip=True),
                "wind": tds[4].get_text(" ", strip=True),
                "air_quality": tds[5].get_text(" ", strip=True),
            }
        )
    return rows


def fetch_one_month(
    session: requests.Session, area_id: int, year: int, month: int, retries: int = 3
) -> List[Dict[str, str]]:
    params = {
        "areaInfo[areaId]": str(area_id),
        "areaInfo[areaType]": "2",
        "date[year]": str(year),
        "date[month]": str(month),
    }
    last_error: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            response = session.get(AJAX_HISTORY_URL, params=params, timeout=20)
            response.raise_for_status()
            payload = response.json()
            if payload.get("code") != 1:
                return []
            return parse_rows(payload.get("data", ""))
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(min(attempt, 3))
    if last_error:
        raise last_error
    return []


def scrape_weather(city: str, months_back: int, sleep_seconds: float) -> List[Dict[str, str]]:
    area_id = CITY_MAP[city]
    referer = BASE_HISTORY_URL.format(area_id=area_id)
    now = dt.datetime.now()
    collected: Dict[str, Dict[str, str]] = {}

    with requests.Session() as session:
        session.headers.update(HEADERS)
        session.headers["Referer"] = referer

        for year, month in month_iter(now.year, now.month, months_back):
            try:
                month_rows = fetch_one_month(session, area_id, year, month)
            except Exception as exc:  # noqa: BLE001
                print(f"{year}-{month:02d} failed: {exc}")
                month_rows = []
            for row in month_rows:
                collected[row["date"]] = row
            print(f"{year}-{month:02d} done, rows: {len(month_rows)}")
            time.sleep(sleep_seconds)

    return [collected[k] for k in sorted(collected)]


def filter_rows_by_date(
    rows: List[Dict[str, str]], start_date: Optional[dt.date], end_date: Optional[dt.date]
) -> List[Dict[str, str]]:
    filtered: List[Dict[str, str]] = []
    for row in rows:
        current = dt.datetime.strptime(row["date"], "%Y-%m-%d").date()
        if start_date and current < start_date:
            continue
        if end_date and current > end_date:
            continue
        filtered.append(row)
    return filtered


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape historical weather from 2345.")
    parser.add_argument(
        "--city",
        default="jiuzhaigou",
        choices=sorted(CITY_MAP.keys()),
        help="Target city to crawl.",
    )
    parser.add_argument(
        "--months-back",
        type=int,
        default=240,
        help="How many months to crawl backwards from current month.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.3,
        help="Sleep time between monthly requests.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path. Default: weather_<city>.csv",
    )
    parser.add_argument("--start-date", default=None, help="Filter start date (YYYY-MM-DD).")
    parser.add_argument("--end-date", default=None, help="Filter end date (YYYY-MM-DD).")
    args = parser.parse_args()

    start_date = parse_iso_date(args.start_date)
    end_date = parse_iso_date(args.end_date)
    today = dt.date.today()
    crawl_end = min(end_date, today) if end_date else today
    if start_date and crawl_end and start_date > crawl_end:
        raise ValueError("start-date cannot be later than end-date/today")

    months_back = args.months_back
    if start_date:
        months_back = month_diff_inclusive(start_date, crawl_end)

    output = args.output or f"weather_{args.city}.csv"
    rows = scrape_weather(args.city, months_back, args.sleep_seconds)
    rows = filter_rows_by_date(rows, start_date, end_date)
    with open(output, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "date",
                "temperature_high",
                "temperature_low",
                "weather",
                "wind",
                "air_quality",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"done, total rows: {len(rows)}, output: {output}")


if __name__ == "__main__":
    main()
