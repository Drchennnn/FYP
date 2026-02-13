"""Scrape Jiuzhaigou tourism numbers from https://www.jiuzhai.com/news/number-of-tourists."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import random
import re
import time
from typing import Iterable, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.jiuzhai.com/news/number-of-tourists"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/121.0.0.0 Safari/537.36"
    )
}


def fetch_page(session: requests.Session, page_index: int) -> List[Tuple[str, str]]:
    params = {}
    if page_index > 0:
        params["start"] = page_index * 20

    response = session.get(BASE_URL, params=params, timeout=20)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "lxml")
    dates = soup.find_all("td", attrs={"class": "list-date small"})
    titles = soup.find_all("td", attrs={"class": "list-title"})
    pattern = re.compile(r"-?\d+")

    rows: List[Tuple[str, str]] = []
    for date_cell, title_cell in zip(dates, titles):
        match = pattern.search(title_cell.get_text(strip=True))
        if match:
            rows.append((date_cell.get_text(strip=True), match.group()))
    return rows


def scrape(max_pages: int, sleep_min: float, sleep_max: float) -> Iterable[Tuple[str, str]]:
    with requests.Session() as session:
        session.headers.update(HEADERS)
        seen_dates = set()
        for page_index in range(max_pages):
            page_rows = fetch_page(session, page_index)
            if not page_rows:
                break

            new_count = 0
            for row in page_rows:
                if row[0] in seen_dates:
                    continue
                seen_dates.add(row[0])
                new_count += 1
                yield row

            print(f"page {page_index} done, new rows: {new_count}")
            if new_count == 0:
                break
            time.sleep(random.uniform(sleep_min, sleep_max))


def parse_iso_date(value: Optional[str]) -> Optional[dt.date]:
    if not value:
        return None
    return dt.datetime.strptime(value, "%Y-%m-%d").date()


def in_date_range(date_text: str, start_date: Optional[dt.date], end_date: Optional[dt.date]) -> bool:
    current = dt.datetime.strptime(date_text, "%Y-%m-%d").date()
    if start_date and current < start_date:
        return False
    if end_date and current > end_date:
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape Jiuzhaigou daily tourist counts.")
    parser.add_argument("--max-pages", type=int, default=200, help="Maximum pages to crawl.")
    parser.add_argument(
        "--output",
        default="tourism_num_jiuzhaigou.csv",
        help="Output CSV path.",
    )
    parser.add_argument("--sleep-min", type=float, default=0.5, help="Minimum sleep seconds.")
    parser.add_argument("--sleep-max", type=float, default=1.5, help="Maximum sleep seconds.")
    parser.add_argument("--start-date", default=None, help="Filter start date (YYYY-MM-DD).")
    parser.add_argument("--end-date", default=None, help="Filter end date (YYYY-MM-DD).")
    args = parser.parse_args()

    start_date = parse_iso_date(args.start_date)
    end_date = parse_iso_date(args.end_date)
    if start_date and end_date and start_date > end_date:
        raise ValueError("start-date cannot be later than end-date")

    rows = list(scrape(args.max_pages, args.sleep_min, args.sleep_max))
    rows = [row for row in rows if in_date_range(row[0], start_date, end_date)]
    rows.sort(key=lambda x: x[0])

    with open(args.output, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "tourism_num"])
        writer.writerows(rows)

    print(f"done, total rows: {len(rows)}, output: {args.output}")


if __name__ == "__main__":
    main()
