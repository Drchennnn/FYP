"""
整合爬虫脚本：自动执行客流爬取、天气爬取和数据合并。
"""

import os
import argparse
import datetime as dt
from tourism_num import scrape as scrape_tourism
from weather_jiuzhaigou import scrape_weather, filter_rows_by_date, parse_iso_date as parse_date, month_diff_inclusive, CITY_MAP
from merge_tourism_weather import read_csv_by_date, merge_rows
import csv

def main():
    parser = argparse.ArgumentParser(description="Integrated Crawler for Jiuzhaigou Tourism & Weather")
    parser.add_argument("--start-date", default="2024-01-01", help="YYYY-MM-DD")
    parser.add_argument("--end-date", default=dt.date.today().strftime("%Y-%m-%d"), help="YYYY-MM-DD")
    # 先不设默认值，后面根据解析结果动态生成
    parser.add_argument("--output", default=None, help="Output CSV path")
    parser.add_argument("--max-pages", type=int, default=10, help="Max pages for tourism crawler")
    args, _ = parser.parse_known_args()

    # 动态生成默认输出路径
    if args.output is None:
        filename = f"jiuzhaigou_raw_{args.start_date}_{args.end_date}.csv"
        # 修正：之前用了 ".." 导致存到了项目外面。
        # 如果是在项目根目录运行脚本，应该直接指向 data/raw
        # 或者更稳妥地使用绝对路径
        current_file_path = os.path.abspath(__file__)
        crawler_dir = os.path.dirname(current_file_path) # D:\vscode\FYP\Jiuzhaigou_Crawler
        project_root = os.path.dirname(crawler_dir)      # D:\vscode\FYP
        args.output = os.path.join(project_root, "data", "raw", filename)

    print(f"=== Starting Integrated Crawler ===")
    print(f"Range: {args.start_date} to {args.end_date}")
    
    start_date = parse_date(args.start_date)
    end_date = parse_date(args.end_date)
    
    # 1. 爬取客流数据
    print("\n[Step 1/3] Scraping Tourism Numbers...")
    tourism_data = {}
    # tourism_num.scrape returns generator of (date_str, num_str)
    for date_str, num_str in scrape_tourism(args.max_pages, 0.5, 1.5):
        d = parse_date(date_str)
        if start_date and d < start_date: continue
        if end_date and d > end_date: continue
        tourism_data[date_str] = {"date": date_str, "tourism_num": num_str}
    print(f"Collected {len(tourism_data)} tourism records.")

    # 2. 爬取天气数据
    print("\n[Step 2/3] Scraping Weather Data...")
    # Calculate months back needed
    today = dt.date.today()
    crawl_end = min(end_date, today) if end_date else today
    months_back = month_diff_inclusive(start_date, crawl_end) if start_date else 24
    
    weather_list = scrape_weather("jiuzhaigou", months_back, 0.3)
    weather_list = filter_rows_by_date(weather_list, start_date, end_date)
    weather_data = {row["date"]: row for row in weather_list}
    print(f"Collected {len(weather_data)} weather records.")

    # 3. 合并数据
    print("\n[Step 3/3] Merging Data...")
    # 转换 tourism_data 为 merge_rows 需要的格式 (Dict[str, Dict])
    # tourism_data 已经是这个格式了
    
    # 转换 weather_data 为 merge_rows 需要的格式
    # weather_data 已经是这个格式了

    # 调用合并逻辑
    # 注意：这里我们简化了逻辑，直接在内存中处理，不需要写中间文件
    # 但为了复用 merge_rows 函数，我们需要适配它的参数
    # merge_rows 需要的是 Dict[str, Dict]
    
    # 我们需要模拟 read_csv_by_date 的返回结果，其实就是 tourism_data 和 weather_data
    
    # 只有当有客流数据时，才进行合并 (以客流为主表)
    # 或者取并集？通常以客流为主，因为没有客流的数据对训练无用
    
    merged_list = []
    # 复用 merge_rows 的核心逻辑，或者直接重写一个简单的
    # 这里我们直接调用 merge_rows 函数 (需确保 import 正确)
    # 由于 merge_rows 在 merge_tourism_weather.py 中定义，我们已经 import 了
    
    # 注意：merge_rows 内部会调用 Open-Meteo 补全缺失天气
    merged_list = merge_rows(
        tourism_rows=tourism_data,
        weather_rows=weather_data,
        start_date=start_date,
        end_date=end_date,
        fill_missing_weather=True, # 默认开启补全
        latitude=33.252,
        longitude=103.918
    )
    
    # 4. 保存结果
    output_path = args.output
    # 确保目录存在
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    fieldnames = [
        "date", "tourism_num", "temperature_high", "temperature_low", 
        "weather", "wind", "air_quality", "weather_source"
    ]
    
    with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged_list)
        
    print(f"\n[Success] Merged data saved to: {output_path}")
    print(f"Total records: {len(merged_list)}")

if __name__ == "__main__":
    main()
