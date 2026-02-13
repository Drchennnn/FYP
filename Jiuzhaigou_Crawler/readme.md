# Jiuzhaigou Crawler

用于抓取九寨沟旅游人数和历史天气数据。

## 1. 环境要求

- Python 3.9+
- Windows / macOS / Linux

## 2. 安装依赖

```bash
pip install -r requirements.txt
```

## 3. 旅游人数爬虫

脚本：`tourism_num.py`

```bash
python tourism_num.py
```

常用参数：

```bash
python tourism_num.py --max-pages 200 --output tourism_num_jiuzhaigou.csv
```

日期范围：

```bash
python tourism_num.py --start-date 2024-01-01 --end-date 2026-12-31
```

输出字段：

- `date`
- `tourism_num`

## 4. 历史天气爬虫

脚本：`weather_jiuzhaigou.py`

```bash
python weather_jiuzhaigou.py --city jiuzhaigou
```

常用参数：

```bash
python weather_jiuzhaigou.py --city jiuzhaigou --months-back 240 --output weather_jiuzhaigou.csv
```

日期范围：

```bash
python weather_jiuzhaigou.py --city jiuzhaigou --start-date 2024-01-01 --end-date 2026-12-31
```

`--city` 可选：

- `jiuzhaigou`
- `siguniangshan`

输出字段：

- `date`
- `temperature_high`
- `temperature_low`
- `weather`
- `wind`
- `air_quality`

## 5. 说明

- 当前版本已移除 `PhantomJS` 和旧版 `selenium` 依赖。
- 直接使用 `requests + BeautifulSoup` 抓取，不需要浏览器驱动。
- 天气主数据来源：`tianqi.2345.com` 的历史天气接口 `/Pc/GetHistory`。

## 6. 合并输出（人数+天气）

脚本：`merge_tourism_weather.py`

```bash
python merge_tourism_weather.py --start-date 2024-01-01 --end-date 2026-12-31 --fill-missing-weather
```

说明：

- 合并基于 `date` 字段。
- 若某些日期在 2345 天气缺失，可加 `--fill-missing-weather`，自动从 Open-Meteo 补齐缺失天气。
