"""
实时数据获取模块

功能：
1. 从九寨沟官网获取最新客流数据（昨天的数据）
2. 从 Open-Meteo API 获取实时天气和未来预报
3. 缓存机制避免频繁请求
"""

import requests
from datetime import datetime, timedelta
import pandas as pd
from typing import Dict, Optional
import time
import sys
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from realtime.jiuzhaigou_crawler import JiuzhaigouCrawler

class RealtimeDataFetcher:
    """实时数据获取器"""
    
    def __init__(self, cache_ttl: int = 3600):
        """
        Args:
            cache_ttl: 缓存有效期（秒），默认 1 小时（因为数据是昨天的，不需要频繁刷新）
        """
        self.cache_ttl = cache_ttl
        self._cache = {}
        self._cache_time = {}
        
        # 九寨沟坐标
        self.lat = 33.2
        self.lon = 103.9
        
        # 初始化爬虫
        self.crawler = JiuzhaigouCrawler()
    
    def get_current_visitor_count(self) -> Optional[Dict]:
        """
        获取最新客流数据（昨天的数据）
        
        Returns:
            {
                'date': '2026-04-02',
                'visitor_count': 15000,
                'timestamp': '2026-04-03 17:00:00',
                'is_realtime': False  # 实际是昨天的数据
            }
        """
        cache_key = 'visitor_count'
        
        # 检查缓存
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]
        
        try:
            # 使用真实爬虫获取数据
            data = self.crawler.fetch_latest_visitor_count()
            
            if data is None:
                # 如果爬取失败，从数据库获取最新数据
                data = self.crawler.get_latest_from_database()
            
            if data:
                # 转换为统一格式
                result = {
                    'date': data['date'],
                    'visitor_count': data['visitor_count'],
                    'timestamp': data['crawled_at'],
                    'is_realtime': False  # 标记为昨天的数据
                }
                
                # 更新缓存
                self._cache[cache_key] = result
                self._cache_time[cache_key] = time.time()
                
                return result
            else:
                return None
            
        except Exception as e:
            print(f'Error fetching visitor count: {e}')
            return None
    
    def get_current_weather(self) -> Optional[Dict]:
        """
        获取当天实时天气
        
        Returns:
            {
                'date': '2026-04-03',
                'temperature': 18.5,
                'precipitation': 0.0,
                'temp_high': 22.0,
                'temp_low': 12.0,
                'timestamp': '2026-04-03 16:30:00'
            }
        """
        cache_key = 'current_weather'
        
        # 检查缓存
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]
        
        try:
            # Open-Meteo API: 当前天气
            url = 'https://api.open-meteo.com/v1/forecast'
            params = {
                'latitude': self.lat,
                'longitude': self.lon,
                'current': 'temperature_2m,precipitation',
                'daily': 'temperature_2m_max,temperature_2m_min',
                'timezone': 'Asia/Shanghai',
                'forecast_days': 1
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            result = {
                'date': datetime.now().strftime('%Y-%m-%d'),
                'temperature': data['current']['temperature_2m'],
                'precipitation': data['current']['precipitation'],
                'temp_high': data['daily']['temperature_2m_max'][0],
                'temp_low': data['daily']['temperature_2m_min'][0],
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # 更新缓存
            self._cache[cache_key] = result
            self._cache_time[cache_key] = time.time()
            
            return result
            
        except Exception as e:
            print(f'Error fetching weather: {e}')
            return None
    
    def get_weather_forecast(self, days: int = 7) -> Optional[pd.DataFrame]:
        """
        获取未来天气预报
        
        Args:
            days: 预报天数
            
        Returns:
            DataFrame with columns: date, temp_high, temp_low, precipitation
        """
        cache_key = f'weather_forecast_{days}'
        
        # 检查缓存
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]
        
        try:
            url = 'https://api.open-meteo.com/v1/forecast'
            params = {
                'latitude': self.lat,
                'longitude': self.lon,
                'daily': 'temperature_2m_max,temperature_2m_min,precipitation_sum',
                'timezone': 'Asia/Shanghai',
                'forecast_days': days
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            df = pd.DataFrame({
                'date': data['daily']['time'],
                'temp_high': data['daily']['temperature_2m_max'],
                'temp_low': data['daily']['temperature_2m_min'],
                'precipitation': data['daily']['precipitation_sum']
            })
            
            # 更新缓存
            self._cache[cache_key] = df
            self._cache_time[cache_key] = time.time()
            
            return df
            
        except Exception as e:
            print(f'Error fetching weather forecast: {e}')
            return None
    
    def _is_cache_valid(self, key: str) -> bool:
        """检查缓存是否有效"""
        if key not in self._cache:
            return False
        
        elapsed = time.time() - self._cache_time[key]
        return elapsed < self.cache_ttl


if __name__ == '__main__':
    # 测试
    fetcher = RealtimeDataFetcher(cache_ttl=300)
    
    print('Testing realtime data fetcher...')
    print()
    
    # 测试当前客流
    visitor = fetcher.get_current_visitor_count()
    print('Current visitor count:', visitor)
    print()
    
    # 测试当前天气
    weather = fetcher.get_current_weather()
    print('Current weather:', weather)
    print()
    
    # 测试天气预报
    forecast = fetcher.get_weather_forecast(days=7)
    print('Weather forecast:')
    print(forecast)
