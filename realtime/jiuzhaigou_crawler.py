"""
九寨沟官网爬虫模块

功能：
1. 爬取九寨沟官网最新客流数据（昨天的数据）
2. 数据验证和清洗
3. 保存到数据库
"""

import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta, date
import re
import time
import random
from typing import Optional, Dict, List
import sqlite3
from pathlib import Path

TOURIST_URL = 'https://www.jiuzhai.com/news/number-of-tourists'
_HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}


class JiuzhaigouCrawler:
    """九寨沟官网爬虫"""
    
    def __init__(self, db_path: str = 'data/jiuzhaigou_realtime.db'):
        """
        Args:
            db_path: SQLite 数据库路径
        """
        self.base_url = 'http://www.jiuzhai.com'
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """初始化数据库"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 创建客流数据表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS visitor_data (
                date TEXT PRIMARY KEY,
                visitor_count INTEGER NOT NULL,
                crawled_at TEXT NOT NULL,
                source TEXT DEFAULT 'jiuzhai.com'
            )
        ''')
        
        # 创建预测历史表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_date TEXT NOT NULL,
                target_date TEXT NOT NULL,
                predicted_value INTEGER NOT NULL,
                actual_value INTEGER,
                model_version TEXT,
                created_at TEXT NOT NULL,
                UNIQUE(prediction_date, target_date)
            )
        ''')
        
        conn.commit()
        conn.close()
        
        print(f'Database initialized: {self.db_path}')
    
    def _fetch_page(self, start: int = 0) -> List[Dict]:
        """爬取一页数据（20条），返回 [{date, visitor_count}] 列表。"""
        url = TOURIST_URL if start == 0 else f'{TOURIST_URL}?start={start}'
        r = requests.get(url, headers=_HEADERS, timeout=15)
        r.encoding = 'utf-8'
        if r.status_code != 200:
            raise RuntimeError(f'HTTP {r.status_code}')
        soup = BeautifulSoup(r.text, 'lxml')
        dates = soup.find_all('td', attrs={'class': 'list-date small'})
        nums  = soup.find_all('td', attrs={'class': 'list-title'})
        results = []
        for d, n in zip(dates, nums):
            date_str = d.text.strip()
            m = re.search(r'\d+', n.text)
            if m and re.match(r'\d{4}-\d{2}-\d{2}', date_str):
                count = int(m.group())
                if 100 <= count <= 200000:
                    results.append({'date': date_str, 'visitor_count': count})
        return results

    def fetch_latest_visitor_count(self) -> Optional[Dict]:
        """爬取官网第一页，返回最新一天（昨天）的数据。"""
        try:
            rows = self._fetch_page(0)
            if not rows:
                print('Failed to extract visitor count from page')
                return None
            # 第一页第一条即最新数据
            data = {**rows[0], 'crawled_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            self._save_to_database(data)
            return data
        except Exception as e:
            print(f'Error crawling Jiuzhaigou website: {e}')
            return None

    def fetch_by_date_range(self, start_date: str, end_date: str) -> List[Dict]:
        """
        爬取指定日期范围内的历史客流数据。

        Args:
            start_date: 起始日期 'YYYY-MM-DD'（含）
            end_date:   截止日期 'YYYY-MM-DD'（含）

        Returns:
            按日期升序排列的 [{date, visitor_count, crawled_at}] 列表
        """
        start = date.fromisoformat(start_date)
        end   = date.fromisoformat(end_date)
        needed = {d.isoformat() for d in (start + timedelta(n) for n in range((end - start).days + 1))}
        collected: Dict[str, int] = {}

        page = 0
        while needed - set(collected):
            try:
                rows = self._fetch_page(page * 20)
            except Exception as e:
                print(f'  页面 {page} 爬取失败: {e}')
                break
            if not rows:
                break  # 没有更多数据

            for row in rows:
                d = row['date']
                if d in needed:
                    collected[d] = row['visitor_count']
                # 如果页面最旧日期已早于 start，停止翻页
                if d < start_date:
                    needed.clear()  # 触发退出
                    break

            print(f'  第 {page+1} 页：已收集 {len(collected)}/{len(needed) + len(collected)} 条')
            page += 1
            time.sleep(random.uniform(1, 3))

        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        result = []
        for d_str in sorted(collected):
            entry = {'date': d_str, 'visitor_count': collected[d_str], 'crawled_at': now_str}
            self._save_to_database(entry)
            result.append(entry)

        missing = sorted(needed - set(collected))
        if missing:
            print(f'  警告：以下日期未能爬取到数据: {missing}')

        return result
    
    def _save_to_database(self, data: Dict):
        """保存数据到数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO visitor_data (date, visitor_count, crawled_at)
                VALUES (?, ?, ?)
            ''', (data['date'], data['visitor_count'], data['crawled_at']))
            
            conn.commit()
            print(f"Saved to database: {data['date']} - {data['visitor_count']} visitors")
            
        except Exception as e:
            print(f'Error saving to database: {e}')
            conn.rollback()
        finally:
            conn.close()
    
    def get_latest_from_database(self) -> Optional[Dict]:
        """从数据库获取最新数据"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT date, visitor_count, crawled_at
            FROM visitor_data
            ORDER BY date DESC
            LIMIT 1
        ''')
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                'date': row[0],
                'visitor_count': row[1],
                'crawled_at': row[2]
            }
        
        return None
    
    def get_historical_data(self, days: int = 40) -> list:
        """
        从数据库获取历史数据
        
        Args:
            days: 获取最近多少天的数据
            
        Returns:
            List of dicts with date and visitor_count
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT date, visitor_count
            FROM visitor_data
            ORDER BY date DESC
            LIMIT ?
        ''', (days,))
        
        rows = cursor.fetchall()
        conn.close()
        
        # 反转顺序（从旧到新）
        return [{'date': row[0], 'visitor_count': row[1]} for row in reversed(rows)]
    
    def save_prediction(
        self, 
        prediction_date: str,
        target_date: str,
        predicted_value: int,
        model_version: str = 'GRU-v1'
    ):
        """
        保存预测结果
        
        Args:
            prediction_date: 预测生成日期
            target_date: 预测目标日期
            predicted_value: 预测值
            model_version: 模型版本
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO prediction_history 
                (prediction_date, target_date, predicted_value, model_version, created_at)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                prediction_date,
                target_date,
                predicted_value,
                model_version,
                datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            ))
            
            conn.commit()
            
        except Exception as e:
            print(f'Error saving prediction: {e}')
            conn.rollback()
        finally:
            conn.close()
    
    def update_prediction_with_actual(self, target_date: str, actual_value: int):
        """
        更新预测记录的实际值
        
        Args:
            target_date: 目标日期
            actual_value: 实际客流量
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                UPDATE prediction_history
                SET actual_value = ?
                WHERE target_date = ? AND actual_value IS NULL
            ''', (actual_value, target_date))
            
            conn.commit()
            print(f'Updated actual value for {target_date}: {actual_value}')
            
        except Exception as e:
            print(f'Error updating actual value: {e}')
            conn.rollback()
        finally:
            conn.close()
    
    def get_prediction_accuracy(self, days: int = 30) -> Dict:
        """
        计算预测准确度
        
        Args:
            days: 计算最近多少天的准确度
            
        Returns:
            {
                'mae': 平均绝对误差,
                'rmse': 均方根误差,
                'mape': 平均绝对百分比误差,
                'count': 样本数
            }
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 获取有实际值的预测记录
        cursor.execute('''
            SELECT predicted_value, actual_value
            FROM prediction_history
            WHERE actual_value IS NOT NULL
            AND target_date >= date('now', '-' || ? || ' days')
        ''', (days,))
        
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            return {'mae': 0, 'rmse': 0, 'mape': 0, 'count': 0}
        
        import numpy as np
        
        predicted = np.array([row[0] for row in rows])
        actual = np.array([row[1] for row in rows])
        
        mae = np.mean(np.abs(predicted - actual))
        rmse = np.sqrt(np.mean((predicted - actual) ** 2))
        mape = np.mean(np.abs((predicted - actual) / actual)) * 100
        
        return {
            'mae': float(mae),
            'rmse': float(rmse),
            'mape': float(mape),
            'count': len(rows)
        }


if __name__ == '__main__':
    # 测试爬虫
    crawler = JiuzhaigouCrawler()
    
    print('Testing Jiuzhaigou crawler...')
    print()
    
    # 测试爬取
    data = crawler.fetch_latest_visitor_count()
    if data:
        print('Crawled data:', data)
    else:
        print('Failed to crawl data (using fallback)')
        # 如果爬取失败，从数据库获取最新数据
        data = crawler.get_latest_from_database()
        if data:
            print('Latest from database:', data)
    
    print()
    
    # 测试历史数据
    historical = crawler.get_historical_data(days=10)
    print(f'Historical data (last 10 days): {len(historical)} records')
    
    print()
    
    # 测试预测准确度
    accuracy = crawler.get_prediction_accuracy(days=30)
    print('Prediction accuracy (last 30 days):', accuracy)
