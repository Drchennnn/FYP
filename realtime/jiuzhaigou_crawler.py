"""
九寨沟官网爬虫模块

功能：
1. 爬取九寨沟官网最新客流数据（昨天的数据）
2. 数据验证和清洗
3. 保存到数据库
"""

import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import re
from typing import Optional, Dict
import sqlite3
from pathlib import Path


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
    
    def fetch_latest_visitor_count(self) -> Optional[Dict]:
        """
        爬取最新客流数据（昨天的数据）
        
        Returns:
            {
                'date': '2026-04-02',
                'visitor_count': 15000,
                'crawled_at': '2026-04-03 17:00:00'
            }
        """
        try:
            # 九寨沟官网客流数据页面
            # 注意：实际 URL 可能需要根据官网结构调整
            url = f'{self.base_url}/index.php'
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            response.encoding = 'utf-8'
            
            if response.status_code != 200:
                print(f'Failed to fetch page: HTTP {response.status_code}')
                return None
            
            # 解析 HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 方法 1: 查找包含"游客"或"人数"的元素
            visitor_count = self._extract_visitor_count_method1(soup)
            
            if visitor_count is None:
                # 方法 2: 查找特定 class 或 id
                visitor_count = self._extract_visitor_count_method2(soup)
            
            if visitor_count is None:
                print('Failed to extract visitor count from page')
                return None
            
            # 昨天的日期
            yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
            
            data = {
                'date': yesterday,
                'visitor_count': visitor_count,
                'crawled_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # 保存到数据库
            self._save_to_database(data)
            
            return data
            
        except Exception as e:
            print(f'Error crawling Jiuzhaigou website: {e}')
            return None
    
    def _extract_visitor_count_method1(self, soup: BeautifulSoup) -> Optional[int]:
        """
        方法 1: 通过关键词搜索提取客流数据
        
        常见模式：
        - "今日游客：15000人"
        - "游客人数：15000"
        - "接待游客 15000 人次"
        """
        # 搜索包含"游客"或"人数"的文本
        patterns = [
            r'游客[：:]\s*(\d+)',
            r'人数[：:]\s*(\d+)',
            r'接待.*?(\d+).*?人',
            r'(\d+).*?人次'
        ]
        
        text = soup.get_text()
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    count = int(match.group(1))
                    # 验证数据合理性（1000 - 80000）
                    if 1000 <= count <= 80000:
                        return count
                except ValueError:
                    continue
        
        return None
    
    def _extract_visitor_count_method2(self, soup: BeautifulSoup) -> Optional[int]:
        """
        方法 2: 通过特定 HTML 元素提取
        
        需要根据实际网站结构调整选择器
        """
        # 常见的 class 或 id 名称
        selectors = [
            '.visitor-count',
            '#visitor-count',
            '.tourist-number',
            '#tourist-number',
            '[data-visitor-count]'
        ]
        
        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                text = element.get_text()
                match = re.search(r'(\d+)', text)
                if match:
                    try:
                        count = int(match.group(1))
                        if 1000 <= count <= 80000:
                            return count
                    except ValueError:
                        continue
        
        return None
    
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
