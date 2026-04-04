"""
定时任务脚本 - 每日自动爬取和更新

功能：
1. 每日爬取九寨沟官网最新数据
2. 更新预测历史中的实际值
3. 可配合 cron 或 Windows 任务计划程序使用

使用方法：
- Linux/Mac: 添加到 crontab
  0 9 * * * cd /path/to/FYP && python realtime/daily_update.py
  
- Windows: 使用任务计划程序
  每天 9:00 AM 运行此脚本
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from realtime.jiuzhaigou_crawler import JiuzhaigouCrawler


def daily_update():
    """每日更新任务"""
    print('='*80)
    print(f'Daily Update Task - {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print('='*80)
    print()
    
    crawler = JiuzhaigouCrawler()
    
    # 1. 爬取最新数据
    print('Step 1: Fetching latest visitor data...')
    try:
        data = crawler.fetch_latest_visitor_count()
        
        if data:
            print(f'✅ Successfully fetched data for {data["date"]}')
            print(f'   Visitor count: {data["visitor_count"]:,}')
            
            # 2. 更新预测历史中的实际值
            print()
            print('Step 2: Updating prediction history with actual values...')
            crawler.update_prediction_with_actual(
                target_date=data['date'],
                actual_value=data['visitor_count']
            )
            print('✅ Prediction history updated')
            
        else:
            print('⚠️  Failed to fetch data (website might be down or structure changed)')
            
    except Exception as e:
        print(f'❌ Error during update: {e}')
        return False
    
    print()
    
    # 3. 显示最近的预测准确度
    print('Step 3: Calculating prediction accuracy...')
    try:
        accuracy = crawler.get_prediction_accuracy(days=7)
        
        if accuracy['count'] > 0:
            print(f'✅ Accuracy (last 7 days):')
            print(f'   MAE:   {accuracy["mae"]:.2f}')
            print(f'   RMSE:  {accuracy["rmse"]:.2f}')
            print(f'   MAPE:  {accuracy["mape"]:.2f}%')
            print(f'   Count: {accuracy["count"]} days')
        else:
            print('⚠️  Not enough data to calculate accuracy')
            
    except Exception as e:
        print(f'⚠️  Error calculating accuracy: {e}')
    
    print()
    print('='*80)
    print('Daily update completed!')
    print('='*80)
    
    return True


if __name__ == '__main__':
    success = daily_update()
    sys.exit(0 if success else 1)
