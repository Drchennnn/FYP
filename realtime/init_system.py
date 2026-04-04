"""
实时预测系统初始化脚本

功能：
1. 生成并保存 MinMaxScaler
2. 初始化数据库
3. 验证模型和配置
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from realtime.scaler_utils import save_scaler_from_training_data
from realtime.jiuzhaigou_crawler import JiuzhaigouCrawler

def main():
    print('='*80)
    print('实时预测系统初始化')
    print('='*80)
    print()
    
    # 1. 生成 Scaler
    print('Step 1: 生成 MinMaxScaler...')
    data_path = 'data/processed/jiuzhaigou_daily_features_2016-01-01_2026-04-02.csv'
    scaler_path = 'models/scalers/feature_scalers.pkl'
    
    if Path(data_path).exists():
        scalers = save_scaler_from_training_data(data_path, scaler_path)
        print('✅ Scaler 生成成功')
    else:
        print(f'❌ 数据文件不存在: {data_path}')
        return
    
    print()
    
    # 2. 初始化数据库
    print('Step 2: 初始化数据库...')
    crawler = JiuzhaigouCrawler()
    print('✅ 数据库初始化成功')
    print()
    
    # 3. 验证模型
    print('Step 3: 验证模型文件...')
    model_paths = {
        'LSTM': 'output/runs/lstm_8features_20260403_164549/runs/run_20260403_164549_lb30_ep120_lstm_8features/weights/lstm_jiuzhaigou.h5',
        'GRU': 'output/runs/gru_8features_20260403_163039/runs/run_20260403_163039_lb30_ep120_gru_8features/weights/gru_jiuzhaigou.h5',
        'Seq2Seq+Attention': 'output/runs/seq2seq_attention_8features_20260403_164843/runs/run_20260403_164843_lb30_ep120_seq2seq_attention_8features/weights/seq2seq_attention_jiuzhaigou.h5'
    }
    
    all_models_exist = True
    for model_name, model_path in model_paths.items():
        if Path(model_path).exists():
            print(f'✅ {model_name} 模型文件存在')
        else:
            print(f'❌ {model_name} 模型文件不存在: {model_path}')
            all_models_exist = False
    
    if not all_models_exist:
        print('   请先训练所有模型或修改 api_server.py 中的 MODEL_CONFIGS')
    
    print()
    
    # 4. 测试爬虫
    print('Step 4: 测试爬虫...')
    try:
        data = crawler.fetch_latest_visitor_count()
        if data:
            print(f'✅ 爬虫测试成功: {data}')
        else:
            print('⚠️  爬虫返回空数据（可能是网站结构变化，需要调整爬虫逻辑）')
    except Exception as e:
        print(f'⚠️  爬虫测试失败: {e}')
    
    print()
    print('='*80)
    print('初始化完成！')
    print('='*80)
    print()
    print('下一步：')
    print('1. 启动 API 服务器: python realtime/api_server.py')
    print('2. 打开前端页面: realtime/dashboard_realtime.html')
    print()

if __name__ == '__main__':
    main()
