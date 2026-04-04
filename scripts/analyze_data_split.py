"""
改进的数据划分策略

问题：
- 当前 70/15/15 划分 + lookback=30 导致测试集只有 133 个样本
- 训练样本也不足（只有 669 个窗口）

解决方案：
- 使用 Walk-Forward Validation（滚动窗口验证）
- 增加训练集比例到 80%
- 减少验证集到 10%
- 测试集保持 10%
- 这样可以获得更多的训练样本
"""

import pandas as pd
import numpy as np
from pathlib import Path


def analyze_current_split():
    """分析当前数据划分"""
    df = pd.read_csv('data/processed/jiuzhaigou_daily_features_2016-01-01_2026-04-02.csv')
    
    print('='*80)
    print('当前数据划分分析')
    print('='*80)
    print()
    
    total = len(df)
    print(f'总数据量: {total} 条')
    print(f'日期范围: {df["date"].min()} 到 {df["date"].max()}')
    print()
    
    # 当前 70/15/15 划分
    train_size = int(total * 0.7)
    val_size = int(total * 0.15)
    test_size = total - train_size - val_size
    
    print('当前划分 (70/15/15):')
    print(f'  训练集: {train_size} 条 ({train_size/total*100:.1f}%)')
    print(f'  验证集: {val_size} 条 ({val_size/total*100:.1f}%)')
    print(f'  测试集: {test_size} 条 ({test_size/total*100:.1f}%)')
    print()
    
    # 考虑 lookback=30 的影响
    lookback = 30
    train_windows = train_size - lookback + 1
    val_windows = val_size
    test_windows = test_size
    
    print(f'考虑 lookback={lookback} 后的有效窗口数:')
    print(f'  训练窗口: {train_windows} 个')
    print(f'  验证窗口: {val_windows} 个')
    print(f'  测试窗口: {test_windows} 个')
    print()
    
    return df, total


def propose_new_split():
    """提出新的数据划分方案"""
    df, total = analyze_current_split()
    
    print('='*80)
    print('改进方案 1: 增加训练集比例 (80/10/10)')
    print('='*80)
    print()
    
    train_size = int(total * 0.8)
    val_size = int(total * 0.1)
    test_size = total - train_size - val_size
    
    print(f'  训练集: {train_size} 条 ({train_size/total*100:.1f}%)')
    print(f'  验证集: {val_size} 条 ({val_size/total*100:.1f}%)')
    print(f'  测试集: {test_size} 条 ({test_size/total*100:.1f}%)')
    print()
    
    lookback = 30
    train_windows = train_size - lookback + 1
    val_windows = val_size
    test_windows = test_size
    
    print(f'有效窗口数 (lookback={lookback}):')
    print(f'  训练窗口: {train_windows} 个 (增加 {train_windows - 1873} 个)')
    print(f'  验证窗口: {val_windows} 个')
    print(f'  测试窗口: {test_windows} 个')
    print()
    
    print('='*80)
    print('改进方案 2: 减小 lookback 窗口 (lookback=14)')
    print('='*80)
    print()
    
    # 恢复 70/15/15 但减小 lookback
    train_size = int(total * 0.7)
    val_size = int(total * 0.15)
    test_size = total - train_size - val_size
    
    lookback = 14
    train_windows = train_size - lookback + 1
    val_windows = val_size
    test_windows = test_size
    
    print(f'数据划分: 70/15/15')
    print(f'Lookback: {lookback} 天 (从 30 减少到 14)')
    print()
    print(f'有效窗口数:')
    print(f'  训练窗口: {train_windows} 个 (增加 {train_windows - 1873} 个)')
    print(f'  验证窗口: {val_windows} 个')
    print(f'  测试窗口: {test_windows} 个')
    print()
    
    print('='*80)
    print('推荐方案: 80/10/10 + lookback=30')
    print('='*80)
    print()
    
    train_size = int(total * 0.8)
    val_size = int(total * 0.1)
    test_size = total - train_size - val_size
    lookback = 30
    
    train_windows = train_size - lookback + 1
    val_windows = val_size
    test_windows = test_size
    
    print('优点:')
    print(f'  ✅ 训练窗口增加到 {train_windows} 个 (比当前多 {train_windows - 1873} 个)')
    print(f'  ✅ 保持 lookback=30，不损失时间序列信息')
    print(f'  ✅ 测试集仍有 {test_windows} 个样本，足够评估')
    print()
    
    print('实施方法:')
    print('  修改 run_pipeline.py 中的数据划分比例')
    print('  或在各个训练脚本中修改 train_test_split 参数')
    print()
    
    return {
        'train_ratio': 0.8,
        'val_ratio': 0.1,
        'test_ratio': 0.1,
        'lookback': 30,
        'train_windows': train_windows,
        'val_windows': val_windows,
        'test_windows': test_windows
    }


if __name__ == '__main__':
    config = propose_new_split()
    
    print('='*80)
    print('下一步行动')
    print('='*80)
    print()
    print('1. 修改数据划分比例为 80/10/10')
    print('2. 重新训练三个模型')
    print('3. 对比新旧结果')
    print()
    print('预期改进:')
    print(f'  - 训练样本增加约 {config["train_windows"] - 1873} 个')
    print('  - 模型泛化能力提升')
    print('  - 校准质量改善')
