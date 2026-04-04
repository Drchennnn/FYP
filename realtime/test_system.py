"""
系统测试脚本

功能：
1. 测试所有 API 接口
2. 验证数据流程
3. 检查系统健康状态
"""

import sys
from pathlib import Path
import requests
import time

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

API_BASE = 'http://127.0.0.1:5001/api/realtime'


def test_api_endpoint(name: str, url: str, expected_keys: list = None, check_models: bool = False):
    """测试单个 API 接口"""
    print(f'Testing {name}...')
    
    try:
        response = requests.get(url, timeout=10)
        
        if response.status_code != 200:
            print(f'  ❌ HTTP {response.status_code}')
            return False
        
        data = response.json()
        
        if not data.get('success'):
            print(f'  ❌ API returned success=false: {data.get("error")}')
            return False
        
        if expected_keys:
            for key in expected_keys:
                if key not in data.get('data', {}):
                    print(f'  ❌ Missing key: {key}')
                    return False
        
        # 检查多模型数据
        if check_models:
            models = data.get('data', {}).get('models', {})
            if not models:
                print(f'  ❌ No models data found')
                return False
            
            model_count = sum(1 for v in models.values() if v is not None)
            print(f'  ✅ OK ({model_count} models loaded)')
            return True
        
        print(f'  ✅ OK')
        return True
        
    except requests.exceptions.ConnectionError:
        print(f'  ❌ Connection failed (is the server running?)')
        return False
    except Exception as e:
        print(f'  ❌ Error: {e}')
        return False


def main():
    print('='*80)
    print('实时预测系统测试')
    print('='*80)
    print()
    
    # 检查服务器是否运行
    print('Checking if API server is running...')
    try:
        response = requests.get(f'{API_BASE}/status', timeout=5)
        if response.status_code == 200:
            print('✅ API server is running')
        else:
            print('❌ API server returned unexpected status')
            return
    except:
        print('❌ API server is not running!')
        print('   Please start it with: python realtime/api_server.py')
        return
    
    print()
    
    # 测试所有接口
    tests = [
        ('Status', f'{API_BASE}/status', ['models_loaded', 'cache_ttl'], False),
        ('Current Data', f'{API_BASE}/current', ['date', 'visitor_count', 'temperature'], False),
        ('Forecast (Multi-Model)', f'{API_BASE}/forecast?days=7', ['models', 'generated_at'], True),
        ('Weather', f'{API_BASE}/weather?days=7', ['current', 'forecast'], False),
        ('Accuracy', f'{API_BASE}/accuracy?days=30', ['mae', 'rmse', 'mape'], False),
        ('History', f'{API_BASE}/history?days=7', ['predictions'], False),
    ]
    
    results = []
    for name, url, keys, check_models in tests:
        result = test_api_endpoint(name, url, keys, check_models)
        results.append((name, result))
        time.sleep(0.5)  # 避免请求过快
    
    print()
    print('='*80)
    print('测试结果汇总')
    print('='*80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = '✅ PASS' if result else '❌ FAIL'
        print(f'{status} - {name}')
    
    print()
    print(f'Total: {passed}/{total} passed')
    
    if passed == total:
        print()
        print('🎉 所有测试通过！系统运行正常。')
    else:
        print()
        print('⚠️  部分测试失败，请检查日志。')


if __name__ == '__main__':
    main()
