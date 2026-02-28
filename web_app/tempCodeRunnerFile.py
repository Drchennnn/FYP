    """加载节假日配置文件"""
    try:
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'holidays.json')
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading holidays config: {e}")
    return []

# 全局节假日配置
HOLIDAYS_CONFIG = load_holidays_config()

def mark_core_holiday(date_val):