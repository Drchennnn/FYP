import os

class Config:
    """Flask 应用配置类"""
    # 获取当前文件所在目录的父目录（即项目根目录）
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # SQLite 数据库文件路径
    # 数据库文件名为 jiuzhaigou_fyp.db，存放在项目根目录下
    SQLALCHEMY_DATABASE_URI = f"sqlite:///{os.path.join(BASE_DIR, 'jiuzhaigou_fyp.db')}"
    
    # 关闭 SQLAlchemy 的事件追踪系统，节省内存
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # 密钥配置（建议在生产环境中使用环境变量）
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key'
