from flask_sqlalchemy import SQLAlchemy
from datetime import date

# 初始化 SQLAlchemy 实例
db = SQLAlchemy()

class TrafficRecord(db.Model):
    """
    客流记录表模型
    用于存储历史真实客流数据和模型预测数据
    """
    __tablename__ = 'traffic_records'

    # 主键 ID
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    
    # 记录日期（唯一索引，防止同一天重复数据）
    record_date = db.Column(db.Date, unique=True, nullable=False, index=True)
    
    # 真实游客数量 (允许为空，未来的预测数据可能暂时没有真实值)
    actual_visitor = db.Column(db.Integer, nullable=True)
    
    # 预测游客数量
    predicted_visitor = db.Column(db.Integer, nullable=True)
    
    # 是否为预测数据 (True=预测/未来数据, False=历史/验证数据)
    is_forecast = db.Column(db.Boolean, default=False)

    def to_dict(self):
        """将模型对象转换为字典，方便 API 返回 JSON"""
        return {
            'date': self.record_date.strftime('%Y-%m-%d'),
            'actual': self.actual_visitor,
            'predicted': self.predicted_visitor,
            'is_forecast': self.is_forecast
        }
