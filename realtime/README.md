# 九寨沟客流实时预测系统 - 三模型对比版

## 📋 系统概述

实时预测系统提供以下功能：
1. **最新数据展示**：昨天的客流量（官网每日更新）、实时天气
2. **未来预测**：基于三个 8 特征模型（LSTM、GRU、Seq2Seq+Attention）预测未来 7 天客流
3. **多模型对比**：前端可视化展示三个模型的预测结果对比
4. **预测历史**：保存所有预测记录到数据库
5. **准确度评估**：自动计算预测准确度（MAE、RMSE、MAPE）
6. **可视化展示**：ECharts 交互式图表

---

## 🚀 快速启动

### 1. 安装依赖

```bash
pip install flask flask-cors requests pandas numpy tensorflow chinese-calendar beautifulsoup4 scikit-learn
```

### 2. 初始化系统

```bash
cd E:\openclaw\my_project\workspace\FYP
python realtime/init_system.py
```

这将：
- ✅ 生成 MinMaxScaler（用于准确的归一化/反归一化）
- ✅ 初始化 SQLite 数据库
- ✅ 验证三个模型文件（LSTM、GRU、Seq2Seq+Attention）
- ✅ 测试爬虫

### 3. 启动 API 服务器

```bash
python realtime/api_server.py
```

服务器将在 `http://127.0.0.1:5001` 启动

### 4. 打开前端页面

在浏览器中打开：
```
E:\openclaw\my_project\workspace\FYP\realtime\dashboard_realtime.html
```

---

## 📡 API 接口文档

### 1. 获取最新数据（昨天的）

**Endpoint:** `GET /api/realtime/current`

**Response:**
```json
{
  "success": true,
  "data": {
    "date": "2026-04-02",
    "visitor_count": 15000,
    "temperature": 18.5,
    "precipitation": 0.0,
    "temp_high": 22.0,
    "temp_low": 12.0,
    "timestamp": "2026-04-03 17:00:00",
    "note": "Data is from yesterday (official website updates daily)"
  }
}
```

### 2. 获取未来预测（三模型对比）

**Endpoint:** `GET /api/realtime/forecast?days=7`

**Response:**
```json
{
  "success": true,
  "data": {
    "models": {
      "lstm": [
        {
          "date": "2026-04-04",
          "predicted_visitor_count": 18000,
          "temp_high": 22.0,
          "temp_low": 12.0,
          "precipitation": 0.5
        }
      ],
      "gru": [...],
      "seq2seq": [...]
    },
    "generated_at": "2026-04-03 17:30:00"
  }
}
```

### 3. 获取预测准确度

**Endpoint:** `GET /api/realtime/accuracy?days=30`

### 4. 获取预测历史

**Endpoint:** `GET /api/realtime/history?days=7`

### 5. 获取系统状态

**Endpoint:** `GET /api/realtime/status`

**Response:**
```json
{
  "success": true,
  "data": {
    "models_loaded": {
      "lstm": "LSTM",
      "gru": "GRU",
      "seq2seq": "Seq2Seq+Attention"
    },
    "scaler_loaded": true,
    "cache_ttl": 3600,
    "timestamp": "2026-04-03 17:30:00"
  }
}
```

---

## 🎯 三模型对比

### 模型配置

| 模型 | 架构 | 参数量 | 训练数据 |
|------|------|--------|---------|
| LSTM | 单层 LSTM | ~50K | 2145 训练窗口 |
| GRU | 单层 GRU | ~40K | 2145 训练窗口 |
| Seq2Seq+Attention | Encoder-Decoder | ~80K | 2145 训练窗口 |

### 性能对比（80/10/10 数据集）

| 模型 | MAE | RMSE | MAPE | R² |
|------|-----|------|------|-----|
| LSTM | 3582.77 | 4788.66 | 18.51% | 0.8544 |
| GRU | 2687.90 | 3902.02 | 14.86% | 0.9033 |
| Seq2Seq+Attention | 5319.76 | 7384.39 | - | - |

**结论：GRU 模型表现最佳**

---

## 🔧 核心改进

### 1. 多模型支持
- ✅ 同时加载三个模型
- ✅ 并行预测
- ✅ 前端对比展示

### 2. 真实爬虫实现
- ✅ BeautifulSoup 解析九寨沟官网
- ✅ 多种提取策略
- ✅ 数据验证和降级策略

### 3. 准确的归一化
- ✅ 保存训练时的 MinMaxScaler
- ✅ 预测时使用相同的 Scaler

### 4. 预测历史记录
- ✅ SQLite 数据库存储
- ✅ 自动更新实际值
- ✅ 准确度自动计算

---

## 📊 前端展示

### 实时数据卡片
- 最新客流（昨天的）
- 当前温度
- 今日最高/最低温
- 降水量

### 预测图表（三模型对比）
- LSTM 曲线（红色）
- GRU 曲线（蓝色，带阴影）
- Seq2Seq+Attention 曲线（青色）
- 交互式 Tooltip

### 准确度面板
- MAE（平均绝对误差）
- RMSE（均方根误差）
- MAPE（平均百分比误差）
- 评估样本数

---

## 📝 文件清单

```
FYP/realtime/
├── jiuzhaigou_crawler.py      # 爬虫 + 数据库管理
├── scaler_utils.py             # Scaler 工具
├── data_fetcher.py             # 数据获取模块
├── multi_model_engine.py       # 多模型预测引擎 ⭐ 新增
├── api_server.py               # Flask API（已更新支持多模型）
├── dashboard_realtime.html     # 前端页面（已更新三模型对比）
├── init_system.py              # 初始化脚本（已更新验证三模型）
├── daily_update.py             # 定时任务
├── test_system.py              # 测试脚本（已更新）
├── README.md                   # 本文件
├── DEPLOYMENT.md               # 部署指南
└── SUMMARY.md                  # 完成总结
```

---

## 🐛 故障排查

### 问题 1: 某个模型加载失败

**现象**: API 返回的 `models` 中某个模型为 `null`

**解决**:
1. 检查模型文件路径是否正确
2. 运行 `python realtime/init_system.py` 验证所有模型
3. 查看 API 服务器日志

### 问题 2: 前端只显示部分模型

**原因**: 某些模型预测失败

**解决**:
1. 打开浏览器开发者工具查看 Console
2. 检查 API 返回的 `models` 数据
3. 确认所有模型都已正确加载

---

## 📝 TODO

- [x] 实现真实的九寨沟官网爬虫
- [x] 保存训练时的 MinMaxScaler
- [x] 添加预测历史记录
- [x] 实现预测准确度评估
- [x] 支持三模型对比展示
- [ ] 添加定时任务（每日自动爬取）
- [ ] 添加模型性能对比表格
- [ ] 实现最佳模型自动选择
- [ ] 移动端适配

---

老大，三模型对比系统已完成！🎉
