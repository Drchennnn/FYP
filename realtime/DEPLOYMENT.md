# 实时预测系统部署指南

## 📦 完整文件清单

```
FYP/realtime/
├── jiuzhaigou_crawler.py      # 九寨沟官网爬虫 + 数据库管理
├── scaler_utils.py             # MinMaxScaler 工具
├── data_fetcher.py             # 数据获取模块（天气 + 客流）
├── prediction_engine.py        # 预测引擎（GRU 模型）
├── api_server.py               # Flask API 服务器
├── dashboard_realtime.html     # 前端页面
├── init_system.py              # 系统初始化脚本
├── daily_update.py             # 每日自动更新脚本
├── test_system.py              # 系统测试脚本
├── README.md                   # 使用文档
└── DEPLOYMENT.md               # 本文件
```

---

## 🚀 部署步骤

### 步骤 1: 环境准备

```bash
# 安装依赖
pip install flask flask-cors requests pandas numpy tensorflow chinese-calendar beautifulsoup4 scikit-learn

# 验证安装
python -c "import flask, tensorflow; print('OK')"
```

### 步骤 2: 初始化系统

```bash
cd E:\openclaw\my_project\workspace\FYP

# 运行初始化脚本
python realtime/init_system.py
```

**预期输出：**
```
================================================================================
实时预测系统初始化
================================================================================

Step 1: 生成 MinMaxScaler...
Scalers saved to: models/scalers/feature_scalers.pkl
Features: ['visitor_count', 'tourism_num_lag_7', 'precipitation', 'temp_high', 'temp_low']
✅ Scaler 生成成功

Step 2: 初始化数据库...
Database initialized: data/jiuzhaigou_realtime.db
✅ 数据库初始化成功

Step 3: 验证模型文件...
✅ 模型文件存在: output/runs/gru_8features_20260403_163039/.../gru_jiuzhaigou.h5

Step 4: 测试爬虫...
✅ 爬虫测试成功: {'date': '2026-04-02', 'visitor_count': 15000, ...}

================================================================================
初始化完成！
================================================================================
```

### 步骤 3: 启动 API 服务器

```bash
# 启动服务器
python realtime/api_server.py
```

**预期输出：**
```
Scalers loaded from: models/scalers/feature_scalers.pkl
Features: ['visitor_count', 'tourism_num_lag_7', 'precipitation', 'temp_high', 'temp_low']
Model loaded from: output/runs/.../gru_jiuzhaigou.h5
Lookback: 30
Prediction engine loaded successfully
 * Running on http://127.0.0.1:5001
```

### 步骤 4: 测试系统

**打开新终端：**

```bash
cd E:\openclaw\my_project\workspace\FYP
python realtime/test_system.py
```

**预期输出：**
```
================================================================================
实时预测系统测试
================================================================================

Checking if API server is running...
✅ API server is running

Testing Status...
  ✅ OK
Testing Current Data...
  ✅ OK
Testing Forecast...
  ✅ OK
Testing Weather...
  ✅ OK
Testing Accuracy...
  ✅ OK
Testing History...
  ✅ OK

================================================================================
测试结果汇总
================================================================================
✅ PASS - Status
✅ PASS - Current Data
✅ PASS - Forecast
✅ PASS - Weather
✅ PASS - Accuracy
✅ PASS - History

Total: 6/6 passed

🎉 所有测试通过！系统运行正常。
```

### 步骤 5: 打开前端页面

在浏览器中打开：
```
E:\openclaw\my_project\workspace\FYP\realtime\dashboard_realtime.html
```

点击"🔄 刷新数据"按钮，应该看到：
- ✅ 最新客流数据卡片
- ✅ 未来 7 天预测图表
- ✅ 预测准确度面板

---

## ⏰ 配置定时任务

### Windows 任务计划程序

1. 打开"任务计划程序"
2. 创建基本任务
3. 名称：`Jiuzhaigou Daily Update`
4. 触发器：每天 9:00 AM
5. 操作：启动程序
   - 程序：`python`
   - 参数：`E:\openclaw\my_project\workspace\FYP\realtime\daily_update.py`
   - 起始于：`E:\openclaw\my_project\workspace\FYP`

### Linux/Mac Crontab

```bash
# 编辑 crontab
crontab -e

# 添加以下行（每天 9:00 AM 运行）
0 9 * * * cd /path/to/FYP && python realtime/daily_update.py >> logs/daily_update.log 2>&1
```

---

## 🔧 配置调整

### 修改模型路径

编辑 `realtime/api_server.py`：

```python
MODEL_PATH = 'output/runs/gru_8features_YYYYMMDD_HHMMSS/.../gru_jiuzhaigou.h5'
SCALER_PATH = 'models/scalers/feature_scalers.pkl'
```

### 修改缓存时间

编辑 `realtime/api_server.py`：

```python
data_fetcher = RealtimeDataFetcher(cache_ttl=3600)  # 秒
```

### 修改数据库路径

编辑 `realtime/jiuzhaigou_crawler.py`：

```python
crawler = JiuzhaigouCrawler(db_path='data/jiuzhaigou_realtime.db')
```

---

## 🐛 常见问题

### Q1: 初始化时提示"数据文件不存在"

**A:** 确保已运行完整的数据处理流程：
```bash
python data_processing/fetch_weather_data.py
python data_processing/merge_features.py
```

### Q2: API 服务器启动失败

**A:** 检查端口 5001 是否被占用：
```bash
# Windows
netstat -ano | findstr :5001

# Linux/Mac
lsof -i :5001
```

### Q3: 爬虫返回空数据

**A:** 九寨沟官网结构可能变化，需要更新爬虫逻辑：
1. 手动访问 http://www.jiuzhai.com
2. 查看客流数据的 HTML 结构
3. 更新 `jiuzhaigou_crawler.py` 中的提取逻辑

### Q4: 预测值异常（过大或过小）

**A:** 检查 Scaler 是否正确加载：
```bash
# 重新生成 Scaler
python realtime/init_system.py

# 查看 API 服务器日志
# 应该看到：Scalers loaded from: models/scalers/feature_scalers.pkl
```

---

## 📊 监控和维护

### 每日检查清单

- [ ] 检查 API 服务器是否运行
- [ ] 查看最新爬取的数据是否正常
- [ ] 检查预测准确度是否在合理范围（MAPE < 20%）
- [ ] 查看数据库大小是否正常增长

### 日志位置

- API 服务器日志：控制台输出
- 定时任务日志：`logs/daily_update.log`（如果配置了）
- 数据库：`data/jiuzhaigou_realtime.db`

### 数据库维护

```bash
# 查看数据库大小
ls -lh data/jiuzhaigou_realtime.db

# 备份数据库
cp data/jiuzhaigou_realtime.db data/jiuzhaigou_realtime_backup_$(date +%Y%m%d).db

# 清理旧预测记录（保留最近 90 天）
sqlite3 data/jiuzhaigou_realtime.db "DELETE FROM prediction_history WHERE target_date < date('now', '-90 days')"
```

---

## 🚀 生产环境部署建议

### 1. 使用 Gunicorn + Nginx

```bash
# 安装 Gunicorn
pip install gunicorn

# 启动（4 个 worker）
gunicorn -w 4 -b 127.0.0.1:5001 realtime.api_server:app
```

### 2. 使用 Supervisor 管理进程

创建 `/etc/supervisor/conf.d/jiuzhaigou_api.conf`：

```ini
[program:jiuzhaigou_api]
command=/path/to/venv/bin/gunicorn -w 4 -b 127.0.0.1:5001 realtime.api_server:app
directory=/path/to/FYP
user=www-data
autostart=true
autorestart=true
stderr_logfile=/var/log/jiuzhaigou_api.err.log
stdout_logfile=/var/log/jiuzhaigou_api.out.log
```

### 3. Nginx 反向代理

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location /api/realtime/ {
        proxy_pass http://127.0.0.1:5001/api/realtime/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    location / {
        root /path/to/FYP/realtime;
        index dashboard_realtime.html;
    }
}
```

---

## ✅ 部署检查清单

- [ ] 所有依赖已安装
- [ ] 初始化脚本运行成功
- [ ] Scaler 文件已生成
- [ ] 数据库已初始化
- [ ] 模型文件路径正确
- [ ] API 服务器启动成功
- [ ] 所有 API 接口测试通过
- [ ] 前端页面可以正常访问
- [ ] 定时任务已配置
- [ ] 日志和监控已设置

---

老大，完整的部署指南已完成！🎉
