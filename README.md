# 基于 LSTM 的九寨沟景区客流动态预测系统

本项目实现了一个端到端的客流预测系统，涵盖从数据爬取、清洗、特征工程、LSTM 模型训练到 Web 前端展示的全流程。

## 📋 目录
1. [环境搭建](#1-环境搭建)
2. [数据流程](#2-数据流程)
3. [模型训练](#3-模型训练)
4. [Web 应用](#4-web-应用)
5. [常见问题与解决方案](#5-常见问题与解决方案)

---

## 1. 环境搭建

**⚠️ 关键提示**：本项目对 `tensorflow` 和 `keras` 的版本有严格要求，请务必按照以下步骤操作，否则会出现 `Model not loaded` 或 `AttributeError`。

### 推荐方式：使用 Anaconda
```powershell
# 1. 创建环境 (Python 3.11)
conda create -n FYP python=3.11
conda activate FYP

# 2. 安装基础依赖
pip install -r requirements_all.txt

# 3. (如果遇到 TF 报错) 手动强制重装核心库
pip uninstall -y tensorflow tensorflow-intel tensorflow-cpu keras h5py
pip install tensorflow==2.15.0 tensorflow-intel==2.15.0 keras==2.15.0 h5py==3.10.0 protobuf==4.25.3
```

---

## 2. 数据流程

### 2.1 数据爬取
从阿坝旅游网爬取最新的客流数据。
```powershell
python scripts/scrape_jiuzhaigou.py
```
*   **输出**：`data/raw/jiuzhaigou_tourist_data.csv`

### 2.2 数据预处理
清洗原始数据，处理缺失值，合并天气和节假日特征。
```powershell
python scripts/preprocess.py
```
*   **输出**：`data/processed/jiuzhaigou_daily_features_YYYY-MM-DD_YYYY-MM-DD.csv`

### 2.3 数据库同步
将最新的 CSV 数据（真实客流）和模型预测结果（测试集预测）同步到 SQLite 数据库，供 Web 端使用。
```powershell
python scripts/sync_to_cloud.py
```
*   **功能**：
    *   自动识别最新的 `processed` 数据并覆盖数据库中的真实记录。
    *   自动识别 `output/runs` 下最新的预测结果并更新数据库。
    *   **注意**：`app.py` 启动时也会自动调用此脚本。

---

## 3. 模型训练

使用 LSTM (Long Short-Term Memory) 神经网络进行训练。

```powershell
# 运行完整流程（训练 + 评估）
python run_pipeline.py --epochs 50 --batch-size 32

# 快速测试
python run_pipeline.py --epochs 5
```

*   **参数说明**：
    *   `--epochs`: 训练轮数
    *   `--lookback`: 时间窗口大小（默认 30 天）
*   **输出**：
    *   模型文件：`model/runs/run_TIME/lstm_jiuzhaigou.h5`
    *   预测结果：`output/runs/run_TIME/lstm_test_predictions.csv`

---

## 4. Web 应用

启动 Flask 后端服务器，提供 RESTful API 和前端页面。

```powershell
# 确保在项目根目录下运行
python web_app/app.py
```

*   **访问地址**：`http://127.0.0.1:5000/`
*   **功能**：
    *   **历史回溯**：展示真实客流与模型在历史数据上的拟合曲线。
    *   **未来预测**：点击“立即运算”，调用模型预测未来 7 天（或指定日期）的客流。
    *   **节假日标注**：图表中自动标注春节、国庆等重要节假日。

---

## 5. 常见问题与解决方案

### Q1: `Model not loaded` 或 `AttributeError: module 'tensorflow' has no attribute...`
*   **原因**：环境中的 TensorFlow 和 Keras 版本不匹配（例如安装了 Keras 3 但 TF 是 2.15）。
*   **解决**：执行[环境搭建](#1-环境搭建)中的“强制重装核心库”命令。务必确保 `pip list` 显示 TF 和 Keras 均为 `2.15.0`。

### Q2: 前端显示的“最新数据日期”滞后
*   **原因**：数据库未同步，或者前端读取了旧的预测数据。
*   **解决**：
    1.  确认 `processed` 目录下有最新的 CSV 文件。
    2.  重启 `app.py`（会自动同步）。
    3.  前端代码已优化，会自动寻找真实数据的最后一天作为显示基准。

### Q3: `ImportError: cannot import name '...' from 'keras'`
*   **原因**：代码中混用了 `import keras` 和 `from tensorflow import keras`。
*   **解决**：本项目已统一使用 `tensorflow==2.15.0`。请检查是否误装了 `keras` (v3) 包。如果是，请卸载它并重装 `keras==2.15.0`。

### Q4: 数据库报错 `no such column: id`
*   **原因**：数据库表结构与 `models.py` 定义不一致（旧版脚本创建的表没有主键 ID）。
*   **解决**：删除 `jiuzhaigou_fyp.db`，重启 `app.py` 或运行 `sync_to_cloud.py`，它会自动重建正确的表结构。
