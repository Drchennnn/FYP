# 九寨沟景区客流动态预测系统 (Jiuzhaigou Tourist Flow Prediction System)

本项目是一个端到端的客流预测解决方案，集成了数据爬取、特征工程、LSTM 深度学习模型训练以及基于 Web 的可视化展示平台。系统旨在利用历史客流数据和时间特征，准确预测未来九寨沟景区的游客接待量，辅助景区管理和游客出行决策。

---

## 📚 目录 (Table of Contents)

1.  [项目整体架构 (Project Architecture)](#1-项目整体架构-project-architecture)
2.  [快速开始 (Quick Start)](#2-快速开始-quick-start)
3.  [数据爬取子系统 (Crawler Subsystem)](#3-数据爬取子系统-crawler-subsystem)
4.  [数据预处理与特征工程 (Data Preprocessing)](#4-数据预处理与特征工程-data-preprocessing)
5.  [LSTM 模型架构与设计 (LSTM Model Design)](#5-lstm-模型架构与设计-lstm-model-design)
6.  [Web 应用架构 (Web Application)](#6-web-应用架构-web-application)
7.  [数据库设计 (Database Schema)](#7-数据库设计-database-schema)
8.  [数据同步机制 (Data Synchronization)](#8-数据同步机制-data-synchronization)

---

## 1. 项目整体架构 (Project Architecture)

项目采用模块化设计，主要包含以下核心流水线：

1.  **数据层 (Data Layer)**:
    *   **Crawler**: 从九寨沟官网和天气网抓取原始数据。
    *   **Storage**: CSV 文件用于离线训练，SQLite 数据库用于在线服务。
2.  **模型层 (Model Layer)**:
    *   **Preprocessing**: 清洗数据，生成滞后特征、周期性时间编码和节假日标记。
    *   **Training**: 基于 TensorFlow/Keras 的 LSTM 模型，支持自动超参数调整（EarlyStopping, ReduceLROnPlateau）。
3.  **应用层 (Application Layer)**:
    *   **Backend**: Flask 提供 RESTful API 和模型推理服务。
    *   **Frontend**: 基于 Bootstrap 5 和 ECharts 的响应式数据仪表盘。

**目录结构说明**:
```text
FYP/
├── Jiuzhaigou_Crawler/    # 爬虫子系统
│   ├── tourism_num.py     # 客流抓取
│   ├── weather_jiuzhaigou.py # 天气抓取
│   └── merge_tourism_weather.py # 数据合并与 Open-Meteo 补全
├── data/                  # 数据存储
│   ├── raw/               # 原始爬取数据 (CSV)
│   └── processed/         # 预处理后的特征数据 (CSV)
├── models/                # 模型相关代码
│   ├── common/preprocess.py # 特征工程脚本
│   └── lstm/train_lstm.py   # LSTM 训练脚本
├── model/runs/            # 训练好的模型文件 (.h5/.keras)
├── output/runs/           # 训练过程中的指标、图表和预测结果
├── web_app/               # Web 应用程序
│   ├── app.py             # Flask 后端入口
│   ├── models.py          # SQLAlchemy 数据库模型
│   ├── static/            # JS/CSS 资源
│   └── templates/         # HTML 模板
├── scripts/               # 辅助脚本
│   └── sync_to_cloud.py   # 数据同步脚本 (CSV -> SQLite)
├── run_pipeline.py        # 一键运行流水线脚本
└── jiuzhaigou_fyp.db      # SQLite 数据库文件
```

---

## 2. 快速开始 (Quick Start)

### 环境准备

推荐使用 **Conda** 创建独立的虚拟环境：

```bash

# 激活环境
conda activate FYP

# 安装项目依赖
pip install -r requirements_all.txt #本机已安装
```

### 运行全流程 (Pipeline)
使用 `run_pipeline.py` 可以一键执行“预处理 -> 训练”流程：
```bash
# 运行预处理和训练（默认 120 epochs, lookback 30）
python run_pipeline.py

# 仅运行训练，跳过预处理
python run_pipeline.py --skip-preprocess
```

### 启动 Web 应用
1.  **同步数据**: 将最新的训练结果写入数据库。
    ```bash
    python scripts/sync_to_cloud.py
    ```
2.  **启动服务**:
    ```bash
    cd web_app
    python app.py
    ```
3.  访问浏览器: `http://localhost:5000`

---

## 3. 数据爬取子系统 (Crawler Subsystem)

位于 `Jiuzhaigou_Crawler/` 目录下。

*   **客流抓取 (`tourism_num.py`)**:
    *   **源站**: 九寨沟官网新闻公告栏。
    *   **方法**: 使用 `requests` 获取页面，`BeautifulSoup` 解析 HTML 表格。
    *   **逻辑**: 支持分页 (`start` 参数)，利用正则表达式从标题中提取“入园人数”。
*   **天气抓取 (`weather_jiuzhaigou.py`)**:
    *   **源站**: 2345 天气王历史数据接口。
    *   **方法**: 直接请求后端 AJAX 接口 (`/Pc/GetHistory`)，规避了页面渲染问题。
    *   **内容**: 获取每日最高/最低温、天气状况、风向风力、AQI。
*   **数据合并 (`merge_tourism_weather.py`)**:
    *   **逻辑**: 以日期为主键，执行 Inner Join 合并客流和天气数据。
    *   **容错**: 集成 **Open-Meteo API**。当主天气源缺失数据时，根据经纬度自动调用 Open-Meteo 的历史存档或预报接口进行补全，确保数据连续性。

---

## 4. 数据预处理与特征工程 (Data Preprocessing)

位于 `models/common/preprocess.py`。

*   **数据清洗**:
    *   单位去除（如 "15℃" -> 15.0）。
    *   天气描述标准化（中文 -> 英文代码，如 "晴" -> "SUNNY"）。
    *   缺失值填充（Open-Meteo 补全或前向填充）。
*   **特征工程**:
    *   **时间特征**: 提取 Month, Day of Week，并进行 **Sin/Cos 周期性编码** (`month_sin`, `month_cos`)，保留时间的循环特性。
    *   **节假日特征**: 集成 `chinese_calendar` 库，精准标记中国法定节假日 (`is_holiday`) 和调休工作日。
    *   **滞后特征 (Lag Features)**: 生成 `lag_1`, `lag_7`, `lag_14`, `lag_28` (过去 1/7/14/28 天的客流)。
    *   **滚动统计 (Rolling Stats)**: 计算 7 天和 14 天的滑动平均值 (`rolling_mean`) 和标准差 (`rolling_std`)。

---

## 5. LSTM 模型架构与设计 (LSTM Model Design)

位于 `models/lstm/train_lstm.py`。

### 模型结构
基于 TensorFlow/Keras 构建的 Sequential 模型：
1.  **Input Layer**: 形状 `(Lookback=30, Features=4)`。
2.  **LSTM Layer 1**: 64 单元，`return_sequences=True`，提取序列特征。
3.  **Dropout**: 0.2，防止过拟合。
4.  **LSTM Layer 2**: 32 单元，提取深层依赖。
5.  **Dropout**: 0.2。
6.  **Dense Layer**: 16 单元，ReLU 激活。
7.  **Output Layer**: 1 单元，线性激活（回归预测）。

### 输入特征
目前模型主要依赖以下 4 个强相关特征：
*   `visitor_count_scaled`: 归一化后的历史客流。
*   `month_norm`: 归一化月份。
*   `day_of_week_norm`: 归一化星期。
*   `is_holiday`: 节假日标记 (0/1)。

### 训练策略
*   **损失函数**: Huber Loss (对异常值更鲁棒)。
*   **优化器**: Adam (LR=1e-3)。
*   **回调**:
    *   `EarlyStopping`: 验证集 Loss 15 轮不降则停止。
    *   `ReduceLROnPlateau`: 验证集 Loss 7 轮不降则学习率减半。
*   **数据集划分**: 严格按时间顺序划分 Train/Val/Test，避免未来信息泄露。

---

## 6. Web 应用架构 (Web Application)

位于 `web_app/` 目录下，采用 **Flask** 框架。

### 后端 (Backend)
*   **API**:
    *   `GET /api/data`: 返回全量历史数据和节假日配置。
    *   `POST /api/predict`: 接收预测请求（指定日期或天数）。
*   **在线推理引擎**:
    *   应用启动时自动加载最新的 `.h5` 或 `.keras` 模型。
    *   **滚动预测 (Rolling Prediction)**: 针对未来多天的预测，系统采用递归方式，将前一天的预测值作为当天的输入特征（配合动态生成的日期和节假日特征），实现连续推理。
    *   **归一化处理**: 使用 `MinMaxScaler` 在线对输入数据进行缩放，输出后反归一化。

### 前端 (Frontend)
*   **设计**: 深色科技风 UI，基于 Bootstrap 5。
*   **可视化**: 使用 **ECharts** 绘制交互式折线图，支持缩放、拖拽。
    *   **实线**: 真实历史数据。
    *   **虚线**: AI 预测数据。
    *   **区域高亮**: 自动在图表背景中标记法定节假日（春节、国庆等），不同节日显示不同颜色。
*   **交互**:
    *   支持一键预测未来 1/3/7 天。
    *   支持中英文国际化切换 (i18n)。
    *   预测时显示 Loading 动画，完成后弹出模态框展示结果。

---

## 7. 数据库设计 (Database Schema)

使用 **SQLite** 数据库 (`jiuzhaigou_fyp.db`)，通过 SQLAlchemy ORM 管理。

**表名**: `traffic_records`

| 字段名 | 类型 | 描述 |
| :--- | :--- | :--- |
| `id` | Integer (PK) | 自增主键 |
| `record_date` | Date (Unique) | 记录日期，核心索引 |
| `actual_visitor` | Integer | 真实游客人数 (历史数据) |
| `predicted_visitor` | Integer | 模型预测人数 |
| `is_forecast` | Boolean | 标识位 (0=历史验证, 1=未来预测) |

---

## 8. 数据同步机制 (Data Synchronization)

脚本: `scripts/sync_to_cloud.py`

为了连接离线训练和在线服务，系统设计了同步机制：
1.  **读取**: 自动扫描 `data/processed/` 获取最新的历史数据 CSV，扫描 `output/runs/` 获取最新的测试集预测结果 CSV。
2.  **重置**: 每次同步时重建 `traffic_records` 表，确保数据纯净。
3.  **写入**:
    *   将历史真实数据批量写入 `actual_visitor`。
    *   使用 **Upsert** (Insert or Update) 逻辑将模型的测试集预测结果写入 `predicted_visitor`，实现历史真实值与回测预测值的对比展示。
4.  **触发**: 该脚本可在 Web 应用启动时自动调用，也可手动运行。

## 存在问题与未来改进计划 (Problems & Future Work)

尽管当前系统已初步实现端到端的客流预测与可视化，但在模型泛化能力、极端峰值预测以及系统工程化方面仍存在一定局限性。接下来的开发与研究工作将围绕以下四个核心维度展开：

### 模型评估与基线对比 (Model Benchmarking)
* **当前局限 (Insufficient Empirical Benchmarking)**: 
  目前的评估主要集中在 LSTM 架构上，缺乏与其他主流时间序列预测基线（如 GRU、Prophet 等）的全面对比，难以严格量化本模型在相对性能和泛化能力上的提升。
* **改进计划 (Model Expansion & Standardized Evaluation)**: 
  * 系统性地引入 **GRU (Gated Recurrent Unit)** 和基于统计学的 **Prophet** 模型，完善基线对比分析框架。
  * 建立并固化包含 MAE、RMSE 和 MAPE 在内的标准定量评估指标体系，提供更严谨的交叉验证。

### 极端峰值预测与特征工程 (Peak Prediction & Feature Engineering)
* **当前局限 (Feature Heterogeneity & Peak-interval Volatility)**: 
  模型目前无法动态区分不同外部特征的影响权重（例如“节假日”对客流的压倒性影响）。这导致模型在面对“十一黄金周”等流量激增的极端时段时，预测保真度下降，表现出较高的方差和严重的预测不足 (Under-prediction)。
* **改进计划 (Attention Mechanism & Hybrid Modeling)**:
  * **引入注意力机制 (Attention Mechanisms)**: 在网络架构中增加 Attention 层，使模型能够动态地为关键特征（如节假日标记）分配更高的权重。
  * **混合模型补偿 (Hybrid Extreme Event Modeling)**: 采用双架构设计来解耦常规季节性模式与异常客流激增。引入轻量级梯度提升树（如 LightGBM）或基于规则的专家系统，专门针对 LSTM 预测的残差 (Residual errors) 进行二次训练补偿，大幅提升高峰期的预测精度。
  * **高维特征融合 (Advanced Feature Engineering)**: 拓展多变量数据流水线，接入更细粒度的气象数据和社交媒体搜索指数（如百度指数 Baidu Index），以更灵敏地捕捉真实的游客出行意愿。
  * **自定义损失函数 (Custom Loss Functions)**: 设计不对称损失函数，对峰值区间的预测偏低施加更重的惩罚。

### 长序列预测漂移与架构升级 (Long-Term Forecasting Drift)
* **当前局限 (Error Accumulation & Exposure Bias)**: 
  系统当前采用的“自回归滚动预测机制 (Autoregressive rolling)”极易产生误差累积。在预测较长周期（如 T+14 天）时，早期步骤的微小预测偏差会被作为输入不断前馈，导致远期的预测轨迹出现严重漂移或过度平滑。
* **改进计划 (Seq2Seq Architecture)**:
  * 将预测策略从单步滚动窗口全面升级为 **多对多 (Many-to-Many) Sequence-to-Sequence (Seq2Seq)** 架构。
  * 利用 Encoder-Decoder 结构一次性并行输出整个未来的预测序列，从根本上削弱迭代偏差 (Iterative bias) 的传播。

### 系统部署、可解释性与概念漂移 (Deployment & Explainability)
* **当前局限 (Black-box & Concept Drift)**: 
  作为深度神经网络，LSTM 本质上是一个“黑盒”，缺乏向景区管理者解释预测依据的透明度。同时，面对由于基础设施建设或政策调整导致的长期旅游模式改变（概念漂移 Concept Drift），静态训练的历史数据可能面临失效风险。
* **改进计划 (Cloud Integration & Continual Learning)**:
  * **可解释 AI (Explainable AI, XAI)**: 集成 **SHAP (SHapley Additive exPlanations)** 等可解释性工具，实现针对特定日期预测结果的特征贡献度可视化，提升系统的可信度。
  * **云端部署与系统集成 (Cloud Deployment)**: 将本地原型完整迁移至国内云服务器（如腾讯云）。配置 Nginx Web 服务器，建立健壮的数据库自动更新脚本，实现 ECharts 前端与 Flask API 的稳定远程访问。
  * **持续学习 (Continual Learning)**: 设计支持周期性自动重训练的数据同步流水线，使模型能够自适应新兴的旅游趋势，有效应对概念漂移。