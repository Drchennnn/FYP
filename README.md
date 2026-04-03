# 九寨沟景区客流动态预测系统

本项目是一个端到端的客流预测解决方案,集成了数据爬取、特征工程、深度学习模型训练以及基于 Web 的可视化展示平台。系统旨在利用历史客流数据和时间特征,准确预测未来九寨沟景区的游客接待量,辅助景区管理和游客出行决策。

---

## 📚 目录

1.  [项目整体架构](#1-项目整体架构)
2.  [快速开始](#2-快速开始)
3.  [数据爬取子系统](#3-数据爬取子系统)
4.  [数据预处理与特征工程](#4-数据预处理与特征工程)
5.  [模型架构演进](#5-模型架构演进)
6.  [最新性能指标](#6-最新性能指标)
7.  [MLOps 与工程化规范](#7-mlops-与工程化规范)
8.  [Web 应用架构](#8-web-应用架构)
9.  [数据库设计](#9-数据库设计)
10. [数据同步机制](#10-数据同步机制)
11. [项目成果与技术亮点](#11-项目成果与技术亮点)
12. [存在问题与未来改进计划](#12-存在问题与未来改进计划)
13. [项目维护信息](#13-项目维护信息)

---

## 1. 项目整体架构

项目采用模块化设计,主要包含以下核心流水线:

1.  **数据层**:
    *   **爬虫系统**: 从九寨沟官网和天气网抓取原始数据
    *   **存储系统**: CSV 文件用于离线训练,SQLite 数据库用于在线服务
2.  **模型层**:
    *   **预处理**: 清洗数据,生成滞后特征、周期性时间编码和节假日标记
    *   **训练**: 基于 TensorFlow/Keras 的多种模型架构,支持自动超参数调整
3.  **应用层**:
    *   **后端**: Flask 提供 RESTful API 和模型推理服务
    *   **前端**: 基于 Bootstrap 5 和 ECharts 的响应式数据仪表盘

### 目录结构

```text
FYP/
├── Jiuzhaigou_Crawler/    # 爬虫子系统
│   ├── tourism_num.py     # 客流数据抓取
│   ├── weather_jiuzhaigou.py # 天气数据抓取
│   └── merge_tourism_weather.py # 数据合并与 Open-Meteo 补全
├── data/                  # 数据存储
│   ├── raw/               # 原始爬取数据 (CSV)
│   └── processed/         # 预处理后的特征数据 (CSV)
├── models/                # 模型代码
│   ├── common/            # 通用模块
│   │   ├── preprocess.py  # 特征工程
│   │   ├── evaluator.py   # 评估模块
│   │   └── visualization.py # 可视化模块
│   ├── lstm/              # LSTM 系列模型
│   │   ├── train_seq2seq_attention_8features.py
│   │   ├── train_lstm_8features.py
│   │   └── train_lstm_4features.py
│   └── gru/               # GRU 系列模型
│       ├── train_gru_8features.py
│       └── train_gru_4features.py
├── output/                # 训练输出目录
│   ├── runs/              # 当前运行结果
│   │   └── run_YYYYMMDD_HHMMSS_lb{lookback}_ep{epochs}_{model}_{features}features/
│   │       ├── figures/   # 可视化图表(纯英文)
│   │       └── weights/   # 模型权重 (.h5/.keras)
│   └── backups/           # 历史备份
│       └── backup_YYYYMMDD_HHMM_{tag}/
│           └── run_*/     # 备份的运行结果
├── web_app/               # Web 应用
│   ├── app.py             # Flask 后端入口
│   ├── models.py          # SQLAlchemy 数据库模型
│   ├── static/            # 静态资源
│   │   ├── js/            # JavaScript 文件
│   │   │   ├── dashboard_v2.js  # Dashboard v2 前端逻辑
│   │   │   └── ...
│   │   └── css/           # 样式文件
│   │       ├── dashboard_v2.css
│   │       └── uiw_components.css
│   └── templates/         # HTML 模板
│       ├── base.html      # 基础模板
│       ├── dashboard_v2.html  # Dashboard v2 页面
│       └── ...
├── scripts/               # 辅助脚本
│   └── sync_to_cloud.py   # 数据同步脚本 (CSV -> SQLite)
├── run_pipeline.py        # 一键运行流水线
├── run_benchmark.py       # 全模型基准测试
└── jiuzhaigou_fyp.db      # SQLite 数据库
```

---

## 2. 快速开始

### 环境准备

推荐使用 **Conda** 创建独立的虚拟环境:

```bash
# 激活环境
conda activate FYP

# 安装项目依赖
pip install -r requirements_all.txt
```

### 数据获取与准备

在运行训练流水线之前,必须先获取最新的数据:

```bash
# 一键执行:客流抓取、天气抓取、数据合并补全
python Jiuzhaigou_Crawler/run_crawler.py
```

如需单独调试,可分别运行 `tourism_num.py` 和 `weather_jiuzhaigou.py`。

### 运行训练流程

使用 `run_pipeline.py` 一键执行"特征工程 → 模型训练"流程:

```bash
# 运行 LSTM 8特征模型(默认 120 epochs, lookback 30)
python run_pipeline.py --model lstm --features 8

# 运行 Seq2Seq+Attention 8特征模型
python run_pipeline.py --model seq2seq_attention --features 8

# 仅运行训练,跳过预处理
python run_pipeline.py --skip-preprocess --model gru --features 8
```

### 基准测试

使用 `run_benchmark.py` 同时运行所有模型并生成性能对比报告:

```bash
# 运行全模型基准测试
python run_benchmark.py
```

### 启动 Web 应用

1.  **同步数据**: 将最新的训练结果写入数据库
    ```bash
    python scripts/sync_to_cloud.py
    ```

2.  **启动服务**:
    ```bash
    cd web_app
    python app.py
    ```

3.  访问浏览器: `http://localhost:5000/dashboard/v2`

---

## 3. 数据爬取子系统

位于 `Jiuzhaigou_Crawler/` 目录下。

*   **客流抓取 (`tourism_num.py`)**:
    *   **数据源**: 九寨沟官网新闻公告栏
    *   **方法**: 使用 `requests` 获取页面,`BeautifulSoup` 解析 HTML 表格
    *   **逻辑**: 支持分页抓取,利用正则表达式从标题中提取入园人数

*   **天气抓取 (`weather_jiuzhaigou.py`)**:
    *   **数据源**: 2345 天气王历史数据接口
    *   **方法**: 直接请求后端 AJAX 接口 (`/Pc/GetHistory`)
    *   **内容**: 获取每日最高/最低温、天气状况、风向风力、AQI

*   **数据合并 (`merge_tourism_weather.py`)**:
    *   **逻辑**: 以日期为主键,执行 Inner Join 合并客流和天气数据
    *   **容错**: 集成 **Open-Meteo API**,当主天气源缺失数据时,根据经纬度自动调用 Open-Meteo 的历史存档或预报接口进行补全,确保数据连续性

---

## 4. 数据预处理与特征工程

位于 `models/common/preprocess.py`。

### 数据清洗

*   单位去除(如 "15°C" → 15.0)
*   天气描述标准化(中文 → 英文代码,如 "晴" → "SUNNY")
*   缺失值填充(Open-Meteo 补全或前向填充)

### 特征工程

*   **时间特征**: 提取 Month, Day of Week,并进行 **Sin/Cos 周期性编码** (`month_sin`, `month_cos`),保留时间的循环特性
*   **节假日特征**: 集成 `chinese_calendar` 库,精准标记中国法定节假日 (`is_holiday`) 和调休工作日
*   **滞后特征**: 生成 `lag_1`, `lag_7`, `lag_14`, `lag_28`(过去 1/7/14/28 天的客流)
*   **滚动统计**: 计算 7 天和 14 天的滑动平均值 (`rolling_mean`) 和标准差 (`rolling_std`)

### 当前特征版本

**4特征版本(基础版)**:
1.  `visitor_count_scaled`: 归一化后的历史客流
2.  `month_norm`: 归一化月份
3.  `day_of_week_norm`: 归一化星期
4.  `is_holiday`: 节假日标记 (0/1)

**8特征版本(增强版)**:
1.  `visitor_count_scaled`: 归一化后的历史客流
2.  `month_norm`: 归一化月份
3.  `day_of_week_norm`: 归一化星期
4.  `is_holiday`: 节假日标记 (0/1)
5.  `tourism_num_lag_7_scaled`: 滞后7天客流(归一化)
6.  `meteo_precip_sum_scaled`: 降水量(归一化)
7.  `temp_high_scaled`: 最高温度(归一化)
8.  `temp_low_scaled`: 最低温度(归一化)

---

## 5. 模型架构演进

### 基线模型

**LSTM (Long Short-Term Memory)**
- **位置**: `models/lstm/train_lstm_4features.py` (4特征) 和 `models/lstm/train_lstm_8features.py` (8特征)
- **架构**: 双向LSTM + Dropout + 注意力机制
- **特点**: 支持序列预测,适合处理时间序列数据的长期依赖关系

**GRU (Gated Recurrent Unit)**
- **位置**: `models/gru/train_gru_4features.py` (4特征) 和 `models/gru/train_gru_8features.py` (8特征)
- **架构**: GRU + Dropout + 自定义损失函数
- **特点**: 相比LSTM参数更少,训练更快

### 核心架构:Seq2Seq + Attention(非自回归直接多步预测)

**位置**: `models/lstm/train_seq2seq_attention_8features.py`

**架构特点**:
- **编码器**: 双向LSTM,将30步历史序列编码为隐藏状态(8个特征)
- **解码器**: 单向LSTM + Bahdanau注意力,一次性输出未来步预测(7个外部特征)
- **特征压缩**: 1D卷积压缩(Encoder: 8→128维,Decoder: 7→128维)
- **注意力机制**: 动态分配特征权重,对节假日等关键特征给予更高权重

**核心优化**:
- Decoder输入不含visitor_count,纯外部特征驱动
- 杜绝自回归毒药,直接多步预测
- 自定义非对称损失函数,对节假日预测偏低给予更高惩罚

---

## 6. 最新性能指标

### 2026年4月3日基准测试结果

基于 `output/backups/backup_20260403_0209_weatherhaz/` 的完整训练结果:

#### 核心指标
| 模型 | MAE | RMSE | MAPE | R2 |
|------|-----|------|------|----|
| LSTM (8特征) | 4623.18 | 6987.45 | 25.34% | 0.68 |
| GRU (8特征) | 4512.90 | 6876.34 | 24.89% | 0.70 |
| **Seq2Seq+Attention (8特征)** | **4428.82** | **6499.69** | **22.05%** | **0.73** |

#### 关键发现
- **最佳模型**: Seq2Seq+Attention (8特征) 综合表现最佳
- **特征提升**: 8特征版本相比4特征版本在所有模型上都有显著提升
- **架构优势**: Seq2Seq+Attention架构在RMSE和MAPE指标上有明显优势

#### 推荐方案
- **生产部署**: 使用 Seq2Seq+Attention (8特征) 模型
- **快速预测**: 使用 GRU (8特征) 模型
- **基线对比**: 保留 LSTM (4特征) 作为基准模型

---

## 7. MLOps 与工程化规范

### 7.1 中央调度流水线

项目引入了统一的中央调度脚本 `run_pipeline.py`,负责协调整个训练流程。

#### 支持的模型白名单

目前支持的模型版本:
- **LSTM 模型**:4特征版和8特征版
- **GRU 模型**:4特征版和8特征版
- **Seq2Seq+Attention 模型**:8特征版

#### 使用方法

```bash
# 运行 LSTM 8特征模型
python run_pipeline.py --model lstm --features 8

# 运行 GRU 4特征模型
python run_pipeline.py --model gru --features 4

# 运行 Seq2Seq+Attention 8特征模型
python run_pipeline.py --model seq2seq_attention --features 8
```

### 7.2 标准化输出结构

所有模型的训练输出统一到 `output/runs/` 目录下的时间戳文件夹中:

```
output/
├── runs/                  # 当前运行结果
│   └── run_YYYYMMDD_HHMMSS_lb{lookback}_ep{epochs}_{model}_{features}features/
│       ├── figures/       # 可视化图表(纯英文)
│       │   ├── confusion_matrix_1.png
│       │   ├── confusion_matrix_2.png
│       │   ├── loss.png
│       │   └── true_vs_pred.png
│       └── weights/       # 模型权重文件
│           └── model.h5
└── backups/               # 历史备份
    └── backup_YYYYMMDD_HHMM_{tag}/
        └── run_*/         # 备份的运行结果
```

### 7.3 统一评估与可视化规范

所有模型训练完成后,会自动调用 `models/common/evaluator.py` 和 `models/common/visualization.py` 进行标准化评估和可视化。

#### 强制输出的四大标准图表

1. **Confusion Matrix 1 (Count)**:显示分类结果的绝对数量
2. **Confusion Matrix 2 (Normalized)**:显示归一化后的百分比
3. **Training vs Validation Loss**:训练和验证过程的损失曲线
4. **Test Set: True vs Predicted**:测试集的真实值与预测值的时序对比图

#### 严格的 UI 规范与 Zero Chinese Policy

**强制英文输出**:
- 所有图表强制纯英文输出(消除编码问题)
- 混淆矩阵标签:`['non_peak', 'peak']`
- Loss 图 Legend:`['train_loss', 'val_loss']`
- 对比图配色:True 蓝色,Pred 橙色
- 所有轴标签、标题、刻度均为英文

**可视化参数设置**:
- 混淆矩阵配色:`cmap=plt.cm.Blues`(高级蓝色系)
- 图表分辨率:300 DPI(保证清晰度)
- 时间轴格式:ISO 日期格式(YYYY-MM-DD)

#### 峰值阈值定义

在本系统的评估模块中,混淆矩阵分类所依据的 peak(极端峰值)与 non_peak(日常客流),是基于真实的客流绝对人数阈值进行划分的。系统设定的峰值阈值为 **18500人**。预测值和真实值在参与阈值判定和绘制混淆矩阵前,均已严格通过 Scaler 的逆变换还原为真实的客流人数。

### 7.4 预警定义与风险等级

#### 关键常量与默认值

| 名称 | 默认值 | 含义 |
|---|---:|---|
| `H` | 7 | 预测步数(未来 7 天) |
| `DEFAULT_PEAK_THRESHOLD` | 18500 | 峰值客流阈值 |
| Weather quantiles | `Q0.90 / Q0.10` | 天气危险阈值按训练期历史分位数确定 |

#### 拥挤预警 (crowd_alert)

基于固定阈值的定义:
- `crowd_alert = 1` 当 `visitor_count ≥ 18500`
- `crowd_alert = 0` 否则

四级拥挤风险映射(基于阈值 T = 18500 的百分比):
- **绿色**: `visitor_count < 0.70·T`
- **黄色**: `0.70·T ≤ visitor_count < 0.85·T`
- **橙色**: `0.85·T ≤ visitor_count < 1.00·T`
- **红色**: `visitor_count ≥ 1.00·T`

#### 天气危险 (weather_hazard)

基于历史分位数阈值的定义(仅使用训练期数据计算阈值):

1. 计算训练期阈值:
   - `P90 = Quantile(meteo_precip_sum_train, 0.90)`
   - `TH90 = Quantile(temp_high_train, 0.90)`
   - `TL10 = Quantile(temp_low_train, 0.10)`

2. 应用固定阈值到验证/测试集:
   - `weather_hazard = 1` 当 (`meteo_precip_sum ≥ P90`) 或 (`temp_high ≥ TH90`) 或 (`temp_low ≤ TL10`)
   - `weather_hazard = 0` 否则

#### 综合适宜性预警 (suitability_warning)

推荐组合规则(OR):
- `suitability_warning` 当拥挤风险或天气危险任一指示风险时触发
- 严重程度取两个组件的最大值

二值化形式(用于评估/F1报告):
- `suitability_warning_bin = 1` 当 `level_suitability ∈ {橙色, 红色}`
- `suitability_warning_bin = 0` 当 `level_suitability ∈ {绿色, 黄色}`

### 7.5 模型选择策略

选择冠军模型时采用"预警优先"策略:

**步骤 0 - 定义目标标签**:
- 使用 `suitability_warning_bin` 作为主要分类标签

**步骤 1 - 硬约束(安全/业务要求)**:
- 要求 **`Recall_warning ≥ 0.80`**(在滚动评估下测量)
- 任何未达到此约束的模型不具备成为冠军的资格

**步骤 2 - 主要指标(排序合格模型)**:
- 最大化 **`suitability_warning_bin` 的加权 F1**

**步骤 3 - 平局决胜(概率预警质量)**:
1. 更低的 **Brier score**
2. 更低的 **ECE (Expected Calibration Error)**

**辅助报告(回归)**:
- 报告 MAE / RMSE / sMAPE,对于多步预测使用水平权重进行加权汇总

---

## 8. Web 应用架构

位于 `web_app/` 目录下，采用 **Flask** 框架。

### 后端
*   **API 端点**:
    *   `GET /api/forecast`: 返回预测数据、历史数据和节假日配置（推荐使用）
    *   `GET /api/data`: 返回全量历史数据（旧版接口，保留兼容性）
    *   `POST /api/predict`: 接收预测请求（指定日期或天数）

*   **在线推理引擎**:
    *   应用启动时自动加载最新的 `.h5` 或 `.keras` 模型
    *   **滚动预测**: 针对未来多天的预测，系统采用递归方式，将前一天的预测值作为当天的输入特征
    *   **归一化处理**: 使用 `MinMaxScaler` 在线对输入数据进行缩放，输出后反归一化

### 前端

#### Dashboard v2（推荐使用）
*   **访问路径**: `http://localhost:5000/dashboard/v2`
*   **设计**: 深色科技风 UI，基于 Bootstrap 5
*   **可视化**: 使用 **ECharts** 绘制交互式折线图，支持缩放、拖拽
    *   **实线**: 真实历史数据
    *   **虚线**: AI 预测数据
    *   **区域高亮**: 自动在图表背景中标记法定节假日
*   **交互功能**:
    *   支持一键预测未来 1/3/7 天
    *   支持中英文国际化切换
    *   支持日间/夜间模式切换
    *   预测时显示 Loading 动画
    *   点击图表数据点可查看详细信息

#### 旧版 Dashboard
*   **访问路径**: `http://localhost:5000/dashboard`
*   **状态**: 保留用于兼容性测试

---

## 9. 数据库设计

使用 **SQLite** 数据库 (`jiuzhaigou_fyp.db`)，通过 SQLAlchemy ORM 管理。

**表名**: `traffic_records`

| 字段名 | 类型 | 描述 |
| :--- | :--- | :--- |
| `id` | Integer (PK) | 自增主键 |
| `record_date` | Date (Unique) | 记录日期，核心索引 |
| `actual_visitor` | Integer | 真实游客人数（历史数据） |
| `predicted_visitor` | Integer | 模型预测人数 |
| `is_forecast` | Boolean | 标识位（0=历史验证, 1=未来预测） |

---

## 10. 数据同步机制

脚本: `scripts/sync_to_cloud.py`

为了连接离线训练和在线服务，系统设计了同步机制：

1.  **读取**: 自动扫描 `data/processed/` 获取最新的历史数据 CSV，扫描 `output/runs/` 获取最新的测试集预测结果 CSV
2.  **重置**: 每次同步时重建 `traffic_records` 表，确保数据纯净
3.  **写入**:
    *   将历史真实数据批量写入 `actual_visitor`
    *   使用 **Upsert** 逻辑将模型的测试集预测结果写入 `predicted_visitor`
4.  **触发**: 该脚本可在 Web 应用启动时自动调用，也可手动运行

---

## 11. 项目成果与技术亮点

### 架构优化
1. **统一代码结构**: 所有模型脚本按照特征数量和架构类型进行规范化命名
2. **版本管理**: 支持4特征和8特征版本的基线模型，便于对比分析
3. **自动化部署**: 提供一键运行和基准测试脚本，简化部署流程

### 性能提升
1. **特征工程优化**: 从4特征扩展到8特征，包含天气和滞后特征
2. **架构升级**: 引入Seq2Seq+Attention架构，解决数据泄露问题
3. **损失函数优化**: 自定义非对称损失函数，对节假日预测偏低给予更高惩罚

### 工程化改进
1. **运行名称规范化**: 使用标准格式 `run_YYYYMMDD_HHMMSS_lb{lookback}_ep{epochs}_{model_name}_{features}features`
2. **项目文档更新**: 详细记录架构演进和性能指标
3. **测试体系完善**: 提供快速测试和基准测试脚本，确保代码质量
4. **备份机制**: 自动备份重要训练结果到 `output/backups/` 目录

### 实际应用价值
1. **景区管理**: 辅助景区提前调配资源，应对客流高峰
2. **游客服务**: 提供准确的客流预测，帮助游客合理安排行程
3. **数据驱动决策**: 基于历史数据和预测结果，优化景区运营策略

---

## 12. 存在问题与未来改进计划

### 模型泛化能力
* **当前挑战**: 模型在极端天气和特殊事件下的预测精度仍有提升空间
* **改进方向**: 引入更多外部特征（如社交媒体数据、交通数据），增强模型鲁棒性

### 实时预测（增量学习）
* **当前挑战**: 目前为离线训练，无法实时更新模型
* **改进方向**: 设计增量学习机制，支持在线模型更新

### 可解释性（SHAP）
* **当前挑战**: 深度学习模型缺乏可解释性
* **改进方向**: 集成 SHAP 等可解释性工具，可视化特征重要性

### Walk-forward Evaluation
* **目的**: 用贴近真实部署的方式评估"未来预测"性能，避免一次性随机切分带来的时间泄露
* **计划**:
  1. 以时间为轴构建多个滚动窗口：`train → val → test`
  2. 每个窗口独立训练并记录回归指标与预警指标
  3. 汇总跨窗口均值与方差，用作冠军模型选择的主要依据

### Ablation Study
* **目的**: 量化关键模块对性能与预警质量的贡献
* **计划**:
  - 特征消融：移除天气特征与滞后特征，观察预警 F1 与回归误差变化
  - 架构消融：对比 LSTM/GRU/Seq2Seq(+Attention)，控制超参数
  - 损失函数消融：对比对称/非对称损失，重点观察峰值/节假日段误差

### Calibration
* **目的**: 将预警输出作为概率使用时，保证概率可解释
* **计划**:
  - 评估：Reliability diagram、ECE、Brier score
  - 校准方法：Platt scaling / Isotonic regression
  - 报告：校准前后对比，说明阈值选择对 F1 与误报/漏报的影响

### 部署优化
* **当前挑战**: Web应用部署在本地环境，访问速度有限
* **改进方向**: 部署到云端服务器，配置负载均衡和自动缩放

---

## 13. 项目维护信息

**开发团队**: 个人项目  
**技术栈**: Python, TensorFlow/Keras, Flask, ECharts, SQLite  
**项目状态**: 持续开发中  
**最后更新**: 2026年4月3日
