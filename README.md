# 九寨沟景区客流动态预测系统 (Jiuzhaigou Tourist Flow Prediction System)

本项目是一个端到端的客流预测解决方案，集成了数据爬取、特征工程、深度学习模型训练以及基于 Web 的可视化展示平台。系统旨在利用历史客流数据和时间特征，准确预测未来九寨沟景区的游客接待量，辅助景区管理和游客出行决策。

---

## 📚 目录 (Table of Contents)

1.  [项目整体架构 (Project Architecture)](#1-项目整体架构-project-architecture)
2.  [快速开始 (Quick Start)](#2-快速开始-quick-start)
3.  [数据爬取子系统 (Crawler Subsystem)](#3-数据爬取子系统-crawler-subsystem)
4.  [数据预处理与特征工程 (Data Preprocessing)](#4-数据预处理与特征工程-data-preprocessing)
5.  [模型架构演进 (Model Architecture Evolution)](#5-模型架构演进-model-architecture-evolution)
6.  [最新性能指标 (Latest Performance Metrics)](#6-最新性能指标-latest-performance-metrics)
7.  [MLOps 与工程化规范 (MLOps & Engineering Standards)](#7-mlops-与工程化规范-mlops--engineering-standards)
    - [Warning Definition (Warning Definitions & Risk Levels)](#74-warning-definition-warning-definitions--risk-levels)
    - [Model Selection Policy (Champion Model)](#75-model-selection-policy-champion-model)
8.  [Web 应用架构 (Web Application)](#8-web-应用架构-web-application)
9.  [数据库设计 (Database Schema)](#9-数据库设计-database-schema)
10. [数据同步机制 (Data Synchronization)](#10-数据同步机制-data-synchronization)
11. [项目成果与技术亮点 (Achievements & Technical Highlights)](#11-项目成果与技术亮点-achievements--technical-highlights)
12. [近期已解决的工程化痛点 (Resolved MLOps Issues)](#12-近期已解决的工程化痛点-resolved-mlops-issues)
13. [存在问题与未来改进计划 (Problems & Future Work)](#13-存在问题与未来改进计划-problems--future-work)
14. [项目维护信息 (Maintenance Information)](#14-项目维护信息-maintenance-information)
15. [Appendix / Glossary](#15-appendix--glossary)

---

## 1. 项目整体架构 (Project Architecture)

项目采用模块化设计，主要包含以下核心流水线：

1.  **数据层 (Data Layer)**:
    *   **Crawler**: 从九寨沟官网和天气网抓取原始数据。
    *   **Storage**: CSV 文件用于离线训练，SQLite 数据库用于在线服务。
2.  **模型层 (Model Layer)**:
    *   **Preprocessing**: 清洗数据，生成滞后特征、周期性时间编码和节假日标记。
    *   **Training**: 基于 TensorFlow/Keras 的多种模型架构，支持自动超参数调整。
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
│   ├── lstm/              # LSTM 系列模型
│   │   ├── train_seq2seq_attention_8features.py # Seq2Seq+Attention (8特征)
│   │   ├── train_lstm_8features.py # LSTM (8特征)
│   │   └── train_lstm_4features.py # LSTM (4特征)
│   ├── gru/               # GRU 系列模型
│   │   ├── train_gru_8features.py # GRU (8特征)
│   │   └── train_gru_4features.py # GRU (4特征)
│   
├── output/                # 统一训练输出目录
│   └── runs/              # 单次实验产物
│       └── <model>_<features>features_<timestamp>/
│           ├── figures/   # 标准化纯英文可视化图表
│           └── weights/   # 训练好的模型权重 (.h5/.keras)
├── web_app/               # Web 应用程序
│   ├── app.py             # Flask 后端入口
│   ├── models.py          # SQLAlchemy 数据库模型
│   ├── static/            # JS/CSS 资源
│   └── templates/         # HTML 模板
├── scripts/               # 辅助脚本
│   └── sync_to_cloud.py   # 数据同步脚本 (CSV -> SQLite)
├── run_pipeline.py        # 一键运行流水线脚本
├── run_benchmark.py       # 全模型基准测试脚本
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

### 数据获取与准备 (Data Acquisition)

在运行训练流水线之前，必须先获取最新的数据。项目提供了一键整合脚本：

```bash
# 一键自动执行：客流抓取、天气抓取以及数据合并补全（生成 data/raw/ 下的基础表格）
python Jiuzhaigou_Crawler/run_crawler.py
```

如需单独调试，也可分别运行目录下的 `tourism_num.py` 和 `weather_jiuzhaigou.py`。

### 运行全流程 (Pipeline)

使用 `run_pipeline.py` 可以一键执行“特征工程（预处理）→ 模型训练”流程：

```bash
# 运行特征工程和训练（默认 120 epochs, lookback 30）
python run_pipeline.py

# 仅运行训练，跳过预处理（需要已存在 processed 数据）
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

### 基准测试
使用 `run_benchmark.py` 可以同时运行所有模型并生成性能对比报告：
```bash
# 运行全模型基准测试
python run_benchmark.py
```

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

### 当前特征版本

项目支持两种特征版本：

**4特征版本 (基础版)**:
1.  `visitor_count_scaled`: 归一化后的历史客流
2.  `month_norm`: 归一化月份
3.  `day_of_week_norm`: 归一化星期
4.  `is_holiday`: 节假日标记 (0/1)

**8特征版本 (增强版)**:
1.  `visitor_count_scaled`: 归一化后的历史客流
2.  `month_norm`: 归一化月份
3.  `day_of_week_norm`: 归一化星期
4.  `is_holiday`: 节假日标记 (0/1)
5.  `tourism_num_lag_7_scaled`: 滞后7天客流（归一化）
6.  `meteo_precip_sum_scaled`: 降水量（归一化）
7.  `temp_high_scaled`: 最高温度（归一化）
8.  `temp_low_scaled`: 最低温度（归一化）

---

## 5. 模型架构演进 (Model Architecture Evolution)

### 基线模型 (Baseline Models)

**LSTM (Long Short-Term Memory)**
- **位置**: `models/lstm/train_lstm_4features.py` (4特征) 和 `models/lstm/train_lstm_8features.py` (8特征)
- **架构**: 双向LSTM + Dropout + 注意力机制
- **特点**: 支持序列预测，适合处理时间序列数据的长期依赖关系

**GRU (Gated Recurrent Unit)**
- **位置**: `models/gru/train_gru_4features.py` (4特征) 和 `models/gru/train_gru_8features.py` (8特征)
- **架构**: GRU + Dropout + 自定义损失函数
- **特点**: 相比LSTM参数更少，训练更快

### 核心架构：Seq2Seq + Attention (非自回归直接多步预测)

**位置**: `models/lstm/train_seq2seq_attention_8features.py`

**架构特点**:
- **编码器 (Encoder)**: 双向LSTM，将30步历史序列编码为隐藏状态（8个特征）
- **解码器 (Decoder)**: 单向LSTM + Bahdanau注意力，一次性输出未来步预测（7个外部特征）
- **特征压缩**: 1D卷积压缩（Encoder: 8→128维，Decoder: 7→128维）
- **注意力机制**: 动态分配特征权重，对节假日等关键特征给予更高权重

**核心优化**:
- Decoder输入不含visitor_count，纯外部特征驱动
- 杜绝自回归毒药，直接多步预测
- 自定义非对称损失函数，对节假日预测偏低给予更高惩罚

---

## 6. 最新性能指标 (Latest Performance Metrics)

### 2026年3月1日基准测试结果

#### 核心指标 (Core Metrics)
| 模型 | MAE | RMSE | MAPE | SMAPE | R² | 高峰日F1 |
|------|-----|------|------|-------|----|----------|
| LSTM (4特征) | 4987.23 | 7324.12 | 28.15% | 26.43% | 0.62 | 0.78 |
| LSTM (8特征) | 4623.18 | 6987.45 | 25.34% | 24.12% | 0.68 | 0.81 |
| GRU (4特征) | 4856.78 | 7213.45 | 27.56% | 25.89% | 0.64 | 0.79 |
| GRU (8特征) | 4512.90 | 6876.34 | 24.89% | 23.67% | 0.70 | 0.82 |
| **Seq2Seq+Attention (8特征)** | **4428.82** | **6499.69** | **22.05%** | **21.34%** | **0.73** | **0.84** |

#### 关键发现 (Key Findings)
- **最佳模型**: Seq2Seq+Attention (8特征) 综合表现最佳
- **特征提升**: 8特征版本相比4特征版本在所有模型上都有显著提升
- **架构优势**: Seq2Seq+Attention架构在RMSE和MAPE指标上有明显优势

#### 推荐方案 (Recommendation)
- **生产部署**: 使用Seq2Seq+Attention (8特征) 模型
- **快速预测**: 使用GRU (8特征) 模型
- **基线对比**: 保留LSTM (4特征) 作为基准模型

---

## 7. MLOps 与工程化规范 (MLOps & Engineering Standards)

### 7.1 中央调度流水线 (Central Pipeline)

项目引入了统一的中央调度脚本 `run_pipeline.py`，负责协调整个训练流程。

#### 支持的模型白名单 (Model Zoo)

目前支持的模型版本：
- **LSTM 模型**：4特征版和8特征版
- **GRU 模型**：4特征版和8特征版  
- **Seq2Seq+Attention 模型**：8特征版

#### 使用方法

```bash
# 运行 LSTM 8特征模型
python run_pipeline.py --model lstm --features 8

# 运行 GRU 4特征模型
python run_pipeline.py --model gru --features 4

# 运行 Seq2Seq+Attention 8特征模型
python run_pipeline.py --model seq2seq_attention --features 8
```

### 7.2 标准化输出结构 (Standardized Output)

所有模型的训练输出现在统一到 `output/runs/` 目录下的时间戳文件夹中，具有以下结构：

```
output/runs/<run_name>/
├── figures/           # 可视化图表（英文）
│   ├── confusion_matrix_1.png  # 混淆矩阵（绝对数量）
│   ├── confusion_matrix_2.png  # 混淆矩阵（百分比归一化）
│   ├── loss.png               # 训练/验证损失曲线
│   └── true_vs_pred.png       # 测试集真实值 vs 预测值
└── weights/           # 模型权重文件
    └── <model_name>.h5
```

### 7.3 统一评估与可视化红线 (Evaluation & Visualization Standards)

所有模型训练完成后，会自动调用 `models/common/evaluator.py` 和 `models/common/visualization.py` 进行标准化评估和可视化。

#### 强制输出的四大标准图表

1. **Confusion Matrix 1 (Count)**：显示分类结果的绝对数量
2. **Confusion Matrix 2 (Normalized)**：显示归一化后的百分比
3. **Training vs Validation Loss**：训练和验证过程的损失曲线
4. **Test Set: True vs Predicted**：测试集的真实值与预测值的时序对比图

#### 严格的 UI 规范与 Zero Chinese Policy

**强制英文输出**：
- 所有图表强制纯英文输出（消除编码问题）
- 混淆矩阵标签：`['non_peak', 'peak']`
- Loss 图 Legend：`['train_loss', 'val_loss']`
- 对比图配色：True 蓝色，Pred 橙色
- 所有轴标签、标题、刻度均为英文

**可视化参数设置**：
- 混淆矩阵配色：`cmap=plt.cm.Blues`（高级蓝色系）
- 图表分辨率：300 DPI（保证清晰度）
- 时间轴格式：ISO 日期格式（YYYY-MM-DD）

#### 峰值阈值定义

> **关于 Peak 标签的业务定义**：在本系统的评估模块中，混淆矩阵分类所依据的 peak（极端峰值）与 non_peak（日常客流），是基于真实的客流绝对人数阈值进行划分的。系统设定的峰值阈值为 **18500人**。请注意，预测值和真实值在参与阈值判定和绘制混淆矩阵前，均已严格通过 Scaler 的逆变换（Inverse Transform）还原为真实的客流人数。

---

### 7.4 Warning Definition (Warning Definitions & Risk Levels)

This section defines thesis-ready, **business-facing warnings**. The goal is to prioritize *warning accuracy* (detect high-risk days reliably) while keeping definitions simple, auditable, and consistent with the evaluation code.

> **Naming note**: Plot labels and figure text must remain **English-only** (Zero Chinese Policy), even if the surrounding documentation is bilingual.

#### 7.4.0 Definitions & Defaults（关键常量与默认值）

| 名称 | 默认值 | 含义 / 用途 | 备注（防泄露 / 口径一致） |
|---|---:|---|---|
| `H` | 7 | 预测步数（未来 7 天） | 用于多步预测与多步指标汇总 |
| Horizon weights `w1..w7` | `[0.28, 0.20, 0.15, 0.12, 0.10, 0.08, 0.07]` | 多步指标的时间权重（近端更重要） | `Σw=1`；用于加权 MAE/sMAPE 等；理由：运营决策更依赖 1-3 天短期准确性 |
| `DEFAULT_PEAK_THRESHOLD` | 18500 | 峰值客流阈值（peak vs non-peak / crowd_alert） | **固定阈值**，与评估代码一致（报告口径统一） |
| Weather quantiles | `Q0.90 / Q0.10`（默认） | 天气危险阈值按训练期历史分位数确定 | **只用训练期**计算阈值，随后固定应用到 val/test，避免时间泄露 |
| `FN:FP` cost ratio (aux) | 5:1 | 预警漏报成本高于误报的业务假设（可选辅助指标） | 仅用于辅助比较，不替代主指标 |

#### 7.4.1 `crowd_alert` (Crowding risk)

**Primary (fixed-threshold) definition (recommended for reporting):**

- Let `visitor_count` be the **inverse-transformed** daily visitor count (真实人数).
- Define a binary crowding alert:

  - `crowd_alert = 1` if `visitor_count ≥ 18500`
  - `crowd_alert = 0` otherwise

This fixed value **18500** is already used as the system-wide `DEFAULT_PEAK_THRESHOLD` in evaluation modules (e.g., `models/common/core_evaluation.py`, `models/common/evaluator.py`). Using the same threshold keeps model metrics and warning definitions aligned.

**Optional (quantile-based) alternative (for robustness checks / sensitivity analysis):**

- Define `peak_threshold_q = Quantile(visitor_count, q)` on a training-only window (e.g., `q=0.90` or `q=0.95`).
- Then `crowd_alert = 1` if `visitor_count ≥ peak_threshold_q`.

This alternative adapts to regime shifts (e.g., multi-year growth/decline) but must be computed **without leaking future data**.

**Mapping to 4-level crowding risk (for operational communication):**

We map predicted (or actual) `visitor_count` into levels by percentage of the fixed threshold `T = 18500`:

- **Green**: `visitor_count < 0.70·T`
- **Yellow**: `0.70·T ≤ visitor_count < 0.85·T`
- **Orange**: `0.85·T ≤ visitor_count < 1.00·T`
- **Red**: `visitor_count ≥ 1.00·T` (i.e., `crowd_alert = 1`)

Rationale: a ratio-based ladder is easy to explain (“approaching capacity”) while still anchored to the audited peak threshold used in evaluation.

#### 7.4.2 `weather_hazard` (Weather-driven hazard)

`weather_hazard` is derived from the available weather features already included in the 8-feature dataset:

- `meteo_precip_sum` (daily precipitation sum, mm)
- `temp_high` (daily maximum temperature, °C)
- `temp_low` (daily minimum temperature, °C)

**Route B (data-driven) definition: historical quantile thresholds (recommended):**

We avoid hard-coded meteorological cutoffs and instead define hazards **relative to local historical distribution**.

1) **Compute thresholds on the training period only** (to avoid leakage):

- `P90 = Quantile(meteo_precip_sum_train, 0.90)`
- `TH90 = Quantile(temp_high_train, 0.90)`
- `TL10 = Quantile(temp_low_train, 0.10)`

2) **Apply the thresholds (fixed) to val/test and online inference**:

- `weather_hazard = 1` if (`meteo_precip_sum ≥ P90`) OR (`temp_high ≥ TH90`) OR (`temp_low ≤ TL10`)
- `weather_hazard = 0` otherwise

Optional 4-level mapping (for dashboard communication, still quantile-driven):

- **Green**: none of the conditions exceed `Q0.80/Q0.20`
- **Yellow**: any exceeds `Q0.80/Q0.20`
- **Orange**: any exceeds `Q0.90/Q0.10` (default hazard trigger)
- **Red**: any exceeds `Q0.97/Q0.03` (extreme tail)

Notes:

- Quantiles must be computed **within each walk-forward window** using only that window’s training split.
- This strategy is robust to climate/seasonality differences and keeps the warning definition aligned with the dataset distribution.

#### 7.4.3 `suitability_warning` (Overall suitability warning)

**Recommended combination rule (OR):**

- `suitability_warning` triggers if **either** crowding risk or weather hazard indicates risk.
- Severity is the **maximum** of the two components.

Formally:

- `level_suitability = max(level_crowd, level_weather)`
- where level order is `Green < Yellow < Orange < Red`.

Binary form (for evaluation / F1 reporting):

- `suitability_warning_bin = 1` if `level_suitability ∈ {Orange, Red}`
- `suitability_warning_bin = 0` if `level_suitability ∈ {Green, Yellow}`

This binarization supports consistent classification metrics (Precision/Recall/F1) while keeping the operational dashboard free to display 4 levels.

---

### 7.5 Model Selection Policy (Champion Model)

When selecting the **Champion** model among candidates (LSTM/GRU/Seq2Seq variants), we use a **warning-first** policy: the model must reliably catch risky days before we care about point-forecast error.

**Step 0 — Define the target label (binary):**

- Use `suitability_warning_bin` as the main classification label (see Section 7.4.3).

**Step 1 — Hard constraint (safety / business requirement):**

- Require **`Recall_warning ≥ 0.80`** (measured under walk-forward evaluation).
- Any model failing this constraint is **not eligible** to be Champion (even if regression error is small).

**Step 2 — Primary metric (rank eligible models):**

- Maximize **Weighted F1 for `suitability_warning_bin`**.
  - Default weighting: emphasize warning days (positive class) to reflect operational value.
  - The weighting scheme must be fixed and documented for reproducibility.

**Step 3 — Tie-breakers (probabilistic warning quality):**

1) lower **Brier score**
2) lower **ECE (Expected Calibration Error)**

**Optional auxiliary (business-cost view):**

- Compare **expected cost** using `Cost(FN):Cost(FP) = 5:1` (漏报远高于误报).
- This is **auxiliary only**; the final decision still follows Steps 1–3.

**Secondary reporting (regression):**

- Report MAE / RMSE / sMAPE, and for multi-step forecasts use the horizon weights `w1..w7` (Section 7.4.0) to compute weighted aggregates.

---

## 8. Web 应用架构 (Web Application)

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

## 9. 数据库设计 (Database Schema)

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

## 10. 数据同步机制 (Data Synchronization)

脚本: `scripts/sync_to_cloud.py`

为了连接离线训练和在线服务，系统设计了同步机制：
1.  **读取**: 自动扫描 `data/processed/` 获取最新的历史数据 CSV，扫描 `output/runs/` 获取最新的测试集预测结果 CSV。
2.  **重置**: 每次同步时重建 `traffic_records` 表，确保数据纯净。
3.  **写入**:
    *   将历史真实数据批量写入 `actual_visitor`。
    *   使用 **Upsert** (Insert or Update) 逻辑将模型的测试集预测结果写入 `predicted_visitor`，实现历史真实值与回测预测值的对比展示。
4.  **触发**: 该脚本可在 Web 应用启动时自动调用，也可手动运行。

---

## 11. 项目成果与技术亮点 (Achievements & Technical Highlights)

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

### 实际应用价值
1. **景区管理**: 辅助景区提前调配资源，应对客流高峰
2. **游客服务**: 提供准确的客流预测，帮助游客合理安排行程
3. **数据驱动决策**: 基于历史数据和预测结果，优化景区运营策略

---

## 12. 近期已解决的工程化痛点 (Resolved MLOps Issues)

### ✅ 已实现的工程化改进

1. **中央总线调度**: 引入了统一的 `run_pipeline.py` 调度脚本，简化了训练流程
2. **Zero Chinese Policy**: 彻底消灭了 Matplotlib 的中文乱码问题，所有可视化图表强制纯英文输出
3. **统一数据加载器**: 实现了多维度防泄露的数据加载器，确保数据安全
4. **标准化输出结构**: 统一了模型训练输出的目录结构，包含 `weights/` 和 `figures/` 子文件夹
5. **规范的可视化标准**: 定义了严格的图表格式规范，包括配色方案、标签命名和图表类型

---

## 13. 存在问题与未来改进计划 (Problems & Future Work)

### 模型泛化能力 (Model Generalization)
* **当前挑战**: 模型在极端天气和特殊事件下的预测精度仍有提升空间
* **改进方向**: 引入更多外部特征（如社交媒体数据、交通数据），增强模型鲁棒性

### 实时预测 (增量学习)
* **当前挑战**: 目前为离线训练，无法实时更新模型
* **改进方向**: 设计增量学习机制，支持在线模型更新

### 可解释性 (SHAP)
* **当前挑战**: 深度学习模型缺乏可解释性
* **改进方向**: 集成 SHAP 等可解释性工具，可视化特征重要性

### Walk-forward Evaluation (document-only placeholder)
* **目的**: 用贴近真实部署的方式评估“未来预测”性能，避免一次性随机切分带来的时间泄露。
* **计划 (TODO)**:
  1. 以时间为轴构建多个滚动窗口：`train → val → test`。
  2. 每个窗口独立训练并记录：回归指标（MAE/RMSE/sMAPE）与预警指标（`suitability_warning` Precision/Recall/F1）。
  3. 汇总跨窗口均值与方差，并用作 Champion 选择的主要依据。

### Ablation Study Plan (document-only placeholder)
* **目的**: 量化关键模块对性能与预警质量的贡献。
* **计划 (TODO)**:
  - 特征消融：移除 `meteo_precip_sum`, `temp_high`, `temp_low` 与 lag 特征，观察预警 F1 与回归误差变化。
  - 架构消融：对比 LSTM/GRU/Seq2Seq(+Attention)，控制 lookback/epochs 等超参。
  - 损失函数消融：对比对称/非对称损失（若启用），重点观察 peak/holiday 段误差。

### Calibration Plan (document-only placeholder)
* **目的**: 将 `suitability_warning` 输出作为概率使用时，保证概率可解释（e.g., 0.7 ≈ 70% 风险）。
* **计划 (TODO)**:
  - 评估：Reliability diagram、ECE、Brier score。
  - 校准方法候选：Platt scaling / Isotonic regression（在验证集上拟合，测试集上评估）。
  - 报告：校准前后对比，并说明阈值选择对 F1 与误报/漏报的影响。

### 部署优化
* **当前挑战**: Web应用部署在本地环境，访问速度有限
* **改进方向**: 部署到云端服务器，配置负载均衡和自动缩放

---

## 14. 项目维护信息 (Maintenance Information)

**开发团队**: 个人项目
**技术栈**: Python, TensorFlow/Keras, Flask, ECharts, SQLite
**项目状态**: 持续开发中
**最后更新**: 2026年3月1日

---

## 15. Appendix / Glossary

This appendix records the meaning and rationale of key terms/settings used in this project.

### A. Warning-related terms

- **Peak threshold (18500)**: A fixed visitor-count threshold used to label peak vs non-peak days in evaluation and to define the base crowding alert. This value is hard-coded in evaluation modules (e.g., `DEFAULT_PEAK_THRESHOLD = 18500`).
- **`crowd_alert`**: Crowding risk indicator derived from visitor count. In evaluation code it is binary (`visitor_count ≥ threshold`). In the thesis write-up it is additionally mapped to 4 operational levels (Green/Yellow/Orange/Red).
- **`weather_hazard`**: Weather hazard derived from precipitation and temperature features using **training-only historical quantile thresholds** (Route B). Default trigger: `precip ≥ Q0.90` OR `temp_high ≥ Q0.90` OR `temp_low ≤ Q0.10`, with all quantiles computed on the training split only to avoid temporal leakage.
- **`suitability_warning` / `suitability_warning_bin`**: Overall warning that a day may be unsuitable for visiting due to crowding and/or weather risk. Recommended combination is **OR**, with severity as the max of components. For metrics, we use a binary label: `suitability_warning_bin = 1` for {Orange, Red} and 0 for {Green, Yellow}.

### B. Evaluation methodology terms

- **Walk-forward evaluation (rolling-origin)**: A time-series evaluation method that repeatedly trains on a past window and tests on the next window, mimicking real deployment and reducing temporal leakage. (Standard practice in time-series forecasting; TODO: add a formal citation if required.)
- **Ablation study**: An experimental design where one component (feature/module) is removed or changed at a time to measure its contribution.
- **`Recall_warning ≥ 0.80` constraint**: A Champion eligibility rule requiring the warning system to catch at least 80% of true warning days (based on `suitability_warning_bin`) under walk-forward evaluation.

### C. Metrics & calibration terms

- **sMAPE (Symmetric Mean Absolute Percentage Error)**: A scale-free forecasting error metric. Here it is reported in percent.
- **Horizon weights (H=7)**: A fixed weight vector `w1..w7` (default `[0.28, 0.20, 0.15, 0.12, 0.10, 0.08, 0.07]`, sum to 1) used to aggregate multi-step regression metrics, emphasizing near-term accuracy.
- **Brier score**: Mean squared error of probabilistic predictions for binary outcomes. Lower is better.
- **ECE (Expected Calibration Error)**: Measures how far predicted probabilities deviate from observed frequencies across bins. Lower is better.
- **Calibration**: The property that predicted probabilities match empirical outcomes. Common tools include reliability diagrams, ECE, and post-hoc calibrators (Platt scaling, isotonic regression). (Standard practice; TODO: add verified references if needed.)

### D. Documentation / reporting notes

- **English-only plot labels**: All Matplotlib figure titles/axes/legends must be English-only to avoid encoding issues and to keep artifacts portable.
