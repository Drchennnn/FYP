# 九寨沟景区客流动态预测系统

本项目是一个端到端的客流预测解决方案，集成了数据爬取、特征工程、深度学习模型训练以及基于 Web 的可视化展示平台。系统旨在利用历史客流数据和时间特征，准确预测未来九寨沟景区的游客接待量，辅助景区管理和游客出行决策。

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
12. [已实现的方法论改进](#12-已实现的方法论改进)
13. [项目维护信息](#13-项目维护信息)

---

## 1. 项目整体架构

项目采用模块化设计，主要包含以下核心流水线：

1.  **数据层**：
    *   **爬虫系统**：从九寨沟官网和天气网抓取原始数据
    *   **存储系统**：CSV 文件用于离线训练，SQLite 数据库用于在线服务
2.  **模型层**：
    *   **预处理**：清洗数据，生成滞后特征、周期性时间编码和节假日标记
    *   **训练**：基于 TensorFlow/Keras 的多种模型架构，支持自动超参数调整
3.  **应用层**：
    *   **后端**：Flask 提供 RESTful API 和模型推理服务
    *   **前端**：基于 Bootstrap 5 和 ECharts 的响应式数据仪表盘

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
│   │   ├── core_evaluation.py  # 统一评估模块（含动态阈值）
│   │   ├── evaluator.py        # 基础评估模块
│   │   └── visualization.py   # 可视化模块
│   ├── lstm/              # LSTM 系列模型
│   │   ├── train_seq2seq_attention_8features.py  # Seq2Seq+Attention（含 Attention 热力图）
│   │   ├── train_lstm_8features.py
│   │   └── train_lstm_4features.py
│   └── gru/               # GRU 系列模型
│       ├── train_gru_8features.py
│       └── train_gru_4features.py
├── output/                # 训练输出目录
│   ├── runs/              # 当前运行结果
│   │   └── run_YYYYMMDD_HHMMSS_lb{lookback}_ep{epochs}_{model}_{features}features/
│   │       ├── figures/   # 可视化图表（含 attention_heatmap_mean.png）
│   │       └── weights/   # 模型权重 (.h5/.keras)
│   ├── backups/           # 历史备份
│   └── walk_forward/      # Walk-forward 评估结果
├── scripts/               # 辅助脚本
│   ├── walk_forward_eval.py   # Walk-forward 滚动窗口评估
│   └── sync_to_cloud.py       # 数据同步脚本 (CSV -> SQLite)
├── web_app/               # Web 应用
│   ├── app.py             # Flask 后端（含 APScheduler 定时爬取）
│   ├── static/js/dashboard_v2.js  # 三模型对比前端
│   └── templates/dashboard_v2.html
├── realtime/              # 实时服务模块
│   ├── daily_update.py    # 每日更新脚本
│   └── jiuzhaigou_crawler.py
├── run_pipeline.py        # 一键运行流水线
├── run_benchmark.py       # 全模型基准测试
└── jiuzhaigou_fyp.db      # SQLite 数据库
```

---

## 2. 快速开始

### 环境准备

推荐使用 **Conda** 创建独立的虚拟环境：

```bash
conda activate FYP
pip install -r requirements_all.txt
```

### 数据获取与准备

```bash
# 一键执行：客流抓取、天气抓取、数据合并补全
python Jiuzhaigou_Crawler/run_crawler.py
```

### 运行训练流程

```bash
# 使用10年数据训练（默认路径已配置）
python run_pipeline.py --model lstm --features 8
python run_pipeline.py --model gru --features 8
python run_pipeline.py --model seq2seq_attention
```

### Walk-forward 评估

```bash
# 4折滚动窗口评估（每折测试90天）
python scripts/walk_forward_eval.py --model gru --folds 4
python scripts/walk_forward_eval.py --model all --folds 4
```

### 基准测试

```bash
python run_benchmark.py
```

### 启动 Web 应用

```bash
cd web_app && python app.py
# 访问 http://localhost:5000/dashboard/v2
```

---

## 3. 数据爬取子系统

位于 `Jiuzhaigou_Crawler/` 目录下。

*   **客流抓取 (`tourism_num.py`)**：使用 `requests` + `BeautifulSoup` 从九寨沟官网解析入园人数
*   **天气抓取 (`weather_jiuzhaigou.py`)**：请求 2345 天气王历史 AJAX 接口，获取每日温度、天气、风力、AQI
*   **数据合并 (`merge_tourism_weather.py`)**：以日期为主键合并，集成 Open-Meteo API 补全缺失天气数据

**数据范围**：当前使用 2016-05-01 ~ 2026-04-02 共约 10 年数据（2719 条），存储于 `data/processed/jiuzhaigou_daily_features_2016-01-01_2026-04-02.csv`。

---

## 4. 数据预处理与特征工程

### 特征版本

**4特征版本（基础版）**：
1. `visitor_count_scaled`：归一化历史客流
2. `month_norm`：归一化月份
3. `day_of_week_norm`：归一化星期
4. `is_holiday`：节假日标记 (0/1)

**8特征版本（增强版）**：
在4特征基础上增加：
5. `tourism_num_lag_7_scaled`：滞后7天客流（归一化）
6. `meteo_precip_sum_scaled`：降水量（归一化）
7. `temp_high_scaled`：最高温度（归一化）
8. `temp_low_scaled`：最低温度（归一化）

---

## 5. 模型架构演进

### 基线模型

**LSTM**（`models/lstm/train_lstm_8features.py`）：LSTM-128-64-32 + Dropout(0.2)，单步预测

**GRU**（`models/gru/train_gru_8features.py`）：GRU-128-64 + Dropout(0.2)，参数更少，训练更快

### 核心架构：Seq2Seq + Attention（非自回归直接多步预测）

**位置**：`models/lstm/train_seq2seq_attention_8features.py`

**架构**：
- **编码器**：双向 LSTM（128 单元），将 30 步历史序列编码为隐藏状态
- **解码器**：单向 LSTM（256 单元）+ Bahdanau 注意力，直接输出未来 7 天预测
- **特征压缩**：1D 卷积（Encoder: 8→128 维，Decoder: 7→128 维）
- **Decoder 输入**：纯外部特征（不含 visitor_count），杜绝自回归数据泄露

**自定义非对称损失函数**：
- 节假日预测偏低：惩罚权重 ×2.0
- 非节假日高客流预测偏低：惩罚权重 ×1.5

**Attention 可视化**：训练完成后自动生成 `figures/attention_heatmap_mean.png`（测试集均值热力图）和 `figures/attention_heatmap_sample.png`（最新预测窗口样本），直观展示模型对历史时间步的关注分布。

---

## 6. 最新性能指标

### 2026年4月3日（10年数据重训后）

| 模型 | MAE | RMSE | SMAPE | Suitability F1 | Brier |
|------|-----|------|-------|----------------|-------|
| GRU (8特征) | **2,809** | **3,993** | **15.0%** | 0.971 | 0.036 |
| LSTM (8特征) | 3,881 | 5,104 | 20.3% | 0.966 | 0.043 |
| Seq2Seq+Attention (8特征) | 3,846 | 5,326 | 21.0% | 0.964 | 0.042 |

> 注：使用 2016-2026 共 10 年数据训练（2688 样本），测试集 268 天。峰值阈值由训练集 75 分位数动态计算。

**关键发现**：
- GRU 在回归精度上表现最佳（MAE 最低）
- Seq2Seq 相比旧版（2年数据）MAE 从 5320 降至 3846（-28%），10年数据效果显著
- 三个模型 Suitability Recall 均 ≥ 0.93，满足预警安全约束（≥ 0.80）

---

## 7. MLOps 与工程化规范

### 7.1 中央调度流水线

`run_pipeline.py` 统一管理训练流程，所有模型默认使用 10 年处理数据：

```bash
python run_pipeline.py --model lstm --features 8 --epochs 120
python run_pipeline.py --model gru --features 8 --epochs 120
python run_pipeline.py --model seq2seq_attention --epochs 120
```

### 7.2 标准化输出结构

```
output/runs/<run_name>/
├── metrics.json              # 完整评估指标（含动态峰值阈值）
├── metrics.csv
├── *_test_predictions.csv    # 测试集预测结果
├── figures/
│   ├── true_vs_pred.png
│   ├── confusion_matrix_crowd_alert.png
│   ├── reliability_diagram.png
│   ├── suitability_warning_timeline.png
│   ├── attention_heatmap_mean.png    # Seq2Seq 专属：均值注意力热力图
│   └── attention_heatmap_sample.png  # Seq2Seq 专属：样本注意力热力图
└── weights/
    └── *.h5 / *.keras
```

### 7.3 预警定义与风险等级

#### 动态峰值阈值（数据驱动）

峰值阈值不再硬编码，而是从**训练集**动态计算：

```python
peak_threshold = Quantile(train_visitor_counts, q=0.75)
```

- 仅使用训练集数据，避免测试集泄露
- 默认取 75 分位数，与景区"高峰期"定义对齐
- 合理性约束：阈值限制在 [5000, 80000] 范围内
- 每次重训练自动更新，反映数据分布变化

#### 拥挤预警 (crowd_alert)

- `crowd_alert = 1` 当 `visitor_count ≥ peak_threshold`

四级风险映射（基于阈值 T）：
- **绿色**：`< 0.70·T`
- **黄色**：`0.70·T ~ 0.85·T`
- **橙色**：`0.85·T ~ 1.00·T`
- **红色**：`≥ 1.00·T`

#### 天气危险 (weather_hazard)

基于训练集历史分位数（Route B，防止数据泄露）：
- `P90 = Quantile(train_precip, 0.90)`
- `TH90 = Quantile(train_temp_high, 0.90)`
- `TL10 = Quantile(train_temp_low, 0.10)`
- `weather_hazard = 1` 当任一条件触发

#### 综合适宜性预警 (suitability_warning)

`suitability_warning = crowd_alert OR weather_hazard`

> **校准说明**：当前预警概率通过 logistic 变换（temperature=1000）从点预测转换而来，本质上是确定性分类器的近似，而非真正的概率输出。Brier Score 和 ECE 用于衡量这一近似的质量，但不应与真正概率模型的校准指标直接比较。

#### 模型选择策略（预警优先）

1. **硬约束**：`Recall_warning ≥ 0.80`（漏报代价远高于误报）
2. **主指标**：最大化 `suitability_warning_bin` 的加权 F1
3. **平局决胜**：更低 Brier Score → 更低 ECE

### 7.4 定时自动爬取

Web 应用集成 APScheduler，每日 09:00（Asia/Shanghai）自动执行爬取任务：

```bash
# 查看调度状态
GET /api/scheduler/status
```

---

## 8. Web 应用架构

### 后端 API

| 端点 | 说明 |
|------|------|
| `GET /api/forecast` | 返回三模型预测、历史数据、天气、风险 |
| `GET /api/models` | 返回冠军/亚军/第三模型元数据 |
| `GET /api/metrics` | 返回指定模型的 metrics.json |
| `GET /api/scheduler/status` | 返回定时任务状态 |

### Dashboard v2（推荐）

访问：`http://localhost:5000/dashboard/v2`

**功能**：
- 三模型预测对比（Champion / Runner-up / Third，可单独切换）
- 预测窗口：1 / 3 / 7 天
- 风险温度计（实时风险评分 0~100）
- 天气卡片（点击图表任意日期查看详情）
- 节假日区域高亮
- 中英文切换 / 深色模式

---

## 9. 数据库设计

SQLite（`jiuzhaigou_fyp.db`），表名 `traffic_records`：

| 字段 | 类型 | 描述 |
|------|------|------|
| `id` | Integer (PK) | 自增主键 |
| `record_date` | Date (Unique) | 记录日期 |
| `actual_visitor` | Integer | 真实游客人数 |
| `predicted_visitor` | Integer | 模型预测人数 |
| `is_forecast` | Boolean | 0=历史验证，1=未来预测 |

---

## 10. 数据同步机制

`scripts/sync_to_cloud.py`：扫描 `data/processed/` 和 `output/runs/` 最新结果，重建 `traffic_records` 表并写入历史数据与预测结果。

---

## 11. 项目成果与技术亮点

### 数据工程
- 10 年历史数据（2016-2026），2719 条日级记录
- Open-Meteo API 自动补全缺失天气数据
- 8 特征工程（时间编码 + 节假日 + 滞后特征 + 天气）

### 模型工程
- 三模型并行（GRU / LSTM / Seq2Seq+Attention）
- 非自回归 Seq2Seq 架构，杜绝自回归数据泄露
- 自定义非对称损失函数，强化节假日高峰预测
- 动态峰值阈值（训练集分位数，每次重训自动更新）
- Attention 权重热力图可视化（模型可解释性）

### 评估工程
- Walk-forward Expanding Window 评估（4 折，每折 90 天）
- 多维度指标：回归（MAE/RMSE/SMAPE）+ 分类（F1/Recall/Precision）+ 校准（Brier/ECE）
- 天气危险阈值仅从训练集计算（Route B，防止数据泄露）
- 预警优先的模型选择策略（Recall ≥ 0.80 硬约束）

### 系统工程
- Flask + APScheduler 每日自动爬取
- 三模型对比 Dashboard（ECharts 交互式可视化）
- Seq2Seq `.keras` 格式自定义类注册（解决加载失败问题）

---

## 12. 已实现的方法论改进

### Walk-forward Evaluation（已实现）

**位置**：`scripts/walk_forward_eval.py`

**方法**：Expanding Window 策略，模拟真实部署场景：
- 每个 fold 的训练集从数据起点扩展到切割点
- 测试集为切割点后的固定窗口（默认 90 天）
- 每个 fold 独立训练，记录回归指标与预警指标
- 汇总跨 fold 的均值与标准差

**与固定切分的区别**：固定切分只在最后 10% 测试，无法反映模型在不同季节的稳定性。Walk-forward 覆盖多个时间段（包括春节、五一、国庆等高峰期），评估结果更贴近真实部署。

```bash
python scripts/walk_forward_eval.py --model gru --folds 4 --test-window 90
```

输出：`output/walk_forward/<model>_<timestamp>/`
- `walk_forward_folds.csv`：每折详细指标
- `walk_forward_summary.json`：跨折均值与标准差
- `walk_forward_metrics.png`：指标趋势图

### Attention 权重可视化（已实现）

**位置**：`models/lstm/train_seq2seq_attention_8features.py`，`_save_attention_heatmap()`

每次 Seq2Seq 训练完成后自动生成：
- **均值热力图**（`attention_heatmap_mean.png`）：所有测试样本的平均注意力分布，反映模型整体上"最关注"哪些历史时间位置
- **样本热力图**（`attention_heatmap_sample.png`）：最新预测窗口的注意力分布
- **原始权重**（`attention_weights_mean.npy`）：供后续分析

热力图 X 轴为 Encoder 历史天（-30 ~ -1），Y 轴为 Decoder 预测步（Day+1 ~ Day+7），颜色越深表示该历史天对该预测步影响越大。

### 动态峰值阈值（已实现）

**位置**：`models/common/core_evaluation.py`，`compute_dynamic_peak_threshold()`

峰值阈值从训练集 75 分位数动态计算，替代原来的硬编码常量 18500：
- 仅使用训练集数据，避免测试集泄露
- 每次重训练自动更新，反映数据分布变化
- 合理性约束：限制在 [5000, 80000] 范围内

### 三模型对比 Dashboard（已实现）

前端支持 Champion / Runner-up / Third 三模型独立切换与全量对比，后端 `/api/forecast` 同时返回三条预测序列及各自的风险评估。

### Seq2Seq 模型加载修复（已实现）

`web_app/app.py` 在加载 `.keras` 格式时注册 `AttentionLayer`、`Seq2SeqWithAttention`、`create_custom_asymmetric_loss` 自定义类，解决 `ValueError: Unknown layer` 错误。

### 待实现

- **增量学习**：目前为离线训练，新数据仅用于评估。计划实现每月滑动窗口重训练（用最近 2 年数据重训，自动触发 `run_pipeline.py`）
- **SHAP 可解释性**：集成 SHAP 工具，对 GRU/LSTM 生成特征重要性图
- **Ablation Study（系统性）**：逐特征消融（去掉天气特征、去掉 lag_7 等），量化各特征对预警 F1 的贡献

---

## 13. 项目维护信息

**技术栈**：Python 3.10+, TensorFlow 2.15, Flask 3.0, ECharts, SQLite, APScheduler  
**项目状态**：持续开发中  
**最后更新**：2026年4月3日
