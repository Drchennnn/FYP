# 九寨沟景区客流动态预测系统

本项目是一个端到端的客流预测解决方案，集成了数据爬取、特征工程、深度学习模型训练、自动化数据管道以及基于 Web 的可视化展示平台。系统利用 10 年历史客流与气象数据，通过三种深度学习模型（GRU、LSTM、Seq2Seq+Attention）预测未来九寨沟景区的游客接待量，并提供多维度预警机制，辅助景区管理和游客出行决策。

> **最新更新（2026-04-05）**：确立五模型预测架构（单步离线回测 + 多步/Seq2Seq 实时在线推理），废弃在线/离线切换开关，Dashboard 始终展示最新预测结果。详见[预测展示逻辑](#预测展示逻辑)。

---

## 每日数据更新

每天运行一次以下命令，将昨日官网客流数据追加到训练集并刷新预测：

```bash
python scripts/append_and_retrain.py --append
```

执行后效果：
- **单步模型**（GRU/LSTM）：backfill 向后延伸一天，历史回测段右端 +1
- **多步模型**（MIMO/Seq2Seq）：下次页面加载时自动以最新 30 天为输入重新推理，预测窗口自动后移
- **天气数据**：由 Open-Meteo 自动拉取，无需手动操作

如需重训模型权重（建议每月一次）：

```bash
python scripts/append_and_retrain.py --retrain
```

---

## 📚 目录

1. [项目整体架构](#1-项目整体架构)
2. [快速开始](#2-快速开始)
3. [数据爬取子系统](#3-数据爬取子系统)
4. [数据预处理与特征工程](#4-数据预处理与特征工程)
5. [模型架构](#5-模型架构)
6. [评估体系](#6-评估体系)
7. [预警定义与风险等级](#7-预警定义与风险等级)
8. [最新性能指标](#8-最新性能指标)
9. [自动化数据管道](#9-自动化数据管道)
10. [可解释性分析](#10-可解释性分析)
11. [Web 应用架构](#11-web-应用架构)
12. [MLOps 工程规范](#12-mlops-工程规范)
13. [待完成事项](#13-待完成事项)
14. [项目维护信息](#14-项目维护信息)

---

## 1. 项目整体架构

```
数据层                    模型层                    应用层
─────────────────         ─────────────────         ─────────────────
爬虫（客流+天气）  →      特征工程（8特征）  →      Flask API
Open-Meteo API    →      GRU / LSTM / Seq2Seq →    Dashboard v3
SQLite 数据库     ←      评估+预警系统      →      APScheduler
CSV 训练数据      ←      自动追加+月度重训          在线预测引擎
```

### 目录结构

```text
FYP/
├── data/
│   ├── raw/                    # 原始爬取数据
│   ├── processed/              # 特征工程后的训练数据（10年）
│   │   └── jiuzhaigou_daily_features_2016-01-01_2026-04-02.csv
│   └── append_log.json         # 每日追加记录
├── models/
│   ├── common/
│   │   ├── core_evaluation.py  # 统一评估（含动态峰值阈值）
│   │   ├── evaluator.py        # 基础评估模块
│   │   └── visualization.py    # 可视化模块
│   ├── lstm/
│   │   ├── train_seq2seq_attention_8features.py  # Seq2Seq+Attention
│   │   └── train_lstm_8features.py
│   ├── gru/
│   │   └── train_gru_8features.py
│   └── scalers/
│       └── feature_scalers.pkl # MinMax 归一化器（训练集拟合）
├── output/
│   ├── runs/                   # 训练输出（仅保留最新3个模型）
│   │   ├── gru_8features_<timestamp>/
│   │   ├── lstm_8features_<timestamp>/
│   │   └── seq2seq_attention_8features_<timestamp>/
│   ├── walk_forward/           # Walk-forward 评估结果
│   ├── ablation/               # Ablation Study 结果
│   └── shap/                   # SHAP 特征重要性结果
├── scripts/
│   ├── walk_forward_eval.py    # Walk-forward 滚动窗口评估
│   ├── ablation_study.py       # 特征消融实验
│   ├── shap_analysis.py        # SHAP 特征重要性分析
│   ├── append_and_retrain.py   # 数据追加 + 月度重训管道
│   ├── backfill_predictions.py # 历史预测回填（补全测试集空缺）
│   ├── online_predict.py       # Seq2Seq 在线推断脚本
│   ├── calibrate_model.py      # 概率校准工具（Platt/Isotonic）
│   ├── analyze_data_split.py   # 数据切分分析
│   └── sync_to_cloud.py        # CSV → SQLite 同步
├── realtime/
│   ├── jiuzhaigou_crawler.py   # 实时爬虫
│   ├── data_fetcher.py         # Open-Meteo 天气获取
│   └── daily_update.py         # 每日更新脚本
├── web_app/
│   ├── app.py                  # Flask 后端（含在线预测+APScheduler）
│   ├── static/
│   │   ├── css/dashboard_v3.css
│   │   └── js/dashboard_v3.js
│   └── templates/
│       └── dashboard_v3.html   # 主 Dashboard（Apple 风格）
├── run_pipeline.py             # 一键训练流水线
└── run_benchmark.py            # 全模型基准测试
```

---

## 2. 快速开始

### 环境准备

```bash
conda activate FYP
pip install -r requirements_all.txt
# 可解释性分析额外依赖
pip install shap
```

### 数据获取

```bash
# 手动触发每日爬取与数据追加
python realtime/daily_update.py
python scripts/append_and_retrain.py --append
```

### 训练模型

```bash
# 使用 10 年数据训练（默认路径已配置）
python run_pipeline.py --model gru --features 8 --epochs 120
python run_pipeline.py --model lstm --features 8 --epochs 120
python run_pipeline.py --model seq2seq_attention --epochs 120
```

### 评估与分析

```bash
# Walk-forward 滚动评估（4折，每折90天）
python scripts/walk_forward_eval.py --model gru --folds 4

# 特征消融实验
python scripts/ablation_study.py --epochs 80

# SHAP 特征重要性
python scripts/shap_analysis.py --model gru

# 全模型基准测试
python run_benchmark.py
```

### 数据管道（手动触发）

```bash
python scripts/append_and_retrain.py --append          # 追加昨日数据
python scripts/append_and_retrain.py --retrain         # 重训三个模型
python scripts/append_and_retrain.py --append --retrain  # 追加+重训
```

### 历史预测回填（首次部署或重训后需要执行）

新模型训练完成后，测试集 CSV 只覆盖训练时的数据窗口。若需要补全 CSV 截止日期之后到今天的曲线，运行：

```bash
python scripts/backfill_predictions.py
```

脚本会用已训练的 GRU / LSTM 模型对处理数据集中尚未覆盖的日期逐日滚动推断，结果自动追加到对应的 `*_test_predictions.csv`。

### 启动 Web 应用

```bash
cd web_app && python app.py
# 访问 http://localhost:5000/dashboard/v3
```

---

## 3. 数据爬取子系统

爬虫逻辑位于 `realtime/` 目录（已整合至主系统）。

- **客流抓取 (`realtime/jiuzhaigou_crawler.py`)**：`requests` + `BeautifulSoup` 从九寨沟官网解析每日入园人数
- **天气获取 (`realtime/data_fetcher.py`)**：从 Open-Meteo API 获取历史天气（温度、降水、风速）及未来 7 天预报
- **每日更新 (`realtime/daily_update.py`)**：集成爬取 + 追加 + 写入 SQLite 的完整日更新流程

**当前数据范围**：2016-05-01 ~ 2026-04-02，约 10 年，2719 条日级记录，存储于：
`data/processed/jiuzhaigou_daily_features_2016-01-01_2026-04-02.csv`

**实时数据获取 (`realtime/data_fetcher.py`)**：
- `get_current_visitor_count()`：获取昨日客流（官网每日更新）
- `get_weather_forecast(days=7)`：从 Open-Meteo 获取未来 7 天天气预报（用于在线预测）

---

## 4. 数据预处理与特征工程

### 8特征版本（当前主力）

| # | 特征名 | 说明 |
|---|--------|------|
| 1 | `visitor_count_scaled` | 归一化历史客流（MinMax，训练集拟合） |
| 2 | `month_norm` | 归一化月份 `(month-1)/11` |
| 3 | `day_of_week_norm` | 归一化星期 `weekday/6` |
| 4 | `is_holiday` | 中国法定节假日标记（`chinese_calendar`） |
| 5 | `tourism_num_lag_7_scaled` | 滞后7天客流（归一化） |
| 6 | `meteo_precip_sum_scaled` | 降水量（归一化） |
| 7 | `temp_high_scaled` | 最高温度（归一化） |
| 8 | `temp_low_scaled` | 最低温度（归一化） |

### 4特征版本（基线对比）

`visitor_count_scaled`、`month_norm`、`day_of_week_norm`、`is_holiday`

---

## 5. 模型架构

系统包含五个模型，分为**单步**和**多步**两类：

### 单步模型（GRU / LSTM）

输入 `(30, 8)` → 输出标量（下一天），滚动推理产生多天预测，误差随步数累积。

| 模型 | 位置 | 架构 |
|------|------|------|
| GRU | `models/gru/train_gru_8features.py` | `GRU(128)→Dropout→GRU(64)→Dropout→Dense(32)→Dense(1)` |
| LSTM | `models/lstm/train_lstm_8features.py` | `LSTM(128)→Dropout→LSTM(64)→Dropout→Dense(32)→Dense(1)` |

- Huber Loss，Adam(lr=1e-3)，EarlyStopping(patience=15)

### 多步模型（GRU-MIMO / LSTM-MIMO）

输入 `(30, 8)` → 输出 `(7,)`，一次性预测未来7天，无误差积累。

| 模型 | 位置 | 架构 |
|------|------|------|
| GRU-MIMO | `models/gru/train_gru_mimo_8features.py` | `GRU(128)→Dropout→GRU(64)→Dropout→Dense(32)→Dense(7)` |
| LSTM-MIMO | `models/lstm/train_lstm_mimo_8features.py` | `LSTM(128)→Dropout→LSTM(64)→Dropout→Dense(32)→Dense(7)` |

### Seq2Seq + Attention（多步，可解释性最强）

**位置**：`models/lstm/train_seq2seq_attention_8features.py`

**架构**：
- **Encoder**：双向 LSTM（128 单元）+ 1D CNN 特征压缩（8→128 维），编码 30 步历史
- **Decoder**：单向 LSTM（256 单元）+ Bahdanau 注意力，直接输出未来 7 天
- **Decoder 输入**：纯外部特征（7维，不含 visitor_count），杜绝自回归数据泄露
- **输出**：`(batch, 7, 1)` 未来7天预测值

**自定义非对称损失函数**：
- 节假日预测偏低：惩罚权重 ×2.0
- 非节假日高客流预测偏低：惩罚权重 ×1.5

**Attention 可视化**：训练完成后自动生成：
- `figures/attention_heatmap_mean.png`：测试集均值热力图（X轴=历史天 -30~-1，Y轴=预测步 Day+1~Day+7）
- `figures/attention_heatmap_sample.png`：最新预测窗口样本热力图
- `figures/attention_weights_mean.npy`：原始权重矩阵

---

## 6. 评估体系

### 6.1 固定切分评估（标准基准）

训练/验证/测试 = 80% / 10% / 10%，测试集固定为最后 268 天。

**指标**：
- 回归：MAE、RMSE、SMAPE
- 分类：Crowd Alert F1/Recall/Precision
- 预警：Suitability Warning F1/Recall/Brier/ECE
- 多步（Seq2Seq）：按 horizon 加权汇总

### 6.2 Walk-forward Evaluation（时序泛化评估）

**位置**：`scripts/walk_forward_eval.py`

**方法**：Expanding Window 策略，模拟真实部署场景：

```
Fold 1: train=[0, N-4×90)  test=[N-4×90, N-3×90)
Fold 2: train=[0, N-3×90)  test=[N-3×90, N-2×90)
Fold 3: train=[0, N-2×90)  test=[N-2×90, N-1×90)
Fold 4: train=[0, N-1×90)  test=[N-1×90, N)
```

每个 fold 独立训练，覆盖不同季节（包括春节、五一、国庆等高峰期），输出跨折均值与标准差。

```bash
python scripts/walk_forward_eval.py --model gru --folds 4 --test-window 90
python scripts/walk_forward_eval.py --model all --folds 4
```

**输出**：`output/walk_forward/<model>_<timestamp>/`
- `walk_forward_folds.csv`：每折详细指标
- `walk_forward_summary.json`：跨折均值 ± 标准差
- `walk_forward_metrics.png`：MAE / SMAPE / Suitability F1 / Recall 趋势图

**与固定切分的区别**：固定切分只在最后 10% 测试，无法反映模型在不同季节的稳定性。Walk-forward 是时序模型泛化能力的更可靠估计。

### 6.3 Ablation Study（特征贡献量化）

**位置**：`scripts/ablation_study.py`

6个消融方案，在 GRU 上训练对比，量化各特征对预警 F1 和回归误差的贡献：

| 方案 | 特征组合 | 目的 |
|------|---------|------|
| `baseline_4feat` | visitor, month, dow, holiday | 基线参考 |
| `no_weather` | 去掉 precip, temp_high, temp_low | 量化天气特征价值 |
| `no_lag7` | 去掉 tourism_num_lag_7_scaled | 量化滞后特征价值 |
| `no_holiday` | 去掉 is_holiday | 量化节假日特征价值 |
| `no_weather_lag7` | 仅时间特征 | 量化外部特征整体价值 |
| `full_8feat` | 完整8特征 | 对照组 |

```bash
python scripts/ablation_study.py --epochs 80
```

**输出**：`output/ablation/<timestamp>/`
- `ablation_results.csv`：各方案完整指标
- `ablation_study.png`：MAE / Suitability F1 / Recall 对比柱状图

---

## 7. 预警定义与风险等级

### 动态峰值阈值

**位置**：`models/common/core_evaluation.py`，`compute_dynamic_peak_threshold()`

```python
peak_threshold = Quantile(train_visitor_counts, q=0.75)
```

- 仅使用训练集数据，避免测试集泄露
- 合理性约束：限制在 [5000, 80000] 范围内
- 每次重训练自动更新，反映数据分布变化

### 拥挤预警 (crowd_alert)

`crowd_alert = 1` 当 `visitor_count ≥ peak_threshold`

四级风险映射（T = peak_threshold）：

| 等级 | 条件 | 颜色 |
|------|------|------|
| 低风险 | `< 0.70·T` | 绿色 |
| 中风险 | `0.70·T ~ 0.85·T` | 黄色 |
| 高风险 | `0.85·T ~ 1.00·T` | 橙色 |
| 极高风险 | `≥ 1.00·T` | 红色 |

### 天气危险 (weather_hazard)

基于训练集历史分位数（Route B，防止数据泄露）：
- `P90 = Quantile(train_precip, 0.90)`
- `TH90 = Quantile(train_temp_high, 0.90)`
- `TL10 = Quantile(train_temp_low, 0.10)`
- `weather_hazard = 1` 当任一条件触发

### 综合适宜性预警 (suitability_warning)

`suitability_warning = crowd_alert OR weather_hazard`

> **校准说明**：预警概率通过 logistic 变换（temperature=1000）从点预测转换而来，本质上是确定性分类器的近似。Brier Score 和 ECE 用于衡量近似质量，不应与真正概率模型的校准指标直接比较。

---

## 8. 最新性能指标

### 2026年4月3日（10年数据训练）

| 模型 | MAE | RMSE | SMAPE | Suitability F1 | Suitability Recall | Brier |
|------|-----|------|-------|----------------|--------------------|-------|
| **GRU (8特征)** | **2,809** | **3,993** | **15.0%** | 0.971 | 0.964 | 0.036 |
| LSTM (8特征) | 3,881 | 5,104 | 20.3% | 0.966 | 0.945 | 0.043 |
| Seq2Seq+Attention (8特征) | 3,846 | 5,326 | 21.0% | 0.964 | 0.952 | 0.042 |

> 训练集 2016-2026（2688 样本），测试集 268 天，峰值阈值由训练集 75 分位数动态计算。

**关键发现**：
- GRU 回归精度最佳（MAE 最低），适合单日精确预测
- Seq2Seq 相比旧版（2年数据）MAE 从 5320 降至 3846（-28%），10年数据效果显著
- 三个模型 Suitability Recall 均 ≥ 0.93，满足预警安全约束（≥ 0.80）
- Seq2Seq 提供 7 天多步预测，适合中期规划

---

## 9. 自动化数据管道

### 9.1 每日数据追加

**位置**：`scripts/append_and_retrain.py`，`append_new_data()`

**自动触发**：每日 08:30（APScheduler，Asia/Shanghai）

**流程**：
1. 从爬虫获取昨日客流数据
2. 从 Open-Meteo 历史存档获取对应天气（温度、降水）
3. 追加到 `data/processed/jiuzhaigou_daily_features_2016-01-01_2026-04-02.csv`
4. 重新计算 scaled 特征（MinMax，基于全量数据）
5. 记录到 `data/append_log.json`

### 9.2 月度重训

**自动触发**：每月1日 02:00（APScheduler）

**流程**：依次调用 `run_pipeline.py` 重训 GRU、LSTM、Seq2Seq，新模型自动进入 `output/runs/`，Dashboard 下次加载时使用最新版本。

### 9.3 手动操作

```bash
# 追加指定日期数据
python scripts/append_and_retrain.py --append --date 2026-04-03

# 仅重训（不追加）
python scripts/append_and_retrain.py --retrain --epochs 120

# 追加+重训
python scripts/append_and_retrain.py --append --retrain

# 预览（不写入）
python scripts/append_and_retrain.py --append --dry-run
```

### 9.4 每日爬取（独立任务）

**自动触发**：每日 09:00（APScheduler）

```bash
# 手动触发
python realtime/daily_update.py
```

---

## 10. 可解释性分析

### 10.1 Attention 权重热力图（Seq2Seq 专属）

**位置**：`models/lstm/train_seq2seq_attention_8features.py`，`_save_attention_heatmap()`

每次 Seq2Seq 训练完成后自动生成，无需额外操作：

- `figures/attention_heatmap_mean.png`：测试集所有样本的均值注意力分布
- `figures/attention_heatmap_sample.png`：最新预测窗口的注意力分布

**解读**：X 轴为 Encoder 历史天（-30~-1），Y 轴为 Decoder 预测步（Day+1~Day+7），颜色越深表示该历史天对该预测步影响越大。通常可观察到模型对 lag-7（7天前同期）和近期数据的高度关注。

### 10.2 SHAP 特征重要性（GRU / LSTM）

**位置**：`scripts/shap_analysis.py`

使用 `shap.GradientExplainer` 计算每个特征对预测结果的贡献：

```bash
# 安装依赖
pip install shap

# 分析 GRU 模型
python scripts/shap_analysis.py --model gru

# 同时分析 GRU 和 LSTM
python scripts/shap_analysis.py --model all

# 自定义参数
python scripts/shap_analysis.py --model gru --n-background 100 --n-explain 200
```

**输出**：`output/shap/<timestamp>/`
- `shap_summary_bar_<model>.png`：全局特征重要性柱状图（均值绝对 SHAP 值）
- `shap_importance_<model>.csv`：特征重要性排名表
- `shap_values_<model>.npy`：原始 SHAP 值矩阵（供进一步分析）

**参数说明**：
- `--n-background`：背景样本数（默认100，越大越准确但越慢）
- `--n-explain`：解释样本数（默认200）

---

## 11. Web 应用架构

### 11.1 后端 API

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/forecast` | GET | 五模型预测、历史数据、天气、风险 |
| `/api/metrics` | GET | 指定模型的 metrics.json |
| `/api/scheduler/status` | GET | 定时任务状态（爬取/追加/重训） |

**`/api/forecast` 参数**：
- `h`：展示窗口长度（默认7，影响图表缩放，不影响模型推理步数）

---

### 预测展示逻辑

<a name="预测展示逻辑"></a>

Dashboard 展示五条预测曲线，根据模型类型采用不同策略：

#### 单步模型（GRU 单步 / LSTM 单步）

- **展示**：在线推理，从 `last_real_date+1` 滚动到 `today+6`，但**只展示 today 起的7天**
- **数据来源**：页面加载时实时滚动推理（误差随步数累积）
- **空白段**：测试集截止日（如4月2日）到当天之间**留白**，这是官网数据滞后 1-2 天的正常现象
- **中间过渡步**：`last_real_date+1` 到 `today-1` 的推理结果作为滚动窗口的中间输入，不展示

#### 多步模型（GRU MIMO / LSTM MIMO / Seq2Seq+Attention）

- **展示**：实时在线推理，输出直接映射到 today~today+6（7天）
- **数据来源**：页面加载时自动调用模型权重实时推理
- **输入窗口**：最近 30 天有真实访客记录的历史数据（encoder 输入与 today 无关，始终用 last_real_date 前30天）
- **输出**：`today` 到 `today + 6`（7天，直接对应当天起的预测）

**具体示例（假设今天是4月5日，官网数据最新到4月2日）**：

```
最新真实数据：4月2日
数据空缺段：4月3日、4月4日（官网滞后，无真实访客数）

单步 GRU/LSTM：
  历史回测 ████████████████ [4/2 结束]
  空白段    ░░░░░░░░░░░░░░░░ [4/3 ~ 4/4，官网数据未到，不展示]
  在线预测  ████████████████ [4/5 ~ 4/11]（内部从4/3开始滚动，4/5前的步骤不展示）

MIMO GRU/LSTM / Seq2Seq：
  输入：[3/4 ~ 4/2] 30天真实历史（visitor_count + 天气特征）
  输出：[4/5, 4/6, 4/7, 4/8, 4/9, 4/10, 4/11] 直接映射到今天起7天
        ↑ encoder 输入不变，输出日期标签对齐 today
```

**论文意义**：单步留白 vs 多步连续的对比，直接展示了 MIMO 在实际部署中相对于滚动单步的工程优势——不依赖未来访客数据，不受数据滞后影响，可连续输出无断层。

#### 数据管道（每日自动）

```
08:30  抓取昨天官网数据，append 到处理后 CSV
09:00  重新运行单步模型 backfill inference（不重训）
月初   重训所有模型权重
```

### 11.2 Dashboard v3（当前主版本）

访问：`http://localhost:5000/dashboard/v3`

**设计**：Apple 风格三栏布局（ECharts + CSS Grid）

**功能**：
- 五模型预测曲线（GRU单步 / GRU多步 / LSTM单步 / LSTM多步 / Seq2Seq+Attention），独立 chip 切换
- 始终显示在线预测结果（MIMO/Seq2Seq 实时推理，单步只读历史 CSV）
- 顶部天气预报横向滚动条（未来14天）
- 右侧面板：天气卡片 + 适宜性风险温度计 + 推荐出行窗口
- 测试集开始竖线标注
- 节假日区域高亮
- 中英文切换 / 深色模式

---

## 12. MLOps 工程规范

### 12.1 训练流水线

```bash
python run_pipeline.py --model [lstm|gru|seq2seq_attention] --features 8 --epochs 120
```

所有模型默认使用 10 年处理数据（`data/processed/jiuzhaigou_daily_features_2016-01-01_2026-04-02.csv`）。

### 12.2 标准化输出结构

```
output/runs/<run_name>/
├── metrics.json              # 完整评估指标（含动态峰值阈值来源）
├── metrics.csv
├── *_test_predictions.csv
├── figures/
│   ├── true_vs_pred.png
│   ├── confusion_matrix_crowd_alert.png
│   ├── reliability_diagram.png
│   ├── suitability_warning_timeline.png
│   ├── attention_heatmap_mean.png    # Seq2Seq 专属
│   └── attention_heatmap_sample.png  # Seq2Seq 专属
└── weights/
    └── *.h5 / *.keras
```

### 12.3 定时任务总览

| 任务 | 触发时间 | 说明 |
|------|---------|------|
| 每日爬取 | 09:00 | 爬取昨日客流，更新 SQLite |
| 每日追加 | 08:30 | 追加数据到训练 CSV，补充天气 |
| 月度重训 | 每月1日 02:00 | 重训三个模型，更新 output/runs/ |

查看状态：`GET /api/scheduler/status`

---

## 13. 待完成事项

以下内容已有代码框架，但需要实际运行或进一步完善后才能作为最终结果使用。

### 13.1 Ablation Study 实验结果

`scripts/ablation_study.py` 已实现，但尚未实际运行获取图表。需要执行：

```bash
python scripts/ablation_study.py --epochs 80
```

运行后将在 `output/ablation/<timestamp>/` 生成 `ablation_study.png` 和 `ablation_results.csv`，可直接用于论文图表。

### 13.2 SHAP 特征重要性结果

`scripts/shap_analysis.py` 已实现，但尚未实际运行。需要执行：

```bash
pip install shap
python scripts/shap_analysis.py --model all
```

运行后将在 `output/shap/<timestamp>/` 生成特征重要性柱状图。

### 13.3 校准问题（论文说明）

当前预警概率通过 logistic 变换（temperature=1000）从点预测转换而来，本质上是**确定性分类器的概率近似**，而非真正的概率输出。Reliability diagram 呈现双峰分布（大量样本集中在 0 和 1 附近）是这一设计的直接结果。

**论文中需要明确说明**：
- Brier Score 和 ECE 用于衡量这一近似的质量，不应与真正概率模型（如 MC Dropout、Conformal Prediction）的校准指标直接比较
- 如需真正的概率输出，可考虑 Platt Scaling 或 Isotonic Regression 后校准（`scripts/calibrate_model.py` 有框架）

---

## 14. 项目维护信息

**技术栈**：Python 3.10+, TensorFlow 2.15, Flask 3.0, ECharts 5, SQLite, APScheduler, SHAP

**依赖安装**：
```bash
pip install -r requirements_all.txt
pip install APScheduler==3.10.4  # 自动调度
pip install shap                  # 可解释性分析（可选）
```

**最后更新**：2026年4月5日

### 更新日志

| 日期 | 内容 |
|------|------|
| 2026-04-05 | 确立五模型架构（单步 GRU/LSTM + 多步 MIMO + Seq2Seq），预测窗口对齐今天，前端天气直连 Open-Meteo，废弃在线/离线切换开关 |
| 2026-04-03 | Dashboard v3 发布（Apple 风格），完成10年数据训练（GRU/LSTM/MIMO/Seq2Seq） |
