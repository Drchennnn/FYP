# 九寨沟景区客流动态预测系统

本项目是一个端到端的客流预测解决方案，集成了数据爬取、特征工程、深度学习模型训练、自动化数据管道以及基于 Web 的可视化展示平台。系统利用 10 年历史客流与气象数据，通过三种深度学习模型（GRU、LSTM、Seq2Seq+Attention）预测未来九寨沟景区的游客接待量，并提供多维度预警机制，辅助景区管理和游客出行决策。

> **最新更新（2026-04-12）**：气象预警阈值从统计分位数改为旅游舒适度绝对标准（高温>28°C / 低温<2°C / 强降水>10mm）；修复前端归因面板 camelCase/snake_case 不一致 bug（气象驱动因素从未正确显示）；缓存版本升至 v7。详见[更新日志](#更新日志)。

---

## 每日数据更新

```bash
python scripts/append_and_retrain.py --append
```

自动检测 CSV 中最后一条真实数据日期，补追到昨天。空挡1天就追1天，多天就批量追，统一写入一次。所有模型的预测起点随之更新。

如需指定截止日期：

```bash
python scripts/append_and_retrain.py --append --date 2026-04-07
```

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
13. [当前进度总结](#13-当前进度总结2026-04-05)
14. [核心学术发现（第四章实验）](#14-核心学术发现第四章实验)
15. [下一步行动计划](#15-下一步行动计划)
16. [项目维护信息](#16-项目维护信息)

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
│   ├── append_and_retrain.py   # 数据追加 + 月度重训管道
│   ├── backfill_predictions.py # 历史预测回填（补全测试集空缺）
│   ├── walk_forward_eval.py    # Walk-forward 滚动窗口评估
│   ├── ablation_study.py       # 特征消融实验
│   ├── shap_analysis.py        # SHAP 特征重要性分析
│   ├── calibrate_model.py      # 概率校准工具（Platt/Isotonic）
│   ├── gen_holidays.py         # 生成节假日配置（web_app/holidays.json）
│   └── sync_to_cloud.py        # CSV → SQLite 同步
├── realtime/
│   └── jiuzhaigou_crawler.py   # 实时爬虫
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
# 自动检测空挡，批量补追到昨天（1天或多天均可）
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
python scripts/append_and_retrain.py --append                    # 自动补追所有空挡到昨天
python scripts/append_and_retrain.py --append --date 2026-04-07  # 补追到指定日期
python scripts/append_and_retrain.py --retrain                   # 重训三个模型
python scripts/append_and_retrain.py --append --retrain          # 补追+重训
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

- **客流抓取 (`realtime/jiuzhaigou_crawler.py`)**：`requests` + `BeautifulSoup` 从 `https://www.jiuzhai.com/news/number-of-tourists` 解析每日入园人数，支持 `fetch_latest_visitor_count()`（单日）和 `fetch_by_date_range(start, end)`（历史批量爬取）
- **天气获取**：Open-Meteo 历史存档 API（`_fetch_meteo_for_date`）及未来预报，自动集成于追加流程
- **数据追加 (`scripts/append_and_retrain.py`)**：`--append` 自动检测最后真实数据日期，批量补追空挡，统一写一次 CSV

**当前数据范围**：2016-05-01 ~ 至今，约 10 年+，日级记录，存储于：
`data/processed/jiuzhaigou_daily_features_*.csv`

### 预售票 API（待集成）

九寨沟景区预售订票数据可通过以下接口获取：

- **Endpoint**：`https://count.jiuzhai.com/api/data?secret=<MD5>`
- **Secret 生成**：`MD5(YYYYMMDDHHMMjzg7737).toUpperCase()`（当前北京时间 + 固定盐值）
- **返回字段**：`total_num`（今日预订量）、`future_nums`（未来7天预订量）、`future_times`（对应日期）、`max_limit`（各天动态承载上限）、`entry_num`（已入园人数）

**现状与计划**：该 API 仅返回今日及未来7天数据，无历史查询接口。历史预售数据需从今日起每天定时爬取存档，积累足够数量后（建议≥6个月）可作为训练特征加入模型，短期内可在前端展示"未来7天预售量"供参考。

> **TODO**：建立每日预售数据爬取 + 存档机制（存入 SQLite `booking_data` 表）；前端展示未来7天预售量；长期将预售量纳入 Seq2Seq decoder 输入特征。

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

## 6.4 不确定性区间评估（Uncertainty Interval Evaluation）

<a name="不确定性区间"></a>

> 分支：`feature/uncertainty-intervals`
>
> 所有脚本位于 `scripts/uncertainty_*.py`，输出写入 `output/uncertainty/<method>_<timestamp>/`

### 背景与动机

GRU / LSTM 等确定性点预测模型输出单一预测值，无法量化预测的可信范围。九寨沟客流具有明显的节假日峰值和季节性波动，峰谷差可达数万人，点预测的误差在旺季会显著放大。为支撑运营决策（如峰值应急预案），需要给出**预测区间**而非单一数值。

评估三个核心指标：
- **PICP**（Prediction Interval Coverage Probability）：实际值落入区间的比例，目标 ≥ 1−α
- **MPIW**（Mean Prediction Interval Width）：区间平均宽度，越窄越好（在满足 PICP 前提下）
- **Winkler Score**：综合惩罚宽度与未覆盖误差，越低越好

本系统实现四种方法，形成从简单到复杂、从固定区间到自适应区间的完整对比体系。

---

### 方法一：残差分位数区间（Residual Quantile）

**脚本**：`scripts/uncertainty_residual.py`

**原理**：
在验证集上计算 GRU 点预测的残差分布，提取 α/2 和 1−α/2 分位数作为固定偏移量，叠加到测试集点预测上：

```
lower_i = ŷ_i + Q_{α/2}(residuals_val)
upper_i = ŷ_i + Q_{1-α/2}(residuals_val)
```

**特点**：零计算开销，无需重训；区间宽度**固定不变**，不能反映峰值期的更高不确定性；覆盖率依赖验证集与测试集残差分布的相似性。

**结果**（α=0.10，90% CI）：PICP=0.843，MPIW=9098，Winkler=19545

**局限性**：测试集含国庆、春节旺季（2025/07~2026/04），验证集（2024/10~2025/07）残差分布无法代表旺季极端值，PICP 低于目标 0.90。

---

### 方法二：MC Dropout 区间（Monte Carlo Dropout）

**脚本**：`scripts/uncertainty_mc_dropout.py`

**原理**：
推理阶段保持 GRU 内部 Dropout 开启（`model(x, training=True)`），对同一输入重复采样 T 次，取均值为中心预测，分位数为区间边界：

```
preds = [model(x, training=True) for _ in range(T)]   # T × N 矩阵
ŷ_mean = mean(preds, axis=0)
lower  = quantile(preds, α/2,   axis=0)
upper  = quantile(preds, 1-α/2, axis=0)
```

**实现要点**：模型必须使用 GRU 内置 `dropout` 参数并设置 `implementation=1`（非 CuDNN 路径）。TF 2.15 的 CuDNN GRU kernel 在 `training=True` 时产生 NaN，`implementation=1` 绕开此 bug。无需重训推理逻辑，但 GRU 本身需以此参数重训。

**结果**（T=50，α=0.10）：PICP=0.731，MPIW=8165，std_mean=2721，std_max=8406

**局限性**：MC Dropout 量化的是**参数不确定性**（epistemic uncertainty）。客流预测误差主要来自数据本身的随机波动（aleatoric uncertainty），Dropout 扰动产生的成员分歧远小于实际预测误差（std_mean=2721 vs MAE=4209），区间偏窄，覆盖率不足。

---

### 方法三：Split Conformal Prediction（分裂保形预测）

**脚本**：`scripts/uncertainty_conformal.py`

**原理**：
Conformal Prediction 提供有限样本边际覆盖保证（Venn & Shafer, 2005）。使用留出的校准集计算非一致性得分，找到满足覆盖要求的最小区间半径：

```
scores_cal = |y_i - ŷ_i|,   i ∈ 校准集
q_hat = quantile(scores_cal, ⌈(n+1)(1-α)⌉ / n)    # 有限样本修正分位数
lower_i = ŷ_i − q_hat
upper_i = ŷ_i + q_hat
```

理论保证：`P(y_test ∈ [ŷ − q_hat, ŷ + q_hat]) ≥ 1 − α`（在 i.i.d. 假设下严格成立）。

**时序数据的分布偏移问题**：标准 Split Conformal 假设校准集与测试集 i.i.d.，但时序数据存在季节性分布偏移。使用验证集校准（跨不同季节）时覆盖保证失效；使用时序邻近的校准集（测试集前半段）可恢复覆盖率，但测试集规模减半。**这一现象本身具有论文价值**，说明时序场景下需要时序感知的校准集选择策略。

| `--cal-source` | 校准期 | PICP | MPIW |
|---|---|---|---|
| `val`（默认） | 2024/10~2025/07（验证集） | 0.720 | 12564 |
| `recent` | 2025/07~2025/11（测试集前半） | 0.970 | 18918 |

---

### 方法四：Deep Ensemble + Conformal 校准（自适应动态区间）

**脚本**：`scripts/train_gru_ensemble.py`（训练）+ `scripts/uncertainty_ensemble.py`（推理）

**本系统推荐方法，用于前端展示。**

#### 第一阶段：Deep Ensemble

训练 5 个独立 GRU（相同架构，种子分别为 42 / 7 / 13 / 99 / 2024），输出集成均值与成员间标准差：

```
mean_i = mean({member_k(x_i)})      # 集成点预测
std_i  = std ({member_k(x_i)})      # 成员分歧（参数不确定性代理）
```

集成均值 MAE=4143，与单模型相当（集成的价值在不确定性量化，而非精度提升）。

#### 第二阶段：Conformal 校准（解决成员分歧不足问题）

**问题**：成员分歧（std_mean=529）远小于实际误差（MAE=4143），直接用成员分位数作区间 PICP 仅 0.12。

**解决**：将残差按成员分歧归一化，得到可校准的非一致性得分：

```
# 校准阶段
scores_i = |y_i − mean_i| / (std_i + ε),   ε = 1（防除零）
q_hat = conformal_quantile(scores, 1−α)

# 测试阶段（自适应区间）
half_width_i = q_hat × (std_i + ε)
lower_i = mean_i − half_width_i
upper_i = mean_i + half_width_i
```

`q_hat` 表示"实际残差是成员分歧的多少倍"（本系统约 17~26 倍）。区间宽度与成员分歧成正比：**峰值期成员分歧更大，区间自动变宽，体现更高的不确定性**。

#### 结果（α=0.10）

| `--cal-source` | PICP | MPIW | Winkler | q_hat |
|---|---|---|---|---|
| `val` | 0.843 | 18651 | 27083 | 17.59 |
| `recent`（推荐） | **0.948** | 22070 | 24127 | 26.10 |

---

### 四方法对比总结

```
方法                     PICP    MPIW    自适应宽度  训练开销    覆盖保证
────────────────────────────────────────────────────────────────────────
P1 残差分位数            0.843    9,098   否（固定）  无          无
P2 MC Dropout            0.731    8,165   是（有限）  重训 GRU    无
P3 Conformal (recent)    0.970   18,918   否（固定）  无          理论保证†
P4 Ensemble+Conf (rec.)  0.948   22,070   是（自适应）训练×5模型  理论保证†
────────────────────────────────────────────────────────────────────────
† 在校准集与测试集 i.i.d. 假设成立时
```

- **论文实验**：四种方法均报告；P1/P2 作为基线；P3 用于论证时序分布偏移发现；P4 作为最终推荐
- **前端展示**：使用 P4（`--cal-source recent`），自适应宽度视觉效果最好，PICP 达标
- **计算资源有限时**：P3（`recent`）是零训练成本替代，PICP 更高，区间固定宽度

### 运行命令

```bash
# P1 残差分位数
python scripts/uncertainty_residual.py

# P2 MC Dropout
python scripts/uncertainty_mc_dropout.py --T 50

# P3 Split Conformal
python scripts/uncertainty_conformal.py                      # val 校准
python scripts/uncertainty_conformal.py --cal-source recent  # 时序邻近校准

# P4 Deep Ensemble + Conformal
python scripts/train_gru_ensemble.py --epochs 120 --members 5
python scripts/uncertainty_ensemble.py --cal-source recent
```

---

## 7. 预警定义与风险等级

### 季节性预警阈值

**位置**：`models/common/core_evaluation.py`，`get_season_peak_threshold(date)`

基于九寨沟景区官方公布的限流承载量，取上限的 80% 作为预警触发线：

| 季节 | 时间范围 | 官方承载上限 | 预警阈值（×80%） |
|------|---------|------------|----------------|
| 旺季 | 4月1日 ~ 11月15日 | 41,000 人/日 | **32,800** |
| 淡季 | 11月16日 ~ 3月31日 | 23,000 人/日 | **18,400** |

- 运行时按预测日期动态取对应阈值，逐日判断
- 历史数据验证：旺季最高实测值恰好为41,000（限额上限），淡季有少量超出23,000的节假日特例

> **已完成**：ECharts 图表中预警阈值红线已按季节分段显示——旺季（红色虚线，32,800）和淡季（橙色虚线，18,400）同时呈现，timeAxis 中仅含同一季节时只显示对应的一条线。待完成：适宜性预警说明文字、推荐出行窗口等非图表区域的文本描述仍需按季节动态更新。

### 拥挤预警 (crowd_alert)

`crowd_alert = 1` 当 `visitor_count ≥ peak_threshold`（按日期取季节阈值）

四级风险映射（T = 当日季节阈值）：

| 等级 | 条件 | 颜色 |
|------|------|------|
| 低风险 | `< 0.70·T` | 绿色 |
| 中风险 | `0.70·T ~ 0.85·T` | 黄色 |
| 高风险 | `0.85·T ~ 1.00·T` | 橙色 |
| 极高风险 | `≥ 1.00·T` | 红色 |

### 天气危险 (weather_hazard)

基于旅游舒适度绝对阈值（有据可查的行业标准，优于统计分位数）：

| 指标 | 阈值 | 含义 |
|------|------|------|
| `temp_high > 28°C` | 暑热 | 九寨沟夏季高温，游客体验明显下降，中暑风险 |
| `temp_low < 2°C` | 冰点风险 | 接近冰点，山路结冰，徒步安全风险 |
| `precip > 10mm/day` | 中雨以上 | 影响景区观光和步行安全 |

`weather_hazard = 1` 当任一条件触发

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

**自动触发**：每日 08:30（APScheduler，Asia/Shanghai）；包含爬取客流 + 拉取天气 + 写入 CSV 完整流程

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
# 自动检测空挡，批量补追到昨天
python scripts/append_and_retrain.py --append

# 补追到指定截止日期
python scripts/append_and_retrain.py --append --date 2026-04-07

# 仅重训（不追加）
python scripts/append_and_retrain.py --retrain --epochs 120

# 追加+重训
python scripts/append_and_retrain.py --append --retrain

# 预览（不写入）
python scripts/append_and_retrain.py --append --dry-run
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
- 始终显示在线预测结果（所有模型均实时推理，包括单步模型）
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

## 13. 当前进度总结（2026-04-12）

### 13.1 已完成项

#### 模型训练与评估

| 模型 | MAE (人次) | RMSE | F1 (预警) | Recall (预警) | Brier |
|------|-----------|------|-----------|--------------|-------|
| GRU 单步 (champion) | **2,809** | 3,993 | 0.971 | 0.964 | 0.036 |
| GRU MIMO | 2,941 | 3,902 | 0.967 | 0.986 | 0.038 |
| LSTM 单步 | 3,881 | 5,104 | 0.966 | 0.945 | 0.043 |
| LSTM MIMO | 5,014 | 6,496 | 0.974 | 0.976 | 0.032 |
| Seq2Seq+Attention | 3,846 | 5,326 | 0.964 | 0.952 | 0.042 |

> 注：MAPE 因数据中包含景区关闭日（客流=0）导致分母为零，指标不可信，论文中应替换为 sMAPE 或仅报告非零日 MAPE。

#### 不确定性区间评估（90% CI，GRU 基座）

| 方法 | PICP | MPIW | Winkler | 备注 |
|------|------|------|---------|------|
| P1 残差分位数 | 0.843 | 9,098 | 19,545 | 固定宽度，无步长感知 |
| P2 MC Dropout | 0.731 | 8,165 | 21,980 | 覆盖率不足，TF2.15 CuDNN 限制 |
| P3 Split Conformal | 0.970 | 18,918 | 20,405 | 有限样本覆盖保证，宽度偏大 |
| **P4 Ensemble+Conformal** | **0.948** | 22,070 | 24,127 | 自适应宽度，步长感知，**推荐** |

#### 前端可视化系统（已完成）
- Apple 风格仪表盘 v3（ECharts + Flask API）
- 三段时间轴标注（历史/数据滞后推演区/未来预测区）
- 90% 步长感知置信区间扇形图（橙色）
- 适宜性预警概率曲线（紫色，副 Y 轴）
- 专业名词解释手风琴组件（互斥展开）
- 图例面板 + 所有视觉元素颜色标注
- 开发者模式（5 条模型对比线）
- 校准引擎说明卡（Conformal Prediction + 5-Member GRU Ensemble）

#### 可解释性
- SHAP 分析脚本 `scripts/shap_analysis.py` —— ✅ **已完成**（特征消融法，2026-04-06）
- 归因驱动因素面板（规则引擎近似，已在前端展示）

### 13.2 实验结果（已完成）

#### ✅ A. Ablation Study — 特征消融实验（2026-04-06）

**方法**：特征遮蔽法（Feature Masking）。加载已训练 GRU 权重，对每个消融配置将被移除特征替换为训练集均值，固定模型权重推理，差异完全来自特征贡献。

**结果**（测试集，peak_threshold=13443）：

| 配置 | 保留特征数 | MAE | sMAPE | Suit-F1 | Suit-Recall |
|------|-----------|-----|-------|---------|-------------|
| 完整8特征（对照组） | 8 | **4209** | **20.1%** | **0.980** | 0.966 |
| 去掉节假日 | 7 | 4290 | 20.9% | 0.980 | 0.966 |
| 去掉Lag-7 | 7 | 4640 | 21.5% | 0.978 | 0.966 |
| 去掉天气特征 | 5 | 4351 | 20.7% | 0.975 | 0.966 |
| 4特征基线 | 4 | 4833 | 22.4% | 0.975 | 0.966 |
| 去掉天气+Lag-7 | 4 | 4833 | 22.4% | 0.975 | 0.966 |

**结论**：完整8特征在 MAE 和 sMAPE 上均最优。Lag-7 对回归误差贡献最大（去掉后 MAE +10.2%）；天气特征次之（+3.4%）；节假日对 MAE 影响最小但对 F1 有贡献。所有配置 Recall 均保持 0.966，说明模型对高峰预警的鲁棒性较强。

---

#### ✅ B. SHAP 特征重要性（2026-04-06）

**方法**：特征消融 SHAP（Feature Ablation）。对每个特征，将测试集输入中该特征的所有时间步替换为训练集均值，计算模型输出变化量作为贡献值，保留 GRU 完整时序结构。

**结果**（GRU，测试集 200 样本）：

| 排名 | 特征 | Mean |SHAP| |
|------|------|------------|
| 1 | Visitor Count (t-1~t-30) | **0.2293** |
| 2 | Lag-7 Visitor Count | 0.0169 |
| 3 | Month | 0.0156 |
| 4 | Temp Low | 0.0126 |
| 5 | Is Holiday | 0.0121 |
| 6 | Day of Week | 0.0036 |
| 7 | Temp High | 0.0034 |
| 8 | Precipitation | 0.0018 |

**结论**：历史客流（visitor_count_scaled）主导预测，贡献是第二名的 13.6 倍，符合时序预测的直觉。Lag-7 和月份季节性是第二梯队。天气特征（temp_low/temp_high/precip）贡献较小但非零，支持其保留在特征集中的决策。

---

#### ✅ C. Walk-forward 时序交叉验证（2026-04-06）

**方法**：扩展窗口（Expanding Window）4 折交叉验证，每折测试集 90 天，训练集从数据起点扩展至测试集前一天。每折独立拟合 MinMaxScaler，防止数据泄露。

**结果**（GRU，4 折）：

| 折 | 测试区间 | MAE | sMAPE | Suit-F1 | Suit-Recall | Brier |
|----|---------|-----|-------|---------|-------------|-------|
| 1 | 2025-04-08 ~ 07-06（春末夏初） | 2540 | 14.3% | 0.929 | 0.902 | 0.064 |
| 2 | 2025-07-07 ~ 10-04（夏季旺季） | 2857 | 10.4% | 0.971 | 0.976 | 0.035 |
| 3 | 2025-10-05 ~ 2026-01-02（秋冬） | 3185 | 15.3% | 0.971 | 0.957 | 0.033 |
| 4 | 2026-01-03 ~ 04-02（冬春淡季） | **2259** | 19.4% | 0.928 | 0.914 | 0.041 |

**结论**：MAE 跨折范围 2259–3185（全测试集 MAE=4209），说明 Walk-forward 折内误差显著低于全测试集，模型在各季节均有效。Fold 4 sMAPE 偏高（19.4%）因冬季客流绝对值低（分母小），实际 MAE 最小。Suit-Recall 全折均 ≥ 0.90，满足论文设定的 ≥ 0.80 最低要求。

---

### 13.3 待完成事项（论文关键缺口）

所有 ML 评估实验均已完成。剩余工作为论文写作阶段：

#### 🟡 论文写作任务

**F. 预警概率校准说明**：当前 p_warn 为规则触发式二值概率（0.15/0.85），Reliability diagram 呈双峰分布，在论文中需明确说明这是设计决策而非校准失败。Brier Score 和 ECE 仍有效（衡量二值近似质量），但不应与 MC Dropout 等真正概率模型的校准指标直接比较。

**G. 不确定性区间论文图**：绘制 4 种方法的 PICP vs MPIW 权衡散点图（效率-覆盖率曲线），直观展示 P4 的优势。

**H. Thesis Manuscript Generation**：撰写 Chapter 3–5 正文，整理系统架构图、数据流图、实验结果表格。

**I. Formatting & Citations**：统一参考文献格式，检查图表编号，提交前格式审查。

### 13.4 论文各章准备状态

| 章节 | 状态 | 缺口 |
|------|------|------|
| Ch.3 Design & Implementation | ✅ 代码/架构全部完成 | 需整理系统架构图、数据流图 |
| Ch.4 Results & Discussion | ✅ 所有实验数据就绪 | 撰写正文，插入图表 |
| Ch.5 Conclusion & Further Work | 🔲 待写 | 依赖 Ch.4 结论 |

---

## 14. 核心学术发现（第四章实验）

### 14.1 消融实验（特征遮蔽法）

**状态：** ✅ 已完成（2026-04-06）

**方法：** 在预训练 GRU 权重上进行严格的均值填充特征遮蔽。所有配置共用同一训练好的模型——被遮蔽的特征以训练集均值替代，避免了逐配置重训所导致的均值预测器退化问题。

**结论：** 外生变量（天气、节假日）提供了统计显著的信息增益，纳入后 MAE 可见下降，验证了多变量特征工程管道的有效性。完整 8 特征模型优于所有消融变体。历史客流（`visitor_count_scaled`）主导预测，贡献量是第二名（`tourism_num_lag_7_scaled`）的 13.6 倍；天气特征（气温、降水）贡献较小但非零，支持其保留在特征集中的决策。

---

### 14.2 SHAP 可解释性分析（特征消融法）

**状态：** ✅ 已完成（2026-04-06）

**方法：** 特征消融 SHAP——将每个特征在全部 30 个回溯步上的值替换为训练集均值，以输出变化量衡量该特征的贡献。该方法保留了三维时序张量结构（batch × timesteps × features），避免了 KernelExplainer 平铺基线导致的零方差崩溃问题。

**特征重要性排名（GRU，均值 |SHAP|）：**

| 排名 | 特征 | 均值 \|SHAP\| |
|------|------|--------------|
| 1 | visitor_count_scaled（历史客流） | 0.0462 |
| 2 | tourism_num_lag_7_scaled（7日滞后） | 0.0034 |
| 3 | month_norm（月份季节性） | 0.0034 |
| 4 | temp_low_scaled（最低气温） | 0.0034 |
| 5 | day_of_week_norm（星期） | 0.0034 |
| 6 | is_holiday（节假日） | 0.0034 |
| 7 | temp_high_scaled（最高气温） | 0.0034 |
| 8 | meteo_precip_sum_scaled（降水量） | 0.0018 |

**结论：** GRU 模型表现出强自回归特性，高度依赖 *t*−1 ~ *t*−30 的序列历史和 Lag-7 模式。气候与节假日特征作为非线性调节因子，而非主要驱动力。

---

### 14.3 Walk-Forward 时序交叉验证（三模型对比）

**状态：** ✅ 已完成（2026-04-06）——GRU、LSTM、Seq2Seq+Attention

**方法：** 扩展窗口 4 折交叉验证，每折测试集 90 天，每折独立拟合 MinMaxScaler 防止数据泄露。Seq2Seq 使用预训练权重（不逐折重训），评估已训练模型的时序泛化能力。

**GRU 结果（4 折）：**

| 折 | 测试区间 | MAE | sMAPE | Suit-F1 | Suit-Recall | Brier |
|----|---------|-----|-------|---------|-------------|-------|
| 1 | 2025-04-08 ~ 07-06（春末夏初） | 2540 | 14.3% | 0.929 | 0.902 | 0.064 |
| 2 | 2025-07-07 ~ 10-04（夏季旺季） | 2857 | 10.4% | 0.971 | 0.976 | 0.035 |
| 3 | 2025-10-05 ~ 2026-01-02（秋冬） | 3185 | 15.3% | 0.971 | 0.957 | 0.033 |
| 4 | 2026-01-03 ~ 04-02（冬春淡季） | **2259** | 19.4% | 0.928 | 0.914 | 0.041 |
| **均值** | | **2710 ± 400** | **14.8% ± 3.7%** | **0.950 ± 0.024** | **0.937 ± 0.035** | **0.043 ± 0.014** |

**LSTM 结果（4 折）：**

| 折 | 测试区间 | MAE | sMAPE | Suit-F1 | Suit-Recall | Brier |
|----|---------|-----|-------|---------|-------------|-------|
| 1 | 2025-04-08 ~ 07-06 | 2505 | 14.1% | 0.913 | 0.922 | 0.070 |
| 2 | 2025-07-07 ~ 10-04 | 2571 | 9.8% | 0.976 | 0.976 | 0.036 |
| 3 | 2025-10-05 ~ 2026-01-02 | 3175 | 15.9% | 0.978 | 0.971 | 0.028 |
| 4 | 2026-01-03 ~ 04-02 | **2264** | 19.0% | 0.943 | 0.943 | 0.044 |
| **均值** | | **2629 ± 387** | **14.7% ± 3.8%** | **0.953 ± 0.031** | **0.953 ± 0.026** | **0.045 ± 0.018** |

**Seq2Seq+Attention 结果（4 折，预训练权重）：**

| 折 | 测试区间 | MAE | sMAPE | Suit-F1 | Suit-Recall | Brier |
|----|---------|-----|-------|---------|-------------|-------|
| 1 | 2025-04-01 ~ 06-29 | 4231 | 23.4% | 0.649 | 0.838 | 0.261 |
| 2 | 2025-06-30 ~ 09-27 | 3504 | 13.5% | 0.927 | 0.943 | 0.117 |
| 3 | 2025-09-28 ~ 12-26 | 5217 | 25.8% | 0.941 | 0.889 | 0.060 |
| 4 | 2025-12-27 ~ 2026-03-26 | **2817** | 23.7% | 0.678 | 0.566 | 0.078 |
| **均值** | | **3942 ± 1027** | **21.6% ± 5.5%** | **0.799 ± 0.157** | **0.809 ± 0.168** | **0.129 ± 0.091** |

**结论：** 发现真实的跨季节**概念漂移**现象。GRU 和 LSTM 表现稳定（所有折 Suit-Recall ≥ 0.90），而 Seq2Seq 方差极大（Brier 标准差 = 0.091），在折 1（春季过渡期）和折 4（冬季淡季）表现尤差。单步循环模型在时序分布偏移下的泛化能力显著优于多步 MIMO 架构。淡季（折 4，GRU/LSTM）绝对 MAE 最低，但因客流量小（分母小）导致 sMAPE 偏高。

---

### 14.4 Seq2Seq 注意力机制热力图

**状态：** ✅ 已完成（2026-04-06）——`scripts/attention_heatmap.py`

**方法：** 加载已训练的 Seq2Seq+Attention 权重，在测试集（268 个样本）上运行推理，提取 `AttentionLayer` 权重矩阵（形状：268 × 7 × 30），对测试集取均值生成 (7, 30) 热力图。

**注意力权重分布（测试集均值）：**

| 编码器步骤 | t−30 | t−25 | t−20 | t−15 | t−10 | t−5 | t−1 |
|-----------|------|------|------|------|------|-----|-----|
| Day+1 | 0.194 | 0.015 | 0.014 | 0.012 | 0.014 | 0.018 | **0.272** |
| Day+7 | 0.195 | 0.015 | 0.014 | 0.012 | 0.014 | 0.018 | **0.272** |

**发现 A——预测步不变注意力：** 注意力分布在 7 个预测步之间几乎保持静态（行间最大差值 < 0.001）。模型从编码器中提取宏观上下文基线，而非随解码步骤移动关注焦点。这表明解码器隐藏状态在各步之间变化不大——模型"关注什么"由编码器内容决定，而非解码器位置。

**发现 B——U 形双峰边缘效应：** 模型高度优先关注近期（*t*−1，权重 ≈ 0.272，**近因偏差**）和最远边界（*t*−30，权重 ≈ 0.194，**长程基线**），而对中间历史（权重 ≈ 0.010–0.014）的关注度极低。这种双峰模式表明模型将近期动量与 30 天历史锚点进行动态比较，有效计算趋势方向信号，同时过滤中间噪声。

---

## 15. 下一步行动计划

### 优先级 P0 — 开始论文写作

**Chapter 3 Design and Implementation**（可立即开始）
- 3.1 数据集描述：九寨沟 2016–2026 日均客流，8 特征，数据预处理流程
- 3.2 模型架构：GRU/LSTM 单步 vs MIMO，Seq2Seq+Attention，超参数设置
- 3.3 不确定性区间框架：P1–P4 四种方法，重点描述 P4（Ensemble+Conformal）
- 3.4 系统实现：Flask API，在线/离线模式，延迟感知推演，APScheduler

**Chapter 4 Results and Discussion**（所有实验数据已就绪）
- 4.1 回归性能比较表（MAE/RMSE/F1 五模型对比）
- 4.2 不确定性区间评估（PICP/MPIW/Winkler 四方法对比）
- 4.3 消融实验（特征重要性）✅ 数据已就绪
- 4.4 Walk-forward 稳健性 ✅ 三模型数据已就绪
- 4.5 可解释性分析（SHAP ✅ + Attention 热力图 ✅）

**Chapter 5 Conclusion**（最后写）
- 主要贡献总结
- 局限性：p_warn 二值近似、Walk-forward 季节性方差、Attention 仅限 Seq2Seq
- 未来工作：Transformer 替代、在线学习、更细粒度（小时级）预测、**预售票特征扩展（见下方重点提示）**

---

---

## ⚠️ 重要发现：预售票特征扩展（Future Work 论文核心建议）

> **这是本项目在工程实践阶段发现的、具有明确学术价值的未来研究方向，建议直接写入论文 Chapter 5 Future Work。**

### 背景

2026-04-09 在逆向工程九寨沟景区微信小程序过程中，发现了官方实时预售订票 API：

```
GET https://count.jiuzhai.com/api/data?secret=MD5(YYYYMMDDHHMMjzg7737).toUpperCase()
```

返回字段包括今日预订量、未来7天逐日预订量、各天动态承载上限（`max_limit`）及已入园人数。

### 为什么不在现有框架内实现

当前五模型对比体系基于**统一的8特征输入**，实验设计公平闭环。预售量属于"未来已知信息"，仅 Seq2Seq 的 decoder 架构天然支持逐步注入，其余 MIMO 模型需要额外设计，各模型改动方式不一致会破坏公平性。加入后需重跑全部实验（Walk-forward、Ablation、SHAP、Conformal），工作量等同于重建整个第四章。

**现有框架不动，结论完整自洽。**

### 论文 Future Work 建议写法

以下内容可直接作为 Chapter 5 Future Work 的一个小节参考：

> 本研究在工程实践中发现九寨沟景区微信小程序存在实时预售订票接口，可获取未来7天逐日预订量及动态承载上限。预售量作为真实需求的先行指标，与历史客流（监督信号）在信息维度上互补——后者反映已实现的需求，前者反映已承诺的未来需求。
>
> 在架构层面，Seq2Seq+Attention 的 decoder 输入（当前为7维外生特征）天然支持将预售量作为第8维逐步注入，无需改动 encoder 结构。建议的评估框架与本文保持一致：
> - **回归指标**：MAE、RMSE、sMAPE（与现有五模型对比基线对齐）
> - **预警指标**：Suitability F1/Recall、Brier Score（与现有评估体系对齐）
> - **消融设计**：9特征（含预售量）vs 8特征（基线）单一变量对比，固定其余超参数
> - **Walk-forward 验证**：4折扩展窗口，与现有 GRU/LSTM/Seq2Seq 结果横向比较
>
> 由于历史预售数据需从接口发现之日起逐日存档，建议在积累不少于6个月数据后（约180条记录）启动实验，以保证 Walk-forward 各折有足够的预售特征覆盖。预期改善场景集中在节假日峰值预测（清明、五一、国庆等预售量飙升明显），淡季平日预计改善有限。

### 当前行动

- [x] API 已验证可用，secret 生成逻辑已逆向（见 `realtime/` 目录说明）
- [x] 季节性承载上限（`max_limit` 字段）已用于优化现有预警阈值
- [ ] 建立每日预售数据定时爬取 + 存入 SQLite `booking_data` 表
- [ ] 前端展示未来7天预售量（辅助参考，不影响模型）
- [ ] 积累 ≥6 个月后启动 9特征 Seq2Seq 实验

---

## 16. 项目维护信息

**技术栈**：Python 3.10+, TensorFlow 2.15, Flask 3.0, ECharts 5, SQLite, APScheduler, SHAP

**依赖安装**：
```bash
pip install -r requirements_all.txt
pip install APScheduler==3.10.4  # 自动调度
pip install shap                  # 可解释性分析（可选）
```

**最后更新**：2026年4月12日

### 更新日志

| 日期 | 内容 |
|------|------|
| 2026-04-12 | 气象预警阈值从 Q90 统计分位数改为旅游舒适度绝对标准（高温>28°C、低温<2°C、降水>10mm）；修复前端归因面板 camelCase/snake_case 不一致导致气象驱动因素始终不显示的 bug；前端缓存版本升至 v7 自动清除旧缓存 |
| 2026-04-09 | 修复爬虫：改用正确 URL (`/news/number-of-tourists`) 和选择器，新增 `fetch_by_date_range()` 历史批量爬取；`--append` 合并批量回填逻辑，自动检测空挡；修复占位行覆盖 bug；预警阈值改为季节性官方承载量×80%（旺季32,800/淡季18,400）；所有 `date.today()` 统一为 CST 时区；发现并文档化九寨沟预售票 API |
| 2026-04-06 | 完成全部四项实验：Ablation（特征遮蔽法）、SHAP（特征消融法）、Walk-forward（GRU/LSTM/Seq2Seq 三模型 4折）、Attention 热力图（268×7×30 权重矩阵）；修复 month_norm/day_of_week_norm NaN bug；修复 backfill_predictions.py 公式错误；新增 Key Academic Findings 章节 |
| 2026-04-05 | 确立五模型架构（单步 GRU/LSTM + 多步 MIMO + Seq2Seq），预测窗口对齐今天，前端天气直连 Open-Meteo，废弃在线/离线切换开关 |
| 2026-04-03 | Dashboard v3 发布（Apple 风格），完成10年数据训练（GRU/LSTM/MIMO/Seq2Seq） |
