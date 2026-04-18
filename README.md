# 九寨沟景区客流动态预测系统

端到端客流预测解决方案：10年历史数据（2016–2026）→ 三种深度学习模型（GRU、LSTM、Seq2Seq+Attention）→ 多维度预警 → Apple 风格 Web 仪表盘。辅助景区管理与游客出行决策。

> **最新更新（2026-04-12）**：气象预警阈值从统计分位数改为旅游舒适度绝对标准（高温>28°C / 低温<2°C / 强降水>10mm）；修复前端归因面板 camelCase/snake_case 不一致 bug；缓存版本升至 v7。

---

## 目录

1. [项目架构](#1-项目架构)
2. [快速开始](#2-快速开始)
3. [数据子系统](#3-数据子系统)
4. [特征工程](#4-特征工程)
5. [模型架构](#5-模型架构)
6. [评估体系](#6-评估体系)
7. [预警定义与风险等级](#7-预警定义与风险等级)
8. [最新性能指标](#8-最新性能指标)
9. [自动化数据管道](#9-自动化数据管道)
10. [Web 应用架构](#10-web-应用架构)
11. [当前进度与后续计划](#11-当前进度与后续计划)
12. [项目维护信息](#12-项目维护信息)

---

## 1. 项目架构

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
│   ├── processed/              # 特征工程后的训练数据（10年）
│   └── append_log.json         # 每日追加记录
├── models/
│   ├── common/
│   │   ├── core_evaluation.py  # 统一评估（含动态峰值阈值）
│   │   └── preprocess.py       # 特征工程
│   ├── lstm/                   # LSTM + Seq2Seq 训练脚本
│   ├── gru/                    # GRU 训练脚本
│   └── scalers/                # MinMax 归一化器
├── output/
│   └── runs/                   # 训练输出（每模型保留最新一次）
├── scripts/
│   ├── append_and_retrain.py   # 数据追加 + 月度重训管道
│   ├── backfill_predictions.py # 历史预测回填
│   ├── ablation_study.py       # 特征消融实验
│   ├── attention_heatmap.py    # Seq2Seq 注意力热力图
│   ├── calibrate_model.py      # 概率校准工具
│   ├── gen_holidays.py         # 生成节假日配置
│   └── sync_to_cloud.py        # CSV → SQLite 同步
├── realtime/
│   └── jiuzhaigou_crawler.py   # 实时爬虫
├── web_app/
│   ├── app.py                  # Flask 后端（含在线预测 + APScheduler）
│   ├── static/js/dashboard_v3.js
│   └── templates/dashboard_v3.html
├── archive/
│   └── uncertainty_shap_walkforward/  # 已归档的补充实验脚本与结果
├── run_pipeline.py             # 一键训练流水线
└── run_benchmark.py            # 全模型基准测试
```

---

## 2. 快速开始

```bash
# 激活环境
conda activate FYP

# 训练模型
python run_pipeline.py --model gru --features 8 --epochs 120
python run_pipeline.py --model lstm --features 8 --epochs 120
python run_pipeline.py --model seq2seq_attention --epochs 120

# 数据追加（自动检测空挡，补追到昨天）
python scripts/append_and_retrain.py --append

# 启动 Web 应用
cd web_app && python app.py
# 访问 http://localhost:5000/dashboard/v3
```

---

## 3. 数据子系统

- **客流爬取**：`realtime/jiuzhaigou_crawler.py`，从九寨沟官网解析每日入园人数，支持单日及历史批量爬取
- **天气获取**：Open-Meteo 历史存档 API + 未来预报，自动集成于追加流程
- **数据追加**：`--append` 自动检测最后真实数据日期，批量补追空挡，统一写入一次 CSV

**当前数据范围**：2016-05-01 ~ 至今，约 10 年，日级记录  
`data/processed/jiuzhaigou_daily_features_2016-01-01_2026-04-02.csv`

### 预售票 API（待集成）

2026-04-09 逆向工程发现官方实时预售订票接口：

```
GET https://count.jiuzhai.com/api/data?secret=MD5(YYYYMMDDHHMMjzg7737).toUpperCase()
```

返回字段：今日预订量、未来 7 天逐日预订量、动态承载上限（`max_limit`）、已入园人数。  
现状：历史数据需从接口发现日起逐日存档，积累 ≥6 个月后可作为训练特征（见[后续计划](#11-当前进度与后续计划)）。

---

## 4. 特征工程

### 8特征版本（当前主力）

| # | 特征名 | 说明 |
|---|--------|------|
| 1 | `visitor_count_scaled` | 归一化历史客流（MinMax，训练集拟合） |
| 2 | `month_norm` | 归一化月份 `(month-1)/11` |
| 3 | `day_of_week_norm` | 归一化星期 `weekday/6` |
| 4 | `is_holiday` | 中国法定节假日标记 |
| 5 | `tourism_num_lag_7_scaled` | 滞后7天客流（归一化） |
| 6 | `meteo_precip_sum_scaled` | 降水量（归一化） |
| 7 | `temp_high_scaled` | 最高温度（归一化） |
| 8 | `temp_low_scaled` | 最低温度（归一化） |

### 4特征版本（消融基线）

`visitor_count_scaled`、`month_norm`、`day_of_week_norm`、`is_holiday`

---

## 5. 模型架构

### GRU / LSTM（单步滚动）

输入 `(30, 8)` → 输出标量（下一天），滚动推理产生7天预测，误差随步数累积。

| 模型 | 架构 |
|------|------|
| GRU | `GRU(128)→Dropout→GRU(64)→Dropout→Dense(32)→Dense(1)` |
| LSTM | `LSTM(128)→Dropout→LSTM(64)→Dropout→Dense(32)→Dense(1)` |

训练配置：Huber Loss，Adam(lr=1e-3)，EarlyStopping(patience=15)，ReduceLROnPlateau

### Seq2Seq + Attention（多步，可解释性最强）

**位置**：`models/lstm/train_seq2seq_attention_8features.py`

- **Encoder**：双向 LSTM（128 单元）+ 1D CNN 特征压缩，编码 30 步历史
- **Decoder**：单向 LSTM（256 单元）+ Bahdanau 注意力，直接输出未来 7 天
- **Decoder 输入**：纯外部特征（7 维，不含 visitor_count），杜绝自回归数据泄露
- **输出**：`(batch, 7, 1)` 未来 7 天预测值
- **自定义非对称损失**：节假日预测偏低惩罚 ×2.0，非节假日高客流偏低惩罚 ×1.5

---

## 6. 评估体系

训练/验证/测试 = 80% / 10% / 10%，测试集固定为最后 268 天。

**核心指标**：

| 类别 | 指标 |
|------|------|
| 回归精度 | MAE、RMSE、sMAPE、NRMSE |
| 拥挤预警 | Crowd Alert F1 / Recall / Precision |
| 适游性预警 | Suitability Warning F1 / Recall / Brier Score |
| 多步（Seq2Seq） | 按 horizon 加权汇总 |

### 消融实验（特征贡献量化）

**位置**：`scripts/ablation_study.py`

6个消融方案，在 GRU 上对比各特征对预警 F1 和回归误差的贡献：

| 方案 | 特征组合 |
|------|---------|
| `full_8feat`（对照） | 完整 8 特征 |
| `no_weather` | 去掉 precip、temp_high、temp_low |
| `no_lag7` | 去掉 tourism_num_lag_7_scaled |
| `no_holiday` | 去掉 is_holiday |
| `no_weather_lag7` | 仅时间特征 |
| `baseline_4feat` | visitor、month、dow、holiday |

**已完成结果**（2026-04-06）：完整 8 特征在 MAE 和 sMAPE 上均最优；Lag-7 对回归误差贡献最大（去掉后 MAE +10.2%）；天气特征次之（+3.4%）。

```bash
python scripts/ablation_study.py --epochs 80
```

---

## 7. 预警定义与风险等级

### 季节性拥挤预警（crowd_alert）

基于官方限流承载量 ×80%：

| 季节 | 时间范围 | 预警阈值 |
|------|---------|---------|
| 旺季 | 4月1日 ~ 11月15日 | **32,800** 人/日 |
| 淡季 | 11月16日 ~ 3月31日 | **18,400** 人/日 |

四级风险（T = 当日季节阈值）：

| 等级 | 条件 | 颜色 |
|------|------|------|
| 低风险 | `< 0.70·T` | 绿色 |
| 中风险 | `0.70·T ~ 0.85·T` | 黄色 |
| 高风险 | `0.85·T ~ 1.00·T` | 橙色 |
| 极高风险 | `≥ 1.00·T` | 红色 |

### 天气危险（weather_hazard）

| 指标 | 阈值 | 含义 |
|------|------|------|
| `temp_high > 28°C` | 暑热 | 高温，中暑风险 |
| `temp_low < 2°C` | 冰点风险 | 山路结冰，安全风险 |
| `precip > 10mm/day` | 中雨以上 | 影响观光和步行安全 |

### 综合适游性预警（suitability_warning）

`suitability_warning = crowd_alert OR weather_hazard`

---

## 8. 最新性能指标

### 2026-04-03（10年数据训练，测试集 268 天）

| 模型 | MAE | RMSE | sMAPE | Suitability F1 | Suitability Recall | Brier |
|------|-----|------|-------|----------------|--------------------|-------|
| **GRU（champion）** | **2,809** | **3,993** | **15.0%** | 0.971 | 0.964 | 0.036 |
| LSTM | 3,881 | 5,104 | 20.3% | 0.966 | 0.945 | 0.043 |
| Seq2Seq+Attention | 3,846 | 5,326 | 21.0% | 0.964 | 0.952 | 0.042 |

NRMSE（值域归一化）= RMSE / (41,000 − 15)：

| 模型 | NRMSE |
|------|-------|
| GRU（最优 run） | **9.7%** |
| GRU（当前 run） | 13.7% |
| Seq2Seq | 13.0% |

**关键发现**：
- GRU 回归精度最佳，适合单日精确预测
- 三个模型 Suitability Recall 均 ≥ 0.93，满足安全约束（≥ 0.80）
- Seq2Seq 提供 7 天多步预测，适合中期规划

---

## 9. 自动化数据管道

| 任务 | 触发时间 | 说明 |
|------|---------|------|
| 每日追加 | 08:30 | 追加数据到训练 CSV，补充天气 |
| 每日爬取 | 09:00 | 爬取昨日客流，更新 SQLite |
| 月度重训 | 每月1日 02:00 | 重训三个模型，更新 output/runs/ |

```bash
python scripts/append_and_retrain.py --append                    # 自动补追到昨天
python scripts/append_and_retrain.py --append --date 2026-04-07  # 补追到指定日期
python scripts/append_and_retrain.py --retrain                   # 仅重训
python scripts/append_and_retrain.py --append --retrain          # 补追+重训
python scripts/backfill_predictions.py                           # 历史预测回填
```

查看调度状态：`GET /api/scheduler/status`

---

## 10. Web 应用架构

### 后端 API

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/forecast` | GET | 预测、历史数据、天气、风险 |
| `/api/metrics` | GET | 指定模型的 metrics.json |
| `/api/scheduler/status` | GET | 定时任务状态 |

### Dashboard v3

访问：`http://localhost:5000/dashboard/v3`

- Apple 风格三栏布局（ECharts + CSS Grid）
- 三条预测曲线（GRU / LSTM / Seq2Seq+Attention）
- 顶部天气预报横向滚动条（未来 14 天）
- 右侧面板：天气卡片 + 适游性风险温度计 + 推荐出行窗口
- 季节性预警阈值双红线（旺季/淡季自动切换）
- 节假日区域高亮 / 中英文切换 / 深色模式

---

## 11. 当前进度与后续计划

### 已完成

| 项目 | 状态 |
|------|------|
| 10年数据训练（GRU/LSTM/Seq2Seq） | ✅ |
| 消融实验（特征遮蔽法，6配置） | ✅ |
| Seq2Seq 注意力热力图 | ✅ |
| 季节性预警阈值（官方承载量×80%） | ✅ |
| 气象危险绝对阈值（旅游舒适度标准） | ✅ |
| 自动化数据管道（APScheduler） | ✅ |
| Docker 容器化部署 | ✅ |

### 后续调优方向（答辩前）

#### P0 — 降低 MAE/RMSE

1. **特征扩展**：新增 `lag_14`（14天滞后）、`rolling_mean_14`（14天滚动均值）、`is_peak_season`（旺淡季标记）
2. **超参调优**：lookback 从 30 → 45/60，测试对 MAE 的影响；GRU units 64→96
3. **训练期样本权重**：COVID/地震异常期（2020-2022）赋权 0.3，正常期权重 1.0

#### P1 — 新增对比模型

计划新增以下模型，形成完整方法演进链（统计基线 → 传统 ML → 深度学习）：

| 模型 | 定位 | 特征兼容性 |
|------|------|----------|
| **XGBoost** | 传统 ML 基线，证明 DL 优越性 | ✅ lag 特征展开输入 |
| **Transformer Encoder** | 前沿自注意力对比 | ✅ 8特征原生支持 |

XGBoost 使用 lag 特征展开（lag_1/7/14/28 + rolling_mean_7/14 + 天气特征 + 时间编码），Transformer 使用 Encoder-only + Dense Head 架构（输入 `(30, 8)`）。

所有新模型共用相同特征集、相同测试集切分，保证对比公平性。

#### P2 — 论文写作

| 章节 | 状态 | 待完成 |
|------|------|-------|
| Ch.3 Design & Implementation | 代码/架构全部完成 | 整理系统架构图、数据流图 |
| Ch.4 Results & Discussion | 现有实验数据就绪 | 新模型结果 + 对比分析撰写 |
| Ch.5 Conclusion & Future Work | 待写 | 依赖 Ch.4 完成 |

#### P3 — Future Work（论文建议方向）

- **预售票特征扩展**：将已发现的预售订票 API（未来7天逐日预订量）作为 Seq2Seq decoder 第9维输入，需积累 ≥6 个月历史数据后启动实验
- **小时级预测**：当前日级，若官网数据支持小时级爬取，可大幅提升运营决策精度
- **Transformer/TFT**：Temporal Fusion Transformer 对多步旅游预测有文献支撑，可替代 Seq2Seq

---

## 12. 项目维护信息

**技术栈**：Python 3.10+, TensorFlow 2.15, Flask 3.0, ECharts 5, SQLite, APScheduler

```bash
pip install -r requirements_all.txt
```

**最后更新**：2026年4月18日

### 更新日志

| 日期 | 内容 |
|------|------|
| 2026-04-18 | 移除 README 中 Uncertainty Interval、SHAP、Walk-forward 相关描述；相关脚本和结果归档至 `archive/uncertainty_shap_walkforward/`；新增后续调优方向与新模型计划 |
| 2026-04-12 | 气象预警阈值从 Q90 统计分位数改为旅游舒适度绝对标准；修复前端归因面板 camelCase/snake_case bug；前端缓存版本升至 v7 |
| 2026-04-09 | 修复爬虫 URL 和选择器；`--append` 合并批量回填逻辑；预警阈值改为季节性官方承载量×80%；发现并文档化九寨沟预售票 API |
| 2026-04-06 | 完成消融实验（特征遮蔽法）、SHAP 分析、Walk-forward 三模型4折、Seq2Seq 注意力热力图 |
| 2026-04-05 | 确立三模型架构（GRU/LSTM/Seq2Seq），废弃在线/离线切换开关 |
| 2026-04-03 | Dashboard v3 发布，完成10年数据训练 |
