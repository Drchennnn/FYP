# Codex 任务书 v2 — 五模型统一重训对比

> **当前状态**（2026-04-18）：
> - ✅ XGBoost 已完整训练（14特征，MAE=4326，NRMSE=16.7%）
> - ✅ Transformer 已完整训练（11特征，120ep→早停58ep，MAE=2909，NRMSE=12.1%）
> - ✅ Seq2Seq 已完整训练（8特征，MAE=3846）
> - ❌ GRU/LSTM 只跑了 1 epoch（无效），需要用 **11特征** 正式重训
> - ❌ 五模型 peak_threshold 不统一（旧run=18500，新run=32800），对比无意义
> - ❌ GRU/LSTM 脚本尚未加入 P0 三个新特征
>
> **你的核心任务**：改 GRU/LSTM 脚本加入 11 特征 → 正式重训 GRU+LSTM → 跑五模型对比

---

## 任务 1：GRU 脚本加入 P0 三个新特征

**文件**：`models/gru/train_gru_8features.py`

### 1-A：在 `load_and_engineer_features()` 中新增特征构造

找到函数内构造 lag_7 的位置，在其后追加：

```python
# P0 扩展特征
df["tourism_num_lag_14"] = df["tourism_num"].shift(14)
df["tourism_num_rolling_mean_14"] = df["tourism_num"].rolling(14).mean()
df["is_peak_season"] = df["date"].apply(is_peak_season).astype(float)
```

同时在 scaler fit/transform 段（lag_7_scaler 附近）追加：

```python
lag14_scaler = MinMaxScaler()
lag14_scaler.fit(train_df[["tourism_num_lag_14"]])
df["tourism_num_lag_14_scaled"] = lag14_scaler.transform(df[["tourism_num_lag_14"]]).reshape(-1)

rolling14_scaler = MinMaxScaler()
rolling14_scaler.fit(train_df[["tourism_num_rolling_mean_14"]])
df["rolling_mean_14_scaled"] = rolling14_scaler.transform(df[["tourism_num_rolling_mean_14"]]).reshape(-1)
# is_peak_season 不需要归一化（已是 0/1）
```

**重要**：scaler 只在训练集（`train_df`）上 fit，然后 transform 全量 df。与现有 lag_7 逻辑完全一致。

### 1-B：在 `build_sequences()` 内更新 `feature_cols` 列表

将原来的 8 个特征改为 11 个：

```python
feature_cols = [
    "visitor_count_scaled",
    "month_norm",
    "day_of_week_norm",
    "is_holiday",
    "is_peak_season",            # 新增
    "tourism_num_lag_7_scaled",
    "tourism_num_lag_14_scaled",  # 新增
    "rolling_mean_14_scaled",    # 新增
    "meteo_precip_sum_scaled",
    "temp_high_scaled",
    "temp_low_scaled",
]
```

### 1-C：确认 `is_peak_season` 函数存在

在文件顶部（`mark_core_holiday` 附近）加入：

```python
def is_peak_season(date_val: pd.Timestamp) -> int:
    """旺季：4月1日 ~ 11月15日"""
    m, d = int(date_val.month), int(date_val.day)
    if m < 4 or m > 11:
        return 0
    if m == 11 and d > 15:
        return 0
    return 1
```

### 1-D：dropna 覆盖新增列

确保 dropna 的 subset 包含新列：
```python
df = df.dropna(subset=[..., "tourism_num_lag_14", "tourism_num_rolling_mean_14"])
```

### 验证

```bash
python models/gru/train_gru_8features.py --epochs 2
# 输出应包含：使用特征: [..., 'is_peak_season', 'tourism_num_lag_14_scaled', 'rolling_mean_14_scaled']
# feature_count 应为 11
```

---

## 任务 2：LSTM 脚本同步相同修改

**文件**：`models/lstm/train_lstm_8features.py`

与任务 1 完全一致，对 `train_lstm_8features.py` 做同样的四步修改（1-A 到 1-D）。

### 验证

```bash
python models/lstm/train_lstm_8features.py --epochs 2
# feature_count 应为 11
```

---

## 任务 3：正式重训 GRU 和 LSTM（120 epoch）

任务 1、2 验证通过后，运行完整训练：

```bash
python run_pipeline.py --model gru --features 8 --epochs 120
python run_pipeline.py --model lstm --features 8 --epochs 120
```

> 注意：`run_pipeline.py` 的 `--features 8` 参数是脚本选择用的，实际特征数已由脚本内部决定（现在是11）。不需要改 run_pipeline.py。

**预期结果**：
- GRU：MAE 目标 < 3500，peak_threshold 应为 32800（与 Transformer/XGBoost 一致）
- LSTM：MAE 目标 < 4500

---

## 任务 4：运行五模型对比

五个模型全部完整训练后，运行：

```bash
python run_benchmark.py
```

`run_benchmark.py` 会自动发现 `output/runs/` 下各模型类型的最新有效 run，生成：
- `output/runs/run_compare_<timestamp>/report.md`
- `output/runs/run_compare_<timestamp>/compare_metrics.csv`

### 对比报告必须包含这五个模型

| 模型 | 特征数 | 预期 MAE | 备注 |
|------|--------|---------|------|
| GRU | 11 | < 3500 | 任务3重训后 |
| LSTM | 11 | < 4500 | 任务3重训后 |
| Seq2Seq | 8 | ~3846 | 已有，直接用 |
| XGBoost | 14 | ~4326 | 已有，直接用 |
| Transformer | 11 | ~2909 | 已有，直接用 |

**成功标准**：report.md 的 Metrics Table 中出现五行，且 GRU/LSTM 的 `epochs_trained > 30`（确认是完整训练）。

---

## 注意事项

1. **不要修改** `models/common/core_evaluation.py`（已有 NRMSE，不动）
2. **不要修改** Seq2Seq、XGBoost、Transformer 脚本（已完成）
3. lag_14 / rolling_mean_14 的 scaler 必须只在训练集 fit，逻辑与 lag_7 完全一致
4. `dropna` 要覆盖新增列，否则 lag_14（需要14天历史）会导致前14行有 NaN 未清理

---

## 完成标志

```bash
# 检查 GRU/LSTM 是否用了 11 特征且完整训练
python -c "
import json, glob
for pattern in ['output/runs/gru_8features_*/runs/*/metrics.json',
                'output/runs/lstm_8features_*/runs/*/metrics.json']:
    files = sorted(glob.glob(pattern))
    if files:
        d = json.load(open(files[-1]))
        m = d['meta']
        print(m['model_name'], 'feat=', m['feature_count'],
              'epochs=', m.get('epochs_trained','?'),
              'peak_thr=', m['peak_threshold'])
"
# 预期输出：
# gru  feat=11  epochs=XX(>30)  peak_thr=32800.0
# lstm feat=11  epochs=XX(>30)  peak_thr=32800.0
```
