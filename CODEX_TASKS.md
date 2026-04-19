# Codex 任务书 v11 — 特征工程优化（删冗余 + 加节假日距离特征）

> **当前状态**（2026-04-20）：
> - ✅ 三模型已训练，集成预测已接入前端
> - ✅ GRU/Transformer 11特征，XGBoost 14特征
> - ❌ `rolling_mean_14_scaled` 与 lag_7/lag_14 高度冗余，待删除
> - ❌ `tourism_num_lag_28_scaled`（XGBoost）贡献低，待删除
> - ❌ 缺少节假日距离特征，清明节等阶跃峰值召回率仅33%

---

## 当前模型状态（必读）

| 模型 | 权重路径 | 当前特征数 | MAE |
|------|---------|-----------|-----|
| GRU | `output/runs/gru_8features_20260418_194417/runs/run_20260418_194417_lb30_ep150_gru_8features/weights/gru_jiuzhaigou.h5` | 11 | 2,871 |
| Transformer | `output/runs/transformer_8features_20260418_195030/runs/run_20260418_195030_lb45_ep150_transformer_8features/weights/transformer_8features.h5` | 11 | 3,100 |
| XGBoost | `output/runs/xgboost_8features_20260418_181137/runs/run_20260418_181137_xgboost_8features/weights/xgboost_model.json` | 14 | 2,531 |

**重训后特征数**：GRU/Transformer → 12，XGBoost → 13

---

## 特征变更说明

### 删除

| 特征 | 影响模型 | 原因 |
|------|---------|------|
| `rolling_mean_14_scaled` | GRU, Transformer, XGBoost | 与 lag_7/lag_14 高度相关（r>0.9），信息冗余 |
| `tourism_num_lag_28_scaled` | XGBoost | 28天前客流对3天预测贡献低，月度周期已由 month_norm 覆盖 |

### 新增

| 特征 | 影响模型 | 计算方式 | 作用 |
|------|---------|---------|------|
| `days_to_next_holiday` | 全部 | 距下一个法定节假日的天数，上限截断为14，归一化为 `/14` | 让模型提前感知节假日峰值 |
| `days_since_last_holiday` | 全部 | 距上一个法定节假日结束的天数，上限截断为14，归一化为 `/14` | 捕捉节后客流回落信号 |

**节假日定义**：使用 `chinese_calendar` 库（已安装），与训练脚本现有的 `is_holiday` 特征保持一致。

---

## 任务 1：GRU（`models/gru/train_gru_8features.py`）

### 1a. 在 `load_and_engineer_features` 函数里加新特征计算

找到 `df["is_holiday"] = df["date"].apply(mark_core_holiday).astype(float)` 这行之后，加入：

```python
# 节假日距离特征
def days_to_next_hol(date_val: pd.Timestamp) -> float:
    for delta in range(1, 15):
        try:
            if cncal.is_holiday((date_val + pd.Timedelta(days=delta)).date()):
                return delta / 14.0
        except Exception:
            pass
    return 1.0  # 超过14天，归一化为1

def days_since_last_hol(date_val: pd.Timestamp) -> float:
    for delta in range(1, 15):
        try:
            if cncal.is_holiday((date_val - pd.Timedelta(days=delta)).date()):
                return delta / 14.0
        except Exception:
            pass
    return 1.0

df["days_to_next_holiday"] = df["date"].apply(days_to_next_hol).astype(float)
df["days_since_last_holiday"] = df["date"].apply(days_since_last_hol).astype(float)
```

### 1b. 更新 `feature_cols`

将：
```python
feature_cols = [
    "visitor_count_scaled",
    "month_norm",
    "day_of_week_norm",
    "is_holiday",
    "is_peak_season",
    "tourism_num_lag_7_scaled",
    "tourism_num_lag_14_scaled",
    "rolling_mean_14_scaled",      # ← 删除
    "meteo_precip_sum_scaled",
    "temp_high_scaled",
    "temp_low_scaled",
]
```

改为：
```python
feature_cols = [
    "visitor_count_scaled",
    "month_norm",
    "day_of_week_norm",
    "is_holiday",
    "is_peak_season",
    "days_to_next_holiday",        # ← 新增
    "days_since_last_holiday",     # ← 新增
    "tourism_num_lag_7_scaled",
    "tourism_num_lag_14_scaled",
    "meteo_precip_sum_scaled",
    "temp_high_scaled",
    "temp_low_scaled",
]
```

**最终12特征，顺序固定，不可改变。**

---

## 任务 2：Transformer（`models/transformer/train_transformer_8features.py`）

与 GRU 完全相同的改动：
- 在特征工程函数里加 `days_to_next_holiday` 和 `days_since_last_holiday` 的计算（同上）
- 更新 `feature_cols`，删除 `rolling_mean_14_scaled`，新增两个节假日距离特征
- 顺序与 GRU 保持一致（12特征）

---

## 任务 3：XGBoost（`models/xgboost/train_xgboost_8features.py`）

### 3a. 加新特征计算（同上，在 `is_holiday` 计算后加入）

### 3b. 更新 `FEATURE_COLS` 列表

将现有14特征改为13特征：
- 删除 `tourism_num_lag_28_scaled`
- 删除 `rolling_mean_14_scaled`
- 新增 `days_to_next_holiday_scaled`（注意XGBoost用 `_scaled` 后缀命名，但这两个特征已经是0-1归一化，直接用原值即可，列名用 `days_to_next_holiday` 和 `days_since_last_holiday`）

**最终13特征**：
```python
FEATURE_COLS = [
    "month_norm",
    "day_of_week_norm",
    "is_holiday",
    "is_peak_season",
    "days_to_next_holiday",        # ← 新增
    "days_since_last_holiday",     # ← 新增
    "tourism_num_lag_1_scaled",
    "tourism_num_lag_7_scaled",
    "tourism_num_lag_14_scaled",
    # tourism_num_lag_28_scaled    ← 删除
    "rolling_mean_7_scaled",
    # rolling_mean_14_scaled       ← 删除
    "rolling_std_7_scaled",
    "meteo_precip_sum_scaled",
    "temp_high_scaled",
    "temp_low_scaled",
]
```

---

## 任务 4：同步 `web_app/app.py` 在线推理

找到 `_build_window_feat` 函数，同步更新特征构造逻辑。

### 4a. 在函数开头加节假日距离计算辅助函数

在 `_build_window_feat` 函数定义之前（或内部），加：

```python
def _days_to_next_hol(d):
    try:
        import chinese_calendar as _cc
        for delta in range(1, 15):
            if _cc.is_holiday((d + timedelta(days=delta))):
                return delta / 14.0
    except Exception:
        pass
    return 1.0

def _days_since_last_hol(d):
    try:
        import chinese_calendar as _cc
        for delta in range(1, 15):
            if _cc.is_holiday((d - timedelta(days=delta))):
                return delta / 14.0
    except Exception:
        pass
    return 1.0
```

### 4b. 更新 `_build_window_feat` 函数体

将现有的特征构造改为12特征（GRU/Transformer）：

```python
def _build_window_feat(visitor_window, lag7_window, dates, n_features=12):
    rows = []
    for i, d in enumerate(dates):
        m_norm = (d.month - 1) / 11.0
        dow_norm = d.weekday() / 6.0
        hol = float(_is_holiday(d))
        m = d.month
        is_peak = float((4 <= m <= 10) or (m == 11 and d.day <= 15))
        d2n = _days_to_next_hol(d)
        d2s = _days_since_last_hol(d)
        row = [
            visitor_window[i],   # visitor_count_scaled
            m_norm,              # month_norm
            dow_norm,            # day_of_week_norm
            hol,                 # is_holiday
            is_peak,             # is_peak_season
            d2n,                 # days_to_next_holiday
            d2s,                 # days_since_last_holiday
            lag7_window[i],      # tourism_num_lag_7_scaled
        ]
        if n_features >= 12:
            lag14_s = visitor_window[max(0, i - 7)]  # 近似 lag14
            row.append(lag14_s)  # tourism_num_lag_14_scaled
        row.extend([
            0.0,   # meteo_precip_sum_scaled
            0.5,   # temp_high_scaled
            0.5,   # temp_low_scaled
        ])
        rows.append(row)
    return np.array(rows, dtype=np.float32)
```

**注意**：`n_features` 默认改为12（新的GRU/Transformer特征数）。天气特征索引也要更新：
- `precip_idx = 9`（原来是5/8）
- `temp_h_idx = 10`（原来是6/9）
- `temp_l_idx = 11`（原来是7/10）

在 `_predict_single_step_model` 函数里，更新天气特征索引的计算：

```python
# 天气特征固定在最后3位
precip_idx = n_feat - 3
temp_h_idx = n_feat - 2
temp_l_idx = n_feat - 1
```

### 4c. 更新 XGBoost 在线推理特征字典

找到 XGBoost 在线推理的 `_feat` 字典，删除 `tourism_num_lag_28_scaled` 和 `rolling_mean_14_scaled`，加入两个新特征：

```python
_feat = {
    'month_norm':                _m_norm,
    'day_of_week_norm':          _dow_norm,
    'is_holiday':                _hol,
    'is_peak_season':            _is_peak,
    'days_to_next_holiday':      _days_to_next_hol(_pred_date),   # ← 新增
    'days_since_last_holiday':   _days_since_last_hol(_pred_date), # ← 新增
    'tourism_num_lag_1_scaled':  _lag(_win, 1),
    'tourism_num_lag_7_scaled':  _lag(_win, 7),
    'tourism_num_lag_14_scaled': _lag(_win, 14),
    # tourism_num_lag_28_scaled  ← 删除
    'rolling_mean_7_scaled':     float(np.mean(_win[-7:])) if len(_win) >= 7 else float(np.mean(_win)),
    # rolling_mean_14_scaled     ← 删除
    'rolling_std_7_scaled':      float(np.std(_win[-7:])) if len(_win) >= 7 else 0.0,
    'meteo_precip_sum_scaled':   _scale_precip(_precip_raw),
    'temp_high_scaled':          _scale_temp_high(_th_raw),
    'temp_low_scaled':           _scale_temp_low(_tl_raw),
}
```

---

## 任务 5：重训三个模型

```bash
# GRU（约60-90分钟）
python run_pipeline.py --model gru --features 8 --epochs 150

# Transformer（约60-90分钟）
python models/transformer/train_transformer_8features.py

# XGBoost（约5分钟）
python models/xgboost/train_xgboost_8features.py
```

---

## 任务 6：验证

```bash
python -c "
import json, glob
for name, pat in [('gru','output/runs/gru_8features_*/runs/*/metrics.json'),
                  ('transformer','output/runs/transformer_8features_*/runs/*/metrics.json'),
                  ('xgboost','output/runs/xgboost_8features_*/runs/*/metrics.json')]:
    files = sorted(glob.glob(pat))
    if files:
        d = json.load(open(files[-1]))
        r = d.get('regression', {})
        sw = d.get('suitability_warning_weighted', d.get('suitability_warning', {}))
        print(f'{name}: MAE={r.get(\"mae\",\"?\"):.0f} recall={sw.get(\"recall_weighted\",sw.get(\"recall\",\"?\"))}')
"
```

---

## 成功标准

- 三个模型重训完成，metrics.json 存在
- GRU/Transformer 特征数为12（日志打印 `特征维度: 12`）
- XGBoost 特征数为13（`model.feature_names_in_` 长度为13，包含 `days_to_next_holiday`）
- GRU MAE ≤ 2,871（不退步），清明节召回率提升（目标 > 50%）
- `web_app/app.py` 在线推理不报特征维度错误

---

## 目标指标

| 模型 | 当前 MAE | 目标 MAE | 清明召回（当前） | 目标 |
|------|---------|---------|--------------|------|
| GRU | 2,871 | ≤ 2,800 | 33% | > 50% |
| Transformer | 3,100 | ≤ 3,000 | 33% | > 50% |
| XGBoost | 2,531 | ≤ 2,500 | 33% | > 50% |
