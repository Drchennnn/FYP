# Codex 任务书 v10 — 集成预测（Ensemble）

> **当前状态**（2026-04-20）：
> - ✅ 三模型已训练完毕，权重文件存在，CSV预测文件存在
> - ✅ 在线推理已修复（GRU/Transformer/XGBoost 均可滚动推理3步）
> - ✅ 预警逻辑已改为客流主导（超阈值即预警，天气仅辅助）
> - ❌ 尚无集成预测列（ensemble_pred），前端无第四条曲线

---

## 当前模型状态（重要，必读）

### 已训练模型及权重路径

| 模型 | 权重文件 | 输入形状 | 特征数 | MAE | NRMSE | Suit-Recall |
|------|---------|---------|--------|-----|-------|-------------|
| GRU | `output/runs/gru_8features_20260418_194417/runs/run_20260418_194417_lb30_ep150_gru_8features/weights/gru_jiuzhaigou.h5` | (30, 11) | 11 | 2,871 | 13.0% | 0.960 |
| Transformer | `output/runs/transformer_8features_20260418_195030/runs/run_20260418_195030_lb45_ep150_transformer_8features/weights/transformer_8features.h5` | (45, 11) | 11 | 3,100 | 13.5% | 0.960 |
| XGBoost | `output/runs/xgboost_8features_20260418_181137/runs/run_20260418_181137_xgboost_8features/weights/xgboost_model.json` | 14维表格 | 14 | 2,531 | 12.1% | 0.963 |

### 预测CSV路径

| 模型 | 预测CSV |
|------|---------|
| GRU | `output/runs/gru_8features_20260418_194417/runs/run_20260418_194417_lb30_ep150_gru_8features/gru_test_predictions.csv` |
| Transformer | `output/runs/transformer_8features_20260418_195030/runs/run_20260418_195030_lb45_ep150_transformer_8features/transformer_test_predictions.csv` |
| XGBoost | `output/runs/xgboost_8features_20260418_181137/runs/run_20260418_181137_xgboost_8features/xgboost_test_predictions.csv` |

CSV格式：`date, y_true, y_pred`（列名固定，BOM UTF-8）

### 特征说明

**GRU / Transformer（11特征，顺序固定）**：
```
visitor_count_scaled, month_norm, day_of_week_norm, is_holiday, is_peak_season,
tourism_num_lag_7_scaled, tourism_num_lag_14_scaled, rolling_mean_14_scaled,
meteo_precip_sum_scaled, temp_high_scaled, temp_low_scaled
```

**XGBoost（14特征，model.feature_names_in_ 顺序）**：
```
month_norm, day_of_week_norm, is_holiday, is_peak_season,
tourism_num_lag_1_scaled, tourism_num_lag_7_scaled, tourism_num_lag_14_scaled, tourism_num_lag_28_scaled,
rolling_mean_7_scaled, rolling_mean_14_scaled, rolling_std_7_scaled,
meteo_precip_sum_scaled, temp_high_scaled, temp_low_scaled
```

---

## 本轮目标：集成预测（无需重训）

**方案**：加权平均集成

```
ensemble_pred = XGBoost × 0.5 + GRU × 0.3 + Transformer × 0.2
```

权重依据：MAE反比加权（MAE越低权重越高）。

---

## 任务 1：离线CSV集成脚本

新建 `scripts/ensemble_predict.py`，读取三个模型的预测CSV，按日期对齐后计算加权均值，输出集成预测CSV并打印MAE对比：

```python
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
import glob, os

def find_latest_pred_csv(model_prefix):
    pattern = f'output/runs/{model_prefix}_*/runs/run_*/*_test_predictions.csv'
    files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    return files[0] if files else None

gru_csv  = find_latest_pred_csv('gru_8features')
tr_csv   = find_latest_pred_csv('transformer_8features')
xgb_csv  = find_latest_pred_csv('xgboost_8features')

dfs = {}
for name, path in [('gru', gru_csv), ('transformer', tr_csv), ('xgboost', xgb_csv)]:
    if path:
        df = pd.read_csv(path)
        df['date'] = pd.to_datetime(df['date']).dt.date
        dfs[name] = df.set_index('date')[['y_true', 'y_pred']].rename(columns={'y_pred': name})

# 对齐日期
base = dfs['gru'][['y_true']].copy()
for name, df in dfs.items():
    base[name] = df[name]
base = base.dropna(subset=['gru', 'transformer', 'xgboost'])

# 加权集成
W = {'gru': 0.3, 'transformer': 0.2, 'xgboost': 0.5}
base['ensemble'] = sum(base[k] * w for k, w in W.items())

# MAE对比
mask = base['y_true'].notna()
for col in ['gru', 'transformer', 'xgboost', 'ensemble']:
    mae = mean_absolute_error(base.loc[mask, 'y_true'], base.loc[mask, col])
    print(f'{col:12s} MAE: {mae:.2f}')

# 保存
out = base.reset_index().rename(columns={'index': 'date'})
out.to_csv('output/ensemble_test_predictions.csv', index=False)
print('Saved: output/ensemble_test_predictions.csv')
```

---

## 任务 2：app.py 加入 ensemble_pred 列

在 `web_app/app.py` 的 `api_forecast` 函数里，找到：

```python
df_merge = df_base.sort_values('date').reset_index(drop=True)
```

在这行**之后**插入：

```python
# 集成预测列（加权平均，缺任一模型则按可用模型归一化权重计算）
_ens_weights = {'gru_pred': 0.3, 'transformer_pred': 0.2, 'xgboost_pred': 0.5}
_avail = {c: w for c, w in _ens_weights.items() if c in df_merge.columns}
if _avail:
    _total_w = sum(_avail.values())
    df_merge['ensemble_pred'] = sum(
        pd.to_numeric(df_merge[c], errors='coerce') * (w / _total_w)
        for c, w in _avail.items()
    )
else:
    df_merge['ensemble_pred'] = np.nan
```

在返回的 `series` 字典里加入（找到 `'xgboost_pred': _to_num_list('xgboost_pred'),` 这行后面）：
```python
'ensemble_pred':    _to_num_list('ensemble_pred'),
```

在线推理的 `out_rows` 组装循环里（找到 `out_rows.append(row)` 前面），加：
```python
_ens_vals = [row.get(c, float('nan')) for c in ['gru_pred', 'transformer_pred', 'xgboost_pred']]
_ens_valid = [v for v in _ens_vals if not (isinstance(v, float) and np.isnan(v))]
row['ensemble_pred'] = float(np.mean(_ens_valid)) if _ens_valid else float('nan')
```

在 `df_future` 的 concat 列表里加 `'ensemble_pred'`：
```python
df_future[['date', 'actual', 'precip_mm', 'temp_high_c', 'temp_low_c',
           'weather_code_en', 'wind_level', 'wind_dir_en', 'wind_max',
           'aqi_value', 'aqi_level_en',
           'gru_pred', 'transformer_pred', 'xgboost_pred', 'ensemble_pred']]
```

---

## 任务 3：前端加第四条集成曲线

### dashboard_v3.js

**normalizeForecastPayload** 里加（紧接 xgboost 那行后面）：
```javascript
out.series.ensemble = safeArr(s.ensemble_pred || [], n, safeNum);
```

**CURVE_DEFS** 里加第四条：
```javascript
{ key: 'ensemble', nameZh: '集成预测', nameEn: 'Ensemble', color: '#bf5af2' },
```

**initLegendToggles** 的 `resolveSeriesName` 函数里加：
```javascript
if (key === 'ensemble') return state.lang === 'zh' ? '集成预测' : 'Ensemble';
```

**buildChartOption** 里 hidden legend data 改为：
```javascript
legend: { show: false, data: [
  state.lang === 'zh' ? '实际客流' : 'Actual',
  'GRU', 'Transformer', 'XGBoost',
  state.lang === 'zh' ? '集成预测' : 'Ensemble'
]},
```

### dashboard_v3.html

在 XGBoost 图例按钮后面加：
```html
<button class="v3-chart-key__item v3-chart-key__item--toggle v3-chart-key__item--active"
        type="button" data-series-name="ensemble" title="点击显示/隐藏">
  <span class="v3-chart-key__swatch" style="background:#bf5af2"></span>
  <span class="v3-chart-key__label">集成预测（三模型加权均值）</span>
</button>
```

---

## 任务 4：验证

```bash
# 离线集成MAE
python scripts/ensemble_predict.py
# 期望：Ensemble MAE < XGBoost MAE (2531)

# 启动服务验证前端
cd web_app && python app.py
# 访问 http://localhost:5000/dashboard/v3
# 验证：图表出现第四条紫色曲线（集成预测），可通过图例按钮切换
```

---

## 成功标准

- `scripts/ensemble_predict.py` 打印四行MAE对比，Ensemble MAE < 2,531
- `/api/forecast` 响应的 `series.ensemble_pred` 非全null（离线模式下有100+个值）
- 前端图表出现紫色集成预测曲线，在线模式下延伸至未来3天
- 图例按钮可切换集成曲线显示/隐藏

---

## 目标指标

| 模型 | 整体 MAE | 备注 |
|------|---------|------|
| XGBoost（当前最优） | 2,531 | 基准 |
| GRU | 2,871 | |
| Transformer | 3,100 | |
| **集成（目标）** | **< 2,400** | 加权平均，无需重训 |
