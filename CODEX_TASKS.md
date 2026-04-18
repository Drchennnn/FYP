# Codex 任务书 v3 — 四模型调参 + 重跑对比

> **当前状态**（2026-04-18）：
> - ✅ 五模型已完整训练，最新对比见 `output/runs/run_compare_20260418_172530/report.md`
> - ✅ Seq2Seq 保持 8 特征不动（结果最稳定，MAE=3846）
> - ❌ GRU/LSTM 表现差于预期（MAE≈4300+，Suit-Recall 0.786/0.816），需要调参
> - ❌ Transformer 有进一步提升空间（MAE=2909，lookback=30）
> - ❌ XGBoost 超参未优化（MAE=4326，n_estimators=300 固定）
>
> **你的任务**：按下方调参方案修改各模型脚本默认值 → 重训四个模型 → 重跑五模型对比

---

## 当前五模型基准（对比参考）

| 模型 | 特征 | MAE | NRMSE | sMAPE | Suit-Recall |
|------|------|-----|-------|-------|-------------|
| Transformer | 11 | 2,909 | 12.1% | 16.1% | 0.981 |
| Seq2Seq | 8 | 3,846 | — | 21.0% | 0.952 ← 不动 |
| LSTM | 11 | 4,306 | 16.2% | 22.3% | 0.816 |
| XGBoost | 14 | 4,326 | 16.7% | 20.7% | 0.819 |
| GRU | 11 | 4,389 | 16.4% | 21.7% | 0.786 |

---

## 任务 1：GRU 调参

**文件**：`models/gru/train_gru_8features.py`

找到 `argparse` 参数定义，修改以下默认值：

```python
# 原来
parser.add_argument("--look-back", type=int, default=30)
parser.add_argument("--epochs",    type=int, default=120)

# 改为
parser.add_argument("--look-back", type=int, default=45)
parser.add_argument("--epochs",    type=int, default=150)
```

找到模型 `build_model()` 或内联的 GRU 层定义，修改单元数和 dropout：

```python
# 原来（大约是）
GRU(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)
GRU(64,  return_sequences=False, dropout=0.2, recurrent_dropout=0.2)

# 改为
GRU(128, return_sequences=True,  dropout=0.1, recurrent_dropout=0.1)
GRU(96,  return_sequences=False, dropout=0.1, recurrent_dropout=0.1)
```

找到 EarlyStopping callback，修改 patience：

```python
# 原来
EarlyStopping(patience=15, ...)

# 改为
EarlyStopping(patience=20, restore_best_weights=True, ...)
```

### 验证

```bash
python models/gru/train_gru_8features.py --epochs 3
# 确认输出包含：look_back=45，无报错
```

---

## 任务 2：LSTM 调参

**文件**：`models/lstm/train_lstm_8features.py`

与 GRU 完全相同的四处修改：

1. `--look-back` 默认值：`30` → `45`
2. `--epochs` 默认值：`120` → `150`
3. LSTM 第二层单元数：`64` → `96`，dropout：`0.2` → `0.1`
4. EarlyStopping patience：`15` → `20`

### 验证

```bash
python models/lstm/train_lstm_8features.py --epochs 3
# 确认输出包含：look_back=45，无报错
```

---

## 任务 3：Transformer 调参

**文件**：`models/transformer/train_transformer_8features.py`

修改 argparse 默认值：

```python
# 原来
parser.add_argument("--look-back", type=int, default=30)
parser.add_argument("--dropout",   type=float, default=0.1)
parser.add_argument("--epochs",    type=int, default=120)

# 改为
parser.add_argument("--look-back", type=int, default=45)
parser.add_argument("--dropout",   type=float, default=0.15)
parser.add_argument("--epochs",    type=int, default=150)
```

EarlyStopping patience 同样改为 20。

### 验证

```bash
python models/transformer/train_transformer_8features.py --epochs 3
# 确认输出包含：look_back=45，无报错
```

---

## 任务 4：XGBoost 调参

**文件**：`models/xgboost/train_xgboost_8features.py`

修改 `build_model()` 中的超参：

```python
# 原来
XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.5,
    reg_lambda=1.0,
    ...
)

# 改为
XGBRegressor(
    n_estimators=500,       # 300 → 500
    max_depth=5,            # 6 → 5（防过拟合）
    learning_rate=0.03,     # 0.05 → 0.03（配合更多树）
    subsample=0.8,
    colsample_bytree=0.7,   # 0.8 → 0.7
    min_child_weight=3,     # 新增，防止稀疏节点
    reg_alpha=0.3,          # 0.5 → 0.3
    reg_lambda=1.5,         # 1.0 → 1.5
    early_stopping_rounds=40,  # 30 → 40
    ...
)
```

同时修改 argparse 默认值：

```python
parser.add_argument("--n-estimators", type=int, default=500)
parser.add_argument("--max-depth",    type=int, default=5)
parser.add_argument("--lr",           type=float, default=0.03)
```

### 验证

```bash
python models/xgboost/train_xgboost_8features.py --n-estimators 50 --max-depth 4
# 确认正常运行，无报错
```

---

## 任务 5：重训四个模型

四个脚本验证通过后，**按顺序**运行完整训练：

```bash
python run_pipeline.py --model gru --features 8 --epochs 150
python run_pipeline.py --model lstm --features 8 --epochs 150
python models/transformer/train_transformer_8features.py
python models/xgboost/train_xgboost_8features.py
```

> Seq2Seq 不重训，`run_benchmark.py` 会自动复用已有结果。

**预期训练时间**：GRU ~10min，LSTM ~15min，Transformer ~20min，XGBoost ~5min

---

## 任务 6：重跑五模型对比

```bash
python run_benchmark.py
```

生成新的 `output/runs/run_compare_<timestamp>/report.md`。

**目标指标**（调参后预期）：

| 模型 | MAE 目标 | Suit-Recall 目标 |
|------|---------|----------------|
| GRU | < 3,500 | > 0.90 |
| LSTM | < 3,800 | > 0.90 |
| Transformer | < 2,700 | > 0.97 |
| XGBoost | < 4,000 | > 0.85 |

---

## 完成标志

```bash
python -c "
import json, glob
models = {
    'gru':         'output/runs/gru_8features_*/runs/*/metrics.json',
    'lstm':        'output/runs/lstm_8features_*/runs/*/metrics.json',
    'transformer': 'output/runs/transformer_8features_*/runs/*/metrics.json',
    'xgboost':     'output/runs/xgboost_8features_*/runs/*/metrics.json',
}
for name, pattern in models.items():
    files = sorted(glob.glob(pattern))
    if files:
        d = json.load(open(files[-1]))
        r = d['regression']
        m = d['meta']
        sw = d.get('suitability_warning', {})
        print(f\"{name:12s} MAE={r['mae']:7.0f}  NRMSE={r.get('nrmse', float('nan')):.3f}  epochs={m.get('epochs_trained','?'):>3}  recall={sw.get('recall','?')}\")
"
# 预期：四个模型 MAE 均低于当前值，epochs > 30
```
