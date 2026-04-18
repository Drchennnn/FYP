# Codex 任务书 — 九寨沟客流预测系统扩展

> **写给 Codex**：本项目是一个端到端旅游客流预测系统，已有 GRU / LSTM / Seq2Seq+Attention 三个深度学习模型跑通。你的任务是完成两项扩展（P0 特征扩展 + P1 新模型），所有代码需与现有框架风格一致。
>
> **读懂再动手**：先读 `models/gru/train_gru_8features.py`（~500行），理解整体数据流、特征工程、evaluate_and_save_run 调用方式，再逐项完成下方任务。

---

## 项目结构速览

```
FYP/
├── data/processed/jiuzhaigou_daily_features_2016-01-01_2026-04-02.csv  ← 训练数据
├── models/
│   ├── common/
│   │   ├── core_evaluation.py    ← evaluate_and_save_run()（不要改）
│   │   ├── evaluator.py          ← calculate_metrics()（不要改）
│   │   └── preprocess.py
│   ├── gru/train_gru_8features.py      ← 参考实现（最完整）
│   ├── lstm/train_lstm_8features.py
│   ├── xgboost/train_xgboost_8features.py   ← 骨架，待你补全
│   └── transformer/train_transformer_8features.py  ← 骨架，待你补全
├── output/runs/   ← 所有训练结果输出目录（自动创建）
└── run_pipeline.py
```

---

## P0 — 特征扩展（所有模型共用，优先完成）

**目标**：在现有 8 特征基础上新增 3 个特征，形成 **11 特征版本**。

### 新增特征

| 特征名 | 说明 | 实现方式 |
|--------|------|---------|
| `tourism_num_lag_14_scaled` | 14天滞后客流（归一化） | `df['tourism_num'].shift(14)`，用训练集 scaler transform |
| `rolling_mean_14_scaled` | 14天滚动均值（归一化） | `df['tourism_num'].rolling(14).mean()`，同 scaler |
| `is_peak_season` | 旺淡季标记（0/1） | 4月1日~11月15日=1，其余=0，无需归一化 |

### 需要修改的文件

1. **`models/gru/train_gru_8features.py`** 的 `load_and_engineer_features()`：
   - 在现有 8 特征构造之后，追加 3 个新特征
   - 更新 `FEATURE_COLS` 列表（8→11列）
   - **关键**：`lag_14` 和 `rolling_mean_14` 的 scaler 必须只在训练集 fit，然后 transform 全集（与现有 lag_7 逻辑完全一致）

2. **`models/lstm/train_lstm_8features.py`**：同上，保持与 GRU 特征完全一致

3. **`models/lstm/train_seq2seq_attention_8features.py`**：
   - Encoder 输入从 `(30, 8)` → `(30, 11)`
   - Decoder 输入（外生特征）从 `(7, 7)` → `(7, 10)`（不含 visitor_count_scaled，其余 10 个）
   - 注意：`encoder_features=11`, `decoder_features=10` 参数需更新

### 验证标准

修改后运行以下命令，若无报错且输出 MAE 数值则通过：
```bash
python run_pipeline.py --model gru --features 11 --epochs 5
```

---

## P0.5 — NRMSE 归一化指标（P0 完成后立即做）

**目标**：在所有模型的评估结果中，将 RMSE 替换为 **NRMSE（值域归一化 RMSE）**，作为论文主要回归指标。

**定义**：
```
NRMSE = RMSE / (y_true_max - y_true_min)
```
其中 `y_true` 为**反归一化后的真实客流量**（人次/天，非 scaled 值）。

### 需要修改的文件：`models/common/core_evaluation.py`

**Step 1**：在 `_rmse()` 函数下方（约第 188 行）新增 `_nrmse()` 函数：

```python
def _nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = _to_1d(y_true).astype(float)
    y_pred = _to_1d(y_pred).astype(float)
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    value_range = float(y_true.max() - y_true.min())
    if value_range == 0:
        return float("nan")
    return rmse / value_range
```

**Step 2**：在 `reg` 字典（约第 338 行）中加入 `nrmse`：

```python
reg = {
    "mae":   _mae(y_true_flat, y_pred_flat),
    "rmse":  _rmse(y_true_flat, y_pred_flat),
    "nrmse": _nrmse(y_true_flat, y_pred_flat),   # ← 新增
    "smape": _smape(y_true_flat, y_pred_flat),
}
```

**Step 3**：在 per-horizon 指标字典（约第 447 行，`for h in range(H)` 循环内）同样加入 `nrmse`：

```python
{
    "mae":   _mae(y_true[:, h], y_pred[:, h]),
    "rmse":  _rmse(y_true[:, h], y_pred[:, h]),
    "nrmse": _nrmse(y_true[:, h], y_pred[:, h]),   # ← 新增
    "smape": _smape(y_true[:, h], y_pred[:, h]),
}
```

**Step 4**：在约第 620 行的顶层 metrics 汇总处加入 `nrmse`：

```python
"mae":   metrics["regression"]["mae"],
"rmse":  metrics["regression"]["rmse"],
"nrmse": metrics["regression"]["nrmse"],   # ← 新增
"smape": metrics["regression"]["smape"],
```

### 验证

```bash
python -c "
import json
f = open('output/runs/gru_8features_20260406_172031/runs/run_20260406_172031_lb30_ep120_gru_8features/metrics.json')
m = json.load(f)
print('nrmse' in m['regression'], m['regression'].get('nrmse'))
"
# 预期：True 0.137...（约13.7%）
```

> **注意**：`core_evaluation.py` 中 RMSE 保留不删（`metrics.json` 向后兼容），`nrmse` 作为新增字段并列存在。

---

## P1-A — XGBoost 模型补全

**文件**：`models/xgboost/train_xgboost_8features.py`

框架骨架已写好，所有 `TODO` 标注了需要实现的位置。按以下顺序补全：

### 步骤 1：安装依赖

```bash
pip install xgboost
```

在文件顶部取消注释：
```python
import xgboost as xgb
from xgboost import XGBRegressor
```

### 步骤 2：`load_and_engineer_features()`

参照 `models/gru/train_gru_8features.py` 的同名函数实现，但输出是 **表格形式**（非序列），包含以下列：

```python
FEATURE_COLS = [
    "month_norm", "day_of_week_norm", "is_holiday", "is_peak_season",
    "tourism_num_lag_1_scaled",   # 新增 lag_1
    "tourism_num_lag_7_scaled",
    "tourism_num_lag_14_scaled",  # 新增 lag_14
    "tourism_num_lag_28_scaled",  # 新增 lag_28
    "rolling_mean_7_scaled",
    "rolling_mean_14_scaled",     # 新增 rolling_14
    "rolling_std_7_scaled",       # 新增 rolling_std
    "meteo_precip_sum_scaled",
    "temp_high_scaled",
    "temp_low_scaled",
]
```

注意：XGBoost 不需要 `build_sequences()`，每行直接是一个样本。

### 步骤 3：`build_model()`

```python
return XGBRegressor(
    n_estimators=args.n_estimators,   # 默认300
    max_depth=args.max_depth,          # 默认6
    learning_rate=args.lr,             # 默认0.05
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.5,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=30,          # 配合 eval_set 提前停止
)
```

### 步骤 4：`compute_sample_weights()`

```python
# 2020-01-01 ~ 2022-12-31 赋权重 0.3，其余 1.0
mask = (train_df["date"] >= "2020-01-01") & (train_df["date"] <= "2022-12-31")
weights = np.where(mask, 0.3, 1.0)
return weights
```

### 步骤 5：`main()` 中补全训练与评估逻辑

参照骨架注释中的 TODO 伪代码，逐段实现。  
**关键**：`evaluate_and_save_run()` 的调用签名必须与 GRU 脚本完全一致（见骨架注释）。

### 运行验证

```bash
python models/xgboost/train_xgboost_8features.py --n-estimators 100 --max-depth 4
# 预期输出：run_dir 路径 + MAE 数值
# 输出目录：output/runs/xgboost_8features_<timestamp>/
```

---

## P1-B — Transformer 模型补全

**文件**：`models/transformer/train_transformer_8features.py`

### 步骤 1：`PositionalEncoding`

```python
# __init__ 中预计算
positions = np.arange(max_len)[:, np.newaxis]           # (max_len, 1)
dims = np.arange(d_model)[np.newaxis, :]                # (1, d_model)
angles = positions / np.power(10000, (2*(dims//2)) / d_model)
angles[:, 0::2] = np.sin(angles[:, 0::2])
angles[:, 1::2] = np.cos(angles[:, 1::2])
self.pe = tf.cast(angles[np.newaxis, :, :], tf.float32)  # (1, max_len, d_model)

# call 中
def call(self, x):
    return x + self.pe[:, :tf.shape(x)[1], :]
```

### 步骤 2：`TransformerEncoderBlock`

```python
# __init__
self.mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model//num_heads)
self.ffn1 = tf.keras.layers.Dense(dff, activation="relu")
self.ffn2 = tf.keras.layers.Dense(d_model)
self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
self.drop1 = tf.keras.layers.Dropout(dropout_rate)
self.drop2 = tf.keras.layers.Dropout(dropout_rate)

# call
def call(self, x, training=False):
    attn = self.mha(x, x, training=training)
    attn = self.drop1(attn, training=training)
    x = self.norm1(x + attn)
    ffn = self.ffn2(self.ffn1(x))
    ffn = self.drop2(ffn, training=training)
    return self.norm2(x + ffn)
```

### 步骤 3：`build_model()`

```python
inp = tf.keras.Input(shape=(look_back, n_features))
x = tf.keras.layers.Dense(d_model)(inp)          # 特征投影
x = PositionalEncoding(d_model)(x)
for _ in range(num_layers):
    x = TransformerEncoderBlock(num_heads, d_model, dff, dropout_rate)(x)
x = tf.keras.layers.GlobalAveragePooling1D()(x)
x = tf.keras.layers.Dense(64, activation="relu")(x)
x = tf.keras.layers.Dropout(dropout_rate)(x)
out = tf.keras.layers.Dense(1)(x)
model = tf.keras.Model(inp, out)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss=tf.keras.losses.Huber())
return model
```

### 步骤 4：`load_and_engineer_features()` 和 `build_sequences()`

- `load_and_engineer_features`：与 P0 扩展后的 GRU 脚本完全一致（11特征）
- `build_sequences`：直接复制 GRU 脚本的实现

### 步骤 5：`main()` 训练与评估

callbacks 与 GRU 相同：EarlyStopping(patience=15) + ReduceLROnPlateau(patience=8) + ModelCheckpoint。  
评估调用 `evaluate_and_save_run()`，`model_name="transformer"`。

### 运行验证

```bash
python models/transformer/train_transformer_8features.py --epochs 10 --d-model 32 --num-layers 1
# 预期：训练10个epoch，输出 run_dir + MAE
# 输出目录：output/runs/transformer_8features_<timestamp>/
```

---

## P2 — run_pipeline.py 扩展（P1 完成后）

在 `run_pipeline.py` 中增加对 xgboost 和 transformer 的支持：

```python
# 在 --model 参数的 choices 中加入：
choices=["gru", "lstm", "seq2seq_attention", "xgboost", "transformer"]

# 在模型分发逻辑中加入对应的训练函数调用
```

同时更新 `run_benchmark.py`，使其能汇总5个模型的 metrics.json 并输出对比表。

---

## 代码规范（必须遵守）

1. **输出目录结构**必须与 GRU 完全一致：
   ```
   output/runs/<model>_8features_<timestamp>/runs/run_<timestamp>_<model>_8features/
   ├── metrics.json      ← evaluate_and_save_run() 自动生成，不要手动创建
   ├── weights/
   └── figures/
   ```

2. **不要修改** `models/common/core_evaluation.py` 和 `models/common/evaluator.py`

3. **日期切分逻辑**：80/10/10 时序切分，不能打乱顺序，不能有数据泄露（scaler 只在训练集 fit）

4. **MinMaxScaler**：visitor_count 和 lag/rolling 特征使用独立 scaler，保存到 `models/scalers/` 或 run_dir

5. 所有 print 使用中文（与现有脚本风格一致）

---

## 完成标志

| 任务 | 验证命令 | 成功标准 |
|------|---------|---------|
| P0 特征扩展 | `python run_pipeline.py --model gru --features 11 --epochs 5` | 无报错，输出 MAE |
| P1-A XGBoost | `python models/xgboost/train_xgboost_8features.py --n-estimators 50` | 生成 output/runs/xgboost_*/metrics.json |
| P1-B Transformer | `python models/transformer/train_transformer_8features.py --epochs 5` | 生成 output/runs/transformer_*/metrics.json |
| P2 Pipeline | `python run_benchmark.py` | 打印5模型对比表（含 xgboost 和 transformer 行） |

所有任务完成后，运行完整训练：
```bash
python run_pipeline.py --model xgboost
python run_pipeline.py --model transformer --epochs 120
```
