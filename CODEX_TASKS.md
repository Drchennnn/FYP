# Codex 任务书 v13b — Transformer 重训（look_back=45，保存 history CSV）

> **背景**（2026-04-20）：
> - 上次重训（20260420_155113）因 look_back 默认为 30，与原始训练（look_back=45）不一致，
>   导致 MAE 变差（3343 vs 3044），需废弃。
> - 本次任务：用正确超参 look_back=45 重训，同时确认 history CSV 已保存（上次已修改脚本）。

---

## 任务：用正确超参重训 Transformer

### 步骤 1：确认训练脚本已有 history 保存代码

检查 `models/transformer/train_transformer_8features.py` 第 437 行附近是否已有以下代码：

```python
    history_df = pd.DataFrame(history.history)
    history_df.insert(0, "epoch", range(1, len(history_df) + 1))
    history_df.to_csv(run_dir / "transformer_history.csv", index=False, encoding="utf-8-sig")
```

如果没有，先插入（参考上次任务）。

### 步骤 2：重训（look_back=45）

```bash
cd e:/openclaw/my_project/workspace/FYP
python run_pipeline.py --model transformer --features 8 --epochs 150 --look-back 45
```

### 验证

训练完成后确认：

1. 新 run 目录存在：
```
output/runs/transformer_8features_<新timestamp>/runs/run_<timestamp>/
```

2. history CSV 存在且有 epoch/loss/val_loss 列：
```
output/runs/transformer_8features_<新timestamp>/runs/run_<timestamp>/transformer_history.csv
```

3. metrics.json 里 `meta.look_back == 45`，MAE 应在 2900~3200 范围内（与旧 run 3044 接近）：
```bash
python -c "
import json, glob, os
top = max(glob.glob('output/runs/transformer_8features_*'), key=os.path.getmtime)
subs = glob.glob(os.path.join(top, 'runs', 'run_*'))
run_dir = max(subs, key=os.path.getmtime)
m = json.load(open(os.path.join(run_dir, 'metrics.json')))
print('look_back:', m['meta']['look_back'])
print('MAE:', m['regression']['mae'])
print('history exists:', os.path.exists(os.path.join(run_dir, 'transformer_history.csv')))
"
```

期望输出：
```
look_back: 45
MAE: ~3000（±300）
history exists: True
```
