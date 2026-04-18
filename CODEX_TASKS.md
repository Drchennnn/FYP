# Codex 任务书 v4 — 近三年数据集重训五模型

> **当前状态**（2026-04-18）：
> - ✅ 五个训练脚本已由 Claude 加入日期过滤 `df[df["date"] >= "2023-06-01"]`
> - ✅ 调参已在 v3 完成（lookback=45，dropout↓，patience=20，XGB n_est=500）
> - ❌ 五个模型尚未用新数据集重训
> - ❌ 对比报告尚未生成
>
> **背景**：全量10年数据训练集均值 9,044 vs 测试集均值 21,341，分布严重不匹配。
> 改用 2023-06-01 起的近三年数据（约 1,016 天连续记录），训练/测试分布对齐，
> 预期 MAE 可从 ~3,700 降至 ~2,000-2,500。
>
> **你的任务**：直接重训五个模型 → 跑对比报告

---

## 数据集信息（供参考）

```
原始 CSV：data/processed/jiuzhaigou_daily_features_2016-01-01_2026-04-02.csv
过滤后范围：2023-06-01 ~ 2026-04-16，约 1,016 天
  训练集（80%）：2023-06-01 ~ 2025-09-XX，约 812 天，mean≈15,604
  验证集（10%）：约 101 天
  测试集（10%）：约 101 天，mean≈12,160
缺失说明：2023-03~05 为冬季闭园，已通过 2023-06-01 起点自动跳过
```

---

## 任务 1：重训五个模型

**直接运行，不需要改任何脚本**（日期过滤已内置）：

```bash
# GRU（约10分钟）
python run_pipeline.py --model gru --features 8 --epochs 150

# LSTM（约15分钟）
python run_pipeline.py --model lstm --features 8 --epochs 150

# Seq2Seq（约20分钟）
python run_pipeline.py --model seq2seq_attention --epochs 150

# Transformer（约20分钟）
python models/transformer/train_transformer_8features.py

# XGBoost（约5分钟）
python models/xgboost/train_xgboost_8features.py
```

**预期训练样本数**：
- GRU/LSTM/Transformer：序列约 767 个（812 - lookback=45）
- Seq2Seq：编码器序列约 767 个，7步输出
- XGBoost：表格样本约 784 行（812 - lag_28=28）

---

## 任务 2：重跑五模型对比

```bash
python run_benchmark.py
```

生成：`output/runs/run_compare_<timestamp>/report.md`

---

## 完成验证

```bash
python -c "
import json, glob
models = {
    'gru':         'output/runs/gru_8features_*/runs/*/metrics.json',
    'lstm':        'output/runs/lstm_8features_*/runs/*/metrics.json',
    'seq2seq':     'output/runs/seq2seq_attention_8features_*/runs/*/metrics.json',
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
        n = m.get('n_samples', '?')
        ep = m.get('epochs_trained', '?')
        print(f'{name:12s} n={n:>4}  MAE={r[\"mae\"]:6.0f}  NRMSE={r.get(\"nrmse\",float(\"nan\")):.3f}  Recall={sw.get(\"recall\",\"?\")!s:>5}  epochs={ep}')
"
# 成功标准：
# - 所有模型 n_samples < 300（说明用的是近三年测试集，不是全量）
# - GRU/LSTM/Transformer MAE < 3000
# - Suit-Recall 全部 > 0.88
```

---

## 目标指标（近三年数据预期）

| 模型 | MAE 目标 | NRMSE 目标 | Suit-Recall 目标 |
|------|---------|-----------|----------------|
| Transformer | < 2,200 | < 9% | > 0.97 |
| GRU | < 2,500 | < 10% | > 0.92 |
| LSTM | < 2,800 | < 11% | > 0.92 |
| Seq2Seq | < 3,000 | — | > 0.95 |
| XGBoost | < 3,200 | < 13% | > 0.88 |
