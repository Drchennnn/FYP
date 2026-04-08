# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

九寨沟景区客流动态预测系统 (Jiuzhaigou Visitor Flow Prediction System) — an end-to-end ML pipeline with three models (GRU, LSTM, Seq2Seq+Attention), a Flask API, and an Apple-style dashboard (v3).

## Commands

**Run the web app:**
```bash
cd web_app && python app.py
# Dashboard: http://localhost:5000/dashboard/v3
```

**Train models:**
```bash
python run_pipeline.py --model gru --features 8 --epochs 120
python run_pipeline.py --model lstm --features 8 --epochs 120
python run_pipeline.py --model seq2seq_attention --epochs 120
python run_benchmark.py  # compare all models
```

**Data pipeline:**
```bash
python scripts/append_and_retrain.py --append          # auto-detect gap, backfill to yesterday (1 day or many)
python scripts/append_and_retrain.py --append --date YYYY-MM-DD  # backfill to specific date
python scripts/append_and_retrain.py --retrain         # retrain all 3 models
python scripts/append_and_retrain.py --append --retrain
python scripts/backfill_predictions.py                 # fill historical prediction gaps (rolling inference)
```

**Evaluation:**
```bash
python scripts/walk_forward_eval.py --model gru --folds 4
python scripts/ablation_study.py                    # feature-masking, loads trained GRU weights (~30s)
python scripts/shap_analysis.py --model gru
```

## Architecture

### Data Flow

```
Raw (crawler + Open-Meteo) → data/processed/jiuzhaigou_daily_features_*.csv
                           → models/scalers/feature_scalers.pkl
                           → run_pipeline.py (train)
                           → output/runs/<model>_8features_<timestamp>/
                           → web_app/app.py (serve)
                           → /api/forecast → dashboard_v3.js
```

### 8 Input Features (all MinMax scaled)
`visitor_count_scaled`, `month_norm`, `day_of_week_norm`, `is_holiday`, `tourism_num_lag_7_scaled`, `meteo_precip_sum_scaled`, `temp_high_scaled`, `temp_low_scaled`

### Three Models

| Model | Type | Input | Output | Weights format |
|-------|------|-------|--------|----------------|
| GRU | Single-step | (30, 8) | scalar | `.h5` |
| LSTM | Single-step | (30, 8) | scalar | `.h5` |
| Seq2Seq+Attention | Multi-step | encoder (30,8) + decoder (7,7) | (7,1) | `.keras` |

Champion is selected at runtime by `_pick_champion_and_runner_up()` — highest `suitability_warning_f1_weighted` with `recall_weighted ≥ 0.80`. Currently: GRU=champion, Seq2Seq=runner_up, LSTM=third.

### Flask API (`web_app/app.py`)

Key endpoint: `GET /api/forecast?h=1|7|14&mode=offline|online&include_all=0|1`

**Offline mode**: reads `*_test_predictions.csv` artifacts from `output/runs/`. Fast, no model loading.

**Online mode**: loads model weights → fetches Open-Meteo weather → generates true future predictions.
- GRU/LSTM: rolling single-step (30-step lookback window, 7 iterations)
- Seq2Seq: one-shot 7-day prediction with encoder+decoder inputs

Model discovery: `_synthesise_compare_metrics()` scans `output/runs/` for latest run per model type, reads `metrics.json`, builds comparison DataFrame. `_resolve_backup_run_dir()` resolves run paths (handles relative and absolute).

### Seq2Seq Custom Objects

The Seq2Seq model requires `@tf.keras.utils.register_keras_serializable(package='Custom')` on both `AttentionLayer` and `Seq2SeqWithAttention` in `models/lstm/train_seq2seq_attention_8features.py`. These are imported at Flask startup into `_seq2seq_custom_objects` dict (with multiple key aliases: `custom_asymmetric_loss`, `custom_loss`, `CustomAsymmetricLoss`).

### Frontend (`web_app/static/js/dashboard_v3.js`)

ECharts-based dashboard. Cache key: `v3_forecast_v2_${mode}_h${h}` (30-min TTL for online, no TTL for offline). Weather fallback: searches ±3 days for nearest valid weather data when a date has null weather (e.g. boundary dates outside Open-Meteo forecast window).

### Automated Scheduler (APScheduler in app.py)
- 08:30 daily: append yesterday's data
- 09:00 daily: crawl visitor count
- 02:00 on 1st of month: retrain all models

## Key File Locations

- Processed training data: `data/processed/jiuzhaigou_daily_features_2016-01-01_2026-04-02.csv`
- Feature scalers: `models/scalers/feature_scalers.pkl`
- Model runs: `output/runs/<model>_8features_<timestamp>/runs/run_<timestamp>/`
  - `metrics.json` — full evaluation including `suitability_warning_weighted`
  - `weights/*.keras` or `weights/*.h5` — model weights
  - `*_test_predictions.csv` — backtest predictions
- Champion selection logic: `web_app/app.py:_pick_champion_and_runner_up()`
- Core evaluation metrics: `models/common/core_evaluation.py`
- Feature engineering: `models/common/preprocess.py`

## Known Issues / Ongoing Work

- **Attention heatmap**: No `attention_heatmap*.png` files exist yet — need to re-run Seq2Seq inference on test set to generate them.
- **Walk-forward data**: `scripts/walk_forward_eval.py` has never been run; `walk_forward` key is absent from `metrics.json`. Model Analysis page in dashboard shows empty skeleton.
- **GRU/LSTM `model_architecture`**: Only Seq2Seq has this field in `metrics.json['meta']`.
- **Backfill script**: `scripts/backfill_predictions.py` extends GRU/LSTM prediction CSVs via rolling inference (not retraining). Run after long gaps in offline predictions.

---

## Academic Research Skills

Skills for thesis writing, literature research, peer review, and pipeline orchestration. Located in `.claude/skills/`.

| Skill | Purpose | Key Modes |
|-------|---------|-----------|
| `deep-research` | 13-agent research team | full, quick, socratic, lit-review, fact-check, systematic-review |
| `academic-paper` | 12-agent paper writing | full, plan, outline, revision, citation-check, bilingual-abstract |
| `academic-paper-reviewer` | Multi-perspective review (5 reviewers) | full, re-review, quick, methodology-focus, guided |
| `academic-pipeline` | Full pipeline orchestrator | coordinates all above |

### Recommended Flow for This Thesis

```
deep-research (lit-review)          ← Chapter 2 literature
  → academic-paper (plan/outline)   ← Chapter structure
    → academic-paper (full)         ← Draft chapters
      → academic-paper-reviewer     ← Self-review
        → academic-paper (revision) ← Revise
```

### Routing Rules

- Use **academic-pipeline** for end-to-end runs; use individual skills for single tasks.
- Use **deep-research socratic** when the research question is unclear; **full** for direct output.
- Use **academic-paper plan** to think through structure; **full** to produce draft directly.
- Use **academic-paper-reviewer guided** to learn from review; **full** for a standard report.

### Key Rules

- All claims must have citations; evidence hierarchy respected (meta-analyses > RCTs > cohort > expert opinion).
- Default output language matches user input (Simplified Chinese or English).
- AI disclosure included in all reports.
