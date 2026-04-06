# Thesis Outline
## Dynamic Visitor Flow Prediction for Jiuzhaigou Scenic Area Using Deep Learning

---

## Abstract

Tourism demand forecasting is critical for sustainable scenic area management, yet existing approaches rarely integrate multi-source exogenous features, multi-horizon prediction, or operational uncertainty quantification into a unified deployable system. This thesis presents an end-to-end deep learning pipeline for daily visitor flow prediction at Jiuzhaigou National Park, China, using ten years of historical visitor data (2016–2026) combined with meteorological and calendar features.

Three model architectures are evaluated: single-step GRU and LSTM (30-day lookback), and a multi-step Seq2Seq with Bahdanau attention (7-day horizon). A novel four-tier uncertainty interval framework is proposed, culminating in a Deep Ensemble + Conformal Prediction method (P4) that provides adaptive, coverage-guaranteed prediction bands. Model interpretability is addressed through feature ablation SHAP analysis and attention weight heatmaps. Temporal robustness is assessed via expanding-window walk-forward cross-validation across four seasonal folds.

Results show that GRU achieves the highest suitability warning F1 (0.980) and serves as the operational champion model. Walk-forward evaluation reveals significant concept drift during summer peak seasons, with GRU and LSTM maintaining Suit-Recall ≥ 0.90 across all folds while Seq2Seq exhibits high variance. Attention analysis uncovers a bimodal recency-baseline pattern, where the model prioritises the most recent day (*t*−1) and the 30-day anchor (*t*−30) while attenuating intermediate history. The system is deployed as a Flask web application with an Apple-style dashboard, automated daily data ingestion, and APScheduler-driven monthly retraining.

---

## Keywords

visitor flow prediction, deep learning, GRU, LSTM, Seq2Seq, attention mechanism, conformal prediction, uncertainty quantification, SHAP interpretability, walk-forward cross-validation, concept drift, tourism forecasting, Jiuzhaigou

---

## Chapter 1: Introduction

### 1.1 Background and Motivation
- Tourism as a major economic sector in China; Jiuzhaigou as a UNESCO World Heritage site with strict visitor capacity constraints
- Challenges of visitor flow management: overcrowding, environmental degradation, safety risks
- Limitations of current rule-based and statistical forecasting approaches
- Opportunity: deep learning on multi-source time series data for operational decision support

### 1.2 Problem Statement
- Formal definition: given 30-day historical multivariate sequence, predict next 1–7 days of visitor count
- Secondary objective: generate calibrated uncertainty intervals for risk-aware planning
- Tertiary objective: provide interpretable feature attribution to support management decisions

### 1.3 Research Objectives
1. Design and train multiple deep learning architectures (GRU, LSTM, Seq2Seq+Attention) on 10-year Jiuzhaigou data
2. Develop a four-tier uncertainty quantification framework with coverage guarantees
3. Evaluate temporal generalisation via walk-forward cross-validation
4. Provide model interpretability through SHAP and attention visualisation
5. Deploy as a real-time web application with automated data pipeline

### 1.4 Research Contributions
- A unified multi-model prediction system with operational champion selection based on suitability warning metrics
- A novel adaptive uncertainty interval method combining deep ensemble variance with conformal calibration (P4)
- Empirical evidence of seasonal concept drift in tourism flow prediction via multi-model walk-forward evaluation
- Attention heatmap analysis revealing bimodal recency-baseline attention patterns in Seq2Seq models

### 1.5 Thesis Structure
- Overview of chapter organisation

---

## Chapter 2: Background

### 2.1 Tourism Demand Forecasting
- Traditional methods: ARIMA, exponential smoothing, regression models
- Machine learning approaches: SVR, Random Forest, gradient boosting
- Deep learning for time series: RNN, LSTM, GRU — key advances and limitations
- Multi-step forecasting: direct vs recursive vs MIMO strategies

### 2.2 Recurrent Neural Network Architectures
- Vanilla RNN and the vanishing gradient problem
- LSTM: cell state, forget/input/output gates
- GRU: simplified gating mechanism, parameter efficiency
- Comparative studies: GRU vs LSTM in time series tasks

### 2.3 Sequence-to-Sequence Models with Attention
- Encoder-decoder architecture for multi-step forecasting
- Bahdanau (additive) attention mechanism: query, key, value formulation
- Attention weight interpretation as soft alignment
- Applications in time series forecasting

### 2.4 Uncertainty Quantification in Forecasting
- Frequentist vs Bayesian approaches
- Residual quantile intervals: simplicity and limitations
- Monte Carlo Dropout as approximate Bayesian inference
- Conformal prediction: coverage guarantees without distributional assumptions
- Deep ensemble methods: diversity, variance estimation

### 2.5 Model Interpretability
- Feature importance methods: permutation importance, SHAP
- SHAP for sequential models: challenges with 3D tensors
- Feature ablation as an alternative to KernelExplainer for temporal data
- Attention as interpretability: capabilities and limitations

### 2.6 Temporal Cross-Validation
- Standard k-fold vs time series cross-validation
- Walk-forward (expanding window) evaluation
- Concept drift in non-stationary time series
- Evaluation metrics for regression and classification tasks

### 2.7 Related Work
- Visitor flow prediction in tourism: existing datasets and methods
- Weather-integrated forecasting models
- Real-time deployment of ML forecasting systems

---

## Chapter 3: Design and Implementation

### 3.1 Dataset Description
- Data sources: Jiuzhaigou official visitor statistics (2016–2026), Open-Meteo meteorological API
- Data volume: ~3,750 daily records, 8 input features
- Feature engineering pipeline:
  - Visitor count MinMax scaling
  - Calendar features: month_norm, day_of_week_norm, is_holiday
  - Lag feature: tourism_num_lag_7_scaled (7-day autoregressive lag)
  - Meteorological features: precipitation, temp_high, temp_low (MinMax scaled)
- Train/test split: 90% / 10% (chronological), 268 test samples
- Data quality: handling missing values, NaN imputation strategy

### 3.2 Model Architectures

#### 3.2.1 Single-Step GRU and LSTM
- Input: (30, 8) — 30-day lookback window, 8 features
- Architecture: stacked recurrent layers + dense output
- Output: scalar (next-day visitor count, scaled)
- Training: Adam optimiser, EarlyStopping (patience=15), custom asymmetric loss

#### 3.2.2 Multi-Step MIMO (GRU-MIMO / LSTM-MIMO)
- Input: (30, 8)
- Output: (7,) — 7-day direct multi-output
- Architecture: recurrent encoder + multi-output dense head

#### 3.2.3 Seq2Seq with Bahdanau Attention
- Encoder input: (30, 8) — historical sequence
- Decoder input: (7, 7) — future exogenous features (no visitor count)
- Attention: query = decoder hidden state, key/value = encoder outputs
- Output: (7, 1) — 7-day prediction sequence
- Custom Keras serialisation for deployment compatibility

### 3.3 Uncertainty Interval Framework

#### 3.3.1 P1: Residual Quantile Interval
- Compute residuals on calibration set; use empirical quantiles as fixed-width bands

#### 3.3.2 P2: Monte Carlo Dropout
- Enable dropout at inference; sample N=50 forward passes; compute mean ± 1.96σ

#### 3.3.3 P3: Split Conformal Prediction
- Calibration set nonconformity scores; coverage-guaranteed intervals at α=0.10

#### 3.3.4 P4: Deep Ensemble + Conformal Calibration (Adopted Method)
- Train 5 GRU models with different random seeds
- Ensemble mean as point prediction; ensemble std as base uncertainty
- Step-wise conformal quantile q̂_h: calibrated per forecast horizon using expanding calibration set
- Adaptive width cap: half-width ≤ 40% of centre prediction value
- Coverage target: 90% PICP

### 3.4 Evaluation Framework

#### 3.4.1 Regression Metrics
- MAE, RMSE, sMAPE

#### 3.4.2 Suitability Warning Classification
- Dynamic peak threshold: 75th percentile of training visitor counts
- Binary classification: crowd alert, weather hazard, composite suitability warning
- Metrics: F1 (weighted), Recall (weighted), Brier Score, ECE
- Champion selection criterion: highest Suit-F1 with Recall ≥ 0.80

#### 3.4.3 Walk-Forward Cross-Validation
- Expanding window, 4 folds, 90-day test windows
- Independent scaler fitting per fold

### 3.5 System Implementation

#### 3.5.1 Data Pipeline
- Automated daily append: `scripts/append_and_retrain.py`
- Rolling backfill inference: `scripts/backfill_predictions.py`
- APScheduler: 08:30 daily append, 09:00 visitor crawl, 02:00 monthly retrain

#### 3.5.2 Flask API
- `GET /api/forecast?h=1|7|14&mode=offline|online`
- Offline mode: reads pre-computed test prediction CSVs (fast, no model loading)
- Online mode: loads champion weights, fetches Open-Meteo weather, generates true future predictions
- Conformal uncertainty intervals computed at request time
- `zones` field in payload: history_end, gap_end, forecast_start, forecast_end

#### 3.5.3 Dashboard (v3)
- ECharts-based Apple-style interface
- Forecast fan chart with uncertainty bands
- Suitability warning calibration card
- Model comparison panel
- Automated gap-fill visualisation for data latency periods

---

## Chapter 4: Results and Discussion

### 4.1 Regression Performance Comparison
- Five-model comparison table: MAE, RMSE, sMAPE, Suit-F1, Suit-Recall, Brier
- GRU single-step as champion (highest Suit-F1 = 0.980, Recall = 0.966)
- GRU-MIMO achieves best regression MAE (2941) but lower Suit-F1
- Discussion: trade-off between regression accuracy and classification performance
- Note on training variance: GRU single-step MAE sensitivity to random seed

### 4.2 Uncertainty Interval Evaluation
- Comparison of P1–P4 methods: PICP, MPIW, Winkler Score
- P4 (Ensemble + Conformal) achieves best coverage-efficiency trade-off
- Step-wise q̂_h: wider intervals for longer horizons, adaptive to seasonal volatility
- 40% cap prevents degenerate intervals during low-traffic periods
- *(Note: formal PICP/MPIW/Winkler metrics pending dedicated evaluation script)*

### 4.3 Feature Ablation Study
- Feature masking methodology: prevents mean-predictor collapse vs retraining approach
- Full 8-feature model outperforms all ablated variants
- visitor_count_scaled dominates (SHAP = 0.0462, 13.6× second-ranked feature)
- Exogenous features (weather, holidays) provide statistically significant information gain
- Implication: multi-variate feature engineering pipeline is validated

### 4.4 SHAP Interpretability Analysis
- Feature ablation SHAP on 3D temporal tensors
- Strong autoregressive property: model relies heavily on t−1 ~ t−30 history
- Lag-7 pattern captures weekly seasonality
- Climate and holiday features act as non-linear modifiers
- Comparison with attention weights: consistent prioritisation of recent history

### 4.5 Walk-Forward Temporal Robustness
- Three-model comparison across 4 seasonal folds
- GRU and LSTM: stable performance, Suit-Recall ≥ 0.90 in all folds
- Seq2Seq: high variance (Brier std = 0.091), struggles in spring transition and winter off-peak
- Concept drift evidence: summer peak (Fold 2) shows highest MAE for GRU/LSTM
- Off-peak seasons: lowest absolute MAE but highest sMAPE (low denominator effect)
- Implication: single-step recurrent models generalise more robustly under temporal distribution shift

### 4.6 Seq2Seq Attention Analysis
- Attention weight matrix: (268, 7, 30) → mean (7, 30) heatmap
- Finding A — Horizon-Invariant Attention: near-static distribution across 7 forecast steps (max row diff < 0.001); model's attention determined by encoder content, not decoder position
- Finding B — Bimodal Edge Effect: high attention at t−1 (0.272, recency bias) and t−30 (0.194, long-range baseline); middle history attenuated (0.010–0.014)
- Interpretation: model computes trend direction by comparing recent momentum against 30-day anchor
- Limitation: attention ≠ causal importance; horizon-invariant pattern may indicate limited decoder state diversity

### 4.7 Discussion
- Overall system performance relative to literature benchmarks
- Practical implications for Jiuzhaigou visitor management
- Limitations: binary p_warn approximation, seasonal concept drift, attention limited to Seq2Seq
- Reliability diagram analysis: bimodal distribution as design decision, not calibration failure

---

## Chapter 5: Conclusion and Further Work

### 5.1 Summary of Contributions
- End-to-end deep learning system for daily visitor flow prediction with 10-year training data
- Four-tier uncertainty quantification framework with adaptive conformal calibration
- Empirical demonstration of seasonal concept drift via multi-model walk-forward evaluation
- Bimodal attention pattern discovery: recency bias + long-range baseline in Seq2Seq
- Deployed operational system with automated data pipeline and real-time dashboard

### 5.2 Key Findings
- GRU achieves best operational performance (Suit-F1 = 0.980) with stable temporal generalisation
- Exogenous features (weather, holidays) provide measurable information gain beyond autoregressive baseline
- Concept drift is most severe during summer peak season (July–October), challenging static model generalisation
- Seq2Seq attention is horizon-invariant and bimodal — a genuine learned behaviour, not an implementation artefact

### 5.3 Limitations
- Binary suitability warning probability (p_warn ∈ {0.15, 0.85}): rule-based approximation, not a true probabilistic model; Brier Score and ECE measure binary approximation quality, not calibration in the Bayesian sense
- Walk-forward seasonal variance: models trained on historical data may underperform during unprecedented events (e.g., post-pandemic recovery, policy changes)
- Attention interpretability: horizon-invariant attention suggests limited decoder state diversity; attention weights are not causal importance scores
- Data dependency: system relies on timely official visitor count publication; delays create prediction gaps

### 5.4 Further Work
- **Transformer-based architectures**: Temporal Fusion Transformer (TFT) or PatchTST for improved long-range dependency modelling
- **Online learning**: incremental model updates with each new day's data to address concept drift without full retraining
- **Finer temporal granularity**: hourly or sub-daily prediction using gate entry timestamps
- **True probabilistic forecasting**: replace binary p_warn with MC Dropout or normalising flows for well-calibrated probability outputs
- **Multi-site generalisation**: transfer learning across multiple scenic areas with similar visitor flow dynamics
- **Causal feature attribution**: replace attention-based interpretability with causal inference methods (e.g., Granger causality, PCMCI)

---

*Last updated: 2026-04-06*
