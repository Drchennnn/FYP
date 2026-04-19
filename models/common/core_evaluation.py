#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Core evaluation + unified output artifacts.

This module standardizes metric computation and plot generation across:
- LSTM (single-step)
- GRU (single-step)
- Seq2Seq + Attention (multi-horizon)

All plots must be English-only.

Output structure (per run directory):

  output/runs/<run>/
    metrics.json
    metrics.csv
    metrics_by_horizon.csv   (only when horizon > 1)
    figures/
      true_vs_pred.png
      error_by_horizon.png   (only when horizon > 1)
      confusion_matrix_crowd_alert.png
      suitability_warning_timeline.png
      reliability_diagram.png
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
)


# 官方承载量上限：旺季（4/1~11/15）41000人，淡季（11/16~3/31）23000人
# 预警阈值取上限的80%：旺季32800，淡季18400
CAPACITY_PEAK_SEASON = 41000
CAPACITY_OFF_SEASON  = 23000
THRESHOLD_RATIO      = 0.80
PEAK_THRESHOLD_PEAK  = int(CAPACITY_PEAK_SEASON * THRESHOLD_RATIO)  # 32800
PEAK_THRESHOLD_OFF   = int(CAPACITY_OFF_SEASON  * THRESHOLD_RATIO)  # 18400
DEFAULT_PEAK_THRESHOLD = PEAK_THRESHOLD_PEAK  # 向后兼容，默认取旺季值


def get_season_peak_threshold(query_date=None) -> float:
    """根据日期返回对应季节的预警阈值（官方承载量上限×80%）。

    旺季（4/1~11/15）: 32800  淡季（11/16~3/31）: 18400
    """
    if query_date is None:
        return float(PEAK_THRESHOLD_PEAK)
    from datetime import date as _d
    if isinstance(query_date, str):
        query_date = _d.fromisoformat(query_date)
    m, d = query_date.month, query_date.day
    is_peak = (4 <= m <= 10) or (m == 11 and d <= 15)
    return float(PEAK_THRESHOLD_PEAK if is_peak else PEAK_THRESHOLD_OFF)


# Default horizon weights from README (Section 7.4.0), used for H=7.
DEFAULT_HORIZON_WEIGHTS_H7 = [0.28, 0.20, 0.15, 0.12, 0.10, 0.08, 0.07]


# Route B (data-driven) default quantiles for weather hazard thresholds.
# We treat hazard as: precip >= q_precip_high OR temp_high >= q_temp_high OR temp_low <= q_temp_low
DEFAULT_WEATHER_QUANTILES = {
    "precip_high": 0.90,
    "temp_high": 0.90,
    "temp_low": 0.10,
}


def compute_dynamic_peak_threshold(
    train_visitor_counts: np.ndarray,
    quantile: float = 0.75,
    fallback: float = DEFAULT_PEAK_THRESHOLD,
) -> float:
    """返回旺季预警阈值（官方承载量上限×80%=32800）。

    训练时无法区分预测日期，统一用旺季值；运行时由 get_season_peak_threshold() 按日期动态调整。
    """
    return float(PEAK_THRESHOLD_PEAK)


def _nanquantile(x: np.ndarray, q: float) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(np.quantile(x, q))


def compute_weather_thresholds_quantile(
    *,
    train_precip: np.ndarray,
    train_temp_high: np.ndarray,
    train_temp_low: np.ndarray,
    quantiles: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Compute Route B weather hazard thresholds from TRAIN split only.

    All inputs must be in real units (NOT scaled). This avoids leakage by
    computing quantiles only on the training window.
    """

    q = dict(DEFAULT_WEATHER_QUANTILES)
    if quantiles:
        q.update({k: float(v) for k, v in quantiles.items()})

    thresholds = {
        "method": "quantile_route_b",
        "quantiles": q,
        "precip_high": _nanquantile(train_precip, q["precip_high"]),
        "temp_high": _nanquantile(train_temp_high, q["temp_high"]),
        "temp_low": _nanquantile(train_temp_low, q["temp_low"]),
    }
    return thresholds


def compute_weather_hazard(
    *,
    precip: np.ndarray,
    temp_high: np.ndarray,
    temp_low: np.ndarray,
    thresholds: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute weather hazard binary label and a simple severity score.

    Returns:
      - weather_hazard_bin: 0/1
      - weather_hazard_severity: integer in [0, 3] (# triggered conditions)

    Shapes follow broadcasting rules; caller should pass arrays aligned to y.
    """

    precip = np.asarray(precip, dtype=float)
    temp_high = np.asarray(temp_high, dtype=float)
    temp_low = np.asarray(temp_low, dtype=float)

    thr_p = float(thresholds.get("precip_high", float("nan")))
    thr_th = float(thresholds.get("temp_high", float("nan")))
    thr_tl = float(thresholds.get("temp_low", float("nan")))

    cond_p = np.isfinite(precip) & np.isfinite(thr_p) & (precip >= thr_p)
    cond_th = np.isfinite(temp_high) & np.isfinite(thr_th) & (temp_high >= thr_th)
    cond_tl = np.isfinite(temp_low) & np.isfinite(thr_tl) & (temp_low <= thr_tl)

    severity = cond_p.astype(int) + cond_th.astype(int) + cond_tl.astype(int)
    hazard = (severity > 0).astype(int)
    return hazard, severity


def _to_1d(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    return a.reshape(-1)


def _smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    y_true = _to_1d(y_true).astype(float)
    y_pred = _to_1d(y_pred).astype(float)
    denom = np.maximum(np.abs(y_true) + np.abs(y_pred), eps)
    return float(100.0 * np.mean(2.0 * np.abs(y_pred - y_true) / denom))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = _to_1d(y_true).astype(float)
    y_pred = _to_1d(y_pred).astype(float)
    return float(np.mean(np.abs(y_pred - y_true)))


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = _to_1d(y_true).astype(float)
    y_pred = _to_1d(y_pred).astype(float)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def _nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = _to_1d(y_true).astype(float)
    y_pred = _to_1d(y_pred).astype(float)
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    value_range = float(y_true.max() - y_true.min())
    return rmse / value_range if value_range > 0 else float("nan")


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    # numerically stable sigmoid
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    return out


def visitor_count_to_warning_prob(
    y_pred_count: np.ndarray,
    peak_threshold: float,
    temperature: float = 1000.0,
) -> np.ndarray:
    """Convert point forecasts into a pseudo-probability of 'suitability_warning'.

    Since our models are deterministic, we derive probabilities via a calibrated
    logistic transform around the peak threshold.

    temperature controls the softness of the transition.
    """

    y_pred_count = _to_1d(y_pred_count)
    z = (y_pred_count - float(peak_threshold)) / float(temperature)
    return _sigmoid(z)


def brier_score(y_true_bin: np.ndarray, y_prob: np.ndarray) -> float:
    y_true_bin = _to_1d(y_true_bin).astype(float)
    y_prob = _to_1d(y_prob).astype(float)
    return float(np.mean((y_prob - y_true_bin) ** 2))


def expected_calibration_error(
    y_true_bin: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Tuple[float, pd.DataFrame]:
    """ECE with equal-width bins.

    Returns (ece, bin_table).
    """

    y_true_bin = _to_1d(y_true_bin).astype(int)
    y_prob = _to_1d(y_prob).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    # right-inclusive last bin
    bin_ids = np.digitize(y_prob, bins, right=False) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    rows = []
    total = len(y_prob)
    ece = 0.0
    for b in range(n_bins):
        mask = bin_ids == b
        cnt = int(np.sum(mask))
        if cnt == 0:
            rows.append(
                {
                    "bin": b,
                    "bin_left": float(bins[b]),
                    "bin_right": float(bins[b + 1]),
                    "count": 0,
                    "avg_confidence": float("nan"),
                    "avg_accuracy": float("nan"),
                }
            )
            continue

        conf = float(np.mean(y_prob[mask]))
        acc = float(np.mean(y_true_bin[mask]))
        w = cnt / total
        ece += w * abs(acc - conf)
        rows.append(
            {
                "bin": b,
                "bin_left": float(bins[b]),
                "bin_right": float(bins[b + 1]),
                "count": cnt,
                "avg_confidence": conf,
                "avg_accuracy": acc,
            }
        )

    return float(ece), pd.DataFrame(rows)


def _clf_prf(y_true_bin: np.ndarray, y_pred_bin: np.ndarray) -> Dict[str, float]:
    p, r, f1, _ = precision_recall_fscore_support(
        y_true_bin, y_pred_bin, average="binary", zero_division=0
    )
    tn = int(np.sum((y_true_bin == 0) & (y_pred_bin == 0)))
    fp = int(np.sum((y_true_bin == 0) & (y_pred_bin == 1)))
    fn = int(np.sum((y_true_bin == 1) & (y_pred_bin == 0)))
    tp = int(np.sum((y_true_bin == 1) & (y_pred_bin == 1)))
    return {"precision": float(p), "recall": float(r), "f1": float(f1),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def compute_core_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    peak_threshold: float = DEFAULT_PEAK_THRESHOLD,
    dates: Optional[np.ndarray] = None,   # 若提供，按日期动态取旺淡季阈值
    horizon: Optional[int] = None,
    warning_temperature: float = 1000.0,
    horizon_weights: Optional[Iterable[float]] = None,
    fn_fp_cost_ratio: Tuple[float, float] = (5.0, 1.0),
    # --- Weather hazard (Route B: train-quantile thresholds) ---
    # If provided, these arrays must be in real units and aligned with y_true/y_pred
    # (shape (N,) for single-step, shape (N,H) for multi-horizon).
    weather_precip: Optional[np.ndarray] = None,
    weather_temp_high: Optional[np.ndarray] = None,
    weather_temp_low: Optional[np.ndarray] = None,
    # Provide ONE of:
    #   - weather_thresholds: precomputed thresholds (recommended for walk-forward)
    #   - weather_train_* arrays: training window used to compute quantile thresholds
    weather_thresholds: Optional[Dict[str, Any]] = None,
    weather_train_precip: Optional[np.ndarray] = None,
    weather_train_temp_high: Optional[np.ndarray] = None,
    weather_train_temp_low: Optional[np.ndarray] = None,
    weather_quantiles: Optional[Dict[str, float]] = None,
) -> Tuple[Dict[str, Any], Optional[pd.DataFrame]]:
    """Compute the unified core metrics.

    y_true/y_pred can be:
      - shape (N,) for single-step
      - shape (N, H) for multi-horizon

    If horizon is provided, it must match y_true.shape[1] when y_true is 2D.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    by_horizon_df: Optional[pd.DataFrame] = None

    # Determine horizon
    if y_true.ndim == 2:
        H = int(y_true.shape[1])
        if horizon is not None and int(horizon) != H:
            raise ValueError(f"horizon mismatch: horizon={horizon} vs y_true.shape[1]={H}")
        horizon = H
    else:
        horizon = int(horizon or 1)

    # Overall regression metrics (flatten)
    y_true_flat = _to_1d(y_true)
    y_pred_flat = _to_1d(y_pred)

    reg = {
        "mae": _mae(y_true_flat, y_pred_flat),
        "rmse": _rmse(y_true_flat, y_pred_flat),
        "nrmse": _nrmse(y_true_flat, y_pred_flat),
        "smape": _smape(y_true_flat, y_pred_flat),
    }

    # crowd_alert + peak_mask — 按日期取旺淡季阈值（若提供 dates），否则用固定阈值
    if dates is not None and len(dates) > 0:
        import pandas as _pd
        _dates_flat = np.asarray(dates).flatten()
        _repeat = len(y_true_flat) // len(_dates_flat)
        if _repeat > 1:
            _dates_flat = np.repeat(_dates_flat, _repeat)
        _thr_arr = np.array([
            get_season_peak_threshold(_pd.Timestamp(d).date() if not hasattr(d, 'month') else d)
            for d in _dates_flat[:len(y_true_flat)]
        ], dtype=float)
        y_true_peak = (y_true_flat >= _thr_arr).astype(int)
        y_pred_peak = (y_pred_flat >= _thr_arr).astype(int)
        peak_mask = y_true_flat >= _thr_arr
    else:
        _thr_arr = None
        y_true_peak = (y_true_flat >= float(peak_threshold)).astype(int)
        y_pred_peak = (y_pred_flat >= float(peak_threshold)).astype(int)
        peak_mask = y_true_flat >= float(peak_threshold)
    crowd_alert = _clf_prf(y_true_peak, y_pred_peak)

    # Peak-only MAE
    if np.any(peak_mask):
        peak_mae = _mae(y_true_flat[peak_mask], y_pred_flat[peak_mask])
    else:
        peak_mae = float("nan")

    # --- Weather hazard (optional) ---
    weather_hazard_bin_flat = np.zeros_like(y_true_peak, dtype=int)
    weather_hazard_sev_flat = np.zeros_like(y_true_peak, dtype=int)
    weather_meta: Dict[str, Any] = {
        "enabled": False,
        "assumption": (
            "If the model does not predict weather, weather inputs are treated as exogenous "
            "(forecast/observed from dataset) and used directly for hazard."
        ),
    }

    weather_inputs_provided = (
        weather_precip is not None and weather_temp_high is not None and weather_temp_low is not None
    )
    if weather_inputs_provided:
        # thresholds: prefer explicitly provided thresholds; else compute from TRAIN arrays
        thr = weather_thresholds
        if thr is None:
            if (
                weather_train_precip is not None
                and weather_train_temp_high is not None
                and weather_train_temp_low is not None
            ):
                thr = compute_weather_thresholds_quantile(
                    train_precip=weather_train_precip,
                    train_temp_high=weather_train_temp_high,
                    train_temp_low=weather_train_temp_low,
                    quantiles=weather_quantiles,
                )
            else:
                # Safety default: no thresholds => disable hazard (avoid leakage).
                thr = None

        if thr is not None:
            wh, sev = compute_weather_hazard(
                precip=_to_1d(np.asarray(weather_precip)),
                temp_high=_to_1d(np.asarray(weather_temp_high)),
                temp_low=_to_1d(np.asarray(weather_temp_low)),
                thresholds=thr,
            )
            weather_hazard_bin_flat = wh.astype(int)
            weather_hazard_sev_flat = sev.astype(int)
            weather_meta = {
                "enabled": True,
                "thresholds": {
                    "method": thr.get("method"),
                    "quantiles": thr.get("quantiles"),
                    "precip_high": thr.get("precip_high"),
                    "temp_high": thr.get("temp_high"),
                    "temp_low": thr.get("temp_low"),
                },
                "prevalence": float(np.mean(weather_hazard_bin_flat)) if len(weather_hazard_bin_flat) else float("nan"),
                "assumption": (
                    "If the model does not predict weather, weather inputs are treated as exogenous "
                    "(forecast/observed from dataset) and used directly for hazard."
                ),
            }

    # suitability_warning = crowd_alert OR weather_hazard
    suitability_true = ((y_true_peak == 1) | (weather_hazard_bin_flat == 1)).astype(int)

    # probabilistic warning: 用日期阈值数组（若有）或固定阈值
    if _thr_arr is not None:
        p_crowd = np.array([
            visitor_count_to_warning_prob(np.array([yp]), peak_threshold=thr, temperature=float(warning_temperature))[0]
            for yp, thr in zip(y_pred_flat, _thr_arr)
        ])
    else:
        p_crowd = visitor_count_to_warning_prob(
            y_pred_flat, peak_threshold=float(peak_threshold), temperature=float(warning_temperature)
        )
    p_weather = weather_hazard_bin_flat.astype(float)  # deterministic (0 or 1)
    y_prob_warn = 1.0 - (1.0 - p_crowd) * (1.0 - p_weather)
    y_pred_warn = (y_prob_warn >= 0.5).astype(int)
    suitability_warning = _clf_prf(suitability_true, y_pred_warn)

    # auxiliary business-cost view (default FN:FP = 5:1)
    fn_cost, fp_cost = float(fn_fp_cost_ratio[0]), float(fn_fp_cost_ratio[1])
    fn = float(np.sum((suitability_true == 1) & (y_pred_warn == 0)))
    fp = float(np.sum((suitability_true == 0) & (y_pred_warn == 1)))
    expected_cost = (
        (fn_cost * fn + fp_cost * fp) / float(len(suitability_true))
        if len(suitability_true)
        else float("nan")
    )

    brier = brier_score(suitability_true, y_prob_warn)
    ece, ece_table = expected_calibration_error(suitability_true, y_prob_warn, n_bins=10)

    # Per-horizon regression metrics
    if np.asarray(y_true).ndim == 2 and horizon and horizon > 1:
        rows = []
        for h in range(horizon):
            rows.append(
                {
                    "horizon": h + 1,
                    "mae": _mae(y_true[:, h], y_pred[:, h]),
                    "rmse": _rmse(y_true[:, h], y_pred[:, h]),
                    "nrmse": _nrmse(y_true[:, h], y_pred[:, h]),
                    "smape": _smape(y_true[:, h], y_pred[:, h]),
                }
            )
        by_horizon_df = pd.DataFrame(rows)

    # Per-horizon warning metrics + weighted aggregation (for multi-horizon models)
    warning_by_h_df: Optional[pd.DataFrame] = None
    warning_weighted: Optional[Dict[str, float]] = None
    if y_true.ndim == 2 and horizon and horizon > 1:
        if horizon_weights is None:
            if int(horizon) == 7:
                horizon_weights = DEFAULT_HORIZON_WEIGHTS_H7
            else:
                horizon_weights = [1.0 / float(horizon)] * int(horizon)

        w = np.asarray(list(horizon_weights), dtype=float)
        if w.shape[0] != int(horizon):
            raise ValueError(
                f"horizon_weights length mismatch: len={len(w)} vs horizon={horizon}"
            )
        if not np.isfinite(w).all():
            raise ValueError("horizon_weights contains non-finite values")
        if float(np.sum(w)) <= 0:
            raise ValueError("horizon_weights must sum to a positive value")
        w = w / float(np.sum(w))

        rows = []
        f1s = []
        recalls = []
        briers = []
        eces = []
        costs = []

        fn_cost, fp_cost = float(fn_fp_cost_ratio[0]), float(fn_fp_cost_ratio[1])

        # Prepare weather hazard matrix (optional)
        wh_mat: Optional[np.ndarray] = None
        if (
            weather_inputs_provided
            and weather_thresholds is not None
            and np.asarray(weather_precip).ndim == 2
        ):
            wh_mat, _ = compute_weather_hazard(
                precip=np.asarray(weather_precip),
                temp_high=np.asarray(weather_temp_high),
                temp_low=np.asarray(weather_temp_low),
                thresholds=weather_thresholds,
            )
        elif (
            weather_inputs_provided
            and weather_thresholds is None
            and weather_meta.get("enabled")
            and np.asarray(weather_precip).ndim == 2
        ):
            # thresholds were computed above into weather_meta
            thr2 = weather_meta.get("thresholds") or {}
            wh_mat, _ = compute_weather_hazard(
                precip=np.asarray(weather_precip),
                temp_high=np.asarray(weather_temp_high),
                temp_low=np.asarray(weather_temp_low),
                thresholds=thr2,
            )

        for h in range(int(horizon)):
            yt_peak = (y_true[:, h] >= float(peak_threshold)).astype(int)
            yt_weather = (
                wh_mat[:, h].astype(int) if wh_mat is not None else np.zeros_like(yt_peak)
            )
            yt = ((yt_peak == 1) | (yt_weather == 1)).astype(int)

            p_c = visitor_count_to_warning_prob(
                y_pred[:, h],
                peak_threshold=float(peak_threshold),
                temperature=float(warning_temperature),
            )
            p_w = yt_weather.astype(float)
            yp_prob = 1.0 - (1.0 - p_c) * (1.0 - p_w)
            yp = (yp_prob >= 0.5).astype(int)
            prf = _clf_prf(yt, yp)

            # cost per-sample (aux): 5*FN + 1*FP
            fn = float(np.sum((yt == 1) & (yp == 0)))
            fp = float(np.sum((yt == 0) & (yp == 1)))
            exp_cost = (fn_cost * fn + fp_cost * fp) / float(len(yt)) if len(yt) else float("nan")

            br = brier_score(yt, yp_prob)
            e, _ = expected_calibration_error(yt, yp_prob, n_bins=10)

            rows.append(
                {
                    "horizon": h + 1,
                    "weight": float(w[h]),
                    "precision": prf["precision"],
                    "recall": prf["recall"],
                    "f1": prf["f1"],
                    "brier": float(br),
                    "ece": float(e),
                    "expected_cost": float(exp_cost),
                }
            )
            f1s.append(prf["f1"])
            recalls.append(prf["recall"])
            briers.append(float(br))
            eces.append(float(e))
            costs.append(float(exp_cost))

        warning_by_h_df = pd.DataFrame(rows)
        warning_weighted = {
            "f1_weighted": float(np.sum(w * np.asarray(f1s, dtype=float))),
            "recall_weighted": float(np.sum(w * np.asarray(recalls, dtype=float))),
            "brier_weighted": float(np.sum(w * np.asarray(briers, dtype=float))),
            "ece_weighted": float(np.sum(w * np.asarray(eces, dtype=float))),
            "expected_cost_weighted": float(np.sum(w * np.asarray(costs, dtype=float))),
        }

    metrics = {
        "regression": reg,
        "peak_only_mae": float(peak_mae),
        "crowd_alert": crowd_alert,
        "weather_hazard": {
            **weather_meta,
            "severity_mean": float(np.mean(weather_hazard_sev_flat)) if len(weather_hazard_sev_flat) else float("nan"),
        },
        "suitability_warning": {
            **suitability_warning,
            "brier": float(brier),
            "ece": float(ece),
            "warning_temperature": float(warning_temperature),
            "expected_cost": float(expected_cost),
        },
        "suitability_warning_by_horizon": (
            warning_by_h_df.to_dict(orient="records") if warning_by_h_df is not None else None
        ),
        "suitability_warning_weighted": warning_weighted,
        "meta": {
            "peak_threshold": float(peak_threshold),
            "horizon": int(horizon),
            "n_samples": int(len(y_true_flat)),
        },
        "calibration_bins": ece_table.to_dict(orient="records"),
    }

    return metrics, by_horizon_df


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.generic,)):
        return o.item()
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, (pd.Timestamp,)):
        return str(o)
    raise TypeError(f"Object of type {type(o)} is not JSON serializable")


def save_metrics_artifacts(
    run_dir: str,
    metrics: Dict[str, Any],
    by_horizon_df: Optional[pd.DataFrame],
) -> None:
    _ensure_dir(run_dir)

    json_path = os.path.join(run_dir, "metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2, default=_json_default)

    # metrics.csv (one-row summary)
    flat = {
        "mae": metrics["regression"]["mae"],
        "rmse": metrics["regression"]["rmse"],
        "nrmse": metrics["regression"]["nrmse"],
        "smape": metrics["regression"]["smape"],
        "peak_only_mae": metrics["peak_only_mae"],
        "crowd_alert_precision": metrics["crowd_alert"]["precision"],
        "crowd_alert_recall": metrics["crowd_alert"]["recall"],
        "crowd_alert_f1": metrics["crowd_alert"]["f1"],
        "weather_hazard_enabled": (metrics.get("weather_hazard") or {}).get("enabled"),
        "weather_hazard_prevalence": (metrics.get("weather_hazard") or {}).get("prevalence"),
        "weather_hazard_severity_mean": (metrics.get("weather_hazard") or {}).get("severity_mean"),
        "weather_thr_precip_high": ((metrics.get("weather_hazard") or {}).get("thresholds") or {}).get(
            "precip_high"
        ),
        "weather_thr_temp_high": ((metrics.get("weather_hazard") or {}).get("thresholds") or {}).get(
            "temp_high"
        ),
        "weather_thr_temp_low": ((metrics.get("weather_hazard") or {}).get("thresholds") or {}).get(
            "temp_low"
        ),
        "suitability_warning_precision": metrics["suitability_warning"]["precision"],
        "suitability_warning_recall": metrics["suitability_warning"]["recall"],
        "suitability_warning_f1": metrics["suitability_warning"]["f1"],
        "suitability_warning_brier": metrics["suitability_warning"]["brier"],
        "suitability_warning_ece": metrics["suitability_warning"]["ece"],
        "suitability_warning_expected_cost": metrics["suitability_warning"].get("expected_cost"),
        "suitability_warning_f1_weighted": (metrics.get("suitability_warning_weighted") or {}).get("f1_weighted"),
        "suitability_warning_recall_weighted": (metrics.get("suitability_warning_weighted") or {}).get(
            "recall_weighted"
        ),
        "suitability_warning_brier_weighted": (metrics.get("suitability_warning_weighted") or {}).get(
            "brier_weighted"
        ),
        "suitability_warning_ece_weighted": (metrics.get("suitability_warning_weighted") or {}).get(
            "ece_weighted"
        ),
        "suitability_warning_expected_cost_weighted": (
            (metrics.get("suitability_warning_weighted") or {}).get("expected_cost_weighted")
        ),
        "peak_threshold": metrics["meta"]["peak_threshold"],
        "horizon": metrics["meta"]["horizon"],
        "n_samples": metrics["meta"]["n_samples"],
    }
    pd.DataFrame([flat]).to_csv(os.path.join(run_dir, "metrics.csv"), index=False)

    if by_horizon_df is not None:
        by_horizon_df.to_csv(os.path.join(run_dir, "metrics_by_horizon.csv"), index=False)

    # suitability warning by horizon (if present)
    warn_by_h = metrics.get("suitability_warning_by_horizon")
    if isinstance(warn_by_h, list) and len(warn_by_h) > 0:
        pd.DataFrame(warn_by_h).to_csv(
            os.path.join(run_dir, "suitability_warning_by_horizon.csv"), index=False
        )


def plot_true_vs_pred(
    fig_dir: str,
    dates: Optional[Iterable[Any]],
    y_true_flat: np.ndarray,
    y_pred_flat: np.ndarray,
    max_points: int = 600,
) -> None:
    _ensure_dir(fig_dir)
    y_true_flat = _to_1d(y_true_flat)
    y_pred_flat = _to_1d(y_pred_flat)

    if dates is None:
        x = np.arange(len(y_true_flat))
        x_label = "Index"
    else:
        d = pd.to_datetime(pd.Series(list(dates)))
        x = d
        x_label = "Date"

    if len(y_true_flat) > max_points:
        idx = np.linspace(0, len(y_true_flat) - 1, max_points).astype(int)
        y_true_flat = y_true_flat[idx]
        y_pred_flat = y_pred_flat[idx]
        x = np.asarray(x)[idx]

    plt.figure(figsize=(12, 5))
    plt.plot(x, y_true_flat, label="True", linewidth=1.6)
    plt.plot(x, y_pred_flat, label="Pred", linewidth=1.6)
    plt.xlabel(x_label)
    plt.ylabel("Visitor Count")
    plt.title("True vs Pred")
    if dates is not None:
        plt.xticks(rotation=30)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "true_vs_pred.png"), dpi=150)
    plt.close()


def plot_error_by_horizon(fig_dir: str, by_horizon_df: pd.DataFrame) -> None:
    _ensure_dir(fig_dir)
    plt.figure(figsize=(10, 4))
    plt.bar(by_horizon_df["horizon"].astype(int), by_horizon_df["mae"].astype(float), color="#1f77b4")
    plt.xlabel("Horizon")
    plt.ylabel("MAE")
    plt.title("MAE by Horizon")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "error_by_horizon.png"), dpi=150)
    plt.close()


def plot_confusion_matrix_crowd_alert(
    fig_dir: str,
    y_true_bin: np.ndarray,
    y_pred_bin: np.ndarray,
) -> None:
    _ensure_dir(fig_dir)
    cm = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1])
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.title("Confusion Matrix (crowd_alert)")
    plt.colorbar()
    plt.xticks([0, 1], ["non_peak", "peak"])
    plt.yticks([0, 1], ["non_peak", "peak"])

    thresh = cm.max() / 2.0 if cm.size else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                int(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "confusion_matrix_crowd_alert.png"), dpi=150)
    plt.close()


def plot_suitability_warning_timeline(
    fig_dir: str,
    dates: Optional[Iterable[Any]],
    y_true_bin: np.ndarray,
    y_prob: np.ndarray,
    max_points: int = 800,
) -> None:
    _ensure_dir(fig_dir)
    y_true_bin = _to_1d(y_true_bin)
    y_prob = _to_1d(y_prob)

    if dates is None:
        x = np.arange(len(y_prob))
        x_label = "Index"
    else:
        x = pd.to_datetime(pd.Series(list(dates)))
        x_label = "Date"

    if len(y_prob) > max_points:
        idx = np.linspace(0, len(y_prob) - 1, max_points).astype(int)
        y_true_bin = y_true_bin[idx]
        y_prob = y_prob[idx]
        x = np.asarray(x)[idx]

    plt.figure(figsize=(12, 4))
    plt.plot(x, y_prob, label="Predicted warning probability", linewidth=1.6)
    plt.plot(x, y_true_bin, label="True warning (0/1)", linewidth=1.0, alpha=0.8)
    plt.xlabel(x_label)
    plt.ylabel("Probability")
    plt.title("Suitability Warning Timeline")
    if dates is not None:
        plt.xticks(rotation=30)
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "suitability_warning_timeline.png"), dpi=150)
    plt.close()


def plot_reliability_diagram(
    fig_dir: str,
    ece_table: pd.DataFrame,
) -> None:
    _ensure_dir(fig_dir)
    tbl = ece_table.copy()
    tbl = tbl.dropna(subset=["avg_confidence", "avg_accuracy"], how="any")
    if tbl.empty:
        # still save an empty placeholder for pipeline consistency
        plt.figure(figsize=(5, 5))
        plt.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
        plt.xlabel("Confidence")
        plt.ylabel("Accuracy")
        plt.title("Reliability Diagram")
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "reliability_diagram.png"), dpi=150)
        plt.close()
        return

    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    plt.scatter(tbl["avg_confidence"], tbl["avg_accuracy"], color="#1f77b4")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title("Reliability Diagram")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "reliability_diagram.png"), dpi=150)
    plt.close()


def generate_core_figures(
    run_dir: str,
    *,
    dates: Optional[Iterable[Any]],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    peak_threshold: float,
    warning_temperature: float,
    by_horizon_df: Optional[pd.DataFrame],
    # Weather inputs (optional, for suitability_warning = crowd OR weather)
    weather_precip: Optional[np.ndarray] = None,
    weather_temp_high: Optional[np.ndarray] = None,
    weather_temp_low: Optional[np.ndarray] = None,
    weather_thresholds: Optional[Dict[str, Any]] = None,
) -> None:
    fig_dir = os.path.join(run_dir, "figures")
    _ensure_dir(fig_dir)

    y_true_flat = _to_1d(y_true)
    y_pred_flat = _to_1d(y_pred)
    y_true_peak = (y_true_flat >= float(peak_threshold)).astype(int)
    y_pred_bin = (y_pred_flat >= float(peak_threshold)).astype(int)

    # Weather hazard (optional)
    wh_bin = np.zeros_like(y_true_peak, dtype=int)
    if (
        weather_precip is not None
        and weather_temp_high is not None
        and weather_temp_low is not None
        and weather_thresholds is not None
    ):
        wh_bin, _ = compute_weather_hazard(
            precip=_to_1d(np.asarray(weather_precip)),
            temp_high=_to_1d(np.asarray(weather_temp_high)),
            temp_low=_to_1d(np.asarray(weather_temp_low)),
            thresholds=weather_thresholds,
        )
        wh_bin = wh_bin.astype(int)

    y_true_bin = ((y_true_peak == 1) | (wh_bin == 1)).astype(int)

    p_crowd = visitor_count_to_warning_prob(
        y_pred_flat, peak_threshold=float(peak_threshold), temperature=float(warning_temperature)
    )
    p_weather = wh_bin.astype(float)
    y_prob = 1.0 - (1.0 - p_crowd) * (1.0 - p_weather)

    plot_true_vs_pred(fig_dir, dates, y_true_flat, y_pred_flat)
    if by_horizon_df is not None and not by_horizon_df.empty:
        plot_error_by_horizon(fig_dir, by_horizon_df)
    plot_confusion_matrix_crowd_alert(fig_dir, y_true_peak, y_pred_bin)
    plot_suitability_warning_timeline(fig_dir, dates, y_true_bin, y_prob)

    # reliability diagram uses the same binning as metrics
    _, ece_tbl = expected_calibration_error(y_true_bin, y_prob, n_bins=10)
    plot_reliability_diagram(fig_dir, ece_tbl)


def evaluate_and_save_run(
    run_dir: str,
    *,
    model_name: str,
    feature_count: int,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dates: Optional[Iterable[Any]] = None,
    peak_threshold: float = DEFAULT_PEAK_THRESHOLD,
    horizon: Optional[int] = None,
    warning_temperature: float = 1000.0,
    horizon_weights: Optional[Iterable[float]] = None,
    fn_fp_cost_ratio: Tuple[float, float] = (5.0, 1.0),
    # Weather hazard (Route B quantile thresholds).
    # All arrays must be in real units (NOT scaled).
    weather_precip: Optional[np.ndarray] = None,
    weather_temp_high: Optional[np.ndarray] = None,
    weather_temp_low: Optional[np.ndarray] = None,
    weather_thresholds: Optional[Dict[str, Any]] = None,
    weather_train_precip: Optional[np.ndarray] = None,
    weather_train_temp_high: Optional[np.ndarray] = None,
    weather_train_temp_low: Optional[np.ndarray] = None,
    weather_quantiles: Optional[Dict[str, float]] = None,
    extra_meta: Optional[Dict[str, Any]] = None,
    save_figures: bool = True,
) -> Dict[str, Any]:
    metrics, by_horizon_df = compute_core_metrics(
        y_true,
        y_pred,
        peak_threshold=peak_threshold,
        dates=np.asarray(dates) if dates is not None else None,
        horizon=horizon,
        warning_temperature=warning_temperature,
        horizon_weights=horizon_weights,
        fn_fp_cost_ratio=fn_fp_cost_ratio,
        weather_precip=weather_precip,
        weather_temp_high=weather_temp_high,
        weather_temp_low=weather_temp_low,
        weather_thresholds=weather_thresholds,
        weather_train_precip=weather_train_precip,
        weather_train_temp_high=weather_train_temp_high,
        weather_train_temp_low=weather_train_temp_low,
        weather_quantiles=weather_quantiles,
    )

    metrics["meta"].update(
        {
            "model_name": str(model_name),
            "feature_count": int(feature_count),
            "horizon_weights": list(horizon_weights)
            if horizon_weights is not None
            else (
                DEFAULT_HORIZON_WEIGHTS_H7
                if int(metrics.get("meta", {}).get("horizon", 1)) == 7
                else None
            ),
            "fn_fp_cost_ratio": [float(fn_fp_cost_ratio[0]), float(fn_fp_cost_ratio[1])],
        }
    )
    if extra_meta:
        metrics["meta"].update(extra_meta)

    save_metrics_artifacts(run_dir, metrics, by_horizon_df)
    if save_figures:
        # If thresholds were computed inside compute_core_metrics, pull them from metrics.
        thr_for_fig = weather_thresholds
        if thr_for_fig is None:
            thr_for_fig = (metrics.get("weather_hazard") or {}).get("thresholds")
        generate_core_figures(
            run_dir,
            dates=dates,
            y_true=y_true,
            y_pred=y_pred,
            peak_threshold=peak_threshold,
            warning_temperature=warning_temperature,
            by_horizon_df=by_horizon_df,
            weather_precip=weather_precip,
            weather_temp_high=weather_temp_high,
            weather_temp_low=weather_temp_low,
            weather_thresholds=thr_for_fig,
        )

    return metrics
