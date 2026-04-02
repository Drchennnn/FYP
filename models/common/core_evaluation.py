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


DEFAULT_PEAK_THRESHOLD = 18500


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
    return {"precision": float(p), "recall": float(r), "f1": float(f1)}


def compute_core_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    peak_threshold: float = DEFAULT_PEAK_THRESHOLD,
    horizon: Optional[int] = None,
    warning_temperature: float = 1000.0,
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
        "smape": _smape(y_true_flat, y_pred_flat),
    }

    # Peak-only MAE
    peak_mask = y_true_flat >= float(peak_threshold)
    if np.any(peak_mask):
        peak_mae = _mae(y_true_flat[peak_mask], y_pred_flat[peak_mask])
    else:
        peak_mae = float("nan")

    # crowd_alert (deterministic)
    y_true_peak = (y_true_flat >= float(peak_threshold)).astype(int)
    y_pred_peak = (y_pred_flat >= float(peak_threshold)).astype(int)
    crowd_alert = _clf_prf(y_true_peak, y_pred_peak)

    # suitability_warning (probabilistic via transform)
    y_prob_warn = visitor_count_to_warning_prob(
        y_pred_flat, peak_threshold=float(peak_threshold), temperature=float(warning_temperature)
    )
    y_pred_warn = (y_prob_warn >= 0.5).astype(int)
    suitability_warning = _clf_prf(y_true_peak, y_pred_warn)
    brier = brier_score(y_true_peak, y_prob_warn)
    ece, ece_table = expected_calibration_error(y_true_peak, y_prob_warn, n_bins=10)

    # Per-horizon regression metrics
    if np.asarray(y_true).ndim == 2 and horizon and horizon > 1:
        rows = []
        for h in range(horizon):
            rows.append(
                {
                    "horizon": h + 1,
                    "mae": _mae(y_true[:, h], y_pred[:, h]),
                    "rmse": _rmse(y_true[:, h], y_pred[:, h]),
                    "smape": _smape(y_true[:, h], y_pred[:, h]),
                }
            )
        by_horizon_df = pd.DataFrame(rows)

    metrics = {
        "regression": reg,
        "peak_only_mae": float(peak_mae),
        "crowd_alert": crowd_alert,
        "suitability_warning": {
            **suitability_warning,
            "brier": float(brier),
            "ece": float(ece),
            "warning_temperature": float(warning_temperature),
        },
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
        "smape": metrics["regression"]["smape"],
        "peak_only_mae": metrics["peak_only_mae"],
        "crowd_alert_precision": metrics["crowd_alert"]["precision"],
        "crowd_alert_recall": metrics["crowd_alert"]["recall"],
        "crowd_alert_f1": metrics["crowd_alert"]["f1"],
        "suitability_warning_precision": metrics["suitability_warning"]["precision"],
        "suitability_warning_recall": metrics["suitability_warning"]["recall"],
        "suitability_warning_f1": metrics["suitability_warning"]["f1"],
        "suitability_warning_brier": metrics["suitability_warning"]["brier"],
        "suitability_warning_ece": metrics["suitability_warning"]["ece"],
        "peak_threshold": metrics["meta"]["peak_threshold"],
        "horizon": metrics["meta"]["horizon"],
        "n_samples": metrics["meta"]["n_samples"],
    }
    pd.DataFrame([flat]).to_csv(os.path.join(run_dir, "metrics.csv"), index=False)

    if by_horizon_df is not None:
        by_horizon_df.to_csv(os.path.join(run_dir, "metrics_by_horizon.csv"), index=False)


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
) -> None:
    fig_dir = os.path.join(run_dir, "figures")
    _ensure_dir(fig_dir)

    y_true_flat = _to_1d(y_true)
    y_pred_flat = _to_1d(y_pred)
    y_true_bin = (y_true_flat >= float(peak_threshold)).astype(int)
    y_pred_bin = (y_pred_flat >= float(peak_threshold)).astype(int)
    y_prob = visitor_count_to_warning_prob(
        y_pred_flat, peak_threshold=float(peak_threshold), temperature=float(warning_temperature)
    )

    plot_true_vs_pred(fig_dir, dates, y_true_flat, y_pred_flat)
    if by_horizon_df is not None and not by_horizon_df.empty:
        plot_error_by_horizon(fig_dir, by_horizon_df)
    plot_confusion_matrix_crowd_alert(fig_dir, y_true_bin, y_pred_bin)
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
    extra_meta: Optional[Dict[str, Any]] = None,
    save_figures: bool = True,
) -> Dict[str, Any]:
    metrics, by_horizon_df = compute_core_metrics(
        y_true,
        y_pred,
        peak_threshold=peak_threshold,
        horizon=horizon,
        warning_temperature=warning_temperature,
    )

    metrics["meta"].update(
        {
            "model_name": str(model_name),
            "feature_count": int(feature_count),
        }
    )
    if extra_meta:
        metrics["meta"].update(extra_meta)

    save_metrics_artifacts(run_dir, metrics, by_horizon_df)
    if save_figures:
        generate_core_figures(
            run_dir,
            dates=dates,
            y_true=y_true,
            y_pred=y_pred,
            peak_threshold=peak_threshold,
            warning_temperature=warning_temperature,
            by_horizon_df=by_horizon_df,
        )

    return metrics

