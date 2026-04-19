#!/usr/bin/env python3
"""Benchmark runner (5 models) with unified core metrics.

Models:
  - LSTM (8 features)
  - GRU (8 features)
  - Seq2Seq+Attention (8 features)
  - XGBoost
  - Transformer

Outputs:
  output/runs/run_compare_<timestamp>/
    compare_metrics.csv
    report.md
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd


ROOT = Path(__file__).resolve().parent


def run_cmd(cmd: List[str], name: str) -> None:
    print("=" * 80)
    print(f"Running: {name}")
    print("Command:", " ".join(cmd))
    print("=" * 80)
    subprocess.run(cmd, check=True)


def load_metrics(run_dir: Path) -> Dict:
    path = run_dir / "metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics.json: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def required_artifacts_exist(run_dir: Path) -> Dict[str, bool]:
    figs = run_dir / "figures"
    return {
        "metrics.json": (run_dir / "metrics.json").exists(),
        "metrics.csv": (run_dir / "metrics.csv").exists(),
        "metrics_by_horizon.csv": (run_dir / "metrics_by_horizon.csv").exists(),
        "fig_true_vs_pred": (figs / "true_vs_pred.png").exists(),
        "fig_confusion_matrix_crowd_alert": (figs / "confusion_matrix_crowd_alert.png").exists(),
        "fig_suitability_warning_timeline": (figs / "suitability_warning_timeline.png").exists(),
        "fig_reliability_diagram": (figs / "reliability_diagram.png").exists(),
    }


def metrics_to_row(model_label: str, m: Dict) -> Dict:
    weighted = m.get("suitability_warning_weighted") or {}
    return {
        "model": model_label,
        "feature_count": m.get("meta", {}).get("feature_count"),
        "horizon": m.get("meta", {}).get("horizon"),
        "mae": m.get("regression", {}).get("mae"),
        "rmse": m.get("regression", {}).get("rmse"),
        "nrmse": m.get("regression", {}).get("nrmse"),
        "smape": m.get("regression", {}).get("smape"),
        "peak_only_mae": m.get("peak_only_mae"),
        "crowd_alert_precision": m.get("crowd_alert", {}).get("precision"),
        "crowd_alert_recall": m.get("crowd_alert", {}).get("recall"),
        "crowd_alert_f1": m.get("crowd_alert", {}).get("f1"),
        "suitability_warning_precision": m.get("suitability_warning", {}).get("precision"),
        "suitability_warning_recall": m.get("suitability_warning", {}).get("recall"),
        "suitability_warning_f1": m.get("suitability_warning", {}).get("f1"),
        "suitability_warning_brier": m.get("suitability_warning", {}).get("brier"),
        "suitability_warning_ece": m.get("suitability_warning", {}).get("ece"),
        "suitability_warning_expected_cost": m.get("suitability_warning", {}).get("expected_cost"),
        "suitability_warning_recall_weighted": weighted.get(
            "recall_weighted", m.get("suitability_warning", {}).get("recall")
        ),
        "suitability_warning_f1_weighted": weighted.get(
            "f1_weighted", m.get("suitability_warning", {}).get("f1")
        ),
        "suitability_warning_brier_weighted": weighted.get(
            "brier_weighted", m.get("suitability_warning", {}).get("brier")
        ),
        "suitability_warning_ece_weighted": weighted.get(
            "ece_weighted", m.get("suitability_warning", {}).get("ece")
        ),
        "suitability_warning_expected_cost_weighted": weighted.get(
            "expected_cost_weighted", m.get("suitability_warning", {}).get("expected_cost")
        ),
    }


def find_latest_nested_run_dir(runs_root: Path, model_name: str, min_mtime: float | None = None) -> Path:
    pattern = f"{model_name}_8features_*/runs/run_*_{model_name}_8features"
    candidates = []
    for p in runs_root.glob(pattern):
        metrics_path = p / "metrics.json"
        if not metrics_path.exists():
            continue
        mtime = metrics_path.stat().st_mtime
        if min_mtime is not None and mtime < min_mtime:
            continue
        candidates.append((mtime, p))
    if not candidates:
        raise FileNotFoundError(f"未找到 {model_name} 的有效运行目录（pattern={pattern}）")
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def find_latest_direct_run_dir(runs_root: Path, model_suffix: str, min_mtime: float | None = None) -> Path:
    pattern = f"run_*_{model_suffix}"
    candidates = []
    for p in runs_root.glob(pattern):
        if not p.is_dir():
            continue
        metrics_path = p / "metrics.json"
        if not metrics_path.exists():
            continue
        mtime = metrics_path.stat().st_mtime
        if min_mtime is not None and mtime < min_mtime:
            continue
        candidates.append((mtime, p))
    if not candidates:
        raise FileNotFoundError(f"未找到 {model_suffix} 的有效运行目录（pattern={pattern}）")
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def find_latest_general_run_dir(runs_root: Path, model_name: str, min_mtime: float | None = None) -> Path:
    candidates = []

    # 结构1：output/runs/run_<...>_<model>_8features/
    for p in runs_root.glob(f"run_*_{model_name}_8features"):
        metrics_path = p / "metrics.json"
        if not metrics_path.exists():
            continue
        mtime = metrics_path.stat().st_mtime
        if min_mtime is not None and mtime < min_mtime:
            continue
        candidates.append((mtime, p))

    # 结构2：output/runs/<model>_8features_<ts>/runs/run_<...>_<model>_8features/
    for p in runs_root.glob(f"{model_name}_8features_*/runs/run_*_{model_name}_8features"):
        metrics_path = p / "metrics.json"
        if not metrics_path.exists():
            continue
        mtime = metrics_path.stat().st_mtime
        if min_mtime is not None and mtime < min_mtime:
            continue
        candidates.append((mtime, p))

    if not candidates:
        raise FileNotFoundError(f"未找到 {model_name}_8features 的有效运行目录。")
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def find_latest_seq2seq_run_dir(runs_root: Path, min_mtime: float | None = None) -> Path:
    candidates = []

    # 结构1：output/runs/run_<...>_seq2seq_attention_8features/
    for p in runs_root.glob("run_*_seq2seq_attention_8features"):
        metrics_path = p / "metrics.json"
        if metrics_path.exists():
            mtime = metrics_path.stat().st_mtime
            if min_mtime is None or mtime >= min_mtime:
                candidates.append((mtime, p))

    # 结构2：output/runs/seq2seq_attention_8features_<ts>/runs/run_<...>_seq2seq_attention_8features/
    for p in runs_root.glob("seq2seq_attention_8features_*/runs/run_*_seq2seq_attention_8features"):
        metrics_path = p / "metrics.json"
        if metrics_path.exists():
            mtime = metrics_path.stat().st_mtime
            if min_mtime is None or mtime >= min_mtime:
                candidates.append((mtime, p))

    if not candidates:
        raise FileNotFoundError("未找到 seq2seq_attention_8features 的有效运行目录。")
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def main() -> None:
    ap = argparse.ArgumentParser(description="Run benchmark for 5 models")
    ap.add_argument(
        "--skip-train",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否跳过重训（默认 True，仅汇总 output/runs 下最新有效 run）。",
    )
    ap.add_argument("--epochs", type=int, default=120, help="Epochs for DL benchmark runs.")
    ap.add_argument("--look-back", type=int, default=30)
    ap.add_argument("--decoder-steps", type=int, default=7)
    ap.add_argument("--output-dir", default="output")
    ap.add_argument("--model-dir", default="model")
    ap.add_argument(
        "--input-csv-8f",
        default="data/processed/jiuzhaigou_daily_features_2016-01-01_2026-04-02.csv",
    )
    ap.add_argument("--run-tag", default=None)
    args, _ = ap.parse_known_args()

    tag = args.run_tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    runs_root = ROOT / args.output_dir / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)

    lstm8_run = f"run_{tag}_lb{args.look_back}_ep{args.epochs}_lstm_8features"
    gru8_run = f"run_{tag}_lb{args.look_back}_ep{args.epochs}_gru_8features"
    seq_run = f"run_{tag}_lb{args.look_back}_ep{args.epochs}_seq2seq_attention_8features"

    py = sys.executable
    train_start_ts = datetime.now().timestamp()

    if not args.skip_train:
        run_cmd(
            [
                py,
                str(ROOT / "models" / "lstm" / "train_lstm_8features.py"),
                "--input-csv",
                args.input_csv_8f,
                "--look-back",
                str(args.look_back),
                "--epochs",
                str(args.epochs),
                "--output-dir",
                args.output_dir,
                "--model-dir",
                args.model_dir,
                "--run-name",
                lstm8_run,
            ],
            "LSTM (8 features)",
        )

        run_cmd(
            [
                py,
                str(ROOT / "models" / "gru" / "train_gru_8features.py"),
                "--input-csv",
                args.input_csv_8f,
                "--look-back",
                str(args.look_back),
                "--epochs",
                str(args.epochs),
                "--output-dir",
                args.output_dir,
                "--model-dir",
                args.model_dir,
                "--run-name",
                gru8_run,
            ],
            "GRU (8 features)",
        )

        run_cmd(
            [
                py,
                str(ROOT / "models" / "lstm" / "train_seq2seq_attention_8features.py"),
                "--input-csv",
                args.input_csv_8f,
                "--encoder-steps",
                str(args.look_back),
                "--decoder-steps",
                str(args.decoder_steps),
                "--epochs",
                str(args.epochs),
                "--output-dir",
                args.output_dir,
                "--model-dir",
                args.model_dir,
                "--run-name",
                seq_run,
            ],
            "Seq2Seq+Attention (8 features)",
        )

        run_cmd(
            [
                py,
                str(ROOT / "models" / "xgboost" / "train_xgboost_8features.py"),
                "--data",
                args.input_csv_8f,
            ],
            "XGBoost",
        )

        run_cmd(
            [
                py,
                str(ROOT / "models" / "transformer" / "train_transformer_8features.py"),
                "--data",
                args.input_csv_8f,
                "--look-back",
                str(args.look_back),
                "--epochs",
                str(args.epochs),
            ],
            "Transformer",
        )

    # --- Resolve run directories ---
    if args.skip_train:
        lstm8_dir = find_latest_general_run_dir(runs_root, "lstm")
        gru8_dir = find_latest_general_run_dir(runs_root, "gru")
        seq_dir = find_latest_seq2seq_run_dir(runs_root)
        xgb_dir = find_latest_nested_run_dir(runs_root, "xgboost")
        transformer_dir = find_latest_nested_run_dir(runs_root, "transformer")
    else:
        lstm8_dir = find_latest_general_run_dir(runs_root, "lstm", min_mtime=train_start_ts - 2)
        gru8_dir = find_latest_general_run_dir(runs_root, "gru", min_mtime=train_start_ts - 2)
        seq_dir = find_latest_seq2seq_run_dir(runs_root, min_mtime=train_start_ts - 2)
        xgb_dir = find_latest_nested_run_dir(runs_root, "xgboost", min_mtime=train_start_ts - 2)
        transformer_dir = find_latest_nested_run_dir(
            runs_root, "transformer", min_mtime=train_start_ts - 2
        )

    # --- Load metrics ---
    lstm8_m = load_metrics(lstm8_dir)
    gru8_m = load_metrics(gru8_dir)
    seq_m = load_metrics(seq_dir)
    xgb_m = load_metrics(xgb_dir)
    transformer_m = load_metrics(transformer_dir)

    df = pd.DataFrame(
        [
            metrics_to_row("lstm_8features", lstm8_m),
            metrics_to_row("gru_8features", gru8_m),
            metrics_to_row("seq2seq_attention_8features", seq_m),
            metrics_to_row("xgboost_8features", xgb_m),
            metrics_to_row("transformer_8features", transformer_m),
        ]
    )

    compare_dir = runs_root / f"run_compare_{tag}"
    compare_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(compare_dir / "compare_metrics.csv", index=False)

    lines = []
    lines.append("# Benchmark Compare Report")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("## Run Folders")
    lines.append("")
    lines.append(f"- LSTM (8 features): {lstm8_dir}")
    lines.append(f"- GRU (8 features): {gru8_dir}")
    lines.append(f"- Seq2Seq+Attention (8 features): {seq_dir}")
    lines.append(f"- XGBoost: {xgb_dir}")
    lines.append(f"- Transformer: {transformer_dir}")
    lines.append("")
    lines.append("## Artifact Checks")
    lines.append("")
    for label, rdir in [
        ("lstm_8f", lstm8_dir),
        ("gru_8f", gru8_dir),
        ("seq2seq", seq_dir),
        ("xgboost", xgb_dir),
        ("transformer", transformer_dir),
    ]:
        checks = required_artifacts_exist(rdir)
        lines.append(f"### {label}")
        for k, v in checks.items():
            lines.append(f"- {k}: {'OK' if v else 'MISSING'}")
        lines.append("")

    lines.append("## Metrics Table")
    lines.append("")
    lines.append("```text")
    lines.append(df.to_string(index=False))
    lines.append("```")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- 该报告按 5 个模型类型汇总，不再按 4/8 特征变体展开。")
    lines.append("- XGBoost 与 Transformer 的运行目录由 output/runs 下最新有效 metrics.json 自动发现。")

    (compare_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")

    print("\nCompare report saved to:", compare_dir)


if __name__ == "__main__":
    main()
