#!/usr/bin/env python3
"""Benchmark runner (LSTM/GRU/Seq2Seq+Attention) with unified core metrics.

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
from typing import Dict, List, Tuple

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
        "fig_error_by_horizon": (figs / "error_by_horizon.png").exists(),
        "fig_confusion_matrix_crowd_alert": (figs / "confusion_matrix_crowd_alert.png").exists(),
        "fig_suitability_warning_timeline": (figs / "suitability_warning_timeline.png").exists(),
        "fig_reliability_diagram": (figs / "reliability_diagram.png").exists(),
    }


def metrics_to_row(model_label: str, m: Dict) -> Dict:
    return {
        "model": model_label,
        "feature_count": m.get("meta", {}).get("feature_count"),
        "horizon": m.get("meta", {}).get("horizon"),
        "mae": m.get("regression", {}).get("mae"),
        "rmse": m.get("regression", {}).get("rmse"),
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
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Run benchmark for LSTM/GRU/Seq2Seq+Attention")
    ap.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training and only generate the compare report for the given --run-tag.",
    )
    ap.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Epochs for quick benchmark runs (override as needed).",
    )
    ap.add_argument("--look-back", type=int, default=30)
    ap.add_argument("--decoder-steps", type=int, default=7)
    ap.add_argument("--output-dir", default="output")
    ap.add_argument("--model-dir", default="model")
    ap.add_argument("--input-csv-4f", default="data/raw/jiuzhaigou_tourism_weather_2024_2026_latest.csv")
    ap.add_argument("--input-csv-8f", default="data/processed/jiuzhaigou_8features_latest.csv")
    ap.add_argument("--run-tag", default=None)
    args, _ = ap.parse_known_args()

    tag = args.run_tag or datetime.now().strftime("%Y%m%d_%H%M%S")

    runs_root = ROOT / args.output_dir / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)

    # --- Run training scripts ---
    lstm_run = f"run_{tag}_lb{args.look_back}_ep{args.epochs}_lstm_4features"
    gru_run = f"run_{tag}_lb{args.look_back}_ep{args.epochs}_gru_4features"
    seq_run = f"run_{tag}_lb{args.look_back}_ep{args.epochs}_seq2seq_attention_8features"

    py = sys.executable

    if not args.skip_train:
        run_cmd(
            [
                py,
                str(ROOT / "models" / "lstm" / "train_lstm_4features.py"),
                "--input-csv",
                args.input_csv_4f,
                "--look-back",
                str(args.look_back),
                "--epochs",
                str(args.epochs),
                "--output-dir",
                args.output_dir,
                "--model-dir",
                args.model_dir,
                "--run-name",
                lstm_run,
            ],
            "LSTM (4 features)",
        )

        run_cmd(
            [
                py,
                str(ROOT / "models" / "gru" / "train_gru_4features.py"),
                "--input-csv",
                args.input_csv_4f,
                "--look-back",
                str(args.look_back),
                "--epochs",
                str(args.epochs),
                "--output-dir",
                args.output_dir,
                "--model-dir",
                args.model_dir,
                "--run-name",
                gru_run,
            ],
            "GRU (4 features)",
        )

        # Seq2Seq+Attention currently supports 8 features (encoder=8, decoder=7)
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

    # --- Load metrics ---
    lstm_dir = runs_root / lstm_run
    gru_dir = runs_root / gru_run
    seq_dir = runs_root / seq_run

    lstm_m = load_metrics(lstm_dir)
    gru_m = load_metrics(gru_dir)
    seq_m = load_metrics(seq_dir)

    df = pd.DataFrame(
        [
            metrics_to_row("lstm_4features", lstm_m),
            metrics_to_row("gru_4features", gru_m),
            metrics_to_row("seq2seq_attention_8features", seq_m),
        ]
    )

    compare_dir = runs_root / f"run_compare_{tag}"
    compare_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(compare_dir / "compare_metrics.csv", index=False)

    # --- Report ---
    lines = []
    lines.append("# Benchmark Compare Report")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("## Run Folders")
    lines.append("")
    lines.append(f"- LSTM (4 features): {lstm_dir}")
    lines.append(f"- GRU (4 features): {gru_dir}")
    lines.append(f"- Seq2Seq+Attention (8 features): {seq_dir}")
    lines.append("")
    lines.append("## Artifact Checks")
    lines.append("")
    for label, rdir in [("lstm", lstm_dir), ("gru", gru_dir), ("seq2seq", seq_dir)]:
        checks = required_artifacts_exist(rdir)
        lines.append(f"### {label}")
        for k, v in checks.items():
            # metrics_by_horizon and error_by_horizon only required for multi-horizon
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
    lines.append(
        "- Seq2Seq+Attention is evaluated as a multi-horizon model (metrics_by_horizon.csv + error_by_horizon.png)."
    )
    lines.append("- Seq2Seq+Attention currently requires 8 features (encoder=8, decoder=7); 4-feature mode is not supported.")

    (compare_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")

    print("\nCompare report saved to:", compare_dir)


if __name__ == "__main__":
    main()
