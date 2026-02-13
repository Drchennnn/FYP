"""流水线入口：通用预处理 + LSTM 训练。"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_step(cmd: list[str]) -> None:
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description="运行预处理 + LSTM 训练。")
    parser.add_argument("--skip-preprocess", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lookback", type=int, default=30)
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--model-dir", default="model")
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--save-plots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否生成可视化图，默认 True。",
    )
    args = parser.parse_args()

    python_bin = sys.executable
    root = Path(__file__).resolve().parent

    if not args.skip_preprocess:
        run_step([python_bin, str(root / "models" / "common" / "preprocess.py")])

    if not args.skip_train:
        train_cmd = [
            python_bin,
            str(root / "models" / "lstm" / "train_lstm.py"),
            "--epochs",
            str(args.epochs),
            "--look-back",
            str(args.lookback),
            "--output-dir",
            str(root / args.output_dir),
            "--model-dir",
            str(root / args.model_dir),
        ]
        if args.run_name:
            train_cmd.extend(["--run-name", args.run_name])
        train_cmd.append("--save-plots" if args.save_plots else "--no-save-plots")
        run_step(train_cmd)


if __name__ == "__main__":
    main()
