"""
流水线入口：通用预处理 + LSTM 训练。

流程：
1. 预处理 (Preprocess): 读取 data/raw 下最新的爬虫数据 -> 生成 data/processed/jiuzhaigou_daily_features.csv (包含特征工程)
2. 训练 (Train): 读取 data/processed 下的数据 -> 训练 LSTM 模型 -> 输出模型和评估结果
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
from pathlib import Path

def get_latest_data_file(data_dir: Path) -> Path:
    """获取指定目录下最新的 .csv 文件，优先选择 jiuzhaigou_raw_ 开头的文件"""
    # Exclude known output files to prevent loop
    files = list(data_dir.glob("*.csv"))
    
    # Filter out the intermediate output file if others exist
    raw_files = [f for f in files if f.name.startswith("jiuzhaigou_raw_")]
    
    if raw_files:
        # If we have files matching the raw pattern, use the latest of those
        return max(raw_files, key=os.path.getmtime)
    
    # Fallback to any csv if no specific raw pattern found
    valid_files = [f for f in files if f.name != "jiuzhaigou_tourism_weather_2024_2026_latest.csv"]
    if valid_files:
        return max(valid_files, key=os.path.getmtime)
        
    if not files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
        
    return max(files, key=os.path.getmtime)

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
        # 自动查找 data/raw 下最新的爬虫文件
        raw_dir = root / "data" / "raw"
        processed_dir = root / "data" / "processed"
        
        try:
            latest_raw = get_latest_data_file(raw_dir)
            print(f"Using latest raw data for preprocessing: {latest_raw}")
            
            # 构造对应的 processed 文件名
            # 从 jiuzhaigou_raw_2024-01-01_2026-02-26.csv 提取日期
            raw_filename = latest_raw.name
            if raw_filename.startswith("jiuzhaigou_raw_"):
                date_part = raw_filename.replace("jiuzhaigou_raw_", "").replace(".csv", "")
                processed_filename = f"jiuzhaigou_daily_features_{date_part}.csv"
            else:
                processed_filename = f"jiuzhaigou_daily_features_latest.csv"
            
            processed_output = processed_dir / processed_filename
            
            # 调用预处理脚本
            run_step([
                python_bin, 
                str(root / "models" / "common" / "preprocess.py"),
                "--input-csv", str(latest_raw),
                "--output-csv", str(processed_output)
            ])
        except FileNotFoundError:
            print("Warning: No raw data found in data/raw, running preprocess with default.")
            run_step([python_bin, str(root / "models" / "common" / "preprocess.py")])

    if not args.skip_train:
        # 获取最新的 processed 数据
        data_processed_dir = root / "data" / "processed"
        try:
            latest_input = get_latest_data_file(data_processed_dir)
            print(f"Using latest training data: {latest_input}")
        except FileNotFoundError:
            print(f"Warning: No processed data found in {data_processed_dir}, using default.")
            latest_input = root / "data" / "processed" / "jiuzhaigou_daily_features.csv"

        train_cmd = [
            python_bin,
            str(root / "models" / "lstm" / "train_lstm.py"),
            "--input-csv",
            str(latest_input),
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
