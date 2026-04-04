#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中央调度流水线 - 九寨沟景区客流动态预测系统

统一管理所有模型的训练、评估和可视化流程
支持命令行参数选择模型和特征版本
严格规范输出路径和可视化标准
"""

import argparse
import datetime
import os
import sys
from pathlib import Path
import tensorflow as tf

# ==================== 常量配置 ====================
SUPPORTED_MODELS = ["lstm", "gru", "seq2seq_attention", "gru_mimo", "lstm_mimo"]
SUPPORTED_FEATURES = [4, 8]
BASE_OUTPUT_DIR = Path("E:/openclaw/my_project/workspace/FYP/output/runs")
ROOT_DIR = Path(__file__).resolve().parent

def get_timestamp() -> str:
    """生成时间戳格式：YYYYMMDD_HHMMSS"""
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def create_output_dir(model: str, features: int) -> Path:
    """创建规范化的输出目录结构"""
    timestamp = get_timestamp()
    if model == "seq2seq_attention":
        run_name = f"{model}_8features_{timestamp}"
    else:
        run_name = f"{model}_{features}features_{timestamp}"
    
    output_dir = BASE_OUTPUT_DIR / run_name
    figures_dir = output_dir / "figures"
    weights_dir = output_dir / "weights"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory created: {output_dir}")
    return output_dir, figures_dir, weights_dir

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="九寨沟景区客流动态预测系统 - 中央调度流水线",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
支持的模型白名单:
  - lstm: LSTM模型（单步，支持4/8特征）
  - gru: GRU模型（单步，支持4/8特征）
  - seq2seq_attention: Seq2Seq+Attention模型（多步，仅8特征）
  - gru_mimo: GRU MIMO模型（多步直接输出，仅8特征）
  - lstm_mimo: LSTM MIMO模型（多步直接输出，仅8特征）
  
示例用法:
  python run_pipeline.py --model lstm --features 8
  python run_pipeline.py --model gru --features 4
  python run_pipeline.py --model seq2seq_attention
        """.strip()
    )
    
    parser.add_argument(
        "--model", 
        required=True,
        choices=SUPPORTED_MODELS,
        help="选择要训练的模型类型"
    )
    
    parser.add_argument(
        "--features",
        type=int,
        choices=SUPPORTED_FEATURES,
        default=8,
        help="选择特征版本（默认为8）"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=120,
        help="训练轮数（默认120）"
    )
    
    parser.add_argument(
        "--look-back",
        type=int,
        default=30,
        help="lookback窗口大小（默认30）"
    )
    
    parser.add_argument(
        "--save-plots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否保存可视化图表（默认True）"
    )
    
    parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否显示详细日志（默认True）"
    )
    
    return parser.parse_args()

def run_lstm(
    features: int,
    epochs: int,
    look_back: int,
    output_dir: Path,
    figures_dir: Path,
    weights_dir: Path,
    save_plots: bool
) -> None:
    """运行LSTM模型"""
    print("Running LSTM model...")
    
    # 直接运行训练脚本（使用命令行调用）
    import subprocess
    
    cmd = [
        sys.executable,
        f"models/lstm/train_lstm_{features}features.py",
        "--input-csv", "data/processed/jiuzhaigou_daily_features_2016-01-01_2026-04-02.csv",
        "--epochs", str(epochs),
        "--look-back", str(look_back),
        "--save-plots" if save_plots else "--no-save-plots",
        "--output-dir", str(output_dir),
        "--model-dir", str(ROOT_DIR / "model"),
        "--run-name", f"run_{get_timestamp()}_lb{look_back}_ep{epochs}_lstm_{features}features"
    ]
    
    subprocess.run(cmd, check=True)

def run_gru(
    features: int,
    epochs: int,
    look_back: int,
    output_dir: Path,
    figures_dir: Path,
    weights_dir: Path,
    save_plots: bool
) -> None:
    """运行GRU模型"""
    print("Running GRU model...")
    
    # 直接运行训练脚本（使用命令行调用）
    import subprocess
    
    cmd = [
        sys.executable,
        f"models/gru/train_gru_{features}features.py",
        "--input-csv", "data/processed/jiuzhaigou_daily_features_2016-01-01_2026-04-02.csv",
        "--epochs", str(epochs),
        "--look-back", str(look_back),
        "--save-plots" if save_plots else "--no-save-plots",
        "--output-dir", str(output_dir),
        "--model-dir", str(ROOT_DIR / "model"),
        "--run-name", f"run_{get_timestamp()}_lb{look_back}_ep{epochs}_gru_{features}features"
    ]
    
    subprocess.run(cmd, check=True)

def run_gru_mimo(
    epochs: int,
    look_back: int,
    output_dir: Path,
    figures_dir: Path,
    weights_dir: Path,
    save_plots: bool
) -> None:
    """运行GRU MIMO多步输出模型"""
    print("Running GRU MIMO model...")
    import subprocess
    cmd = [
        sys.executable,
        "models/gru/train_gru_mimo_8features.py",
        "--input-csv", "data/processed/jiuzhaigou_daily_features_2016-01-01_2026-04-02.csv",
        "--epochs", str(epochs),
        "--look-back", str(look_back),
        "--save-plots" if save_plots else "--no-save-plots",
        "--output-dir", str(output_dir),
        "--model-dir", str(ROOT_DIR / "model"),
        "--run-name", f"run_{get_timestamp()}_lb{look_back}_ep{epochs}_gru_mimo_8features"
    ]
    subprocess.run(cmd, check=True)


def run_lstm_mimo(
    epochs: int,
    look_back: int,
    output_dir: Path,
    figures_dir: Path,
    weights_dir: Path,
    save_plots: bool
) -> None:
    """运行LSTM MIMO多步输出模型"""
    print("Running LSTM MIMO model...")
    import subprocess
    cmd = [
        sys.executable,
        "models/lstm/train_lstm_mimo_8features.py",
        "--input-csv", "data/processed/jiuzhaigou_daily_features_2016-01-01_2026-04-02.csv",
        "--epochs", str(epochs),
        "--look-back", str(look_back),
        "--save-plots" if save_plots else "--no-save-plots",
        "--output-dir", str(output_dir),
        "--model-dir", str(ROOT_DIR / "model"),
        "--run-name", f"run_{get_timestamp()}_lb{look_back}_ep{epochs}_lstm_mimo_8features"
    ]
    subprocess.run(cmd, check=True)


def run_seq2seq_attention(
    epochs: int,
    look_back: int,
    output_dir: Path,
    figures_dir: Path,
    weights_dir: Path,
    save_plots: bool
) -> None:
    """运行Seq2Seq+Attention模型（仅支持8特征）
    
    ⚠️ 关键：必须进行维度适配处理
    """
    print("Running Seq2Seq+Attention model...")
    
    # 直接运行训练脚本（使用命令行调用）
    import subprocess
    
    cmd = [
        sys.executable,
        "models/lstm/train_seq2seq_attention_8features.py",
        "--input-csv", "data/processed/jiuzhaigou_daily_features_2016-01-01_2026-04-02.csv",
        "--epochs", str(epochs),
        "--look-back", str(look_back),
        "--save-plots" if save_plots else "--no-save-plots",
        "--output-dir", str(output_dir),
        "--model-dir", str(ROOT_DIR / "model"),
        "--run-name", f"run_{get_timestamp()}_lb{look_back}_ep{epochs}_seq2seq_attention_8features"
    ]
    
    subprocess.run(cmd, check=True)

def main():
    """主函数"""
    args = parse_args()
    
    # 验证参数
    if args.model in ("seq2seq_attention", "gru_mimo", "lstm_mimo") and args.features != 8:
        print(f"Warning: {args.model} only supports 8 features, automatically set to 8")
        args.features = 8
    
    # 创建输出目录
    output_dir, figures_dir, weights_dir = create_output_dir(
        args.model, 
        args.features
    )
    
    # 根据模型类型运行训练
    try:
        if args.model == "lstm":
            run_lstm(
                args.features,
                args.epochs,
                args.look_back,
                output_dir,
                figures_dir,
                weights_dir,
                args.save_plots
            )
        elif args.model == "gru":
            run_gru(
                args.features,
                args.epochs,
                args.look_back,
                output_dir,
                figures_dir,
                weights_dir,
                args.save_plots
            )
        elif args.model == "seq2seq_attention":
            run_seq2seq_attention(
                args.epochs, args.look_back, output_dir, figures_dir, weights_dir, args.save_plots
            )
        elif args.model == "gru_mimo":
            run_gru_mimo(
                args.epochs, args.look_back, output_dir, figures_dir, weights_dir, args.save_plots
            )
        elif args.model == "lstm_mimo":
            run_lstm_mimo(
                args.epochs, args.look_back, output_dir, figures_dir, weights_dir, args.save_plots
            )
            
        print("\nTraining completed!")
        print(f"Output directory: {output_dir}")
        
    except Exception as e:
        print(f"\nError: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
