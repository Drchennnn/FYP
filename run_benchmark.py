#!/usr/bin/env python3
"""
全模型联合训练与预测对比脚本

同时训练LSTM、GRU、Prophet三个模型，使用相同的8个特征，
进行性能对比分析，特别关注tourism_num_lag_7的效果提升。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np


def get_latest_data_file(data_dir: Path) -> Path:
    """获取指定目录下最新的 .csv 文件"""
    files = list(data_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    return max(files, key=lambda f: f.stat().st_mtime)


def run_command(cmd: List[str], model_name: str) -> Tuple[bool, str]:
    """运行命令并返回结果"""
    print(f"\n{'='*60}")
    print(f"开始训练 {model_name} 模型")
    print(f"命令: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8"
        )
        
        elapsed = time.time() - start_time
        print(f"{model_name} 训练完成！耗时: {elapsed:.1f}秒")
        
        # 显示最后几行输出
        if result.stdout:
            lines = result.stdout.strip().split('\n')
            last_lines = lines[-10:] if len(lines) > 10 else lines
            print("最后输出:")
            for line in last_lines:
                print(f"  {line}")
        
        return True, f"训练成功，耗时{elapsed:.1f}秒"
        
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"{model_name} 训练失败！耗时: {elapsed:.1f}秒")
        print(f"错误输出:")
        print(e.stderr[:500])
        
        return False, f"训练失败: {e.stderr[:200]}"


def load_metrics(model_dir: Path, run_name: str, model_type: str) -> Dict:
    """加载模型评估指标"""
    metrics_path = model_dir / "runs" / run_name / f"{model_type}_metrics.json"
    
    if not metrics_path.exists():
        # 尝试其他可能的路径
        alt_path = model_dir / run_name / f"{model_type}_metrics.json"
        if alt_path.exists():
            metrics_path = alt_path
        else:
            raise FileNotFoundError(f"指标文件不存在: {metrics_path}")
    
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    
    return metrics


def compare_models(metrics_lstm: Dict, metrics_gru: Dict, metrics_prophet: Dict) -> pd.DataFrame:
    """对比三个模型的性能指标"""
    comparison_data = []
    
    # 提取关键回归指标
    for model_name, metrics in [
        ("LSTM", metrics_lstm),
        ("GRU", metrics_gru),
        ("Prophet", metrics_prophet)
    ]:
        reg_metrics = metrics.get("regression", {})
        cls_metrics = metrics.get("classification", {})
        
        row = {
            "Model": model_name,
            "MAE": reg_metrics.get("mae", 0),
            "RMSE": reg_metrics.get("rmse", 0),
            "MAPE": reg_metrics.get("mape", 0),
            "SMAPE": reg_metrics.get("smape", 0),
            "R2": reg_metrics.get("r2", 0),
            "Peak_Accuracy": cls_metrics.get("accuracy", 0),
            "Peak_Precision": cls_metrics.get("precision", 0),
            "Peak_Recall": cls_metrics.get("recall", 0),
            "Peak_F1": cls_metrics.get("f1", 0),
        }
        
        # 添加运行信息
        run_info = metrics.get("run_info", {})
        if run_info:
            row.update({
                "Samples": run_info.get("samples", 0),
                "Test_Samples": run_info.get("test_samples", 0),
                "Features": run_info.get("input_dim", 0),
                "Architecture": run_info.get("model_architecture", ""),
            })
        
        comparison_data.append(row)
    
    df = pd.DataFrame(comparison_data)
    
    # 计算相对提升（以LSTM为基准）
    if not df.empty and "LSTM" in df["Model"].values:
        lstm_mae = df.loc[df["Model"] == "LSTM", "MAE"].values[0]
        lstm_rmse = df.loc[df["Model"] == "LSTM", "RMSE"].values[0]
        lstm_mape = df.loc[df["Model"] == "LSTM", "MAPE"].values[0]
        
        for idx, row in df.iterrows():
            if row["Model"] != "LSTM":
                if lstm_mae > 0:
                    df.loc[idx, "MAE_Improvement"] = (lstm_mae - row["MAE"]) / lstm_mae * 100
                if lstm_rmse > 0:
                    df.loc[idx, "RMSE_Improvement"] = (lstm_rmse - row["RMSE"]) / lstm_rmse * 100
                if lstm_mape > 0:
                    df.loc[idx, "MAPE_Improvement"] = (lstm_mape - row["MAPE"]) / lstm_mape * 100
    
    return df


def analyze_lag7_effect(metrics_4features: Dict, metrics_8features: Dict) -> Dict:
    """分析滞后7天特征的效果提升
    
    Args:
        metrics_4features: 4特征版本的指标
        metrics_8features: 8特征版本的指标
        
    Returns:
        提升分析结果
    """
    analysis = {}
    
    if not metrics_4features or not metrics_8features:
        return analysis
    
    # 提取关键指标
    for key in ["mae", "rmse", "mape", "smape", "r2"]:
        old_val = metrics_4features.get("regression", {}).get(key)
        new_val = metrics_8features.get("regression", {}).get(key)
        
        if old_val is not None and new_val is not None:
            if key in ["mae", "rmse", "mape", "smape"]:
                # 误差指标，越小越好
                if old_val > 0:
                    improvement = (old_val - new_val) / old_val * 100
                    direction = "降低" if improvement > 0 else "增加"
                else:
                    improvement = 0
                    direction = "无变化"
            else:  # r2
                # R2指标，越大越好
                if abs(old_val) > 0:
                    improvement = (new_val - old_val) / abs(old_val) * 100
                    direction = "提升" if improvement > 0 else "下降"
                else:
                    improvement = 0
                    direction = "无变化"
            
            analysis[key] = {
                "old": old_val,
                "new": new_val,
                "improvement": improvement,
                "direction": direction,
                "interpretation": f"{key.upper()} {direction}了 {abs(improvement):.1f}%"
            }
    
    return analysis


def save_benchmark_report(
    comparison_df: pd.DataFrame,
    lag7_analysis: Dict,
    output_dir: Path,
    run_name: str
) -> None:
    """保存基准测试报告"""
    report_dir = output_dir / "benchmark_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 保存对比表格
    csv_path = report_dir / f"{run_name}_comparison.csv"
    comparison_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    
    # 2. 保存详细报告
    report_path = report_dir / f"{run_name}_report.md"
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# 全模型联合训练基准测试报告\n\n")
        f.write(f"**运行名称**: {run_name}\n")
        f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**特征版本**: 8特征 (含tourism_num_lag_7_scaled)\n\n")
        
        f.write("## 1. 模型性能对比\n\n")
        f.write(comparison_df.to_markdown(index=False))
        f.write("\n\n")
        
        f.write("## 2. 关键指标分析\n\n")
        
        # MAE分析
        if not comparison_df.empty and "MAE" in comparison_df.columns:
            best_mae_idx = comparison_df["MAE"].idxmin()
            best_mae = comparison_df.loc[best_mae_idx]
            f.write(f"### 2.1 平均绝对误差 (MAE)\n")
            f.write(f"- **最佳模型**: {best_mae['Model']} (MAE = {best_mae['MAE']:.1f})\n")
            
            for _, row in comparison_df.iterrows():
                if row["Model"] != best_mae["Model"]:
                    diff = row["MAE"] - best_mae["MAE"]
                    perc = (row["MAE"] / best_mae["MAE"] - 1) * 100 if best_mae["MAE"] > 0 else 0
                    f.write(f"- {row['Model']}: {row['MAE']:.1f} (差 {diff:.1f}, 高 {perc:.1f}%)\n")
            
            f.write("\n")
        
        # R2分析
        if not comparison_df.empty and "R2" in comparison_df.columns:
            best_r2_idx = comparison_df["R2"].idxmax()
            best_r2 = comparison_df.loc[best_r2_idx]
            f.write(f"### 2.2 决定系数 (R2)\n")
            f.write(f"- **最佳模型**: {best_r2['Model']} (R2 = {best_r2['R2']:.4f})\n")
            
            for _, row in comparison_df.iterrows():
                if row["Model"] != best_r2["Model"]:
                    diff = best_r2["R2"] - row["R2"]
                    f.write(f"- {row['Model']}: {row['R2']:.4f} (差 {diff:.4f})\n")
            
            f.write("\n")
        
        # 高峰日预测分析
        if not comparison_df.empty and "Peak_F1" in comparison_df.columns:
            best_f1_idx = comparison_df["Peak_F1"].idxmax()
            best_f1 = comparison_df.loc[best_f1_idx]
            f.write(f"### 2.3 高峰日预测 (F1 Score)\n")
            f.write(f"- **最佳模型**: {best_f1['Model']} (F1 = {best_f1['Peak_F1']:.4f})\n")
            
            for _, row in comparison_df.iterrows():
                if row["Model"] != best_f1["Model"]:
                    diff = best_f1["Peak_F1"] - row["Peak_F1"]
                    f.write(f"- {row['Model']}: {row['Peak_F1']:.4f} (差 {diff:.4f})\n")
            
            f.write("\n")
        
        f.write("## 3. tourism_num_lag_7 效果分析\n\n")
        
        if lag7_analysis:
            f.write("### 3.1 指标提升对比\n\n")
            
            for key, analysis in lag7_analysis.items():
                if key in ["mae", "rmse", "mape"]:
                    f.write(f"- **{key.upper()}**: {analysis['old']:.2f} → {analysis['new']:.2f} ")
                    f.write(f"({analysis['direction']} {abs(analysis['improvement']):.1f}%)\n")
            
            f.write("\n### 3.2 关键发现\n\n")
            
            # 分析最重要的提升
            mae_improvement = lag7_analysis.get("mae", {}).get("improvement", 0)
            mape_improvement = lag7_analysis.get("mape", {}).get("improvement", 0)
            
            if mae_improvement > 5:
                f.write(f"1. **MAE显著降低**: 平均绝对误差降低了 {mae_improvement:.1f}%，模型预测更准确\n")
            elif mae_improvement > 0:
                f.write(f"1. **MAE轻微改善**: 平均绝对误差降低了 {mae_improvement:.1f}%\n")
            
            if mape_improvement > 5:
                f.write(f"2. **MAPE明显改善**: 相对误差降低了 {mape_improvement:.1f}%，百分比预测更可靠\n")
            elif mape_improvement > 0:
                f.write(f"2. **MAPE轻微改善**: 相对误差降低了 {mape_improvement:.1f}%\n")
            
            if mae_improvement > 5 or mape_improvement > 5:
                f.write(f"3. **滞后特征价值高**: tourism_num_lag_7 对预测精度有显著提升，建议保留\n")
            elif mae_improvement > 0 or mape_improvement > 0:
                f.write(f"3. **滞后特征有效**: 有一定提升效果，建议保留\n")
            else:
                f.write(f"3. **滞后特征效果有限**: 提升不明显，可考虑优化特征工程\n")
        
        f.write("\n")
        
        f.write("## 4. 推荐方案\n\n")
        
        if not comparison_df.empty:
            # 基于MAE推荐
            best_mae_idx = comparison_df["MAE"].idxmin()
            best_mae_model = comparison_df.loc[best_mae_idx, "Model"]
            
            # 基于R2推荐
            best_r2_idx = comparison_df["R2"].idxmax()
            best_r2_model = comparison_df.loc[best_r2_idx, "Model"]
            
            # 基于F1推荐
            best_f1_idx = comparison_df["Peak_F1"].idxmax()
            best_f1_model = comparison_df.loc[best_f1_idx, "Model"]
            
            f.write("### 4.1 模型选择建议\n\n")
            f.write(f"- **追求最低误差**: 选择 {best_mae_model} (MAE最低)\n")
            f.write(f"- **追求最佳拟合**: 选择 {best_r2_model} (R2最高)\n")
            f.write(f"- **追求高峰日预测**: 选择 {best_f1_model} (F1最高)\n")
            
            # 综合推荐
            model_scores = {}
            for model in comparison_df["Model"].unique():
                score = 0
                if model == best_mae_model:
                    score += 3
                if model == best_r2_model:
                    score += 2
                if model == best_f1_model:
                    score += 1
                model_scores[model] = score
            
            best_overall = max(model_scores, key=model_scores.get)
            f.write(f"- **综合推荐**: {best_overall} (综合表现最佳)\n")
        
        f.write("\n### 4.2 部署建议\n\n")
        f.write("1. **生产环境**: 选择综合表现最佳的模型\n")
        f.write("2. **实时预测**: 考虑模型推理速度，GRU通常比LSTM更快\n")
        f.write("3. **可解释性**: Prophet提供趋势分解和预测区间，适合业务分析\n")
        f.write("4. **特征工程**: 保留tourism_num_lag_7特征，继续优化气象特征\n")
    
    print(f"\n基准测试报告已保存:")
    print(f"  - 对比表格: {csv_path}")
    print(f"  - 详细报告: {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="全模型联合训练与对比分析")
    parser.add_argument("--skip-preprocess", action="store_true", help="跳过预处理")
    parser.add_argument("--skip-train", action="store_true", help="跳过训练，只分析已有结果")
    parser.add_argument("--run-name", default=None, help="运行名称")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮次")
    parser.add_argument("--lookback", type=int, default=30, help="历史窗口长度")
    parser.add_argument("--test-ratio", type=float, default=0.2, help="测试集比例")
    parser.add_argument("--output-dir", default="output", help="输出目录")
    parser.add_argument("--model-dir", default="model", help="模型目录")
    args, _ = parser.parse_known_args()
    
    python_bin = sys.executable
    root = Path(__file__).resolve().parent
    
    # 生成运行名称
    if args.run_name:
        run_name = args.run_name
    else:
        run_name = f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print(f"{'='*80}")
    print(f"全模型联合训练与对比分析")
    print(f"运行名称: {run_name}")
    print(f"特征版本: 8特征 (含tourism_num_lag_)")