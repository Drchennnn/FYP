"""
Transformer Encoder 模型（自注意力时序预测）

架构：Encoder-only Transformer + 全局平均池化 + Dense 输出头。
输入与 GRU/LSTM 完全相同：(batch, lookback=30, features=8+扩展)
输出：标量（下一天客流，归一化），滚动推理生成7天预测。

与 GRU/LSTM 的本质区别：
  - GRU/LSTM: 隐状态递归传递，捕捉局部时序依赖
  - Transformer: Multi-Head Self-Attention，全局注意力，
    可直接建模任意距离的时间步依赖，无梯度消失问题

超参基准：
  d_model=64, num_heads=4, num_layers=2, dff=128, dropout=0.1
  (小数据集不宜过深，过拟合风险高)
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import chinese_calendar as cncal
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# ── 项目根路径 ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.common.core_evaluation import evaluate_and_save_run, compute_dynamic_peak_threshold
from models.common.evaluator import calculate_metrics, save_metrics_to_files

# ── 常量 ─────────────────────────────────────────────────────────────────────
DEFAULT_DATA_PATH = (
    PROJECT_ROOT
    / "data/processed/jiuzhaigou_daily_features_2016-01-01_2026-04-02.csv"
)
OUTPUT_BASE = PROJECT_ROOT / "output" / "runs"

TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
TEST_RATIO  = 0.10
LOOK_BACK   = 30          # 与 GRU/LSTM 保持一致


# ── 节假日 / 旺季标记（与 GRU 保持一致）─────────────────────────────────────
def mark_core_holiday(date_val: pd.Timestamp) -> int:
    m, d = int(date_val.month), int(date_val.day)
    if m == 10 and 1 <= d <= 7:
        return 1
    if m == 5 and 1 <= d <= 5:
        return 1
    try:
        return int(cncal.is_holiday(date_val.date()))
    except Exception:
        return 0


def is_peak_season(date_val: pd.Timestamp) -> int:
    """旺季：4月1日 ~ 11月15日"""
    m, d = int(date_val.month), int(date_val.day)
    if m < 4 or m > 11:
        return 0
    if m == 11 and d > 15:
        return 0
    return 1


# ── 特征工程（复用 GRU 逻辑，P0 扩展版）────────────────────────────────────
def load_and_engineer_features(input_csv: Path) -> tuple[pd.DataFrame, MinMaxScaler]:
    """
    加载 CSV，构造与 GRU 相同的 8 基础特征 + P0 扩展特征。

    TODO (Codex):
        与 GRU 脚本的 load_and_engineer_features 保持一致，额外增加：
          - tourism_num_lag_14_scaled   (14天滞后)
          - rolling_mean_14_scaled      (14天滚动均值)
          - is_peak_season              (旺淡季标记，0/1)
        共 11 个特征（原8 + 3个P0新增），构成扩展特征集。

        IMPORTANT: MinMaxScaler 只在训练集上 fit，然后 transform 全集。
    """
    df = pd.read_csv(input_csv)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # TODO: 参照 GRU 脚本补全特征工程逻辑
    raise NotImplementedError("TODO: implement feature engineering")


# ── 序列构建（与 GRU 完全相同）───────────────────────────────────────────────
def build_sequences(
    data: np.ndarray, look_back: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    将时序数组切成滑动窗口。

    TODO (Codex):
        X shape: (N, look_back, n_features)
        y shape: (N,)  — 每窗口的下一步目标值（visitor_count_scaled，index=0）
        逻辑与 GRU 脚本完全相同，可直接复制。
    """
    raise NotImplementedError("TODO: implement build_sequences()")


# ── Transformer 组件 ──────────────────────────────────────────────────────────

class PositionalEncoding(tf.keras.layers.Layer):
    """
    正弦/余弦位置编码（固定，非可学习）。

    TODO (Codex):
        公式：PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
              PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        输入 shape: (batch, seq_len, d_model)
        输出 shape: (batch, seq_len, d_model)  — x + positional_encoding
    """
    def __init__(self, d_model: int, max_len: int = 512, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        # TODO: 预计算 pe 矩阵，shape=(1, max_len, d_model)

    def call(self, x):
        # TODO: return x + self.pe[:, :tf.shape(x)[1], :]
        raise NotImplementedError("TODO: implement PositionalEncoding.call()")


class TransformerEncoderBlock(tf.keras.layers.Layer):
    """
    单层 Transformer Encoder Block。

    结构：
      x → MultiHeadAttention(self) → Add & Norm
        → FFN(Dense→ReLU→Dense) → Add & Norm

    TODO (Codex):
        参数：num_heads, d_model, dff, dropout_rate
        使用 tf.keras.layers.MultiHeadAttention
        LayerNormalization 在残差连接之后（Post-Norm 风格）
    """
    def __init__(self, num_heads: int, d_model: int, dff: int, dropout_rate: float, **kwargs):
        super().__init__(**kwargs)
        # TODO: 初始化 MHA、FFN Dense 层、LayerNorm、Dropout

    def call(self, x, training=False):
        # TODO: 实现前向传播
        raise NotImplementedError("TODO: implement TransformerEncoderBlock.call()")


def build_model(
    look_back: int,
    n_features: int,
    d_model: int = 64,
    num_heads: int = 4,
    num_layers: int = 2,
    dff: int = 128,
    dropout_rate: float = 0.1,
) -> tf.keras.Model:
    """
    构建 Transformer Encoder 回归模型。

    架构：
      Input(look_back, n_features)
        → Dense(d_model)               # 特征投影
        → PositionalEncoding
        → N × TransformerEncoderBlock
        → GlobalAveragePooling1D
        → Dense(64, relu)
        → Dropout
        → Dense(1)                     # 输出标量

    TODO (Codex):
        1. 使用 tf.keras.Input 定义输入
        2. 按上方架构堆叠各层
        3. compile: optimizer=Adam(1e-3), loss=Huber()
        4. 返回 model
    """
    raise NotImplementedError("TODO: implement build_model()")


# ── 主流程 ────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Transformer Encoder 客流预测")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--look-back", type=int, default=LOOK_BACK)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dff", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--peak-quantile", type=float, default=0.75)
    args = parser.parse_args()

    # ── 数据准备 ────────────────────────────────────────────────────────────
    # TODO (Codex):
    #   df, scaler = load_and_engineer_features(args.data)
    #   feature_cols 包含 11 个扩展特征（P0）
    #   按 80/10/10 切分，build_sequences() 构建 X/y
    #   与 GRU 脚本保持完全一致的切分逻辑

    # ── 输出目录 ─────────────────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_{ts}_lb{args.look_back}_ep{args.epochs}_transformer_8features"
    runs_dir = OUTPUT_BASE / f"transformer_8features_{ts}" / "runs"
    run_dir = runs_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = run_dir / "weights"
    fig_dir = run_dir / "figures"
    weights_dir.mkdir(exist_ok=True)
    fig_dir.mkdir(exist_ok=True)

    # ── 模型训练 ─────────────────────────────────────────────────────────────
    # TODO (Codex):
    #   model = build_model(args.look_back, n_features, args.d_model,
    #                       args.num_heads, args.num_layers, args.dff, args.dropout)
    #   callbacks: EarlyStopping(patience=15, restore_best_weights=True)
    #              ReduceLROnPlateau(factor=0.5, patience=8, min_lr=1e-6)
    #              ModelCheckpoint(weights_dir/"transformer_best.h5", save_best_only=True)
    #   history = model.fit(X_train, y_train,
    #                       validation_data=(X_val, y_val),
    #                       epochs=args.epochs, batch_size=args.batch_size,
    #                       callbacks=callbacks)
    #   model.save(weights_dir / "transformer_8features.h5")

    # ── 预测与反归一化 ───────────────────────────────────────────────────────
    # TODO (Codex):
    #   与 GRU 脚本完全一致：
    #   y_pred_scaled = model.predict(X_test).ravel()
    #   y_true = scaler.inverse_transform(y_test.reshape(-1,1)).ravel()
    #   y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1,1)).ravel()
    #   y_pred = np.clip(y_pred, 0, None)

    # ── 评估（复用统一框架）────────────────────────────────────────────────
    # TODO (Codex):
    #   evaluate_and_save_run(
    #       str(run_dir), model_name="transformer", feature_count=n_features,
    #       y_true=y_true, y_pred=y_pred,
    #       dates=pd.to_datetime(test_df["date"]).values,
    #       horizon=1, peak_threshold=dynamic_peak_threshold,
    #       warning_temperature=1000.0, fn_fp_cost_ratio=(5.0, 1.0),
    #       weather_precip=..., weather_temp_high=..., weather_temp_low=...,
    #       weather_train_precip=..., weather_train_temp_high=..., weather_train_temp_low=...,
    #       extra_meta={"d_model": args.d_model, "num_heads": args.num_heads,
    #                   "num_layers": args.num_layers, "look_back": args.look_back},
    #   )

    print("Transformer 框架骨架已就绪，等待 Codex 补全 TODO 部分。")


if __name__ == "__main__":
    main()
