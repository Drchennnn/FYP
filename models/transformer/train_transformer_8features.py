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

LOOK_BACK   = 45          # 与 GRU/LSTM 保持一致
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10
PRED_HORIZON = 3


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
    df = pd.read_csv(input_csv, encoding="utf-8-sig")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)
    df = df[df["date"] >= "2023-06-01"].reset_index(drop=True)

    if "tourism_num" in df.columns:
        target_col = "tourism_num"
    elif "visitor_count" in df.columns:
        target_col = "visitor_count"
    else:
        raise ValueError("未找到目标列，请包含 'tourism_num' 或 'visitor_count'。")

    df["visitor_count"] = pd.to_numeric(df[target_col], errors="coerce")
    df = df.dropna(subset=["visitor_count"]).reset_index(drop=True)

    df["month_norm"] = (df["date"].dt.month - 1) / 11.0
    df["day_of_week_norm"] = df["date"].dt.weekday / 6.0
    df["is_holiday"] = df["date"].apply(mark_core_holiday).astype(float)
    df["is_peak_season"] = df["date"].apply(is_peak_season).astype(float)

    # P0 扩展：lag_14 + rolling_mean_14
    df["tourism_num_lag_7"] = df["visitor_count"].shift(7)
    df["tourism_num_lag_14"] = df["visitor_count"].shift(14)
    df["tourism_num_rolling_mean_14"] = df["visitor_count"].rolling(14).mean()

    def _pick_col(candidates: list[str]) -> str | None:
        for col in candidates:
            if col in df.columns:
                return col
        return None

    precip_src = _pick_col(["meteo_precip_sum", "meteo_rain_sum", "precip_sum"])
    temp_high_src = _pick_col(["meteo_temp_max", "temp_high_c", "temp_high"])
    temp_low_src = _pick_col(["meteo_temp_min", "temp_low_c", "temp_low"])

    if precip_src is None:
        df["precip_raw"] = 0.0
    else:
        df["precip_raw"] = pd.to_numeric(df[precip_src], errors="coerce")

    if temp_high_src is None:
        df["temp_high_raw"] = 0.0
    else:
        df["temp_high_raw"] = pd.to_numeric(df[temp_high_src], errors="coerce")

    if temp_low_src is None:
        df["temp_low_raw"] = 0.0
    else:
        df["temp_low_raw"] = pd.to_numeric(df[temp_low_src], errors="coerce")

    df = df.dropna(
        subset=[
            "visitor_count",
            "tourism_num_lag_7",
            "tourism_num_lag_14",
            "tourism_num_rolling_mean_14",
            "precip_raw",
            "temp_high_raw",
            "temp_low_raw",
        ]
    ).reset_index(drop=True)

    train_end = int(len(df) * TRAIN_RATIO)
    if train_end <= 0:
        raise ValueError("数据量不足，无法完成训练集拟合。")

    def _fit_transform_col(raw_col: str, scaled_col: str) -> MinMaxScaler:
        scaler = MinMaxScaler()
        scaler.fit(df.iloc[:train_end][[raw_col]])
        df[scaled_col] = scaler.transform(df[[raw_col]]).reshape(-1)
        return scaler

    visitor_scaler = _fit_transform_col("visitor_count", "visitor_count_scaled")
    _fit_transform_col("tourism_num_lag_7", "tourism_num_lag_7_scaled")
    _fit_transform_col("tourism_num_lag_14", "tourism_num_lag_14_scaled")
    _fit_transform_col("tourism_num_rolling_mean_14", "rolling_mean_14_scaled")
    _fit_transform_col("precip_raw", "meteo_precip_sum_scaled")
    _fit_transform_col("temp_high_raw", "temp_high_scaled")
    _fit_transform_col("temp_low_raw", "temp_low_scaled")

    return df, visitor_scaler


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
    x_list, y_list = [], []
    for i in range(look_back, len(data) - PRED_HORIZON + 1):
        x_list.append(data[i - look_back : i, :])
        y_list.append(data[i : i + PRED_HORIZON, 0])
    return np.asarray(x_list, dtype=np.float32), np.asarray(y_list, dtype=np.float32)


def compute_sample_weights(months: np.ndarray) -> np.ndarray:
    """旺季(4-11月)权重×2，淡季权重×1"""
    weights = np.where(
        np.isin(months, [4, 5, 6, 7, 8, 9, 10, 11]),
        2.0,
        1.0,
    )
    return weights.astype(np.float32)


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
        positions = np.arange(max_len)[:, np.newaxis]
        dims = np.arange(d_model)[np.newaxis, :]
        angles = positions / np.power(10000.0, (2 * (dims // 2)) / float(d_model))
        angles[:, 0::2] = np.sin(angles[:, 0::2])
        angles[:, 1::2] = np.cos(angles[:, 1::2])
        self.pe = tf.cast(angles[np.newaxis, :, :], tf.float32)

    def call(self, x):
        return x + self.pe[:, :tf.shape(x)[1], :]


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
        if d_model % num_heads != 0:
            raise ValueError(f"d_model={d_model} 必须能被 num_heads={num_heads} 整除。")
        self.mha = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model // num_heads
        )
        self.ffn1 = tf.keras.layers.Dense(dff, activation="relu")
        self.ffn2 = tf.keras.layers.Dense(d_model)
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = tf.keras.layers.Dropout(dropout_rate)
        self.drop2 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, training=False):
        attn = self.mha(x, x, training=training)
        attn = self.drop1(attn, training=training)
        x = self.norm1(x + attn)
        ffn = self.ffn2(self.ffn1(x))
        ffn = self.drop2(ffn, training=training)
        return self.norm2(x + ffn)


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
    inp = tf.keras.Input(shape=(look_back, n_features))
    x = tf.keras.layers.Dense(d_model)(inp)
    x = PositionalEncoding(d_model)(x)
    for _ in range(num_layers):
        x = TransformerEncoderBlock(num_heads, d_model, dff, dropout_rate)(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    out = tf.keras.layers.Dense(PRED_HORIZON)(x)

    model = tf.keras.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.Huber(),
    )
    return model


# ── 主流程 ────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Transformer Encoder 客流预测")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--look-back", type=int, default=LOOK_BACK)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dff", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--peak-quantile", type=float, default=0.75)
    args = parser.parse_args()

    np.random.seed(42)
    tf.random.set_seed(42)

    # ── 数据准备 ────────────────────────────────────────────────────────────
    df, scaler = load_and_engineer_features(args.data)
    feature_cols = [
        "visitor_count_scaled",
        "month_norm",
        "day_of_week_norm",
        "is_holiday",
        "is_peak_season",
        "tourism_num_lag_7_scaled",
        "tourism_num_lag_14_scaled",
        "rolling_mean_14_scaled",
        "meteo_precip_sum_scaled",
        "temp_high_scaled",
        "temp_low_scaled",
    ]

    data = df[feature_cols].values.astype(np.float32)
    X_all, y_all = build_sequences(data, args.look_back)
    n_samples = len(X_all)
    all_dates = pd.to_datetime(df["date"]).values[args.look_back : args.look_back + n_samples]
    precip_raw_all = df["precip_raw"].values.astype(float)
    temp_high_raw_all = df["temp_high_raw"].values.astype(float)
    temp_low_raw_all = df["temp_low_raw"].values.astype(float)
    weather_precip_all = np.stack(
        [precip_raw_all[args.look_back + h : args.look_back + h + n_samples] for h in range(PRED_HORIZON)],
        axis=1,
    )
    weather_temp_high_all = np.stack(
        [temp_high_raw_all[args.look_back + h : args.look_back + h + n_samples] for h in range(PRED_HORIZON)],
        axis=1,
    )
    weather_temp_low_all = np.stack(
        [temp_low_raw_all[args.look_back + h : args.look_back + h + n_samples] for h in range(PRED_HORIZON)],
        axis=1,
    )

    n = len(X_all)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

    X_train, y_train = X_all[:train_end], y_all[:train_end]
    X_val, y_val = X_all[train_end:val_end], y_all[train_end:val_end]
    X_test, y_test = X_all[val_end:], y_all[val_end:]
    dates_train = all_dates[:train_end]
    d_test = all_dates[val_end:]

    weather_train_precip = weather_precip_all[:train_end]
    weather_train_temp_high = weather_temp_high_all[:train_end]
    weather_train_temp_low = weather_temp_low_all[:train_end]
    weather_test_precip = weather_precip_all[val_end:]
    weather_test_temp_high = weather_temp_high_all[val_end:]
    weather_test_temp_low = weather_temp_low_all[val_end:]

    train_visitor_counts = scaler.inverse_transform(y_train.reshape(-1, 1)).reshape(-1)
    dynamic_peak_threshold = compute_dynamic_peak_threshold(
        train_visitor_counts, quantile=args.peak_quantile
    )

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
    n_features = len(feature_cols)
    model = build_model(
        args.look_back,
        n_features,
        args.d_model,
        args.num_heads,
        args.num_layers,
        args.dff,
        args.dropout,
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=20, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=8, min_lr=1e-6
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(weights_dir / "transformer_best.h5"),
            monitor="val_loss",
            save_best_only=True,
        ),
    ]
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        shuffle=False,
        callbacks=callbacks,
        sample_weight=compute_sample_weights(pd.to_datetime(dates_train).month.values),
        verbose=1,
    )
    model.save(weights_dir / "transformer_8features.h5")

    # ── 预测与反归一化 ───────────────────────────────────────────────────────
    y_pred_scaled = model.predict(X_test, verbose=0)
    y_true = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(-1, PRED_HORIZON)
    y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).reshape(-1, PRED_HORIZON)
    y_pred = np.clip(y_pred, 0, None)
    print(f"y_pred shape: {y_pred.shape}")
    for h in range(PRED_HORIZON):
        mae_h = np.mean(np.abs(y_true[:, h] - y_pred[:, h]))
        print(f"day+{h+1} MAE: {mae_h:.2f}")

    pred_rows = []
    pred_dates = []
    for i in range(len(d_test)):
        base_date = pd.to_datetime(d_test[i])
        for h in range(PRED_HORIZON):
            target_date = base_date + pd.Timedelta(days=h)
            pred_dates.append(target_date)
            pred_rows.append(
                {
                    "date": target_date,
                    "horizon": h + 1,
                    "y_true": y_true[i, h],
                    "y_pred": y_pred[i, h],
                }
            )
    pred_df = pd.DataFrame(pred_rows).sort_values(["date", "horizon"])
    pred_df.to_csv(run_dir / "transformer_test_predictions.csv", index=False, encoding="utf-8-sig")

    # ── 评估（复用统一框架）────────────────────────────────────────────────
    evaluate_and_save_run(
        str(run_dir),
        model_name="transformer",
        feature_count=n_features,
        y_true=np.asarray(y_true),
        y_pred=np.asarray(y_pred),
        dates=pred_dates,
        horizon=PRED_HORIZON,
        peak_threshold=dynamic_peak_threshold,
        warning_temperature=1000.0,
        fn_fp_cost_ratio=(5.0, 1.0),
        weather_precip=np.asarray(weather_test_precip),
        weather_temp_high=np.asarray(weather_test_temp_high),
        weather_temp_low=np.asarray(weather_test_temp_low),
        weather_train_precip=np.asarray(weather_train_precip),
        weather_train_temp_high=np.asarray(weather_train_temp_high),
        weather_train_temp_low=np.asarray(weather_train_temp_low),
        extra_meta={
            "d_model": int(args.d_model),
            "num_heads": int(args.num_heads),
            "num_layers": int(args.num_layers),
            "dff": int(args.dff),
            "dropout": float(args.dropout),
            "look_back": int(args.look_back),
            "epochs_requested": int(args.epochs),
            "epochs_trained": int(len(history.history.get("loss", []))),
            "peak_threshold_source": "dynamic_train_quantile",
            "peak_quantile": float(args.peak_quantile),
        },
    )

    metrics = calculate_metrics(y_true=y_true.reshape(-1), y_pred=y_pred.reshape(-1), scaler=scaler)
    try:
        save_metrics_to_files(metrics, str(run_dir), "transformer_baseline")
    except TypeError:
        pass

    print("Transformer 训练完成！")
    print(f"运行目录: {run_dir}")
    print(f"特征维度: {n_features}")
    print(f"MAE: {metrics['regression']['mae']:.4f}")


if __name__ == "__main__":
    main()
