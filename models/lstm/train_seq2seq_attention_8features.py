"""九寨沟客流预测 Seq2Seq + Attention 训练脚本（专业版）。

主要改进：
1. 升级为 Many-to-Many Seq2Seq 架构，避免滚动预测误差累积
2. 集成 Bahdanau Attention 机制，动态分配特征权重（支持 is_holiday 等特征）
3. 自定义非对称损失函数，对节假日预测偏低给予更高惩罚
4. 使用双向 LSTM 编码器 + 单向 LSTM 解码器架构
5. 1D CNN 特征压缩，优化特征维度到 128 维

架构设计方案已通过审核，实现如下。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Tuple, List

import chinese_calendar as cncal
import matplotlib
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import MinMaxScaler

# 导入通用评估器
# Ensure project root is on sys.path when running as a script
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.common.core_evaluation import evaluate_and_save_run

matplotlib.use("Agg")


def mark_core_holiday(date_val: pd.Timestamp) -> int:
    """核心节假日标记函数。

    显式标记：
    - 国庆：10/01 - 10/07
    - 劳动节：05/01 - 05/05
    - 春节及其他法定节假日：由 chinese_calendar 动态判断
    """
    m, d = int(date_val.month), int(date_val.day)

    # 国庆黄金周
    if m == 10 and 1 <= d <= 7:
        return 1
    # 劳动节窗口
    if m == 5 and 1 <= d <= 5:
        return 1

    # 动态法定节假日（覆盖春节等）
    try:
        return int(cncal.is_holiday(date_val.date()))
    except Exception:
        return 0


def load_and_engineer_features(input_csv: Path) -> pd.DataFrame:
    """加载数据并进行特征工程（8特征版本）
    
    包含8个核心特征：
    1. visitor_count_scaled (目标值，归一化)
    2. month_norm
    3. day_of_week_norm
    4. is_holiday
    5. tourism_num_lag_7_scaled
    6. meteo_precip_sum_scaled
    7. temp_high_scaled
    8. temp_low_scaled
    
    Args:
        input_csv: 输入CSV路径
        
    Returns:
        包含8个特征的DataFrame
    """
    df = pd.read_csv(input_csv, encoding="utf-8-sig")

    if "tourism_num" in df.columns:
        target_col = "tourism_num"
    elif "visitor_count" in df.columns:
        target_col = "visitor_count"
    else:
        raise ValueError("未找到目标列，请包含 'tourism_num' 或 'visitor_count'。")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)
    df["visitor_count"] = pd.to_numeric(df[target_col], errors="coerce")
    df = df.dropna(subset=["visitor_count"]).reset_index(drop=True)

    # 必要时间特征
    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.weekday
    df["is_holiday"] = df["date"].apply(mark_core_holiday).astype(float)

    # 时间特征归一化
    df["month_norm"] = (df["month"] - 1) / 11.0
    df["day_of_week_norm"] = df["day_of_week"] / 6.0

    # 检查并确保所有8个特征都存在
    required_features = [
        "tourism_num_lag_7_scaled",
        "meteo_precip_sum_scaled", 
        "temp_high_scaled",
        "temp_low_scaled"
    ]
    
    for feature in required_features:
        if feature not in df.columns:
            print(f"警告: 特征 '{feature}' 不存在，将用0填充")
            df[feature] = 0.0
    
    return df


class AttentionLayer(tf.keras.layers.Layer):
    """Bahdanau 注意力机制层（动态权重分配）
    
    输入:
        - encoder_outputs: Encoder的输出序列 (batch, encoder_steps, units)
        - decoder_hidden: Decoder的隐藏状态 (batch, units)
    
    输出:
        - context_vector: 注意力权重加权后的上下文向量 (batch, units)
        - attention_weights: 注意力权重分布 (batch, encoder_steps, 1)
    """
    def __init__(self, units):
        super(AttentionLayer, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, decoder_hidden, encoder_outputs):
        # 扩展维度以便广播加法
        hidden_with_time_axis = tf.expand_dims(decoder_hidden, 1)
        
        # 计算注意力分数
        score = self.V(tf.nn.tanh(
            self.W1(encoder_outputs) + self.W2(hidden_with_time_axis)
        ))
        
        # 计算权重分布
        attention_weights = tf.nn.softmax(score, axis=1)
        
        # 加权求和得到上下文向量
        context_vector = attention_weights * encoder_outputs
        context_vector = tf.reduce_sum(context_vector, axis=1)
        
        return context_vector, attention_weights


class Seq2SeqWithAttention(tf.keras.Model):
    """带有注意力机制的非自回归Seq2Seq模型（支持8/7特征输入）
    
    架构说明：
    - Encoder: 双向LSTM，将30步历史序列编码为隐藏状态（8个特征）
    - Decoder: 单向LSTM + Bahdanau注意力，一次性输出未来步预测（7个外部特征）
    - 特征压缩: 1D卷积压缩（Encoder: 8→128维，Decoder: 7→128维）
    
    核心优化：
    - Decoder输入不含visitor_count，纯外部特征驱动
    - 杜绝自回归毒药，直接多步预测
    """
    def __init__(self, encoder_units, decoder_units, encoder_features=8, decoder_features=7):
        super(Seq2SeqWithAttention, self).__init__()
        self.encoder_units = encoder_units
        self.decoder_units = decoder_units
        self.encoder_features = encoder_features
        self.decoder_features = decoder_features
        
        # Encoder特征压缩层：将8个原始特征压缩到128维
        self.encoder_feature_compress = tf.keras.Sequential([
            tf.keras.layers.Conv1D(filters=64, kernel_size=1, activation='relu'),
            tf.keras.layers.Conv1D(filters=128, kernel_size=1, activation='relu')
        ])
        
        # Decoder特征压缩层：将7个外部特征压缩到128维
        self.decoder_feature_compress = tf.keras.Sequential([
            tf.keras.layers.Conv1D(filters=64, kernel_size=1, activation='relu'),
            tf.keras.layers.Conv1D(filters=128, kernel_size=1, activation='relu')
        ])
        
        # Encoder：双向LSTM
        self.encoder = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                encoder_units,
                return_sequences=True,
                return_state=True,
                dropout=0.2,
                recurrent_dropout=0.2
            )
        )
        
        # 状态转换层：解决双向LSTM输出维度翻倍问题
        self.state_transition = tf.keras.layers.Dense(decoder_units)
        
        # Decoder LSTM
        self.decoder_lstm = tf.keras.layers.LSTM(
            decoder_units,
            return_sequences=True,
            return_state=True,
            dropout=0.2,
            recurrent_dropout=0.2
        )
        
        # Attention机制
        self.attention = AttentionLayer(decoder_units)
        
        # 输出层：预测客流值
        self.fc = tf.keras.layers.Dense(1)
        
        # 特征权重分析层（用于可视化）
        self.feature_importance = tf.keras.layers.Dense(decoder_features, activation='softmax')

    def call(self, inputs, training=True):
        """前向传播（非自回归版本）
        
        Args:
            inputs: [encoder_input, decoder_input]
            encoder_input: (batch, encoder_steps, 8)  - 历史30天数据
            decoder_input: (batch, decoder_steps, 7)  - 预测未来7天的外部特征
            
        Returns:
            predictions: (batch, decoder_steps, 1)    - 预测值（只返回这个用于训练）
        """
        encoder_input, decoder_input = inputs
        
        # 步骤1：特征压缩
        encoder_compressed = self.encoder_feature_compress(encoder_input)
        decoder_compressed = self.decoder_feature_compress(decoder_input)
        
        # 步骤2：Encoder编码
        encoder_outputs, forward_h, forward_c, backward_h, backward_c = \
            self.encoder(encoder_compressed, training=training)
        
        # 合并双向LSTM的状态（维度转换）
        encoder_hidden = tf.keras.layers.concatenate([forward_h, backward_h])
        encoder_cell = tf.keras.layers.concatenate([forward_c, backward_c])
        
        # 解决维度匹配问题：双向状态 → Decoder状态
        decoder_initial_hidden = self.state_transition(encoder_hidden)
        decoder_initial_cell = self.state_transition(encoder_cell)
        
        # 步骤3：Decoder初始化
        decoder_hidden = decoder_initial_hidden
        decoder_cell = decoder_initial_cell
        decoder_outputs = []
        
        # 步骤4：逐个时间步解码（with Attention）
        for t in range(decoder_input.shape[1]):
            # 获取当前时间步的特征
            decoder_input_t = tf.expand_dims(decoder_input[:, t, :], axis=1)
            
            # 注意力机制：获取上下文向量
            context_vector, attention_weights = self.attention(
                decoder_hidden, encoder_outputs
            )
            
            # 拼接上下文向量与当前特征
            decoder_input_with_context = tf.concat([
                tf.expand_dims(context_vector, 1), 
                decoder_compressed[:, t:t+1, :]
            ], axis=-1)
            
            # LSTM前向传播
            decoder_output, decoder_hidden, decoder_cell = self.decoder_lstm(
                decoder_input_with_context,
                initial_state=[decoder_hidden, decoder_cell],
                training=training
            )
            
            # 输出预测值
            prediction = self.fc(decoder_output)
            decoder_outputs.append(prediction)
        
        # 整理输出
        predictions = tf.concat(decoder_outputs, axis=1)
        
        return predictions


def create_custom_asymmetric_loss(holiday_weight=2.0, peak_weight=1.5):
    """创建自定义非对称损失函数（解决高峰期预测偏低问题）
    
    设计理念：
    - 常规样本：正常权重
    - 节假日预测偏低：权重加倍 (holiday_weight=2.0)
    - 非节假日但高客流预测偏低：权重增大 (peak_weight=1.5)
    
    Args:
        holiday_weight: 节假日预测偏低的惩罚权重
        peak_weight: 高客流预测偏低的惩罚权重
        
    Returns:
        custom_loss_fn: 自定义损失函数
    """
    def custom_loss(y_true, y_pred):
        # 确保维度一致性
        y_pred = tf.reshape(y_pred, tf.shape(y_true[:, :, :1]))
        
        # 获取节假日特征
        is_holiday = y_true[:, :, 1]  # y_true 格式: [value, is_holiday]
        
        # 计算基础误差
        error = tf.abs(y_true[:, :, 0] - y_pred[:, :, 0])
        
        # 计算非对称权重
        weight = tf.ones_like(error)
        
        # 情况1：节假日且预测偏低
        holiday_underpred = tf.logical_and(
            tf.cast(is_holiday, tf.bool),
            y_pred[:, :, 0] < y_true[:, :, 0]
        )
        weight = tf.where(holiday_underpred, holiday_weight, weight)
        
        # 情况2：非节假日但高客流且预测偏低
        peak_underpred = tf.logical_and(
            tf.logical_not(tf.cast(is_holiday, tf.bool)),
            tf.logical_and(
                y_true[:, :, 0] > 0.75,  # 归一化后的高客流阈值
                y_pred[:, :, 0] < y_true[:, :, 0]
            )
        )
        weight = tf.where(peak_underpred, peak_weight, weight)
        
        # 加权误差
        weighted_error = error * weight
        
        return tf.reduce_mean(weighted_error)
        
    return custom_loss


def prepare_seq2seq_data(
    df: pd.DataFrame,
    encoder_steps: int = 30,
    decoder_steps: int = 7,
    test_ratio: float = 0.2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """准备非自回归Seq2Seq模型的训练/测试数据
    
    架构：非自回归直接多步预测 + Attention
    - Encoder输入：历史30天的8个特征（包含visitor_count）
    - Decoder输入：未来7天的外部特征（不含visitor_count，杜绝自回归）
    - 特征列表：[month_norm, day_of_week_norm, is_holiday, tourism_num_lag_7_scaled, meteo_precip_sum_scaled, temp_high_scaled, temp_low_scaled]
    
    Args:
        df: 特征工程后的DataFrame
        encoder_steps: Encoder输入序列长度（历史天数）
        decoder_steps: Decoder输出序列长度（预测天数）
        test_ratio: 测试集比例
        
    Returns:
        x_encoder_train, x_decoder_train, y_train,
        x_encoder_test, x_decoder_test, y_test,
        weather_train, weather_test

    weather_* have shape (N, decoder_steps, 3) in real units:
      [:, :, 0] precip_sum
      [:, :, 1] temp_high
      [:, :, 2] temp_low
    """
    # Encoder特征列（包含visitor_count）
    encoder_feature_cols = [
        "visitor_count_scaled",      # 目标客流（归一化）
        "month_norm",                # 月份归一化
        "day_of_week_norm",          # 星期归一化
        "is_holiday",                # 节假日标记
        "tourism_num_lag_7_scaled",  # 滞后7天客流（归一化）
        "meteo_precip_sum_scaled",   # 降水量（归一化）
        "temp_high_scaled",          # 最高温度（归一化）
        "temp_low_scaled"            # 最低温度（归一化）
    ]
    
    # Decoder特征列（纯外部特征，不含visitor_count，杜绝自回归）
    decoder_feature_cols = [
        "month_norm",                # 月份归一化
        "day_of_week_norm",          # 星期归一化
        "is_holiday",                # 节假日标记
        "tourism_num_lag_7_scaled",  # 滞后7天客流（归一化）
        "meteo_precip_sum_scaled",   # 降水量（归一化）
        "temp_high_scaled",          # 最高温度（归一化）
        "temp_low_scaled"            # 最低温度（归一化）
    ]
    
    values_encoder = df[encoder_feature_cols].values.astype(np.float32)
    values_decoder = df[decoder_feature_cols].values.astype(np.float32)
    dates = df["date"].values

    # Weather columns (real units; treated as exogenous forecast/observed from dataset)
    def _pick_col(candidates):
        for c in candidates:
            if c in df.columns:
                return c
        return None

    precip_col = _pick_col(["meteo_precip_sum", "meteo_rain_sum", "precip_sum"])
    temp_high_col = _pick_col(["meteo_temp_max", "temp_high_c", "temp_high"])
    temp_low_col = _pick_col(["meteo_temp_min", "temp_low_c", "temp_low"])
    if precip_col is None or temp_high_col is None or temp_low_col is None:
        raise ValueError(
            f"Missing required weather columns for hazard: precip={precip_col}, temp_high={temp_high_col}, temp_low={temp_low_col}"
        )

    precip_real = df[precip_col].values.astype(float)
    temp_high_real = df[temp_high_col].values.astype(float)
    temp_low_real = df[temp_low_col].values.astype(float)
    
    # 创建序列数据
    encoder_input = []
    decoder_input = []
    decoder_target = []
    weather_target = []
    
    total_steps = encoder_steps + decoder_steps
    for i in range(total_steps, len(df)):
        # Encoder输入：历史encoder_steps天的8个特征
        enc_seq = values_encoder[i - total_steps : i - decoder_steps, :]
        encoder_input.append(enc_seq)
        
        # Decoder输入：未来decoder_steps天的外部特征（不含visitor_count，纯外部驱动）
        dec_seq = values_decoder[i - decoder_steps : i, :]
        decoder_input.append(dec_seq)
        
        # Decoder目标：未来decoder_steps天的目标值 + 节假日标记
        dec_target = []
        dec_weather = []
        for j in range(decoder_steps):
            idx = i - decoder_steps + j
            # 格式：[真实值, 节假日标记]
            dec_target.append([values_encoder[idx, 0], values_encoder[idx, 3]])
            dec_weather.append([precip_real[idx], temp_high_real[idx], temp_low_real[idx]])
        decoder_target.append(dec_target)
        weather_target.append(dec_weather)
    
    encoder_input = np.array(encoder_input)
    decoder_input = np.array(decoder_input)
    decoder_target = np.array(decoder_target)
    weather_target = np.array(weather_target, dtype=float)
    
    # 时间划分训练/测试集
    n = len(encoder_input)
    test_size = int(n * test_ratio)
    train_size = n - test_size
    
    x_encoder_train = encoder_input[:train_size]
    x_decoder_train = decoder_input[:train_size]
    y_train = decoder_target[:train_size]
    
    x_encoder_test = encoder_input[train_size:]
    x_decoder_test = decoder_input[train_size:]
    y_test = decoder_target[train_size:]

    weather_train = weather_target[:train_size]
    weather_test = weather_target[train_size:]
    
    print(f"数据准备完成:")
    print(f"  总样本数: {n}")
    print(f"  训练样本: {train_size}, 测试样本: {test_size}")
    print(f"  Encoder输入形状: {x_encoder_train.shape}")
    print(f"  Decoder输入形状: {x_decoder_train.shape}")
    print(f"  目标形状: {y_train.shape}")
    
    return (
        x_encoder_train,
        x_decoder_train,
        y_train,
        x_encoder_test,
        x_decoder_test,
        y_test,
        weather_train,
        weather_test,
        np.array([precip_col, temp_high_col, temp_low_col], dtype=object),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Seq2Seq + Attention 客流预测训练脚本。")
    parser.add_argument(
        "--input-csv",
        default="data/processed/jiuzhaigou_8features_latest.csv",
        help="输入 CSV（需包含 date + 客流列）。",
    )
    parser.add_argument("--encoder-steps", type=int, default=30, help="历史窗口长度。")
    parser.add_argument("--decoder-steps", type=int, default=7, help="预测天数。")
    parser.add_argument("--epochs", type=int, default=120, help="训练轮次（建议 >=100）。")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--peak-quantile", type=float, default=0.75)
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--model-dir", default="model")
    parser.add_argument(
        "--run-name",
        default=None,
        help="可选轮次目录名，不传则自动按时间戳生成。",
    )
    parser.add_argument(
        "--save-plots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否保存可视化图。默认 True。",
    )
    args, _ = parser.parse_known_args()

    np.random.seed(42)
    tf.random.set_seed(42)

    output_dir = Path(args.output_dir)
    model_root_dir = Path(args.model_dir)
    runs_dir = output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    model_runs_dir = model_root_dir / "runs"
    model_runs_dir.mkdir(parents=True, exist_ok=True)
    auto_run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_lb{args.encoder_steps}_ep{args.epochs}_seq2seq_attention_8features"
    run_name = args.run_name or auto_run_name
    run_name_pattern = r"^run_\d{8}_\d{6}_.+$"
    if not re.fullmatch(run_name_pattern, run_name):
        raise ValueError(
            "run_name 格式不符合要求，应为：run_YYYYMMDD_HHMMSS_lb<lookback>_ep<epochs>"
        )
    run_dir = runs_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建必要的子目录
    weights_dir = run_dir / "weights"
    fig_dir = run_dir / "figures"
    weights_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 模型权重保存在 output/runs/<run_name>/weights/ 目录中
    model_path = weights_dir / "seq2seq_jiuzhaigou.keras"
    pred_path = run_dir / "seq2seq_test_predictions.csv"
    history_path = run_dir / "seq2seq_history.csv"

    # 1. 加载数据并特征工程
    df = load_and_engineer_features(Path(args.input_csv))

    # 2. 对目标客流做 MinMax 归一化
    scaler = MinMaxScaler()
    df["visitor_count_scaled"] = scaler.fit_transform(df[["visitor_count"]]).reshape(-1)

    # 3. 准备Seq2Seq数据
    (
        x_encoder_train,
        x_decoder_train,
        y_train,
        x_encoder_test,
        x_decoder_test,
        y_test,
        weather_train,
        weather_test,
        weather_cols,
    ) = prepare_seq2seq_data(
        df,
        encoder_steps=args.encoder_steps,
        decoder_steps=args.decoder_steps,
        test_ratio=args.test_ratio,
    )

    # 4. 创建Seq2Seq + Attention模型
    model = Seq2SeqWithAttention(
        encoder_units=128,
        decoder_units=256,  # 解决双向LSTM输出维度翻倍问题
        encoder_features=8,
        decoder_features=7
    )

    # 5. 编译模型 - 使用自定义非对称损失
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=create_custom_asymmetric_loss(),
        metrics=["mae"]
    )

    # 6. 训练模型
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=15,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=7,
            min_lr=1e-5,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            str(model_path),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,
        )
    ]

    history = model.fit(
        [x_encoder_train, x_decoder_train],
        y_train,  # 传递完整的目标值（包含真实值和节假日标记）
        validation_split=0.15,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1,
        shuffle=True
    )

    # 7. 预测：非自回归直接多步预测（纯外部特征驱动）
    print("使用非自回归直接多步预测模式进行测试阶段预测...")
    
    # 在非自回归架构中，Decoder输入已经是纯外部特征（不含visitor_count）
    # 我们直接使用准备好的数据
    y_pred = model.predict(
        [x_encoder_test, x_decoder_test],
        verbose=0
    )
    
    # 8. 反归一化
    y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1)).reshape(-1, args.decoder_steps)
    y_true = scaler.inverse_transform(y_test[:, :, 0].reshape(-1, 1)).reshape(-1, args.decoder_steps)

    # 10. 保存模型
    model.save(model_path)
    
    extra_meta = {
        "samples": int(len(df)),
        "encoder_steps": int(args.encoder_steps),
        "decoder_steps": int(args.decoder_steps),
        "epochs_requested": int(args.epochs),
        "epochs_trained": int(len(history.history["loss"])),
        "train_samples": int(len(x_encoder_train)),
        "val_samples": int(len(x_encoder_train) * 0.15),
        "test_samples": int(len(x_encoder_test)),
        "features": [
            "visitor_count_scaled",      # 目标客流（归一化）
            "month_norm",                # 月份归一化
            "day_of_week_norm",          # 星期归一化
            "is_holiday",                # 节假日标记
            "tourism_num_lag_7_scaled",  # 滞后7天客流（归一化）
            "meteo_precip_sum_scaled",   # 降水量（归一化）
            "temp_high_scaled",          # 最高温度（归一化）
            "temp_low_scaled"            # 最低温度（归一化）
        ],
        "model_architecture": "Seq2Seq+Attention-128-256",
        "feature_version": "8_features_v1",
        "loss_function": "CustomAsymmetricLoss",
    }

    # 11. 保存训练历史
    history_df = pd.DataFrame(history.history)
    history_df.insert(0, "epoch", np.arange(1, len(history_df) + 1))
    history_df.to_csv(history_path, index=False, encoding="utf-8-sig")

    # 12. 保存测试预测结果 (also used for dates in plots)
    pred_rows = []
    pred_dates = []
    for i in range(len(x_encoder_test)):
        for j in range(args.decoder_steps):
            date_idx = len(df) - len(x_encoder_test) - args.decoder_steps + i + j
            if date_idx >= len(df):
                continue
                
            pred_dates.append(pd.to_datetime(df["date"].iloc[date_idx]))
            pred_rows.append({
                "date": str(df["date"].iloc[date_idx]),
                "y_true": y_true[i, j],
                "y_pred": y_pred[i, j]
            })
    pd.DataFrame(pred_rows).to_csv(pred_path, index=False, encoding="utf-8-sig")

    # 13. Unified core metrics + figures (multi-horizon)
    evaluate_and_save_run(
        run_dir=str(run_dir),
        model_name="seq2seq_attention",
        feature_count=8,
        y_true=y_true,
        y_pred=y_pred,
        dates=pred_dates,
        horizon=int(args.decoder_steps),
        # Weather hazard uses Route B quantile thresholds computed on TRAIN split only.
        # Weather is treated as exogenous (from dataset/API) since the model does not predict weather.
        weather_precip=np.asarray(weather_test[:, :, 0], dtype=float),
        weather_temp_high=np.asarray(weather_test[:, :, 1], dtype=float),
        weather_temp_low=np.asarray(weather_test[:, :, 2], dtype=float),
        weather_train_precip=np.asarray(weather_train[:, :, 0], dtype=float).reshape(-1),
        weather_train_temp_high=np.asarray(weather_train[:, :, 1], dtype=float).reshape(-1),
        weather_train_temp_low=np.asarray(weather_train[:, :, 2], dtype=float).reshape(-1),
        extra_meta=extra_meta,
        save_figures=bool(args.save_plots),
    )

    # 14. 输出结果
    print(f"\n{'='*80}")
    print(f"训练完成！")
    print(f"{'='*80}")
    print(f"运行目录: {run_dir}")
    print(f"模型保存: {model_path}")
    print(f"预测结果: {pred_path}")
    print(f"训练历史: {history_path}")
    
    print("Core metrics saved to:", run_dir / "metrics.json")


if __name__ == "__main__":
    main()
