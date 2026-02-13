# Forecast Pipeline

## Directory Structure

- `models/common/preprocess.py`: 通用预处理
- `models/lstm/train_lstm.py`: LSTM 训练脚本
- `run_pipeline.py`: 一键运行入口

## Run

```bash
python run_pipeline.py
```

仅训练（跳过预处理）：

```bash
python run_pipeline.py --skip-preprocess
```

## Output Layout

- 每次运行一个独立轮次目录（固定命名规范）：
  - `output/runs/run_YYYYMMDD_HHMMSS_lb30_ep120/`
- 该目录下包含：
  - `lstm_history.csv`
  - `lstm_metrics.json`
  - `lstm_metrics.csv`
  - `lstm_test_predictions.csv`
  - `figures/*.png`（若启用可视化）

- 模型文件单独放在：
  - `model/runs/run_YYYYMMDD_HHMMSS_lb30_ep120/lstm_jiuzhaigou.keras`

## Run Name Rule

- 允许自动生成或手动指定 `--run-name`
- 手动指定时必须匹配：
  - `run_YYYYMMDD_HHMMSS_lb<lookback>_ep<epochs>`
