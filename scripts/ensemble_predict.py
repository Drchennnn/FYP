from __future__ import annotations

import glob
import os
from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_absolute_error


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_CSV = PROJECT_ROOT / "output" / "ensemble_test_predictions.csv"


def find_latest_pred_csv(model_prefix: str) -> str | None:
    pattern = str(PROJECT_ROOT / f"output/runs/{model_prefix}_*/runs/run_*/*_test_predictions.csv")
    files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    return files[0] if files else None


def pick_cols(df: pd.DataFrame) -> tuple[str, str]:
    y_true_col = "y_true" if "y_true" in df.columns else "y_true_h1"
    y_pred_col = "y_pred" if "y_pred" in df.columns else "y_pred_h1"
    if y_true_col not in df.columns or y_pred_col not in df.columns:
        raise ValueError(f"Prediction CSV missing expected columns. got={list(df.columns)}")
    return y_true_col, y_pred_col


def main() -> None:
    gru_csv = find_latest_pred_csv("gru_8features")
    tr_csv = find_latest_pred_csv("transformer_8features")
    xgb_csv = find_latest_pred_csv("xgboost_8features")

    paths = {"gru": gru_csv, "transformer": tr_csv, "xgboost": xgb_csv}
    missing = [k for k, v in paths.items() if not v]
    if missing:
        raise FileNotFoundError(f"Missing prediction CSV for: {', '.join(missing)}")

    dfs: dict[str, pd.DataFrame] = {}
    for name, path in paths.items():
        df = pd.read_csv(path)
        y_true_col, y_pred_col = pick_cols(df)
        df["date"] = pd.to_datetime(df["date"]).dt.date
        dfs[name] = (
            df.set_index("date")[[y_true_col, y_pred_col]]
            .rename(columns={y_true_col: "y_true", y_pred_col: name})
        )
        print(f"{name:12s} CSV: {path}")

    base = dfs["gru"][["y_true"]].copy()
    for name, df_m in dfs.items():
        base[name] = df_m[name]
    base = base.dropna(subset=["gru", "transformer", "xgboost"])

    w = {"gru": 0.3, "transformer": 0.2, "xgboost": 0.5}
    base["ensemble"] = sum(base[k] * wt for k, wt in w.items())

    mask = base["y_true"].notna()
    for col in ["gru", "transformer", "xgboost", "ensemble"]:
        mae = mean_absolute_error(base.loc[mask, "y_true"], base.loc[mask, col])
        print(f"{col:12s} MAE: {mae:.2f}")

    out = base.reset_index()
    out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"Saved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
