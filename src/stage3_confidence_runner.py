from __future__ import annotations

from pathlib import Path

import pandas as pd

from .backtesting import run_long_flat_backtest, summarize_backtest
from .config import load_config, resolve_path
from .evaluation import classification_metrics, confidence_summary, predicted_distribution


def apply_confidence_rule(df: pd.DataFrame, threshold: float) -> pd.Series:
    buy = df["probability_buy"]
    sell = df["probability_sell"]
    labels = pd.Series("HOLD", index=df.index, dtype="object")
    buy_active = (buy >= threshold) & (buy > sell)
    sell_active = (sell >= threshold) & (sell > buy)
    labels = labels.mask(buy_active, "BUY")
    labels = labels.mask(sell_active, "SELL")
    return labels


def quick_backtest_score(bt: pd.DataFrame) -> dict[str, float]:
    returns = bt["strategy_return_net"].fillna(0.0)
    total_return = float(bt["equity_curve"].iloc[-1] - 1)
    trade_count = int((((bt["position_prev"] == 0) & (bt["position"] == 1)).astype(int)).sum())
    return {
        "total_return": total_return,
        "num_trades": trade_count,
    }


def run_stage3_confidence(config_path: str | None = None) -> None:
    config = load_config(config_path)
    metrics_dir = resolve_path(config["paths"]["metrics_dir"])
    predictions_dir = resolve_path(config["paths"]["predictions_dir"])
    labeled_path = resolve_path(config["paths"]["labeled_1h"])
    candidates = config["stage3"]["confidence_threshold_candidates"]

    base_predictions = pd.read_parquet(predictions_dir / "stage3_predictions.parquet")
    dual_predictions = pd.read_parquet(predictions_dir / "stage3_dual_binary_predictions.parquet")
    for frame in (base_predictions, dual_predictions):
        frame["datetime_utc"] = pd.to_datetime(frame["datetime_utc"], utc=True)

    lookup = pd.read_parquet(labeled_path).set_index("timestamp")[["Open", "Close"]]
    model_frames = {
        "logistic_regression": base_predictions[base_predictions["model"] == "logistic_regression"].copy(),
        "xgboost": base_predictions[base_predictions["model"] == "xgboost"].copy(),
        "dual_binary_logistic_regression": dual_predictions[dual_predictions["model"] == "dual_binary_logistic_regression"].copy(),
        "dual_binary_xgboost": dual_predictions[dual_predictions["model"] == "dual_binary_xgboost"].copy(),
    }

    selection_rows: list[dict] = []
    final_metric_rows: list[dict] = []
    final_backtest_rows: list[dict] = []
    final_prediction_frames: list[pd.DataFrame] = []

    for model_name, pred_df in model_frames.items():
        candidate_rows: list[dict] = []
        for threshold in candidates:
            fold_rows: list[dict] = []
            for fold, fold_df in pred_df[pred_df["split"] == "validation"].groupby("fold"):
                adjusted = fold_df.copy()
                adjusted["predicted_label"] = apply_confidence_rule(adjusted, threshold)
                bt_input = pd.DataFrame(
                    {
                        "datetime_utc": adjusted["datetime_utc"],
                        "Open": lookup.loc[adjusted["timestamp"], "Open"].to_numpy(),
                        "Close": lookup.loc[adjusted["timestamp"], "Close"].to_numpy(),
                        "signal": adjusted["predicted_label"].to_numpy(),
                    }
                )
                bt = run_long_flat_backtest(bt_input, signal_col="signal", fee=config["backtest"]["fee"], slippage=config["backtest"]["slippage"])
                summary = quick_backtest_score(bt)
                metrics = classification_metrics(adjusted["actual_label"], adjusted["predicted_label"])
                fold_rows.append(
                    {
                        "model": model_name,
                        "threshold": threshold,
                        "fold": fold,
                        "macro_f1": metrics["macro_f1"],
                        "balanced_accuracy": metrics["balanced_accuracy"],
                        "bt_total_return": summary["total_return"],
                        "bt_num_trades": summary["num_trades"],
                    }
                )
            fold_df = pd.DataFrame(fold_rows)
            candidate_rows.append(
                {
                    "model": model_name,
                    "confidence_threshold": threshold,
                    "mean_macro_f1": fold_df["macro_f1"].mean(),
                    "mean_balanced_accuracy": fold_df["balanced_accuracy"].mean(),
                    "mean_bt_total_return": fold_df["bt_total_return"].mean(),
                    "mean_bt_num_trades": fold_df["bt_num_trades"].mean(),
                }
            )
        selection_df = pd.DataFrame(candidate_rows).sort_values(
            ["mean_bt_total_return", "mean_macro_f1", "mean_bt_num_trades"],
            ascending=[False, False, True],
        )
        best_threshold = float(selection_df.iloc[0]["confidence_threshold"])
        selection_rows.extend(selection_df.to_dict("records"))

        for split_name in ["validation", "test"]:
            adjusted = pred_df[pred_df["split"] == split_name].copy()
            adjusted["predicted_label"] = apply_confidence_rule(adjusted, best_threshold)
            metrics = {
                "model": model_name,
                "split": split_name,
                "confidence_threshold": best_threshold,
            }
            metrics.update(classification_metrics(adjusted["actual_label"], adjusted["predicted_label"]))
            metrics.update(predicted_distribution(adjusted["predicted_label"]))
            metrics.update(
                confidence_summary(
                    adjusted[["probability_buy", "probability_hold", "probability_sell"]]
                )
            )
            final_metric_rows.append(metrics)

            adjusted["model"] = f"{model_name}_confthr"
            adjusted["selected_confidence_threshold"] = best_threshold
            final_prediction_frames.append(adjusted)

        test_adjusted = pred_df[pred_df["split"] == "test"].copy()
        test_adjusted["predicted_label"] = apply_confidence_rule(test_adjusted, best_threshold)
        bt_input = pd.DataFrame(
            {
                "datetime_utc": test_adjusted["datetime_utc"],
                "Open": lookup.loc[test_adjusted["timestamp"], "Open"].to_numpy(),
                "Close": lookup.loc[test_adjusted["timestamp"], "Close"].to_numpy(),
                "signal": test_adjusted["predicted_label"].to_numpy(),
            }
        )
        bt = run_long_flat_backtest(bt_input, signal_col="signal", fee=config["backtest"]["fee"], slippage=config["backtest"]["slippage"])
        summary = summarize_backtest(bt)
        summary.update({"model": model_name, "confidence_threshold": best_threshold})
        final_backtest_rows.append(summary)
        bt.to_csv(metrics_dir / f"backtest_{model_name}_confidence_threshold.csv", index=False)

    pd.DataFrame(selection_rows).to_csv(metrics_dir / "stage3_confidence_threshold_selection.csv", index=False)
    pd.DataFrame(final_metric_rows).to_csv(metrics_dir / "stage3_confidence_classification_metrics.csv", index=False)
    pd.DataFrame(final_backtest_rows).to_csv(metrics_dir / "stage3_confidence_backtest_metrics.csv", index=False)
    pd.concat(final_prediction_frames, ignore_index=True).to_parquet(predictions_dir / "stage3_confidence_predictions.parquet", index=False)
    print(f"Saved confidence-threshold metrics to {metrics_dir}")
    print(f"Saved confidence-threshold predictions to {predictions_dir}")
