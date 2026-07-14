from __future__ import annotations

import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score

from .backtesting import run_long_flat_backtest, summarize_backtest
from .config import load_config, resolve_path
from .evaluation import classification_metrics, confidence_summary, predicted_distribution


PROB_COLS = ["probability_buy", "probability_hold", "probability_sell"]
BASE_MODELS = [
    "logistic_regression",
    "xgboost",
    "dual_binary_logistic_regression",
    "dual_binary_xgboost",
]


def _load_prediction_pool(predictions_dir) -> pd.DataFrame:
    base = pd.read_parquet(predictions_dir / "stage3_predictions.parquet")
    dual = pd.read_parquet(predictions_dir / "stage3_dual_binary_predictions.parquet")
    df = pd.concat(
        [
            base[base["model"].isin(["logistic_regression", "xgboost"])],
            dual[dual["model"].isin(["dual_binary_logistic_regression", "dual_binary_xgboost"])],
        ],
        ignore_index=True,
    )
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)
    return df


def _candidate_weight_sets() -> list[dict[str, float]]:
    return [
        {"logistic_regression": 1.0},
        {"xgboost": 1.0},
        {"dual_binary_logistic_regression": 1.0},
        {"dual_binary_xgboost": 1.0},
        {"logistic_regression": 0.5, "xgboost": 0.5},
        {"logistic_regression": 0.5, "dual_binary_logistic_regression": 0.5},
        {"xgboost": 0.5, "dual_binary_xgboost": 0.5},
        {"logistic_regression": 0.25, "xgboost": 0.25, "dual_binary_logistic_regression": 0.25, "dual_binary_xgboost": 0.25},
        {"logistic_regression": 0.2, "xgboost": 0.4, "dual_binary_logistic_regression": 0.1, "dual_binary_xgboost": 0.3},
        {"logistic_regression": 0.4, "xgboost": 0.2, "dual_binary_logistic_regression": 0.3, "dual_binary_xgboost": 0.1},
    ]


def _compose_ensemble(split_df: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    merged = None
    keys = ["timestamp", "datetime_utc", "actual_label", "future_return", "fold", "split", "window_type"]
    for model_name, weight in weights.items():
        part = split_df[split_df["model"] == model_name][keys + PROB_COLS].copy()
        part = part.rename(columns={col: f"{col}_{model_name}" for col in PROB_COLS})
        merged = part if merged is None else merged.merge(part, on=keys, how="inner")
    if merged is None:
        raise RuntimeError("No models supplied for ensemble.")
    for col in PROB_COLS:
        merged[col] = 0.0
        for model_name, weight in weights.items():
            merged[col] += weight * merged[f"{col}_{model_name}"]
    merged["predicted_label"] = merged[PROB_COLS].idxmax(axis=1).str.replace("probability_", "").str.upper()
    return merged


def _quick_backtest_score(bt: pd.DataFrame) -> dict[str, float]:
    return {
        "total_return": float(bt["equity_curve"].iloc[-1] - 1),
        "num_trades": int((((bt["position_prev"] == 0) & (bt["position"] == 1)).astype(int)).sum()),
    }


def _quick_classification_score(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def run_stage_ensemble(config_path: str | None = None) -> None:
    config = load_config(config_path)
    predictions_dir = resolve_path(config["paths"]["predictions_dir"])
    metrics_dir = resolve_path(config["paths"]["metrics_dir"])
    labeled_path = resolve_path(config["paths"]["labeled_1h"])
    metrics_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    pred_pool = _load_prediction_pool(predictions_dir)
    lookup = pd.read_parquet(labeled_path).set_index("timestamp")[["Open", "Close"]]

    candidate_rows: list[dict] = []
    for weights in _candidate_weight_sets():
        fold_rows: list[dict] = []
        for fold, fold_df in pred_pool[pred_pool["split"] == "validation"].groupby("fold"):
            ensemble = _compose_ensemble(fold_df, weights)
            metrics = _quick_classification_score(ensemble["actual_label"], ensemble["predicted_label"])
            bt_input = pd.DataFrame(
                {
                    "datetime_utc": ensemble["datetime_utc"],
                    "Open": lookup.loc[ensemble["timestamp"], "Open"].to_numpy(),
                    "Close": lookup.loc[ensemble["timestamp"], "Close"].to_numpy(),
                    "signal": ensemble["predicted_label"].to_numpy(),
                }
            )
            bt = run_long_flat_backtest(bt_input, signal_col="signal", fee=config["backtest"]["fee"], slippage=config["backtest"]["slippage"])
            bt_summary = _quick_backtest_score(bt)
            fold_rows.append(
                {
                    "fold": fold,
                    "weights": str(weights),
                    "macro_f1": metrics["macro_f1"],
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "bt_total_return": bt_summary["total_return"],
                    "bt_num_trades": bt_summary["num_trades"],
                }
            )
        fold_df = pd.DataFrame(fold_rows)
        candidate_rows.append(
            {
                "weights": str(weights),
                "mean_macro_f1": fold_df["macro_f1"].mean(),
                "mean_balanced_accuracy": fold_df["balanced_accuracy"].mean(),
                "mean_bt_total_return": fold_df["bt_total_return"].mean(),
                "mean_bt_num_trades": fold_df["bt_num_trades"].mean(),
            }
        )

    selection_df = pd.DataFrame(candidate_rows).sort_values(
        ["mean_macro_f1", "mean_bt_total_return", "mean_balanced_accuracy", "mean_bt_num_trades"],
        ascending=[False, False, False, True],
    )
    selection_df.to_csv(metrics_dir / "ensemble_selection.csv", index=False)
    best_weights = eval(selection_df.iloc[0]["weights"])

    metric_rows: list[dict] = []
    prediction_frames: list[pd.DataFrame] = []
    backtest_rows: list[dict] = []
    for split_name in ["validation", "test"]:
        split_df = pred_pool[pred_pool["split"] == split_name].copy()
        ensemble = _compose_ensemble(split_df, best_weights)
        row = {"model": "weighted_soft_voting_ensemble", "split": split_name, "weights": str(best_weights)}
        row.update(classification_metrics(ensemble["actual_label"], ensemble["predicted_label"]))
        row.update(predicted_distribution(ensemble["predicted_label"]))
        row.update(confidence_summary(ensemble[PROB_COLS]))
        metric_rows.append(row)

        ensemble["model"] = "weighted_soft_voting_ensemble"
        ensemble["weights"] = str(best_weights)
        prediction_frames.append(
            ensemble[
                ["timestamp", "datetime_utc", "actual_label", "predicted_label", "future_return", "fold", "split", "model", "window_type", "weights"]
                + PROB_COLS
            ].copy()
        )

        if split_name == "test":
            bt_input = pd.DataFrame(
                {
                    "datetime_utc": ensemble["datetime_utc"],
                    "Open": lookup.loc[ensemble["timestamp"], "Open"].to_numpy(),
                    "Close": lookup.loc[ensemble["timestamp"], "Close"].to_numpy(),
                    "signal": ensemble["predicted_label"].to_numpy(),
                }
            )
            bt = run_long_flat_backtest(bt_input, signal_col="signal", fee=config["backtest"]["fee"], slippage=config["backtest"]["slippage"])
            bt_summary = summarize_backtest(bt)
            bt_summary.update({"model": "weighted_soft_voting_ensemble", "weights": str(best_weights)})
            backtest_rows.append(bt_summary)
            bt.to_csv(metrics_dir / "backtest_weighted_soft_voting_ensemble.csv", index=False)

    pd.DataFrame(metric_rows).to_csv(metrics_dir / "ensemble_classification_metrics.csv", index=False)
    pd.DataFrame(backtest_rows).to_csv(metrics_dir / "ensemble_backtest_metrics.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_parquet(predictions_dir / "ensemble_predictions.parquet", index=False)
    print(f"Saved ensemble metrics to {metrics_dir}")
    print(f"Saved ensemble predictions to {predictions_dir}")
