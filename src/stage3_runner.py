from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import numpy as np

from .backtesting import run_long_flat_backtest, summarize_backtest
from .baselines import (
    CLASS_ORDER,
    always_buy,
    always_hold,
    bollinger_band_strategy,
    combined_macd_bollinger_strategy,
    macd_crossover_strategy,
    majority_class,
)
from .config import load_config, resolve_path
from .evaluation import classification_metrics, confidence_summary, plot_confusion_matrix, predicted_distribution
from .experiment_scope import apply_experiment_min_datetime
from .temporal_split import generate_expanding_folds
from .train_classical_models import fit_logistic_regression
from .train_xgboost import decode_labels, fit_xgboost


NON_FEATURE_COLUMNS = {
    "datetime_utc",
    "timestamp",
    "future_return_6h",
    "future_return_24h",
}


@dataclass
class PredictionBundle:
    labels: pd.Series
    probabilities: pd.DataFrame


def _feature_columns(df: pd.DataFrame) -> list[str]:
    return [
        col
        for col in df.columns
        if not col.startswith("label_")
        and col not in NON_FEATURE_COLUMNS
        and not col.startswith("future_return_")
        and col not in {"Open", "High", "Low", "Close", "Volume"}
    ] + ["Open", "High", "Low", "Close", "Volume"]


def _prepare_label_col(horizon: int, threshold: float) -> str:
    return f"label_h{horizon}_thr_{int(round(threshold * 10000))}bp"


def _sanitize_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    clean = df[feature_cols].copy()
    clean = clean.replace([np.inf, -np.inf], np.nan)
    return clean


def _subset_for_fold(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, label_col: str) -> pd.DataFrame:
    part = df[(df["datetime_utc"] >= start) & (df["datetime_utc"] <= end)].copy()
    return part[part[label_col].notna()].reset_index(drop=True)


def _build_predictions_frame(
    df: pd.DataFrame,
    label_col: str,
    pred: PredictionBundle,
    fold_name: str,
    split: str,
    model_name: str,
    window_type: str,
    future_return_col: str,
) -> pd.DataFrame:
    result = pd.DataFrame(
        {
            "timestamp": df["timestamp"].to_numpy(),
            "datetime_utc": df["datetime_utc"].to_numpy(),
            "actual_label": df[label_col].to_numpy(),
            "predicted_label": pred.labels.to_numpy(),
            "future_return": df[future_return_col].to_numpy(),
            "fold": fold_name,
            "split": split,
            "model": model_name,
            "window_type": window_type,
        }
    )
    result = pd.concat([result, pred.probabilities.reset_index(drop=True)], axis=1)
    return result


def _metrics_row(
    y_true: pd.Series,
    pred: PredictionBundle,
    model_name: str,
    fold_name: str,
    split: str,
    window_type: str,
    threshold: float,
    horizon: int,
) -> dict:
    row = {
        "model": model_name,
        "fold": fold_name,
        "split": split,
        "window_type": window_type,
        "threshold": threshold,
        "horizon_hours": horizon,
    }
    row.update(classification_metrics(y_true, pred.labels))
    row.update(predicted_distribution(pred.labels))
    row.update(confidence_summary(pred.probabilities))
    return row


def _run_technical_baseline(model_name: str, df: pd.DataFrame):
    if model_name == "always_hold":
        return always_hold(df.index)
    if model_name == "majority_class":
        raise ValueError("majority_class requires training labels")
    if model_name == "macd":
        return macd_crossover_strategy(df)
    if model_name == "bollinger":
        return bollinger_band_strategy(df)
    if model_name == "macd_bollinger":
        return combined_macd_bollinger_strategy(df)
    raise ValueError(model_name)


def run_stage3(config_path: str | None = None) -> None:
    config = load_config(config_path)
    labeled_path = resolve_path(config["paths"]["labeled_1h"])
    metrics_dir = resolve_path(config["paths"]["metrics_dir"])
    predictions_dir = resolve_path(config["paths"]["predictions_dir"])
    figures_dir = resolve_path(config["paths"]["figures_dir"])
    metrics_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(labeled_path)
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)
    df = apply_experiment_min_datetime(df, config["dataset"]["experiment_min_datetime"])

    stage3_cfg = config["stage3"]
    start_year = config["dataset"]["experiment_start_year"]
    validation_start_year = config["dataset"]["validation_start_year"]
    last_pre_holdout_year = config["dataset"]["last_pre_holdout_year"]
    horizon = stage3_cfg["primary_horizon_hours"]
    threshold_candidates = config["labels"]["threshold_candidates"]
    feature_cols = _feature_columns(df)
    xgb_grid = stage3_cfg["xgboost_grid"]
    logreg_cfg = stage3_cfg["logistic_regression"]
    folds = generate_expanding_folds(
        start_train_year=start_year,
        first_validation_year=validation_start_year,
        last_test_year=last_pre_holdout_year,
    )

    threshold_rows: list[dict] = []
    for threshold in threshold_candidates:
        label_col = _prepare_label_col(horizon, threshold)
        future_return_col = f"future_return_{horizon}h"
        fold_scores: list[dict] = []
        for fold in folds:
            train_df = _subset_for_fold(df, fold.train_start, fold.train_end, label_col)
            valid_df = _subset_for_fold(df, fold.validation_start, fold.validation_end, label_col)

            logistic = fit_logistic_regression(
                _sanitize_features(train_df, feature_cols),
                train_df[label_col],
                C=logreg_cfg["C"],
                max_iter=logreg_cfg["max_iter"],
            )
            X_valid = _sanitize_features(valid_df, feature_cols)
            valid_pred_labels = pd.Series(logistic.model.predict(X_valid), index=valid_df.index)
            valid_pred_prob = pd.DataFrame(
                logistic.model.predict_proba(X_valid),
                columns=[f"probability_{c.lower()}" for c in logistic.model.classes_],
                index=valid_df.index,
            ).reindex(columns=[f"probability_{c.lower()}" for c in CLASS_ORDER], fill_value=0.0)
            pred_bundle = PredictionBundle(valid_pred_labels, valid_pred_prob)
            valid_metrics = _metrics_row(
                valid_df[label_col],
                pred_bundle,
                "logistic_regression",
                fold.fold_name,
                "validation",
                fold.window_type,
                threshold,
                horizon,
            )
            bt_input = pd.concat(
                [
                    valid_df[["datetime_utc", "Open", "Close", future_return_col]].reset_index(drop=True),
                    pred_bundle.labels.rename("signal").reset_index(drop=True),
                ],
                axis=1,
            )
            bt_metrics = summarize_backtest(run_long_flat_backtest(bt_input, signal_col="signal", fee=config["backtest"]["fee"], slippage=config["backtest"]["slippage"]))
            valid_metrics.update({f"bt_{k}": v for k, v in bt_metrics.items()})
            fold_scores.append(valid_metrics)
        metrics_df = pd.DataFrame(fold_scores)
        threshold_rows.append(
            {
                "threshold": threshold,
                "horizon_hours": horizon,
                "mean_macro_f1": metrics_df["macro_f1"].mean(),
                "std_macro_f1": metrics_df["macro_f1"].std(ddof=0),
                "mean_balanced_accuracy": metrics_df["balanced_accuracy"].mean(),
                "mean_bt_total_return": metrics_df["bt_total_return"].mean(),
                "mean_bt_num_trades": metrics_df["bt_num_trades"].mean(),
            }
        )

    threshold_df = pd.DataFrame(threshold_rows).sort_values(
        ["mean_macro_f1", "mean_bt_total_return", "mean_bt_num_trades"],
        ascending=[False, False, True],
    )
    threshold_df.to_csv(metrics_dir / "threshold_selection_expanding.csv", index=False)
    selected_threshold = stage3_cfg["selected_threshold"] or float(threshold_df.iloc[0]["threshold"])

    label_col = _prepare_label_col(horizon, selected_threshold)
    future_return_col = f"future_return_{horizon}h"
    metrics_rows: list[dict] = []
    prediction_frames: list[pd.DataFrame] = []

    for fold in folds:
        train_df = _subset_for_fold(df, fold.train_start, fold.train_end, label_col)
        valid_df = _subset_for_fold(df, fold.validation_start, fold.validation_end, label_col)
        test_df = _subset_for_fold(df, fold.test_start, fold.test_end, label_col)

        baseline_models = ["always_hold", "macd", "bollinger", "macd_bollinger"]
        for split_name, split_df in [("validation", valid_df), ("test", test_df)]:
            for baseline_name in baseline_models:
                baseline_pred_raw = _run_technical_baseline(baseline_name, split_df)
                pred_bundle = PredictionBundle(baseline_pred_raw.predicted_label, baseline_pred_raw.probabilities)
                metrics_rows.append(
                    _metrics_row(split_df[label_col], pred_bundle, baseline_name, fold.fold_name, split_name, fold.window_type, selected_threshold, horizon)
                )
                prediction_frames.append(
                    _build_predictions_frame(split_df, label_col, pred_bundle, fold.fold_name, split_name, baseline_name, fold.window_type, future_return_col)
                )

            majority_pred_raw = majority_class(train_df[label_col], split_df.index)
            majority_labels = pd.Series(majority_pred_raw.predicted_label.to_numpy(), index=split_df.index)
            majority_prob = majority_pred_raw.probabilities.copy()
            majority_prob.index = split_df.index
            pred_bundle = PredictionBundle(majority_labels, majority_prob)
            metrics_rows.append(
                _metrics_row(split_df[label_col], pred_bundle, "majority_class", fold.fold_name, split_name, fold.window_type, selected_threshold, horizon)
            )
            prediction_frames.append(
                _build_predictions_frame(split_df, label_col, pred_bundle, fold.fold_name, split_name, "majority_class", fold.window_type, future_return_col)
            )

        X_train = _sanitize_features(train_df, feature_cols)
        X_valid = _sanitize_features(valid_df, feature_cols)
        X_test = _sanitize_features(test_df, feature_cols)
        logistic = fit_logistic_regression(X_train, train_df[label_col], C=logreg_cfg["C"], max_iter=logreg_cfg["max_iter"])
        xgb = fit_xgboost(X_train, train_df[label_col], X_valid, valid_df[label_col], xgb_grid, seed=config["random_seed"])

        for split_name, split_df, X_split in [("validation", valid_df, X_valid), ("test", test_df, X_test)]:
            logreg_labels = pd.Series(logistic.model.predict(X_split), index=split_df.index)
            logreg_prob = pd.DataFrame(
                logistic.model.predict_proba(X_split),
                columns=[f"probability_{c.lower()}" for c in logistic.model.classes_],
                index=split_df.index,
            ).reindex(columns=[f"probability_{c.lower()}" for c in CLASS_ORDER], fill_value=0.0)
            logreg_pred = PredictionBundle(logreg_labels, logreg_prob)
            metrics_rows.append(
                _metrics_row(split_df[label_col], logreg_pred, "logistic_regression", fold.fold_name, split_name, fold.window_type, selected_threshold, horizon)
            )
            prediction_frames.append(
                _build_predictions_frame(split_df, label_col, logreg_pred, fold.fold_name, split_name, "logistic_regression", fold.window_type, future_return_col)
            )

            xgb_prob_arr = xgb.model.predict_proba(X_split)
            xgb_pred_arr = xgb.model.predict(X_split)
            xgb_labels = decode_labels(xgb_pred_arr)
            xgb_labels.index = split_df.index
            xgb_prob = pd.DataFrame(
                xgb_prob_arr,
                columns=[f"probability_{c.lower()}" for c in CLASS_ORDER],
                index=split_df.index,
            )
            xgb_pred = PredictionBundle(xgb_labels, xgb_prob)
            row = _metrics_row(split_df[label_col], xgb_pred, "xgboost", fold.fold_name, split_name, fold.window_type, selected_threshold, horizon)
            row["xgb_best_validation_macro_f1"] = xgb.validation_macro_f1
            row["xgb_best_params"] = str(xgb.params)
            metrics_rows.append(row)
            prediction_frames.append(
                _build_predictions_frame(split_df, label_col, xgb_pred, fold.fold_name, split_name, "xgboost", fold.window_type, future_return_col)
            )

    metrics_df = pd.DataFrame(metrics_rows)
    predictions_df = pd.concat(prediction_frames, ignore_index=True)
    metrics_df.to_csv(metrics_dir / "stage3_classification_metrics.csv", index=False)
    predictions_df.to_parquet(predictions_dir / "stage3_predictions.parquet", index=False)
    predictions_df.to_csv(predictions_dir / "stage3_predictions.csv", index=False)

    backtest_rows: list[dict] = []
    for model_name in sorted(predictions_df["model"].unique()):
        test_pred = predictions_df[(predictions_df["model"] == model_name) & (predictions_df["split"] == "test")].sort_values("datetime_utc").reset_index(drop=True)
        bt_input = pd.DataFrame(
            {
                "datetime_utc": pd.to_datetime(test_pred["datetime_utc"], utc=True),
                "Open": df.set_index("timestamp").loc[test_pred["timestamp"], "Open"].to_numpy(),
                "Close": df.set_index("timestamp").loc[test_pred["timestamp"], "Close"].to_numpy(),
                "signal": test_pred["predicted_label"].to_numpy(),
            }
        )
        bt = run_long_flat_backtest(bt_input, signal_col="signal", fee=config["backtest"]["fee"], slippage=config["backtest"]["slippage"])
        summary = summarize_backtest(bt)
        summary.update(
            {
                "model": model_name,
                "window_type": "expanding",
                "threshold": selected_threshold,
                "horizon_hours": horizon,
            }
        )
        backtest_rows.append(summary)
        bt.to_csv(metrics_dir / f"backtest_{model_name}_expanding.csv", index=False)

    buy_hold_test = predictions_df[predictions_df["split"] == "test"].sort_values("datetime_utc").drop_duplicates("timestamp").reset_index(drop=True)
    buy_hold_signal = always_buy(buy_hold_test.index)
    bt_input = pd.DataFrame(
        {
            "datetime_utc": pd.to_datetime(buy_hold_test["datetime_utc"], utc=True),
            "Open": df.set_index("timestamp").loc[buy_hold_test["timestamp"], "Open"].to_numpy(),
            "Close": df.set_index("timestamp").loc[buy_hold_test["timestamp"], "Close"].to_numpy(),
            "signal": buy_hold_signal.predicted_label.to_numpy(),
        }
    )
    bt = run_long_flat_backtest(bt_input, signal_col="signal", fee=config["backtest"]["fee"], slippage=config["backtest"]["slippage"])
    summary = summarize_backtest(bt)
    summary.update(
        {
            "model": "buy_and_hold",
            "window_type": "expanding",
            "threshold": selected_threshold,
            "horizon_hours": horizon,
        }
    )
    backtest_rows.append(summary)
    bt.to_csv(metrics_dir / "backtest_buy_and_hold_expanding.csv", index=False)

    backtest_df = pd.DataFrame(backtest_rows).sort_values("total_return", ascending=False)
    backtest_df.to_csv(metrics_dir / "stage3_backtest_metrics.csv", index=False)

    xgb_test = predictions_df[(predictions_df["model"] == "xgboost") & (predictions_df["split"] == "test")]
    plot_confusion_matrix(
        y_true=xgb_test["actual_label"],
        y_pred=xgb_test["predicted_label"],
        output_path=figures_dir / "confusion_matrix_xgboost_expanding_test.png",
        title="XGBoost Expanding Test Confusion Matrix",
    )
    print(f"Selected threshold for stage 3: {selected_threshold}")
    print(f"Saved metrics to {metrics_dir}")
    print(f"Saved predictions to {predictions_dir}")
