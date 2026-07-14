from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .backtesting import run_long_flat_backtest, summarize_backtest
from .baselines import BUY, CLASS_ORDER, HOLD, SELL
from .config import load_config, resolve_path
from .evaluation import classification_metrics, confidence_summary, plot_confusion_matrix, predicted_distribution
from .experiment_scope import apply_experiment_min_datetime
from .stage3_runner import _feature_columns, _prepare_label_col, _sanitize_features, _subset_for_fold
from .temporal_split import generate_expanding_folds
from .train_binary_models import fit_binary_logistic_regression, fit_binary_xgboost


@dataclass
class DualBinaryPrediction:
    labels: pd.Series
    probabilities: pd.DataFrame


def _make_binary_targets(labels: pd.Series) -> tuple[pd.Series, pd.Series]:
    buy_target = (labels == BUY).astype(int)
    sell_target = (labels == SELL).astype(int)
    return buy_target, sell_target


def _resolve_dual_binary_labels(
    buy_prob: pd.Series,
    sell_prob: pd.Series,
    buy_threshold: float = 0.5,
    sell_threshold: float = 0.5,
) -> pd.Series:
    labels = pd.Series(HOLD, index=buy_prob.index, dtype="object")
    buy_active = buy_prob >= buy_threshold
    sell_active = sell_prob >= sell_threshold

    labels = labels.mask(buy_active & ~sell_active, BUY)
    labels = labels.mask(sell_active & ~buy_active, SELL)

    conflict = buy_active & sell_active
    labels = labels.mask(conflict & (buy_prob > sell_prob), BUY)
    labels = labels.mask(conflict & (sell_prob > buy_prob), SELL)
    return labels


def _build_prob_frame(buy_prob: pd.Series, sell_prob: pd.Series) -> pd.DataFrame:
    hold_prob = (1 - buy_prob.clip(0, 1)) * (1 - sell_prob.clip(0, 1))
    raw = pd.DataFrame(
        {
            "probability_buy": buy_prob,
            "probability_hold": hold_prob,
            "probability_sell": sell_prob,
        }
    )
    row_sum = raw.sum(axis=1).replace(0, np.nan)
    return raw.div(row_sum, axis=0).fillna(1 / 3)


def _prediction_bundle_from_probs(
    buy_prob: np.ndarray,
    sell_prob: np.ndarray,
    index: pd.Index,
    buy_threshold: float = 0.5,
    sell_threshold: float = 0.5,
) -> DualBinaryPrediction:
    buy_series = pd.Series(buy_prob, index=index)
    sell_series = pd.Series(sell_prob, index=index)
    labels = _resolve_dual_binary_labels(buy_series, sell_series, buy_threshold=buy_threshold, sell_threshold=sell_threshold)
    probabilities = _build_prob_frame(buy_series, sell_series)
    probabilities.index = index
    return DualBinaryPrediction(labels=labels, probabilities=probabilities)


def run_stage3_dual_binary(config_path: str | None = None) -> None:
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
    threshold = stage3_cfg["selected_threshold"] or 0.005
    label_col = _prepare_label_col(horizon, threshold)
    future_return_col = f"future_return_{horizon}h"
    feature_cols = _feature_columns(df)
    xgb_grid = stage3_cfg["xgboost_grid"]
    logreg_cfg = stage3_cfg["logistic_regression"]
    folds = generate_expanding_folds(
        start_train_year=start_year,
        first_validation_year=validation_start_year,
        last_test_year=last_pre_holdout_year,
    )

    metrics_rows: list[dict] = []
    prediction_frames: list[pd.DataFrame] = []
    lookup = df.set_index("timestamp")[["Open", "Close"]]

    for fold in folds:
        train_df = _subset_for_fold(df, fold.train_start, fold.train_end, label_col)
        valid_df = _subset_for_fold(df, fold.validation_start, fold.validation_end, label_col)
        test_df = _subset_for_fold(df, fold.test_start, fold.test_end, label_col)

        X_train = _sanitize_features(train_df, feature_cols)
        X_valid = _sanitize_features(valid_df, feature_cols)
        X_test = _sanitize_features(test_df, feature_cols)
        buy_train, sell_train = _make_binary_targets(train_df[label_col])
        buy_valid, sell_valid = _make_binary_targets(valid_df[label_col])

        buy_logreg = fit_binary_logistic_regression(X_train, buy_train, C=logreg_cfg["C"], max_iter=logreg_cfg["max_iter"])
        sell_logreg = fit_binary_logistic_regression(X_train, sell_train, C=logreg_cfg["C"], max_iter=logreg_cfg["max_iter"])
        buy_xgb = fit_binary_xgboost(X_train, buy_train, X_valid, buy_valid, xgb_grid, seed=config["random_seed"])
        sell_xgb = fit_binary_xgboost(X_train, sell_train, X_valid, sell_valid, xgb_grid, seed=config["random_seed"])

        for model_name, buy_model, sell_model in [
            ("dual_binary_logistic_regression", buy_logreg.model, sell_logreg.model),
            ("dual_binary_xgboost", buy_xgb.model, sell_xgb.model),
        ]:
            for split_name, split_df, X_split in [("validation", valid_df, X_valid), ("test", test_df, X_test)]:
                buy_prob = buy_model.predict_proba(X_split)[:, 1]
                sell_prob = sell_model.predict_proba(X_split)[:, 1]
                pred = _prediction_bundle_from_probs(buy_prob, sell_prob, split_df.index)

                row = {
                    "model": model_name,
                    "fold": fold.fold_name,
                    "split": split_name,
                    "window_type": fold.window_type,
                    "threshold": threshold,
                    "horizon_hours": horizon,
                }
                row.update(classification_metrics(split_df[label_col], pred.labels))
                row.update(predicted_distribution(pred.labels))
                row.update(confidence_summary(pred.probabilities))
                metrics_rows.append(row)

                prediction_frame = pd.DataFrame(
                    {
                        "timestamp": split_df["timestamp"].to_numpy(),
                        "datetime_utc": split_df["datetime_utc"].to_numpy(),
                        "actual_label": split_df[label_col].to_numpy(),
                        "predicted_label": pred.labels.to_numpy(),
                        "future_return": split_df[future_return_col].to_numpy(),
                        "fold": fold.fold_name,
                        "split": split_name,
                        "model": model_name,
                        "window_type": fold.window_type,
                    }
                )
                prediction_frame = pd.concat([prediction_frame, pred.probabilities.reset_index(drop=True)], axis=1)
                prediction_frames.append(prediction_frame)

    metrics_df = pd.DataFrame(metrics_rows)
    predictions_df = pd.concat(prediction_frames, ignore_index=True)
    metrics_df.to_csv(metrics_dir / "stage3_dual_binary_classification_metrics.csv", index=False)
    predictions_df.to_parquet(predictions_dir / "stage3_dual_binary_predictions.parquet", index=False)
    predictions_df.to_csv(predictions_dir / "stage3_dual_binary_predictions.csv", index=False)

    backtest_rows: list[dict] = []
    for model_name in sorted(predictions_df["model"].unique()):
        test_pred = predictions_df[(predictions_df["model"] == model_name) & (predictions_df["split"] == "test")].sort_values("datetime_utc").reset_index(drop=True)
        bt_input = pd.DataFrame(
            {
                "datetime_utc": pd.to_datetime(test_pred["datetime_utc"], utc=True),
                "Open": lookup.loc[test_pred["timestamp"], "Open"].to_numpy(),
                "Close": lookup.loc[test_pred["timestamp"], "Close"].to_numpy(),
                "signal": test_pred["predicted_label"].to_numpy(),
            }
        )
        bt = run_long_flat_backtest(bt_input, signal_col="signal", fee=config["backtest"]["fee"], slippage=config["backtest"]["slippage"])
        summary = summarize_backtest(bt)
        summary.update(
            {
                "model": model_name,
                "window_type": "expanding",
                "threshold": threshold,
                "horizon_hours": horizon,
            }
        )
        backtest_rows.append(summary)
        bt.to_csv(metrics_dir / f"backtest_{model_name}_expanding.csv", index=False)

    backtest_df = pd.DataFrame(backtest_rows).sort_values("total_return", ascending=False)
    backtest_df.to_csv(metrics_dir / "stage3_dual_binary_backtest_metrics.csv", index=False)

    xgb_test = predictions_df[(predictions_df["model"] == "dual_binary_xgboost") & (predictions_df["split"] == "test")]
    plot_confusion_matrix(
        y_true=xgb_test["actual_label"],
        y_pred=xgb_test["predicted_label"],
        output_path=figures_dir / "confusion_matrix_dual_binary_xgboost_expanding_test.png",
        title="Dual Binary XGBoost Expanding Test Confusion Matrix",
    )
    print(f"Saved dual-binary metrics to {metrics_dir}")
    print(f"Saved dual-binary predictions to {predictions_dir}")
