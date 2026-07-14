from __future__ import annotations

from typing import Literal

import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score

from .config import load_config, resolve_path
from .evaluation import classification_metrics
from .experiment_scope import apply_experiment_min_datetime
from .stage3_runner import _feature_columns, _prepare_label_col, _sanitize_features, _subset_for_fold
from .temporal_split import generate_expanding_folds
from .train_binary_models import fit_binary_logistic_regression, fit_binary_xgboost
from .train_classical_models import fit_logistic_regression
from .train_xgboost import decode_labels, fit_xgboost


AltTask = Literal["binary-events", "direction-24h", "regime-24h", "direction-24h-neutral"]


def _binary_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_positive": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_positive": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_positive": float(f1_score(y_true, y_pred, zero_division=0)),
        "positive_rate_actual": float(y_true.mean()),
        "positive_rate_pred": float(y_pred.mean()),
    }


def _build_binary_target(labels: pd.Series, positive_label: str) -> pd.Series:
    return (labels == positive_label).astype(int)


def _load_alt_context(config_path: str | None):
    config = load_config(config_path)
    labeled_path = resolve_path(config["paths"]["labeled_1h"])
    metrics_dir = resolve_path(config["paths"]["metrics_dir"])
    predictions_dir = resolve_path(config["paths"]["predictions_dir"])
    metrics_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(labeled_path)
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)
    df = apply_experiment_min_datetime(df, config["dataset"]["experiment_min_datetime"])

    feature_cols = _feature_columns(df)
    xgb_grid = config["stage3"]["xgboost_grid"]
    logreg_cfg = config["stage3"]["logistic_regression"]
    folds = generate_expanding_folds(
        start_train_year=config["dataset"]["experiment_start_year"],
        first_validation_year=config["dataset"]["validation_start_year"],
        last_test_year=config["dataset"]["last_pre_holdout_year"],
    )
    return config, df, feature_cols, xgb_grid, logreg_cfg, folds, metrics_dir, predictions_dir


def run_alt_binary_events(config_path: str | None = None) -> None:
    config, df, feature_cols, xgb_grid, logreg_cfg, folds, metrics_dir, predictions_dir = _load_alt_context(config_path)
    alt_cfg = config["alt_targets"]
    binary_h = alt_cfg["binary_event_horizon_hours"]
    binary_thr = alt_cfg["binary_event_threshold"]
    binary_label_col = _prepare_label_col(binary_h, binary_thr)

    rows: list[dict] = []
    pred_rows: list[pd.DataFrame] = []
    for positive_label in ["BUY", "SELL"]:
        for fold in folds:
            train_df = _subset_for_fold(df, fold.train_start, fold.train_end, binary_label_col)
            valid_df = _subset_for_fold(df, fold.validation_start, fold.validation_end, binary_label_col)
            test_df = _subset_for_fold(df, fold.test_start, fold.test_end, binary_label_col)

            X_train = _sanitize_features(train_df, feature_cols)
            X_valid = _sanitize_features(valid_df, feature_cols)
            X_test = _sanitize_features(test_df, feature_cols)
            y_train = _build_binary_target(train_df[binary_label_col], positive_label)
            y_valid = _build_binary_target(valid_df[binary_label_col], positive_label)
            y_test = _build_binary_target(test_df[binary_label_col], positive_label)

            logreg = fit_binary_logistic_regression(X_train, y_train, C=logreg_cfg["C"], max_iter=logreg_cfg["max_iter"])
            xgb = fit_binary_xgboost(X_train, y_train, X_valid, y_valid, xgb_grid, seed=config["random_seed"])

            for model_name, model in [
                (f"logistic_{positive_label.lower()}_vs_non", logreg.model),
                (f"xgboost_{positive_label.lower()}_vs_non", xgb.model),
            ]:
                for split_name, split_df, X_split, y_split in [
                    ("validation", valid_df, X_valid, y_valid),
                    ("test", test_df, X_test, y_test),
                ]:
                    prob = model.predict_proba(X_split)[:, 1]
                    pred = (prob >= 0.5).astype(int)
                    metric = {
                        "task": f"{positive_label}_vs_non",
                        "model": model_name,
                        "fold": fold.fold_name,
                        "split": split_name,
                        "horizon_hours": binary_h,
                        "threshold": binary_thr,
                    }
                    metric.update(_binary_metrics(y_split, pred))
                    rows.append(metric)
                    pred_rows.append(
                        pd.DataFrame(
                            {
                                "timestamp": split_df["timestamp"].to_numpy(),
                                "datetime_utc": split_df["datetime_utc"].to_numpy(),
                                "actual_binary": y_split.to_numpy(),
                                "predicted_binary": pred,
                                "probability_positive": prob,
                                "positive_label": positive_label,
                                "fold": fold.fold_name,
                                "split": split_name,
                                "model": model_name,
                            }
                        )
                    )
    pd.DataFrame(rows).to_csv(metrics_dir / "alt_binary_event_metrics.csv", index=False)
    pd.concat(pred_rows, ignore_index=True).to_parquet(predictions_dir / "alt_binary_event_predictions.parquet", index=False)
    print(f"Saved binary-event metrics to {metrics_dir}")


def run_alt_direction_24h(config_path: str | None = None) -> None:
    config, df, feature_cols, xgb_grid, logreg_cfg, folds, metrics_dir, predictions_dir = _load_alt_context(config_path)
    horizon = config["alt_targets"]["regime_horizon_hours"]
    direction_col = f"future_return_{horizon}h"

    rows: list[dict] = []
    pred_rows: list[pd.DataFrame] = []
    direction_df = df[df[direction_col].notna()].copy()
    for fold in folds:
        train_df = _subset_for_fold(direction_df, fold.train_start, fold.train_end, "label_h6_thr_50bp")
        valid_df = _subset_for_fold(direction_df, fold.validation_start, fold.validation_end, "label_h6_thr_50bp")
        test_df = _subset_for_fold(direction_df, fold.test_start, fold.test_end, "label_h6_thr_50bp")

        X_train = _sanitize_features(train_df, feature_cols)
        X_valid = _sanitize_features(valid_df, feature_cols)
        X_test = _sanitize_features(test_df, feature_cols)
        y_train = (train_df[direction_col] > 0).astype(int)
        y_valid = (valid_df[direction_col] > 0).astype(int)
        y_test = (test_df[direction_col] > 0).astype(int)

        logreg = fit_binary_logistic_regression(X_train, y_train, C=logreg_cfg["C"], max_iter=logreg_cfg["max_iter"])
        xgb = fit_binary_xgboost(X_train, y_train, X_valid, y_valid, xgb_grid, seed=config["random_seed"])

        for model_name, model in [("logistic_updown_24h", logreg.model), ("xgboost_updown_24h", xgb.model)]:
            for split_name, split_df, X_split, y_split in [
                ("validation", valid_df, X_valid, y_valid),
                ("test", test_df, X_test, y_test),
            ]:
                prob = model.predict_proba(X_split)[:, 1]
                pred = (prob >= 0.5).astype(int)
                metric = {
                    "task": "UP_vs_DOWN_24h",
                    "model": model_name,
                    "fold": fold.fold_name,
                    "split": split_name,
                    "horizon_hours": horizon,
                }
                metric.update(_binary_metrics(y_split, pred))
                rows.append(metric)
                pred_rows.append(
                    pd.DataFrame(
                        {
                            "timestamp": split_df["timestamp"].to_numpy(),
                            "datetime_utc": split_df["datetime_utc"].to_numpy(),
                            "actual_binary": y_split.to_numpy(),
                            "predicted_binary": pred,
                            "probability_up": prob,
                            "fold": fold.fold_name,
                            "split": split_name,
                            "model": model_name,
                        }
                    )
                )
    pd.DataFrame(rows).to_csv(metrics_dir / "alt_direction_24h_metrics.csv", index=False)
    pd.concat(pred_rows, ignore_index=True).to_parquet(predictions_dir / "alt_direction_24h_predictions.parquet", index=False)
    print(f"Saved direction-24h metrics to {metrics_dir}")


def run_alt_direction_24h_neutral(config_path: str | None = None) -> None:
    config, df, feature_cols, xgb_grid, logreg_cfg, folds, metrics_dir, predictions_dir = _load_alt_context(config_path)
    horizon = config["alt_targets"]["regime_horizon_hours"]
    direction_col = f"future_return_{horizon}h"
    thresholds = config["alt_targets"]["regime_threshold_candidates"]

    selection_rows: list[dict] = []
    final_rows: list[dict] = []
    pred_rows: list[pd.DataFrame] = []

    for neutral_threshold in thresholds:
        filtered = df[df[direction_col].notna() & (df[direction_col].abs() >= neutral_threshold)].copy()
        fold_scores: list[dict] = []
        for fold in folds:
            train_df = _subset_for_fold(filtered, fold.train_start, fold.train_end, "label_h6_thr_50bp")
            valid_df = _subset_for_fold(filtered, fold.validation_start, fold.validation_end, "label_h6_thr_50bp")
            if train_df.empty or valid_df.empty:
                continue
            X_train = _sanitize_features(train_df, feature_cols)
            X_valid = _sanitize_features(valid_df, feature_cols)
            y_train = (train_df[direction_col] > 0).astype(int)
            y_valid = (valid_df[direction_col] > 0).astype(int)
            xgb = fit_binary_xgboost(X_train, y_train, X_valid, y_valid, xgb_grid, seed=config["random_seed"])
            pred = (xgb.model.predict_proba(X_valid)[:, 1] >= 0.5).astype(int)
            fold_scores.append(_binary_metrics(y_valid, pred))
        score_df = pd.DataFrame(fold_scores)
        selection_rows.append(
            {
                "neutral_threshold": neutral_threshold,
                "mean_accuracy": score_df["accuracy"].mean(),
                "mean_balanced_accuracy": score_df["balanced_accuracy"].mean(),
                "mean_f1_positive": score_df["f1_positive"].mean(),
            }
        )

    selection_df = pd.DataFrame(selection_rows).sort_values(
        ["mean_accuracy", "mean_balanced_accuracy", "mean_f1_positive"],
        ascending=[False, False, False],
    )
    selection_df.to_csv(metrics_dir / "alt_direction_24h_neutral_selection.csv", index=False)
    selected_threshold = float(selection_df.iloc[0]["neutral_threshold"])
    filtered = df[df[direction_col].notna() & (df[direction_col].abs() >= selected_threshold)].copy()

    for fold in folds:
        train_df = _subset_for_fold(filtered, fold.train_start, fold.train_end, "label_h6_thr_50bp")
        valid_df = _subset_for_fold(filtered, fold.validation_start, fold.validation_end, "label_h6_thr_50bp")
        test_df = _subset_for_fold(filtered, fold.test_start, fold.test_end, "label_h6_thr_50bp")
        if train_df.empty or valid_df.empty or test_df.empty:
            continue

        X_train = _sanitize_features(train_df, feature_cols)
        X_valid = _sanitize_features(valid_df, feature_cols)
        X_test = _sanitize_features(test_df, feature_cols)
        y_train = (train_df[direction_col] > 0).astype(int)
        y_valid = (valid_df[direction_col] > 0).astype(int)
        y_test = (test_df[direction_col] > 0).astype(int)

        logreg = fit_binary_logistic_regression(X_train, y_train, C=logreg_cfg["C"], max_iter=logreg_cfg["max_iter"])

        for split_name, split_df, X_split, y_split in [("validation", valid_df, X_valid, y_valid), ("test", test_df, X_test, y_test)]:
            prob = logreg.model.predict_proba(X_split)[:, 1]
            pred = (prob >= 0.5).astype(int)
            metric = {
                "task": "UP_vs_DOWN_24h_neutral",
                "model": "logistic_updown_24h_neutral",
                "fold": fold.fold_name,
                "split": split_name,
                "horizon_hours": horizon,
                "neutral_threshold": selected_threshold,
            }
            metric.update(_binary_metrics(y_split, pred))
            final_rows.append(metric)
            pred_rows.append(
                pd.DataFrame(
                    {
                        "timestamp": split_df["timestamp"].to_numpy(),
                        "datetime_utc": split_df["datetime_utc"].to_numpy(),
                        "actual_binary": y_split.to_numpy(),
                        "predicted_binary": pred,
                        "probability_up": prob,
                        "fold": fold.fold_name,
                        "split": split_name,
                        "model": "logistic_updown_24h_neutral",
                        "neutral_threshold": selected_threshold,
                    }
                )
            )
    pd.DataFrame(final_rows).to_csv(metrics_dir / "alt_direction_24h_neutral_metrics.csv", index=False)
    pd.concat(pred_rows, ignore_index=True).to_parquet(predictions_dir / "alt_direction_24h_neutral_predictions.parquet", index=False)
    print(f"Saved direction-24h-neutral metrics to {metrics_dir}")


def run_alt_regime_24h(config_path: str | None = None) -> None:
    config, df, feature_cols, xgb_grid, logreg_cfg, folds, metrics_dir, predictions_dir = _load_alt_context(config_path)
    horizon = config["alt_targets"]["regime_horizon_hours"]
    threshold_candidates = config["alt_targets"]["regime_threshold_candidates"]

    threshold_scores: list[dict] = []
    for threshold in threshold_candidates:
        label_col = _prepare_label_col(horizon, threshold)
        fold_scores: list[float] = []
        for fold in folds:
            train_df = _subset_for_fold(df, fold.train_start, fold.train_end, label_col)
            valid_df = _subset_for_fold(df, fold.validation_start, fold.validation_end, label_col)
            X_train = _sanitize_features(train_df, feature_cols)
            X_valid = _sanitize_features(valid_df, feature_cols)
            logistic = fit_logistic_regression(X_train, train_df[label_col], C=logreg_cfg["C"], max_iter=logreg_cfg["max_iter"])
            pred = logistic.model.predict(X_valid)
            fold_scores.append(float(f1_score(valid_df[label_col], pred, average="macro", zero_division=0)))
        threshold_scores.append({"threshold": threshold, "mean_macro_f1": sum(fold_scores) / len(fold_scores)})
    threshold_df = pd.DataFrame(threshold_scores).sort_values("mean_macro_f1", ascending=False)
    threshold_df.to_csv(metrics_dir / "alt_regime_threshold_selection.csv", index=False)
    selected_threshold = float(threshold_df.iloc[0]["threshold"])
    label_col = _prepare_label_col(horizon, selected_threshold)

    rows: list[dict] = []
    pred_rows: list[pd.DataFrame] = []
    for fold in folds:
        train_df = _subset_for_fold(df, fold.train_start, fold.train_end, label_col)
        valid_df = _subset_for_fold(df, fold.validation_start, fold.validation_end, label_col)
        test_df = _subset_for_fold(df, fold.test_start, fold.test_end, label_col)
        X_train = _sanitize_features(train_df, feature_cols)
        X_valid = _sanitize_features(valid_df, feature_cols)
        X_test = _sanitize_features(test_df, feature_cols)

        logistic = fit_logistic_regression(X_train, train_df[label_col], C=logreg_cfg["C"], max_iter=logreg_cfg["max_iter"])
        xgb = fit_xgboost(X_train, train_df[label_col], X_valid, valid_df[label_col], xgb_grid, seed=config["random_seed"])

        for model_name, model_type in [("logistic_regime_24h", "logreg"), ("xgboost_regime_24h", "xgb")]:
            for split_name, split_df, X_split in [("validation", valid_df, X_valid), ("test", test_df, X_test)]:
                if model_type == "logreg":
                    pred = logistic.model.predict(X_split)
                    prob = logistic.model.predict_proba(X_split)
                    classes = list(logistic.model.classes_)
                else:
                    pred = decode_labels(xgb.model.predict(X_split)).to_numpy()
                    prob = xgb.model.predict_proba(X_split)
                    classes = ["BUY", "HOLD", "SELL"]
                metric = {
                    "task": "regime_24h",
                    "model": model_name,
                    "fold": fold.fold_name,
                    "split": split_name,
                    "horizon_hours": horizon,
                    "threshold": selected_threshold,
                }
                metric.update(classification_metrics(split_df[label_col], pd.Series(pred)))
                rows.append(metric)

                prob_df = pd.DataFrame(prob, columns=[f"probability_{c.lower()}" for c in classes])
                for col in ["probability_buy", "probability_hold", "probability_sell"]:
                    if col not in prob_df.columns:
                        prob_df[col] = 0.0
                pred_rows.append(
                    pd.concat(
                        [
                            pd.DataFrame(
                                {
                                    "timestamp": split_df["timestamp"].to_numpy(),
                                    "datetime_utc": split_df["datetime_utc"].to_numpy(),
                                    "actual_label": split_df[label_col].to_numpy(),
                                    "predicted_label": pred,
                                    "fold": fold.fold_name,
                                    "split": split_name,
                                    "model": model_name,
                                }
                            ),
                            prob_df[["probability_buy", "probability_hold", "probability_sell"]].reset_index(drop=True),
                        ],
                        axis=1,
                    )
                )
    pd.DataFrame(rows).to_csv(metrics_dir / "alt_regime_metrics.csv", index=False)
    pd.concat(pred_rows, ignore_index=True).to_parquet(predictions_dir / "alt_regime_predictions.parquet", index=False)
    print(f"Saved regime-24h metrics to {metrics_dir}")


def run_alt_targets(config_path: str | None = None, task: AltTask = "binary-events") -> None:
    if task == "binary-events":
        run_alt_binary_events(config_path)
        return
    if task == "direction-24h":
        run_alt_direction_24h(config_path)
        return
    if task == "direction-24h-neutral":
        run_alt_direction_24h_neutral(config_path)
        return
    if task == "regime-24h":
        run_alt_regime_24h(config_path)
        return
    raise ValueError(f"Unsupported alt task: {task}")
