from __future__ import annotations

import numpy as np
import pandas as pd

from .backtesting import run_long_flat_backtest, summarize_backtest
from .baselines import CLASS_ORDER
from .config import load_config, resolve_path
from .evaluation import classification_metrics, confidence_summary, predicted_distribution
from .experiment_scope import apply_experiment_min_datetime
from .stage3_runner import _build_predictions_frame, _feature_columns, _metrics_row, _prepare_label_col, _sanitize_features, _subset_for_fold, PredictionBundle
from .temporal_split import generate_expanding_folds
from .train_lstm import (
    build_sequences,
    decode_labels,
    encode_labels,
    fit_preprocessors,
    predict_lstm,
    train_lstm_model,
    transform_features,
)


def run_stage4_lstm(config_path: str | None = None) -> None:
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
    threshold = config["stage3"]["selected_threshold"] or 0.005
    horizon = config["stage3"]["primary_horizon_hours"]
    label_col = _prepare_label_col(horizon, threshold)
    future_return_col = f"future_return_{horizon}h"
    folds = generate_expanding_folds(
        start_train_year=config["dataset"]["experiment_start_year"],
        first_validation_year=config["dataset"]["validation_start_year"],
        last_test_year=config["dataset"]["last_pre_holdout_year"],
    )
    stage4 = config["stage4"]
    seq_len = int(stage4["enabled_sequence_lengths"][0])
    fold_years = set(stage4["fold_test_years"])

    metric_rows: list[dict] = []
    prediction_frames: list[pd.DataFrame] = []
    backtest_rows: list[dict] = []
    lookup = df.set_index("timestamp")[["Open", "Close"]]

    for fold in folds:
        if fold.test_start.year not in fold_years:
            continue
        train_df = _subset_for_fold(df, fold.train_start, fold.train_end, label_col)
        valid_df = _subset_for_fold(df, fold.validation_start, fold.validation_end, label_col)
        test_df = _subset_for_fold(df, fold.test_start, fold.test_end, label_col)

        fit_df = train_df.copy()
        valid_context = df[(df["datetime_utc"] >= (fold.validation_start - pd.Timedelta(hours=seq_len - 1))) & (df["datetime_utc"] <= fold.validation_end)].copy()
        test_context = df[(df["datetime_utc"] >= (fold.test_start - pd.Timedelta(hours=seq_len - 1))) & (df["datetime_utc"] <= fold.test_end)].copy()

        imputer, scaler = fit_preprocessors(_sanitize_features(fit_df, feature_cols))
        train_features = transform_features(_sanitize_features(train_df, feature_cols), imputer, scaler)
        valid_features = transform_features(_sanitize_features(valid_context, feature_cols), imputer, scaler)
        test_features = transform_features(_sanitize_features(test_context, feature_cols), imputer, scaler)

        y_train = encode_labels(train_df[label_col]).astype(float)
        y_valid_context = valid_context[label_col].map({c: i for i, c in enumerate(CLASS_ORDER)}).to_numpy(dtype=float)
        y_test_context = test_context[label_col].map({c: i for i, c in enumerate(CLASS_ORDER)}).to_numpy(dtype=float)

        X_train_seq, y_train_seq, train_target_idx = build_sequences(train_features, y_train, seq_len)
        X_valid_seq, y_valid_seq, valid_target_idx = build_sequences(valid_features, y_valid_context, seq_len)
        X_test_seq, y_test_seq, test_target_idx = build_sequences(test_features, y_test_context, seq_len)

        model, valid_score = train_lstm_model(
            X_train_seq=X_train_seq,
            y_train=y_train_seq,
            X_valid_seq=X_valid_seq,
            y_valid=y_valid_seq,
            input_size=X_train_seq.shape[-1],
            hidden_size=stage4["hidden_size"],
            num_layers=stage4["num_layers"],
            dropout=stage4["dropout"],
            batch_size=stage4["batch_size"],
            learning_rate=stage4["learning_rate"],
            max_epochs=stage4["max_epochs"],
            patience=stage4["patience"],
            seed=config["random_seed"],
        )

        for split_name, context_df, X_seq, y_seq, target_idx in [
            ("validation", valid_context, X_valid_seq, y_valid_seq, valid_target_idx),
            ("test", test_context, X_test_seq, y_test_seq, test_target_idx),
        ]:
            pred_enc, prob = predict_lstm(model, X_seq, batch_size=stage4["batch_size"])
            pred_labels = decode_labels(pred_enc)
            prob_df = pd.DataFrame(prob, columns=[f"probability_{c.lower()}" for c in CLASS_ORDER])
            target_rows = context_df.iloc[target_idx].reset_index(drop=True)
            pred_labels.index = target_rows.index
            pred_bundle = PredictionBundle(labels=pred_labels, probabilities=prob_df)
            row = _metrics_row(
                target_rows[label_col],
                pred_bundle,
                "lstm",
                fold.fold_name,
                split_name,
                fold.window_type,
                threshold,
                horizon,
            )
            row["sequence_length"] = seq_len
            row["validation_macro_f1_best"] = valid_score
            metric_rows.append(row)
            prediction_frames.append(
                _build_predictions_frame(target_rows, label_col, pred_bundle, fold.fold_name, split_name, "lstm", fold.window_type, future_return_col)
            )

    metrics_df = pd.DataFrame(metric_rows)
    predictions_df = pd.concat(prediction_frames, ignore_index=True)
    metrics_df.to_csv(metrics_dir / "stage4_lstm_classification_metrics.csv", index=False)
    predictions_df.to_parquet(predictions_dir / "stage4_lstm_predictions.parquet", index=False)
    predictions_df.to_csv(predictions_dir / "stage4_lstm_predictions.csv", index=False)

    test_pred = predictions_df[predictions_df["split"] == "test"].sort_values("datetime_utc").reset_index(drop=True)
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
    summary.update({"model": "lstm", "sequence_length": seq_len})
    backtest_rows.append(summary)
    bt.to_csv(metrics_dir / "backtest_lstm.csv", index=False)
    pd.DataFrame(backtest_rows).to_csv(metrics_dir / "stage4_lstm_backtest_metrics.csv", index=False)
    print(f"Saved LSTM metrics to {metrics_dir}")
    print(f"Saved LSTM predictions to {predictions_dir}")
