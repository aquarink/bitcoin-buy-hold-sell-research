from __future__ import annotations

import argparse
from pathlib import Path

from src.config import load_config, resolve_path
from src.features import build_features
from src.labels import append_label_columns
from src.preprocessing import preprocess_minute_to_hourly
from src.stage_alt_targets_runner import run_alt_targets
from src.stage_ensemble_runner import run_stage_ensemble
from src.stage3_confidence_runner import run_stage3_confidence
from src.stage3_dual_binary_runner import run_stage3_dual_binary
from src.stage3_runner import run_stage3
from src.stage4_lstm_runner import run_stage4_lstm
from src.visualization import plot_hourly_eda


def run_prepare_data(config_path: str | None = None) -> None:
    config = load_config(config_path)
    paths = config["paths"]
    dataset = config["dataset"]
    labels = config["labels"]

    raw_path = resolve_path(paths["raw_data"])
    processed_1h_path = resolve_path(paths["processed_1h"])
    featured_path = resolve_path(paths["featured_1h"])
    labeled_path = resolve_path(paths["labeled_1h"])
    figures_dir = resolve_path(paths["figures_dir"])

    artifacts = preprocess_minute_to_hourly(raw_path, processed_1h_path)
    feature_df = build_features(artifacts.hourly)
    label_df = append_label_columns(
        feature_df,
        horizons=labels["horizons_hours"],
        thresholds=labels["threshold_candidates"],
    )

    featured_path.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_parquet(featured_path, index=False)
    label_df.to_parquet(labeled_path, index=False)
    artifacts.yearly_stats.to_csv(resolve_path("outputs/metrics/hourly_yearly_stats.csv"), index=False)
    plot_hourly_eda(artifacts.hourly, figures_dir)

    print(f"Saved hourly parquet: {processed_1h_path}")
    print(f"Saved feature parquet: {featured_path}")
    print(f"Saved labeled parquet: {labeled_path}")
    print(f"Final holdout start remains fixed at {dataset['final_holdout_start']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BTC signal research pipeline")
    parser.add_argument(
        "--stage",
        default="prepare-data",
        choices=["prepare-data", "stage3-classical", "stage3-dual-binary", "stage3-confidence-threshold", "alt-targets", "ensemble", "stage4-lstm"],
        help="Pipeline stage to run.",
    )
    parser.add_argument("--config", default=None, help="Optional config path.")
    parser.add_argument(
        "--task",
        default="binary-events",
        choices=["binary-events", "direction-24h", "direction-24h-neutral", "regime-24h"],
        help="Sub-task for alt-targets stage.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.stage == "prepare-data":
        run_prepare_data(args.config)
        return
    if args.stage == "stage3-classical":
        run_stage3(args.config)
        return
    if args.stage == "stage3-dual-binary":
        run_stage3_dual_binary(args.config)
        return
    if args.stage == "stage3-confidence-threshold":
        run_stage3_confidence(args.config)
        return
    if args.stage == "alt-targets":
        run_alt_targets(args.config, task=args.task)
        return
    if args.stage == "ensemble":
        run_stage_ensemble(args.config)
        return
    if args.stage == "stage4-lstm":
        run_stage4_lstm(args.config)
        return
    raise ValueError(f"Unsupported stage: {args.stage}")


if __name__ == "__main__":
    main()
