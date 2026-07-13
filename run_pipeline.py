from __future__ import annotations

import argparse
from pathlib import Path

from src.config import load_config, resolve_path
from src.features import build_features
from src.labels import append_label_columns
from src.preprocessing import preprocess_minute_to_hourly
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
        choices=["prepare-data"],
        help="Pipeline stage to run.",
    )
    parser.add_argument("--config", default=None, help="Optional config path.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.stage == "prepare-data":
        run_prepare_data(args.config)
        return
    raise ValueError(f"Unsupported stage: {args.stage}")


if __name__ == "__main__":
    main()
