#!/usr/bin/env python3
"""Generate quick-look SB3 training plots from monitor files."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from stable_baselines3.common import results_plotter  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot SB3 training curves from monitor logs.")
    parser.add_argument(
        "--log-dir",
        required=True,
        help="Directory containing SB3 monitor files (e.g. experiments/<run>/logs/sb3/train_monitor).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write plots (defaults to the experiment's plots/ directory).",
    )
    parser.add_argument(
        "--basename",
        default="sb3_training",
        help="Base filename for generated plots.",
    )
    return parser.parse_args()


def plot_series(log_dir: Path, output_dir: Path, basename: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_specs = [
        (results_plotter.X_TIMESTEPS, "Timesteps"),
        (results_plotter.X_EPISODES, "Episodes"),
    ]

    for axis, label in plot_specs:
        plt.figure(figsize=(10, 5))
        results_plotter.plot_results([str(log_dir)], None, axis, f"Rewards vs {label}")
        plt.tight_layout()
        output_path = output_dir / f"{basename}_{label.lower()}.png"
        plt.savefig(output_path)
        plt.close()


def main() -> None:
    args = parse_args()
    log_dir = Path(args.log_dir).expanduser().resolve()
    if not log_dir.exists():
        raise FileNotFoundError(f"Log directory not found: {log_dir}")

    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
    else:
        output_dir = log_dir.parent.parent / "plots"

    plot_series(log_dir, output_dir, args.basename)


if __name__ == "__main__":
    main()
