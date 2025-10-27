#!/usr/bin/env python3
"""Train a single DRL agent and evaluate it on test/trade periods."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Iterable

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import config
from env import StockTradingEnv
from hyperparams import DEFAULT_ALGO_CONFIG, ENV_PARAMS, PORTFOLIO_INIT
from train import (
    build_env_kwargs,
    evaluate_agent,
    load_dataset,
    make_experiment_dirs,
    setup_logging,
    train_agent,
)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a single DRL agent on KRX data.")
    parser.add_argument("--experiment-name", default=None, help="Name for the experiment directory.")
    parser.add_argument("--output-root", default="experiments", help="Root directory for experiments.")
    parser.add_argument(
        "--algo",
        required=True,
        choices=sorted(DEFAULT_ALGO_CONFIG.keys()),
        help="Which DRL algorithm to train.",
    )
    parser.add_argument(
        "--timesteps-scale",
        type=float,
        default=1.0,
        help="Scale factor applied to the default number of training timesteps.",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=None,
        help="Override base timesteps before scaling (optional).",
    )
    parser.add_argument(
        "--turbulence-threshold",
        type=float,
        default=70.0,
        help="Turbulence threshold used for trade-period evaluation.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)

    paths = make_experiment_dirs(args.experiment_name, args.output_root)
    logger = setup_logging(paths)

    algo_key = args.algo.lower()
    algo_cfg = copy.deepcopy(DEFAULT_ALGO_CONFIG[algo_key])
    base_timesteps = args.total_timesteps or algo_cfg["timesteps"]
    total_timesteps = max(1, int(base_timesteps * args.timesteps_scale))

    logger.info("selected_algo|%s|timesteps|%s", algo_key.upper(), total_timesteps)

    train_df = load_dataset("train_data.csv")
    stock_dim = len(train_df.tic.unique())
    state_space = 1 + 2 * stock_dim + len(config.INDICATORS) * stock_dim
    env_kwargs = build_env_kwargs(stock_dim=stock_dim, state_space=state_space, env_params=ENV_PARAMS, initial_amount=PORTFOLIO_INIT.initial_cash)

    env_train = StockTradingEnv(
        df=train_df,
        random_start=True,
        portfolio_config=PORTFOLIO_INIT,
        **env_kwargs,
    )
    sb_env, _ = env_train.get_sb_env()

    trained_model = train_agent(
        sb_env,
        algo=algo_key,
        algo_cfg=algo_cfg,
        total_timesteps=total_timesteps,
        logger=logger,
    )

    save_path = paths.models / f"agent_{algo_key}.zip"
    trained_model.save(save_path)
    logger.info("saved_model|%s", save_path)

    eval_env_kwargs = build_env_kwargs(
        stock_dim=stock_dim,
        state_space=state_space,
        env_params=ENV_PARAMS,
        initial_amount=PORTFOLIO_INIT.initial_cash,
    )

    for label, dataset, turbulence in (
        ("test", "stock_test_data.csv", None),
        ("trade", "trade_data.csv", args.turbulence_threshold),
    ):
        df = load_dataset(dataset)
        evaluate_agent(
            trained_model,
            df,
            env_kwargs=eval_env_kwargs,
            results_dir=paths.results,
            plots_dir=paths.plots,
            logger=logger,
            label=label,
            turbulence_threshold=turbulence,
        )

    logger.info("experiment_root|%s", paths.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
