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
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize


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
    parser.add_argument(
        "--disable-eval",
        action="store_true",
        help="Skip evaluation during training and post-training assessment.",
    )
    parser.add_argument(
        "--eval-freq-scale",
        type=float,
        default=1.0,
        help="Multiplier applied to the default evaluation frequency (>=0 to disable).",
    )
    parser.add_argument(
        "--n-eval-episodes",
        type=int,
        default=5,
        help="Number of evaluation episodes when running EvalCallback.",
    )
    parser.add_argument(
        "--summary-log-every",
        type=int,
        default=1,
        help="Log environment episode summaries every N rollouts (<=0 disables logging).",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=8,
        help="Number of parallel environments used for training.",
    )
    parser.add_argument(
        "--vec-env",
        choices=("subproc", "dummy"),
        default="subproc",
        help="Vectorized environment type for training.",
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
    env_kwargs = build_env_kwargs(
        stock_dim=stock_dim,
        state_space=state_space,
        env_params=ENV_PARAMS,
        initial_amount=PORTFOLIO_INIT.initial_cash,
    )

    sb3_log_dir = paths.logs / "sb3"
    train_monitor_dir = sb3_log_dir / "train_monitor"
    eval_monitor_dir = sb3_log_dir / "eval_monitor"

    n_envs = max(1, args.n_envs)
    vec_env_cls = SubprocVecEnv if (args.vec_env == "subproc" and n_envs > 1) else DummyVecEnv

    def make_train_env():
        kwargs = {k: (v.copy() if isinstance(v, list) else v) for k, v in env_kwargs.items()}
        return StockTradingEnv(
            df=train_df,
            random_start=True,
            portfolio_config=PORTFOLIO_INIT,
            **kwargs,
        )

    train_env = make_vec_env(
        make_train_env,
        n_envs=n_envs,
        vec_env_cls=vec_env_cls,
        monitor_dir=str(train_monitor_dir),
        seed=0,
    )
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    eval_vec_env = None
    eval_freq = None
    if not args.disable_eval:
        eval_df = load_dataset("stock_test_data.csv")
        eval_env_kwargs = build_env_kwargs(
            stock_dim=stock_dim,
            state_space=state_space,
            env_params=ENV_PARAMS,
            initial_amount=PORTFOLIO_INIT.initial_cash,
        )

        def make_eval_env():
            kwargs = {k: (v.copy() if isinstance(v, list) else v) for k, v in eval_env_kwargs.items()}
            return StockTradingEnv(
                df=eval_df,
                random_start=False,
                **kwargs,
            )

        eval_vec_env = make_vec_env(
            make_eval_env,
            n_envs=1,
            monitor_dir=str(eval_monitor_dir),
            seed=10,
        )
        eval_vec_env = VecNormalize(eval_vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
        eval_vec_env.training = False
        eval_vec_env.norm_reward = False
        if hasattr(train_env, "obs_rms") and train_env.obs_rms is not None:
            eval_vec_env.obs_rms = copy.deepcopy(train_env.obs_rms)
        if hasattr(train_env, "ret_rms") and train_env.ret_rms is not None:
            eval_vec_env.ret_rms = copy.deepcopy(train_env.ret_rms)
        if hasattr(train_env, "clip_obs"):
            eval_vec_env.clip_obs = train_env.clip_obs
        if hasattr(train_env, "clip_reward"):
            eval_vec_env.clip_reward = train_env.clip_reward

        base_eval_freq = max(1, total_timesteps // 10)
        scaled = int(base_eval_freq * max(0.0, args.eval_freq_scale))
        eval_freq = max(1, scaled) if scaled > 0 else None
        logger.info("eval_config|freq|%s|episodes|%s", eval_freq, args.n_eval_episodes)
    else:
        logger.info("eval_config|disabled|true")

    trained_model = train_agent(
        train_env,
        algo=algo_key,
        algo_cfg=algo_cfg,
        total_timesteps=total_timesteps,
        logger=logger,
        log_dir=sb3_log_dir,
        eval_env=eval_vec_env,
        eval_freq=eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        summary_log_every=args.summary_log_every,
    )

    save_path = paths.models / f"agent_{algo_key}.zip"
    trained_model.save(save_path)
    logger.info("saved_model|%s", save_path)

    vecnorm_path = paths.models / "vecnormalize.pkl"
    train_env.save(str(vecnorm_path))

    if not args.disable_eval:
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
                vecnormalize_path=vecnorm_path,
            )
    else:
        logger.info("post_training_evaluation|skipped|true")

    logger.info("experiment_root|%s", paths.root)

    train_env.close()
    if eval_vec_env is not None:
        eval_vec_env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
