#!/usr/bin/env python
"""End-to-end training & evaluation pipeline extracted from p3.ipynb.

The script trains the five baseline DRL agents, runs a trade-period
backtest, executes the ensemble rebalancing strategy, and collects all
artifacts (models, logs, metrics, plots) under a dedicated experiment
directory.
"""

from __future__ import annotations

import argparse
import logging
import sys
import io
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable

import matplotlib


CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3.common.logger import configure as sb3_configure

try:  # pragma: no cover - optional dependency
    from torch.utils.tensorboard import SummaryWriter  # noqa: F401

    TENSORBOARD_AVAILABLE = True
except ImportError:  # pragma: no cover
    TENSORBOARD_AVAILABLE = False

import config
from env import StockTradingEnv
from helper_function import (
    backtest_stats,
    check_and_make_directories,
    get_baseline,
)
from models import DRLAgent, DRLEnsembleAgent


DEFAULT_ALGOS: Dict[str, dict] = {
    "a2c": {
        "total_timesteps": 1_000_000,
        "model_kwargs": None,
    },
    "ddpg": {
        "total_timesteps": 100_000,
        "model_kwargs": None,
    },
    "ppo": {
        "total_timesteps": 200_000,
        "model_kwargs": {
            "n_steps": 2048,
            "ent_coef": 0.01,
            "learning_rate": 0.00025,
            "batch_size": 256,
        },
    },
    "td3": {
        "total_timesteps": 200_000,
        "model_kwargs": {
            "batch_size": 200,
            "buffer_size": 1_000_000,
            "learning_rate": 0.001,
        },
    },
    "sac": {
        "total_timesteps": 200_000,
        "model_kwargs": {
            "batch_size": 512,
            "buffer_size": 100_000,
            "learning_rate": 0.0001,
            "learning_starts": 100,
            "ent_coef": "auto_0.1",
        },
    },
}


BASE_ENSEMBLE_TIMESTEPS = {algo: 10_000 for algo in DEFAULT_ALGOS}


@dataclass
class ExperimentPaths:
    root: Path
    models: Path
    tensorboard: Path
    results: Path
    logs: Path
    plots: Path
    sb3_logs: Path


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate DRL agents on KRX data.")
    parser.add_argument(
        "--experiment-name",
        default=None,
        help="Optional name for the experiment directory (defaults to timestamp).",
    )
    parser.add_argument(
        "--output-root",
        default="experiments",
        help="Root directory where experiment folders will be created.",
    )
    parser.add_argument(
        "--turbulence-threshold",
        type=float,
        default=70.0,
        help="Turbulence threshold used during out-of-sample trading.",
    )
    parser.add_argument(
        "--timesteps-scale",
        type=float,
        default=1.0,
        help="Scale factor applied to every training timetable (e.g., 0.1 for a quick dry run).",
    )
    return parser.parse_args(argv)


def make_experiment_dirs(args: argparse.Namespace) -> ExperimentPaths:
    name = args.experiment_name or datetime.now().strftime("%Y%m%d-%H%M%S")
    root = Path(args.output_root).expanduser().resolve() / name
    models_dir = root / "models"
    tensorboard_dir = root / "tensorboard"
    results_dir = root / "results"
    logs_dir = root / "logs"
    plots_dir = root / "plots"
    sb3_logs_dir = root / "sb3_logs"

    for path in [models_dir, tensorboard_dir, results_dir, logs_dir, plots_dir, sb3_logs_dir]:
        path.mkdir(parents=True, exist_ok=True)

    # Update global config directories so downstream utilities respect the experiment layout.
    config.TRAINED_MODEL_DIR = str(models_dir)
    config.TENSORBOARD_LOG_DIR = str(tensorboard_dir)
    config.RESULTS_DIR = str(results_dir)

    check_and_make_directories(
        [
            config.TRAINED_MODEL_DIR,
            config.TENSORBOARD_LOG_DIR,
            config.RESULTS_DIR,
        ]
    )

    return ExperimentPaths(
        root=root,
        models=models_dir,
        tensorboard=tensorboard_dir,
        results=results_dir,
        logs=logs_dir,
        plots=plots_dir,
        sb3_logs=sb3_logs_dir,
    )


class _StreamToLogger(io.TextIOBase):
    """Redirects writes to a logger instance."""

    def __init__(self, logger: logging.Logger, level: int) -> None:
        super().__init__()
        self.logger = logger
        self.level = level

    def write(self, message: str) -> int:  # type: ignore[override]
        message = message.strip()
        if message:
            self.logger.log(self.level, message)
        return len(message)

    def flush(self) -> None:  # pragma: no cover
        pass


def setup_logging(paths: ExperimentPaths) -> logging.Logger:
    log_path = paths.logs / "experiment.log"
    handler = logging.FileHandler(log_path, encoding="utf-8")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[handler],
    )
    logger = logging.getLogger("drl_experiment")
    logger.propagate = False
    logger.info("Logging to %s", log_path)

    logging.captureWarnings(True)
    sys.stdout = _StreamToLogger(logger, logging.INFO)
    return logger


def load_dataset(name: str) -> pd.DataFrame:
    df = pd.read_csv(Path("data") / name, dtype={"tic": str})
    df = df.sort_values(["date", "tic"]).reset_index(drop=True)
    df.index = df["date"].factorize()[0]
    return df


def build_env_kwargs(stock_dim: int, state_space: int, initial_amount: int = 10_000_000) -> dict:
    cost = 0.001
    return {
        "hmax": 100,
        "initial_amount": initial_amount,
        "num_stock_shares": [0] * stock_dim,
        "buy_cost_pct": [cost] * stock_dim,
        "sell_cost_pct": [cost] * stock_dim,
        "state_space": state_space,
        "stock_dim": stock_dim,
        "tech_indicator_list": config.INDICATORS,
        "action_space": stock_dim,
        "reward_scaling": 1e-4,
    }


def train_agents(
    env_train,
    algorithms: Dict[str, dict],
    tensorboard_dir: Path,
    sb3_log_dir: Path,
    logger: logging.Logger,
    scale: float,
) -> Dict[str, object]:
    trained_models = {}
    for algo, cfg in algorithms.items():
        timesteps = max(1, int(cfg["total_timesteps"] * scale))
        logger.info(
            "Training %s for %s timesteps (scale %.3f)", algo.upper(), timesteps, scale
        )
        agent = DRLAgent(env_train)
        model = agent.get_model(
            algo,
            model_kwargs=cfg.get("model_kwargs"),
            tensorboard_log=str(tensorboard_dir / algo),
        )

        log_path = sb3_log_dir / algo
        log_path.mkdir(parents=True, exist_ok=True)
        sink_formats = ["stdout", "csv"]
        if TENSORBOARD_AVAILABLE:
            sink_formats.append("tensorboard")
        sb3_logger = sb3_configure(str(log_path), sink_formats)
        model.set_logger(sb3_logger)

        trained_model = agent.train_model(
            model=model,
            tb_log_name=algo,
            total_timesteps=timesteps,
        )
        save_path = Path(config.TRAINED_MODEL_DIR) / f"agent_{algo}.zip"
        trained_model.save(save_path)
        logger.info("Saved %s model to %s", algo.upper(), save_path)
        trained_models[algo] = trained_model
    return trained_models


def run_trade_backtest(
    models: Dict[str, object],
    trade_df: pd.DataFrame,
    env_kwargs: dict,
    results_dir: Path,
    plots_dir: Path,
    turbulence_threshold: float,
    logger: logging.Logger,
) -> pd.DataFrame:
    logger.info("Running trade backtest for individual agents")
    e_trade_gym = StockTradingEnv(
        df=trade_df,
        turbulence_threshold=turbulence_threshold,
        risk_indicator_col="vix",
        **env_kwargs,
    )

    account_values = {}
    for algo, model in models.items():
        df_account, df_actions = DRLAgent.DRL_prediction(model=model, environment=e_trade_gym)
        df_account = df_account.copy()
        algo_prefix = algo.lower()
        account_path = results_dir / f"trade_account_value_{algo_prefix}.csv"
        actions_path = results_dir / f"trade_actions_{algo_prefix}.csv"
        df_account.to_csv(account_path, index=False)
        df_actions.to_csv(actions_path, index=False)
        logger.info("Saved %s account value to %s", algo.upper(), account_path)
        account_values[algo] = df_account.set_index("date")["account_value"]

    result_df = pd.DataFrame(account_values)
    result_df.index.name = "date"
    result_df.index = pd.to_datetime(result_df.index)
    comparison_csv = results_dir / "trade_account_comparison.csv"
    result_df.to_csv(comparison_csv)
    logger.info("Stored trade comparison curve to %s", comparison_csv)

    plt.figure(figsize=(14, 5))
    for algo, series in result_df.items():
        plt.plot(series.index, series.values, label=algo.upper())
    plt.legend()
    plt.title("Trade Period Account Values")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plot_path = plots_dir / "trade_account_values.png"
    plt.savefig(plot_path)
    plt.close()
    logger.info("Saved trade backtest plot to %s", plot_path)

    return result_df


def run_ensemble(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    trade_df: pd.DataFrame,
    logger: logging.Logger,
    results_dir: Path,
    plots_dir: Path,
    timesteps: Dict[str, int],
) -> pd.DataFrame:
    data = pd.concat([train_df, test_df, trade_df], axis=0, ignore_index=True)
    stock_dim = len(train_df.tic.unique())
    state_space = 1 + 2 * stock_dim + len(config.INDICATORS) * stock_dim

    unique_trade_date = test_df.date.unique()
    trade_len = len(unique_trade_date)
    rebalance_window = config.REBALANCE_WINDOW
    validation_window = config.VALIDATION_WINDOW
    if rebalance_window + validation_window >= trade_len:
        rebalance_window = max(1, trade_len // 4)
        validation_window = max(1, trade_len // 4)
        logger.warning(
            "Adjusting ensemble windows to rebalance=%s, validation=%s for trade length %s",
            rebalance_window,
            validation_window,
            trade_len,
        )

    env_kwargs = {
        "stock_dim": stock_dim,
        "hmax": 100,
        "initial_amount": 1_000_000,
        "buy_cost_pct": 0.001,
        "sell_cost_pct": 0.001,
        "reward_scaling": 1e-4,
        "state_space": state_space,
        "action_space": stock_dim,
        "tech_indicator_list": config.INDICATORS,
        "print_verbosity": 5,
    }

    ensemble_agent = DRLEnsembleAgent(
        df=data,
        train_period=(config.TRAIN_START_DATE, config.TRAIN_END_DATE),
        val_test_period=(config.TEST_START_DATE, config.TEST_END_DATE),
        rebalance_window=rebalance_window,
        validation_window=validation_window,
        **env_kwargs,
    )

    logger.info("Running ensemble strategy")
    summary = ensemble_agent.run_ensemble_strategy(
        config.A2C_PARAMS,
        config.PPO_PARAMS,
        config.DDPG_PARAMS,
        config.SAC_PARAMS,
        config.TD3_PARAMS,
        timesteps,
    )

    summary_path = results_dir / "ensemble_summary.csv"
    summary.to_csv(summary_path, index=False)
    logger.info("Saved ensemble summary to %s", summary_path)

    account_frames = []
    for i in range(
        rebalance_window + validation_window,
        len(unique_trade_date) + 1,
        rebalance_window,
    ):
        path = results_dir / f"account_value_trade_ensemble_{i}.csv"
        if not path.exists():
            logger.warning("Expected ensemble account value file missing: %s", path)
            continue
        account_frames.append(pd.read_csv(path))

    if not account_frames:
        available = sorted(
            p.name for p in results_dir.glob("account_value_trade_ensemble_*.csv")
        )
        logger.error("No ensemble account value files were produced. Found: %s", available)
        raise RuntimeError("No ensemble account value files were produced.")

    df_account_value = pd.concat(account_frames, ignore_index=True)
    df_account_value = (
        df_account_value.drop_duplicates(subset="date")
        .assign(date=lambda df: pd.to_datetime(df["date"]))
        .sort_values("date")
        .reset_index(drop=True)
    )
    df_account_value.to_csv(results_dir / "ensemble_account_value.csv", index=False)

    df_account_value.set_index("date")["account_value"].plot(
        figsize=(14, 5), title="Ensemble Account Value"
    )
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.xticks(rotation=45)
    plt.tight_layout()
    ensemble_plot_path = plots_dir / "ensemble_account_value.png"
    plt.savefig(ensemble_plot_path)
    plt.close()
    logger.info("Saved ensemble backtest plot to %s", ensemble_plot_path)

    return df_account_value


def compute_metrics(
    df_account_value: pd.DataFrame,
    logger: logging.Logger,
    results_dir: Path,
    plots_dir: Path,
    baseline_symbol: str = "^KS11",
) -> None:
    df_account_value = df_account_value.copy()
    df_account_value["date"] = pd.to_datetime(df_account_value["date"]) 
    df_account_value["daily_return"] = df_account_value["account_value"].pct_change()
    sharpe = (
        np.sqrt(252)
        * df_account_value["daily_return"].mean()
        / df_account_value["daily_return"].std()
    )
    logger.info("Ensemble Sharpe (approx.): %.4f", sharpe)

    try:
        baseline = get_baseline(
            ticker=baseline_symbol,
            start=df_account_value.iloc[0]["date"].strftime("%Y-%m-%d"),
            end=df_account_value.iloc[-1]["date"].strftime("%Y-%m-%d"),
        )
        baseline["date"] = pd.to_datetime(baseline["date"])
        baseline_path = results_dir / "baseline.csv"
        baseline.to_csv(baseline_path, index=False)
        logger.info("Saved baseline series (%s) to %s", baseline_symbol, baseline_path)
    except Exception as exc:  # pragma: no cover - network failures
        logger.warning("Unable to download baseline data: %s", exc)
        baseline = None

    try:
        stats = backtest_stats(df_account_value)
        stats_path = results_dir / "ensemble_backtest_stats.csv"
        stats.to_csv(stats_path)
        logger.info("Saved detailed backtest stats to %s", stats_path)
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.warning("Failed to compute full backtest stats: %s", exc)

    if baseline is not None:
        aligned = pd.merge(
            df_account_value[["date", "account_value"]],
            baseline[["date", "close"]],
            on="date",
            how="left",
        ).ffill()
        aligned["baseline_value"] = (
            aligned["close"] / aligned["close"].iloc[0] * aligned["account_value"].iloc[0]
        )
        aligned.to_csv(results_dir / "ensemble_vs_baseline.csv", index=False)

        plt.figure(figsize=(14, 5))
        plt.plot(aligned["date"], aligned["account_value"], label="Ensemble")
        plt.plot(aligned["date"], aligned["baseline_value"], label=baseline_symbol)
        plt.legend()
        plt.title("Ensemble vs Baseline")
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plots_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(plots_dir / "ensemble_vs_baseline.png")
        plt.close()


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    paths = make_experiment_dirs(args)
    logger = setup_logging(paths)

    logger.info("Loading datasets")
    train_df = load_dataset("train_data.csv")
    test_df = load_dataset("stock_test_data.csv")
    trade_df = load_dataset("trade_data.csv")
    stock_dim = len(train_df.tic.unique())
    state_space = 1 + 2 * stock_dim + len(config.INDICATORS) * stock_dim
    logger.info("Stock dimension: %s | State space: %s", stock_dim, state_space)

    env_kwargs = build_env_kwargs(stock_dim=stock_dim, state_space=state_space)
    e_train_gym = StockTradingEnv(df=train_df, **env_kwargs)
    env_train, _ = e_train_gym.get_sb_env()

    trained_models = train_agents(
        env_train,
        DEFAULT_ALGOS,
        tensorboard_dir=paths.tensorboard,
        sb3_log_dir=paths.sb3_logs,
        logger=logger,
        scale=args.timesteps_scale,
    )

    trade_env_kwargs = build_env_kwargs(stock_dim=stock_dim, state_space=state_space, initial_amount=1_000_000)
    trade_results = run_trade_backtest(
        trained_models,
        trade_df=trade_df,
        env_kwargs=trade_env_kwargs,
        results_dir=paths.results,
        plots_dir=paths.plots,
        turbulence_threshold=args.turbulence_threshold,
        logger=logger,
    )
    logger.info("Trade backtest completed with final values: %s", trade_results.tail(1).to_dict("records"))

    ensemble_timesteps = {
        algo: max(1, int(base * args.timesteps_scale))
        for algo, base in BASE_ENSEMBLE_TIMESTEPS.items()
    }

    df_account_value = run_ensemble(
        train_df=train_df,
        test_df=test_df,
        trade_df=trade_df,
        logger=logger,
        results_dir=paths.results,
        plots_dir=paths.plots,
        timesteps=ensemble_timesteps,
    )

    compute_metrics(
        df_account_value,
        logger=logger,
        results_dir=paths.results,
        plots_dir=paths.plots,
    )

    logger.info("Experiment artifacts available under %s", paths.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
