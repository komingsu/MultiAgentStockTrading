"""Utilities for single-agent DRL training and evaluation."""

from __future__ import annotations

import io
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
from stable_baselines3.common.logger import KVWriter, Logger

import config
from env import StockTradingEnv
from helper_function import check_and_make_directories
from models import DRLAgent
from vis_util import plot_account_values
from hyperparams import ENV_PARAMS, PORTFOLIO_INIT, EnvHyperParams

@dataclass
class ExperimentPaths:
    root: Path
    models: Path
    results: Path
    logs: Path
    plots: Path


class _StreamToLogger(io.TextIOBase):
    """Redirect stdout/stderr to the experiment logger."""

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


class LoggingOutputFormat(KVWriter):
    """Simple SB3 logger output that prints key/value pairs to experiment.log."""

    def __init__(self, logger: logging.Logger) -> None:
        self.logger = logger

    def write(self, kvs, key_excluded=None, step=None):
        pairs = [f"{key}|{kvs[key]}" for key in sorted(kvs.keys())]
        if step is not None:
            pairs.append(f"step|{step}")
        self.logger.info(" | ".join(pairs))

    def close(self) -> None:  # pragma: no cover
        pass


def make_experiment_dirs(experiment_name: str | None, output_root: str = "experiments") -> ExperimentPaths:
    name = experiment_name or datetime.now().strftime("%Y%m%d-%H%M%S")
    root = Path(output_root).expanduser() / name
    models_dir = root / "models"
    results_dir = root / "results"
    logs_dir = root / "logs"
    plots_dir = root / "plots"

    for path in [models_dir, results_dir, logs_dir, plots_dir]:
        path.mkdir(parents=True, exist_ok=True)

    config.TRAINED_MODEL_DIR = str(models_dir)
    config.RESULTS_DIR = str(results_dir)

    check_and_make_directories([config.TRAINED_MODEL_DIR, config.RESULTS_DIR])

    return ExperimentPaths(
        root=root,
        models=models_dir,
        results=results_dir,
        logs=logs_dir,
        plots=plots_dir,
    )


def setup_logging(paths: ExperimentPaths, logger_name: str = "drl_experiment") -> logging.Logger:
    log_path = paths.logs / "experiment.log"
    logger = logging.getLogger(logger_name)
    logger.handlers.clear()
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s|%(levelname)s|%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    sys.stdout = _StreamToLogger(logger, logging.INFO)
    sys.stderr = _StreamToLogger(logger, logging.ERROR)
    logger.info("log_path|%s", log_path)
    return logger


def load_dataset(name: str) -> pd.DataFrame:
    df = pd.read_csv(Path("data") / name, dtype={"tic": str})
    df = df.sort_values(["date", "tic"]).reset_index(drop=True)
    df.index = df["date"].factorize()[0]
    return df


def build_env_kwargs(
    stock_dim: int,
    state_space: int,
    *,
    env_params: EnvHyperParams = ENV_PARAMS,
    initial_amount: float | None = None,
) -> dict:
    amount = initial_amount if initial_amount is not None else PORTFOLIO_INIT.initial_cash
    cost = env_params.transaction_cost_pct
    return {
        "hmax": env_params.hmax,
        "initial_amount": amount,
        "num_stock_shares": [0] * stock_dim,
        "buy_cost_pct": [cost] * stock_dim,
        "sell_cost_pct": [cost] * stock_dim,
        "state_space": state_space,
        "stock_dim": stock_dim,
        "tech_indicator_list": config.INDICATORS,
        "action_space": stock_dim,
        "reward_scaling": env_params.reward_scaling,
    }


def train_agent(
    env_train,
    *,
    algo: str,
    algo_cfg: dict,
    total_timesteps: int,
    logger: logging.Logger,
) -> object:
    logger.info("train_algo|%s|timesteps|%s", algo.upper(), total_timesteps)
    agent = DRLAgent(env_train)
    model = agent.get_model(
        algo,
        model_kwargs=algo_cfg.get("model_kwargs"),
        tensorboard_log=None,
    )

    sb3_logger = Logger(folder=None, output_formats=[LoggingOutputFormat(logger)])
    model.set_logger(sb3_logger)

    trained_model = agent.train_model(
        model=model,
        tb_log_name=algo,
        total_timesteps=total_timesteps,
    )
    return trained_model


def evaluate_agent(
    model,
    df: pd.DataFrame,
    *,
    env_kwargs: dict,
    results_dir: Path,
    plots_dir: Path,
    logger: logging.Logger,
    label: str,
    turbulence_threshold: float | None = None,
    risk_indicator_col: str = "vix",
) -> pd.DataFrame:
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    env = StockTradingEnv(
        df=df,
        turbulence_threshold=turbulence_threshold,
        risk_indicator_col=risk_indicator_col,
        **{k: (v.copy() if isinstance(v, list) else v) for k, v in env_kwargs.items()},
    )

    df_account, df_actions = DRLAgent.DRL_prediction(model=model, environment=env)
    df_account = df_account.copy()

    prefix = f"{label}_account_value"
    account_path = results_dir / f"{prefix}.csv"
    actions_path = results_dir / f"{label}_actions.csv"
    df_account.to_csv(account_path, index=False)
    df_actions.to_csv(actions_path, index=False)
    logger.info("saved_metric|%s|path|%s", prefix, account_path)

    value_series = df_account.set_index("date")["account_value"].to_frame("account_value")
    plot_account_values(value_series, plots_dir / f"{label}_account_values.png", title=f"{label.capitalize()} Account Value")

    return df_account
