"""Utilities for single-agent DRL training and evaluation."""

from __future__ import annotations

import io
import logging
import sys
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
from stable_baselines3.common.logger import KVWriter, configure
from stable_baselines3.common.callbacks import CallbackList, EvalCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

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


class EpisodeSummaryCallback(BaseCallback):
    """Collect per-episode summaries from StockTradingEnv instances and log them."""

    def __init__(self, logger: logging.Logger, log_every_rollout: int = 1) -> None:
        super().__init__()
        self._summary_logger = logger
        self.rollout_counter = 0
        self.log_every_rollout = max(1, log_every_rollout)

    def _on_step(self) -> bool:  # pragma: no cover - required by BaseCallback
        return True

    def _on_rollout_end(self) -> bool:
        self.rollout_counter += 1
        if self.rollout_counter % self.log_every_rollout != 0:
            return True

        summaries_nested = self.training_env.env_method("pop_episode_summaries")
        for worker_summaries in summaries_nested:
            if not worker_summaries:
                continue
            for summary in worker_summaries:
                parts = [
                    "episode_summary",
                    f"day={summary.get('day')}",
                    f"episode={summary.get('episode')}",
                    f"begin_total_asset={summary.get('begin_total_asset'):.2f}",
                    f"end_total_asset={summary.get('end_total_asset'):.2f}",
                    f"total_reward={summary.get('total_reward'):.2f}",
                    f"total_cost={summary.get('total_cost'):.2f}",
                    f"total_trades={summary.get('total_trades')}",
                ]
                sharpe = summary.get("sharpe")
                if sharpe is not None:
                    parts.append(f"sharpe={sharpe:.3f}")
                mode = summary.get("mode")
                if mode:
                    parts.append(f"mode={mode}")
                model_name = summary.get("model_name")
                if model_name:
                    parts.append(f"model={model_name}")
                self._summary_logger.info("|".join(parts))
        return True


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
    buy_cost = env_params.buy_cost_pct
    sell_cost = env_params.sell_cost_pct
    return {
        "hmax": env_params.hmax,
        "initial_amount": amount,
        "num_stock_shares": [0] * stock_dim,
        "buy_cost_pct": [buy_cost] * stock_dim,
        "sell_cost_pct": [sell_cost] * stock_dim,
        "state_space": state_space,
        "stock_dim": stock_dim,
        "tech_indicator_list": config.INDICATORS,
        "action_space": stock_dim,
        "reward_scaling": env_params.reward_scaling,
        "print_verbosity": None,
    }


def train_agent(
    env_train,
    *,
    algo: str,
    algo_cfg: dict,
    total_timesteps: int,
    logger: logging.Logger,
    log_dir: Path,
    eval_env=None,
    eval_freq: int | None = None,
    n_eval_episodes: int = 5,
    summary_log_every: int = 1,
) -> object:
    logger.info("train_algo|%s|timesteps|%s", algo.upper(), total_timesteps)
    agent = DRLAgent(env_train)
    model_kwargs = algo_cfg.get("model_kwargs")
    if model_kwargs is not None:
        model_kwargs = deepcopy(model_kwargs)
    policy_kwargs = algo_cfg.get("policy_kwargs")
    if policy_kwargs is not None:
        policy_kwargs = deepcopy(policy_kwargs)
    policy_name = algo_cfg.get("policy", "MlpPolicy")

    model = agent.get_model(
        algo,
        policy=policy_name,
        model_kwargs=model_kwargs,
        policy_kwargs=policy_kwargs,
        tensorboard_log=None,
    )

    if algo in {"ppo", "ppo_lstm"}:
        logger.info(
            "algo=%s|gSDE=on|sde_freq=%s|deterministic_eval=true|log_std_init=%s|full_std=%s|use_expln=%s",
            algo,
            (model_kwargs or {}).get("sde_sample_freq"),
            (policy_kwargs or {}).get("log_std_init"),
            (policy_kwargs or {}).get("full_std"),
            (policy_kwargs or {}).get("use_expln"),
        )

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    format_strings = ["stdout", "csv"]
    try:  # pragma: no cover - optional tensorboard dependency
        import tensorboard  # type: ignore  # noqa: F401
    except ImportError:
        logger.info("tensorboard is not installed; skipping tensorboard logging output")
    else:
        format_strings.append("tensorboard")
    sb3_logger = configure(str(log_dir), format_strings)
    sb3_logger.output_formats.append(LoggingOutputFormat(logger))
    model.set_logger(sb3_logger)

    callbacks = []
    if summary_log_every and summary_log_every > 0:
        callbacks.append(EpisodeSummaryCallback(logger=logger, log_every_rollout=summary_log_every))
    if eval_env is not None and eval_freq:
        eval_log_dir = log_dir / "eval"
        eval_log_dir.mkdir(parents=True, exist_ok=True)
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(eval_log_dir),
            log_path=str(eval_log_dir),
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            deterministic=True,
        )
        callbacks.append(eval_callback)

    callback_list = CallbackList(callbacks) if callbacks else None

    trained_model = agent.train_model(
        model=model,
        tb_log_name=algo,
        total_timesteps=total_timesteps,
        callback=callback_list,
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
    risk_indicator_col: str = config.RISK_INDICATOR_COL,
    vecnormalize_path: Path | None = None,
) -> pd.DataFrame:
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    def _make_env():
        return StockTradingEnv(
            df=df,
            turbulence_threshold=turbulence_threshold,
            risk_indicator_col=risk_indicator_col,
            **{k: (v.copy() if isinstance(v, list) else v) for k, v in env_kwargs.items()},
        )

    vec_env = DummyVecEnv([_make_env])
    if vecnormalize_path is not None:
        vecnorm_file = Path(vecnormalize_path)
        if not vecnorm_file.exists():
            raise FileNotFoundError(f"VecNormalize statistics not found at {vecnorm_file}")
        vec_env = VecNormalize.load(str(vecnorm_file), vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
    else:
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
        vec_env.training = False

    initial_obs = vec_env.reset()

    env = vec_env.envs[0]

    model_obs_shape = getattr(getattr(model, "observation_space", None), "shape", None)
    vec_obs_shape = getattr(vec_env.observation_space, "shape", None)
    if model_obs_shape is not None and vec_obs_shape is not None and tuple(model_obs_shape) != tuple(vec_obs_shape):
        raise RuntimeError(
            "VecNormalize observation shape %s does not match model observation shape %s. "
            "Verify indicator configuration and normalization artifacts."
            % (vec_obs_shape, model_obs_shape)
        )

    episode_summaries: list[dict] = []

    try:
        df_account, df_actions = DRLAgent.DRL_prediction(
            model=model,
            environment=env,
            vec_env=vec_env,
            initial_obs=initial_obs,
        )
        episode_summaries = env.pop_episode_summaries()
    finally:
        vec_env.close()
    df_account = df_account.copy()

    for summary in episode_summaries:
        parts = [
            "evaluation_summary",
            f"label={label}",
            f"day={summary.get('day')}",
            f"episode={summary.get('episode')}",
            f"begin_total_asset={summary.get('begin_total_asset'):.2f}",
            f"end_total_asset={summary.get('end_total_asset'):.2f}",
            f"total_reward={summary.get('total_reward'):.2f}",
            f"total_cost={summary.get('total_cost'):.2f}",
            f"total_trades={summary.get('total_trades')}",
        ]
        sharpe = summary.get("sharpe")
        if sharpe is not None:
            parts.append(f"sharpe={sharpe:.3f}")
        logger.info("|".join(parts))

    prefix = f"{label}_account_value"
    account_path = results_dir / f"{prefix}.csv"
    actions_path = results_dir / f"{label}_actions.csv"
    df_account.to_csv(account_path, index=False)
    df_actions.to_csv(actions_path, index=False)
    logger.info("saved_metric|%s|path|%s", prefix, account_path)

    value_series = df_account.set_index("date")["account_value"].to_frame("account_value")
    plot_account_values(value_series, plots_dir / f"{label}_account_values.png", title=f"{label.capitalize()} Account Value")

    return df_account
