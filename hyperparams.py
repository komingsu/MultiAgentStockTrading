from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

import torch.nn as nn


def linear_schedule(initial_value: float, final_value: float) -> Callable[[float], float]:
    """Create a linear learning-rate schedule from initial to final value."""

    def _schedule(progress_remaining: float) -> float:
        return final_value + (initial_value - final_value) * progress_remaining

    return _schedule


@dataclass(frozen=True)
class EnvHyperParams:
    hmax: int = 20
    buy_cost_pct: float = 0.004
    sell_cost_pct: float = 0.001
    reward_scaling: float = 1e-4


@dataclass(frozen=True)
class PortfolioInitParams:
    total_capital: int = 100_000_000
    initial_cash: int = 50_000_000
    target_stock_value: int = 50_000_000
    per_ticker_value_cap_ratio: float = 0.25
    min_positions: int = 3
    max_positions: int = 8


ENV_PARAMS = EnvHyperParams()
PORTFOLIO_INIT = PortfolioInitParams()


DEFAULT_ALGO_CONFIG: Dict[str, dict] = {
    "a2c": {
        "timesteps": 1_000_000,
        "model_kwargs": {
            "n_steps": 5,
            "ent_coef": 0.01,
            "learning_rate": 0.0007,
        },
    },
    "ddpg": {
        "timesteps": 100_000,
        "model_kwargs": {
            "batch_size": 128,
            "buffer_size": 50_000,
            "learning_rate": 0.001,
            "tau": 0.001,
            "gamma": 0.99,
        },
    },
    "ppo": {
        "timesteps": 1_000_000,
        "model_kwargs": {
            "n_steps": 512,
            "batch_size": 512,
            "ent_coef": 0.01,
            "learning_rate": linear_schedule(3e-4, 3e-5),
            "target_kl": 0.015,
            "use_sde": True,
            "sde_sample_freq": 4,
        },
        "policy_kwargs": {
            "net_arch": [
                {
                    "pi": [256, 256, 128],
                    "vf": [256, 256, 128],
                }
            ],
            "activation_fn": nn.ReLU,
            "ortho_init": False,
            "log_std_init": -2,
            "full_std": False,
            "use_expln": True,
        },
    },
    "ppo_lstm": {
        "timesteps": 1_000_000,
        "policy": "MlpLstmPolicy",
        "model_kwargs": {
            "n_steps": 512,
            "batch_size": 512,
            "ent_coef": 0.01,
            "learning_rate": linear_schedule(3e-4, 3e-5),
            "target_kl": 0.015,
            "use_sde": True,
            "sde_sample_freq": 4,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
        },
        "policy_kwargs": {
            "net_arch": [
                {
                    "pi": [256, 256, 128],
                    "vf": [256, 256, 128],
                }
            ],
            "activation_fn": nn.ReLU,
            "ortho_init": False,
            "log_std_init": -2,
            "full_std": False,
            "use_expln": True,
            "lstm_hidden_size": 256,
            "n_lstm_layers": 1,
            "shared_lstm": False,
            "enable_critic_lstm": True,
        },
    },
    "td3": {
        "timesteps": 200_000,
        "model_kwargs": {
            "batch_size": 100,
            "buffer_size": 1_000_000,
            "learning_rate": 0.001,
        },
    },
    "sac": {
        "timesteps": 200_000,
        "model_kwargs": {
            "batch_size": 64,
            "buffer_size": 100_000,
            "learning_rate": 0.0001,
            "learning_starts": 100,
            "ent_coef": "auto_0.1",
        },
    },
}
