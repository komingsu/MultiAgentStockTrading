from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


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
            "n_steps": 2048,
            "ent_coef": 0.01,
            "learning_rate": 0.00025,
            "batch_size": 64,
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
