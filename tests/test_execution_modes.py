"""env의 MOO/LIMIT_OHLC 규칙이 기대대로 동작하는지 검증하는 테스트."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env import ExecutionMode, StockTradingEnv


def _make_env(
    day0: dict,
    day1: dict,
    *,
    exec_mode: str,
    adv_fraction: float = 1.0,
    limit_offset_bps: float = 0.0,
    slippage_bps: float = 0.0,
    buy_cost: float = 0.0,
    sell_cost: float = 0.0,
    initial_cash: float = 1_000.0,
    hmax: int = 10,
) -> StockTradingEnv:
    df = pd.DataFrame([day0, day1])
    df = df.sort_values(["date", "tic"]).reset_index(drop=True)
    df.index = df["date"].factorize()[0]

    env = StockTradingEnv(
        df=df,
        stock_dim=1,
        hmax=hmax,
        initial_amount=initial_cash,
        num_stock_shares=[0],
        buy_cost_pct=[buy_cost],
        sell_cost_pct=[sell_cost],
        reward_scaling=1.0,
        state_space=1 + 2 * 1,
        action_space=1,
        tech_indicator_list=[],
        print_verbosity=None,
        exec_mode=exec_mode,
        adv_fraction=adv_fraction,
        limit_offset_bps=limit_offset_bps,
        slippage_bps=slippage_bps,
        turbulence_threshold=None,
    )
    env.reset(seed=42)
    return env


def test_moo_executes_at_open_price():
    day0 = {
        "date": "2023-01-02",
        "tic": "TEST",
        "open": 100.0,
        "high": 102.0,
        "low": 98.0,
        "close": 100.0,
        "volume": 10_000,
    }
    day1 = {
        "date": "2023-01-03",
        "tic": "TEST",
        "open": 110.0,
        "high": 115.0,
        "low": 108.0,
        "close": 120.0,
        "volume": 10_000,
    }
    env = _make_env(day0, day1, exec_mode=ExecutionMode.MOO.value)

    state, reward, done, truncated, info = env.step(np.array([0.5]))

    assert not done
    assert not truncated
    assert info["fills"][0]["filled_qty"] == 5
    assert info["fills"][0]["fill_price"] == pytest.approx(110.0)
    assert state[0] == pytest.approx(1_000 - 5 * 110.0)
    assert state[2] == 5
    expected_reward = (state[0] + state[1] * state[2]) - 1_000.0
    assert reward == pytest.approx(expected_reward)


def test_limit_order_fills_at_limit_when_touching_intraday():
    day0 = {
        "date": "2023-01-02",
        "tic": "TEST",
        "open": 100.0,
        "high": 102.0,
        "low": 98.0,
        "close": 100.0,
        "volume": 10_000,
    }
    day1 = {
        "date": "2023-01-03",
        "tic": "TEST",
        "open": 105.0,
        "high": 108.0,
        "low": 99.0,
        "close": 106.0,
        "volume": 10_000,
    }
    env = _make_env(
        day0,
        day1,
        exec_mode=ExecutionMode.LIMIT_OHLC.value,
        limit_offset_bps=0.0,
    )

    state, reward, _, _, info = env.step(np.array([0.5]))
    fill = info["fills"][0]
    assert fill["filled_qty"] == 5
    assert fill["fill_price"] == pytest.approx(100.0)
    assert fill["limit_price"] == pytest.approx(100.0)
    assert state[0] == pytest.approx(1_000 - 5 * 100.0)
    assert state[2] == 5


def test_limit_order_expires_when_price_not_reached():
    day0 = {
        "date": "2023-01-02",
        "tic": "TEST",
        "open": 100.0,
        "high": 102.0,
        "low": 98.0,
        "close": 100.0,
        "volume": 10_000,
    }
    day1 = {
        "date": "2023-01-03",
        "tic": "TEST",
        "open": 105.0,
        "high": 104.0,
        "low": 101.0,
        "close": 103.0,
        "volume": 10_000,
    }
    env = _make_env(
        day0,
        day1,
        exec_mode=ExecutionMode.LIMIT_OHLC.value,
        limit_offset_bps=0.0,
    )

    state, reward, _, _, info = env.step(np.array([0.5]))
    fill = info["fills"][0]
    assert fill["filled_qty"] == 0
    assert fill["reason"] == "limit_not_reached"
    assert state[0] == pytest.approx(1_000.0)
    assert state[2] == 0
    assert reward == pytest.approx((state[0] + state[1] * state[2]) - 1_000.0)


def test_adv_fraction_caps_order_size():
    day0 = {
        "date": "2023-01-02",
        "tic": "TEST",
        "open": 50.0,
        "high": 51.0,
        "low": 49.0,
        "close": 50.0,
        "volume": 10_000,
    }
    day1 = {
        "date": "2023-01-03",
        "tic": "TEST",
        "open": 60.0,
        "high": 61.0,
        "low": 59.0,
        "close": 62.0,
        "volume": 25,
    }
    env = _make_env(
        day0,
        day1,
        exec_mode=ExecutionMode.MOO.value,
        adv_fraction=0.1,
    )

    state, _, _, _, info = env.step(np.array([0.8]))
    fill = info["fills"][0]
    assert fill["filled_qty"] == 2  # floor(0.1 * 25)
    assert state[2] == 2


def test_transaction_costs_and_slippage_reduce_cash():
    day0 = {
        "date": "2023-01-02",
        "tic": "TEST",
        "open": 100.0,
        "high": 101.0,
        "low": 99.0,
        "close": 100.0,
        "volume": 10_000,
    }
    day1 = {
        "date": "2023-01-03",
        "tic": "TEST",
        "open": 110.0,
        "high": 111.0,
        "low": 109.0,
        "close": 112.0,
        "volume": 10_000,
    }
    env = _make_env(
        day0,
        day1,
        exec_mode=ExecutionMode.MOO.value,
        buy_cost=0.002,
        sell_cost=0.001,
        slippage_bps=10.0,
    )

    state, _, _, _, info = env.step(np.array([0.5]))
    fill = info["fills"][0]
    assert fill["filled_qty"] == 5
    total_multiplier = 1.0 + 0.002 + 0.001
    expected_cash = 1_000.0 - 5 * 110.0 * total_multiplier
    assert state[0] == pytest.approx(expected_cash)
    # Total trading costs should be recorded (fee + slippage component)
    expected_cost = 5 * 110.0 * (0.002 + 0.001)
    assert env.cost == pytest.approx(expected_cost)
