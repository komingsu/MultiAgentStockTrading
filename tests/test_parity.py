"""SimBroker와 env.step() 체결 결과가 동등(Parity)함을 검증하는 테스트."""
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
from execution.sim_broker import SimBroker
from live.route import to_orders


def _make_dataframe(day0: dict, day1: dict) -> pd.DataFrame:
    rows = []
    for idx, (date, day) in enumerate([
        ("2023-01-02", day0),
        ("2023-01-03", day1),
    ]):
        row = {
            "date": date,
            "tic": "TEST",
            "open": day["open"],
            "high": day["high"],
            "low": day["low"],
            "close": day["close"],
            "volume": day["volume"],
            "turbulence": 0.0,
            "day": idx,
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    return df


def _make_env(exec_mode: ExecutionMode, day0: dict, day1: dict, *, adv_frac: float = 1.0, limit_offset_bps: float = 0.0) -> StockTradingEnv:
    df = _make_dataframe(day0, day1)
    env = StockTradingEnv(
        df=df,
        stock_dim=1,
        hmax=1,
        initial_amount=100_000.0,
        num_stock_shares=[0],
        buy_cost_pct=[0.0],
        sell_cost_pct=[0.0],
        reward_scaling=1.0,
        state_space=3,
        action_space=1,
        tech_indicator_list=[],
        exec_mode=exec_mode.value,
        adv_fraction=adv_frac,
        limit_offset_bps=limit_offset_bps,
        slippage_bps=0.0,
        day_order_only=True,
        make_plots=False,
        print_verbosity=0,
    )
    env.hmax = 1
    return env


def _snapshot(env: StockTradingEnv) -> dict:
    holdings = np.array(env.state[(env.stock_dim + 1) : (env.stock_dim * 2 + 1)], dtype=float)
    return {
        "cash": float(env.state[0]),
        "holdings": holdings,
        "tickers": env.ticker_list,
        "today_frame": env.data.copy(),
        "next_frame": env._get_day_frame(env.day + 1),
    }


def _compare_fill(sim_report: dict, env_fill: dict) -> None:
    assert sim_report["filled"] == env_fill["filled_qty"]
    if env_fill["fill_price"] is not None:
        assert pytest.approx(sim_report["price"], rel=1e-6) == env_fill["fill_price"]
    else:
        assert sim_report["price"] is None
    assert sim_report["reason"] == env_fill["reason"]


def test_parity_moo_open_fill_exact():
    day0 = {"open": 100.0, "high": 102.0, "low": 98.0, "close": 100.0, "volume": 1000.0}
    day1 = {"open": 110.0, "high": 115.0, "low": 108.0, "close": 112.0, "volume": 1000.0}
    env = _make_env(ExecutionMode.MOO, day0, day1)
    desired = np.array([5])
    orders = to_orders(env.execution_spec.mode, desired, env.ticker_list, np.array([day0["close"]]), env.execution_spec.limit_offset_bps)
    broker = SimBroker(env.execution_spec)
    reports = broker.simulate(0, orders, _snapshot(env))

    state, _, _, _, info = env.step(desired)
    env_fill = info["fills"][0]

    _compare_fill(reports[0], env_fill)


def test_parity_limit_touch_intraday():
    day0 = {"open": 100.0, "high": 102.0, "low": 98.0, "close": 100.0, "volume": 1000.0}
    day1 = {"open": 105.0, "high": 110.0, "low": 99.0, "close": 106.0, "volume": 1000.0}
    env = _make_env(ExecutionMode.LIMIT_OHLC, day0, day1, limit_offset_bps=0.0)
    desired = np.array([3])
    orders = to_orders(env.execution_spec.mode, desired, env.ticker_list, np.array([day0["close"]]), env.execution_spec.limit_offset_bps)
    reports = SimBroker(env.execution_spec).simulate(0, orders, _snapshot(env))
    _, _, _, _, info = env.step(desired)
    _compare_fill(reports[0], info["fills"][0])


def test_parity_limit_not_reached():
    day0 = {"open": 100.0, "high": 102.0, "low": 98.0, "close": 100.0, "volume": 1000.0}
    day1 = {"open": 104.0, "high": 105.0, "low": 101.0, "close": 104.0, "volume": 1000.0}
    env = _make_env(ExecutionMode.LIMIT_OHLC, day0, day1, limit_offset_bps=0.0)
    desired = np.array([4])
    orders = to_orders(env.execution_spec.mode, desired, env.ticker_list, np.array([day0["close"]]), env.execution_spec.limit_offset_bps)
    reports = SimBroker(env.execution_spec).simulate(0, orders, _snapshot(env))
    _, _, _, _, info = env.step(desired)
    _compare_fill(reports[0], info["fills"][0])


def test_parity_adv_cap_respected():
    day0 = {"open": 100.0, "high": 102.0, "low": 98.0, "close": 100.0, "volume": 1_000.0}
    day1 = {"open": 110.0, "high": 112.0, "low": 108.0, "close": 111.0, "volume": 20.0}
    env = _make_env(ExecutionMode.MOO, day0, day1, adv_frac=0.1)
    desired = np.array([10])
    orders = to_orders(env.execution_spec.mode, desired, env.ticker_list, np.array([day0["close"]]), env.execution_spec.limit_offset_bps)
    reports = SimBroker(env.execution_spec).simulate(0, orders, _snapshot(env))
    _, _, _, _, info = env.step(desired)
    _compare_fill(reports[0], info["fills"][0])


def test_parity_cash_holdings_clipping():
    day0 = {"open": 100.0, "high": 102.0, "low": 98.0, "close": 100.0, "volume": 500.0}
    day1 = {"open": 95.0, "high": 100.0, "low": 90.0, "close": 92.0, "volume": 500.0}
    env = _make_env(ExecutionMode.MOO, day0, day1, adv_frac=1.0)
    # Deplete cash by setting initial cash small
    env.state[0] = 100.0
    desired = np.array([5])
    orders = to_orders(env.execution_spec.mode, desired, env.ticker_list, np.array([day0["close"]]), env.execution_spec.limit_offset_bps)
    reports = SimBroker(env.execution_spec).simulate(0, orders, _snapshot(env))
    _, _, _, _, info = env.step(desired)
    _compare_fill(reports[0], info["fills"][0])
