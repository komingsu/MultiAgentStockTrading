#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import sys

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3

import config
from env import StockTradingEnv
from hyperparams import ENV_PARAMS

ALGOS = {
    "ppo": PPO,
    "td3": TD3,
    "sac": SAC,
    "a2c": A2C,
    "ddpg": DDPG,
}


def _load_snapshot(df: pd.DataFrame, date: str | None) -> pd.DataFrame:
    if "date" not in df.columns or "tic" not in df.columns:
        missing = {"date", "tic"} - set(df.columns)
        raise KeyError(f"Required columns missing from data: {missing}")

    df = df.sort_values(["date", "tic"]).reset_index(drop=True)
    if date:
        snap = df[df["date"] == date]
        if snap.empty:
            raise ValueError(f"date {date} not found in data")
    else:
        last = df["date"].iloc[-1]
        snap = df[df["date"] == last]
    if snap.empty:
        raise ValueError("Snapshot dataframe is empty")

    d0 = pd.to_datetime(snap["date"].iloc[0])
    d1 = (d0 + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    snap2 = snap.copy()
    snap2["date"] = d1
    out = pd.concat([snap, snap2], ignore_index=True)
    codes, _ = pd.factorize(out["date"])
    out.index = codes
    return out


def _ensure_iterable(cost: float | Iterable[float], stock_dim: int) -> list[float]:
    if isinstance(cost, (int, float)):
        return [float(cost)] * stock_dim
    cost_list = list(cost)
    if len(cost_list) != stock_dim:
        raise ValueError(
            "Cost list length must match number of tickers. "
            f"Expected {stock_dim}, got {len(cost_list)}"
        )
    return [float(c) for c in cost_list]


def _build_env(
    snapshot_df: pd.DataFrame,
    cash: float,
    holdings: Dict[str, int | float],
    hmax: int,
    buy_cost: float | Iterable[float],
    sell_cost: float | Iterable[float],
    turbulence_threshold: float | None,
) -> Tuple[StockTradingEnv, np.ndarray, list[str]]:
    tickers = sorted(snapshot_df["tic"].unique().tolist())
    stock_dim = len(tickers)
    if stock_dim == 0:
        raise ValueError("No tickers found in snapshot data")

    num_shares = [int(holdings.get(t, 0)) for t in tickers]
    buy_list = _ensure_iterable(buy_cost, stock_dim)
    sell_list = _ensure_iterable(sell_cost, stock_dim)

    env = StockTradingEnv(
        df=snapshot_df,
        stock_dim=stock_dim,
        hmax=hmax,
        initial_amount=float(cash),
        num_stock_shares=num_shares,
        buy_cost_pct=buy_list,
        sell_cost_pct=sell_list,
        reward_scaling=1e-4,
        state_space=1 + 2 * stock_dim + len(config.INDICATORS) * stock_dim,
        action_space=stock_dim,
        tech_indicator_list=config.INDICATORS,
        turbulence_threshold=turbulence_threshold,
        make_plots=False,
        print_verbosity=0,
        initial=True,
        previous_state=[],
        model_name="inference",
        mode="inference",
        iteration=0,
    )
    obs = np.asarray(env.state, dtype=np.float32)
    return env, obs, tickers


def _load_holdings(payload: str) -> Dict[str, int]:
    candidate_path = Path(payload)
    try:
        if candidate_path.exists():
            with candidate_path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        else:
            data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "`--holdings` must be a path to a JSON file or a JSON string"
        ) from exc
    if not isinstance(data, dict):
        raise ValueError("Holdings data must be a JSON object mapping ticker to quantity")
    clean: Dict[str, int] = {}
    for key, value in data.items():
        clean[str(key)] = int(value)
    return clean


def main() -> None:
    parser = argparse.ArgumentParser(description="Infer next action from a trained RL agent")
    parser.add_argument("--experiment", required=True, help="Path to experiments/<name> folder")
    parser.add_argument("--algo", default="ppo", choices=list(ALGOS))
    parser.add_argument("--cash", type=float, required=True, help="Available cash balance")
    parser.add_argument(
        "--holdings",
        required=True,
        help="Path to holdings JSON file or inline JSON (e.g. '{\"005930\": 10}')",
    )
    parser.add_argument("--date", default=None, help="YYYY-MM-DD snapshot date. Defaults to last date")
    parser.add_argument(
        "--data",
        default=Path("data") / "trade_data.csv",
        help="CSV with trade data (default: data/trade_data.csv)",
    )
    parser.add_argument("--hmax", type=int, default=ENV_PARAMS.hmax, help="Max shares per order (same as training)")
    parser.add_argument("--buy-cost", type=float, default=ENV_PARAMS.transaction_cost_pct, help="Buy transaction cost percentage")
    parser.add_argument("--sell-cost", type=float, default=ENV_PARAMS.transaction_cost_pct, help="Sell transaction cost percentage")
    parser.add_argument(
        "--turbulence-threshold",
        type=float,
        default=None,
        help="Same turbulence threshold used during training",
    )
    parser.add_argument("--device", default="cpu", help="Torch device for loading the model")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_csv(data_path, dtype={"tic": str})
    snapshot = _load_snapshot(df, args.date)

    holdings = _load_holdings(args.holdings)

    env, obs, tickers = _build_env(
        snapshot_df=snapshot,
        cash=args.cash,
        holdings=holdings,
        hmax=args.hmax,
        buy_cost=args.buy_cost,
        sell_cost=args.sell_cost,
        turbulence_threshold=args.turbulence_threshold,
    )

    model_path = Path(args.experiment) / "models" / f"agent_{args.algo}.zip"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model_cls = ALGOS[args.algo]
    model = model_cls.load(model_path, device=args.device)

    raw_action, _ = model.predict(obs, deterministic=True)
    planned_shares = (raw_action * args.hmax).astype(int)

    prices_today = snapshot[snapshot["date"] == snapshot["date"].iloc[0]].sort_values("tic")["close"].to_numpy()
    before_qty = np.array([holdings.get(t, 0) for t in tickers], dtype=int)

    next_state, _, _, _, _ = env.step(planned_shares)

    after_qty = np.array(next_state[1 + env.stock_dim : 1 + 2 * env.stock_dim], dtype=int)
    executed = after_qty - before_qty

    prices_next = np.array(next_state[1 : 1 + env.stock_dim], dtype=float)

    report = pd.DataFrame(
        {
            "tic": tickers,
            "price": prices_today,
            "qty_before": before_qty,
            "action_raw": planned_shares.astype(int),
            "action_executed": executed,
            "qty_after": after_qty,
        }
    )

    cash_after = float(next_state[0])
    total_before = float(args.cash + np.dot(before_qty, prices_today))
    total_after = float(cash_after + np.dot(after_qty, prices_next))

    pd.set_option("display.max_rows", None)
    print(report.to_string(index=False))
    print(f"cash_before: {args.cash:.2f}")
    print(f"cash_after : {cash_after:.2f}")
    print(f"total_before: {total_before:.2f}")
    print(f"total_after: {total_after:.2f}")


if __name__ == "__main__":
    main()
