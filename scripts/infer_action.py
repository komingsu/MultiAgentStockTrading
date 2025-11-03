#!/usr/bin/env python3
from __future__ import annotations

"""
infer_action.py — 학습된 에이전트로부터 내일 주문 계획을 생성(AOC 포함)

흐름:
  1) 데이터 스냅샷(오늘) 구성 → 환경(env) 생성 (inference 모드)
  2) 모델 로드 후 행동 예측 → AOC로 실행가능 수량(desired_qty) 산출
  3) AOC 리포트 JSON 저장 + 제출용 Plan JSON 저장([{symbol,desired_qty,close}])

중요 CLI 옵션(요약):
  --exec-mode: MOO / LIMIT_OHLC (체결 규칙)
  --adv-frac: ADV 캡 비율 (0.1=10%)
  --limit-offset-bps: LIMIT 지정가 오프셋(bps)
  --cash-buffer-pct: 계획 후 유지할 현금 비율
  --write-plan-json: 제출 스크립트가 소비할 플랜 JSON 경로

주의: 코드 동작은 변경하지 않고, 주석/도큐스트링만 추가했다.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
from datetime import datetime

import sys

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3

import config
from env import ExecutionMode, StockTradingEnv
from execution.order_convert import (
    AdvGuard,
    ExecKnobs,
    Limits,
    PortfolioSnapshot,
    PriceContext,
    plan_from_weights,
)
from hyperparams import ENV_PARAMS

ALGOS = {
    "ppo": PPO,
    "td3": TD3,
    "sac": SAC,
    "a2c": A2C,
    "ddpg": DDPG,
}


def _load_snapshot(df: pd.DataFrame, date: str | None) -> pd.DataFrame:
    """주어진 데이터프레임에서 오늘/내일 2일 스냅샷을 생성.

    date가 지정되지 않으면 마지막 거래일을 오늘로 간주한다.
    반환 데이터프레임은 index=0(오늘),1(내일)로 factorize된다.
    """
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
    """단일 값 또는 리스트를 종목 수 길이의 리스트로 보정."""
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
    exec_mode: str = ENV_PARAMS.exec_mode,
    adv_fraction: float = ENV_PARAMS.adv_fraction,
    limit_offset_bps: float = ENV_PARAMS.limit_offset_bps,
    slippage_bps: float = ENV_PARAMS.slippage_bps,
    day_only: bool = ENV_PARAMS.day_order_only,
) -> Tuple[StockTradingEnv, np.ndarray, list[str]]:
    """인퍼런스 전용 환경 생성.

    - env.mode="inference"로 추적/플롯 비활성
    - hmax=1로 강제(행동 스케일 의존성 제거)
    """
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
        exec_mode=exec_mode,
        adv_fraction=adv_fraction,
        limit_offset_bps=limit_offset_bps,
        slippage_bps=slippage_bps,
        day_order_only=day_only,
    )
    obs = np.asarray(env.state, dtype=np.float32)
    env.hmax = 1
    obs = np.asarray(env.state, dtype=np.float32)
    return env, obs, tickers


def _compute_adv_reference(
    full_df: pd.DataFrame,
    tickers: List[str],
    date: str,
    mode: str,
) -> np.ndarray:
    """ADV 참조 시계열(t-1 또는 SMA5) 계산."""
    out: List[float] = []
    grouped = full_df[full_df["tic"].isin(tickers)].sort_values(["tic", "date"])
    for tic in tickers:
        history = grouped[grouped["tic"] == tic]
        if mode == "prev_volume":
            prev = history[history["date"] < date].tail(1)
            if prev.empty:
                prev = history[history["date"] == date].tail(1)
            value = float(prev["volume"].iloc[-1]) if not prev.empty else 0.0
        else:  # sma5_volume
            window = history[history["date"] <= date].tail(5)
            if window.empty:
                window = history[history["date"] == date]
            value = float(window["volume"].mean()) if not window.empty else 0.0
        out.append(value)
    return np.asarray(out, dtype=float)


def _latest_day_frame(snapshot_df: pd.DataFrame) -> Tuple[str, pd.DataFrame]:
    """스냅샷에서 오늘 날짜와 해당 일자 프레임을 반환."""
    dates = snapshot_df["date"].unique()
    if len(dates) == 0:
        raise ValueError("Snapshot dataframe is empty")
    current_date = dates[0]
    day_frame = snapshot_df[snapshot_df["date"] == current_date].sort_values("tic")
    return current_date, day_frame


def _json_default(obj):
    """넘파이/세트 타입 JSON 직렬화 헬퍼."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, set):
        return list(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _load_holdings(payload: str) -> Dict[str, int]:
    """보유수량 JSON 로드. 파일 경로나 인라인 JSON 모두 허용."""
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
    """에이전트 추론→AOC 계획→Plan JSON/리포트 저장까지 한 번에 수행."""
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
    parser.add_argument("--buy-cost", type=float, default=ENV_PARAMS.buy_cost_pct, help="Buy transaction cost percentage")
    parser.add_argument("--sell-cost", type=float, default=ENV_PARAMS.sell_cost_pct, help="Sell transaction cost percentage")
    parser.add_argument(
        "--turbulence-threshold",
        type=float,
        default=None,
        help="Same turbulence threshold used during training",
    )
    parser.add_argument(
        "--action-schema",
        default="WEIGHT",
        choices=["WEIGHT", "DELTA"],
        help="Interpret agent output as target weights or share deltas",
    )
    parser.add_argument("--w-max", type=float, default=0.10, help="Per-symbol weight cap")
    parser.add_argument("--turnover-max", type=float, default=0.20, help="Daily turnover cap (fraction of NAV)")
    parser.add_argument("--min-notional", type=float, default=50_000.0, help="Minimum order notional")
    parser.add_argument("--epsilon-w", type=float, default=0.005, help="No-trade band width")
    parser.add_argument(
        "--cash-buffer-pct",
        type=float,
        default=0.001,
        help="Fraction of NAV to retain as cash after planning",
    )
    parser.add_argument(
        "--max-orders",
        type=int,
        default=20,
        help="Maximum number of symbols to trade per day (0 disables cap)",
    )
    parser.add_argument(
        "--adv-ref",
        default="prev_volume",
        choices=["prev_volume", "sma5_volume"],
        help="ADV reference series to use during planning",
    )
    parser.add_argument(
        "--exec-mode",
        default=ENV_PARAMS.exec_mode,
        choices=[mode.value for mode in ExecutionMode],
        help="Execution mode for next-day fills",
    )
    parser.add_argument(
        "--adv-frac",
        type=float,
        default=ENV_PARAMS.adv_fraction,
        help="Cap fills to fraction of next-day ADV",
    )
    parser.add_argument(
        "--limit-offset-bps",
        type=float,
        default=ENV_PARAMS.limit_offset_bps,
        help="Limit price offset (bps) for LIMIT_OHLC mode",
    )
    parser.add_argument(
        "--slippage-bps",
        type=float,
        default=ENV_PARAMS.slippage_bps,
        help="Additional slippage in basis points applied to executions",
    )
    parser.add_argument("--day-only", dest="day_only", action="store_true", help="Submit DAY orders (default)")
    parser.add_argument(
        "--allow-gtc",
        dest="day_only",
        action="store_false",
        help="Allow orders to persist beyond one day in simulation",
    )
    parser.set_defaults(day_only=ENV_PARAMS.day_order_only)
    parser.add_argument(
        "--report-dir",
        default=Path("logs") / "aoc_reports",
        help="Directory to store planning and execution reports",
    )
    parser.add_argument(
        "--write-plan-json",
        default=None,
        help="Optional path to persist plan JSON; default logs/plans/plan_{date}.json",
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
        exec_mode=args.exec_mode,
        adv_fraction=args.adv_frac,
        limit_offset_bps=args.limit_offset_bps,
        slippage_bps=args.slippage_bps,
        day_only=args.day_only,
    )

    model_path = Path(args.experiment) / "models" / f"agent_{args.algo}.zip"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model_cls = ALGOS[args.algo]
    model = model_cls.load(model_path, device=args.device)

    raw_action, _ = model.predict(obs, deterministic=True)
    raw_action = np.asarray(raw_action, dtype=float).reshape(-1)
    raw_action = np.nan_to_num(raw_action, nan=0.0, posinf=0.0, neginf=0.0)
    if raw_action.shape[0] != env.stock_dim:
        raise ValueError(
            f"Model output dimension {raw_action.shape[0]} does not match stock_dim {env.stock_dim}"
        )

    today_date, today_frame = _latest_day_frame(snapshot)
    prices_today = today_frame.sort_values("tic")["close"].to_numpy(dtype=float)
    before_qty = np.array([holdings.get(t, 0) for t in tickers], dtype=int)
    adv_reference = _compute_adv_reference(df, tickers, today_date, args.adv_ref)

    total_value = float(args.cash + np.dot(before_qty, prices_today))
    limits = Limits(
        w_max=args.w_max,
        turnover_max=args.turnover_max,
        min_notional=args.min_notional,
        epsilon_w=args.epsilon_w,
        lot_size=1,
        max_orders=None if args.max_orders and args.max_orders <= 0 else args.max_orders,
    )
    exec_spec = env.execution_spec
    knobs = ExecKnobs(
        fee_buy=exec_spec.fee_buy,
        fee_sell=exec_spec.fee_sell,
        slippage_bps=exec_spec.slippage_bps,
        cash_buffer_pct=args.cash_buffer_pct,
        limit_offset_bps=exec_spec.limit_offset_bps,
    )
    adv_guard = AdvGuard(
        adv_cap_frac=float(args.adv_frac),
        adv_reference=adv_reference,
    )
    snap_struct = PortfolioSnapshot(cash=args.cash, holdings=before_qty.astype(float))
    price_ctx = PriceContext(close=prices_today)

    if args.action_schema == "WEIGHT":
        target_w = np.clip(raw_action, -1.0, 1.0)
        desired_actions, planning_report = plan_from_weights(
            target_w=target_w,
            snap=snap_struct,
            prices=price_ctx,
            limits=limits,
            adv=adv_guard,
            knobs=knobs,
            short_allowed=False,
        )
        planning_report["schema"] = "WEIGHT"
        planning_report["target_w"] = target_w.tolist()
    else:  # DELTA
        raw_delta = np.rint(raw_action * args.hmax).astype(int)
        raw_delta = np.nan_to_num(raw_delta, nan=0.0).astype(int)
        target_holdings = np.maximum(before_qty + raw_delta, 0)
        if total_value > 0:
            implied_weights = (target_holdings * prices_today) / total_value
        else:
            implied_weights = np.zeros_like(prices_today)
        desired_actions, planning_report = plan_from_weights(
            target_w=implied_weights,
            snap=snap_struct,
            prices=price_ctx,
            limits=limits,
            adv=adv_guard,
            knobs=knobs,
            short_allowed=False,
        )
        planning_report["schema"] = "DELTA"
        planning_report["input_delta"] = raw_delta.tolist()

    planning_report.setdefault("notes", [])
    planning_report["shapes"] = {
        "raw_action": list(raw_action.shape),
        "desired": list(desired_actions.shape),
    }
    planning_report["exec"] = {
        "mode": env.execution_spec.mode.value,
        "slippage_bps": env.execution_spec.slippage_bps,
        "limit_offset_bps": env.execution_spec.limit_offset_bps,
        "adv_fraction_arg": float(args.adv_frac),
    }
    planning_report["adv_reference"] = adv_reference.tolist()

    actions = desired_actions.astype(int)
    if actions.shape != (env.stock_dim,):
        raise ValueError(f"AOC output shape {actions.shape} does not match stock_dim {env.stock_dim}")
    next_state, _, done, trunc, info = env.step(actions)
    info.setdefault("aoc", planning_report)

    after_qty = np.array(next_state[1 + env.stock_dim : 1 + 2 * env.stock_dim], dtype=int)
    executed = after_qty - before_qty
    prices_next = np.array(next_state[1 : 1 + env.stock_dim], dtype=float)

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    fills = info.get("fills", [])
    env_info = {k: v for k, v in info.items() if k != "aoc"}
    payload = {
        "timestamp": datetime.now().isoformat(),
        "date": today_date,
        "schema": args.action_schema,
        "planning": planning_report,
        "fills": fills,
        "raw_action": raw_action.tolist(),
        "actions": actions.tolist(),
        "executed": executed.tolist(),
        "env_info": env_info,
    }
    report_path = report_dir / f"aoc_{today_date}_{args.action_schema.lower()}.json"
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, default=_json_default, ensure_ascii=False, indent=2)

    plan_output = args.write_plan_json
    if plan_output is None:
        plan_dir = Path("logs") / "plans"
        plan_dir.mkdir(parents=True, exist_ok=True)
        plan_output = plan_dir / f"plan_{today_date}.json"
    plan_path = Path(plan_output)
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_rows = [
        {"symbol": tickers[i], "desired_qty": int(actions[i]), "close": float(prices_today[i])}
        for i in range(len(tickers))
    ]
    with plan_path.open("w", encoding="utf-8") as fh:
        json.dump(plan_rows, fh, default=_json_default, ensure_ascii=False, indent=2)

    report = pd.DataFrame(
        {
            "tic": tickers,
            "price": prices_today,
            "qty_before": before_qty,
            "action_model": raw_action,
            "action_planned": actions,
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
