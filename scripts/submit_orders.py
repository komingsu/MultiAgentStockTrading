#!/usr/bin/env python3
from __future__ import annotations

"""
submit_orders.py — AOC 플랜 JSON을 KIS 주문으로 제출(드라이런/모의/실전)

흐름:
  1) Plan JSON 로드([{symbol, desired_qty, close}])
  2) to_orders()로 OrderSpec 생성(MOO=MARKET / LIMIT_OHLC=LIMIT+틱 라운딩)
  3) 거래시간/휴일/플랜 날짜 가드 통과 확인(Fail-Closed)
  4) 프리플라이트(보유/현금 게이트, MARKET 비용은 시세→플랜 close로 추정)
  5) 라우터.place() 호출(드라이런/모의/실전) → 응답/주문 1:1 매핑 저장

주의: 본 패치는 사용법/가드 설명을 주석/도큐스트링으로만 보강한다.
"""

import argparse
import json
import os
import re
from datetime import datetime, time
from pathlib import Path
from typing import Dict, List
from zoneinfo import ZoneInfo

import numpy as np

import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env import ExecutionMode
from execution.order_spec import OrderSpec
from live.kis_router import LiveBroker
from live.route import to_orders


class MockRouter:
    def __init__(self) -> None:
        self.sent: List[OrderSpec] = []

    def place(self, order: OrderSpec) -> Dict[str, object]:
        self.sent.append(order)
        return {
            "order_id": order.client_order_id,
            "accepted": True,
            "message": "mock",
        }

    def positions(self) -> Dict[str, object]:
        return {}

    def cash_available(self) -> float:
        return 0.0


def _load_plan(path: Path) -> List[dict]:
    """플랜 JSON 로드 및 간단 검증."""
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError("Plan JSON must be a list of {symbol, desired_qty, close}")
    return data


def _fetch_prices(router: LiveBroker, orders: List[OrderSpec]) -> Dict[str, float]:
    """시장가 주문의 비용 추정에 쓸 현재가 조회(가능한 경우)."""
    prices: Dict[str, float] = {}
    symbols = {order.symbol for order in orders if order.order_type == "MARKET"}
    for symbol in symbols:
        quote = router.quote_price(symbol) if hasattr(router, "quote_price") else None
        if quote is not None and quote > 0:
            prices[symbol] = quote
    return prices


def _preflight(orders: List[OrderSpec], router: LiveBroker, reference_prices: Dict[str, float], plan_prices: Dict[str, float]) -> List[str]:
    """사전 점검(프리플라이트) — Fail-Closed 원칙.

    체크 항목(오직 3개):
      1) 기본 형식: LIMIT이면 price>0, client_order_id 중복 없음
      2) SELL ≤ 보유(positions())
      3) BUY 총액 ≤ 가용 현금 × (1−1e−3). MARKET은 현재가(가능 시) 또는 plan close로 보수 추정
    """
    issues: List[str] = []
    seen_ids = set()
    for order in orders:
        if order.client_order_id and order.client_order_id in seen_ids:
            issues.append(f"duplicate client_order_id {order.client_order_id}")
        seen_ids.add(order.client_order_id)
        if order.order_type == "LIMIT" and (order.limit_price is None or order.limit_price <= 0):
            issues.append(f"invalid limit price for {order.symbol}")
    if issues:
        return issues

    positions = router.positions()
    cash_available = router.cash_available()

    sell_requirements: Dict[str, int] = {}
    buy_cost = 0.0
    for order in orders:
        if order.side == "SELL":
            sell_requirements[order.symbol] = sell_requirements.get(order.symbol, 0) + order.qty
        else:
            if order.order_type == "LIMIT" and order.limit_price is not None:
                price = float(order.limit_price)
            else:
                price = reference_prices.get(order.symbol, plan_prices.get(order.symbol, 0.0))
            buy_cost += price * order.qty

    for symbol, req_qty in sell_requirements.items():
        held = int(positions.get(symbol, 0))
        if req_qty > held:
            issues.append(f"sell_qty_exceeds_position:{symbol}:{req_qty}>{held}")

    if buy_cost > cash_available * (1.0 - 1e-3):
        issues.append(f"insufficient_cash:{buy_cost:.2f}>{cash_available:.2f}")

    return issues


def _parse_plan_date(path: Path) -> datetime.date:
    """파일명에서 YYYYMMDD 추출. 없으면 오늘(KST)."""
    name = path.name
    match = re.search(r"(20\d{2})(\d{2})(\d{2})", name)
    if match:
        year, month, day = map(int, match.groups())
        return datetime(year, month, day, tzinfo=ZoneInfo("Asia/Seoul")).date()
    return datetime.now(ZoneInfo("Asia/Seoul")).date()


def _load_holidays(path: Path) -> set[datetime.date]:
    """휴장일 JSON 로드. 없거나 실패 시 빈 집합."""
    if not path or not path.exists():
        return set()
    try:
        with path.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
        holidays: set[datetime.date] = set()
        for item in raw:
            dt = datetime.fromisoformat(str(item))
            holidays.add(dt.date())
        return holidays
    except Exception:
        return set()


def _ensure_trading_window(exec_mode: ExecutionMode, plan_date: datetime.date, holidays: set[datetime.date], force: bool = False) -> None:
    """KST 시간창/휴일/플랜 날짜 가드.

    - plan_date != 오늘(KST) → 차단 (테스트는 --force-submit)
    - 주말/휴일 차단
    - 시간창: MOO 08:30–09:00, LIMIT 09:00–15:20 (KST)
    """
    if force:
        return
    tz = ZoneInfo("Asia/Seoul")
    now = datetime.now(tz)
    if now.date() != plan_date:
        raise RuntimeError(f"Plan date {plan_date} does not match today {now.date()} (KST)")
    if now.weekday() >= 5:
        raise RuntimeError("Trading window closed on weekends")
    if plan_date in holidays:
        raise RuntimeError(f"Trading window closed (holiday {plan_date})")
    current_time = now.time()
    if exec_mode is ExecutionMode.MOO:
        if not (time(8, 30) <= current_time <= time(9, 0)):
            raise RuntimeError("MOO orders permitted only 08:30-09:00 KST")
    else:
        if not (time(9, 0) <= current_time <= time(15, 20)):
            raise RuntimeError("LIMIT orders permitted only 09:00-15:20 KST")


def _build_live_router(args: argparse.Namespace) -> LiveBroker:
    """CLI 인자/환경변수에서 KIS 라우터 구성."""
    app_key = args.app_key or os.getenv("KIS_APP_KEY") or os.getenv("KIS_API_KEY")
    app_secret = args.app_secret or os.getenv("KIS_APP_SECRET")
    account_no = args.account_no or os.getenv("KIS_ACCOUNT_NO", "")
    product_code = args.product_code or os.getenv("KIS_PRODUCT_CODE", "01")
    if not app_key or not app_secret:
        raise RuntimeError("KIS app key/secret required")
    base_url = args.base_url or (
        "https://openapivts.koreainvestment.com:29443"
        if args.use_paper
        else "https://openapi.koreainvestment.com:9443"
    )
    return LiveBroker(
        base_url=base_url,
        app_key=app_key,
        app_secret=app_secret,
        account_no=account_no,
        product_code=product_code,
        use_paper=args.use_paper,
        dry_run=args.dry_run,
    )


def main() -> None:
    """플랜→주문 변환/가드/프리플라이트/제출을 실행하고 로그를 저장."""
    parser = argparse.ArgumentParser(description="Submit orders generated by AOC")
    parser.add_argument("--plan-json", required=True, help="Path to JSON plan (list of orders)")
    parser.add_argument("--exec-mode", choices=[mode.value for mode in ExecutionMode], default=ExecutionMode.MOO.value)
    parser.add_argument("--limit-offset-bps", type=float, default=0.0)
    parser.add_argument("--dry-run", action="store_true", help="Do not call KIS, only log")
    parser.add_argument("--use-paper", action="store_true", help="Use paper trading endpoints")
    parser.add_argument("--base-url", default=None, help="Override KIS base URL")
    parser.add_argument("--app-key", default=None)
    parser.add_argument("--app-secret", default=None)
    parser.add_argument("--account-no", default=None)
    parser.add_argument("--product-code", default=None)
    parser.add_argument("--output", default=Path("logs") / "submit_orders.json", help="Where to persist responses")
    parser.add_argument("--holiday-file", default=Path("config") / "holiday_kr.json", help="JSON list of YYYY-MM-DD holidays")
    parser.add_argument("--force-submit", action="store_true", help="Skip trading window checks")
    args = parser.parse_args()

    plan_path = Path(args.plan_json)
    plan_rows = _load_plan(plan_path)
    tickers = [row["symbol"] for row in plan_rows]
    desired = np.array([int(row["desired_qty"]) for row in plan_rows], dtype=int)
    closes = np.array([float(row.get("close", 0.0)) for row in plan_rows], dtype=float)
    plan_prices = {row["symbol"]: float(row.get("close", 0.0)) for row in plan_rows}

    exec_mode = ExecutionMode(args.exec_mode)
    orders = to_orders(exec_mode, desired, tickers, closes, args.limit_offset_bps)
    router = MockRouter() if args.dry_run else _build_live_router(args)

    holiday_path = Path(args.holiday_file) if args.holiday_file else None
    holidays = _load_holidays(holiday_path) if holiday_path else set()
    plan_date = _parse_plan_date(plan_path)
    _ensure_trading_window(exec_mode, plan_date, holidays, args.force_submit)

    ref_prices = _fetch_prices(router, orders)
    issues = _preflight(orders, router, ref_prices, plan_prices)
    if issues:
        raise RuntimeError(f"Preflight check failed: {issues}")

    responses: List[dict] = []
    for order in orders:
        responses.append(router.place(order))

    order_records = [
        {
            "symbol": order.symbol,
            "side": order.side,
            "qty": order.qty,
            "order_type": order.order_type,
            "limit_price": None if order.limit_price is None else int(order.limit_price),
            "client_order_id": order.client_order_id,
        }
        for order in orders
    ]

    response_records = []
    for order, response in zip(orders, responses):
        record = dict(response)
        record.setdefault("order_id", order.client_order_id)
        record["symbol"] = order.symbol
        record["qty"] = order.qty
        record["side"] = order.side
        response_records.append(record)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "plan_path": str(plan_path),
        "plan_date": plan_date.isoformat(),
        "orders": order_records,
        "responses": response_records,
    }
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    print(f"Submitted {len(orders)} orders; responses written to {output_path}")


if __name__ == "__main__":
    main()
