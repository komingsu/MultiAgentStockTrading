from __future__ import annotations

"""주문 라우팅/변환 유틸리티

to_orders(): AOC가 산출한 종목별 수량을 OrderSpec 리스트로 변환한다.

규칙:
- MOO: 시장가 주문 생성(MARKET)
- LIMIT_OHLC: 전일 종가 ± offset(bps)로 리미트 산출 후, KRX 호가가격단위(틱) 규칙으로
  정수 호가로 라운딩(매수=내림, 매도=올림)하여 LIMIT 주문 생성

KRX 틱 표(요약):
  <2000:1, 2000~4999:5, 5000~9999:10, 10000~19999:10, 20000~49999:50,
  50000~99999:100, 100000~199999:100, 200000~499999:500, ≥500000:1000 (원)
"""

from datetime import datetime
from typing import List

import numpy as np

from env import ExecutionMode
from execution.order_spec import OrderSpec


def _today_str() -> str:
    return datetime.now().strftime("%Y%m%d")


def _krx_tick(price: float) -> int:
    """가격대별 호가가격단위(틱) 크기를 반환.

    한국 주식은 가격대에 따라 고정된 틱 사이즈를 따른다. 지정가 위반 시
    주문이 거부되므로, LIMIT 주문에서는 반드시 이 규칙을 적용해야 한다.
    """
    p = int(price)
    if p < 2000:
        return 1
    if p < 5000:
        return 5
    if p < 10000:
        return 10
    if p < 20000:
        return 10
    if p < 50000:
        return 50
    if p < 100000:
        return 100
    if p < 200000:
        return 100
    if p < 500000:
        return 500
    return 1000


def _round_price(price: float, side: str) -> int:
    """KRX 틱 규칙에 따라 정수 호가로 라운딩.

    매수는 불리 방향(내림), 매도는 유리 방향(올림)으로 처리한다.
    """
    tick = _krx_tick(price)
    if side.upper() == "BUY":
        return int(np.floor(price / tick) * tick)
    return int(np.ceil(price / tick) * tick)


def to_orders(
    exec_mode: ExecutionMode,
    desired_qtys: np.ndarray,
    tickers: List[str],
    ref_close: np.ndarray,
    limit_offset_bps: float,
) -> List[OrderSpec]:
    """수량 벡터를 OrderSpec 리스트로 변환.

    Args:
      exec_mode: 실행 모드(MOO/LIMIT_OHLC)
      desired_qtys: 종목별 정수 수량 벡터(부호 포함)
      tickers: 종목 코드 리스트
      ref_close: 전일 종가(리미트 산출 기준)
      limit_offset_bps: 지정가 오프셋(bps)

    Returns:
      OrderSpec 리스트. LIMIT은 정수 호가로 보정되어 반환.
    """
    desired = np.asarray(desired_qtys, dtype=int)
    closes = np.asarray(ref_close, dtype=float)
    if desired.shape[0] != len(tickers):
        raise ValueError("desired_qtys length must match tickers")

    orders: List[OrderSpec] = []
    offset_ratio = float(limit_offset_bps) / 10_000.0
    today = _today_str()

    for idx, qty in enumerate(desired):
        if qty == 0:
            continue
        symbol = tickers[idx]
        side = "BUY" if qty > 0 else "SELL"
        quantity = abs(int(qty))
        if quantity == 0:
            continue

        if exec_mode is ExecutionMode.MOO:
            order_type = "MARKET"
            price = None
        else:
            base_price = float(closes[idx])
            if np.isnan(base_price) or base_price <= 0:
                continue
            if side == "BUY":
                raw = base_price * (1.0 - offset_ratio)
            else:
                raw = base_price * (1.0 + offset_ratio)
            price = _round_price(raw, side)
            order_type = "LIMIT"

        client_order_id = f"{today}-{symbol}-{side}-{quantity}"
        orders.append(
            OrderSpec(
                symbol=symbol,
                side=side,
                qty=quantity,
                order_type=order_type,  # type: ignore[arg-type]
                limit_price=price,
                tif="DAY",
                client_order_id=client_order_id,
            )
        )

    return orders
