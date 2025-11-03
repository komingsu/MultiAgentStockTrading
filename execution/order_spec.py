from __future__ import annotations

"""공통 주문 명세(OrderSpec) 데이터 구조.

에이전트/AOC/시뮬/실거래(라우터)가 동일한 주문 스펙을 공유함으로써
백테스트 ↔ 라이브 동등성(Parity)을 유지한다.

Notes:
  - `order_type`: `MARKET`(시장가) 또는 `LIMIT`(지정가)
  - `tif`: Time-In-Force, 본 레포는 DAY(당일)만 사용
  - `client_order_id`: 멱등성/중복 방지용 클라이언트 생성 ID (예: YYYYMMDD-종목-SIDE-수량)
"""

from dataclasses import dataclass
from typing import Literal, Optional

Side = Literal["BUY", "SELL"]
OrderType = Literal["MARKET", "LIMIT"]
TIF = Literal["DAY"]


@dataclass(frozen=True)
class OrderSpec:
    """
    주문 스펙.

    Args:
      symbol: 종목 코드(예: "005930")
      side: 매수/매도 ("BUY"/"SELL")
      qty: 주문 수량(양수 정수)
      order_type: "MARKET" 또는 "LIMIT"
      limit_price: 지정가 주문일 때의 가격(원). 시장가는 None이어야 함
      tif: 유효기간, DAY 고정
      client_order_id: 클라이언트에서 생성한 멱등 키(중복 전송 방지)

    Raises:
      ValueError: 수량/가격/타입 조합이 잘못된 경우
    """
    symbol: str
    side: Side
    qty: int
    order_type: OrderType
    limit_price: Optional[float] = None
    tif: TIF = "DAY"
    client_order_id: str = ""

    def __post_init__(self) -> None:
        if self.qty <= 0:
            raise ValueError("OrderSpec.qty must be positive")
        if self.order_type == "LIMIT" and self.limit_price is None:
            raise ValueError("LIMIT orders require limit_price")
        if self.order_type == "MARKET" and self.limit_price is not None:
            raise ValueError("MARKET orders must not have limit_price")
