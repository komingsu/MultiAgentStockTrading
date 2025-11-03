from __future__ import annotations

"""SimBroker — 환경(env)의 체결 규칙을 그대로 모사하는 초경량 시뮬레이션 브로커.

목적:
  - 동일한 OrderSpec 묶음을 env.step()과 동일한 규칙으로 채결시켜 Parity 테스트에 사용
  - 체결 사유(reason) 키는 env와 합의된 값만 사용한다.
    {"limit_not_reached","adv_cap_zero","adv_cap_limit","insufficient_cash","no_position"}

주의: 로직/시그니처 변경 없이 주석/도큐스트링만 추가했다.
"""

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from env import ExecutionMode, ExecutionSpec
from execution.order_spec import OrderSpec


@dataclass
class FillReport:
    """체결 결과 레코드(단순화 버전)."""
    symbol: str
    side: str
    requested: int
    filled: int
    price: float | None
    reason: str | None
    cash_after: float


class SimBroker:
    """환경과 동일 규칙으로 체결을 모사하는 시뮬레이터(Parity 용)."""

    def __init__(self, exec_spec: ExecutionSpec):
        self.exec_spec = exec_spec
        self.last_state: Dict[str, object] | None = None

    def simulate(
        self,
        day_t: int,
        order_list: List[OrderSpec],
        env_state_snapshot: Dict[str, object],
    ) -> List[Dict[str, object]]:
        """
        주문 스펙 리스트를 받아 t→t+1 규칙(MOO/LIMIT_OHLC)로 체결을 모사.

        Args:
          day_t: 오늘(day 인덱스)
          order_list: OrderSpec 리스트(양수 수량)
          env_state_snapshot: 현금/보유/오늘/내일 프레임 등 상태 스냅샷

        Returns:
          env.info["fills"]와 동형의 레코드 리스트
        """
        if not order_list:
            return []

        cash = float(env_state_snapshot["cash"])
        holdings = np.asarray(env_state_snapshot["holdings"], dtype=float)
        tickers: List[str] = list(env_state_snapshot["tickers"])
        today_frame: pd.DataFrame = env_state_snapshot["today_frame"]
        next_frame: pd.DataFrame = env_state_snapshot["next_frame"]

        ohlc_next = next_frame.set_index("tic")
        ohlc_today = today_frame.set_index("tic")
        fee_buy = np.asarray(self.exec_spec.fee_buy, dtype=float)
        fee_sell = np.asarray(self.exec_spec.fee_sell, dtype=float)
        slippage = self.exec_spec.slippage_decimal
        adv_frac_value = self.exec_spec.adv_frac

        ticker_index = {tic: idx for idx, tic in enumerate(tickers)}
        reports: List[Dict[str, object]] = []

        for spec in order_list:
            idx = ticker_index.get(spec.symbol)
            if idx is None:
                reports.append(
                    {
                        "symbol": spec.symbol,
                        "side": spec.side,
                        "requested": spec.qty,
                        "filled": 0,
                        "price": None,
                        "reason": "unknown_symbol",
                        "cash_after": cash,
                    }
                )
                continue

            row_next = ohlc_next.loc[spec.symbol]
            volume = float(row_next.get("volume", 0.0) or 0.0)
            if adv_frac_value is None:
                adv_cap = None
            else:
                adv_cap = max(int(np.floor(volume * float(adv_frac_value))), 0)
            requested = int(spec.qty)
            side = spec.side
            filled = 0
            fill_price = None
            reason = None

            if requested == 0:
                reports.append(
                    {
                        "symbol": spec.symbol,
                        "side": side,
                        "requested": requested,
                        "filled": 0,
                        "price": None,
                        "reason": "zero_qty",
                        "cash_after": cash,
                    }
                )
                continue

            if adv_cap == 0:
                reason = "adv_cap_zero"
            else:
                qty_cap = requested if adv_cap is None else min(requested, adv_cap)
                open_px = float(row_next["open"])
                high_px = float(row_next["high"])
                low_px = float(row_next["low"])
                close_px = float(row_next["close"])

                if qty_cap <= 0:
                    reason = "adv_cap_limit"
                else:
                    if self.exec_spec.mode is ExecutionMode.MOO:
                        fill_price = open_px
                        filled = qty_cap
                    else:
                        limit_offset = self.exec_spec.limit_offset_bps / 10_000.0
                        reference_close = float(ohlc_today.loc[spec.symbol]["close"])
                        if side == "BUY":
                            limit_price = reference_close * (1.0 - limit_offset)
                            if open_px <= limit_price:
                                fill_price = open_px
                                filled = qty_cap
                            elif low_px <= limit_price:
                                fill_price = limit_price
                                filled = qty_cap
                            else:
                                reason = "limit_not_reached"
                        else:
                            limit_price = reference_close * (1.0 + limit_offset)
                            if open_px >= limit_price:
                                fill_price = open_px
                                filled = qty_cap
                            elif high_px >= limit_price:
                                fill_price = limit_price
                                filled = qty_cap
                            else:
                                reason = "limit_not_reached"

                    if filled > 0 and fill_price is not None and reason is None:
                        if side == "BUY":
                            eff_mult = 1.0 + fee_buy[idx] + slippage
                            max_qty_cash = int(np.floor(cash / (fill_price * eff_mult)))
                            if max_qty_cash <= 0:
                                filled = 0
                                reason = "insufficient_cash"
                            else:
                                filled = min(filled, max_qty_cash)
                                cash -= fill_price * filled * eff_mult
                                holdings[idx] += filled
                        else:
                            eff_mult = 1.0 - (fee_sell[idx] + slippage)
                            if eff_mult <= 0:
                                filled = 0
                                reason = "invalid_costs"
                            else:
                                available = int(np.floor(holdings[idx]))
                                if available <= 0:
                                    filled = 0
                                    reason = "no_position"
                                else:
                                    filled = min(filled, available)
                                    cash += fill_price * filled * eff_mult
                                    holdings[idx] -= filled

                if filled == 0 and reason is None:
                    reason = None

            reports.append(
                {
                    "symbol": spec.symbol,
                    "side": side,
                    "requested": requested,
                    "filled": filled,
                    "price": fill_price,
                    "reason": reason,
                    "cash_after": cash,
                }
            )

        self.last_state = {"cash": cash, "holdings": holdings.copy(), "day": day_t}
        return reports
