"""KRX 틱 라운딩(호가가격단위) 보정 함수와 LIMIT 변환 검증 테스트."""
from __future__ import annotations

import numpy as np

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env import ExecutionMode
from live.route import _krx_tick, _round_price, to_orders


def test_tick_brackets():
    cases = [
        (1000, 1),
        (2000, 5),
        (4999, 5),
        (5000, 10),
        (19999, 10),
        (20000, 50),
        (50000, 100),
        (500000, 1000),
    ]
    for price, expected in cases:
        assert _krx_tick(price) == expected


def test_round_price_buy_sell():
    assert _round_price(10003, "BUY") == 10000
    assert _round_price(10003, "SELL") == 10010
    assert _round_price(19999, "BUY") == 19990
    assert _round_price(19999, "SELL") == 20000
    assert _round_price(50510, "BUY") == 50500
    assert _round_price(50510, "SELL") == 50600


def test_to_orders_limit_rounding():
    orders = to_orders(
        ExecutionMode.LIMIT_OHLC,
        np.array([10, -15]),
        ["AAA", "BBB"],
        np.array([10003.0, 50510.0]),
        0.0,
    )
    assert orders[0].limit_price == 10000
    assert orders[1].limit_price == 50600
