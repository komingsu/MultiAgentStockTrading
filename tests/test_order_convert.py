"""AOC(plan_from_weights)의 제약/클리핑/리포트 로직을 검증하는 테스트."""
from __future__ import annotations

import numpy as np
import pytest

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from execution.order_convert import (
    AdvGuard,
    ExecKnobs,
    Limits,
    PortfolioSnapshot,
    PriceContext,
    plan_from_weights,
)


def _default_knobs(
    slippage_bps: float = 0.0,
    cash_buffer_pct: float = 0.0,
    n: int = 1,
) -> ExecKnobs:
    fee_buy = np.full(n, 0.0, dtype=float)
    fee_sell = np.full(n, 0.0, dtype=float)
    return ExecKnobs(
        fee_buy=fee_buy,
        fee_sell=fee_sell,
        slippage_bps=slippage_bps,
        cash_buffer_pct=cash_buffer_pct,
        limit_offset_bps=50.0,
    )


def test_sell_first_generates_expected_cash():
    limits = Limits(w_max=1.0, turnover_max=1.0, min_notional=0.0, epsilon_w=0.0)
    prices = PriceContext(close=np.array([100.0]))
    snap = PortfolioSnapshot(cash=0.0, holdings=np.array([10]))
    adv = AdvGuard(adv_cap_frac=1.0, adv_reference=np.array([10.0]))
    knobs = _default_knobs()

    desired, report = plan_from_weights(
        target_w=np.array([0.0]),
        snap=snap,
        prices=prices,
        limits=limits,
        adv=adv,
        knobs=knobs,
    )
    assert desired.tolist() == [-10]
    assert pytest.approx(report["cash_in"], rel=1e-6) == 1000.0
    assert pytest.approx(report["cash_after"], rel=1e-6) == 1000.0
    assert report["turnover_ratio"] == pytest.approx(1.0)
    assert "turnover_ratio_gross" in report


def test_adv_cap_limits_buy_and_sell():
    limits = Limits(w_max=1.0, turnover_max=1.0, min_notional=0.0, epsilon_w=0.0)
    prices = PriceContext(close=np.array([50.0]))
    snap = PortfolioSnapshot(cash=5000.0, holdings=np.array([10]))
    adv = AdvGuard(adv_cap_frac=0.1, adv_reference=np.array([25.0]))
    knobs = _default_knobs()

    desired, report = plan_from_weights(
        target_w=np.array([0.0]),
        snap=snap,
        prices=prices,
        limits=limits,
        adv=adv,
        knobs=knobs,
    )
    # sell side should clip to floor(0.1 * 25) == 2
    assert desired.tolist() == [-2]
    assert report["per_symbol"][0]["sell_req"] == 2

    desired_buy, report_buy = plan_from_weights(
        target_w=np.array([0.8]),
        snap=PortfolioSnapshot(cash=5000.0, holdings=np.array([0])),
        prices=prices,
        limits=limits,
        adv=adv,
        knobs=knobs,
    )
    assert desired_buy.tolist() == [2]
    assert report_buy["per_symbol"][0]["buy_req"] == 2
    assert report_buy["ok"]["adv_respected"] is True


def test_no_trade_band_zeroes_small_changes():
    limits = Limits(w_max=1.0, turnover_max=1.0, min_notional=0.0, epsilon_w=0.01)
    prices = PriceContext(close=np.array([100.0]))
    portfolio_value = 1000.0
    snap = PortfolioSnapshot(cash=500.0, holdings=np.array([5]))
    adv = AdvGuard(adv_cap_frac=0.0, adv_reference=np.array([0.0]))
    knobs = _default_knobs()

    w_curr = (snap.holdings * prices.close) / portfolio_value
    assert pytest.approx(w_curr[0]) == 0.5
    target_w = np.array([w_curr[0] + 0.005])
    desired, report = plan_from_weights(
        target_w=target_w,
        snap=snap,
        prices=prices,
        limits=limits,
        adv=adv,
        knobs=knobs,
    )
    assert desired.tolist() == [0]
    assert 0 in report["clips"]["epsilon"]


def test_turnover_cap_scales_buys():
    limits = Limits(w_max=1.0, turnover_max=0.1, min_notional=0.0, epsilon_w=0.0)
    prices = PriceContext(close=np.array([100.0]))
    snap = PortfolioSnapshot(cash=10_000.0, holdings=np.array([0]))
    adv = AdvGuard(adv_cap_frac=1.0, adv_reference=np.array([100.0]))
    knobs = _default_knobs()

    desired, report = plan_from_weights(
        target_w=np.array([0.8]),
        snap=snap,
        prices=prices,
        limits=limits,
        adv=adv,
        knobs=knobs,
    )
    assert report["clips"]["turnover"] is True
    turnover = report["turnover_ratio"]
    assert turnover <= limits.turnover_max + 1e-6


def test_min_notional_prevents_small_orders():
    limits = Limits(w_max=1.0, turnover_max=1.0, min_notional=50_000.0, epsilon_w=0.0)
    prices = PriceContext(close=np.array([1_000.0]))
    snap = PortfolioSnapshot(cash=10_000.0, holdings=np.array([0]))
    adv = AdvGuard(adv_cap_frac=1.0, adv_reference=np.array([1_000.0]))
    knobs = _default_knobs()

    desired, report = plan_from_weights(
        target_w=np.array([0.05]),
        snap=snap,
        prices=prices,
        limits=limits,
        adv=adv,
        knobs=knobs,
    )
    assert desired.tolist() == [0]
    assert "no_trade_budget_or_min_notional" in report["notes"]


def test_cash_and_holdings_safety_enforced():
    limits = Limits(w_max=1.0, turnover_max=1.0, min_notional=0.0, epsilon_w=0.0)
    prices = PriceContext(close=np.array([100.0]))
    snap = PortfolioSnapshot(cash=0.0, holdings=np.array([0]))
    adv = AdvGuard(adv_cap_frac=0.0, adv_reference=np.array([0.0]))
    knobs = _default_knobs(cash_buffer_pct=0.6)

    desired, report = plan_from_weights(
        target_w=np.array([1.0]),
        snap=snap,
        prices=prices,
        limits=limits,
        adv=adv,
        knobs=knobs,
    )
    assert desired.tolist() == [0]
    assert report["ok"]["cash_nonnegative"] is True


def test_lot_rounding_applies():
    limits = Limits(w_max=1.0, turnover_max=1.0, min_notional=0.0, epsilon_w=0.0, lot_size=10)
    prices = PriceContext(close=np.array([100.0]))
    snap = PortfolioSnapshot(cash=100_000.0, holdings=np.array([0]))
    adv = AdvGuard(adv_cap_frac=1.0, adv_reference=np.array([1_000.0]))
    knobs = _default_knobs()

    desired, report = plan_from_weights(
        target_w=np.array([0.45]),
        snap=snap,
        prices=prices,
        limits=limits,
        adv=adv,
        knobs=knobs,
    )
    assert desired[0] % 10 == 0
    assert report["per_symbol"][0]["buy_req"] % 10 == 0


def test_limit_prices_reported_correctly():
    limits = Limits(w_max=1.0, turnover_max=1.0, min_notional=0.0, epsilon_w=0.0)
    prices = PriceContext(close=np.array([200.0]))
    snap = PortfolioSnapshot(cash=50_000.0, holdings=np.array([0]))
    adv = AdvGuard(adv_cap_frac=1.0, adv_reference=np.array([1_000.0]))
    knobs = _default_knobs(slippage_bps=10.0)

    desired, report = plan_from_weights(
        target_w=np.array([0.2]),
        snap=snap,
        prices=prices,
        limits=limits,
        adv=adv,
        knobs=knobs,
    )
    limit_info = report["per_symbol"][0]["limit_px"]
    assert pytest.approx(limit_info["buy"], rel=1e-6) == 200.0 * (1 - knobs.limit_offset_bps / 10_000.0)
    assert pytest.approx(limit_info["sell"], rel=1e-6) == 200.0 * (1 + knobs.limit_offset_bps / 10_000.0)
    assert desired[0] >= 0


def test_violation_no_trade_preserves_flags():
    limits = Limits(
        w_max=1.0,
        turnover_max=1.0,
        min_notional=0.0,
        epsilon_w=0.0,
    )
    prices = PriceContext(close=np.array([100.0]))
    snap = PortfolioSnapshot(cash=-100.0, holdings=np.array([2]))
    adv = AdvGuard(adv_cap_frac=0.0, adv_reference=np.array([0.0]))
    knobs = _default_knobs()

    desired, report = plan_from_weights(
        target_w=np.array([0.5]),
        snap=snap,
        prices=prices,
        limits=limits,
        adv=adv,
        knobs=knobs,
    )
    assert desired.tolist() == [0]
    assert report["ok"]["cash_nonnegative"] is False
    assert "no_trade_due_to_violation" in report["notes"]


def test_max_orders_cap_applied():
    limits = Limits(
        w_max=1.0,
        turnover_max=1.0,
        min_notional=0.0,
        epsilon_w=0.0,
        max_orders=1,
    )
    prices = PriceContext(close=np.array([100.0, 50.0, 25.0]))
    snap = PortfolioSnapshot(cash=100_000.0, holdings=np.array([0, 0, 0]))
    adv = AdvGuard(adv_cap_frac=1.0, adv_reference=np.array([1_000.0, 1_000.0, 1_000.0]))
    knobs = _default_knobs(n=3)

    desired, report = plan_from_weights(
        target_w=np.array([0.4, 0.3, 0.2]),
        snap=snap,
        prices=prices,
        limits=limits,
        adv=adv,
        knobs=knobs,
    )
    non_zero = np.nonzero(desired)[0]
    assert non_zero.size == 1


def test_turnover_gross_recorded():
    limits = Limits(w_max=1.0, turnover_max=1.0, min_notional=0.0, epsilon_w=0.0)
    prices = PriceContext(close=np.array([100.0]))
    snap = PortfolioSnapshot(cash=50_000.0, holdings=np.array([0]))
    adv = AdvGuard(adv_cap_frac=1.0, adv_reference=np.array([1_000.0]))
    knobs = _default_knobs(slippage_bps=100)

    _, report = plan_from_weights(
        target_w=np.array([0.5]),
        snap=snap,
        prices=prices,
        limits=limits,
        adv=adv,
        knobs=knobs,
    )
    assert "turnover_ratio_gross" in report
    assert report["turnover_ratio_gross"] >= report["turnover_ratio"]
