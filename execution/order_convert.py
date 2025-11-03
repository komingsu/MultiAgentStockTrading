from __future__ import annotations

"""
주문 계획(AOC: Action → Order Converter) 모듈

이 모듈은 에이전트의 목표 비중(또는 델타)을 입력받아, 다음 원칙에 맞는
정수 수량 벡터(종목별 desired_qty)와 리포트를 산출한다.

- 한도 제약: |w_i| ≤ w_max, 일일 턴오버 ≤ turnover_max, 최소주문금액(min_notional)
- ADV 캡: 종목별 t-1(또는 SMA5) 거래량의 adv_cap_frac 비율 이내로 수량 제한
- 비용/슬리피지: buy/sell 수수료와 slippage_bps를 현금 지출/유입 계산에 반영
- 캐시 버퍼: 계획 후 남길 현금 비율 cash_buffer_pct 유지
- 노트레이드 밴드: |w_tgt − w_cur| < epsilon_w 인 경우 거래 생략
- 라운딩: lot_size 단위로 내림 라운딩

반환되는 리포트에는 per_symbol 상세(요청/클립/노테/리미트 가격 등)와
클리핑 사유(clips) 및 안전성 플래그(ok)가 포함된다.

Notes:
  - 본 모듈은 백테스트/실거래 공용의 변환 규칙을 제공해 Parity를 보장한다.
  - 로직/시그니처/동작 변경 없이 주석과 도큐스트링만 추가했다.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass(frozen=True)
class Limits:
    """리스크/체크 제약 파라미터 집합.

    Args:
      w_max: 종목별 최대 절대 비중 한도 (예: 0.10 ⇒ 10%)
      turnover_max: 일일 매수+매도 금액 / 포트총액 상한 (예: 0.20)
      min_notional: 최소 주문금액(원). 이보다 작으면 주문 생략
      epsilon_w: 노트레이드 밴드 폭 (|Δw| < epsilon → 0으로 처리)
      lot_size: 수량 라운딩 단위(한국 주식=1주)
      max_orders: 당일 주문 종목 수 상한(None/0이면 미적용)
    """
    w_max: float
    turnover_max: float
    min_notional: float
    epsilon_w: float
    lot_size: int = 1
    max_orders: int | None = None


@dataclass(frozen=True)
class PriceContext:
    """가격 맥락.

    close: 리밸런싱 기준가(보통 전일 종가)
    ref_for_limits: 한도 계산용 기준가(미지정 시 close 사용)
    """
    close: np.ndarray
    ref_for_limits: np.ndarray | None = None


@dataclass(frozen=True)
class PortfolioSnapshot:
    """현재 포트폴리오 스냅샷.

    cash: 가용 현금(원)
    holdings: 종목별 현재 보유 주식 수
    """
    cash: float
    holdings: np.ndarray


@dataclass(frozen=True)
class AdvGuard:
    """ADV(일일 거래량) 기반 수량 상한 설정.

    adv_cap_frac: t+1 거래량의 비율(0.1=10%) 만큼만 체결 허용
    adv_reference: ADV 추정치(보통 t-1 또는 SMA5 거래량)
    """
    adv_cap_frac: float
    adv_reference: np.ndarray


@dataclass(frozen=True)
class ExecKnobs:
    """실행 비용/버퍼/오프셋 파라미터.

    fee_buy/sell: 종목별 매수/매도 수수료 비율
    slippage_bps: 추가 슬리피지(bps)
    cash_buffer_pct: 계획 후 유지할 현금 비율
    limit_offset_bps: LIMIT_OHLC 시 리미트 산출에 사용하는 오프셋(bps)
    """
    fee_buy: np.ndarray
    fee_sell: np.ndarray
    slippage_bps: float
    cash_buffer_pct: float
    limit_offset_bps: float


def _as_numpy(vector: np.ndarray | List[float]) -> np.ndarray:
    """리스트/넘파이를 1차원 넘파이(float)로 강제 변환."""
    arr = np.asarray(vector, dtype=float)
    if arr.ndim != 1:
        raise ValueError("Expected 1-D vector input")
    return arr


def _sanitize_weights(
    target_w: np.ndarray,
    limits: Limits,
    short_allowed: bool,
) -> Tuple[np.ndarray, Dict[str, List[int]]]:
    """목표 비중을 w_max/공매도 허용 여부에 맞게 클리핑.

    Returns:
      (클리핑된 w_tgt, 클립 인덱스 정보 사전)
    """
    w_tgt = np.nan_to_num(target_w, nan=0.0, posinf=0.0, neginf=0.0)
    clip_w_max_idx = np.where(np.abs(w_tgt) > limits.w_max)[0].tolist()
    w_tgt = np.clip(w_tgt, -limits.w_max if short_allowed else 0.0, limits.w_max)
    if not short_allowed:
        neg_idx = np.where(w_tgt < 0)[0].tolist()
        if neg_idx:
            w_tgt[neg_idx] = 0.0
    else:
        neg_idx = []
    return w_tgt, {"w_max": clip_w_max_idx, "no_short": neg_idx}


def _apply_no_trade_band(
    w_curr: np.ndarray,
    w_tgt: np.ndarray,
    epsilon_w: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """노트레이드 밴드 적용: 작은 비중 변화는 그대로 유지(mask=True)."""
    mask = np.abs(w_tgt - w_curr) < epsilon_w
    w_adj = np.where(mask, w_curr, w_tgt)
    return w_adj, mask


def _lot_round(values: np.ndarray, lot_size: int) -> np.ndarray:
    """수량을 lot_size 단위로 내림 라운딩."""
    if lot_size <= 1:
        return values.astype(int)
    return ((values.astype(int) // lot_size) * lot_size).astype(int)


def _adv_cap_vector(
    adv_guard: AdvGuard,
) -> np.ndarray:
    """ADV 비율을 적용해 종목별 최대 수량 벡터를 생성."""
    frac = max(float(adv_guard.adv_cap_frac), 0.0)
    if frac == 0.0:
        return np.zeros_like(adv_guard.adv_reference, dtype=int)
    return np.floor(np.maximum(adv_guard.adv_reference, 0.0) * frac).astype(int)


def _compute_total_value(cash: float, holdings: np.ndarray, prices: np.ndarray) -> float:
    """현금+보유평가액 합계(포트 총액)."""
    return float(cash + float(np.dot(holdings, prices)))


def plan_from_weights(
    target_w: np.ndarray,
    snap: PortfolioSnapshot,
    prices: PriceContext,
    limits: Limits,
    adv: AdvGuard,
    knobs: ExecKnobs,
    short_allowed: bool = False,
) -> tuple[np.ndarray, dict]:
    """
    목표 비중을 실행가능한 정수 수량 벡터로 변환.

    절차(고정):
      1) 현재 비중 w_curr 계산 → w_max/노트레이드 밴드 적용해 w_tgt 확정
      2) 목표 주식 수(tgt_shares) 및 Δshares 산출
      3) SELL 선반영(현금 유입), 보유/ADV로 클리핑
      4) BUY 예산/최소금액/ADV/lot_size 적용 → 필요 시 비례 스케일
      5) 턴오버 상한 검사 → 필요 시 비례 스케일
      6) 안전성 확인(ok) 실패 시 no-trade로 강등(모두 0)

    Returns:
      (desired_qty: np.ndarray, report: dict)

    Notes:
      - 수수료/슬리피지는 현금 유출입 계산에만 반영(보유 수량에는 영향 없음)
      - report["per_symbol"][i]["limit_px"]는 LIMIT_OHLC 오프셋 참고용 가격
    """
    target_w = _as_numpy(target_w)
    close = _as_numpy(prices.close)
    ref_prices = (
        _as_numpy(prices.ref_for_limits) if prices.ref_for_limits is not None else close
    )
    holdings = _as_numpy(snap.holdings).astype(float)
    fee_buy = _as_numpy(knobs.fee_buy)
    fee_sell = _as_numpy(knobs.fee_sell)
    adv_ref = _as_numpy(adv.adv_reference)

    if not (
        len(target_w)
        == len(close)
        == len(ref_prices)
        == len(holdings)
        == len(fee_buy)
        == len(fee_sell)
        == len(adv_ref)
    ):
        raise ValueError("Input lengths for symbols do not match")

    n_assets = len(close)
    notes: List[str] = []

    total_value = _compute_total_value(snap.cash, holdings, ref_prices)
    if total_value <= 0:
        notes.append("portfolio_value_nonpositive->no_trade")
        desired_zero = np.zeros(n_assets, dtype=int)
        return desired_zero, {
            "V": float(total_value),
            "budget": 0.0,
            "reserve": 0.0,
            "turnover_ratio": 0.0,
            "clips": {"w_max": [], "no_short": [], "epsilon": [], "adv": [], "cash": False, "turnover": False},
            "per_symbol": [],
            "notes": notes,
            "ok": {
                "cash_nonnegative": True,
                "no_short": True,
                "adv_respected": True,
                "turnover_respected": True,
            },
        }

    w_curr = np.divide(
        holdings * ref_prices,
        total_value,
        out=np.zeros_like(close),
        where=total_value > 0,
    )

    w_tgt_raw, clip_info = _sanitize_weights(target_w, limits, short_allowed)
    w_tgt, band_mask = _apply_no_trade_band(w_curr, w_tgt_raw, limits.epsilon_w)

    tgt_dollar = w_tgt * total_value
    tgt_shares = np.floor_divide(
        tgt_dollar,
        np.maximum(ref_prices, 1e-12),
        where=np.maximum(ref_prices, 1e-12) > 0,
    ).astype(int)
    desired_delta = tgt_shares - holdings.astype(int)

    adv_cap = _adv_cap_vector(adv)

    sell_req = np.clip(-desired_delta, 0, None)
    sell_req = np.minimum(sell_req, holdings.astype(int))
    sell_req = np.minimum(sell_req, adv_cap)
    sell_req = _lot_round(sell_req, limits.lot_size)

    slippage_decimal = max(float(knobs.slippage_bps) / 10_000.0, 0.0)
    eff_mult_sell = np.maximum(1.0 - (fee_sell + slippage_decimal), 0.0)
    sell_notional = sell_req * ref_prices
    realized_sell_cash = float(np.dot(sell_req, ref_prices * eff_mult_sell))

    cash_after_sell = snap.cash + realized_sell_cash
    reserve = max(total_value * knobs.cash_buffer_pct, 0.0)
    budget = max(cash_after_sell - reserve, 0.0)

    buy_req = np.clip(desired_delta, 0, None)
    buy_req = np.minimum(buy_req, adv_cap)
    raw_notional_buy = buy_req * ref_prices
    buy_req = np.where(raw_notional_buy >= limits.min_notional, buy_req, 0)
    buy_req = _lot_round(buy_req, limits.lot_size)
    eff_mult_buy = 1.0 + fee_buy + slippage_decimal
    buy_cost_gross = float(np.dot(buy_req, ref_prices * eff_mult_buy))

    clips_adv_idx = np.where((np.clip(-desired_delta, 0, None) > adv_cap) | (np.clip(desired_delta, 0, None) > adv_cap))[0].tolist()

    if buy_cost_gross > budget + 1e-9 and buy_cost_gross > 0:
        scale = budget / buy_cost_gross
        buy_req = np.floor(buy_req * scale).astype(int)
        buy_req = _lot_round(buy_req, limits.lot_size)
        buy_cost_gross = float(np.dot(buy_req, ref_prices * eff_mult_buy))
        clips_cash = True
    else:
        clips_cash = False

    sell_notional_value = float(np.dot(sell_req, ref_prices))
    buy_notional_value = float(np.dot(buy_req, ref_prices))
    turnover = (sell_notional_value + buy_notional_value) / max(total_value, 1e-12)
    turnover_gross = (
        float(np.dot(sell_req, ref_prices * eff_mult_sell))
        + float(np.dot(buy_req, ref_prices * eff_mult_buy))
    ) / max(total_value, 1e-12)
    clips_turnover = False
    if turnover > limits.turnover_max + 1e-9 and buy_notional_value > 0:
        scale = limits.turnover_max / max(turnover, 1e-12)
        buy_req = np.floor(buy_req * scale).astype(int)
        buy_req = _lot_round(buy_req, limits.lot_size)
        buy_notional_value = float(np.dot(buy_req, ref_prices))
        buy_cost_gross = float(np.dot(buy_req, ref_prices * eff_mult_buy))
        turnover = (sell_notional_value + buy_notional_value) / max(total_value, 1e-12)
        clips_turnover = True

    desired_final = buy_req - sell_req

    if limits.max_orders is not None and limits.max_orders > 0:
        active_indices = np.where(np.abs(desired_final) > 0)[0]
        if active_indices.size > limits.max_orders:
            notionals = np.abs(desired_final[active_indices] * ref_prices[active_indices])
            keep_idx = active_indices[np.argsort(-notionals)[: limits.max_orders]]
            mask_keep = np.zeros_like(desired_final, dtype=bool)
            mask_keep[keep_idx] = True
            desired_final = np.where(mask_keep, desired_final, 0)
            buy_req = np.where(mask_keep, buy_req, 0)
            sell_req = np.where(mask_keep, sell_req, 0)
            buy_notional_value = float(np.dot(buy_req, ref_prices))
            sell_notional_value = float(np.dot(sell_req, ref_prices))
            buy_cost_gross = float(np.dot(buy_req, ref_prices * eff_mult_buy))
            turnover = (sell_notional_value + buy_notional_value) / max(total_value, 1e-12)
            turnover_gross = (
                float(np.dot(sell_req, ref_prices * eff_mult_sell))
                + float(np.dot(buy_req, ref_prices * eff_mult_buy))
            ) / max(total_value, 1e-12)

    cash_after_trades = cash_after_sell - buy_cost_gross
    holdings_after = holdings + desired_final

    cash_nonnegative = cash_after_trades >= -1e-6
    no_short_flag = bool(np.all(holdings_after >= -1e-6)) if not short_allowed else True
    adv_respected = bool(np.all(buy_req <= adv_cap) and np.all(sell_req <= adv_cap))
    turnover_respected = turnover <= limits.turnover_max + 1e-6

    if budget <= 0 or (np.all(buy_req == 0) and np.all(sell_req == 0)):
        notes.append("no_trade_budget_or_min_notional")

    # guard for illiquid/locked stocks (volume <=0)
    if np.any(adv_ref <= 0):
        locked_idx = np.where(adv_ref <= 0)[0]
        if locked_idx.size:
            notes.append(f"illiquid_filtered:{locked_idx.tolist()}")
            mask_locked = adv_ref <= 0
            desired_final = np.where(mask_locked, 0, desired_final)
            buy_req = np.where(mask_locked, 0, buy_req)
            sell_req = np.where(mask_locked, 0, sell_req)
            buy_notional_value = float(np.dot(buy_req, ref_prices))
            sell_notional_value = float(np.dot(sell_req, ref_prices))
            buy_cost_gross = float(np.dot(buy_req, ref_prices * eff_mult_buy))
            turnover = (sell_notional_value + buy_notional_value) / max(total_value, 1e-12)
            turnover_gross = (
                float(np.dot(sell_req, ref_prices * eff_mult_sell))
                + float(np.dot(buy_req, ref_prices * eff_mult_buy))
            ) / max(total_value, 1e-12)

    per_symbol: List[Dict[str, object]] = []
    for i in range(n_assets):
        limit_px = {
            "buy": ref_prices[i] * (1.0 - knobs.limit_offset_bps / 10_000.0),
            "sell": ref_prices[i] * (1.0 + knobs.limit_offset_bps / 10_000.0),
        }
        per_symbol.append(
            {
                "index": i,
                "w_curr": float(w_curr[i]),
                "w_tgt": float(w_tgt[i]),
                "delta_shares": int(desired_final[i]),
                "sell_req": int(sell_req[i]),
                "buy_req": int(buy_req[i]),
                "adv_cap": int(adv_cap[i]),
                "notional_buy": float(ref_prices[i] * buy_req[i]),
                "notional_sell": float(ref_prices[i] * sell_req[i]),
                "limit_px": limit_px,
            }
        )

    clips = {
        "w_max": clip_info.get("w_max", []),
        "no_short": clip_info.get("no_short", []),
        "epsilon": np.where(band_mask)[0].tolist(),
        "adv": clips_adv_idx,
        "cash": clips_cash,
        "turnover": clips_turnover,
    }

    ok_flags = {
        "cash_nonnegative": bool(cash_nonnegative),
        "no_short": bool(no_short_flag),
        "adv_respected": bool(adv_respected),
        "turnover_respected": bool(turnover_respected),
    }

    violated = not all(ok_flags.values())
    if violated:
        notes.append("no_trade_due_to_violation")
        desired_final = np.zeros_like(desired_final)
        buy_req = np.zeros_like(buy_req)
        sell_req = np.zeros_like(sell_req)
        per_symbol = [
            {
                **entry,
                "delta_shares": 0,
                "sell_req": 0,
                "buy_req": 0,
                "notional_buy": 0.0,
                "notional_sell": 0.0,
            }
            for entry in per_symbol
        ]
        buy_notional_value = 0.0
        sell_notional_value = 0.0
        buy_cost_gross = 0.0
        realized_sell_cash = 0.0
        cash_after_trades = snap.cash
        holdings_after = holdings.copy()
        turnover = 0.0
        turnover_gross = 0.0

    report = {
        "V": float(total_value),
        "budget": float(budget),
        "reserve": float(reserve),
        "cash_in": float(realized_sell_cash),
        "cash_out": float(buy_cost_gross),
        "turnover_ratio": float(turnover),
        "turnover_ratio_gross": float(turnover_gross),
        "clips": clips,
        "per_symbol": per_symbol,
        "notes": notes,
        "ok": ok_flags,
        "holdings_after": holdings_after.tolist(),
        "cash_after": float(cash_after_trades),
    }

    return desired_final.astype(int), report
