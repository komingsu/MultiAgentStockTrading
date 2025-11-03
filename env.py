from __future__ import annotations

"""
강화학습 환경(StockTradingEnv) — 실거래 동등 규칙(MOO/LIMIT_OHLC) 반영

핵심 개념:
- ExecutionMode: MOO(개장가 시장가) / LIMIT_OHLC(전일 종가±offset으로 t+1 OHLC 판정)
- ExecutionSpec: ADV 캡/슬리피지/수수료/오프셋/TIF 등 실행 파라미터 묶음
- OrderFill: step()에서 심볼별 체결 결과 기록(부분체결/미체결 사유 포함)

본 파일의 주석/도큐스트링은 동작을 바꾸지 않고, env→AOC→to_orders→Sim/LIVE
경로의 Parity를 독자가 이해하기 쉽게 설명하는 목적만 가진다.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List

import logging
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt

import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor

matplotlib.use("Agg")

import config
from hyperparams import PortfolioInitParams


LOGGER = logging.getLogger(__name__)


class ExecutionMode(str, Enum):
    """실행/체결 모드.

    - MOO: 오늘 관측→내일 개장 동시호가에 시장가 주문, 개장가로 전량 체결(초기 단순화)
    - LIMIT_OHLC: 전일 종가±offset(bps)로 지정가를 정하고 t+1 OHLC로 1일 체결 판정
    """
    MOO = "MOO"
    LIMIT_OHLC = "LIMIT_OHLC"


@dataclass(frozen=True)
class ExecutionSpec:
    """실행 파라미터 묶음.

    mode: 실행 모드(MOO/LIMIT_OHLC)
    adv_frac: t+1 거래량 비율(ADV 캡)
    slippage_bps: 추가 슬리피지(bps)
    fee_buy/sell: 종목별 수수료율 배열
    limit_offset_bps: LIMIT 지정가 오프셋(bps)
    day_only: DAY(당일) 주문만 사용 여부
    """
    mode: ExecutionMode
    adv_frac: float
    slippage_bps: float
    fee_buy: np.ndarray
    fee_sell: np.ndarray
    limit_offset_bps: float = 0.0
    day_only: bool = True

    @property
    def slippage_decimal(self) -> float:
        return self.slippage_bps / 10_000.0


@dataclass
class OrderFill:
    """단일 심볼 체결 결과.

    requested_qty: 요청 수량
    filled_qty: 실제 체결 수량(부분 체결 가능)
    fill_price: 체결가격(None은 미체결)
    limit_price: LIMIT 지정가(시장가는 None)
    reason: 미체결/클립 사유 키(예: limit_not_reached, adv_cap_zero, insufficient_cash, no_position)
    """
    symbol: str
    side: str
    requested_qty: int
    filled_qty: int
    fill_price: float | None
    limit_price: float | None
    reason: str | None


class StockTradingEnv(gym.Env):
    """
    Parameters:
    ----------
    df : pd.DataFrame
        주식 시장 데이터를 포함하는 데이터프레임.
    stock_dim : int
        거래할 주식 종목의 수.
    hmax : int
        한 번에 거래할 수 있는 최대 주식 수.
    initial_amount : int
        에이전트가 거래를 시작할 때 보유하고 있는 초기 금액.
    num_stock_shares : list[int]
        각 주식 종목에 대해 에이전트가 보유하고 있는 주식 수의 리스트.
    buy_cost_pct : list[float]
        매수 거래에 따른 비용 비율.
    sell_cost_pct : list[float]
        매도 거래에 따른 비용 비율.
    reward_scaling : float
        보상의 스케일링 인자.
    state_space : int
        관측 공간의 차원.
    action_space : int
        행동 공간의 차원.
    tech_indicator_list : list[str]
        기술적 지표의 이름을 포함하는 리스트.
    turbulence_threshold : float, optional
        터뷸런스 임계값. 이 값 이상이면 시장이 불안정하다고 간주.
    risk_indicator_col : str, default "turbulence"
        위험 지표의 컬럼 이름.
    make_plots : bool, default False
        시뮬레이션 중에 플롯을 만들지 여부.
    print_verbosity : int, default 10
        출력의 상세 수준을 조절하는 정수.
    day : int, default 0
        에피소드 시작일을 나타내는 정수.
    initial : bool, default True
        초기화 여부. 초기화할 경우 True.
    previous_state : list, default []
        이전 상태를 기억하는 리스트.
    model_name : str, default ""
        모델 이름.
    mode : str, default ""
        현재 모드 (예: 훈련, 테스트 등).
    iteration : str, default ""
        현재 반복 횟수.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        stock_dim: int,
        hmax: int,
        initial_amount: int,
        num_stock_shares: list[int],
        buy_cost_pct: list[float],
        sell_cost_pct: list[float],
        reward_scaling: float,
        state_space: int,
        action_space: int,
        tech_indicator_list: list[str],
        turbulence_threshold=None,
        risk_indicator_col=config.RISK_INDICATOR_COL,
        make_plots: bool = False,
        print_verbosity=10,
        day=0,
        initial=True,
        previous_state=[],
        model_name="",
        mode="",
        iteration="",
        exec_mode: str = ExecutionMode.MOO.value,
        adv_fraction: float = 0.2,
        limit_offset_bps: float = 0.0,
        slippage_bps: float = 0.0,
        day_order_only: bool = True,
        random_start: bool = False,
        portfolio_config: PortfolioInitParams | None = None,
    ):
        self.day = day
        self.df = df
        self.stock_dim = stock_dim
        self.hmax = hmax
        self.base_num_stock_shares = list(num_stock_shares)
        self.num_stock_shares = list(num_stock_shares)
        self.base_initial_amount = initial_amount
        self.initial_amount = initial_amount  # refreshed each episode
        self.buy_cost_pct = list(buy_cost_pct)
        self.sell_cost_pct = list(sell_cost_pct)
        self._fee_buy = np.asarray(self.buy_cost_pct, dtype=float)
        self._fee_sell = np.asarray(self.sell_cost_pct, dtype=float)
        mode_value = exec_mode if isinstance(exec_mode, ExecutionMode) else ExecutionMode(exec_mode)
        adv_value = 0.0 if adv_fraction is None else float(adv_fraction)
        slip_value = 0.0 if slippage_bps is None else float(slippage_bps)
        self.execution_spec = ExecutionSpec(
            mode=mode_value,
            adv_frac=float(max(adv_value, 0.0)),
            slippage_bps=float(max(slip_value, 0.0)),
            fee_buy=self._fee_buy,
            fee_sell=self._fee_sell,
            limit_offset_bps=float(limit_offset_bps if limit_offset_bps is not None else 0.0),
            day_only=bool(day_order_only),
        )
        self._last_fills: list[OrderFill] = []
        self._unique_day_indices = np.array(sorted(self.df.index.unique()))
        self._n_trade_days = len(self._unique_day_indices)
        self._max_day_index = (
            int(self._unique_day_indices[-1]) if self._n_trade_days > 0 else 0
        )
        self.reward_scaling = reward_scaling
        self.state_space = state_space
        self.action_space = action_space
        self.tech_indicator_list = tech_indicator_list
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_space,))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_space,)
        )
        initial_slice = self.df.loc[self.day, :]
        if isinstance(initial_slice, pd.Series):
            initial_frame = initial_slice.to_frame().T
        else:
            initial_frame = initial_slice.copy()
        if "tic" in initial_frame.columns:
            self.ticker_list = initial_frame["tic"].tolist()
        else:
            self.ticker_list = [str(initial_frame.iloc[0].get("tic", "TIC0"))]
        self.data = self._ensure_ticker_order(initial_frame)
        self.terminal = False
        self.make_plots = make_plots
        self.print_verbosity = print_verbosity
        self.turbulence_threshold = turbulence_threshold
        self.risk_indicator_col = risk_indicator_col
        self.initial = initial
        self.previous_state = previous_state
        self.model_name = model_name
        self.mode = mode
        self.iteration = iteration
        self.portfolio_config = portfolio_config
        self.random_start = bool(random_start and portfolio_config is not None)
        self._tracking_enabled = not (self.mode == "inference")
        self._episode_summaries: List[dict] = []
        # initalize state
        self._seed()
        self.state = self._initiate_state()
        # initialize reward tracking and state memories
        self.reward = 0
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.episode = 0
        self.asset_memory = [
            self.initial_amount
            + np.sum(
                np.array(self.num_stock_shares)
                * np.array(self.state[1 : 1 + self.stock_dim])
            )
        ]
        self.rewards_memory = []
        self.actions_memory = []
        self.state_memory = []
        self.date_memory = [self._get_date()]

    def _ensure_ticker_order(self, frame: pd.DataFrame) -> pd.DataFrame:
        """
        Return the provided day frame ordered to match the environment's ticker ordering.
        Ensures downstream vector operations align with the agent's state representation.
        """

        if isinstance(frame, pd.Series):
            frame = frame.to_frame().T
        if self.stock_dim <= 1 or "tic" not in frame.columns:
            return frame.reset_index(drop=True)
        if not hasattr(self, "ticker_list"):
            return frame.reset_index(drop=True)
        ordered = frame.set_index("tic")
        missing = [tic for tic in self.ticker_list if tic not in ordered.index]
        if missing:
            raise KeyError(f"Missing tickers {missing} for day frame alignment")
        ordered = ordered.loc[self.ticker_list]
        return ordered.reset_index()

    def _get_day_frame(self, day_idx: int) -> pd.DataFrame:
        """Fetch a day's DataFrame in the canonical ticker order."""

        day_slice = self.df.loc[day_idx, :]
        if isinstance(day_slice, pd.Series):
            frame = day_slice.to_frame().T
        else:
            frame = day_slice.copy()
        return self._ensure_ticker_order(frame)

    def _extract_column_array(self, frame: pd.DataFrame, column: str) -> np.ndarray:
        """Extract the requested column as a float numpy array."""

        return frame[column].to_numpy(dtype=float, copy=True)

    def _current_holdings_array(self) -> np.ndarray:
        return np.array(
            self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)],
            dtype=float,
        )

    @staticmethod
    def _compute_total_asset(cash: float, prices: np.ndarray, holdings: np.ndarray) -> float:
        return float(cash + float(np.dot(prices, holdings)))

    def _adv_cap_qty(self, volume: float) -> int | None:
        adv = self.execution_spec.adv_frac
        if adv is None:
            return None
        adv = max(float(adv), 0.0)
        if adv == 0.0:
            return 0
        return max(int(np.floor(float(volume) * adv)), 0)

    def _compute_limit_price(self, reference_price: float, side: str) -> float:
        offset_ratio = self.execution_spec.limit_offset_bps / 10_000.0
        if side.upper() == "BUY":
            return reference_price * (1.0 - offset_ratio)
        return reference_price * (1.0 + offset_ratio)

    def _execute_orders(
        self,
        desired_qtys: np.ndarray,
        current_close: np.ndarray,
        next_frame: pd.DataFrame,
    ) -> tuple[np.ndarray, list[OrderFill]]:
        """t→t+1 체결 시뮬레이션(MOO/LIMIT_OHLC 규칙).

        - MOO: 개장가(open)로 요청 수량을 체결(ADV/현금/보유로 클립)
        - LIMIT_OHLC: 전일 종가±offset으로 지정가 산출 후,
            BUY: open≤limit → open 체결; else low≤limit → limit 체결; else 미체결
            SELL: open≥limit → open 체결; else high≥limit → limit 체결; else 미체결
        - 이후 수수료/슬리피지 반영, 현금/보유/ADV 캡에 의해 부분체결 가능
        """

        fills = np.zeros_like(desired_qtys, dtype=int)
        records: list[OrderFill] = []

        open_prices = self._extract_column_array(next_frame, "open")
        high_prices = self._extract_column_array(next_frame, "high")
        low_prices = self._extract_column_array(next_frame, "low")
        volumes = self._extract_column_array(next_frame, "volume")

        cash_balance = float(self.state[0])
        holdings = self._current_holdings_array()
        slippage = self.execution_spec.slippage_decimal

        for idx, raw_qty in enumerate(desired_qtys):
            qty = int(raw_qty)
            if qty == 0:
                continue

            side = "BUY" if qty > 0 else "SELL"
            requested_qty = abs(qty)
            ticker = self.ticker_list[idx] if idx < len(self.ticker_list) else str(idx)
            limit_price = None
            fill_price = None
            filled_qty = 0
            reason = None

            adv_cap = self._adv_cap_qty(volumes[idx])
            if adv_cap == 0:
                reason = "adv_cap_zero"
            else:
                effective_qty = requested_qty
                if adv_cap is not None:
                    effective_qty = min(effective_qty, adv_cap)
                if effective_qty <= 0:
                    reason = "adv_cap_limit"
                else:
                    if self.execution_spec.mode is ExecutionMode.MOO:
                        fill_price = float(open_prices[idx])
                        filled_qty = effective_qty
                    else:
                        reference_price = float(current_close[idx])
                        limit_price = float(self._compute_limit_price(reference_price, side))
                        open_px = float(open_prices[idx])
                        high_px = float(high_prices[idx])
                        low_px = float(low_prices[idx])
                        if side == "BUY":
                            if open_px <= limit_price:
                                fill_price = open_px
                                filled_qty = effective_qty
                            elif low_px <= limit_price:
                                fill_price = limit_price
                                filled_qty = effective_qty
                            else:
                                reason = "limit_not_reached"
                        else:
                            if open_px >= limit_price:
                                fill_price = open_px
                                filled_qty = effective_qty
                            elif high_px >= limit_price:
                                fill_price = limit_price
                                filled_qty = effective_qty
                            else:
                                reason = "limit_not_reached"

                    if filled_qty > 0 and reason is None:
                        if side == "BUY":
                            fee = float(self._fee_buy[idx])
                            total_multiplier = 1.0 + fee + slippage
                            max_cash_qty = int(
                                np.floor(cash_balance / (fill_price * total_multiplier))
                            )
                            if max_cash_qty <= 0:
                                filled_qty = 0
                                reason = "insufficient_cash"
                            else:
                                filled_qty = min(filled_qty, max_cash_qty)
                                if filled_qty <= 0:
                                    reason = "insufficient_cash"
                        else:
                            fee = float(self._fee_sell[idx])
                            total_multiplier = 1.0 - (fee + slippage)
                            if total_multiplier <= 0:
                                filled_qty = 0
                                reason = "invalid_costs"
                            else:
                                available = int(np.floor(holdings[idx]))
                                filled_qty = min(filled_qty, available)
                                if filled_qty <= 0:
                                    reason = "no_position"

            if filled_qty > 0 and reason is None:
                if side == "BUY":
                    fee = float(self._fee_buy[idx])
                    total_multiplier = 1.0 + fee + slippage
                    cash_delta = fill_price * filled_qty * total_multiplier
                    cash_balance -= cash_delta
                    holdings[idx] += filled_qty
                    self.cost += fill_price * filled_qty * (fee + slippage)
                    fills[idx] = filled_qty
                else:
                    fee = float(self._fee_sell[idx])
                    total_multiplier = 1.0 - (fee + slippage)
                    cash_delta = fill_price * filled_qty * total_multiplier
                    cash_balance += cash_delta
                    holdings[idx] -= filled_qty
                    self.cost += fill_price * filled_qty * (fee + slippage)
                    fills[idx] = -filled_qty
                self.trades += 1
                records.append(
                    OrderFill(
                        symbol=ticker,
                        side=side,
                        requested_qty=requested_qty,
                        filled_qty=filled_qty,
                        fill_price=fill_price,
                        limit_price=limit_price,
                        reason=None,
                    )
                )
            else:
                records.append(
                    OrderFill(
                        symbol=ticker,
                        side=side,
                        requested_qty=requested_qty,
                        filled_qty=0,
                        fill_price=None,
                        limit_price=limit_price,
                        reason=reason,
                    )
                )

        self.state[0] = float(cash_balance)
        holdings_int = holdings.astype(int)
        self.state[
            (self.stock_dim + 1) : (self.stock_dim * 2 + 1)
        ] = holdings_int.tolist()
        self.num_stock_shares = holdings_int.tolist()
        self._last_fills = records
        return fills.astype(int), records
        #         self.logger = Logger('results',[CSVOutputFormat])
        # self.reset()

    def _sell_stock(self, index, action):
        """
        주식 매도 로직을 수행하는 함수.

        Parameters:
        index : int
            현재 매도하려는 주식의 인덱스.
        action : int
            매도할 주식의 수. 양수인 경우 매수를 의미하며, 음수인 경우 매도를 의미함.

        Returns:
        sell_num_shares : int
            실제 매도한 주식의 수.

        주의: action 값이 음수일 경우만 매도가 실행됨.
        """

        def _do_sell_normal():
            # 주식이 매도 가능한 상태인지 확인 (기술 지표를 통해 매매 가능 여부를 판단)
            if self.state[index + 2 * self.stock_dim + 1] != True:
                # 주식의 현재 가격이 0 이상일 때에만 매도 가능
                if self.state[index + self.stock_dim + 1] > 0:
                    # 현재 보유한 주식 수와 매도하려는 주식 수 중 더 작은 값으로 매도 수량 결정
                    sell_num_shares = min(
                        abs(action), self.state[index + self.stock_dim + 1]
                    )
                    # 매도 금액 계산 (매도 수량 * 주식 가격 * (1 - 매도 비용 비율))
                    sell_amount = (
                        self.state[index + 1]
                        * sell_num_shares
                        * (1 - self.sell_cost_pct[index])
                    )
                    # 현금 자산 업데이트 (매도 금액 추가)
                    self.state[0] += sell_amount

                    # 주식 보유 수 업데이트 (매도 수량만큼 감소)
                    self.state[index + self.stock_dim + 1] -= sell_num_shares
                    # 매도 비용 업데이트
                    self.cost += (
                        self.state[index + 1]
                        * sell_num_shares
                        * self.sell_cost_pct[index]
                    )
                    # 거래 수 업데이트
                    self.trades += 1
                else:
                    # 매도할 주식이 없는 경우, 매도 수량은 0
                    sell_num_shares = 0
            else:
                # 주식이 매도 불가능한 상태인 경우, 매도 수량은 0
                sell_num_shares = 0

            return sell_num_shares

        # 시장의 불안정성이 임계값 이상인지에 따라 행동을 결정
        if self.turbulence_threshold is not None:
            # 시장 불안정성이 높을 때
            if self.turbulence >= self.turbulence_threshold:
                # 주식 가격이 0 이상이고, 보유 주식이 있을 때 모든 주식 매도
                if self.state[index + 1] > 0 and self.state[index + self.stock_dim + 1] > 0:
                    # 보유한 모든 주식 매도
                    sell_num_shares = self.state[index + self.stock_dim + 1]
                    sell_amount = (
                        self.state[index + 1]
                        * sell_num_shares
                        * (1 - self.sell_cost_pct[index])
                    )
                    # 현금 자산 업데이트 (매도 금액 추가)
                    self.state[0] += sell_amount
                    # 주식 보유 수를 0으로 설정
                    self.state[index + self.stock_dim + 1] = 0
                    # 매도 비용 업데이트
                    self.cost += (
                        self.state[index + 1]
                        * sell_num_shares
                        * self.sell_cost_pct[index]
                    )
                    # 거래 수 업데이트
                    self.trades += 1
                else:
                    # 매도할 주식이 없는 경우, 매도 수량은 0
                    sell_num_shares = 0
            else:
                # 시장이 비교적 안정적일 때 정상적인 매도 로직 수행
                sell_num_shares = _do_sell_normal()
        else:
            # 불안정성 임계값이 설정되어 있지 않을 때 정상적인 매도 로직 수행
            sell_num_shares = _do_sell_normal()

        return sell_num_shares

    def _buy_stock(self, index, action):
        """
        주식 매수 로직을 수행하는 함수입니다.

        Parameters:
        index : int
            현재 매수하려는 주식의 인덱스입니다.
        action : int
            매수할 주식의 수입니다. 양수일 경우 매수를 의미합니다.

        Returns:
        buy_num_shares : int
            실제 매수한 주식의 수입니다.

        이 함수는 에이전트가 결정한 액션에 따라 주식을 매수합니다. 주식 매수가 가능한지 여부를 
        체크하고, 가능할 경우에만 매수를 진행합니다.
        """

        def _do_buy():
            # 주식 매수 가능 여부 체크 (기술 지표 등을 통해 판단)
            if self.state[index + 2 * self.stock_dim + 1] != True:
                # 주식 가격이 0 이상인 경우에만 매수를 진행합니다 (데이터가 누락되지 않은 날짜에만 매수)
                available_amount = self.state[0] // (
                    self.state[index + 1] * (1 + self.buy_cost_pct[index])
                )  # 매수 가능한 주식 수를 계산할 때 거래 비용을 고려합니다.

                # 실제로 매수할 수 있는 주식 수를 계산합니다 (가용 금액과 액션 중 작은 값)
                buy_num_shares = min(available_amount, action)
                # 매수 금액을 계산합니다 (주식 가격 * 매수 수량 * (1 + 매수 비용 비율))
                buy_amount = (
                    self.state[index + 1]
                    * buy_num_shares
                    * (1 + self.buy_cost_pct[index])
                )
                # 현금 잔고를 업데이트합니다 (매수 금액만큼 감소)
                self.state[0] -= buy_amount

                # 보유 주식 수를 업데이트합니다 (매수 수량만큼 증가)
                self.state[index + self.stock_dim + 1] += buy_num_shares

                # 매수 비용을 업데이트합니다.
                self.cost += (
                    self.state[index + 1] * buy_num_shares * self.buy_cost_pct[index]
                )
                # 거래 횟수를 증가시킵니다.
                self.trades += 1
            else:
                # 매수 불가능 상태일 경우, 매수 수량은 0입니다.
                buy_num_shares = 0

            return buy_num_shares

        # 시장의 불안정성이 정의된 임계값 이하일 때만 매수를 진행합니다.
        if self.turbulence_threshold is None:
            # 불안정성 임계값이 설정되지 않은 경우, 정상 매수 로직을 수행합니다.
            buy_num_shares = _do_buy()
        else:
            # 불안정성 임계값이 설정된 경우, 불안정성이 임계값 이하일 때만 매수를 수행합니다.
            if self.turbulence < self.turbulence_threshold:
                buy_num_shares = _do_buy()
            else:
                # 불안정성이 임계값 이상일 경우, 매수를 진행하지 않고 매수 수량을 0으로 설정합니다.
                buy_num_shares = 0

        return buy_num_shares

    def _make_plot(self):
        plt.plot(self.asset_memory, "r")
        results_dir = Path(config.RESULTS_DIR)
        results_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(results_dir / f"account_value_trade_{self.episode}.png")
        plt.close()

    def step(self, actions):
        """
        환경에서 한 스텝을 진행하고 결과를 반환하는 메서드입니다.

        Parameters:
        actions : np.array
            에이전트가 선택한 행동들의 배열로, 각 주식에 대한 매수 또는 매도 액션을 포함합니다.

        Returns:
        self.state : np.array
            새로운 상태의 배열입니다.
        self.reward : float
            이번 스텝에서 얻은 보상의 양입니다.
        self.terminal : bool
            에피소드가 종료되었는지 여부를 나타냅니다. 모든 거래일이 끝나면 True가 됩니다.
        False : bool
            추가 정보 제공용, 여기서는 사용되지 않으므로 항상 False를 반환합니다.
        {} : dict
            추가 정보 제공용, 여기서는 사용되지 않으므로 빈 딕셔너리를 반환합니다.

        이 메서드는 에이전트의 행동 배열을 받아 각 주식에 대한 매수/매도를 진행하고,
        새로운 상태, 보상, 종료 여부 등을 계산합니다.
        """
        # 에피소드 종료 여부 판단: 현재 일자가 데이터의 유일한 인덱스 개수보다 많거나 같으면 종료
        self.terminal = self.day >= self._n_trade_days - 1

        # 만약 에피소드가 종료되었다면 (self.terminal == True)
        if self.terminal:
            if not self._tracking_enabled:
                info = {"fills": [fill.__dict__ for fill in getattr(self, "_last_fills", [])]}
                return self.state, self.reward, self.terminal, False, info
            # 에피소드 종료 시 수행할 로직
            
            # 포트폴리오의 총 자산 가치를 계산
            end_total_asset = self.state[0] + sum(
                np.array(self.state[1 : (self.stock_dim + 1)]) *
                np.array(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
            )
            
            # 총 자산 가치의 시계열 데이터를 데이터프레임으로 저장
            df_total_value = pd.DataFrame(self.asset_memory)
            
            # 에피소드 전체에서의 총 보상을 계산 (초기 자산 대비 총 자산의 증가)
            tot_reward = end_total_asset - self.asset_memory[0]
            
            # 계산된 총 자산 가치와 날짜를 데이터프레임에 추가
            df_total_value.columns = ["account_value"]
            df_total_value["date"] = self.date_memory
            
            # 일일 수익률을 계산하고 데이터프레임에 추가
            df_total_value["daily_return"] = df_total_value["account_value"].pct_change(1)
            
            # 샤프 지수 계산 (일일 수익률의 평균 / 일일 수익률의 표준편차)
            sharpe = None
            if df_total_value["daily_return"].std() != 0:
                sharpe = (
                    (252**0.5) *  # 연간화 계수
                    df_total_value["daily_return"].mean() /
                    df_total_value["daily_return"].std()
                )
            
            # 보상의 시계열 데이터를 데이터프레임으로 저장
            df_rewards = pd.DataFrame(self.rewards_memory)
            df_rewards.columns = ["account_rewards"]
            df_rewards["date"] = self.date_memory[:-1]
            
            # 설정된 print_verbosity에 따라 로깅 정보 출력
            summary = {
                "day": int(self.day),
                "episode": int(self.episode),
                "begin_total_asset": float(self.asset_memory[0]),
                "end_total_asset": float(end_total_asset),
                "total_reward": float(tot_reward),
                "total_cost": float(self.cost),
                "total_trades": int(self.trades),
                "sharpe": float(sharpe) if sharpe is not None else None,
                "mode": self.mode,
                "model_name": self.model_name,
            }
            self._episode_summaries.append(summary)
            if len(self._episode_summaries) > 10000:
                self._episode_summaries = self._episode_summaries[-5000:]

            if (
                self.print_verbosity is not None
                and self.print_verbosity > 0
                and self.episode % self.print_verbosity == 0
            ):
                log_parts = [
                    f"day={summary['day']}",
                    f"episode={summary['episode']}",
                    f"begin_total_asset={summary['begin_total_asset']:.2f}",
                    f"end_total_asset={summary['end_total_asset']:.2f}",
                    f"total_reward={summary['total_reward']:.2f}",
                    f"total_cost={summary['total_cost']:.2f}",
                    f"total_trades={summary['total_trades']}",
                ]
                if summary["sharpe"] is not None:
                    log_parts.append(f"sharpe={summary['sharpe']:.3f}")
                LOGGER.info("episode_summary|%s", "|".join(log_parts))
                
            # 모델 이름과 모드가 설정되었다면 결과를 파일로 저장
            if (self.model_name != "") and (self.mode != ""):
                df_actions = self.save_action_memory()
                results_dir = Path(config.RESULTS_DIR)
                results_dir.mkdir(parents=True, exist_ok=True)
                LOGGER.info(
                    "Saving action/value logs to %s (mode=%s, model=%s, iter=%s)",
                    results_dir,
                    self.mode,
                    self.model_name,
                    self.iteration,
                )
                df_actions.to_csv(
                    results_dir
                    / "actions_{}_{}_{}.csv".format(
                        self.mode, self.model_name, self.iteration
                    ),
                    index=False,
                )
                df_total_value.to_csv(
                    results_dir
                    / "account_value_{}_{}_{}.csv".format(
                        self.mode, self.model_name, self.iteration
                    ),
                    index=False,
                )
                df_rewards.to_csv(
                    results_dir
                    / "account_rewards_{}_{}_{}.csv".format(
                        self.mode, self.model_name, self.iteration
                    ),
                    index=False,
                )
                plt.plot(self.asset_memory, "r")
                plt.savefig(
                    results_dir
                    / "account_value_{}_{}_{}.png".format(
                        self.mode, self.model_name, self.iteration
                    )
                )
                plt.close()
            else:
                LOGGER.debug(
                    "Skipping result save (mode=%s, model=%s)",
                    self.mode,
                    self.model_name,
                )

            info = {"fills": [fill.__dict__ for fill in getattr(self, "_last_fills", [])]}
            return self.state, self.reward, self.terminal, False, info

        else:
            action_array = np.asarray(actions, dtype=float) * self.hmax
            desired_qtys = action_array.astype(int)

            if self.turbulence_threshold is not None and self.turbulence >= self.turbulence_threshold:
                desired_qtys = np.array([-self.hmax] * self.stock_dim, dtype=int)

            current_close = self._extract_column_array(self.data, "close")
            current_holdings = self._current_holdings_array()
            cash_before = float(self.state[0])
            begin_total_asset = self._compute_total_asset(cash_before, current_close, current_holdings)

            next_day_index = self.day + 1
            next_frame = self._get_day_frame(next_day_index)
            fills, fill_records = self._execute_orders(desired_qtys, current_close, next_frame)
            if self._tracking_enabled:
                self.actions_memory.append(fills.astype(int).tolist())

            self.day = next_day_index
            self.data = next_frame

            if self.turbulence_threshold is not None:
                if self.stock_dim == 1:
                    self.turbulence = float(self.data[self.risk_indicator_col].iloc[0])
                else:
                    self.turbulence = float(self.data[self.risk_indicator_col].values[0])

            self.state = self._update_state()

            end_close = self._extract_column_array(self.data, "close")
            end_holdings = self._current_holdings_array()
            end_total_asset = self._compute_total_asset(float(self.state[0]), end_close, end_holdings)
            self.num_stock_shares = end_holdings.astype(int).tolist()

            raw_reward = end_total_asset - begin_total_asset
            if self._tracking_enabled:
                self.asset_memory.append(end_total_asset)
                self.date_memory.append(self._get_date())
                self.rewards_memory.append(raw_reward)
                self.state_memory.append(self.state)
            self.reward = raw_reward * self.reward_scaling

            info = {
                "fills": [fill.__dict__ for fill in fill_records],
                "begin_total_asset": begin_total_asset,
                "end_total_asset": end_total_asset,
            }

            return self.state, self.reward, self.terminal, False, info

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ):
        """
        환경을 초기 상태로 리셋하는 메서드입니다.

        Parameters:
        seed : int, optional
            난수 생성기 시드 값입니다.
        options : dict, optional
            리셋 옵션을 지정하는 딕셔너리입니다.

        Returns:
        self.state : np.array
            리셋된 환경의 초기 상태 배열입니다.
        
        이 메서드는 새로운 에피소드를 시작하기 전에 호출됩니다. 현재 일자를 0으로 설정하고
        데이터프레임에서 해당 일자의 데이터를 불러와 상태를 초기화합니다.
        """

        # 거래를 시작할 날짜를 0으로 설정하여 새로운 에피소드를 시작합니다.
        self.day = 0
        # 현재 날짜에 해당하는 데이터를 데이터프레임에서 가져옵니다.
        initial_slice = self.df.loc[self.day, :]
        if isinstance(initial_slice, pd.Series):
            initial_frame = initial_slice.to_frame().T
        else:
            initial_frame = initial_slice.copy()
        if "tic" in initial_frame.columns:
            self.ticker_list = initial_frame["tic"].tolist()
        else:
            self.ticker_list = [str(initial_frame.iloc[0].get("tic", "TIC0"))]
        self.data = self._ensure_ticker_order(initial_frame)
        self.initial_amount = self.base_initial_amount
        self.num_stock_shares = list(self.base_num_stock_shares)
        # 상태를 초기 상태로 설정하는 내부 메서드를 호출합니다.
        self.state = self._initiate_state()

        # 만약 초기 상태에서 시작하는 경우
        if self.initial:
            # 초기 자산 메모리를 설정합니다. 이는 초기 현금과 각 주식의 초기 주식 수와 가격을 기반으로 계산됩니다.
            self.asset_memory = [
                self.initial_amount
                + np.sum(
                    np.array(self.num_stock_shares)
                    * np.array(self.state[1 : 1 + self.stock_dim])
                )
            ]
        # 만약 이전 상태에서 시작하는 경우
        else:
            # 이전 에피소드의 마지막 총 자산을 계산합니다.
            previous_total_asset = self.previous_state[0] + sum(
                np.array(self.state[1 : (self.stock_dim + 1)])
                * np.array(
                    self.previous_state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)]
                )
            )
            # 자산 메모리를 이전 에피소드의 마지막 총 자산으로 설정합니다.
            self.asset_memory = [previous_total_asset]

        # 터뷸런스와 비용, 거래 수를 0으로 설정합니다.
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        # 에피소드 종료 여부를 False로 설정합니다.
        self.terminal = False
        # 에피소드를 나타내는 숫자를 1 증가시킵니다.
        self.episode += 1

        # 보상과 행동 메모리를 비우고 현재 날짜를 메모리에 추가합니다.
        self.rewards_memory = []
        self.actions_memory = []
        self.date_memory = [self._get_date()]
        self._last_fills = []

        # 초기 상태를 반환합니다.
        return self.state, {}

    def render(self, mode="human", close=False):
        return self.state

    def _initiate_state(self):
        """
        환경의 초기 상태 또는 이전 상태를 기반으로 현재 상태를 설정하는 메서드입니다.

        Returns:
        state : list
            현재 상태를 나타내는 리스트로, 현금 잔액, 각 주식의 가격, 보유 주식 수,
            그리고 선택된 기술적 지표들의 값을 포함합니다.

        이 메서드는 reset 메서드에 의해 호출되며, 에이전트의 초기 상태 또는 이전 상태를 준비합니다.
        """

        if self.initial:
            self.initial_amount = self.base_initial_amount
            self.num_stock_shares = list(self.base_num_stock_shares)
            if self.random_start and self.portfolio_config is not None:
                cash, shares = self._sample_random_portfolio()
                self.initial_amount = cash
                self.num_stock_shares = shares

        # 환경이 처음 시작할 때의 상태를 설정합니다.
        if self.initial:
            # 단일 주식이 아닌 여러 주식에 대한 상태를 초기화할 경우
            if len(self.df.tic.unique()) > 1:
                # 초기 자본을 포함하여 상태 리스트를 구성합니다.
                state = (
                    [self.initial_amount]  # 시작할 때 보유 현금
                    + self.data.close.values.tolist()  # 모든 주식의 가격
                    + self.num_stock_shares  # 에이전트가 초기에 보유한 각 주식의 수
                    # 기술적 지표들의 값들을 추가합니다.
                    + sum(
                        (
                            self.data[tech].values.tolist()
                            for tech in self.tech_indicator_list
                        ),
                        [],
                    )
                )
            else:
                # 단일 주식에 대한 상태를 초기화할 경우
                state = (
                    [self.initial_amount]  # 시작할 때 보유 현금
                    + [self.data.close]  # 주식의 가격
                    + list(self.num_stock_shares)  # 초기 보유 주식 수를 반영
                    # 기술적 지표들의 값들을 추가합니다.
                    + sum(([self.data[tech]] for tech in self.tech_indicator_list), [])
                )
        else:
            # 이전 상태를 사용하여 상태를 설정할 경우
            if len(self.df.tic.unique()) > 1:
                # 여러 주식에 대한 상태를 이전 상태에서 업데이트합니다.
                state = (
                    [self.previous_state[0]]  # 이전 자본
                    + self.data.close.values.tolist()  # 모든 주식의 가격
                    + self.previous_state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)]  # 이전에 보유한 각 주식의 수
                    # 기술적 지표들의 값들을 추가합니다.
                    + sum(
                        (
                            self.data[tech].values.tolist()
                            for tech in self.tech_indicator_list
                        ),
                        [],
                    )
                )
            else:
                # 단일 주식에 대한 상태를 이전 상태에서 업데이트합니다.
                state = (
                    [self.previous_state[0]]  # 이전 자본
                    + [self.data.close]  # 주식의 가격
                    + self.previous_state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)]  # 이전에 보유한 주식의 수
                    # 기술적 지표들의 값들을 추가합니다.
                    + sum(([self.data[tech]] for tech in self.tech_indicator_list), [])
                )

        # 설정된 상태를 반환합니다.
        return state

    def _current_price_snapshot(self):
        if len(self.df.tic.unique()) > 1:
            tickers = self.data.tic.values.tolist()
            prices = self.data.close.values.tolist()
        else:
            ticker = self.df.tic.unique()[0]
            tickers = [ticker]
            prices = [float(self.data.close)]
        return tickers, prices

    def _sample_random_portfolio(self):
        cfg = self.portfolio_config
        if cfg is None:
            return float(self.base_initial_amount), list(self.base_num_stock_shares)

        tickers, prices = self._current_price_snapshot()
        if not tickers:
            return float(cfg.initial_cash), list(self.base_num_stock_shares)

        base_rng = getattr(self, "np_random", None)
        if base_rng is None:
            try:
                base_rng = np.random.default_rng()
            except AttributeError:  # pragma: no cover - fallback for older numpy
                base_rng = np.random.RandomState()

        rand_int = base_rng.integers if hasattr(base_rng, "integers") else base_rng.randint
        rand_choice = base_rng.choice
        rand_perm = base_rng.permutation

        min_positions = max(1, min(cfg.min_positions, len(tickers)))
        max_positions = max(min_positions, min(cfg.max_positions, len(tickers)))
        num_positions = int(rand_int(min_positions, max_positions + 1))
        indices = np.arange(len(tickers))
        chosen = rand_choice(indices, size=num_positions, replace=False)
        per_ticker_cap = cfg.per_ticker_value_cap_ratio * cfg.target_stock_value

        share_list = [0] * len(tickers)
        actual_value = 0.0
        remaining_value = cfg.target_stock_value

        for idx in rand_perm(chosen):
            if remaining_value <= 0:
                break
            price = prices[idx]
            if price <= 0:
                continue
            cap_value = min(per_ticker_cap, remaining_value)
            max_shares = int(cap_value // price)
            if max_shares <= 0:
                continue
            shares = int(rand_int(1, max_shares + 1))
            share_list[idx] += int(shares)
            value = shares * price
            actual_value += value
            remaining_value = max(0.0, cfg.target_stock_value - actual_value)

        # If no shares allocated due to price constraints, fall back to holding cash only
        if all(share == 0 for share in share_list):
            return float(cfg.initial_cash), [0] * len(tickers)

        return float(cfg.initial_cash), share_list

    def _update_state(self):
        """
        현재 거래일의 주식 가격과 기술적 지표들의 값을 반영하여 환경의 상태를 업데이트하는 메서드입니다.

        Returns:
        state : list
            업데이트된 상태를 나타내는 리스트입니다. 현금 잔액, 각 주식의 가격, 보유 주식 수,
            그리고 선택된 기술적 지표들의 값을 포함합니다.

        이 메서드는 매 거래일 마다 호출되어 에이전트의 상태를 최신 정보로 업데이트합니다.
        """

        # 여러 주식에 대한 환경일 경우
        if len(self.df.tic.unique()) > 1:
            # 현재 보유 현금은 유지하고 주식 가격을 업데이트합니다.
            state = (
                [self.state[0]]  # 현재 보유 현금
                + self.data.close.values.tolist()  # 새로운 거래일의 각 주식 가격
                + list(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])  # 현재 보유 주식 수는 유지
                # 선택된 기술적 지표들의 새로운 값을 추가합니다.
                + sum(
                    (
                        self.data[tech].values.tolist()
                        for tech in self.tech_indicator_list
                    ),
                    [],
                )
            )

        # 단일 주식에 대한 환경일 경우
        else:
            # 현재 보유 현금은 유지하고 주식 가격을 업데이트합니다.
            state = (
                [self.state[0]]  # 현재 보유 현금
                + [self.data.close]  # 새로운 거래일의 주식 가격
                + list(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])  # 현재 보유 주식 수는 유지
                # 선택된 기술적 지표들의 새로운 값을 추가합니다.
                + sum(([self.data[tech]] for tech in self.tech_indicator_list), [])
            )

        # 업데이트된 상태를 반환합니다.
        return state

    def _get_date(self):
        if isinstance(self.data, pd.DataFrame):
            return self.data["date"].iloc[0]
        return self.data["date"]

    def save_state_memory(self):
        """
        거래 과정에서의 상태를 기록하여 메모리에 저장하는 메서드입니다.

        이 메서드는 각 스텝마다의 에이전트 상태를 데이터프레임 형태로 저장합니다.
        여러 주식을 다루는 경우와 단일 주식을 다루는 경우로 나뉘어 처리합니다.

        Returns:
        df_states : pd.DataFrame
            각 거래일에 대한 상태 정보를 포함하는 데이터프레임입니다.
        """

        # 거래일 리스트를 가져옵니다. 마지막 거래일은 제외합니다.
        date_list = self.date_memory[:-1]
        df_date = pd.DataFrame(date_list)
        df_date.columns = ["date"]

        # 상태 메모리에서 상태 리스트를 가져옵니다.
        state_list = self.state_memory

        # 여러 주식에 대한 처리: 여러 주식을 다루는 경우 해당되는 컬럼 이름을 지정합니다.
        # 제공된 데이터에 맞게 컬럼 이름을 설정합니다.  
        if len(self.df.tic.unique()) > 1:
            # 여러 주식을 다루는 경우의 상태 데이터프레임을 생성합니다.
            df_states = pd.DataFrame(
                state_list,
                columns=[
                    "cash",
                    # "Bitcoin_price", "Gold_price" 등의 컬럼은 예시로 사용되었습니다.
                    # 실제 데이터에 맞는 컬럼으로 대체합니다.
                    "close_price",
                    # "Bitcoin_num", "Gold_num" 등의 컬럼은 보유 주식 수를 나타냈습니다.
                    # "num_shares" 등 실제 데이터에 맞는 컬럼으로 대체할 수 있습니다.
                    "num_shares",
                    # "Bitcoin_Disable", "Gold_Disable" 등의 컬럼은 해당 주식의 거래 가능 여부를 나타냈습니다.
                    # 실제 사용하는 데이터에 따라 필요 없는 컬럼은 제거할 수 있습니다.
                    # 여기서는 예시로 든 컬럼을 제거하고, 실제 데이터의 특성에 맞는 컬럼 이름으로 대체합니다.
                    "volume",
                    "vix",
                    "turbulence",
                    # 기술적 지표 등 추가 컬럼
                    "sma_10",
                    "rsi",
                    # 나머지 필요한 컬럼을 추가합니다.
                ],
            )
            df_states.index = df_date.date
        else:
            # 단일 주식에 대한 처리: 상태 데이터프레임을 생성합니다.
            df_states = pd.DataFrame({"date": date_list, "states": state_list})

        return df_states

    def save_asset_memory(self):
        date_list = self.date_memory
        asset_list = self.asset_memory
        # print(len(date_list))
        # print(len(asset_list))
        df_account_value = pd.DataFrame(
            {"date": date_list, "account_value": asset_list}
        )
        return df_account_value

    def save_action_memory(self):
        if len(self.df.tic.unique()) > 1:
            # date and close price length must match actions length
            date_list = self.date_memory[:-1]
            df_date = pd.DataFrame(date_list)
            df_date.columns = ["date"]

            action_list = self.actions_memory
            df_actions = pd.DataFrame(action_list)
            df_actions.columns = self.data.tic.values
            df_actions.index = df_date.date
            # df_actions = pd.DataFrame({'date':date_list,'actions':action_list})
        else:
            date_list = self.date_memory[:-1]
            action_list = [
                actions[0] if isinstance(actions, (list, tuple, np.ndarray)) else actions
                for actions in self.actions_memory
            ]
            df_actions = pd.DataFrame({"date": date_list, "actions": action_list})
        return df_actions

    def pop_episode_summaries(self):
        summaries = list(self._episode_summaries)
        self._episode_summaries.clear()
        return summaries

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_sb_env(self, log_dir=None):
        e = DummyVecEnv([lambda: self])
        if log_dir is not None:
            log_path = Path(log_dir)
            log_path.mkdir(parents=True, exist_ok=True)
            e = VecMonitor(e, str(log_path))
        e = VecNormalize(e, norm_obs=True, norm_reward=True, clip_obs=10.0)
        obs = e.reset()
        return e, obs
