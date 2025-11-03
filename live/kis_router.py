from __future__ import annotations

"""KIS 주문 라우터(LiveBroker) — 최소 REST 연동 래퍼

지원 기능(모의/실전 공통):
  - place(): 현금 매수/매도 주문 `/uapi/domestic-stock/v1/trading/order-cash`
      TR-ID: (모의) VTTC0802U 매수 / VTTC0801U 매도
             (실전) TTTC0802U 매수 / TTTC0801U 매도
      헤더: authorization(Bearer), appkey, appsecret, tr_id, custtype=P, hashkey(POST 바디)
  - positions(): 보유 수량 조회 `/trading/inquire-balance` (TR-ID: VTTC8432R/TTTC8432R)
  - cash_available(): 매수가능 현금 조회 `/trading/inquire-psbl-order` (TR-ID: VTTC8434R/TTTC8434R)
  - quote_price(): 현재가 조회 `/quotations/inquire-price` (TR-ID: VFHKST01010100/FHKST01010100)

안전장치:
  - 레이트리밋 직렬화(_RateLimiter)
  - 401/429/5xx에서 1회 재시도(토큰 갱신/백오프) 후 실패 시 Fail-Closed 리턴
  - dry_run=True 시 네트워크 호출 없이 안전값 반환

주의: 본 패치는 문서/주석 보강만 포함하며 API 호출 로직은 변경하지 않았다.
"""

import threading
import time
from collections import deque
from typing import Deque, Dict, Optional

import requests

from execution.order_spec import OrderSpec
from kis_auth import get_or_load_access_token


class _RateLimiter:
    def __init__(self, rps: float) -> None:
        self.interval = 1.0 / max(rps, 1e-6)
        self._lock = threading.Lock()
        self._next_time = time.monotonic()

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            if now < self._next_time:
                time.sleep(self._next_time - now)
            self._next_time = time.monotonic() + self.interval


class LiveBroker:
    """KIS REST 주문 래퍼(모의/실전 공용)."""

    def __init__(
        self,
        base_url: str,
        app_key: str,
        app_secret: str,
        account_no: str,
        product_code: str,
        *,
        use_paper: bool = True,
        dry_run: bool = False,
        rate_limit_rps: float = 2.0,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.app_key = app_key
        self.app_secret = app_secret
        self.account_no = account_no
        self.product_code = product_code
        self.use_paper = use_paper
        self.dry_run = dry_run
        self.session = session or requests.Session()
        self._rate_limiter = _RateLimiter(rate_limit_rps)
        self._id_cache: Deque[str] = deque(maxlen=2048)
        env = "mock" if use_paper else "real"
        self._token_env = env
        self._access_token = get_or_load_access_token(env=env)

    def _headers(self, tr_id: str, hashkey: Optional[str] = None) -> Dict[str, str]:
        """공통 헤더 구성. POST 시 hashkey 포함.

        tr_id 예:
          - 주문: TTTC0802U/0801U (모의 VTTC...)
          - 현재가: FHKST01010100 (모의 VFHKST01010100)
        """
        headers = {
            "content-type": "application/json; charset=utf-8",
            "authorization": f"Bearer {self._access_token}",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
            "tr_id": tr_id,
            "custtype": "P",
        }
        if hashkey:
            headers["hashkey"] = hashkey
        return headers

    def _hash_body(self, body: Dict[str, object]) -> str:
        """요청 바디 해시키 생성 엔드포인트 호출."""
        url = f"{self.base_url}/uapi/hashkey"
        res = self.session.post(url, headers=self._headers("HASH"), json=body, timeout=5)
        res.raise_for_status()
        data = res.json()
        key = data.get("HASH") or data.get("hashkey")
        if not key:
            raise RuntimeError(f"Failed to obtain hashkey: {data}")
        return key

    def _ensure_token(self) -> None:
        self._access_token = get_or_load_access_token(env=self._token_env)

    def quote_price(self, symbol: str) -> Optional[float]:
        """현재가 조회.

        TR-ID: (모의) VFHKST01010100 / (실전) FHKST01010100
        실패 시 None 반환.
        """
        if self.dry_run:
            return None
        tr_id = "VFHKST01010100" if self.use_paper else "FHKST01010100"
        params = {
            "fid_cond_mrkt_div_code": "J",
            "fid_input_iscd": symbol,
        }
        try:
            attempts = 0
            while attempts < 2:
                self._rate_limiter.wait()
                res = self.session.get(
                    f"{self.base_url}/uapi/domestic-stock/v1/quotations/inquire-price",
                    headers=self._headers(tr_id),
                    params=params,
                    timeout=5,
                )
                if res.status_code == 401 and attempts == 0:
                    self._ensure_token()
                    attempts += 1
                    continue
                if not res.ok:
                    return None
                data = res.json()
                output = data.get("output") or {}
                price = output.get("stck_prpr") or output.get("STCK_PRPR")
                if price is None:
                    return None
                try:
                    return float(price)
                except (TypeError, ValueError):
                    return None
            return None
        except Exception:  # pragma: no cover
            return None

    def place(self, order: OrderSpec) -> Dict[str, object]:
        """현금 주문 제출(시장가/지정가).

        - TR-ID: 실전 TTTC0802U/0801U, 모의 VTTC0802U/0801U
        - 재시도 정책: 401/429/5xx에서 1회 재시도(401은 토큰 재발급)
        - 실패 시 Fail-Closed 응답 딕셔너리 반환
        - LIMIT 가격은 정수 문자열로 전송(ORD_UNPR)
        """
        if order.client_order_id and order.client_order_id in self._id_cache:
            return {
                "order_id": order.client_order_id,
                "accepted": False,
                "message": "duplicate_client_order_id",
            }

        if self.dry_run:
            report = {
                "order_id": order.client_order_id or "dry-run",
                "accepted": True,
                "message": "dry_run",
                "request_body": None,
                "response_body": None,
            }
            if order.client_order_id:
                self._id_cache.append(order.client_order_id)
            return report

        tr_id = "VTTC0802U" if (self.use_paper and order.side == "BUY") else "VTTC0801U"
        if not self.use_paper:
            tr_id = "TTTC0802U" if order.side == "BUY" else "TTTC0801U"

        ord_dvsn = "01" if order.order_type == "MARKET" else "00"
        limit_price_value = None if order.limit_price is None else int(order.limit_price)
        body = {
            "CANO": self.account_no,
            "ACNT_PRDT_CD": self.product_code,
            "PDNO": order.symbol,
            "ORD_DVSN": ord_dvsn,
            "ORD_QTY": str(order.qty),
            "ORD_UNPR": "0" if order.order_type == "MARKET" else str(limit_price_value),
            "CID": order.client_order_id,
        }

        attempts = 0
        accepted = False
        message = ""
        response_body = None
        while attempts < 2:
            try:
                self._rate_limiter.wait()
                hashkey = self._hash_body(body)
                res = self.session.post(
                    f"{self.base_url}/uapi/domestic-stock/v1/trading/order-cash",
                    headers=self._headers(tr_id, hashkey=hashkey),
                    json=body,
                    timeout=10,
                )
                if res.status_code == 401 and attempts == 0:
                    self._ensure_token()
                    attempts += 1
                    continue
                if res.status_code in {429, 500, 502, 503, 504} and attempts == 0:
                    time.sleep(0.5)
                    attempts += 1
                    continue
                accepted = res.ok
                if res.headers.get("content-type", "").startswith("application/json"):
                    response_body = res.json()
                    message = response_body.get("msg1", "") if isinstance(response_body, dict) else ""
                res.raise_for_status()
                break
            except Exception as exc:  # pragma: no cover
                if attempts == 0:
                    self._ensure_token()
                    attempts += 1
                    continue
                return {
                    "order_id": order.client_order_id,
                    "accepted": False,
                    "message": f"error:{exc}",
                    "request_body": body,
                }
        else:
            return {
                "order_id": order.client_order_id,
                "accepted": False,
                "message": "retry_exhausted",
                "request_body": body,
            }

        if order.client_order_id:
            self._id_cache.append(order.client_order_id)

        return {
            "order_id": order.client_order_id,
            "accepted": accepted,
            "message": message,
            "request_body": body,
            "response_body": response_body,
        }

    def cancel(self, order_id: str) -> Dict[str, object]:
        """주문 취소(스텁). 실제 구현은 별도 엔드포인트 필요."""
        if self.dry_run:
            return {"order_id": order_id, "accepted": True, "message": "dry_run"}
        # Placeholder: actual cancel endpoint not implemented here
        return {"order_id": order_id, "accepted": False, "message": "not_implemented"}

    def positions(self) -> Dict[str, object]:
        """보유 종목/수량 조회.

        TR-ID: (모의) VTTC8432R / (실전) TTTC8432R
        실패/예외 시 빈 딕셔너리 반환.
        """
        if self.dry_run:
            return {}
        tr_id = "VTTC8432R" if self.use_paper else "TTTC8432R"
        params = {
            "CANO": self.account_no,
            "ACNT_PRDT_CD": self.product_code,
        }
        try:
            attempts = 0
            while attempts < 2:
                self._rate_limiter.wait()
                res = self.session.get(
                    f"{self.base_url}/uapi/domestic-stock/v1/trading/inquire-balance",
                    headers=self._headers(tr_id),
                    params=params,
                    timeout=10,
                )
                if res.status_code == 401 and attempts == 0:
                    self._ensure_token()
                    attempts += 1
                    continue
                if not res.ok:
                    return {}
                data = res.json()
                positions = {}
                output = data.get("output", []) or []
                for item in output:
                    symbol = item.get("pdno") or item.get("PDNO")
                    qty = item.get("hldg_qty") or item.get("HLDG_QTY")
                    if symbol is not None and qty is not None:
                        try:
                            positions[str(symbol)] = int(float(qty))
                        except (TypeError, ValueError):
                            continue
                return positions
            return {}
        except Exception:  # pragma: no cover
            return {}

    def cash_available(self) -> float:
        """매수가능 현금 조회.

        TR-ID: (모의) VTTC8434R / (실전) TTTC8434R
        실패/예외 시 0.0 반환.
        """
        if self.dry_run:
            return 0.0
        tr_id = "VTTC8434R" if self.use_paper else "TTTC8434R"
        body = {
            "CANO": self.account_no,
            "ACNT_PRDT_CD": self.product_code,
        }
        try:
            hashkey = self._hash_body(body)
            attempts = 0
            while attempts < 2:
                self._rate_limiter.wait()
                res = self.session.post(
                    f"{self.base_url}/uapi/domestic-stock/v1/trading/inquire-psbl-order",
                    headers=self._headers(tr_id, hashkey=hashkey),
                    json=body,
                    timeout=10,
                )
                if res.status_code == 401 and attempts == 0:
                    self._ensure_token()
                    attempts += 1
                    continue
                if not res.ok:
                    return 0.0
                data = res.json()
                output = data.get("output", {})
                amount = output.get("ord_psbl_cash") or output.get("ORD_PSBL_CASH")
                try:
                    return float(amount)
                except (TypeError, ValueError):
                    return 0.0
            return 0.0
        except Exception:  # pragma: no cover
            return 0.0
