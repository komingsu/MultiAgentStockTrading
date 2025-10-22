"""
kis_auth.py

한국투자증권(KIS) API 토큰 발급/캐싱 유틸리티.

기능
- get_access_token(env): 실시간 토큰 발급 (실전/모의)
- get_or_load_access_token(env, force_refresh=False): .env에 저장된 토큰을 날짜 기준 재사용,
  필요 시 새 토큰 발급 후 .env 갱신
- is_token_fresh_today(env): 오늘 날짜의 토큰인지 확인

환경 변수(.env)
# == 한국 투자 증권 API 실전 키 ==
KIS_API_KEY=...
KIS_API_SECRET=...

# == 한국 투자 증권 API 모의 키 ==
KIS_API_KEY_MOCK=...
KIS_API_SECRET_MOCK=...

# == 토큰(실전/모의)
KIS_ACCESS_TOKEN=...
KIS_ACCESS_TOKEN_DATE=YYYYMMDD
KIS_ACCESS_TOKEN_MOCK=...
KIS_ACCESS_TOKEN_MOCK_DATE=YYYYMMDD

사용 예시
    from packages.core.kis_auth import get_or_load_access_token

    token = get_or_load_access_token(env="real")  # 또는 "mock"
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Tuple

import requests
from dotenv import load_dotenv

# .env 로드 (다른 모듈에서 이미 load_dotenv() 호출해도 무해)
load_dotenv()

# 엔드포인트/키 이름 매핑
_ENV_TABLE = {
    "real": {
        "BASE_URL": "https://openapi.koreainvestment.com:9443",
        "APPKEY": "KIS_API_KEY",
        "APPSECRET": "KIS_API_SECRET",
        "TOKEN": "KIS_ACCESS_TOKEN",
        "TOKEN_DATE": "KIS_ACCESS_TOKEN_DATE",
    },
    "mock": {
        "BASE_URL": "https://openapivts.koreainvestment.com:29443",
        "APPKEY": "KIS_API_KEY_MOCK",
        "APPSECRET": "KIS_API_SECRET_MOCK",
        "TOKEN": "KIS_ACCESS_TOKEN_MOCK",
        "TOKEN_DATE": "KIS_ACCESS_TOKEN_MOCK_DATE",
    },
}


def _today_str() -> str:
    return datetime.now().strftime("%Y%m%d")


def _get_env_keys(env: str) -> Tuple[str, str, str, str, str]:
    if env not in _ENV_TABLE:
        raise ValueError(f"env must be 'real' or 'mock', got: {env}")
    cfg = _ENV_TABLE[env]
    return (
        cfg["BASE_URL"],
        cfg["APPKEY"],
        cfg["APPSECRET"],
        cfg["TOKEN"],
        cfg["TOKEN_DATE"],
    )


def _require_env(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise RuntimeError(
            f"Missing environment variable '{name}'. "
            f"Check your .env file."
        )
    return val


def _upsert_env(k: str, v: str) -> None:
    """
    .env 파일에서 키 k를 값 v로 upsert.
    - 기존 라인 있으면 교체
    - 없으면 마지막에 추가
    - 따옴표 없이 순수값으로 저장 (KIS 토큰은 '='/공백 없음)
    """
    lines = []
    if os.path.exists(".env"):
        with open(".env", "r", encoding="utf-8") as f:
            lines = f.readlines()

    found = False
    out = []
    for line in lines:
        if line.startswith(f"{k}="):
            out.append(f"{k}={v}\n")
            found = True
        else:
            out.append(line)

    if not found:
        out.append(f"{k}={v}\n")

    with open(".env", "w", encoding="utf-8") as f:
        f.writelines(out)


def is_token_fresh_today(env: str = "real") -> bool:
    """
    오늘 날짜 기준으로 .env에 저장된 토큰이 유효(같은 날짜)한지 확인.
    """
    _, _, _, token_key, token_date_key = _get_env_keys(env)
    token = os.getenv(token_key)
    token_date = os.getenv(token_date_key)
    return bool(token and token_date == _today_str())


def get_access_token(env: str = "real") -> str:
    """
    KIS OAuth 토큰 발급 (실전/모의).
    항상 원격으로 새 토큰을 요청한다.
    """
    base_url, appkey_key, appsecret_key, _, _ = _get_env_keys(env)
    appkey = _require_env(appkey_key)
    appsecret = _require_env(appsecret_key)

    url = f"{base_url}/oauth2/tokenP"
    headers = {"content-type": "application/json"}
    body = {
        "grant_type": "client_credentials",
        "appkey": appkey,
        "appsecret": appsecret,
    }

    res = requests.post(url, headers=headers, json=body, timeout=20)
    res.raise_for_status()
    data = res.json()

    # API 표준 응답에 'access_token' 키가 반드시 존재
    token = data.get("access_token")
    if not token:
        raise RuntimeError(
            f"Token response missing 'access_token': {data}"
        )
    return token


def get_or_load_access_token(env: str = "real", force_refresh: bool = False) -> str:
    """
    .env의 토큰을 오늘 날짜 기준으로 재사용.
    - 오늘자면 그대로 반환
    - 아니면 새로 발급 후 .env에 저장

    force_refresh=True 이면 날짜와 무관하게 무조건 새 발급.
    """
    base_url, appkey_key, appsecret_key, token_key, token_date_key = _get_env_keys(env)

    # 키 유효성 선검사
    _require_env(appkey_key)
    _require_env(appsecret_key)

    if not force_refresh:
        token = os.getenv(token_key)
        token_date = os.getenv(token_date_key)
        if token and token_date == _today_str():
            # 재사용
            print(f"✅ reuse {env} token:", token[:20], "...")
            return token

    # 새 발급
    new_token = get_access_token(env=env)
    _upsert_env(token_key, new_token)
    _upsert_env(token_date_key, _today_str())
    print(f"🔄 refreshed {env} token and saved to .env")
    return new_token
