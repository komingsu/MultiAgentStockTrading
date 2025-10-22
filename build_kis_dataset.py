#!/usr/bin/env python
"""Build a FinRL-ready dataset using KIS domestic stock data.

The script mirrors the data layout consumed in `p3.ipynb` by:
1. Collecting OHLCV history for the requested Korean tickers via the KIS API,
2. Engineering technical indicators with `helper_function.FeatureEngineer`,
3. Splitting the dataset into train/test/trade/backtest CSVs under `./data`.

Example:
    python build_kis_dataset.py --start 2023-01-02 --end 2024-10-22 --max-symbols 30
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
from dotenv import load_dotenv

import config
from helper_function import FeatureEngineer, data_split
from kis_auth import get_or_load_access_token
from kis_data_bundle.kis_data.ingest import (
    KisAuth,
    _resolve_kis_auth_from_env,
    get_daily_candle_kis,
)

DATE_FMT = "%Y-%m-%d"
DATE_FMT_KIS = "%Y%m%d"
LOOKBACK_BUFFER_DAYS = 252  # ensures rolling indicators/turbulence have sufficient history
SMA_WINDOWS = (10, 30)
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BOLLINGER_WINDOW = 20
BOLLINGER_STD = 2
STOCH_WINDOW = 14
ATR_PERIOD = 14
DISPARITY_WINDOW = 25
REQUIRED_COMPLETE_COLS = [
    *(f"sma_{window}" for window in SMA_WINDOWS),
    "bollinger_upper",
    "bollinger_lower",
    "stochastic_oscillator",
    "atr",
    "disparity_25",
]
VIX_CACHE_FILENAME = "vix_daily.csv"
TURBULENCE_CACHE_FILENAME = "turbulence_daily.csv"


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect and prepare KIS domestic stock data.")
    parser.add_argument("--env", choices=("real", "mock"), default="real", help="KIS environment to use.")
    parser.add_argument(
        "--dotenv",
        default=".env",
        help="Path to the .env file containing KIS credentials (defaults to ./\\.env).",
    )
    parser.add_argument(
        "--symbols",
        nargs="*",
        default=None,
        help="Optional explicit list of symbols (6-digit codes). Defaults to KIS_SYMBOL_WHITELIST in .env.",
    )
    parser.add_argument(
        "--max-symbols",
        type=int,
        default=30,
        help="Cap the number of symbols to collect (helps avoid large batch jobs).",
    )
    parser.add_argument(
        "--start",
        default=None,
        help="Dataset start date (YYYY-MM-DD). Defaults to the earliest configured split boundary.",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="Dataset end date (YYYY-MM-DD). Defaults to the latest configured split boundary.",
    )
    parser.add_argument(
        "--train-start",
        default=config.TRAIN_START_DATE,
        help="Training start date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--train-end",
        default=config.TRAIN_END_DATE,
        help="Training end date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--test-start",
        default=config.TEST_START_DATE,
        help="Validation/test start date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--test-end",
        default=config.TEST_END_DATE,
        help="Validation/test end date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--trade-start",
        default=config.TRADE_START_DATE,
        help="Trading start date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--trade-end",
        default=config.TRADE_END_DATE,
        help="Trading end date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--backtest-start",
        default=None,
        help="Backtest start date (YYYY-MM-DD). Defaults to trade_end if omitted.",
    )
    parser.add_argument(
        "--backtest-end",
        default=None,
        help="Backtest end date (YYYY-MM-DD). Defaults to dataset end date if omitted.",
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Directory where train/test/trade/backtest CSVs will be written.",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Force a fresh KIS access token before collection.",
    )
    parser.add_argument(
        "--rate-limit-sleep",
        type=float,
        default=0.2,
        help="Seconds to sleep between KIS requests (helps respect API limits).",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def _parse_date(date_str: str) -> datetime:
    return datetime.strptime(date_str, DATE_FMT)


def _format_date(date_obj: datetime) -> str:
    return date_obj.strftime(DATE_FMT)


def _determine_dataset_window(args: argparse.Namespace) -> tuple[str, str]:
    candidates_start = [
        dt_str
        for dt_str in [
            args.start,
            args.train_start,
            args.test_start,
            args.trade_start,
            args.backtest_start,
        ]
        if dt_str
    ]
    candidates_end = [
        dt_str
        for dt_str in [
            args.end,
            args.train_end,
            args.test_end,
            args.trade_end,
            args.backtest_end,
        ]
        if dt_str
    ]
    if not candidates_start or not candidates_end:
        raise ValueError("Unable to infer dataset window; please provide --start/--end explicitly.")
    start = min(candidates_start)
    end = max(candidates_end)
    return start, end


def _clean_symbol(symbol: str) -> str:
    sym = str(symbol).strip()
    return sym.zfill(6) if sym.isdigit() else sym


def _resolve_symbols(args: argparse.Namespace) -> List[str]:
    if args.symbols:
        symbols = [_clean_symbol(sym) for sym in args.symbols]
    else:
        env_list = os.getenv("KIS_SYMBOL_WHITELIST", "")
        symbols = [_clean_symbol(sym) for sym in env_list.split(",") if sym.strip()]
    if not symbols:
        raise ValueError("No symbols provided; supply --symbols or set KIS_SYMBOL_WHITELIST in .env.")
    if args.max_symbols is not None:
        symbols = symbols[: int(args.max_symbols)]
    return symbols


def _resolve_backtest_window(args: argparse.Namespace, dataset_end: str) -> tuple[str, str]:
    start = args.backtest_start or args.trade_end
    end = args.backtest_end or dataset_end
    if _parse_date(start) > _parse_date(end):
        raise ValueError(f"Backtest window invalid: start {start} is after end {end}.")
    return start, end


def _ensure_token_env(env: str, *, force_refresh: bool) -> KisAuth:
    suffix = "" if env == "real" else "_MOCK"
    token = get_or_load_access_token(env=env, force_refresh=force_refresh)
    os.environ[f"KIS_ACCESS_TOKEN{suffix}"] = token
    return _resolve_kis_auth_from_env(env)


def _fetch_symbol_history(
    auth: KisAuth,
    symbol: str,
    start_dt: datetime,
    end_dt: datetime,
    *,
    rate_limit_sleep: float = 0.2,
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    current_end = end_dt
    while current_end >= start_dt:
        batch = get_daily_candle_kis(
            symbol,
            start_dt.strftime(DATE_FMT_KIS),
            current_end.strftime(DATE_FMT_KIS),
            auth,
        )
        if batch.empty:
            break
        frames.append(batch)
        earliest = batch["date"].min()
        if pd.isna(earliest):
            break
        earliest_dt = earliest.to_pydatetime()
        if earliest_dt <= start_dt:
            break
        current_end = earliest_dt - timedelta(days=1)
        if rate_limit_sleep:
            time.sleep(rate_limit_sleep)

    if not frames:
        return pd.DataFrame()

    df = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset=["date"], keep="first")
        .sort_values("date")
    )
    mask = (df["date"] >= start_dt) & (df["date"] <= end_dt)
    df = df.loc[mask].copy()
    return df


def _collect_ohlcv(
    auth: KisAuth,
    symbols: Iterable[str],
    start: str,
    end: str,
    *,
    rate_limit_sleep: float,
) -> pd.DataFrame:
    start_dt = _parse_date(start)
    end_dt = _parse_date(end)
    rows: List[pd.DataFrame] = []

    symbols_list = list(symbols)

    for idx, sym in enumerate(symbols_list, 1):
        history = _fetch_symbol_history(
            auth,
            sym,
            start_dt,
            end_dt,
            rate_limit_sleep=rate_limit_sleep,
        )
        if history.empty:
            print(f"⚠️  No data returned for {sym}; skipping.")
            continue
        keep_cols = ["date", "open", "high", "low", "close", "volume", "symbol"]
        missing = [c for c in keep_cols if c not in history.columns]
        if missing:
            raise RuntimeError(f"Missing expected columns {missing} in KIS response for {sym}.")
        history = history[keep_cols].copy()
        history["tic"] = history["symbol"].astype(str).str.zfill(6)
        history = history.drop(columns=["symbol"])
        rows.append(history)
        print(f"✅ Collected {len(history)} rows for {sym} ({idx}/{len(symbols_list)}).")

    if not rows:
        raise RuntimeError("No data collected for any symbol.")

    df = pd.concat(rows, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df["day"] = df["date"].dt.dayofweek
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values(["date", "tic"]).reset_index(drop=True)
    df["date"] = df["date"].dt.strftime(DATE_FMT)
    return df


def _calculate_rsi(series: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _calculate_atr(group: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    high = group["high"].astype(float)
    low = group["low"].astype(float)
    close = group["close"].astype(float)
    prev_close = close.shift(1)

    high_low = (high - low).abs()
    high_close = (high - prev_close).abs()
    low_close = (low - prev_close).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    return atr


def _enrich_price_indicators(group: pd.DataFrame) -> pd.DataFrame:
    grp = group.sort_values("date").copy()
    close = grp["close"].astype(float)

    for window in SMA_WINDOWS:
        grp[f"sma_{window}"] = close.rolling(window=window, min_periods=window).mean()

    grp["rsi"] = _calculate_rsi(close).clip(lower=0.0, upper=100.0)

    ema_fast = close.ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = close.ewm(span=MACD_SLOW, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=MACD_SIGNAL, adjust=False).mean()
    grp["macd"] = macd
    grp["macd_signal"] = macd_signal
    grp["macd_hist"] = macd - macd_signal

    rolling_mean = close.rolling(window=BOLLINGER_WINDOW, min_periods=BOLLINGER_WINDOW).mean()
    rolling_std = close.rolling(window=BOLLINGER_WINDOW, min_periods=BOLLINGER_WINDOW).std()
    grp["bollinger_upper"] = rolling_mean + (BOLLINGER_STD * rolling_std)
    grp["bollinger_lower"] = rolling_mean - (BOLLINGER_STD * rolling_std)

    highest_high = grp["high"].astype(float).rolling(window=STOCH_WINDOW, min_periods=STOCH_WINDOW).max()
    lowest_low = grp["low"].astype(float).rolling(window=STOCH_WINDOW, min_periods=STOCH_WINDOW).min()
    denominator = (highest_high - lowest_low).replace(0.0, np.nan)
    stoch = (close - lowest_low) / denominator * 100
    grp["stochastic_oscillator"] = stoch.clip(lower=0.0, upper=100.0)

    disparity_ma = close.rolling(window=DISPARITY_WINDOW, min_periods=DISPARITY_WINDOW).mean()
    grp["disparity_25"] = (close / disparity_ma.replace(0.0, np.nan)) * 100

    grp["atr"] = _calculate_atr(grp)
    return grp


def _trim_incomplete_history(df: pd.DataFrame) -> pd.DataFrame:
    trimmed_groups: list[pd.DataFrame] = []
    for _, group in df.groupby("tic", sort=False):
        mask = group[REQUIRED_COMPLETE_COLS].notna().all(axis=1)
        trimmed = group.loc[mask].copy()
        if not trimmed.empty:
            trimmed_groups.append(trimmed)
    if not trimmed_groups:
        return df.iloc[0:0]
    return pd.concat(trimmed_groups, ignore_index=True)


def _download_vix_series(start: datetime, end: datetime) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError:  # pragma: no cover - yfinance optional
        print("⚠️  yfinance not available; unable to compute VIX.")
        return pd.DataFrame()

    try:
        vix_df = yf.download(
            "^VIX",
            start=start.strftime(DATE_FMT),
            end=(end + timedelta(days=1)).strftime(DATE_FMT),
            progress=False,
            auto_adjust=False,
            group_by="column",
        )
    except Exception as exc:  # pragma: no cover - network errors bubble up here
        print(f"⚠️  Unable to download VIX data: {exc}")
        return pd.DataFrame()

    if vix_df.empty:
        return pd.DataFrame()

    if isinstance(vix_df.columns, pd.MultiIndex):
        vix_df.columns = vix_df.columns.droplevel(-1)

    vix = vix_df.reset_index()[["Date", "Close"]].rename(columns={"Date": "date", "Close": "vix"})
    vix["date"] = pd.to_datetime(vix["date"])
    return vix


def _load_or_fetch_vix_series(start: datetime, end: datetime, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    cache_path = output_dir / VIX_CACHE_FILENAME
    if cache_path.exists():
        cached = pd.read_csv(cache_path, parse_dates=["date"])
    else:
        cached = pd.DataFrame(columns=["date", "vix"])

    coverage_ok = not cached.empty and cached["date"].min() <= start and cached["date"].max() >= end

    if not coverage_ok:
        buffer_start = start - timedelta(days=5)
        buffer_end = end + timedelta(days=5)
        min_start = cached["date"].min() if not cached.empty else buffer_start
        max_end = cached["date"].max() if not cached.empty else buffer_end
        download_start = min(buffer_start, min_start)
        download_end = max(buffer_end, max_end)
        fetched = _download_vix_series(download_start, download_end)
        if not fetched.empty:
            cached = pd.concat([cached, fetched], ignore_index=True)
            cached = cached.drop_duplicates(subset="date", keep="last").sort_values("date")
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            to_write = cached.copy()
            to_write["date"] = to_write["date"].dt.strftime(DATE_FMT)
            to_write.to_csv(cache_path, index=False)
        elif cached.empty:
            return pd.DataFrame(), pd.DataFrame()

    filtered = cached[(cached["date"] >= start) & (cached["date"] <= end)].copy()
    return filtered, cached


def _write_daily_series(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        return
    series = df.copy()
    series["date"] = pd.to_datetime(series["date"]).dt.strftime(DATE_FMT)
    series = series.drop_duplicates(subset="date").sort_values("date")
    path.parent.mkdir(parents=True, exist_ok=True)
    series.to_csv(path, index=False)


def _engineer_features(df: pd.DataFrame, *, dataset_start: str, output_dir: Path) -> pd.DataFrame:
    enriched = df.copy()
    enriched["date"] = pd.to_datetime(enriched["date"])
    enriched = enriched.sort_values(["tic", "date"]).reset_index(drop=True)

    groups = []
    for _, group in enriched.groupby("tic", sort=False):
        groups.append(_enrich_price_indicators(group))
    enriched = pd.concat(groups, ignore_index=True)

    enriched = _trim_incomplete_history(enriched)

    start_dt = enriched["date"].min()
    end_dt = enriched["date"].max()
    vix, vix_cache = _load_or_fetch_vix_series(start_dt, end_dt, output_dir)
    if not vix.empty:
        enriched = enriched.merge(vix, on="date", how="left")
        enriched["vix"] = enriched["vix"].ffill().bfill()
    else:
        enriched["vix"] = 0.0

    fe = FeatureEngineer(use_technical_indicator=False, use_turbulence=False)
    try:
        turbulence = fe.calculate_turbulence(enriched[["date", "tic", "close"]])
    except Exception as exc:  # pragma: no cover - occurs when history is too short
        print(f"⚠️  Unable to compute turbulence: {exc}; filling zeros.")
        turbulence = enriched[["date"]].drop_duplicates().copy()
        turbulence["turbulence"] = 0.0
    enriched = enriched.merge(turbulence, on="date", how="left")
    enriched["turbulence"] = enriched["turbulence"].fillna(0.0)

    cutoff = pd.to_datetime(dataset_start)
    enriched = enriched[enriched["date"] >= cutoff].copy()

    enriched = enriched.sort_values(["date", "tic"]).reset_index(drop=True)

    unique_dates = pd.Index(np.sort(enriched["date"].unique()))
    expected_dates = len(unique_dates)
    counts = enriched.groupby("tic")["date"].nunique()
    full_tickers = counts[counts == expected_dates].index
    if full_tickers.empty:
        raise RuntimeError("No tickers retained with complete indicator history for the requested window.")
    enriched = enriched[enriched["tic"].isin(full_tickers)].copy()

    unique_dates = pd.Index(np.sort(enriched["date"].unique()))
    day_lookup = pd.Series(range(len(unique_dates)), index=unique_dates)
    enriched["day"] = enriched["date"].map(day_lookup).astype(int)

    enriched["tic"] = enriched["tic"].astype(str).str.zfill(6)
    enriched["date"] = enriched["date"].dt.strftime(DATE_FMT)

    if not vix_cache.empty:
        _write_daily_series(vix_cache, output_dir / VIX_CACHE_FILENAME)
    _write_daily_series(enriched[["date", "turbulence"]].drop_duplicates(subset="date"), output_dir / TURBULENCE_CACHE_FILENAME)

    desired_order = [
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "tic",
        "vix",
        "turbulence",
        "sma_10",
        "sma_30",
        "rsi",
        "macd",
        "macd_signal",
        "macd_hist",
        "bollinger_upper",
        "bollinger_lower",
        "stochastic_oscillator",
        "disparity_25",
        "atr",
        "day",
    ]

    existing = [col for col in desired_order if col in enriched.columns]
    remaining = [col for col in enriched.columns if col not in existing]
    enriched = enriched[existing + remaining]
    return enriched


def _save_dataset(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(temp_path, index=False)
    try:
        temp_path.replace(path)
    except PermissionError:
        print(f"⚠️  Unable to overwrite {path}; leaving temporary file at {temp_path}.")


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    load_dotenv(args.dotenv)
    symbols = _resolve_symbols(args)
    dataset_start, dataset_end = _determine_dataset_window(args)
    backtest_start, backtest_end = _resolve_backtest_window(args, dataset_end)

    dataset_start_dt = _parse_date(dataset_start)
    extended_start_dt = dataset_start_dt - timedelta(days=LOOKBACK_BUFFER_DAYS)
    prefetch_start = _format_date(extended_start_dt)

    print(
        "➡️  Collecting KIS data for "
        f"{len(symbols)} symbols from {prefetch_start} (prefetch) to {dataset_end} ({args.env})."
    )
    auth = _ensure_token_env(args.env, force_refresh=args.force_refresh)
    try:
        ohlcv = _collect_ohlcv(
            auth,
            symbols,
            prefetch_start,
            dataset_end,
            rate_limit_sleep=args.rate_limit_sleep,
        )
    except Exception as exc:
        print(f"❌ Failed to collect OHLCV: {exc}", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("➡️  Engineering technical indicators...")
    features = _engineer_features(ohlcv, dataset_start=dataset_start, output_dir=output_dir)

    full_path = output_dir / "kis_full_data.csv"
    _save_dataset(features, full_path)

    print("➡️  Splitting dataset using configured windows...")
    train = data_split(features, args.train_start, args.train_end)
    test = data_split(features, args.test_start, args.test_end)
    trade = data_split(features, args.trade_start, args.trade_end)
    backtest = data_split(features, backtest_start, backtest_end)

    _save_dataset(train, output_dir / "train_data.csv")
    _save_dataset(test, output_dir / "stock_test_data.csv")
    _save_dataset(trade, output_dir / "trade_data.csv")
    _save_dataset(backtest, output_dir / "backtest_data.csv")

    print("✅ Dataset build complete.")
    print(
        f"    train: {len(train):6d} rows | {train['date'].min()} → {train['date'].max()}"
    )
    print(
        f"    test : {len(test):6d} rows | {test['date'].min()} → {test['date'].max()}"
        if not test.empty
        else "    test : 0 rows"
    )
    print(
        f"    trade: {len(trade):6d} rows | {trade['date'].min()} → {trade['date'].max()}"
        if not trade.empty
        else "    trade: 0 rows"
    )
    print(
        f"    back : {len(backtest):6d} rows | {backtest['date'].min()} → {backtest['date'].max()}"
        if not backtest.empty
        else "    back : 0 rows"
    )
    print(f"    full : {len(features):6d} rows | {features['date'].min()} → {features['date'].max()}")
    print(f"    saved to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
