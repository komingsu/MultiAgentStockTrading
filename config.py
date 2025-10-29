from __future__ import annotations

DATA_SAVE_DIR = "datasets"
TRAINED_MODEL_DIR = "trained_models"
TENSORBOARD_LOG_DIR = "tensorboard_log"
RESULTS_DIR = "results"

# date format: '%Y-%m-%d'
TRAIN_START_DATE = "2023-01-02"
TRAIN_END_DATE = "2024-12-30"

TEST_START_DATE = "2025-01-02"
TEST_END_DATE = "2025-04-29"

TRADE_START_DATE = "2025-05-02"
TRADE_END_DATE = "2025-08-29"

INDICATORS = [
    "macd",
    "macd_signal",
    "macd_hist",
    "bollinger_upper",
    "bollinger_lower",
    "rsi",
    "stochastic_oscillator",
    "disparity_25",
    "atr",
    "sma_10",
    "sma_30",
]

RISK_INDICATOR_COL = "turbulence"

# Possible time zones
TIME_ZONE_SHANGHAI = "Asia/Shanghai"  # Hang Seng HSI, SSE, CSI
TIME_ZONE_USEASTERN = "US/Eastern"  # Dow, Nasdaq, SP
TIME_ZONE_PARIS = "Europe/Paris"  # CAC,
TIME_ZONE_BERLIN = "Europe/Berlin"  # DAX, TECDAX, MDAX, SDAX
TIME_ZONE_SEOUL = "Asia/Seoul"  # KOSPI, KOSDAQ
TIME_ZONE_SELFDEFINED = "Asia/Seoul"  # If neither of the above is your time zone, you should define it, and set USE_TIME_ZONE_SELFDEFINED 1.
USE_TIME_ZONE_SELFDEFINED = 1  # 0 (default) or 1 (use the self defined)

MY_TICKER = [
    "000250",
    "005930",
    "005935",
    "007660",
    "009150",
    "030530",
    "035900",
    "039030",
    "051910",
    "077970",
    "084370",
    "086520",
    "087010",
    "089030",
    "108490",
    "131970",
    "141080",
    "196170",
    "214370",
    "214450",
    "240810",
    "247540",
    "249420",
    "277810",
    "298040",
    "298380",
    "353200",
    "357780",
]

SINGLE_TICKER = ["005930"]


# %%
