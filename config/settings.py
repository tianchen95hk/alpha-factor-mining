import os

from dotenv import load_dotenv


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, ".env"))


class Settings:
    # Base paths
    DB_PATH = os.path.join(BASE_DIR, "data", "cache")
    OUTPUT_PATH = os.path.join(BASE_DIR, "results")
    FACTOR_FILE = os.path.join(DB_PATH, "market_data_4h.csv")

    # Exchange/auth
    API_KEY = os.getenv("BINANCE_API_KEY", "")
    SECRET_KEY = os.getenv("BINANCE_SECRET_KEY", os.getenv("BINANCE_API_SECRET", ""))
    USE_PROXY = str(os.getenv("USE_PROXY", "False")).lower() == "true"
    PROXY_URL = os.getenv("PROXY_URL", "")

    LIMIT = 6000
    TIMEFRAME = os.getenv("TIMEFRAME", "4h")
    target_symbols_count = 40
    BACKTEST_START_DATE = os.getenv("BACKTEST_START_DATE", "2023-01-01")
    BACKTEST_END_DATE = os.getenv("BACKTEST_END_DATE", "2026-02-22")
    INITIAL_CAPITAL = 100000
    TAKER_FEE = 0.001
    BACKTEST_TOP_QUANTILE = float(os.getenv("BACKTEST_TOP_QUANTILE", "0.2"))
    BACKTEST_MAX_ABS_WEIGHT = float(os.getenv("BACKTEST_MAX_ABS_WEIGHT", "0.2"))
    BACKTEST_VOL_LOOKBACK = int(os.getenv("BACKTEST_VOL_LOOKBACK", "24"))
    BACKTEST_MULTI_FACTOR_TOP_K = int(os.getenv("BACKTEST_MULTI_FACTOR_TOP_K", "8"))
    BACKTEST_SIGNAL_SMOOTHING_SPAN = int(os.getenv("BACKTEST_SIGNAL_SMOOTHING_SPAN", "6"))
    BACKTEST_EXECUTION_ALPHA = float(os.getenv("BACKTEST_EXECUTION_ALPHA", "0.35"))
    BACKTEST_REBALANCE_INTERVAL = int(os.getenv("BACKTEST_REBALANCE_INTERVAL", "0"))
    BACKTEST_TARGET_GROSS_EXPOSURE = float(os.getenv("BACKTEST_TARGET_GROSS_EXPOSURE", "1.0"))
    BACKTEST_SAVE_POSITION_EVENTS = str(os.getenv("BACKTEST_SAVE_POSITION_EVENTS", "True")).lower() == "true"
    BACKTEST_POSITION_EVENT_THRESHOLD = float(os.getenv("BACKTEST_POSITION_EVENT_THRESHOLD", "0.0001"))

    # Research/evaluation
    LOOKBACK_DAYS = int(os.getenv("LOOKBACK_DAYS", "1095"))
    HORIZON_BARS = int(os.getenv("HORIZON_BARS", "12"))
    MIN_ASSETS_PER_TIMESTAMP = int(os.getenv("MIN_ASSETS_PER_TIMESTAMP", "8"))
    TOP_K_FACTORS = int(os.getenv("TOP_K_FACTORS", "5"))
    FACTOR_MAX_COUNT = int(os.getenv("FACTOR_MAX_COUNT", "80"))
    FACTOR_ENABLE_COMBOS = str(os.getenv("FACTOR_ENABLE_COMBOS", "True")).lower() == "true"
    FACTOR_COMBO_SOURCE_TOP_N = int(os.getenv("FACTOR_COMBO_SOURCE_TOP_N", "16"))
    FACTOR_COMBO_MAX_COUNT = int(os.getenv("FACTOR_COMBO_MAX_COUNT", "40"))
    FACTOR_EVAL_HORIZONS = os.getenv("FACTOR_EVAL_HORIZONS", "6,12,24")
    FACTOR_CORR_THRESHOLD = float(os.getenv("FACTOR_CORR_THRESHOLD", "0.85"))
    FACTOR_DIVERSIFY_POOL_SIZE = int(os.getenv("FACTOR_DIVERSIFY_POOL_SIZE", "120"))
    SYMBOLS = [
        s.strip()
        for s in os.getenv("SYMBOLS", "").split(",")
        if s.strip()
    ]

    # Strategy params
    MA_WINDOW = 930
    BB_STD = 2.0
    DIFF_MA_SHORT = 12
    DIFF_MA_LONG = 24
    DIFF_RSI_WIN = 14
    Y_ROC_PERIOD = 14
    Y_ZSCORE_WIN = 30

    # Trading trigger
    BUY_DRAWDOWN = -0.20
    BUY_DIFFUSION = -20
    BUY_Y_RISE_DAYS = 2
    SELL_BB_FACTOR = 1.0
    SELL_DIFFUSION = 20
    SELL_Y_FALL_DAYS = 2

    # Position/risk
    TOP_N = 3
    CORE_BTC_WEIGHT = 0.5
    STOP_LOSS_PCT = 0.08
    STOPLOSS_COOLDOWN_BARS = 2
    TRAILING_STOP_PCT = 0.10
    TARGET_VOLATILITY = 0.55

    # Live trading
    LIVE_ENABLED = str(os.getenv("LIVE_ENABLED", "False")).lower() == "true"
    LIVE_DRY_RUN = str(os.getenv("LIVE_DRY_RUN", "True")).lower() == "true"
    LIVE_ALLOW_SHORT = str(os.getenv("LIVE_ALLOW_SHORT", "True")).lower() == "true"
    LIVE_MIN_ORDER_USDT = float(os.getenv("LIVE_MIN_ORDER_USDT", "25"))
    LIVE_CAPITAL_UTILIZATION = float(os.getenv("LIVE_CAPITAL_UTILIZATION", "0.95"))
    LIVE_MAX_GROSS_EXPOSURE = float(os.getenv("LIVE_MAX_GROSS_EXPOSURE", "1.0"))
    LIVE_LOOKBACK_BARS = int(os.getenv("LIVE_LOOKBACK_BARS", "1300"))
    LIVE_POSITION_SIDE = os.getenv("LIVE_POSITION_SIDE", "BOTH")
    LIVE_SYMBOLS_RAW = os.getenv("LIVE_SYMBOLS", "BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT")
    LIVE_SYMBOLS = [
        s.strip().upper()
        for s in os.getenv("LIVE_SYMBOLS", "BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT").split(",")
        if s.strip()
    ]
    LIVE_STATE_FILE = os.path.join(OUTPUT_PATH, "live_state.json")

    # Compatibility aliases for current factor-mining pipeline
    TOP_N_SYMBOLS = target_symbols_count
    CACHE_PATH = FACTOR_FILE
    RESULTS_DIR = OUTPUT_PATH
