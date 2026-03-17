from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path

try:
    from config.settings import Settings
except ModuleNotFoundError:
    settings_path = Path(__file__).resolve().parents[2] / "config" / "settings.py"
    spec = importlib.util.spec_from_file_location("project_settings", settings_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load settings module from {settings_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    Settings = module.Settings


@dataclass
class RuntimeConfig:
    timeframe: str
    lookback_days: int
    horizon_bars: int
    top_n_symbols: int
    symbols: list[str]
    cache_path: str
    results_dir: str
    min_assets_per_timestamp: int
    top_k_factors: int
    api_key: str
    secret_key: str
    use_proxy: bool
    proxy_url: str
    factor_max_count: int
    factor_enable_combos: bool
    factor_combo_source_top_n: int
    factor_combo_max_count: int
    factor_eval_horizons: list[int]
    factor_corr_threshold: float
    factor_diversify_pool_size: int
    initial_capital: float
    taker_fee: float
    backtest_top_quantile: float
    backtest_max_abs_weight: float
    backtest_vol_lookback: int
    backtest_multi_factor_top_k: int
    backtest_signal_smoothing_span: int
    backtest_execution_alpha: float
    backtest_rebalance_interval: int
    backtest_target_gross_exposure: float
    backtest_save_position_events: bool
    backtest_position_event_threshold: float
    backtest_start_date: str | None
    backtest_end_date: str | None


def _normalize_optional_date(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _parse_horizon_list(raw: object, fallback: int) -> list[int]:
    text = str(raw or "").strip()
    out: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            val = int(part)
        except ValueError:
            continue
        if val > 0:
            out.append(val)
    if fallback > 0:
        out.append(int(fallback))
    return sorted(set(out))


def get_settings() -> RuntimeConfig:
    return RuntimeConfig(
        timeframe=str(Settings.TIMEFRAME).lower(),
        lookback_days=Settings.LOOKBACK_DAYS,
        horizon_bars=Settings.HORIZON_BARS,
        top_n_symbols=Settings.TOP_N_SYMBOLS,
        symbols=list(Settings.SYMBOLS),
        cache_path=Settings.CACHE_PATH,
        results_dir=Settings.RESULTS_DIR,
        min_assets_per_timestamp=Settings.MIN_ASSETS_PER_TIMESTAMP,
        top_k_factors=Settings.TOP_K_FACTORS,
        api_key=Settings.API_KEY,
        secret_key=Settings.SECRET_KEY,
        use_proxy=Settings.USE_PROXY,
        proxy_url=Settings.PROXY_URL,
        factor_max_count=Settings.FACTOR_MAX_COUNT,
        factor_enable_combos=Settings.FACTOR_ENABLE_COMBOS,
        factor_combo_source_top_n=Settings.FACTOR_COMBO_SOURCE_TOP_N,
        factor_combo_max_count=Settings.FACTOR_COMBO_MAX_COUNT,
        factor_eval_horizons=_parse_horizon_list(Settings.FACTOR_EVAL_HORIZONS, Settings.HORIZON_BARS),
        factor_corr_threshold=float(Settings.FACTOR_CORR_THRESHOLD),
        factor_diversify_pool_size=int(Settings.FACTOR_DIVERSIFY_POOL_SIZE),
        initial_capital=float(Settings.INITIAL_CAPITAL),
        taker_fee=float(Settings.TAKER_FEE),
        backtest_top_quantile=float(Settings.BACKTEST_TOP_QUANTILE),
        backtest_max_abs_weight=float(Settings.BACKTEST_MAX_ABS_WEIGHT),
        backtest_vol_lookback=int(Settings.BACKTEST_VOL_LOOKBACK),
        backtest_multi_factor_top_k=int(Settings.BACKTEST_MULTI_FACTOR_TOP_K),
        backtest_signal_smoothing_span=int(Settings.BACKTEST_SIGNAL_SMOOTHING_SPAN),
        backtest_execution_alpha=float(Settings.BACKTEST_EXECUTION_ALPHA),
        backtest_rebalance_interval=int(Settings.BACKTEST_REBALANCE_INTERVAL),
        backtest_target_gross_exposure=float(Settings.BACKTEST_TARGET_GROSS_EXPOSURE),
        backtest_save_position_events=bool(Settings.BACKTEST_SAVE_POSITION_EVENTS),
        backtest_position_event_threshold=float(Settings.BACKTEST_POSITION_EVENT_THRESHOLD),
        backtest_start_date=_normalize_optional_date(Settings.BACKTEST_START_DATE),
        backtest_end_date=_normalize_optional_date(Settings.BACKTEST_END_DATE),
    )


# Backward-compatible aliases.
RunConfig = RuntimeConfig


def load_config(path: str | Path | None = None) -> RuntimeConfig:
    _ = path
    return get_settings()
