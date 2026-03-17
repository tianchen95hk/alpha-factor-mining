from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd


EPS = 1e-9


def _rank_cross_section(wide: pd.DataFrame) -> pd.DataFrame:
    return wide.rank(axis=1, pct=True)


def _cross_sectional_zscore(wide: pd.DataFrame) -> pd.DataFrame:
    row_mean = wide.mean(axis=1)
    row_std = wide.std(axis=1)
    return wide.sub(row_mean, axis=0).div(row_std + EPS, axis=0)


def _factor_score(wide: pd.DataFrame) -> float:
    values = wide.to_numpy(dtype=float, copy=False)
    finite = np.isfinite(values)
    coverage = float(finite.mean())
    if coverage <= 0.0:
        return float("-inf")

    dispersion = float(np.nanstd(values))
    if not np.isfinite(dispersion):
        return float("-inf")
    return coverage * dispersion


def _select_factor_names(
    factors: dict[str, pd.DataFrame],
    max_count: int,
    pinned: list[str] | None = None,
) -> list[str]:
    if max_count <= 0:
        return []

    pinned_names = [name for name in (pinned or []) if name in factors]
    if len(pinned_names) >= max_count:
        return pinned_names[:max_count]

    scores = {name: _factor_score(wide) for name, wide in factors.items()}
    others = [name for name in factors if name not in pinned_names]
    others.sort(
        key=lambda name: (np.isfinite(scores[name]), scores[name], name),
        reverse=True,
    )
    keep = max_count - len(pinned_names)
    return pinned_names + others[:keep]


def _build_core_factors(
    close: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    open_: pd.DataFrame,
    volume: pd.DataFrame,
    funding_rate: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    returns = close.pct_change()
    dollar_volume = close * volume
    factors: dict[str, pd.DataFrame] = {}

    factors["mom_24"] = close.pct_change(24)
    factors["mom_72"] = close.pct_change(72)
    factors["short_reversal_6"] = -close.pct_change(6)

    rolling_mean_48 = close.rolling(48, min_periods=24).mean()
    rolling_std_48 = close.rolling(48, min_periods=24).std()
    factors["volatility_adjusted_trend"] = (close - rolling_mean_48) / (rolling_std_48 + EPS)

    intrabar = (close - open_) / (high - low + EPS)
    factors["intrabar_reversal"] = -intrabar.rolling(12, min_periods=6).mean()

    range_ratio = (high - low) / (close + EPS)
    factors["range_expansion"] = range_ratio.rolling(12, min_periods=6).mean()

    vol_norm = volume / (volume.rolling(48, min_periods=24).mean() + EPS)
    factors["volume_shock"] = vol_norm - 1.0

    amihud = (returns.abs() / (dollar_volume + EPS)).rolling(24, min_periods=12).mean()
    factors["illiquidity_reversal"] = -amihud

    factors["funding_crowding_unwind"] = -funding_rate.rolling(9, min_periods=3).mean()

    fr_mean = funding_rate.rolling(72, min_periods=24).mean()
    fr_std = funding_rate.rolling(72, min_periods=24).std()
    factors["funding_zscore_revert"] = -(funding_rate - fr_mean) / (fr_std + EPS)

    short_vol = returns.rolling(24, min_periods=12).std()
    long_vol = returns.rolling(168, min_periods=48).std()
    factors["vol_regime_reversion"] = -(short_vol / (long_vol + EPS) - 1.0)

    down_vol = returns.clip(upper=0.0).rolling(48, min_periods=24).std()
    total_vol = returns.rolling(48, min_periods=24).std()
    factors["downside_pressure_revert"] = -(down_vol / (total_vol + EPS) - 1.0)

    trend_volume_confirm = close.pct_change(24) * np.sign(vol_norm - 1.0)
    factors["trend_volume_confirmation"] = trend_volume_confirm
    return factors


def _build_expanded_factors(
    close: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    open_: pd.DataFrame,
    volume: pd.DataFrame,
    funding_rate: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    returns = close.pct_change()
    dollar_volume = close * volume
    factors: dict[str, pd.DataFrame] = {}

    for window in [6, 12, 24, 48, 72, 96, 168]:
        factors[f"mom_{window}"] = close.pct_change(window)

    for window in [3, 6, 12, 24]:
        factors[f"short_reversal_{window}"] = -close.pct_change(window)

    for window in [24, 72, 96]:
        mean_ = close.rolling(window, min_periods=max(12, window // 2)).mean()
        std_ = close.rolling(window, min_periods=max(12, window // 2)).std()
        factors[f"volatility_adjusted_trend_{window}"] = (close - mean_) / (std_ + EPS)

    intrabar = (close - open_) / (high - low + EPS)
    for window in [6, 24]:
        factors[f"intrabar_reversal_{window}"] = -intrabar.rolling(window, min_periods=max(3, window // 2)).mean()

    range_ratio = (high - low) / (close + EPS)
    for window in [6, 24, 48]:
        factors[f"range_expansion_{window}"] = range_ratio.rolling(window, min_periods=max(3, window // 2)).mean()

    for window in [24, 96]:
        vol_norm = volume / (volume.rolling(window, min_periods=max(12, window // 2)).mean() + EPS)
        factors[f"volume_shock_{window}"] = vol_norm - 1.0

    for window in [12, 48]:
        amihud = (returns.abs() / (dollar_volume + EPS)).rolling(window, min_periods=max(6, window // 2)).mean()
        factors[f"illiquidity_reversal_{window}"] = -amihud

    for window in [3, 6, 12]:
        factors[f"funding_crowding_unwind_{window}"] = -funding_rate.rolling(window, min_periods=max(2, window // 3)).mean()

    for window in [24, 48, 96]:
        fr_mean = funding_rate.rolling(window, min_periods=max(12, window // 3)).mean()
        fr_std = funding_rate.rolling(window, min_periods=max(12, window // 3)).std()
        factors[f"funding_zscore_revert_{window}"] = -(funding_rate - fr_mean) / (fr_std + EPS)

    for short_window, long_window in [(12, 72), (24, 120), (48, 240)]:
        short_vol = returns.rolling(short_window, min_periods=max(6, short_window // 2)).std()
        long_vol = returns.rolling(long_window, min_periods=max(24, long_window // 3)).std()
        factors[f"vol_regime_reversion_{short_window}_{long_window}"] = -(short_vol / (long_vol + EPS) - 1.0)

    for window in [24, 72]:
        down_vol = returns.clip(upper=0.0).rolling(window, min_periods=max(12, window // 2)).std()
        total_vol = returns.rolling(window, min_periods=max(12, window // 2)).std()
        factors[f"downside_pressure_revert_{window}"] = -(down_vol / (total_vol + EPS) - 1.0)

    for mom_window in [12, 24, 48]:
        for vol_window in [24, 96]:
            vol_norm = volume / (volume.rolling(vol_window, min_periods=max(12, vol_window // 2)).mean() + EPS)
            trend_volume_confirm = close.pct_change(mom_window) * np.sign(vol_norm - 1.0)
            factors[f"trend_volume_confirmation_{mom_window}_{vol_window}"] = trend_volume_confirm

    return factors


def _build_combo_factors(
    base_factors: dict[str, pd.DataFrame],
    max_combos: int,
    source_top_n: int,
) -> dict[str, pd.DataFrame]:
    if max_combos <= 0 or len(base_factors) < 2:
        return {}

    scores = {name: _factor_score(wide) for name, wide in base_factors.items()}
    ranked_names = sorted(
        base_factors,
        key=lambda name: (np.isfinite(scores[name]), scores[name], name),
        reverse=True,
    )
    source_names = ranked_names[: max(2, source_top_n)]
    zscored = {name: _cross_sectional_zscore(base_factors[name]) for name in source_names}

    combos: dict[str, pd.DataFrame] = {}
    for left_name, right_name in combinations(source_names, 2):
        left = zscored[left_name]
        right = zscored[right_name]
        for op_name, op in (
            ("add", lambda a, b: a + b),
            ("sub", lambda a, b: a - b),
            ("mul", lambda a, b: a * b),
        ):
            combo_name = f"combo_{op_name}_{left_name}__{right_name}"
            combos[combo_name] = op(left, right).replace([np.inf, -np.inf], np.nan)
            if len(combos) >= max_combos:
                return combos
    return combos


def _to_long(name: str, wide: pd.DataFrame) -> pd.DataFrame:
    ranked = _rank_cross_section(wide)
    long_df = ranked.stack(future_stack=True).rename("factor_value").reset_index()
    long_df.columns = ["timestamp", "symbol", "factor_value"]
    long_df["factor_name"] = name
    return long_df[["timestamp", "symbol", "factor_name", "factor_value"]]


def _build_market_wides(
    market_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    market_df = market_df.drop_duplicates(subset=["timestamp", "symbol"])
    close = market_df.pivot_table(
        index="timestamp",
        columns="symbol",
        values="close",
        aggfunc="last",
    ).sort_index()
    high = market_df.pivot_table(
        index="timestamp",
        columns="symbol",
        values="high",
        aggfunc="last",
    ).sort_index()
    low = market_df.pivot_table(
        index="timestamp",
        columns="symbol",
        values="low",
        aggfunc="last",
    ).sort_index()
    open_ = market_df.pivot_table(
        index="timestamp",
        columns="symbol",
        values="open",
        aggfunc="last",
    ).sort_index()
    volume = market_df.pivot_table(
        index="timestamp",
        columns="symbol",
        values="volume",
        aggfunc="last",
    ).sort_index()
    funding_rate = market_df.pivot_table(
        index="timestamp",
        columns="symbol",
        values="funding_rate",
        aggfunc="last",
    ).sort_index()
    return close, high, low, open_, volume, funding_rate


def _build_factor_pool(
    market_df: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    close, high, low, open_, volume, funding_rate = _build_market_wides(market_df)

    core_factors = _build_core_factors(
        close=close,
        high=high,
        low=low,
        open_=open_,
        volume=volume,
        funding_rate=funding_rate,
    )
    expanded_factors = _build_expanded_factors(
        close=close,
        high=high,
        low=low,
        open_=open_,
        volume=volume,
        funding_rate=funding_rate,
    )

    all_base: dict[str, pd.DataFrame] = dict(core_factors)
    for name, wide in expanded_factors.items():
        if name not in all_base:
            all_base[name] = wide
    return all_base, list(core_factors.keys())


def _parse_combo_factor_name(name: str) -> tuple[str, str, str] | None:
    for op_name in ("add", "sub", "mul"):
        prefix = f"combo_{op_name}_"
        if not name.startswith(prefix):
            continue
        pair = name[len(prefix) :]
        if "__" not in pair:
            return None
        left_name, right_name = pair.split("__", 1)
        if not left_name or not right_name:
            return None
        return op_name, left_name, right_name
    return None


def compute_selected_factor_library(
    market_df: pd.DataFrame,
    factor_names: list[str],
) -> pd.DataFrame:
    selected = []
    seen: set[str] = set()
    for name in factor_names:
        n = str(name).strip()
        if not n or n in seen:
            continue
        selected.append(n)
        seen.add(n)
    if not selected:
        return pd.DataFrame(columns=["timestamp", "symbol", "factor_name", "factor_value"])

    base_factors, _ = _build_factor_pool(market_df)
    out: dict[str, pd.DataFrame] = {}

    for factor_name in selected:
        if factor_name in base_factors:
            out[factor_name] = base_factors[factor_name]
            continue

        combo = _parse_combo_factor_name(factor_name)
        if combo is None:
            continue
        op_name, left_name, right_name = combo
        if left_name not in base_factors or right_name not in base_factors:
            continue

        left = _cross_sectional_zscore(base_factors[left_name])
        right = _cross_sectional_zscore(base_factors[right_name])
        if op_name == "add":
            wide = left + right
        elif op_name == "sub":
            wide = left - right
        else:
            wide = left * right
        out[factor_name] = wide.replace([np.inf, -np.inf], np.nan)

    if not out:
        return pd.DataFrame(columns=["timestamp", "symbol", "factor_name", "factor_value"])

    out_frames = [_to_long(name, wide) for name, wide in out.items()]
    factor_df = pd.concat(out_frames, ignore_index=True)
    factor_df = factor_df.dropna(subset=["factor_value"]).reset_index(drop=True)
    return factor_df


def compute_factor_library(
    market_df: pd.DataFrame,
    max_factors: int = 200,
    enable_combos: bool = True,
    combo_source_top_n: int = 24,
    max_combo_factors: int = 120,
) -> pd.DataFrame:
    all_base, core_names = _build_factor_pool(market_df)

    combo_room = min(max_combo_factors, max(max_factors - len(core_names), 0)) if enable_combos else 0
    base_limit = max(max_factors - combo_room, 0)
    base_names = _select_factor_names(
        all_base,
        max_count=base_limit,
        pinned=core_names,
    )
    final_factors: dict[str, pd.DataFrame] = {name: all_base[name] for name in base_names}

    if enable_combos and len(final_factors) < max_factors:
        combo_limit = min(max_factors - len(final_factors), combo_room)
        combo_factors = _build_combo_factors(
            base_factors=final_factors,
            max_combos=combo_limit,
            source_top_n=combo_source_top_n,
        )
        final_factors.update(combo_factors)

    out_frames = [_to_long(name, wide) for name, wide in final_factors.items()]
    factor_df = pd.concat(out_frames, ignore_index=True)
    factor_df = factor_df.dropna(subset=["factor_value"]).reset_index(drop=True)
    return factor_df
