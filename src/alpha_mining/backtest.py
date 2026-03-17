from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .evaluation import score_factor_summary, select_diversified_factors


EPS = 1e-9


@dataclass
class BacktestConfig:
    timeframe: str = "4h"
    horizon_bars: int = 12
    top_quantile: float = 0.2
    taker_fee: float = 0.001
    initial_capital: float = 100000.0
    max_abs_weight: float = 0.2
    vol_lookback: int = 24
    min_assets_per_timestamp: int = 8
    multi_factor_top_k: int = 8
    signal_smoothing_span: int = 6
    execution_alpha: float = 0.35
    rebalance_interval: int = 0
    target_gross_exposure: float = 1.0
    start_date: str | None = None
    end_date: str | None = None
    save_position_events: bool = True
    position_event_threshold: float = 1e-4
    factor_corr_threshold: float = 0.85
    factor_diversify_pool_size: int = 120


def _periods_per_year(timeframe: str) -> int:
    try:
        delta = pd.to_timedelta(timeframe.lower())
    except ValueError:
        return 365 * 6  # fallback for 4h-like data
    if delta <= pd.Timedelta(0):
        return 365 * 6
    return max(int(round(pd.Timedelta(days=365) / delta)), 1)


def _close_matrix(market_df: pd.DataFrame) -> pd.DataFrame:
    return market_df.pivot_table(
        index="timestamp",
        columns="symbol",
        values="close",
        aggfunc="last",
    ).sort_index()


def _returns_matrix(market_df: pd.DataFrame) -> pd.DataFrame:
    close = _close_matrix(market_df)
    return close.pct_change()


def _forward_one_bar_returns(market_df: pd.DataFrame) -> pd.DataFrame:
    return _returns_matrix(market_df).shift(-1)


def _smooth_signal(signal_wide: pd.DataFrame, span: int) -> pd.DataFrame:
    if span <= 1:
        return signal_wide
    return signal_wide.ewm(span=span, adjust=False, min_periods=1).mean()


def _signal_to_equal_weight_long_short(
    signal_wide: pd.DataFrame,
    top_quantile: float,
    min_assets: int,
) -> pd.DataFrame:
    signal_wide = signal_wide.replace([np.inf, -np.inf], np.nan)
    ranks = signal_wide.rank(axis=1, pct=True)
    weights = pd.DataFrame(0.0, index=signal_wide.index, columns=signal_wide.columns)

    for ts, row in ranks.iterrows():
        valid = row.dropna()
        if len(valid) < min_assets:
            continue
        q = min(max(top_quantile, 0.05), 0.45)
        long_mask = valid >= (1.0 - q)
        short_mask = valid <= q

        n_long = int(long_mask.sum())
        n_short = int(short_mask.sum())
        if n_long == 0 or n_short == 0:
            continue

        w = pd.Series(0.0, index=signal_wide.columns)
        w.loc[valid.index[long_mask]] = 0.5 / n_long
        w.loc[valid.index[short_mask]] = -0.5 / n_short
        weights.loc[ts] = w
    return weights


def _signal_to_risk_controlled_weights(
    signal_wide: pd.DataFrame,
    realized_returns: pd.DataFrame,
    max_abs_weight: float,
    vol_lookback: int,
    min_assets: int,
) -> pd.DataFrame:
    # Risk scaling uses only historical data up to t (shifted trailing vol).
    hist_ret = realized_returns.shift(1)
    vol = hist_ret.rolling(vol_lookback, min_periods=max(6, vol_lookback // 2)).std()
    inv_vol = 1.0 / (vol + EPS)

    zscore = signal_wide.sub(signal_wide.mean(axis=1), axis=0)
    zscore = zscore.div(signal_wide.std(axis=1) + EPS, axis=0)
    scaled = zscore * inv_vol
    scaled = scaled.sub(scaled.mean(axis=1), axis=0)

    gross = scaled.abs().sum(axis=1)
    weights = scaled.div(gross + EPS, axis=0)
    weights = weights.clip(lower=-max_abs_weight, upper=max_abs_weight)

    gross_after_clip = weights.abs().sum(axis=1)
    weights = weights.div(gross_after_clip + EPS, axis=0)

    valid_assets = signal_wide.notna().sum(axis=1)
    weights = weights.where(valid_assets >= min_assets, 0.0)
    return weights.fillna(0.0)


def _apply_execution_controls(
    target_weights: pd.DataFrame,
    rebalance_interval: int,
    execution_alpha: float,
    target_gross_exposure: float,
) -> pd.DataFrame:
    if target_weights.empty:
        return target_weights

    rebalance_n = max(int(rebalance_interval), 1)
    alpha = float(np.clip(execution_alpha, 0.05, 1.0))
    gross_target = float(max(target_gross_exposure, 0.0))

    out = pd.DataFrame(0.0, index=target_weights.index, columns=target_weights.columns)
    prev = pd.Series(0.0, index=target_weights.columns, dtype=float)

    for i, ts in enumerate(target_weights.index):
        tgt = target_weights.loc[ts].fillna(0.0)
        should_rebalance = i == 0 or (i % rebalance_n == 0)
        if should_rebalance:
            nxt = prev + alpha * (tgt - prev)
            nxt = nxt - nxt.mean()
            gross = float(nxt.abs().sum())
            if gross > EPS and gross_target > 0.0:
                nxt = nxt * (gross_target / gross)
            elif gross_target <= 0.0:
                nxt = nxt * 0.0
        else:
            nxt = prev

        out.loc[ts] = nxt
        prev = nxt
    return out


def _backtest_from_weights(
    strategy_name: str,
    weights: pd.DataFrame,
    returns_fwd: pd.DataFrame,
    taker_fee: float,
    start_ts: pd.Timestamp | None = None,
    end_ts: pd.Timestamp | None = None,
) -> pd.DataFrame:
    idx = returns_fwd.index.intersection(weights.index)
    if start_ts is not None:
        idx = idx[idx >= start_ts]
    if end_ts is not None:
        idx = idx[idx <= end_ts]
    cols = returns_fwd.columns.intersection(weights.columns)
    ret = returns_fwd.loc[idx, cols]
    w = weights.loc[idx, cols].fillna(0.0)

    gross_ret = (w * ret).sum(axis=1, min_count=1).fillna(0.0)
    turnover = w.diff().abs().sum(axis=1)
    if not turnover.empty:
        turnover.iloc[0] = w.iloc[0].abs().sum()
    turnover = turnover.fillna(0.0)

    cost = taker_fee * turnover
    net_ret = gross_ret - cost

    out = pd.DataFrame(
        {
            "timestamp": idx,
            "strategy": strategy_name,
            "gross_return": gross_ret.values,
            "cost": cost.values,
            "turnover": turnover.values,
            "net_return": net_ret.values,
        }
    )
    return out


def _event_type(prev_w: float, new_w: float, eps: float) -> str:
    prev_abs = abs(prev_w)
    new_abs = abs(new_w)
    prev_sign = int(np.sign(prev_w))
    new_sign = int(np.sign(new_w))

    if prev_abs <= eps and new_abs <= eps:
        return "no_position"
    if prev_abs <= eps and new_abs > eps:
        return "open_long" if new_sign > 0 else "open_short"
    if prev_abs > eps and new_abs <= eps:
        return "close_long" if prev_sign > 0 else "close_short"
    if prev_sign != 0 and new_sign != 0 and prev_sign != new_sign:
        return "flip_long_to_short" if prev_sign > 0 else "flip_short_to_long"
    if new_abs > prev_abs + eps:
        return "add_long" if new_sign > 0 else "add_short"
    if new_abs + eps < prev_abs:
        return "reduce_long" if new_sign > 0 else "reduce_short"
    return "rebalance"


def _trigger_reason(
    signal_value: float,
    signal_rank: float,
    mode: str,
    top_quantile: float | None,
) -> str:
    if not np.isfinite(signal_value):
        return "missing_signal"
    if mode == "quantile":
        if signal_rank is None or not np.isfinite(signal_rank):
            return "missing_rank"
        q = float(min(max(top_quantile or 0.2, 0.05), 0.45))
        if signal_rank >= 1.0 - q:
            return "long_quantile_signal"
        if signal_rank <= q:
            return "short_quantile_signal"
        return "execution_adjustment"

    if signal_value > 0:
        return "positive_signal"
    if signal_value < 0:
        return "negative_signal"
    return "neutral_signal"


def _build_position_events(
    strategy_name: str,
    weights: pd.DataFrame,
    signal_wide: pd.DataFrame,
    returns_fwd: pd.DataFrame,
    start_ts: pd.Timestamp | None,
    end_ts: pd.Timestamp | None,
    mode: str,
    top_quantile: float | None,
    threshold: float,
) -> pd.DataFrame:
    idx = returns_fwd.index.intersection(weights.index)
    if start_ts is not None:
        idx = idx[idx >= start_ts]
    if end_ts is not None:
        idx = idx[idx <= end_ts]
    if len(idx) == 0:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "strategy",
                "symbol",
                "event_type",
                "prev_weight",
                "new_weight",
                "delta_weight",
                "signal_value",
                "signal_rank",
                "trigger_reason",
            ]
        )

    cols = returns_fwd.columns.intersection(weights.columns)
    w = weights.loc[idx, cols].fillna(0.0)
    signal_aligned = signal_wide.reindex(index=idx, columns=cols)
    signal_rank = signal_aligned.rank(axis=1, pct=True)

    rows: list[dict[str, object]] = []
    prev = pd.Series(0.0, index=cols, dtype=float)
    eps = float(max(threshold, 1e-9))

    for ts in idx:
        curr = w.loc[ts].astype(float)
        delta = curr - prev
        changed = delta[delta.abs() >= eps]
        if changed.empty:
            prev = curr
            continue

        sig_row = signal_aligned.loc[ts]
        rank_row = signal_rank.loc[ts]
        for symbol, delta_weight in changed.items():
            prev_weight = float(prev.get(symbol, 0.0))
            new_weight = float(curr.get(symbol, 0.0))
            sig_value = float(sig_row.get(symbol)) if pd.notna(sig_row.get(symbol)) else np.nan
            sig_rank = float(rank_row.get(symbol)) if pd.notna(rank_row.get(symbol)) else np.nan
            rows.append(
                {
                    "timestamp": ts,
                    "strategy": strategy_name,
                    "symbol": symbol,
                    "event_type": _event_type(prev_weight, new_weight, eps),
                    "prev_weight": prev_weight,
                    "new_weight": new_weight,
                    "delta_weight": float(delta_weight),
                    "signal_value": sig_value,
                    "signal_rank": sig_rank,
                    "trigger_reason": _trigger_reason(sig_value, sig_rank, mode=mode, top_quantile=top_quantile),
                }
            )
        prev = curr

    if not rows:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "strategy",
                "symbol",
                "event_type",
                "prev_weight",
                "new_weight",
                "delta_weight",
                "signal_value",
                "signal_rank",
                "trigger_reason",
            ]
        )
    return pd.DataFrame(rows).sort_values(["timestamp", "strategy", "symbol"]).reset_index(drop=True)


def _parse_date_boundary(value: str | None) -> pd.Timestamp | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    ts = pd.to_datetime(text, utc=True, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"Invalid date value: {value}")
    return pd.Timestamp(ts)


def _summarize_strategy(
    strategy_returns: pd.DataFrame,
    periods_per_year: int,
    initial_capital: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    returns = strategy_returns["net_return"].fillna(0.0)
    nav = (1.0 + returns).cumprod()
    equity = nav * initial_capital

    running_max = nav.cummax()
    drawdown = nav / (running_max + EPS) - 1.0

    n = len(returns)
    total_return = float(nav.iloc[-1] - 1.0) if n > 0 else 0.0
    ann_return = float((1.0 + total_return) ** (periods_per_year / max(n, 1)) - 1.0) if n > 0 else 0.0
    ann_vol = float(returns.std(ddof=0) * np.sqrt(periods_per_year)) if n > 1 else 0.0
    sharpe = ann_return / (ann_vol + EPS)
    max_dd = float(drawdown.min()) if n > 0 else 0.0
    calmar = ann_return / (abs(max_dd) + EPS)
    win_rate = float((returns > 0).mean()) if n > 0 else 0.0

    summary = pd.DataFrame(
        {
            "strategy": [strategy_returns["strategy"].iloc[0]],
            "periods": [n],
            "total_return": [total_return],
            "annual_return": [ann_return],
            "annual_vol": [ann_vol],
            "sharpe": [sharpe],
            "max_drawdown": [max_dd],
            "calmar": [calmar],
            "win_rate": [win_rate],
            "avg_turnover": [float(strategy_returns["turnover"].mean()) if n > 0 else 0.0],
            "avg_cost": [float(strategy_returns["cost"].mean()) if n > 0 else 0.0],
            "final_equity": [float(equity.iloc[-1]) if n > 0 else initial_capital],
        }
    )

    nav_df = pd.DataFrame(
        {
            "timestamp": strategy_returns["timestamp"].values,
            "strategy": strategy_returns["strategy"].values,
            "nav": nav.values,
            "equity": equity.values,
            "drawdown": drawdown.values,
        }
    )
    return summary, nav_df


def _build_single_factor_signal(
    factor_df: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> tuple[str, pd.DataFrame]:
    scored = _prepare_factor_scoring(summary_df)
    if scored.empty:
        return "single_factor_long_short", pd.DataFrame()

    best = scored.iloc[0]
    factor_name = str(best["factor_name"])
    direction = float(best["direction"])
    signal = factor_df[factor_df["factor_name"] == factor_name].pivot_table(
        index="timestamp",
        columns="symbol",
        values="factor_value",
        aggfunc="last",
    )
    signal = signal * direction
    direction_label = "long_high" if direction >= 0 else "long_low"
    return f"single_factor_long_short[{factor_name}|{direction_label}]", signal.sort_index()


def _prepare_factor_scoring(summary_df: pd.DataFrame) -> pd.DataFrame:
    return score_factor_summary(summary_df)


def _build_composite_signal(
    composite_df: pd.DataFrame,
    factor_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    top_k: int,
    corr_threshold: float,
    diversify_pool_size: int,
) -> tuple[str, pd.DataFrame]:
    scored = _prepare_factor_scoring(summary_df)
    selected, _ = select_diversified_factors(
        scored_summary=scored,
        factor_df=factor_df,
        top_k=max(int(top_k), 1),
        corr_threshold=corr_threshold,
        pool_size=diversify_pool_size,
    )
    if selected:
        scored = scored[scored["factor_name"].isin(selected)].copy()
        scored = scored.set_index("factor_name").loc[selected].reset_index()
        factor_names = scored["factor_name"].tolist()
        weights = scored.set_index("factor_name")["score"] * scored.set_index("factor_name")["direction"]
        weights = weights / (weights.abs().sum() + EPS)

        tmp = factor_df[factor_df["factor_name"].isin(factor_names)].copy()
        if not tmp.empty:
            tmp["w"] = tmp["factor_name"].map(weights)
            tmp["signal"] = tmp["factor_value"] * tmp["w"]
            signal = tmp.groupby(["timestamp", "symbol"], as_index=False)["signal"].sum()
            wide = signal.pivot_table(
                index="timestamp",
                columns="symbol",
                values="signal",
                aggfunc="last",
            )
            return "composite_long_short", wide.sort_index()

    if composite_df.empty:
        return "composite_long_short", pd.DataFrame()
    signal = composite_df.pivot_table(
        index="timestamp",
        columns="symbol",
        values="composite_signal",
        aggfunc="last",
    )
    return "composite_long_short", signal.sort_index()


def _build_multi_factor_signal(
    factor_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    top_k: int,
    corr_threshold: float,
    diversify_pool_size: int,
) -> tuple[str, pd.DataFrame]:
    scored = _prepare_factor_scoring(summary_df)
    selected, _ = select_diversified_factors(
        scored_summary=scored,
        factor_df=factor_df,
        top_k=max(int(top_k), 1),
        corr_threshold=corr_threshold,
        pool_size=diversify_pool_size,
    )
    if not selected:
        return "risk_controlled_multi_factor", pd.DataFrame()
    scored = scored[scored["factor_name"].isin(selected)].copy()
    scored = scored.set_index("factor_name").loc[selected].reset_index()

    factor_names = scored["factor_name"].tolist()
    weights = scored.set_index("factor_name")["score"] * scored.set_index("factor_name")["direction"]
    weights = weights / (weights.abs().sum() + EPS)

    tmp = factor_df[factor_df["factor_name"].isin(factor_names)].copy()
    if tmp.empty:
        return "risk_controlled_multi_factor", pd.DataFrame()
    tmp["w"] = tmp["factor_name"].map(weights)
    tmp["signal"] = tmp["factor_value"] * tmp["w"]

    signal = tmp.groupby(["timestamp", "symbol"], as_index=False)["signal"].sum()
    wide = signal.pivot_table(
        index="timestamp",
        columns="symbol",
        values="signal",
        aggfunc="last",
    )
    return "risk_controlled_multi_factor", wide.sort_index()


def run_strategy_backtests(
    market_df: pd.DataFrame,
    factor_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    composite_df: pd.DataFrame,
    config: BacktestConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if market_df.empty:
        empty_ret = pd.DataFrame(columns=["timestamp", "strategy", "gross_return", "cost", "turnover", "net_return"])
        empty_nav = pd.DataFrame(columns=["timestamp", "strategy", "nav", "equity", "drawdown"])
        empty_summary = pd.DataFrame(
            columns=[
                "strategy",
                "periods",
                "total_return",
                "annual_return",
                "annual_vol",
                "sharpe",
                "max_drawdown",
                "calmar",
                "win_rate",
                "avg_turnover",
                "avg_cost",
                "final_equity",
            ]
        )
        empty_events = pd.DataFrame(
            columns=[
                "timestamp",
                "strategy",
                "symbol",
                "event_type",
                "prev_weight",
                "new_weight",
                "delta_weight",
                "signal_value",
                "signal_rank",
                "trigger_reason",
            ]
        )
        return empty_ret, empty_nav, empty_summary, empty_events

    realized_returns = _returns_matrix(market_df)
    returns_fwd = _forward_one_bar_returns(market_df)
    start_ts = _parse_date_boundary(config.start_date)
    end_ts = _parse_date_boundary(config.end_date)
    if start_ts is not None and end_ts is not None and start_ts > end_ts:
        raise ValueError(f"Invalid backtest range: start_date({config.start_date}) > end_date({config.end_date})")
    if returns_fwd.empty:
        empty_events = pd.DataFrame(
            columns=[
                "timestamp",
                "strategy",
                "symbol",
                "event_type",
                "prev_weight",
                "new_weight",
                "delta_weight",
                "signal_value",
                "signal_rank",
                "trigger_reason",
            ]
        )
        return (
            pd.DataFrame(columns=["timestamp", "strategy", "gross_return", "cost", "turnover", "net_return"]),
            pd.DataFrame(columns=["timestamp", "strategy", "nav", "equity", "drawdown"]),
            pd.DataFrame(columns=["strategy", "periods", "total_return", "annual_return", "annual_vol", "sharpe"]),
            empty_events,
        )

    strategy_returns_list: list[pd.DataFrame] = []
    event_frames: list[pd.DataFrame] = []
    rebalance_interval = config.rebalance_interval if config.rebalance_interval > 0 else max(config.horizon_bars, 1)

    s1_name, s1_signal = _build_single_factor_signal(factor_df=factor_df, summary_df=summary_df)
    if not s1_signal.empty:
        s1_signal = _smooth_signal(s1_signal, span=config.signal_smoothing_span)
        s1_weights = _signal_to_equal_weight_long_short(
            signal_wide=s1_signal,
            top_quantile=config.top_quantile,
            min_assets=config.min_assets_per_timestamp,
        )
        s1_weights = _apply_execution_controls(
            target_weights=s1_weights,
            rebalance_interval=rebalance_interval,
            execution_alpha=config.execution_alpha,
            target_gross_exposure=config.target_gross_exposure,
        )
        strategy_returns_list.append(
            _backtest_from_weights(
                strategy_name=s1_name,
                weights=s1_weights,
                returns_fwd=returns_fwd,
                taker_fee=config.taker_fee,
                start_ts=start_ts,
                end_ts=end_ts,
            )
        )
        if config.save_position_events:
            event_frames.append(
                _build_position_events(
                    strategy_name=s1_name,
                    weights=s1_weights,
                    signal_wide=s1_signal,
                    returns_fwd=returns_fwd,
                    start_ts=start_ts,
                    end_ts=end_ts,
                    mode="quantile",
                    top_quantile=config.top_quantile,
                    threshold=config.position_event_threshold,
                )
            )

    s2_name, s2_signal = _build_composite_signal(
        composite_df=composite_df,
        factor_df=factor_df,
        summary_df=summary_df,
        top_k=max(5, config.multi_factor_top_k // 2),
        corr_threshold=config.factor_corr_threshold,
        diversify_pool_size=config.factor_diversify_pool_size,
    )
    if not s2_signal.empty:
        s2_signal = _smooth_signal(s2_signal, span=config.signal_smoothing_span)
        s2_weights = _signal_to_equal_weight_long_short(
            signal_wide=s2_signal,
            top_quantile=config.top_quantile,
            min_assets=config.min_assets_per_timestamp,
        )
        s2_weights = _apply_execution_controls(
            target_weights=s2_weights,
            rebalance_interval=rebalance_interval,
            execution_alpha=config.execution_alpha,
            target_gross_exposure=config.target_gross_exposure,
        )
        strategy_returns_list.append(
            _backtest_from_weights(
                strategy_name=s2_name,
                weights=s2_weights,
                returns_fwd=returns_fwd,
                taker_fee=config.taker_fee,
                start_ts=start_ts,
                end_ts=end_ts,
            )
        )
        if config.save_position_events:
            event_frames.append(
                _build_position_events(
                    strategy_name=s2_name,
                    weights=s2_weights,
                    signal_wide=s2_signal,
                    returns_fwd=returns_fwd,
                    start_ts=start_ts,
                    end_ts=end_ts,
                    mode="quantile",
                    top_quantile=config.top_quantile,
                    threshold=config.position_event_threshold,
                )
            )

    s3_name, s3_signal = _build_multi_factor_signal(
        factor_df=factor_df,
        summary_df=summary_df,
        top_k=config.multi_factor_top_k,
        corr_threshold=config.factor_corr_threshold,
        diversify_pool_size=config.factor_diversify_pool_size,
    )
    if not s3_signal.empty:
        s3_signal = _smooth_signal(s3_signal, span=config.signal_smoothing_span)
        s3_weights = _signal_to_risk_controlled_weights(
            signal_wide=s3_signal,
            realized_returns=realized_returns,
            max_abs_weight=config.max_abs_weight,
            vol_lookback=config.vol_lookback,
            min_assets=config.min_assets_per_timestamp,
        )
        s3_weights = _apply_execution_controls(
            target_weights=s3_weights,
            rebalance_interval=rebalance_interval,
            execution_alpha=config.execution_alpha,
            target_gross_exposure=config.target_gross_exposure,
        )
        strategy_returns_list.append(
            _backtest_from_weights(
                strategy_name=s3_name,
                weights=s3_weights,
                returns_fwd=returns_fwd,
                taker_fee=config.taker_fee,
                start_ts=start_ts,
                end_ts=end_ts,
            )
        )
        if config.save_position_events:
            event_frames.append(
                _build_position_events(
                    strategy_name=s3_name,
                    weights=s3_weights,
                    signal_wide=s3_signal,
                    returns_fwd=returns_fwd,
                    start_ts=start_ts,
                    end_ts=end_ts,
                    mode="continuous",
                    top_quantile=None,
                    threshold=config.position_event_threshold,
                )
            )

    if not strategy_returns_list:
        empty_events = pd.DataFrame(
            columns=[
                "timestamp",
                "strategy",
                "symbol",
                "event_type",
                "prev_weight",
                "new_weight",
                "delta_weight",
                "signal_value",
                "signal_rank",
                "trigger_reason",
            ]
        )
        return (
            pd.DataFrame(columns=["timestamp", "strategy", "gross_return", "cost", "turnover", "net_return"]),
            pd.DataFrame(columns=["timestamp", "strategy", "nav", "equity", "drawdown"]),
            pd.DataFrame(columns=["strategy", "periods", "total_return", "annual_return", "annual_vol", "sharpe"]),
            empty_events,
        )

    strategy_returns = pd.concat(strategy_returns_list, ignore_index=True)
    if strategy_returns.empty:
        empty_events = pd.DataFrame(
            columns=[
                "timestamp",
                "strategy",
                "symbol",
                "event_type",
                "prev_weight",
                "new_weight",
                "delta_weight",
                "signal_value",
                "signal_rank",
                "trigger_reason",
            ]
        )
        return (
            pd.DataFrame(columns=["timestamp", "strategy", "gross_return", "cost", "turnover", "net_return"]),
            pd.DataFrame(columns=["timestamp", "strategy", "nav", "equity", "drawdown"]),
            pd.DataFrame(columns=["strategy", "periods", "total_return", "annual_return", "annual_vol", "sharpe"]),
            empty_events,
        )
    periods_per_year = _periods_per_year(config.timeframe)

    summary_list: list[pd.DataFrame] = []
    nav_list: list[pd.DataFrame] = []
    for _, grp in strategy_returns.groupby("strategy", sort=False):
        grp = grp.sort_values("timestamp").reset_index(drop=True)
        summary_part, nav_part = _summarize_strategy(
            strategy_returns=grp,
            periods_per_year=periods_per_year,
            initial_capital=config.initial_capital,
        )
        summary_list.append(summary_part)
        nav_list.append(nav_part)

    summary_df_out = pd.concat(summary_list, ignore_index=True).sort_values("sharpe", ascending=False).reset_index(drop=True)
    nav_df_out = pd.concat(nav_list, ignore_index=True).sort_values(["timestamp", "strategy"]).reset_index(drop=True)
    strategy_returns = strategy_returns.sort_values(["timestamp", "strategy"]).reset_index(drop=True)
    if event_frames:
        events_df = pd.concat(event_frames, ignore_index=True).sort_values(["timestamp", "strategy", "symbol"]).reset_index(drop=True)
    else:
        events_df = pd.DataFrame(
            columns=[
                "timestamp",
                "strategy",
                "symbol",
                "event_type",
                "prev_weight",
                "new_weight",
                "delta_weight",
                "signal_value",
                "signal_rank",
                "trigger_reason",
            ]
        )
    return strategy_returns, nav_df_out, summary_df_out, events_df
