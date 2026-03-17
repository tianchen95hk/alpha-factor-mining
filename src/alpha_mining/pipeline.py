from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import BacktestConfig, run_strategy_backtests
from .config import RuntimeConfig, get_settings
from .data_loader import BinanceUSDMDataLoader, load_market_cache, save_market_cache
from .evaluation import (
    build_composite_signal,
    evaluate_factors,
    evaluate_factors_multi_horizon,
    score_factor_summary,
    select_diversified_factors,
)
from .factors import compute_factor_library, compute_selected_factor_library
from .visualization import render_backtest_charts


def _select_from_summary(summary_df: pd.DataFrame, top_n: int) -> list[str]:
    if top_n <= 0:
        return []
    scored = score_factor_summary(summary_df)
    if scored.empty:
        return []
    return scored["factor_name"].dropna().astype(str).head(top_n).tolist()


def _build_top_factor_table(
    scored_summary: pd.DataFrame,
    selected_factors: list[str],
    top_n: int,
) -> pd.DataFrame:
    if selected_factors:
        if scored_summary.empty:
            return pd.DataFrame({"factor_name": selected_factors, "score": np.nan, "direction": np.nan})
        top = scored_summary[scored_summary["factor_name"].isin(selected_factors)].copy()
        top = top.sort_values("score", ascending=False).reset_index(drop=True)
        missing = [name for name in selected_factors if name not in set(top["factor_name"])]
        if missing:
            missing_df = pd.DataFrame({"factor_name": missing, "score": np.nan, "direction": np.nan})
            top = pd.concat([top, missing_df], ignore_index=True)
        return top

    if scored_summary.empty or top_n <= 0:
        return pd.DataFrame(columns=["factor_name", "score", "direction"])
    return scored_summary.head(top_n).reset_index(drop=True)


def _compile_factor_effectiveness_table(
    top_factor_df: pd.DataFrame,
    summary_scoring_df: pd.DataFrame,
    summary_primary_df: pd.DataFrame,
    diversification_df: pd.DataFrame,
) -> pd.DataFrame:
    out = top_factor_df.copy()
    if out.empty:
        return out

    score_cols = [
        "factor_name",
        "mean_ic",
        "ic_ir",
        "hit_rate",
        "t_stat",
        "spread_mean",
        "spread_ir",
        "horizon_count",
        "ic_sign_consistency",
    ]
    existing_score_cols = [c for c in score_cols if c in summary_scoring_df.columns and c not in out.columns]
    if existing_score_cols:
        out = out.merge(summary_scoring_df[existing_score_cols], on="factor_name", how="left")

    primary_cols = ["factor_name", "mean_ic", "ic_ir", "spread_mean", "spread_ir", "obs"]
    existing_primary_cols = [c for c in primary_cols if c in summary_primary_df.columns]
    if existing_primary_cols:
        pri = summary_primary_df[existing_primary_cols].copy()
        rename_map = {c: f"{c}_primary" for c in existing_primary_cols if c != "factor_name"}
        pri = pri.rename(columns=rename_map)
        out = out.merge(pri, on="factor_name", how="left")

    diag_cols = ["factor_name", "selected", "max_abs_corr_to_selected", "reason"]
    if not diversification_df.empty and all(c in diversification_df.columns for c in diag_cols):
        diag_merge_cols = [c for c in diag_cols if c not in out.columns or c == "factor_name"]
        out = out.merge(diversification_df[diag_cols], on="factor_name", how="left")
        if diag_merge_cols != diag_cols:
            extra_cols = [c for c in diag_cols if c not in diag_merge_cols and c != "factor_name"]
            for col in extra_cols:
                if f"{col}_x" in out.columns and f"{col}_y" in out.columns:
                    out[col] = out[f"{col}_x"].fillna(out[f"{col}_y"])
                    out = out.drop(columns=[f"{col}_x", f"{col}_y"], errors="ignore")

    return out


def _save_auto_comparison_report(
    results_dir: Path,
    effective_eval_horizons: list[int],
    factor_effectiveness_df: pd.DataFrame,
    strategy_summary_df: pd.DataFrame,
) -> str:
    report_path = results_dir / "auto_comparison_report.md"
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    lines: list[str] = []
    lines.append("# Auto Comparison Report")
    lines.append("")
    lines.append(f"- Generated at: {now_utc}")
    lines.append(f"- Evaluation horizons (bars): {', '.join(str(x) for x in effective_eval_horizons)}")
    lines.append("")

    lines.append("## Factor Effectiveness")
    if factor_effectiveness_df.empty:
        lines.append("No factor effectiveness output.")
    else:
        lines.append("```text")
        lines.append(factor_effectiveness_df.head(30).to_string(index=False))
        lines.append("```")

    lines.append("")
    lines.append("## Strategy Comparison")
    if strategy_summary_df.empty:
        lines.append("No strategy summary output.")
    else:
        lines.append("```text")
        lines.append(strategy_summary_df.to_string(index=False))
        lines.append("```")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path.name


def run_research_pipeline(
    config_path: str | None = None,
    use_cache_only: bool = False,
    refresh_cache: bool = False,
    top_k: int | None = None,
    max_factors: int | None = None,
    enable_combos: bool | None = None,
    strategy_only: bool = False,
    selected_factors: list[str] | None = None,
    selected_top_n: int = 12,
    generate_charts: bool = True,
    save_selected_factor_values: bool = False,
    backtest_start_date: str | None = None,
    backtest_end_date: str | None = None,
) -> dict[str, object]:
    _ = config_path
    cfg: RuntimeConfig = get_settings()
    effective_top_k = top_k if top_k is not None else cfg.top_k_factors
    effective_max_factors = max_factors if max_factors is not None else cfg.factor_max_count
    effective_enable_combos = enable_combos if enable_combos is not None else cfg.factor_enable_combos
    effective_backtest_start_date = backtest_start_date if backtest_start_date is not None else cfg.backtest_start_date
    effective_backtest_end_date = backtest_end_date if backtest_end_date is not None else cfg.backtest_end_date
    effective_eval_horizons = sorted(set([int(x) for x in cfg.factor_eval_horizons if int(x) > 0] + [cfg.horizon_bars]))

    api_key = cfg.api_key
    api_secret = cfg.secret_key

    cache_file = Path(cfg.cache_path)
    results_dir = Path(cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if use_cache_only:
        if not cache_file.exists():
            raise FileNotFoundError(f"Cache not found: {cache_file}")
        market_df = load_market_cache(cache_file)
    else:
        if cache_file.exists() and not refresh_cache:
            market_df = load_market_cache(cache_file)
        else:
            loader = BinanceUSDMDataLoader(api_key=api_key, api_secret=api_secret)
            universe = loader.get_universe(cfg.symbols, cfg.top_n_symbols)
            market_df = loader.fetch_market_data(
                symbols=universe,
                timeframe=cfg.timeframe,
                lookback_days=cfg.lookback_days,
            )
            if market_df.empty:
                raise RuntimeError("No market data fetched from Binance USD-M.")
            save_market_cache(market_df, cache_file)

    if strategy_only:
        provided = [x.strip() for x in (selected_factors or []) if str(x).strip()]
        summary_cache = pd.DataFrame()
        summary_cache_path = results_dir / "factor_summary_multi_horizon.csv"
        if summary_cache_path.exists():
            summary_cache = pd.read_csv(summary_cache_path)
        elif (results_dir / "factor_summary.csv").exists():
            summary_cache = pd.read_csv(results_dir / "factor_summary.csv")

        selected = provided if provided else _select_from_summary(summary_cache, top_n=selected_top_n)
        if not selected:
            raise RuntimeError(
                "No selected factors found. Provide --selected-factors or run full research once to create factor_summary.csv."
            )

        factor_df = compute_selected_factor_library(
            market_df=market_df,
            factor_names=selected,
        )
        if factor_df.empty:
            raise RuntimeError("Selected factor generation returned empty result.")
    else:
        selected = []
        factor_df = compute_factor_library(
            market_df=market_df,
            max_factors=effective_max_factors,
            enable_combos=effective_enable_combos,
            combo_source_top_n=cfg.factor_combo_source_top_n,
            max_combo_factors=cfg.factor_combo_max_count,
        )
        if factor_df.empty:
            raise RuntimeError("Factor generation returned empty result.")

    summary_multi_df, summary_by_horizon_df, ic_by_horizon_df, spread_by_horizon_df = evaluate_factors_multi_horizon(
        factor_df=factor_df,
        market_df=market_df,
        horizons=effective_eval_horizons,
        min_assets_per_timestamp=cfg.min_assets_per_timestamp,
    )

    if not summary_by_horizon_df.empty:
        summary_df = summary_by_horizon_df[summary_by_horizon_df["horizon_bars"] == cfg.horizon_bars].copy()
        if summary_df.empty:
            primary_h = int(summary_by_horizon_df["horizon_bars"].iloc[0])
            summary_df = summary_by_horizon_df[summary_by_horizon_df["horizon_bars"] == primary_h].copy()
        summary_df = summary_df.drop(columns=["horizon_bars"], errors="ignore").reset_index(drop=True)

        ic_ts_df = ic_by_horizon_df[ic_by_horizon_df["horizon_bars"] == cfg.horizon_bars].copy()
        if ic_ts_df.empty:
            primary_h = int(ic_by_horizon_df["horizon_bars"].iloc[0])
            ic_ts_df = ic_by_horizon_df[ic_by_horizon_df["horizon_bars"] == primary_h].copy()
        ic_ts_df = ic_ts_df.drop(columns=["horizon_bars"], errors="ignore").reset_index(drop=True)

        spread_df = spread_by_horizon_df[spread_by_horizon_df["horizon_bars"] == cfg.horizon_bars].copy()
        if spread_df.empty:
            primary_h = int(spread_by_horizon_df["horizon_bars"].iloc[0])
            spread_df = spread_by_horizon_df[spread_by_horizon_df["horizon_bars"] == primary_h].copy()
        spread_df = spread_df.drop(columns=["horizon_bars"], errors="ignore").reset_index(drop=True)
    else:
        summary_df, ic_ts_df, spread_df = evaluate_factors(
            factor_df=factor_df,
            market_df=market_df,
            horizon_bars=cfg.horizon_bars,
            min_assets_per_timestamp=cfg.min_assets_per_timestamp,
        )

    summary_for_scoring = summary_multi_df if not summary_multi_df.empty else summary_df
    scored_summary = score_factor_summary(summary_for_scoring)

    if strategy_only:
        top_selected = selected
        diversification_df = pd.DataFrame(
            {
                "factor_name": top_selected,
                "selected": True,
                "max_abs_corr_to_selected": np.nan,
                "reason": "manual_selection",
            }
        )
    else:
        top_selected, diversification_df = select_diversified_factors(
            scored_summary=scored_summary,
            factor_df=factor_df,
            top_k=max(selected_top_n, 1),
            corr_threshold=cfg.factor_corr_threshold,
            pool_size=cfg.factor_diversify_pool_size,
        )

    top_factor_df = _build_top_factor_table(
        scored_summary=scored_summary,
        selected_factors=top_selected,
        top_n=max(selected_top_n, 1),
    )
    if not diversification_df.empty:
        top_factor_df = top_factor_df.merge(
            diversification_df[["factor_name", "max_abs_corr_to_selected", "reason"]],
            on="factor_name",
            how="left",
        )

    composite_df = build_composite_signal(
        factor_df=factor_df,
        summary_df=summary_for_scoring,
        top_k=effective_top_k,
        selected_factors=selected if strategy_only and selected else None,
        corr_threshold=cfg.factor_corr_threshold,
        diversify_pool_size=cfg.factor_diversify_pool_size,
    )

    strategy_returns_df, strategy_nav_df, strategy_summary_df, strategy_events_df = run_strategy_backtests(
        market_df=market_df,
        factor_df=factor_df,
        summary_df=summary_for_scoring,
        composite_df=composite_df,
        config=BacktestConfig(
            timeframe=cfg.timeframe,
            horizon_bars=cfg.horizon_bars,
            top_quantile=cfg.backtest_top_quantile,
            taker_fee=cfg.taker_fee,
            initial_capital=cfg.initial_capital,
            max_abs_weight=cfg.backtest_max_abs_weight,
            vol_lookback=cfg.backtest_vol_lookback,
            min_assets_per_timestamp=cfg.min_assets_per_timestamp,
            multi_factor_top_k=cfg.backtest_multi_factor_top_k,
            signal_smoothing_span=cfg.backtest_signal_smoothing_span,
            execution_alpha=cfg.backtest_execution_alpha,
            rebalance_interval=cfg.backtest_rebalance_interval,
            target_gross_exposure=cfg.backtest_target_gross_exposure,
            start_date=effective_backtest_start_date,
            end_date=effective_backtest_end_date,
            save_position_events=cfg.backtest_save_position_events,
            position_event_threshold=cfg.backtest_position_event_threshold,
            factor_corr_threshold=cfg.factor_corr_threshold,
            factor_diversify_pool_size=cfg.factor_diversify_pool_size,
        ),
    )

    if strategy_only:
        if save_selected_factor_values:
            factor_df.to_csv(results_dir / "selected_factor_values.csv", index=False)
    else:
        factor_df.to_csv(results_dir / "factor_values.csv", index=False)

    summary_df.to_csv(results_dir / "factor_summary.csv", index=False)
    summary_for_scoring.to_csv(results_dir / "factor_summary_scoring.csv", index=False)
    summary_by_horizon_df.to_csv(results_dir / "factor_summary_by_horizon.csv", index=False)
    summary_multi_df.to_csv(results_dir / "factor_summary_multi_horizon.csv", index=False)
    ic_ts_df.to_csv(results_dir / "factor_ic_timeseries.csv", index=False)
    ic_by_horizon_df.to_csv(results_dir / "factor_ic_timeseries_by_horizon.csv", index=False)
    spread_df.to_csv(results_dir / "factor_spread_timeseries.csv", index=False)
    spread_by_horizon_df.to_csv(results_dir / "factor_spread_timeseries_by_horizon.csv", index=False)
    composite_df.to_csv(results_dir / "composite_signal.csv", index=False)
    strategy_returns_df.to_csv(results_dir / "strategy_returns.csv", index=False)
    strategy_nav_df.to_csv(results_dir / "strategy_nav.csv", index=False)
    strategy_summary_df.to_csv(results_dir / "strategy_summary.csv", index=False)
    strategy_events_df.to_csv(results_dir / "strategy_position_events.csv", index=False)
    top_factor_df.to_csv(results_dir / "top_strategy_factors.csv", index=False)
    diversification_df.to_csv(results_dir / "factor_diversification.csv", index=False)

    factor_effectiveness_df = _compile_factor_effectiveness_table(
        top_factor_df=top_factor_df,
        summary_scoring_df=summary_for_scoring,
        summary_primary_df=summary_df,
        diversification_df=diversification_df,
    )
    factor_effectiveness_df.to_csv(results_dir / "factor_effectiveness_report.csv", index=False)
    strategy_comparison_df = strategy_summary_df.sort_values("sharpe", ascending=False).reset_index(drop=True)
    strategy_comparison_df.to_csv(results_dir / "strategy_comparison_report.csv", index=False)
    auto_report_file = _save_auto_comparison_report(
        results_dir=results_dir,
        effective_eval_horizons=effective_eval_horizons,
        factor_effectiveness_df=factor_effectiveness_df,
        strategy_summary_df=strategy_comparison_df,
    )

    chart_files: list[str] = []
    if generate_charts:
        chart_files = render_backtest_charts(
            strategy_returns_df=strategy_returns_df,
            strategy_nav_df=strategy_nav_df,
            strategy_summary_df=strategy_summary_df,
            market_df=market_df,
            output_dir=results_dir,
            timeframe=cfg.timeframe,
            initial_capital=cfg.initial_capital,
        )

    return {
        "market": market_df,
        "factors": factor_df,
        "summary": summary_df,
        "summary_scoring": summary_for_scoring,
        "summary_by_horizon": summary_by_horizon_df,
        "summary_multi": summary_multi_df,
        "ic_ts": ic_ts_df,
        "ic_ts_by_horizon": ic_by_horizon_df,
        "spread_ts": spread_df,
        "spread_ts_by_horizon": spread_by_horizon_df,
        "composite": composite_df,
        "strategy_returns": strategy_returns_df,
        "strategy_nav": strategy_nav_df,
        "strategy_summary": strategy_summary_df,
        "strategy_events": strategy_events_df,
        "chart_files": chart_files,
        "selected_factors": selected,
        "strategy_only": strategy_only,
        "saved_selected_factor_values": bool(strategy_only and save_selected_factor_values),
        "backtest_start_date": effective_backtest_start_date,
        "backtest_end_date": effective_backtest_end_date,
        "top_factors": top_factor_df,
        "diversification": diversification_df,
        "factor_effectiveness": factor_effectiveness_df,
        "auto_report_file": auto_report_file,
    }
