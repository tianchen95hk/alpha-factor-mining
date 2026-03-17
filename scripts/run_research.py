from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from alpha_mining.pipeline import run_research_pipeline



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crypto factor mining pipeline (Binance USD-M)")
    parser.add_argument(
        "--config",
        default=None,
        help="Deprecated. Settings are now read from config/settings.py",
    )
    parser.add_argument(
        "--use-cache-only",
        action="store_true",
        help="Use local cache only and skip exchange fetch",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Force refresh market cache from Binance",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Top-k factors for composite signal (default from config/settings.py)",
    )
    parser.add_argument(
        "--max-factors",
        type=int,
        default=None,
        help="Maximum number of generated factors (default from config/settings.py)",
    )
    parser.add_argument(
        "--disable-combos",
        action="store_true",
        help="Disable generated combo factors",
    )
    parser.add_argument(
        "--strategy-only",
        action="store_true",
        help="Skip full factor mining/evaluation and run strategy with selected factors only",
    )
    parser.add_argument(
        "--selected-factors",
        default="",
        help="Comma-separated factor names used in --strategy-only mode",
    )
    parser.add_argument(
        "--selected-top-n",
        type=int,
        default=12,
        help="Use top-N factors from factor_summary.csv when --strategy-only and --selected-factors is empty",
    )
    parser.add_argument(
        "--no-charts",
        action="store_true",
        help="Skip chart rendering for faster runs",
    )
    parser.add_argument(
        "--save-selected-factors",
        action="store_true",
        help="Save selected_factor_values.csv in --strategy-only mode",
    )
    parser.add_argument(
        "--start-date",
        default=None,
        help="Backtest start date, e.g. 2024-01-01 (overrides config)",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="Backtest end date, e.g. 2025-12-31 (overrides config)",
    )
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    selected_factors = [x.strip() for x in args.selected_factors.split(",") if x.strip()]
    outputs = run_research_pipeline(
        config_path=args.config,
        use_cache_only=args.use_cache_only,
        refresh_cache=args.refresh_cache,
        top_k=args.top_k,
        max_factors=args.max_factors,
        enable_combos=False if args.disable_combos else None,
        strategy_only=args.strategy_only,
        selected_factors=selected_factors if selected_factors else None,
        selected_top_n=args.selected_top_n,
        generate_charts=not args.no_charts,
        save_selected_factor_values=args.save_selected_factors,
        backtest_start_date=args.start_date,
        backtest_end_date=args.end_date,
    )

    summary = outputs["summary"]
    factor_count = outputs["factors"]["factor_name"].nunique()
    mode_text = "strategy-only" if outputs.get("strategy_only") else "full-research"
    print(f"\nRun mode: {mode_text}")
    if outputs.get("selected_factors"):
        print(f"Selected factors ({len(outputs['selected_factors'])}): {', '.join(outputs['selected_factors'])}")
    print(
        "Backtest range: "
        f"{outputs.get('backtest_start_date') or 'beginning'} -> {outputs.get('backtest_end_date') or 'latest'}"
    )

    print(f"\nGenerated factors: {factor_count}")
    print("\nTop factors by mean IC:")
    if summary.empty:
        print("No valid factor evaluation.")
    else:
        print(summary.head(10).to_string(index=False))

    top_factors = outputs.get("top_factors")
    print("\nTop strategy factors:")
    if top_factors is None or getattr(top_factors, "empty", True):
        print("No top factor output.")
    else:
        print(top_factors.head(20).to_string(index=False))

    strategy_summary = outputs["strategy_summary"]
    best_strategy_name = None
    print("\nStrategy backtest comparison:")
    if strategy_summary.empty:
        print("No valid strategy backtest output.")
    else:
        print(strategy_summary.to_string(index=False))
        best = strategy_summary.iloc[0]
        best_strategy_name = str(best["strategy"])
        print(
            f"\nBest strategy: {best['strategy']} | Sharpe={best['sharpe']:.3f} | "
            f"AnnualReturn={best['annual_return']:.2%} | MaxDD={best['max_drawdown']:.2%}"
        )

    strategy_events = outputs.get("strategy_events")
    if strategy_events is not None:
        print(f"\nPosition events: {len(strategy_events)} rows")
        if not getattr(strategy_events, "empty", True):
            sample = strategy_events
            if best_strategy_name is not None:
                sample = sample[sample["strategy"] == best_strategy_name]
            sample = sample.tail(10)
            print("\nRecent position events (sample):")
            print(
                sample[
                    [
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
                ].to_string(index=False)
            )

    chart_files = outputs.get("chart_files", [])
    print("\nBacktest charts:")
    if chart_files:
        for name in chart_files:
            print(f"- {name}")
    else:
        print("No chart generated (matplotlib may be missing).")

    print("\nSaved files under results/:")
    if outputs.get("strategy_only"):
        if outputs.get("saved_selected_factor_values"):
            print("- selected_factor_values.csv")
    else:
        print("- factor_values.csv")
    print("- factor_summary.csv")
    print("- factor_summary_scoring.csv")
    print("- factor_summary_by_horizon.csv")
    print("- factor_summary_multi_horizon.csv")
    print("- factor_ic_timeseries.csv")
    print("- factor_ic_timeseries_by_horizon.csv")
    print("- factor_spread_timeseries.csv")
    print("- factor_spread_timeseries_by_horizon.csv")
    print("- composite_signal.csv")
    print("- top_strategy_factors.csv")
    print("- factor_diversification.csv")
    print("- factor_effectiveness_report.csv")
    print("- strategy_returns.csv")
    print("- strategy_nav.csv")
    print("- strategy_summary.csv")
    print("- strategy_comparison_report.csv")
    print("- strategy_position_events.csv")
    print("- auto_comparison_report.md")


if __name__ == "__main__":
    main()
