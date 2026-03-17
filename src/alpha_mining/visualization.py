from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

CHART_FILENAMES = [
    "backtest_equity_curve.png",
    "backtest_equity_vs_btc.png",
    "backtest_drawdown.png",
    "backtest_rolling_sharpe.png",
    "backtest_strategy_metrics.png",
]


def _pick_btc_symbol(market_df: pd.DataFrame) -> str | None:
    if market_df.empty or "symbol" not in market_df.columns:
        return None
    symbols = market_df["symbol"].dropna().astype(str).drop_duplicates()
    if symbols.empty:
        return None

    preferred = ["BTCUSDT", "BTC/USDT:USDT"]
    available = set(symbols.tolist())
    for name in preferred:
        if name in available:
            return name

    for sym in symbols:
        upper = sym.upper()
        if upper.startswith("BTC") and "USDT" in upper:
            return sym
    return None


def _periods_per_year(timeframe: str) -> int:
    try:
        delta = pd.to_timedelta(timeframe.lower())
    except ValueError:
        return 365 * 6
    if delta <= pd.Timedelta(0):
        return 365 * 6
    return max(int(round(pd.Timedelta(days=365) / delta)), 1)


def render_backtest_charts(
    strategy_returns_df: pd.DataFrame,
    strategy_nav_df: pd.DataFrame,
    strategy_summary_df: pd.DataFrame,
    market_df: pd.DataFrame,
    output_dir: str | Path,
    timeframe: str,
    initial_capital: float = 100000.0,
) -> list[str]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mpl_cache = out_dir / ".mpl_cache"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
    os.environ.setdefault("MPLBACKEND", "Agg")

    # Always overwrite previous test charts from last run.
    for name in CHART_FILENAMES:
        chart_path = out_dir / name
        if chart_path.exists():
            chart_path.unlink()

    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return []

    if strategy_nav_df.empty:
        return []
    created: list[str] = []

    nav_wide = strategy_nav_df.pivot_table(
        index="timestamp",
        columns="strategy",
        values="equity",
        aggfunc="last",
    ).sort_index()
    if not nav_wide.empty:
        fig, ax = plt.subplots(figsize=(12, 6))
        nav_wide.plot(ax=ax, linewidth=1.6)
        ax.set_title("Backtest Equity Curve")
        ax.set_ylabel("Equity")
        ax.set_xlabel("Timestamp")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        name = CHART_FILENAMES[0]
        fig.savefig(out_dir / name, dpi=160)
        plt.close(fig)
        created.append(name)

        btc_symbol = _pick_btc_symbol(market_df)
        btc = (
            market_df[market_df["symbol"] == btc_symbol][["timestamp", "close"]].dropna()
            if btc_symbol is not None
            else pd.DataFrame(columns=["timestamp", "close"])
        )
        if not btc.empty:
            btc = btc.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
            nav_idx = pd.DatetimeIndex(nav_wide.index)
            nav_idx_cmp = nav_idx.tz_convert("UTC").tz_localize(None) if nav_idx.tz is not None else nav_idx

            btc_price = btc.set_index("timestamp")["close"]
            btc_idx = pd.DatetimeIndex(btc_price.index)
            btc_price.index = btc_idx.tz_convert("UTC").tz_localize(None) if btc_idx.tz is not None else btc_idx

            btc_aligned = btc_price.reindex(nav_idx_cmp).ffill().bfill()
            btc_aligned.index = nav_wide.index
            if btc_aligned.notna().any():
                btc_aligned = btc_aligned.astype(float)
                btc_equity = btc_aligned / (float(btc_aligned.iloc[0]) + 1e-9) * float(initial_capital)

                fig, ax = plt.subplots(figsize=(12, 6))
                nav_wide.plot(ax=ax, linewidth=1.6)
                btc_label = f"{btc_symbol} buy&hold"
                btc_equity.rename(btc_label).plot(ax=ax, linewidth=2.0, linestyle="--", color="black")
                ax.set_title("Backtest Equity vs BTC (Same Period)")
                ax.set_ylabel("Equity")
                ax.set_xlabel("Timestamp")
                ax.grid(True, alpha=0.3)
                fig.tight_layout()
                name = CHART_FILENAMES[1]
                fig.savefig(out_dir / name, dpi=160)
                plt.close(fig)
                created.append(name)

    drawdown_wide = strategy_nav_df.pivot_table(
        index="timestamp",
        columns="strategy",
        values="drawdown",
        aggfunc="last",
    ).sort_index()
    if not drawdown_wide.empty:
        fig, ax = plt.subplots(figsize=(12, 5))
        drawdown_wide.plot(ax=ax, linewidth=1.4)
        ax.set_title("Backtest Drawdown")
        ax.set_ylabel("Drawdown")
        ax.set_xlabel("Timestamp")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        name = CHART_FILENAMES[2]
        fig.savefig(out_dir / name, dpi=160)
        plt.close(fig)
        created.append(name)

    if not strategy_returns_df.empty:
        rets = strategy_returns_df.pivot_table(
            index="timestamp",
            columns="strategy",
            values="net_return",
            aggfunc="last",
        ).sort_index()
        if not rets.empty:
            ppy = _periods_per_year(timeframe)
            window = max(12, min(96, len(rets) // 6 if len(rets) > 0 else 12))
            rolling_mean = rets.rolling(window).mean()
            rolling_std = rets.rolling(window).std(ddof=0)
            rolling_sharpe = rolling_mean / (rolling_std + 1e-9) * np.sqrt(ppy)

            fig, ax = plt.subplots(figsize=(12, 5))
            rolling_sharpe.plot(ax=ax, linewidth=1.3)
            ax.set_title(f"Rolling Sharpe (window={window})")
            ax.set_ylabel("Sharpe")
            ax.set_xlabel("Timestamp")
            ax.axhline(0.0, color="black", linewidth=0.9, alpha=0.7)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            name = CHART_FILENAMES[3]
            fig.savefig(out_dir / name, dpi=160)
            plt.close(fig)
            created.append(name)

    if not strategy_summary_df.empty:
        display_cols = [
            c
            for c in ["annual_return", "sharpe", "max_drawdown", "win_rate"]
            if c in strategy_summary_df.columns
        ]
        if display_cols:
            summary = strategy_summary_df.set_index("strategy")[display_cols]
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            flat_axes = axes.flatten()

            for i, col in enumerate(display_cols):
                ax = flat_axes[i]
                summary[col].plot(kind="bar", ax=ax, rot=20)
                ax.set_title(col)
                ax.grid(True, axis="y", alpha=0.3)

            for j in range(len(display_cols), len(flat_axes)):
                flat_axes[j].axis("off")

            fig.tight_layout()
            name = CHART_FILENAMES[4]
            fig.savefig(out_dir / name, dpi=160)
            plt.close(fig)
            created.append(name)

    return created
