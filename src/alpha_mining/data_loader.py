from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import ccxt
import pandas as pd

from .utils import timeframe_to_pandas_freq


@dataclass
class BinanceUSDMDataLoader:
    api_key: str | None = None
    api_secret: str | None = None

    def __post_init__(self) -> None:
        self.exchange = ccxt.binanceusdm(
            {
                "apiKey": self.api_key or "",
                "secret": self.api_secret or "",
                "enableRateLimit": True,
                "options": {"defaultType": "future"},
            }
        )

    def get_universe(self, symbols: list[str] | None, top_n: int = 30) -> list[str]:
        if symbols:
            return symbols

        markets = self.exchange.load_markets()
        filtered: list[tuple[str, float]] = []
        for sym, info in markets.items():
            if info.get("quote") != "USDT":
                continue
            if not info.get("contract") or not info.get("linear"):
                continue
            if not info.get("active", True):
                continue

            quote_volume = 0.0
            info_raw = info.get("info", {})
            if isinstance(info_raw, dict):
                if info_raw.get("contractType") not in (None, "PERPETUAL"):
                    continue
                quote_volume = float(info_raw.get("quoteVolume") or 0.0)
            filtered.append((sym, quote_volume))

        filtered.sort(key=lambda x: x[1], reverse=True)
        return [sym for sym, _ in filtered[:top_n]]

    def fetch_ohlcv_history(self, symbol: str, timeframe: str, since_ms: int) -> pd.DataFrame:
        tf_ms = int(self.exchange.parse_timeframe(timeframe) * 1000)
        now_ms = self.exchange.milliseconds()
        cursor = since_ms
        rows: list[list[float]] = []

        while cursor < now_ms:
            batch = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=cursor, limit=1500)
            if not batch:
                break

            rows.extend(batch)
            last_ts = int(batch[-1][0])
            next_cursor = last_ts + tf_ms
            if next_cursor <= cursor:
                break
            cursor = next_cursor

            if len(batch) < 1500:
                break
            time.sleep(max(self.exchange.rateLimit / 1000.0, 0.05))

        if not rows:
            return pd.DataFrame(columns=["timestamp", "symbol", "open", "high", "low", "close", "volume"])

        df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["symbol"] = symbol
        return df[["timestamp", "symbol", "open", "high", "low", "close", "volume"]]

    def fetch_funding_history(self, symbol: str, since_ms: int) -> pd.DataFrame:
        cursor = since_ms
        now_ms = self.exchange.milliseconds()
        rows: list[dict] = []

        while cursor < now_ms:
            batch = self.exchange.fetch_funding_rate_history(symbol=symbol, since=cursor, limit=1000)
            if not batch:
                break

            rows.extend(batch)
            last_ts = int(batch[-1]["timestamp"])
            next_cursor = last_ts + 1
            if next_cursor <= cursor:
                break
            cursor = next_cursor

            if len(batch) < 1000:
                break
            time.sleep(max(self.exchange.rateLimit / 1000.0, 0.05))

        if not rows:
            return pd.DataFrame(columns=["timestamp", "symbol", "funding_rate"])

        out = pd.DataFrame(
            {
                "timestamp": [int(x["timestamp"]) for x in rows],
                "symbol": symbol,
                "funding_rate": [float(x.get("fundingRate") or 0.0) for x in rows],
            }
        )
        return out

    def fetch_market_data(
        self,
        symbols: list[str],
        timeframe: str,
        lookback_days: int,
    ) -> pd.DataFrame:
        since_ms = self.exchange.milliseconds() - lookback_days * 24 * 60 * 60 * 1000
        freq = timeframe_to_pandas_freq(timeframe)

        market_frames: list[pd.DataFrame] = []
        for symbol in symbols:
            ohlcv = self.fetch_ohlcv_history(symbol=symbol, timeframe=timeframe, since_ms=since_ms)
            if ohlcv.empty:
                continue

            try:
                funding = self.fetch_funding_history(symbol=symbol, since_ms=since_ms)
            except Exception:
                funding = pd.DataFrame(columns=["timestamp", "symbol", "funding_rate"])

            ohlcv["timestamp"] = pd.to_datetime(ohlcv["timestamp"], unit="ms", utc=True).dt.floor(freq)
            if not funding.empty:
                funding["timestamp"] = pd.to_datetime(funding["timestamp"], unit="ms", utc=True).dt.floor(freq)

            merged = ohlcv.merge(
                funding[["timestamp", "symbol", "funding_rate"]],
                on=["timestamp", "symbol"],
                how="left",
            )
            merged["funding_rate"] = merged.groupby("symbol")["funding_rate"].ffill().fillna(0.0)
            merged["returns"] = merged.groupby("symbol")["close"].pct_change()
            market_frames.append(merged)

        if not market_frames:
            return pd.DataFrame(
                columns=[
                    "timestamp",
                    "symbol",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "funding_rate",
                    "returns",
                ]
            )

        market = pd.concat(market_frames, ignore_index=True)
        market = market.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
        return market


def save_market_cache(df: pd.DataFrame, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target, index=False)


def load_market_cache(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
    return df
