from __future__ import annotations

_TIMEFRAME_TO_FREQ = {
    "1m": "1min",
    "3m": "3min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1H",
    "2h": "2H",
    "4h": "4H",
    "6h": "6H",
    "8h": "8H",
    "12h": "12H",
    "1d": "1D",
}


def timeframe_to_pandas_freq(timeframe: str) -> str:
    if timeframe not in _TIMEFRAME_TO_FREQ:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    return _TIMEFRAME_TO_FREQ[timeframe]
