"""
Fetch real market data via yfinance (15-minute delayed, free).

Spot prices: equities, FX crosses, commodity futures.
Volatility : 30-day rolling annualised vol from daily log-returns.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

EQUITY_SYMBOLS:    List[str] = ["SPY", "AAPL", "MSFT", "GS", "JPM"]
FX_SYMBOLS:        List[str] = ["EURUSD=X", "GBPUSD=X", "USDJPY=X"]
COMMODITY_SYMBOLS: List[str] = ["GC=F", "CL=F", "NG=F"]   # Gold, Crude, NatGas
ALL_SYMBOLS:       List[str] = EQUITY_SYMBOLS + FX_SYMBOLS + COMMODITY_SYMBOLS


def fetch_spot_prices() -> Dict[str, float]:
    """Return the most recent closing price for every tracked symbol.

    Returns an empty dict entry (symbol omitted) for any symbol that fails.
    Never raises — callers get a best-effort result.
    """
    prices: Dict[str, float] = {}
    try:
        data = yf.download(
            ALL_SYMBOLS,
            period="2d",       # 2 days to guarantee at least one close
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=True,
        )
        # yfinance ≥0.2.31 may return MultiIndex columns; normalize
        if isinstance(data.columns, pd.MultiIndex):
            close = data["Close"]
        elif "Close" in data.columns:
            close = data["Close"]
        else:
            close = data.xs("Close", axis=1, level=0)

        for sym in ALL_SYMBOLS:
            try:
                if isinstance(close, pd.DataFrame) and sym in close.columns:
                    series = close[sym].dropna()
                elif isinstance(close, pd.Series):
                    series = close.dropna()
                else:
                    continue
                if not series.empty:
                    prices[sym] = float(series.iloc[-1])
            except Exception:
                pass
    except Exception as exc:
        logger.warning("yfinance batch download failed: %s", exc)
    return prices


def fetch_historical_volatility(symbol: str, window_days: int = 30) -> float:
    """Annualised 30-day rolling volatility from daily log-returns.

    Falls back to 0.20 if data is unavailable or insufficient.
    """
    try:
        hist = yf.download(
            symbol,
            period="90d",
            interval="1d",
            auto_adjust=True,
            progress=False,
        )
        if hist.empty or len(hist) < window_days + 2:
            return 0.20
        close = hist["Close"]
        # yfinance ≥0.2.31 returns MultiIndex columns even for single symbols
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close = close.dropna()
        log_returns = (close / close.shift(1)).apply(math.log).dropna()
        vol = float(log_returns.tail(window_days).std() * math.sqrt(252))
        return max(vol, 0.01)   # floor at 1 % to avoid zero-vol edge case
    except Exception as exc:
        logger.warning("vol fetch failed for %s: %s — using 0.20 default", symbol, exc)
        return 0.20
