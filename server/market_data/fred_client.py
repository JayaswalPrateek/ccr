"""
Fetch free market rates from the FRED API (St. Louis Fed).

Requires FRED_API_KEY in .env (free registration at fred.stlouisfed.org).
All functions degrade gracefully to sensible defaults when the key is absent
or the API is unreachable, so the server starts cleanly without a key.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import requests

from server.core.config import settings

logger = logging.getLogger(__name__)

# FRED series IDs → our internal labels
FRED_SERIES: Dict[str, str] = {
    "SOFR":  "SOFR",   # Secured Overnight Financing Rate
    "DGS1":  "DGS1",   # 1-Year Treasury Constant Maturity
    "DGS5":  "DGS5",   # 5-Year Treasury Constant Maturity
    "DGS10": "DGS10",  # 10-Year Treasury Constant Maturity
}

_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"
_TIMEOUT  = 10   # seconds


def _fetch_series(series_id: str) -> Optional[float]:
    """Return the most recent observation for a FRED series, or None on failure."""
    if not settings.fred_api_key:
        return None
    try:
        resp = requests.get(
            _BASE_URL,
            params={
                "series_id":     series_id,
                "api_key":       settings.fred_api_key,
                "file_type":     "json",
                "sort_order":    "desc",
                "limit":         5,      # grab a few in case latest is missing
                "observation_start": "2020-01-01",
            },
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        observations = resp.json().get("observations", [])
        for obs in observations:
            val = obs.get("value", ".")
            if val != ".":
                return float(val) / 100.0   # FRED reports as percent; convert to decimal
    except Exception as exc:
        logger.warning("FRED fetch failed for %s: %s", series_id, exc)
    return None


def fetch_rates() -> Dict[str, float]:
    """Return current rates for all tracked FRED series.

    Absent or failed series fall back to conservative defaults so the engine
    always has a usable drift parameter.
    """
    defaults: Dict[str, float] = {
        "SOFR":  0.05,
        "DGS1":  0.05,
        "DGS5":  0.045,
        "DGS10": 0.04,
    }
    rates: Dict[str, float] = {}
    for series_id, label in FRED_SERIES.items():
        val = _fetch_series(series_id)
        rates[label] = val if val is not None else defaults[label]
    return rates
