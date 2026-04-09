"""
Market data orchestrator.

refresh_market_params(db)   — fetch real prices + rates, upsert market_params,
                               append to price_history.
get_sigma_for_symbol(...)   — 30-day vol from DB history or live yfinance.
get_drift_from_db(db)       — SOFR from market_params cache.
get_all_spot_prices(db)     — current snapshot from market_params.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Dict, Optional

from sqlalchemy import select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from server.core.config import settings
from server.market_data.fred_client import fetch_rates
from server.market_data.yfinance_client import (
    ALL_SYMBOLS,
    fetch_historical_volatility,
    fetch_spot_prices,
)
from server.models.db_models import MarketParam, PriceHistory

logger = logging.getLogger(__name__)


# ── Upsert helpers ────────────────────────────────────────────────────────────

async def _upsert_market_param(
    db:         AsyncSession,
    symbol:     str,
    param_type: str,
    value:      float,
    source:     str,
) -> None:
    """Insert or update a single row in market_params."""
    stmt = (
        pg_insert(MarketParam)
        .values(
            symbol     = symbol,
            param_type = param_type,
            value      = value,
            source     = source,
            fetched_at = datetime.now(timezone.utc),
        )
        .on_conflict_do_update(
            index_elements = ["symbol", "param_type"],
            set_ = {
                "value":      value,
                "source":     source,
                "fetched_at": datetime.now(timezone.utc),
            },
        )
    )
    await db.execute(stmt)


async def _append_price_history(
    db:     AsyncSession,
    symbol: str,
    price:  float,
    source: str,
) -> None:
    row = PriceHistory(symbol=symbol, price=price, source=source)
    db.add(row)


# ── Main refresh ──────────────────────────────────────────────────────────────

async def refresh_market_params(db: AsyncSession) -> None:
    """Fetch live prices + rates and persist to the DB.

    Called on server startup and every 15 minutes by the scheduler.
    Failures are logged but do not propagate — stale data is better than
    a crashed server.

    yfinance and FRED calls are synchronous (blocking network I/O).  Running
    them directly on the asyncio event loop stalls the entire server.  We push
    every sync call into the default thread-pool executor so the event loop
    stays responsive.  Volatility fetches are gathered concurrently.
    """
    import asyncio

    loop = asyncio.get_running_loop()

    try:
        # ── Spot prices (yfinance) ────────────────────────────────────────────
        spots: Dict[str, float] = await loop.run_in_executor(None, fetch_spot_prices)
        for sym, price in spots.items():
            await _upsert_market_param(db, sym, "SPOT", price, "yfinance")
            await _append_price_history(db, sym, price, "yfinance")

        # ── Historical vols (yfinance, 30-day) — fetched concurrently ────────
        syms_with_price = [s for s in ALL_SYMBOLS if s in spots]
        vol_results = await asyncio.gather(
            *[loop.run_in_executor(None, fetch_historical_volatility, s)
              for s in syms_with_price],
            return_exceptions=True,
        )
        for sym, vol in zip(syms_with_price, vol_results):
            if isinstance(vol, float):
                await _upsert_market_param(db, sym, "VOL", vol, "yfinance_derived")

        # ── Rates (FRED) ──────────────────────────────────────────────────────
        rates: Dict[str, float] = await loop.run_in_executor(None, fetch_rates)
        for series_id, value in rates.items():
            await _upsert_market_param(db, series_id, "RATE", value, "fred")

        await db.commit()
        logger.info("Market params refreshed: %d spots, %d rates", len(spots), len(rates))
    except Exception as exc:
        await db.rollback()
        logger.error("refresh_market_params failed: %s", exc)


# ── Query helpers ─────────────────────────────────────────────────────────────

async def get_all_spot_prices(db: AsyncSession) -> Dict[str, float]:
    """Return the current SPOT snapshot from market_params."""
    result = await db.execute(
        select(MarketParam).where(MarketParam.param_type == "SPOT")
    )
    return {row.symbol: row.value for row in result.scalars().all()}


async def get_sigma_for_symbol(symbol: str, db: AsyncSession) -> float:
    """30-day vol from DB if fresh, else live yfinance fetch (non-blocking)."""
    import asyncio

    result = await db.execute(
        select(MarketParam).where(
            MarketParam.symbol == symbol,
            MarketParam.param_type == "VOL",
        )
    )
    row = result.scalar_one_or_none()
    if row is not None:
        return float(row.value)
    # No cached vol — fetch live in thread executor and cache it.
    loop = asyncio.get_running_loop()
    vol = await loop.run_in_executor(None, fetch_historical_volatility, symbol)
    await _upsert_market_param(db, symbol, "VOL", vol, "yfinance_derived")
    await db.commit()
    return vol


async def get_drift_from_db(db: AsyncSession) -> float:
    """Return SOFR from market_params, falling back to 0.05."""
    result = await db.execute(
        select(MarketParam).where(
            MarketParam.symbol == "SOFR",
            MarketParam.param_type == "RATE",
        )
    )
    row = result.scalar_one_or_none()
    return float(row.value) if row is not None else 0.05
