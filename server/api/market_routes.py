"""Market data REST endpoints."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from server.auth.rbac import Role, get_current_user, require_role
from server.core.cache import market_cache
from server.core.database import get_db
from server.market_data.fetcher import get_all_spot_prices, refresh_market_params
from server.models.db_models import MarketParam, PriceHistory, User

logger = logging.getLogger(__name__)
market_router = APIRouter(prefix="/api/v1/market", tags=["market"])


# ── Schemas ───────────────────────────────────────────────────────────────────

class MarketPriceItem(BaseModel):
    symbol:     str
    param_type: str
    value:      float
    source:     str
    fetched_at: datetime

    model_config = {"from_attributes": True}


class PriceHistoryItem(BaseModel):
    ts:     datetime
    symbol: str
    price:  float
    source: str

    model_config = {"from_attributes": True}


# ── Endpoints ─────────────────────────────────────────────────────────────────

@market_router.get("/prices", response_model=List[MarketPriceItem])
async def get_market_prices(
    db: AsyncSession = Depends(get_db),
    _u: User         = Depends(get_current_user),
) -> List[MarketPriceItem]:
    """Current snapshot of all market params (cached 60 s)."""
    cached = market_cache.get("all_params")
    if cached is not None:
        return cached

    result = await db.execute(
        select(MarketParam).order_by(MarketParam.symbol, MarketParam.param_type)
    )
    rows = [MarketPriceItem.model_validate(r) for r in result.scalars().all()]
    market_cache.set("all_params", rows)
    return rows


@market_router.get("/prices/{symbol}/history", response_model=List[PriceHistoryItem])
async def get_price_history(
    symbol: str,
    hours:  int          = Query(24, ge=1, le=720),
    db:     AsyncSession = Depends(get_db),
    _u:     User         = Depends(get_current_user),
) -> List[PriceHistoryItem]:
    """Recent price history for a symbol (up to 720 h / 30 days)."""
    since = datetime.now(timezone.utc) - timedelta(hours=hours)
    result = await db.execute(
        select(PriceHistory)
        .where(PriceHistory.symbol == symbol, PriceHistory.ts >= since)
        .order_by(PriceHistory.ts)
    )
    return [PriceHistoryItem.model_validate(r) for r in result.scalars().all()]


@market_router.post("/refresh", status_code=202)
async def trigger_refresh(
    _u: User = Depends(require_role(Role.RISK_MANAGER, Role.ADMIN)),
) -> Dict[str, Any]:
    """Trigger an immediate market data refresh (async — returns 202)."""
    import asyncio

    from server.core.database import AsyncSessionLocal

    market_cache.invalidate("all_params")

    # Must NOT reuse the request-scoped session here: FastAPI closes it when
    # this handler returns, which is before the background task runs.
    async def _refresh() -> None:
        async with AsyncSessionLocal() as bg_db:
            await refresh_market_params(bg_db)

    asyncio.create_task(_refresh())
    return {"status": "refresh queued"}
