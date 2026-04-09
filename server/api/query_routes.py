"""Curated TimescaleDB query templates for the interactive query builder."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy import Float, and_, desc, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from server.auth.rbac import get_current_user
from server.core.database import get_db
from server.models.db_models import (
    Counterparty,
    MarginCall,
    RiskMetric,
    SimulationRun,
    User,
)

logger = logging.getLogger(__name__)
query_router = APIRouter(prefix="/api/v1/query", tags=["query"])


# ── Response models ───────────────────────────────────────────────────────────

class RiskTimelineRow(BaseModel):
    time:              datetime
    counterparty_id:   Optional[str]
    counterparty_name: Optional[str]
    cva:               float
    wwr_cva:           float
    margin_required:   float
    is_stressed:       bool


class ExposureRankRow(BaseModel):
    counterparty_id:   str
    counterparty_name: Optional[str]
    cva:               float
    wwr_cva:           float
    margin_required:   float
    run_count:         int
    last_run_time:     datetime


class PfePeakRow(BaseModel):
    time:              datetime
    simulation_run_id: Optional[str]
    counterparty_id:   Optional[str]
    counterparty_name: Optional[str]
    peak_pfe:          float
    cva:               float


class MarginActivityRow(BaseModel):
    issued_at:         datetime
    counterparty_id:   str
    counterparty_name: Optional[str]
    amount:            float
    excess_exposure:   float
    status:            str
    reason:            str


class VolCvaRow(BaseModel):
    time:              datetime
    sigma:             Optional[float]
    num_paths:         Optional[int]
    cva:               float
    wwr_cva:           float
    counterparty_id:   Optional[str]


class QueryMetadata(BaseModel):
    template:    str
    row_count:   int
    executed_at: datetime


# ── Helper ────────────────────────────────────────────────────────────────────

def _parse_dt(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


# ── Template 1: Risk Timeline ─────────────────────────────────────────────────

@query_router.get("/risk-timeline", response_model=Dict[str, Any])
async def risk_timeline(
    counterparty_id: Optional[str] = Query(None, description="Filter to one counterparty"),
    from_dt:         Optional[str] = Query(None, alias="from",  description="ISO datetime start"),
    to_dt:           Optional[str] = Query(None, alias="to",    description="ISO datetime end"),
    stressed_only:   bool          = Query(False),
    limit:           int           = Query(200, ge=1, le=1000),
    db:              AsyncSession  = Depends(get_db),
    _u:              User          = Depends(get_current_user),
) -> Dict[str, Any]:
    """CVA, WWR-CVA, and margin over time — line chart data."""
    stmt = (
        select(RiskMetric, Counterparty.name.label("cp_name"))
        .outerjoin(Counterparty, RiskMetric.counterparty_id == Counterparty.id)
        .order_by(desc(RiskMetric.time))
        .limit(limit)
    )
    if counterparty_id:
        stmt = stmt.where(RiskMetric.counterparty_id == counterparty_id)
    if stressed_only:
        stmt = stmt.where(RiskMetric.is_stressed.is_(True))
    else:
        stmt = stmt.where(RiskMetric.is_stressed.is_(False))
    from_parsed = _parse_dt(from_dt)
    to_parsed   = _parse_dt(to_dt)
    if from_parsed:
        stmt = stmt.where(RiskMetric.time >= from_parsed)
    if to_parsed:
        stmt = stmt.where(RiskMetric.time <= to_parsed)

    result = await db.execute(stmt)
    rows: List[RiskTimelineRow] = []
    for rm, cp_name in result:
        rows.append(RiskTimelineRow(
            time=rm.time,
            counterparty_id=rm.counterparty_id,
            counterparty_name=cp_name,
            cva=rm.cva,
            wwr_cva=rm.wwr_cva,
            margin_required=rm.margin_required,
            is_stressed=rm.is_stressed,
        ))
    # Reverse so callers get chronological order for charts
    rows.reverse()
    return {
        "meta": QueryMetadata(template="risk-timeline", row_count=len(rows), executed_at=datetime.now(timezone.utc)),
        "rows": [r.model_dump() for r in rows],
    }


# ── Template 2: Exposure Ranking ──────────────────────────────────────────────

@query_router.get("/exposure-ranking", response_model=Dict[str, Any])
async def exposure_ranking(
    from_dt:  Optional[str] = Query(None, alias="from"),
    to_dt:    Optional[str] = Query(None, alias="to"),
    min_cva:  float         = Query(0.0, description="Minimum CVA threshold"),
    limit:    int           = Query(20, ge=1, le=100),
    db:       AsyncSession  = Depends(get_db),
    _u:       User          = Depends(get_current_user),
) -> Dict[str, Any]:
    """Top counterparties ranked by latest CVA — bar chart data."""
    from_parsed = _parse_dt(from_dt)
    to_parsed   = _parse_dt(to_dt)

    # Subquery: latest non-stressed run time + count per counterparty
    sub_filters = [RiskMetric.is_stressed.is_(False), RiskMetric.counterparty_id.isnot(None)]
    if from_parsed:
        sub_filters.append(RiskMetric.time >= from_parsed)
    if to_parsed:
        sub_filters.append(RiskMetric.time <= to_parsed)

    subq = (
        select(
            RiskMetric.counterparty_id,
            func.max(RiskMetric.time).label("max_time"),
            func.count().label("run_count"),
        )
        .where(and_(*sub_filters))
        .group_by(RiskMetric.counterparty_id)
        .subquery()
    )

    stmt = (
        select(RiskMetric, Counterparty.name.label("cp_name"), subq.c.run_count)
        .join(subq, and_(
            RiskMetric.counterparty_id == subq.c.counterparty_id,
            RiskMetric.time            == subq.c.max_time,
        ))
        .outerjoin(Counterparty, RiskMetric.counterparty_id == Counterparty.id)
        .where(RiskMetric.cva >= min_cva)
        .order_by(desc(RiskMetric.cva))
        .limit(limit)
    )

    result = await db.execute(stmt)
    rows: List[ExposureRankRow] = []
    for rm, cp_name, run_count in result:
        rows.append(ExposureRankRow(
            counterparty_id=rm.counterparty_id,
            counterparty_name=cp_name,
            cva=rm.cva,
            wwr_cva=rm.wwr_cva,
            margin_required=rm.margin_required,
            run_count=run_count or 0,
            last_run_time=rm.time,
        ))
    return {
        "meta": QueryMetadata(template="exposure-ranking", row_count=len(rows), executed_at=datetime.now(timezone.utc)),
        "rows": [r.model_dump() for r in rows],
    }


# ── Template 3: PFE Peaks ─────────────────────────────────────────────────────

@query_router.get("/pfe-peaks", response_model=Dict[str, Any])
async def pfe_peaks(
    counterparty_id: Optional[str] = Query(None),
    from_dt:         Optional[str] = Query(None, alias="from"),
    to_dt:           Optional[str] = Query(None, alias="to"),
    limit:           int           = Query(50, ge=1, le=200),
    db:              AsyncSession  = Depends(get_db),
    _u:              User          = Depends(get_current_user),
) -> Dict[str, Any]:
    """Peak PFE per simulation run — scatter/bar chart data."""
    stmt = (
        select(RiskMetric, Counterparty.name.label("cp_name"))
        .outerjoin(Counterparty, RiskMetric.counterparty_id == Counterparty.id)
        .where(RiskMetric.is_stressed.is_(False))
        .order_by(desc(RiskMetric.time))
        .limit(limit)
    )
    if counterparty_id:
        stmt = stmt.where(RiskMetric.counterparty_id == counterparty_id)
    from_parsed = _parse_dt(from_dt)
    to_parsed   = _parse_dt(to_dt)
    if from_parsed:
        stmt = stmt.where(RiskMetric.time >= from_parsed)
    if to_parsed:
        stmt = stmt.where(RiskMetric.time <= to_parsed)

    result = await db.execute(stmt)
    rows: List[PfePeakRow] = []
    for rm, cp_name in result:
        # pfe_profile is stored as JSON text; compute peak in Python
        peak_pfe = 0.0
        if rm.pfe_profile:
            try:
                profile = json.loads(rm.pfe_profile)
                peak_pfe = max(profile) if profile else 0.0
            except (json.JSONDecodeError, ValueError):
                pass
        rows.append(PfePeakRow(
            time=rm.time,
            simulation_run_id=rm.simulation_run_id,
            counterparty_id=rm.counterparty_id,
            counterparty_name=cp_name,
            peak_pfe=peak_pfe,
            cva=rm.cva,
        ))
    rows.sort(key=lambda r: r.peak_pfe, reverse=True)
    return {
        "meta": QueryMetadata(template="pfe-peaks", row_count=len(rows), executed_at=datetime.now(timezone.utc)),
        "rows": [r.model_dump() for r in rows],
    }


# ── Template 4: Margin Activity ───────────────────────────────────────────────

@query_router.get("/margin-activity", response_model=Dict[str, Any])
async def margin_activity(
    counterparty_id: Optional[str] = Query(None),
    from_dt:         Optional[str] = Query(None, alias="from"),
    to_dt:           Optional[str] = Query(None, alias="to"),
    status:          Optional[str] = Query(None, description="PENDING | ACKNOWLEDGED | SETTLED | DISPUTED"),
    limit:           int           = Query(100, ge=1, le=500),
    db:              AsyncSession  = Depends(get_db),
    _u:              User          = Depends(get_current_user),
) -> Dict[str, Any]:
    """Margin call funnel over time — timeline + status breakdown."""
    stmt = (
        select(MarginCall, Counterparty.name.label("cp_name"))
        .outerjoin(Counterparty, MarginCall.counterparty_id == Counterparty.id)
        .order_by(desc(MarginCall.issued_at))
        .limit(limit)
    )
    if counterparty_id:
        stmt = stmt.where(MarginCall.counterparty_id == counterparty_id)
    if status:
        stmt = stmt.where(MarginCall.status == status.upper())
    from_parsed = _parse_dt(from_dt)
    to_parsed   = _parse_dt(to_dt)
    if from_parsed:
        stmt = stmt.where(MarginCall.issued_at >= from_parsed)
    if to_parsed:
        stmt = stmt.where(MarginCall.issued_at <= to_parsed)

    result = await db.execute(stmt)
    rows: List[MarginActivityRow] = []
    for mc, cp_name in result:
        rows.append(MarginActivityRow(
            issued_at=mc.issued_at,
            counterparty_id=mc.counterparty_id,
            counterparty_name=cp_name,
            amount=mc.amount,
            excess_exposure=mc.excess_exposure,
            status=mc.status,
            reason=mc.reason,
        ))

    # Status breakdown summary
    status_counts: Dict[str, int] = {}
    for r in rows:
        status_counts[r.status] = status_counts.get(r.status, 0) + 1

    return {
        "meta": QueryMetadata(template="margin-activity", row_count=len(rows), executed_at=datetime.now(timezone.utc)),
        "rows": [r.model_dump() for r in rows],
        "summary": {"status_breakdown": status_counts, "total_amount": sum(r.amount for r in rows)},
    }


# ── Template 5: Volatility vs CVA ─────────────────────────────────────────────

@query_router.get("/vol-cva", response_model=Dict[str, Any])
async def vol_cva(
    from_dt: Optional[str] = Query(None, alias="from"),
    to_dt:   Optional[str] = Query(None, alias="to"),
    limit:   int           = Query(200, ge=1, le=500),
    db:      AsyncSession  = Depends(get_db),
    _u:      User          = Depends(get_current_user),
) -> Dict[str, Any]:
    """Sigma (volatility) vs CVA scatter — reveals vol/credit sensitivity."""
    stmt = (
        select(RiskMetric, SimulationRun.sim_params_json)
        .outerjoin(SimulationRun, RiskMetric.simulation_run_id == SimulationRun.id)
        .where(RiskMetric.is_stressed.is_(False))
        .order_by(desc(RiskMetric.time))
        .limit(limit)
    )
    from_parsed = _parse_dt(from_dt)
    to_parsed   = _parse_dt(to_dt)
    if from_parsed:
        stmt = stmt.where(RiskMetric.time >= from_parsed)
    if to_parsed:
        stmt = stmt.where(RiskMetric.time <= to_parsed)

    result = await db.execute(stmt)
    rows: List[VolCvaRow] = []
    for rm, params_json in result:
        sigma     = None
        num_paths = None
        if params_json:
            sigma     = params_json.get("sigma")
            num_paths = params_json.get("num_paths")
        rows.append(VolCvaRow(
            time=rm.time,
            sigma=float(sigma) if sigma is not None else None,
            num_paths=int(num_paths) if num_paths is not None else None,
            cva=rm.cva,
            wwr_cva=rm.wwr_cva,
            counterparty_id=rm.counterparty_id,
        ))

    return {
        "meta": QueryMetadata(template="vol-cva", row_count=len(rows), executed_at=datetime.now(timezone.utc)),
        "rows": [r.model_dump() for r in rows],
    }
