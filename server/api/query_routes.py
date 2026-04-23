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
    Portfolio,
    PriceHistory,
    RiskMetric,
    SimulationRun,
    User,
)

# ── Date-bounds response ──────────────────────────────────────────────────────

class _Bounds(BaseModel):
    min: Optional[datetime]
    max: Optional[datetime]

class DateBoundsResponse(BaseModel):
    risk_metrics: _Bounds
    margin_calls: _Bounds

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


# ── Date bounds ──────────────────────────────────────────────────────────────

@query_router.get("/date-bounds", response_model=DateBoundsResponse)
async def get_date_bounds(
    db: AsyncSession = Depends(get_db),
    _u: User         = Depends(get_current_user),
) -> DateBoundsResponse:
    """Return min/max timestamps for risk metrics and margin calls.
    Used by the query builder to constrain date-picker inputs.
    """
    rm_result = await db.execute(
        select(func.min(RiskMetric.time), func.max(RiskMetric.time))
    )
    rm_row = rm_result.one()

    mc_result = await db.execute(
        select(func.min(MarginCall.issued_at), func.max(MarginCall.issued_at))
    )
    mc_row = mc_result.one()

    return DateBoundsResponse(
        risk_metrics=_Bounds(min=rm_row[0], max=rm_row[1]),
        margin_calls=_Bounds(min=mc_row[0], max=mc_row[1]),
    )


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


# ── Historical backtesting ────────────────────────────────────────────────────

class BacktestObservation(BaseModel):
    date:     str
    exposure: float
    breach:   bool


class BacktestResult(BaseModel):
    pfe_profile:   List[float]
    epe_profile:   List[float]
    time_grid:     List[float]
    realised:      List[BacktestObservation]
    breach_count:  int
    coverage_pct:  float


@query_router.get("/counterparties/{cp_id}/backtest", response_model=BacktestResult)
async def get_backtest(
    cp_id:  str,
    days:   int          = Query(90, ge=7, le=365),
    db:     AsyncSession = Depends(get_db),
    _u:     User         = Depends(get_current_user),
) -> BacktestResult:
    """
    Return the latest PFE/EPE profile for this counterparty together with
    realised mark-to-model exposures constructed from price history.

    Each historical price observation is mapped to an approximate portfolio
    exposure using a simple GBM log-return model:
        exposure(t) ≈ Σ notional_i × max(S(t)/S₀ − K_i/S₀, 0)
    where S₀ is the earliest price in the window and K_i = derivative strike.
    This is an indicative approximation only — it ignores netting and
    funding adjustments.
    """
    from datetime import timedelta
    from sqlalchemy.orm import selectinload as _selectinload
    import math

    # Latest non-stressed simulation for this counterparty
    rm_res = await db.execute(
        select(RiskMetric)
        .where(RiskMetric.counterparty_id == cp_id, RiskMetric.is_stressed == False)  # noqa: E712
        .order_by(RiskMetric.time.desc())
        .limit(1)
    )
    rm = rm_res.scalar_one_or_none()
    if rm is None:
        return BacktestResult(
            pfe_profile=[], epe_profile=[], time_grid=[],
            realised=[], breach_count=0, coverage_pct=100.0,
        )

    pfe = json.loads(rm.pfe_profile)     if rm.pfe_profile     else []
    epe = json.loads(rm.epe_profile)     if rm.epe_profile     else []
    tg  = json.loads(rm.time_grid_years) if rm.time_grid_years else []
    peak_pfe = max(pfe) if pfe else 0.0

    # Load counterparty's derivatives for exposure approximation
    cp_res = await db.execute(
        select(Counterparty)
        .options(_selectinload(Counterparty.portfolios).selectinload(Portfolio.derivatives))
        .where(Counterparty.id == cp_id)
    )
    cp = cp_res.scalar_one_or_none()
    derivs = [d for p in (cp.portfolios if cp else []) for d in (p.derivatives or [])]

    # Load price history for all known symbols over the last `days` days
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    ph_res = await db.execute(
        select(PriceHistory)
        .where(PriceHistory.ts >= cutoff)
        .order_by(PriceHistory.symbol, PriceHistory.ts.asc())
    )
    ph_rows = ph_res.scalars().all()

    # Group price history by symbol
    by_symbol: dict[str, list[tuple[datetime, float]]] = {}
    for ph in ph_rows:
        by_symbol.setdefault(ph.symbol, []).append((ph.ts, ph.price))

    # Build daily exposure estimates — one per calendar day
    from collections import defaultdict
    daily_exposure: dict[str, float] = defaultdict(float)
    daily_count:    dict[str, int]   = defaultdict(int)

    for symbol, series in by_symbol.items():
        if len(series) < 2:
            continue
        s0 = series[0][1]  # earliest price as reference
        if s0 <= 0:
            continue
        for ts, price in series[1:]:
            date_str = ts.strftime("%Y-%m-%d")
            total_exposure = 0.0
            for d in derivs:
                k  = d.strike if d.strike > 0 else s0
                # Simplified call payoff as exposure proxy
                payoff = max(price / s0 - k / s0, 0.0) * abs(d.notional)
                total_exposure += payoff
            if not derivs:
                # No derivatives — use log-return as a proxy
                total_exposure = abs(math.log(price / s0)) * 1_000_000
            daily_exposure[date_str] += total_exposure
            daily_count[date_str]    += 1

    realised: list[BacktestObservation] = []
    breach_count = 0
    for date_str in sorted(daily_exposure.keys()):
        exposure = daily_exposure[date_str] / max(daily_count[date_str], 1)
        breach   = peak_pfe > 0 and exposure > peak_pfe
        if breach:
            breach_count += 1
        realised.append(BacktestObservation(date=date_str, exposure=exposure, breach=breach))

    n = len(realised)
    coverage_pct = ((n - breach_count) / n * 100.0) if n > 0 else 100.0

    return BacktestResult(
        pfe_profile  = pfe,
        epe_profile  = epe,
        time_grid    = tg,
        realised     = realised,
        breach_count = breach_count,
        coverage_pct = round(coverage_pct, 1),
    )
