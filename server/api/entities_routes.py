"""Entity CRUD endpoints: counterparties, portfolios, derivatives, margin calls."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

import logging

from server.auth.rbac import Role, get_current_user, require_role
from server.core.config import settings
from server.core.database import get_db
from server.models.db_models import (
    Counterparty,
    Derivative,
    MarginCall,
    Portfolio,
    RiskMetric,
    User,
)
from server.notifications.alerts import send_margin_call_email
from server.notifications.audit import log_event

logger = logging.getLogger(__name__)

entities_router = APIRouter(prefix="/api/v1", tags=["entities"])


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class CounterpartyIn(BaseModel):
    external_id:      str
    name:             str
    credit_rating:    str   = "BBB"
    hazard_rate:      float = 0.02
    recovery_rate:    float = 0.40
    collateral:       float = 0.0
    margin_threshold: float = 0.0
    mpor_days:        int   = 10


class CounterpartyOut(CounterpartyIn):
    id:         str
    created_at: datetime
    updated_at: datetime
    model_config = {"from_attributes": True}


class CounterpartyDetailOut(CounterpartyOut):
    """Extended counterparty response that includes nested portfolios + derivatives."""
    portfolios: List["PortfolioWithDerivatives"] = []


class PortfolioIn(BaseModel):
    external_id:     str
    counterparty_id: str
    collateral:      float = 0.0
    net_value:       float = 0.0
    auto_run:        bool  = False


class PortfolioOut(PortfolioIn):
    id:         str
    created_at: datetime
    updated_at: datetime
    model_config = {"from_attributes": True}


# Forward-declared; filled in after DerivativeOut is defined.
class PortfolioWithDerivatives(PortfolioOut):
    derivatives: List["DerivativeOut"] = []


class DerivativeIn(BaseModel):
    external_id:      str
    deriv_type:       str   = "IRS"
    notional:         float = 1_000_000.0
    maturity_years:   float = 5.0
    underlying_price: float = 0.05
    strike:           float = 0.05
    cash_flow_freq:   float = 2.0


class DerivativeOut(DerivativeIn):
    id:           str
    portfolio_id: str
    created_at:   datetime
    model_config = {"from_attributes": True}


class MarginCallOut(BaseModel):
    id:                str
    counterparty_id:   str
    simulation_run_id: Optional[str]
    amount:            float
    excess_exposure:   float
    status:            str
    reason:            str
    issued_at:         datetime
    acknowledged_at:   Optional[datetime]
    settled_at:        Optional[datetime]
    model_config = {"from_attributes": True}


# Resolve forward references now that all types are defined.
PortfolioWithDerivatives.model_rebuild()
CounterpartyDetailOut.model_rebuild()


# ── Counterparties ────────────────────────────────────────────────────────────

@entities_router.get("/counterparties", response_model=List[CounterpartyOut])
async def list_counterparties(
    db:   AsyncSession = Depends(get_db),
    _u:   User         = Depends(get_current_user),
) -> List[CounterpartyOut]:
    result = await db.execute(select(Counterparty).order_by(Counterparty.name))
    return [CounterpartyOut.model_validate(c) for c in result.scalars().all()]


@entities_router.post("/counterparties", response_model=CounterpartyOut, status_code=201)
async def create_counterparty(
    body: CounterpartyIn,
    db:   AsyncSession = Depends(get_db),
    user: User         = Depends(require_role(Role.RISK_MANAGER, Role.ADMIN)),
) -> CounterpartyOut:
    cp = Counterparty(**body.model_dump(), created_by=user.id)
    db.add(cp)
    await log_event(db, action="create_counterparty", user_id=user.id,
                    resource_type="counterparty", resource_id=None,
                    detail={"name": body.name, "external_id": body.external_id})
    await db.commit()
    await db.refresh(cp)
    return CounterpartyOut.model_validate(cp)


@entities_router.get("/counterparties/{cp_id}", response_model=CounterpartyDetailOut)
async def get_counterparty(
    cp_id: str,
    db:    AsyncSession = Depends(get_db),
    _u:    User         = Depends(get_current_user),
) -> CounterpartyDetailOut:
    # Use explicit selectinload to safely eager-load nested relationships in
    # async sessions (avoids greenlet errors from implicit lazy loading).
    result = await db.execute(
        select(Counterparty)
        .options(
            selectinload(Counterparty.portfolios).selectinload(Portfolio.derivatives)
        )
        .where(Counterparty.id == cp_id)
    )
    cp = result.scalar_one_or_none()
    if cp is None:
        raise HTTPException(status_code=404, detail="Counterparty not found")
    return CounterpartyDetailOut.model_validate(cp)


@entities_router.put("/counterparties/{cp_id}", response_model=CounterpartyOut)
async def update_counterparty(
    cp_id: str,
    body:  CounterpartyIn,
    db:    AsyncSession = Depends(get_db),
    user:  User         = Depends(require_role(Role.RISK_MANAGER, Role.ADMIN)),
) -> CounterpartyOut:
    cp = await _get_or_404(db, Counterparty, cp_id)
    for field, value in body.model_dump().items():
        setattr(cp, field, value)
    cp.updated_at = datetime.now(timezone.utc)
    await log_event(db, action="update_counterparty", user_id=user.id,
                    resource_type="counterparty", resource_id=cp_id,
                    detail={"name": body.name})
    await db.commit()
    await db.refresh(cp)
    return CounterpartyOut.model_validate(cp)


@entities_router.delete("/counterparties/{cp_id}", status_code=204)
async def delete_counterparty(
    cp_id: str,
    db:    AsyncSession = Depends(get_db),
    user:  User         = Depends(require_role(Role.RISK_MANAGER, Role.ADMIN)),
) -> None:
    cp = await _get_or_404(db, Counterparty, cp_id)
    cp_name = cp.name
    db.delete(cp)
    try:
        await log_event(db, action="delete_counterparty", user_id=user.id,
                        resource_type="counterparty", resource_id=cp_id,
                        detail={"name": cp_name})
        await db.commit()
    except IntegrityError:
        await db.rollback()
        raise HTTPException(
            status_code=409,
            detail="Cannot delete counterparty with existing portfolios or margin calls",
        )


# ── Portfolios ────────────────────────────────────────────────────────────────

@entities_router.get("/portfolios", response_model=List[PortfolioOut])
async def list_portfolios(
    counterparty_id: Optional[str] = Query(None),
    db:              AsyncSession   = Depends(get_db),
    _u:              User           = Depends(get_current_user),
) -> List[PortfolioOut]:
    stmt = select(Portfolio)
    if counterparty_id:
        stmt = stmt.where(Portfolio.counterparty_id == counterparty_id)
    result = await db.execute(stmt.order_by(Portfolio.created_at))
    return [PortfolioOut.model_validate(p) for p in result.scalars().all()]


@entities_router.post("/portfolios", response_model=PortfolioOut, status_code=201)
async def create_portfolio(
    body: PortfolioIn,
    db:   AsyncSession = Depends(get_db),
    user: User         = Depends(require_role(Role.RISK_MANAGER, Role.ADMIN)),
) -> PortfolioOut:
    portfolio = Portfolio(**body.model_dump())
    db.add(portfolio)
    await log_event(db, action="create_portfolio", user_id=user.id,
                    resource_type="portfolio", resource_id=None,
                    detail={"external_id": body.external_id, "counterparty_id": body.counterparty_id})
    await db.commit()
    await db.refresh(portfolio)
    return PortfolioOut.model_validate(portfolio)


@entities_router.get("/portfolios/{port_id}", response_model=PortfolioOut)
async def get_portfolio(
    port_id: str,
    db:      AsyncSession = Depends(get_db),
    _u:      User         = Depends(get_current_user),
) -> PortfolioOut:
    p = await _get_or_404(db, Portfolio, port_id)
    return PortfolioOut.model_validate(p)


@entities_router.put("/portfolios/{port_id}", response_model=PortfolioOut)
async def update_portfolio(
    port_id: str,
    body:    PortfolioIn,
    db:      AsyncSession = Depends(get_db),
    user:    User         = Depends(require_role(Role.RISK_MANAGER, Role.ADMIN)),
) -> PortfolioOut:
    p = await _get_or_404(db, Portfolio, port_id)
    for field, value in body.model_dump().items():
        setattr(p, field, value)
    p.updated_at = datetime.now(timezone.utc)
    await log_event(db, action="update_portfolio", user_id=user.id,
                    resource_type="portfolio", resource_id=port_id, detail={})
    await db.commit()
    await db.refresh(p)
    return PortfolioOut.model_validate(p)


@entities_router.delete("/portfolios/{port_id}", status_code=204)
async def delete_portfolio(
    port_id: str,
    db:      AsyncSession = Depends(get_db),
    user:    User         = Depends(require_role(Role.RISK_MANAGER, Role.ADMIN)),
) -> None:
    p = await _get_or_404(db, Portfolio, port_id)
    db.delete(p)
    try:
        await log_event(db, action="delete_portfolio", user_id=user.id,
                        resource_type="portfolio", resource_id=port_id, detail={})
        await db.commit()
    except IntegrityError:
        await db.rollback()
        raise HTTPException(
            status_code=409,
            detail="Cannot delete portfolio with existing derivatives or simulation runs",
        )


# ── Derivatives (sub-resource of portfolios) ──────────────────────────────────

@entities_router.post("/portfolios/{port_id}/derivatives", response_model=DerivativeOut, status_code=201)
async def add_derivative(
    port_id: str,
    body:    DerivativeIn,
    db:      AsyncSession = Depends(get_db),
    user:    User         = Depends(require_role(Role.RISK_MANAGER, Role.ADMIN)),
) -> DerivativeOut:
    await _get_or_404(db, Portfolio, port_id)
    deriv = Derivative(**body.model_dump(), portfolio_id=port_id)
    db.add(deriv)
    await log_event(db, action="create_derivative", user_id=user.id,
                    resource_type="derivative", resource_id=None,
                    detail={"portfolio_id": port_id, "type": body.deriv_type, "notional": body.notional})
    await db.commit()
    await db.refresh(deriv)
    return DerivativeOut.model_validate(deriv)


@entities_router.delete("/portfolios/{port_id}/derivatives/{deriv_id}", status_code=204)
async def delete_derivative(
    port_id:  str,
    deriv_id: str,
    db:       AsyncSession = Depends(get_db),
    user:     User         = Depends(require_role(Role.RISK_MANAGER, Role.ADMIN)),
) -> None:
    result = await db.execute(
        select(Derivative).where(Derivative.id == deriv_id, Derivative.portfolio_id == port_id)
    )
    deriv = result.scalar_one_or_none()
    if deriv is None:
        raise HTTPException(status_code=404, detail="Derivative not found")
    db.delete(deriv)
    try:
        await log_event(db, action="delete_derivative", user_id=user.id,
                        resource_type="derivative", resource_id=deriv_id,
                        detail={"portfolio_id": port_id})
        await db.commit()
    except IntegrityError:
        await db.rollback()
        raise HTTPException(status_code=409, detail="Cannot delete derivative with linked simulation runs")


# ── Margin calls ──────────────────────────────────────────────────────────────

@entities_router.get("/margin-calls", response_model=List[MarginCallOut])
async def list_margin_calls(
    mc_status:       Optional[str] = Query(None, alias="status"),
    counterparty_id: Optional[str] = Query(None),
    db:              AsyncSession   = Depends(get_db),
    _u:              User           = Depends(get_current_user),
) -> List[MarginCallOut]:
    stmt = select(MarginCall)
    if mc_status:
        stmt = stmt.where(MarginCall.status == mc_status.upper())
    if counterparty_id:
        stmt = stmt.where(MarginCall.counterparty_id == counterparty_id)
    result = await db.execute(stmt.order_by(MarginCall.issued_at.desc()))
    return [MarginCallOut.model_validate(mc) for mc in result.scalars().all()]


@entities_router.put("/margin-calls/{mc_id}/acknowledge", response_model=MarginCallOut)
async def acknowledge_margin_call(
    mc_id: str,
    db:    AsyncSession = Depends(get_db),
    user:  User         = Depends(require_role(Role.RISK_MANAGER, Role.ADMIN)),
) -> MarginCallOut:
    mc = await _get_or_404(db, MarginCall, mc_id)
    mc.status = "ACKNOWLEDGED"
    mc.acknowledged_at = datetime.now(timezone.utc)
    await log_event(db, action="acknowledge_margin_call", user_id=user.id,
                    resource_type="margin_call", resource_id=mc_id,
                    detail={"amount": mc.amount})
    await db.commit()
    await db.refresh(mc)
    return MarginCallOut.model_validate(mc)


@entities_router.put("/margin-calls/{mc_id}/settle", response_model=MarginCallOut)
async def settle_margin_call(
    mc_id: str,
    db:    AsyncSession = Depends(get_db),
    user:  User         = Depends(require_role(Role.RISK_MANAGER, Role.ADMIN)),
) -> MarginCallOut:
    mc = await _get_or_404(db, MarginCall, mc_id)
    mc.status = "SETTLED"
    mc.settled_at = datetime.now(timezone.utc)
    await log_event(db, action="settle_margin_call", user_id=user.id,
                    resource_type="margin_call", resource_id=mc_id,
                    detail={"amount": mc.amount})
    await db.commit()
    await db.refresh(mc)
    return MarginCallOut.model_validate(mc)


@entities_router.post("/margin-calls/{mc_id}/notify", status_code=200)
async def notify_counterparty(
    mc_id: str,
    db:    AsyncSession = Depends(get_db),
    user:  User         = Depends(require_role(Role.RISK_MANAGER, Role.ADMIN)),
) -> dict:
    """Send a counterparty notification for a margin call. Falls back to logging if SMTP not configured."""
    mc = await _get_or_404(db, MarginCall, mc_id)
    cp_result = await db.execute(select(Counterparty).where(Counterparty.id == mc.counterparty_id))
    cp = cp_result.scalar_one_or_none()
    cp_name = cp.name if cp else mc.counterparty_id

    if settings.smtp_host:
        import asyncio
        rm_result = await db.execute(
            select(User).where(User.role == "RISK_MANAGER", User.is_active.is_(True))
        )
        emails = [u.email for u in rm_result.scalars().all() if u.email]
        if emails:
            asyncio.create_task(
                send_margin_call_email(emails, f"[COUNTERPARTY NOTICE] {cp_name}", mc.amount, mc.excess_exposure)
            )
    else:
        logger.warning(
            "Counterparty notification (SMTP unconfigured): %s — mc_id=%s amount=%.2f",
            cp_name, mc_id, mc.amount,
        )

    await log_event(db, action="notify_counterparty", user_id=user.id,
                    resource_type="margin_call", resource_id=mc_id,
                    detail={"counterparty_id": mc.counterparty_id, "amount": mc.amount})
    await db.commit()
    return {"status": "notified", "margin_call_id": mc_id}


# ── Counterparty summary ──────────────────────────────────────────────────────

class CounterpartySummary(BaseModel):
    total_runs:          int
    avg_cva:             float
    latest_cva:          Optional[float]
    total_margin_called: float
    pending_calls:       int
    settled_calls:       int
    total_derivatives:   int


@entities_router.get("/counterparties/{cp_id}/summary", response_model=CounterpartySummary)
async def get_counterparty_summary(
    cp_id: str,
    db:    AsyncSession = Depends(get_db),
    _u:    User         = Depends(get_current_user),
) -> CounterpartySummary:
    """Aggregate stats for a single counterparty."""
    await _get_or_404(db, Counterparty, cp_id)

    # Simulation run stats from risk_metrics (base runs only)
    rm_result = await db.execute(
        select(
            func.count().label("run_count"),
            func.avg(RiskMetric.cva).label("avg_cva"),
        )
        .where(RiskMetric.counterparty_id == cp_id, RiskMetric.is_stressed.is_(False))
    )
    rm_row = rm_result.one()

    # Latest CVA
    latest_result = await db.execute(
        select(RiskMetric.cva)
        .where(RiskMetric.counterparty_id == cp_id, RiskMetric.is_stressed.is_(False))
        .order_by(RiskMetric.time.desc())
        .limit(1)
    )
    latest_cva = latest_result.scalar_one_or_none()

    # Margin call stats
    mc_result = await db.execute(
        select(
            func.coalesce(func.sum(MarginCall.amount), 0).label("total_amount"),
            func.count().filter(MarginCall.status == "PENDING").label("pending"),
            func.count().filter(MarginCall.status == "SETTLED").label("settled"),
        )
        .where(MarginCall.counterparty_id == cp_id)
    )
    mc_row = mc_result.one()

    # Derivative count across all portfolios
    port_result = await db.execute(
        select(Portfolio.id).where(Portfolio.counterparty_id == cp_id)
    )
    port_ids = [r[0] for r in port_result.all()]
    deriv_count = 0
    if port_ids:
        d_result = await db.execute(
            select(func.count()).where(Derivative.portfolio_id.in_(port_ids))
        )
        deriv_count = d_result.scalar() or 0

    return CounterpartySummary(
        total_runs          = rm_row.run_count or 0,
        avg_cva             = float(rm_row.avg_cva or 0.0),
        latest_cva          = float(latest_cva) if latest_cva is not None else None,
        total_margin_called = float(mc_row.total_amount or 0.0),
        pending_calls       = mc_row.pending or 0,
        settled_calls       = mc_row.settled or 0,
        total_derivatives   = deriv_count,
    )


# ── Utility ───────────────────────────────────────────────────────────────────

async def _get_or_404(db: AsyncSession, model: Any, obj_id: str) -> Any:
    result = await db.execute(select(model).where(model.id == obj_id))
    obj = result.scalar_one_or_none()
    if obj is None:
        raise HTTPException(status_code=404, detail=f"{model.__tablename__} not found")
    return obj
