"""REST API routes for the CCR engine."""

from __future__ import annotations

import io
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import _ccr_engine as _ccr
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from server.auth.rbac import Role, get_current_user, require_role
from server.bindings.engine_client import build_engine_config
from server.core.database import get_db
from server.core.engine_runner import engine_info, run_simulation
from server.models.db_models import (
    AuditLog,
    Counterparty,
    Derivative,
    MarginCall,
    Portfolio,
    RiskMetric,
    SimulationRun,
    SimStatus,
    TriggerType,
    User,
)
from server.models.schemas import SimulationRequest, SimulationResponse
from server.notifications.alerts import check_and_alert_margin_calls
from server.notifications.audit import log_event

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["simulation"])


# ── Health ────────────────────────────────────────────────────────────────────

@router.get("/health")
async def health():
    """Liveness probe — also returns engine architecture info."""
    return {"status": "ok", "engine": engine_info()}


# ── Simulation history schema ─────────────────────────────────────────────────

class SimulationHistoryItem(BaseModel):
    id:                str
    run_id:            Optional[str]
    counterparty_id:   Optional[str]
    cva:               float
    wwr_cva:           float
    margin_required:   float
    is_stressed:       bool
    compute_time_us:   int
    time:              datetime
    pfe_profile:       List[float]
    epe_profile:       List[float]
    time_grid_years:   List[float]
    note:              Optional[str] = None

    model_config = {"from_attributes": False}


# ── Simulate ──────────────────────────────────────────────────────────────────

@router.post("/simulate", response_model=SimulationResponse)
async def simulate(
    request:      SimulationRequest,
    db:           AsyncSession = Depends(get_db),
    current_user: User         = Depends(require_role(Role.RISK_MANAGER, Role.ADMIN)),
) -> SimulationResponse:
    """Run a CCR Monte Carlo simulation, persist results, and check margin."""
    err = _ccr.CcrEngine.validate_config(build_engine_config(request))
    if err:
        logger.warning(
            "simulate: config validation failed — counterparty=%s paths=%d sigma=%.4f reason=%r",
            request.counterparty.id,
            request.sim_params.num_paths,
            request.sim_params.sigma,
            err,
        )
        raise HTTPException(status_code=422, detail=err)

    # ── Resolve counterparty_id to a real DB UUID ────────────────────────────
    # request.counterparty.id may be a user-defined label (e.g. "CP-001") or a
    # real DB UUID.  We try both the id column and external_id so that either
    # format works.  Falls back to None if no match — nullable FK is fine.
    cp_id_row = await db.execute(
        select(Counterparty.id).where(
            or_(
                Counterparty.id          == request.counterparty.id,
                Counterparty.external_id == request.counterparty.id,
            )
        )
    )
    resolved_cp_id: Optional[str] = cp_id_row.scalar_one_or_none()

    # ── Create simulation run record ─────────────────────────────────────────
    sim_run = SimulationRun(
        triggered_by    = current_user.id,
        trigger_type    = TriggerType.MANUAL,
        sim_params_json = request.sim_params.model_dump(),
        stress_json     = request.stress.model_dump() if request.stress else None,
        status          = SimStatus.RUNNING,
        note            = request.note,
    )
    db.add(sim_run)
    await db.flush()   # get sim_run.id without committing

    # ── Run engine ───────────────────────────────────────────────────────────
    result = await run_simulation(request)

    if not result.success:
        logger.error(
            "simulate: engine failure — run_id=%s counterparty=%s error=%r",
            sim_run.id,
            request.counterparty.id,
            result.error_msg,
        )
        sim_run.status       = SimStatus.FAILED
        sim_run.error_msg    = result.error_msg
        sim_run.completed_at = datetime.now(timezone.utc)
        await db.commit()
        raise HTTPException(status_code=500, detail=result.error_msg)

    # ── Persist base metrics ─────────────────────────────────────────────────
    base = result.base
    await _persist_metrics(db, sim_run.id, resolved_cp_id, base, is_stressed=False)

    if result.stressed:
        await _persist_metrics(db, sim_run.id, resolved_cp_id, result.stressed, is_stressed=True)

    sim_run.status       = SimStatus.DONE
    sim_run.completed_at = datetime.now(timezone.utc)

    # ── Margin call detection ────────────────────────────────────────────────
    collateral = request.counterparty.collateral
    if base.margin_required > collateral:
        logger.warning(
            "simulate: margin call triggered — counterparty=%s run_id=%s "
            "margin_required=%.2f collateral=%.2f excess=%.2f",
            request.counterparty.id,
            sim_run.id,
            base.margin_required,
            collateral,
            base.margin_required - collateral,
        )
    # Only create a MarginCall when we have a real FK-valid counterparty_id
    if resolved_cp_id:
        await check_and_alert_margin_calls(
            db,
            counterparty_id   = resolved_cp_id,
            margin_required   = base.margin_required,
            collateral        = collateral,
            simulation_run_id = sim_run.id,
        )

    # ── Audit log ────────────────────────────────────────────────────────────
    await log_event(
        db,
        action        = "simulate",
        user_id       = current_user.id,
        resource_type = "simulation_run",
        resource_id   = sim_run.id,
        detail        = {
            "counterparty_id": request.counterparty.id,
            "cva":             base.cva,
            "margin_required": base.margin_required,
        },
    )

    await db.commit()
    logger.info(
        "simulate: complete — run_id=%s counterparty=%s cva=%.5f margin=%.2f time_us=%d",
        sim_run.id,
        request.counterparty.id,
        base.cva,
        base.margin_required,
        base.compute_time_us,
    )
    return result


async def _persist_metrics(
    db:             AsyncSession,
    run_id:         str,
    counterparty_id: Optional[str],
    metrics,
    is_stressed:    bool,
) -> None:
    row = RiskMetric(
        simulation_run_id = run_id,
        counterparty_id   = counterparty_id,
        cva               = metrics.cva,
        wwr_cva           = metrics.wwr_cva,
        epe_profile       = json.dumps(metrics.epe_profile),
        pfe_profile       = json.dumps(metrics.pfe_profile),
        time_grid_years   = json.dumps(metrics.time_grid_years),
        margin_required   = metrics.margin_required,
        compute_time_us   = metrics.compute_time_us,
        is_stressed       = is_stressed,
    )
    db.add(row)


# ── Simulation history ────────────────────────────────────────────────────────

@router.get("/simulate/history", response_model=List[SimulationHistoryItem])
async def simulate_history(
    counterparty_id: Optional[str] = Query(None),
    limit:           int            = Query(50, ge=1, le=200),
    offset:          int            = Query(0, ge=0),
    db:              AsyncSession   = Depends(get_db),
    _u:              User           = Depends(get_current_user),
) -> List[SimulationHistoryItem]:
    """Return past simulation results, most recent first, with run note."""
    stmt = (
        select(RiskMetric, SimulationRun.note)
        .outerjoin(SimulationRun, RiskMetric.simulation_run_id == SimulationRun.id)
        .order_by(RiskMetric.time.desc())
        .limit(limit)
        .offset(offset)
    )
    if counterparty_id:
        stmt = stmt.where(RiskMetric.counterparty_id == counterparty_id)
    result = await db.execute(stmt)

    items: List[SimulationHistoryItem] = []
    for row, note in result:
        items.append(SimulationHistoryItem(
            id              = row.id,
            run_id          = row.simulation_run_id,
            counterparty_id = row.counterparty_id,
            cva             = row.cva,
            wwr_cva         = row.wwr_cva,
            margin_required = row.margin_required,
            is_stressed     = row.is_stressed,
            compute_time_us = row.compute_time_us,
            time            = row.time,
            pfe_profile     = json.loads(row.pfe_profile) if row.pfe_profile else [],
            epe_profile     = json.loads(row.epe_profile) if row.epe_profile else [],
            time_grid_years = json.loads(row.time_grid_years) if row.time_grid_years else [],
            note            = note,
        ))
    return items


# ── Simulation comparison ─────────────────────────────────────────────────────

class CompareRequest(BaseModel):
    run_ids: List[str]


@router.post("/simulate/compare", response_model=List[SimulationHistoryItem])
async def simulate_compare(
    body: CompareRequest,
    db:   AsyncSession = Depends(get_db),
    _u:   User         = Depends(get_current_user),
) -> List[SimulationHistoryItem]:
    """Return metrics for a set of simulation run IDs side-by-side."""
    if not body.run_ids:
        return []
    if len(body.run_ids) > 20:
        raise HTTPException(status_code=422, detail="Maximum 20 run_ids per comparison")

    stmt = (
        select(RiskMetric)
        .where(RiskMetric.simulation_run_id.in_(body.run_ids))
        .order_by(RiskMetric.time.desc())
    )
    result = await db.execute(stmt)

    items: List[SimulationHistoryItem] = []
    for row in result.scalars().all():
        items.append(SimulationHistoryItem(
            id              = row.id,
            run_id          = row.simulation_run_id,
            counterparty_id = row.counterparty_id,
            cva             = row.cva,
            wwr_cva         = row.wwr_cva,
            margin_required = row.margin_required,
            is_stressed     = row.is_stressed,
            compute_time_us = row.compute_time_us,
            time            = row.time,
            pfe_profile     = json.loads(row.pfe_profile) if row.pfe_profile else [],
            epe_profile     = json.loads(row.epe_profile) if row.epe_profile else [],
            time_grid_years = json.loads(row.time_grid_years) if row.time_grid_years else [],
        ))
    return items


# ── PDF export ────────────────────────────────────────────────────────────────

@router.get("/simulate/{run_id}/export/pdf")
async def export_pdf(
    run_id:       str,
    db:           AsyncSession = Depends(get_db),
    current_user: User         = Depends(get_current_user),
):
    """Download a PDF report for a completed simulation run."""
    rows = await _load_run_metrics(db, run_id)
    if not rows:
        raise HTTPException(status_code=404, detail="Simulation run not found")

    base_row    = next((r for r in rows if not r.is_stressed), None)
    stress_row  = next((r for r in rows if r.is_stressed),     None)

    if base_row is None:
        raise HTTPException(status_code=404, detail="Base metrics not found for this run")

    # Counterparty info (best-effort — may be deleted)
    cp: Optional[Counterparty] = None
    if base_row.counterparty_id:
        cp_result = await db.execute(
            select(Counterparty).where(Counterparty.id == base_row.counterparty_id)
        )
        cp = cp_result.scalar_one_or_none()

    # Recent margin calls for this counterparty
    mc_rows: list = []
    if base_row.counterparty_id:
        mc_result = await db.execute(
            select(MarginCall)
            .where(MarginCall.counterparty_id == base_row.counterparty_id)
            .order_by(MarginCall.issued_at.desc())
            .limit(10)
        )
        mc_rows = [
            {
                "issued_at":       r.issued_at,
                "amount":          r.amount,
                "excess_exposure": r.excess_exposure,
                "status":          r.status,
                "reason":          r.reason,
            }
            for r in mc_result.scalars().all()
        ]

    from server.core.engine_runner import engine_info as _engine_info
    from server.reports.exporter import export_simulation_pdf

    cp_dict: Dict[str, Any] = {
        "name":          cp.name          if cp else base_row.counterparty_id or "—",
        "credit_rating": cp.credit_rating if cp else "—",
        "hazard_rate":   cp.hazard_rate   if cp else 0.0,
        "recovery_rate": cp.recovery_rate if cp else 0.0,
        "collateral":    cp.collateral    if cp else 0.0,
    }

    pdf_bytes = export_simulation_pdf(
        run_id       = run_id,
        generated_by = current_user.username,
        counterparty = cp_dict,
        base         = _row_to_dict(base_row),
        stressed     = _row_to_dict(stress_row) if stress_row else None,
        margin_calls = mc_rows,
        engine_info  = _engine_info(),
    )

    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="ccr-report-{run_id[:8]}.pdf"'},
    )


# ── CSV export ────────────────────────────────────────────────────────────────

@router.get("/simulate/{run_id}/export/csv")
async def export_csv(
    run_id: str,
    db:     AsyncSession = Depends(get_db),
    _u:     User         = Depends(get_current_user),
):
    """Download a CSV of PFE/EPE profile for a simulation run."""
    rows = await _load_run_metrics(db, run_id)
    if not rows:
        raise HTTPException(status_code=404, detail="Simulation run not found")

    base_row = next((r for r in rows if not r.is_stressed), None)
    if base_row is None:
        raise HTTPException(status_code=404, detail="Base metrics not found for this run")

    from server.reports.exporter import export_simulation_csv

    csv_bytes = export_simulation_csv(
        pfe_profile     = json.loads(base_row.pfe_profile)     if base_row.pfe_profile     else [],
        epe_profile     = json.loads(base_row.epe_profile)     if base_row.epe_profile     else [],
        time_grid_years = json.loads(base_row.time_grid_years) if base_row.time_grid_years else [],
    )

    return StreamingResponse(
        io.BytesIO(csv_bytes),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="ccr-profile-{run_id[:8]}.csv"'},
    )


# ── Margin calls CSV export ───────────────────────────────────────────────────

@router.get("/margin-calls/export/csv")
async def export_margin_calls_csv_endpoint(
    counterparty_id: Optional[str] = Query(None),
    limit:           int            = Query(200, ge=1, le=1000),
    db:              AsyncSession   = Depends(get_db),
    _u:              User           = Depends(get_current_user),
):
    """Download all margin calls as CSV."""
    stmt = select(MarginCall).order_by(MarginCall.issued_at.desc()).limit(limit)
    if counterparty_id:
        stmt = stmt.where(MarginCall.counterparty_id == counterparty_id)
    result = await db.execute(stmt)

    rows = [
        {
            "id":               r.id,
            "counterparty_id":  r.counterparty_id,
            "amount":           r.amount,
            "excess_exposure":  r.excess_exposure,
            "status":           r.status,
            "reason":           r.reason,
            "issued_at":        r.issued_at.isoformat() if r.issued_at else "",
        }
        for r in result.scalars().all()
    ]

    from server.reports.exporter import export_margin_calls_csv

    csv_bytes = export_margin_calls_csv(rows)
    return StreamingResponse(
        io.BytesIO(csv_bytes),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="margin-calls.csv"'},
    )


# ── Audit log ─────────────────────────────────────────────────────────────────

class AuditLogItem(BaseModel):
    id:            str
    time:          datetime
    user_id:       Optional[str]
    action:        str
    resource_type: str
    resource_id:   Optional[str]
    detail:        Optional[Dict[str, Any]]
    ip_address:    Optional[str]

    model_config = {"from_attributes": False}


@router.get("/audit-log", response_model=List[AuditLogItem])
async def get_audit_log(
    action:       Optional[str]      = Query(None),
    resource_type: Optional[str]     = Query(None),
    from_dt:      Optional[datetime] = Query(None, alias="from"),
    to_dt:        Optional[datetime] = Query(None, alias="to"),
    limit:        int                = Query(100, ge=1, le=500),
    db:           AsyncSession       = Depends(get_db),
    _u:           User               = Depends(require_role(Role.ADMIN, Role.AUDITOR)),
) -> List[AuditLogItem]:
    """Query the audit log. ADMIN and AUDITOR only."""
    stmt = select(AuditLog).order_by(AuditLog.time.desc()).limit(limit)
    if action:
        stmt = stmt.where(AuditLog.action == action)
    if resource_type:
        stmt = stmt.where(AuditLog.resource_type == resource_type)
    if from_dt:
        stmt = stmt.where(AuditLog.time >= from_dt)
    if to_dt:
        stmt = stmt.where(AuditLog.time <= to_dt)

    result = await db.execute(stmt)
    return [
        AuditLogItem(
            id            = r.id,
            time          = r.time,
            user_id       = r.user_id,
            action        = r.action,
            resource_type = r.resource_type,
            resource_id   = r.resource_id,
            detail        = r.detail,
            ip_address    = r.ip_address,
        )
        for r in result.scalars().all()
    ]


# ── My activity (user-scoped audit) ──────────────────────────────────────────

@router.get("/me/activity", response_model=List[AuditLogItem])
async def get_my_activity(
    since:        Optional[datetime] = Query(None),
    limit:        int                = Query(20, ge=1, le=100),
    db:           AsyncSession       = Depends(get_db),
    current_user: User               = Depends(get_current_user),
) -> List[AuditLogItem]:
    """Return recent audit events for the current authenticated user."""
    stmt = (
        select(AuditLog)
        .where(AuditLog.user_id == current_user.id)
        .order_by(AuditLog.time.desc())
        .limit(limit)
    )
    if since:
        stmt = stmt.where(AuditLog.time >= since)
    result = await db.execute(stmt)
    return [
        AuditLogItem(
            id            = r.id,
            time          = r.time,
            user_id       = r.user_id,
            action        = r.action,
            resource_type = r.resource_type,
            resource_id   = r.resource_id,
            detail        = r.detail,
            ip_address    = r.ip_address,
        )
        for r in result.scalars().all()
    ]


# ── Risk concentration ────────────────────────────────────────────────────────

class ConcentrationItem(BaseModel):
    counterparty_id:   str
    counterparty_name: Optional[str]
    cva:               float
    margin_required:   float
    last_run_time:     datetime


@router.get("/analytics/concentration", response_model=List[ConcentrationItem])
async def get_concentration(
    limit: int          = Query(20, ge=1, le=100),
    db:    AsyncSession = Depends(get_db),
    _u:    User         = Depends(get_current_user),
) -> List[ConcentrationItem]:
    """Return the latest non-stressed risk metrics per counterparty, ranked by CVA."""
    subq = (
        select(
            RiskMetric.counterparty_id,
            func.max(RiskMetric.time).label("max_time"),
        )
        .where(RiskMetric.counterparty_id.is_not(None))
        .where(RiskMetric.is_stressed.is_(False))
        .group_by(RiskMetric.counterparty_id)
        .subquery()
    )
    stmt = (
        select(RiskMetric, Counterparty.name)
        .join(
            subq,
            (RiskMetric.counterparty_id == subq.c.counterparty_id) &
            (RiskMetric.time == subq.c.max_time),
        )
        .outerjoin(Counterparty, Counterparty.id == RiskMetric.counterparty_id)
        .where(RiskMetric.is_stressed.is_(False))
        .order_by(RiskMetric.cva.desc())
        .limit(limit)
    )
    result = await db.execute(stmt)
    items = []
    for row in result.all():
        rm, name = row[0], row[1]
        items.append(ConcentrationItem(
            counterparty_id   = rm.counterparty_id,
            counterparty_name = name,
            cva               = rm.cva,
            margin_required   = rm.margin_required,
            last_run_time     = rm.time,
        ))
    return items


# ── CVA attribution ───────────────────────────────────────────────────────────

class AttributionItem(BaseModel):
    deriv_id:       str
    deriv_type:     str
    notional:       float
    maturity_years: float
    weight:         float
    allocated_cva:  float


@router.get("/simulate/{run_id}/attribution", response_model=List[AttributionItem])
async def get_attribution(
    run_id: str,
    db:     AsyncSession = Depends(get_db),
    _u:     User         = Depends(get_current_user),
) -> List[AttributionItem]:
    """
    Notional-weighted CVA attribution per derivative for a simulation run.

    CVA_i ≈ CVA_total × (notional_i × maturity_i) / Σ(notional_j × maturity_j)

    Note: the portfolio spec is not persisted at simulation time, so this
    endpoint uses the current live derivatives for the associated counterparty.
    """
    rows = await _load_run_metrics(db, run_id)
    base_row = next((r for r in rows if not r.is_stressed), None)
    if base_row is None:
        raise HTTPException(status_code=404, detail="Simulation run not found")

    cva_total = base_row.cva
    if not base_row.counterparty_id:
        return []

    # Fetch current live portfolio derivatives for this counterparty
    port_result = await db.execute(
        select(Portfolio).where(Portfolio.counterparty_id == base_row.counterparty_id)
    )
    portfolios_list = port_result.scalars().all()

    all_derivs: List[Derivative] = []
    for port in portfolios_list:
        deriv_result = await db.execute(
            select(Derivative).where(Derivative.portfolio_id == port.id)
        )
        all_derivs.extend(deriv_result.scalars().all())

    if not all_derivs:
        return []

    total_weight = sum(d.notional * d.maturity_years for d in all_derivs)
    if total_weight == 0:
        return []

    items = []
    for d in all_derivs:
        w = (d.notional * d.maturity_years) / total_weight
        items.append(AttributionItem(
            deriv_id       = d.id,
            deriv_type     = d.deriv_type,
            notional       = d.notional,
            maturity_years = d.maturity_years,
            weight         = w,
            allocated_cva  = cva_total * w,
        ))
    items.sort(key=lambda x: x.allocated_cva, reverse=True)
    return items


# ── Auto-run (bulk simulation trigger) ───────────────────────────────────────

_DERIV_TYPE_MAP = {"IRS": 0, "CDS": 1, "FX": 2, "EQUITY": 3, "COMMODITY": 4}
_RATING_MAP     = {"AAA": 0, "AA": 1, "A": 2, "BBB": 3, "BB": 4, "B": 5, "CCC": 6, "D": 7}


class AutoRunResult(BaseModel):
    counterparty_id:   str
    counterparty_name: str
    success:           bool
    cva:               float = 0.0
    margin_required:   float = 0.0
    error:             Optional[str] = None


@router.post("/simulate/auto-run", response_model=List[AutoRunResult])
async def trigger_auto_run(
    db:           AsyncSession = Depends(get_db),
    current_user: User         = Depends(require_role(Role.RISK_MANAGER, Role.ADMIN)),
) -> List[AutoRunResult]:
    """
    Run one simulation per counterparty that has at least one auto_run portfolio.
    Uses each counterparty's stored parameters and live portfolio derivatives.
    """
    from server.models.schemas import (
        CounterpartyRequest,
        DerivativeSpecRequest,
        DerivativeType,
        PortfolioRequest,
        SimParamsRequest,
        SimulationRequest,
    )

    # Fetch all portfolios with auto_run=True, with their counterparty
    port_result = await db.execute(
        select(Portfolio, Counterparty)
        .join(Counterparty, Portfolio.counterparty_id == Counterparty.id)
        .where(Portfolio.auto_run.is_(True))
    )
    rows = port_result.all()
    if not rows:
        return []

    # Group portfolios by counterparty
    cp_portfolios: Dict[str, Any] = {}
    for port, cp in rows:
        if cp.id not in cp_portfolios:
            cp_portfolios[cp.id] = {"cp": cp, "ports": []}
        cp_portfolios[cp.id]["ports"].append(port)

    results: List[AutoRunResult] = []
    for cp_id, data in cp_portfolios.items():
        cp   = data["cp"]
        ports = data["ports"]

        # Collect all derivatives from all auto_run portfolios for this CP
        all_derivs: List[Derivative] = []
        for port in ports:
            dr = await db.execute(select(Derivative).where(Derivative.portfolio_id == port.id))
            all_derivs.extend(dr.scalars().all())

        deriv_specs = [
            DerivativeSpecRequest(
                id               = d.id,
                type             = DerivativeType(_DERIV_TYPE_MAP.get(d.deriv_type, 0)),
                notional         = d.notional,
                maturity_years   = d.maturity_years,
                underlying_price = d.underlying_price,
                strike           = d.strike,
                cash_flow_freq   = d.cash_flow_freq,
            )
            for d in all_derivs
        ]
        if not deriv_specs:
            results.append(AutoRunResult(
                counterparty_id=cp.id, counterparty_name=cp.name,
                success=False, error="No derivatives in auto_run portfolios",
            ))
            continue

        request = SimulationRequest(
            sim_params   = SimParamsRequest(),
            counterparty = CounterpartyRequest(
                id               = cp.id,
                name             = cp.name,
                credit_rating    = _RATING_MAP.get(cp.credit_rating, 3),
                hazard_rate      = cp.hazard_rate,
                recovery_rate    = cp.recovery_rate,
                collateral       = cp.collateral,
                margin_threshold = cp.margin_threshold,
                mpor_days        = cp.mpor_days,
            ),
            portfolio = PortfolioRequest(
                id               = ports[0].id,
                counterparty_id  = cp.id,
                derivatives      = deriv_specs,
                collateral       = sum(p.collateral for p in ports),
                net_value        = sum(p.net_value for p in ports),
            ),
            note = "auto_run",
        )

        try:
            err = _ccr.CcrEngine.validate_config(build_engine_config(request))
            if err:
                results.append(AutoRunResult(
                    counterparty_id=cp.id, counterparty_name=cp.name,
                    success=False, error=err,
                ))
                continue

            sim_run = SimulationRun(
                triggered_by    = current_user.id,
                trigger_type    = TriggerType.AUTO_RERUN,
                sim_params_json = request.sim_params.model_dump(),
                status          = SimStatus.RUNNING,
                note            = "auto_run",
            )
            db.add(sim_run)
            await db.flush()

            result = await run_simulation(request)
            if not result.success:
                sim_run.status    = SimStatus.FAILED
                sim_run.error_msg = result.error_msg
                sim_run.completed_at = datetime.now(timezone.utc)
                await db.commit()
                results.append(AutoRunResult(
                    counterparty_id=cp.id, counterparty_name=cp.name,
                    success=False, error=result.error_msg,
                ))
                continue

            await _persist_metrics(db, sim_run.id, cp.id, result.base, is_stressed=False)
            sim_run.status       = SimStatus.DONE
            sim_run.completed_at = datetime.now(timezone.utc)
            await check_and_alert_margin_calls(
                db,
                counterparty_id   = cp.id,
                margin_required   = result.base.margin_required,
                collateral        = cp.collateral,
                simulation_run_id = sim_run.id,
            )
            await db.commit()
            results.append(AutoRunResult(
                counterparty_id=cp.id, counterparty_name=cp.name,
                success=True,
                cva=result.base.cva,
                margin_required=result.base.margin_required,
            ))
        except Exception as exc:
            await db.rollback()
            results.append(AutoRunResult(
                counterparty_id=cp.id, counterparty_name=cp.name,
                success=False, error=str(exc),
            ))

    return results


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _load_run_metrics(db: AsyncSession, run_id: str) -> List[RiskMetric]:
    """Load all RiskMetric rows for a simulation_run_id."""
    result = await db.execute(
        select(RiskMetric)
        .where(RiskMetric.simulation_run_id == run_id)
        .order_by(RiskMetric.is_stressed)
    )
    return list(result.scalars().all())


def _row_to_dict(row: RiskMetric) -> Dict[str, Any]:
    return {
        "cva":             row.cva,
        "wwr_cva":         row.wwr_cva,
        "margin_required": row.margin_required,
        "compute_time_us": row.compute_time_us,
        "pfe_profile":     json.loads(row.pfe_profile)     if row.pfe_profile     else [],
        "epe_profile":     json.loads(row.epe_profile)     if row.epe_profile     else [],
        "time_grid_years": json.loads(row.time_grid_years) if row.time_grid_years else [],
        "arch_used":       "",  # stored in engine_info, not per-run
    }
