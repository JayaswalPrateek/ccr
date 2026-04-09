"""
APScheduler background jobs.

setup_scheduler() — call from the FastAPI lifespan after startup.

Jobs:
    refresh_market_params_job  — every 15 minutes, fetches real prices + rates
    scheduled_rerun_job        — every hour, re-runs simulations for auto_run portfolios
"""

from __future__ import annotations

import json
import logging

from apscheduler.schedulers.asyncio import AsyncIOScheduler

logger    = logging.getLogger(__name__)
scheduler = AsyncIOScheduler(timezone="UTC")


async def refresh_market_params_job() -> None:
    """Fetch live market data and upsert into the DB."""
    from server.core.database import AsyncSessionLocal
    from server.market_data.fetcher import refresh_market_params

    async with AsyncSessionLocal() as db:
        await refresh_market_params(db)


async def scheduled_rerun_job() -> None:
    """Re-run simulations for all portfolios with auto_run=True.

    For each qualifying portfolio:
      1. Load counterparty + derivatives from DB.
      2. Fetch latest sigma / mu from market_params cache.
      3. Build a SimulationRequest and call the engine.
      4. Persist RiskMetric + check margin calls.
    """
    from datetime import datetime, timezone

    from sqlalchemy import select

    from server.core.database import AsyncSessionLocal
    from server.core.engine_runner import run_simulation
    from server.market_data.fetcher import get_drift_from_db, get_sigma_for_symbol
    from server.models.db_models import (
        Counterparty,
        Derivative,
        MarginCall,
        Portfolio,
        RiskMetric,
        SimulationRun,
        SimStatus,
        TriggerType,
    )
    from server.models.schemas import (
        CounterpartyRequest,
        CreditRating,
        DerivativeSpecRequest,
        DerivativeType,
        GridType,
        PortfolioRequest,
        SimMode,
        SimParamsRequest,
        SimulationRequest,
    )
    from server.notifications.alerts import check_and_alert_margin_calls

    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(Portfolio).where(Portfolio.auto_run == True)  # noqa: E712
        )
        portfolios = result.scalars().all()

    if not portfolios:
        return

    logger.info("Scheduled rerun: %d auto-run portfolio(s)", len(portfolios))

    for port in portfolios:
        try:
            async with AsyncSessionLocal() as db:
                # Load counterparty.
                cp_result = await db.execute(
                    select(Counterparty).where(Counterparty.id == port.counterparty_id)
                )
                cp = cp_result.scalar_one_or_none()
                if cp is None:
                    logger.warning("auto_rerun: counterparty %s not found for portfolio %s",
                                   port.counterparty_id, port.id)
                    continue

                # Load derivatives.
                deriv_result = await db.execute(
                    select(Derivative).where(Derivative.portfolio_id == port.id)
                )
                derivs = deriv_result.scalars().all()
                if not derivs:
                    logger.warning("auto_rerun: no derivatives in portfolio %s — skipping", port.id)
                    continue

                # Fetch latest sigma + drift from market params.
                first_sym = "SPY"   # representative equity; could be per-derivative
                sigma = await get_sigma_for_symbol(first_sym, db)
                mu    = await get_drift_from_db(db)

                deriv_specs = [
                    DerivativeSpecRequest(
                        id               = d.external_id,
                        type             = DerivativeType(
                            {"IRS": 0, "CDS": 1, "FX": 2, "EQUITY": 3, "COMMODITY": 4}
                            .get(d.deriv_type, 0)
                        ),
                        notional         = d.notional,
                        maturity_years   = d.maturity_years,
                        underlying_price = d.underlying_price,
                        strike           = d.strike,
                        cash_flow_freq   = d.cash_flow_freq,
                    )
                    for d in derivs
                ]

                request = SimulationRequest(
                    sim_params = SimParamsRequest(
                        num_paths     = 5000,
                        num_timesteps = 12,
                        num_assets    = 1,
                        mu            = mu,
                        sigma         = sigma,
                        rho_wwr       = 0.0,
                        recovery_rate = cp.recovery_rate,
                        horizon_years = 1.0,
                        mode          = SimMode.STANDARD,
                        grid_type     = GridType.MONTHLY,
                    ),
                    counterparty = CounterpartyRequest(
                        id               = cp.id,
                        name             = cp.name,
                        credit_rating    = CreditRating.__members__.get(cp.credit_rating, CreditRating.BBB),
                        hazard_rate      = cp.hazard_rate,
                        recovery_rate    = cp.recovery_rate,
                        collateral       = cp.collateral,
                        margin_threshold = cp.margin_threshold,
                        mpor_days        = cp.mpor_days,
                    ),
                    portfolio = PortfolioRequest(
                        id               = port.id,
                        counterparty_id  = cp.id,
                        derivatives      = deriv_specs,
                        collateral       = port.collateral,
                        net_value        = port.net_value,
                    ),
                    enable_collateral     = True,
                    deterministic_quantile= True,
                    rng_seed              = 0,   # non-deterministic seed for scheduled runs
                )

                # Create run record.
                sim_run = SimulationRun(
                    counterparty_id = cp.id,
                    trigger_type    = TriggerType.SCHEDULED,
                    sim_params_json = request.sim_params.model_dump(),
                    status          = SimStatus.RUNNING,
                )
                db.add(sim_run)
                await db.flush()

                result_obj = await run_simulation(request)

                if not result_obj.success:
                    sim_run.status    = SimStatus.FAILED
                    sim_run.error_msg = result_obj.error_msg
                    await db.commit()
                    logger.warning("auto_rerun failed for portfolio %s: %s",
                                   port.id, result_obj.error_msg)
                    continue

                base = result_obj.base
                metric = RiskMetric(
                    simulation_run_id = sim_run.id,
                    counterparty_id   = cp.id,
                    cva               = base.cva,
                    wwr_cva           = base.wwr_cva,
                    epe_profile       = json.dumps(base.epe_profile),
                    pfe_profile       = json.dumps(base.pfe_profile),
                    time_grid_years   = json.dumps(base.time_grid_years),
                    margin_required   = base.margin_required,
                    compute_time_us   = base.compute_time_us,
                    is_stressed       = False,
                )
                db.add(metric)

                sim_run.status       = SimStatus.DONE
                sim_run.completed_at = datetime.now(timezone.utc)

                await check_and_alert_margin_calls(
                    db,
                    counterparty_id   = cp.id,
                    margin_required   = base.margin_required,
                    collateral        = cp.collateral,
                    simulation_run_id = sim_run.id,
                )

                await db.commit()
                logger.info(
                    "auto_rerun complete: portfolio=%s  CVA=%.5f  margin=%.2f",
                    port.id, base.cva, base.margin_required,
                )

        except Exception as exc:
            logger.error("auto_rerun exception for portfolio %s: %s", port.id, exc, exc_info=True)


def setup_scheduler() -> None:
    """Register all jobs and start the scheduler.

    Must be called from within a running asyncio event loop (FastAPI lifespan).
    Respects the ``SCHEDULER_ENABLED`` env var — set to ``false`` to disable
    all background jobs without redeploying (useful during incidents or tests).
    """
    from server.core.config import settings

    if not settings.scheduler_enabled:
        logger.warning(
            "Scheduler is DISABLED (SCHEDULER_ENABLED=false). "
            "No background jobs will run."
        )
        return

    scheduler.add_job(
        refresh_market_params_job,
        trigger          = "interval",
        minutes          = 15,
        id               = "market_refresh",
        replace_existing = True,
    )
    scheduler.add_job(
        scheduled_rerun_job,
        trigger          = "interval",
        hours            = 1,
        id               = "auto_rerun",
        replace_existing = True,
    )
    scheduler.start()
    logger.info("Scheduler started (market refresh every 15 min, auto-rerun every 1 hr)")
