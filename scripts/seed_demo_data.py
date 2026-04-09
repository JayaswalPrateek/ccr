#!/usr/bin/env python3
"""
Seed demo data for the CCR dashboard.

Creates:
  - An auditor user  (auditor / auditor123)
  - A risk manager   (risk / risk123)
  - Two counterparties (Alpha Bank, Beta Capital)
  - Two portfolios   (one IRS-only, one mixed IRS+CDS)
  - Derivatives in each portfolio
  - Runs two initial simulations so the dashboard has charts to show

Usage:
    python scripts/seed_demo_data.py

Requires:
    - DATABASE_URL env var (or .env file at project root)
    - Server dependencies installed (uv sync)
    - The CCR engine .so present in server/bindings/
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

# ── Path setup ───────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "server" / "bindings"))

# Load .env before importing settings (no python-dotenv required — plain parse).
_env_file = PROJECT_ROOT / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())

os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://ccr:ccr@localhost:5432/ccr")
os.environ.setdefault("JWT_SECRET", "dev-secret-change-me")

from sqlalchemy import select, text

from server.auth.security import hash_password
from server.core.database import AsyncSessionLocal
from server.core.engine_runner import run_simulation
from server.models.db_models import (
    Counterparty,
    Derivative,
    MarginCall,
    Portfolio,
    RiskMetric,
    SimulationRun,
    SimStatus,
    TriggerType,
    User,
    UserRole,
)
from server.models.schemas import (
    CounterpartyRequest,
    DerivativeSpecRequest,
    DerivativeType,
    GridType,
    PortfolioRequest,
    SimMode,
    SimParamsRequest,
    SimulationRequest,
)
from server.notifications.audit import log_event


# ── Helpers ───────────────────────────────────────────────────────────────────

def _print(msg: str) -> None:
    print(f"  {msg}")


async def _get_or_create_user(
    db, username: str, email: str, password: str, role: str
) -> User:
    result = await db.execute(select(User).where(User.username == username))
    user = result.scalar_one_or_none()
    if user:
        _print(f"User '{username}' already exists — skipping")
        return user
    user = User(
        username  = username,
        email     = email,
        hashed_pw = hash_password(password),
        role      = role,
        is_active = True,
    )
    db.add(user)
    await db.flush()
    _print(f"Created user: {username} / {password}  [{role}]")
    return user


async def _get_or_create_counterparty(db, **kwargs) -> Counterparty:
    result = await db.execute(
        select(Counterparty).where(Counterparty.external_id == kwargs["external_id"])
    )
    cp = result.scalar_one_or_none()
    if cp:
        _print(f"Counterparty '{kwargs['name']}' already exists — skipping")
        return cp
    cp = Counterparty(**kwargs)
    db.add(cp)
    await db.flush()
    _print(f"Created counterparty: {kwargs['name']}")
    return cp


async def _get_or_create_portfolio(db, **kwargs) -> Portfolio:
    result = await db.execute(
        select(Portfolio).where(Portfolio.external_id == kwargs["external_id"])
    )
    port = result.scalar_one_or_none()
    if port:
        _print(f"Portfolio '{kwargs['external_id']}' already exists — skipping")
        return port
    port = Portfolio(**kwargs)
    db.add(port)
    await db.flush()
    _print(f"Created portfolio: {kwargs['external_id']}")
    return port


async def _run_demo_simulation(
    db,
    request: SimulationRequest,
    label: str,
    triggered_by: str,
) -> None:
    """Run one simulation and persist metrics."""
    _print(f"Running simulation: {label}…")
    sim_run = SimulationRun(
        triggered_by    = triggered_by,
        trigger_type    = TriggerType.MANUAL,
        sim_params_json = request.sim_params.model_dump(),
        stress_json     = None,
        status          = SimStatus.RUNNING,
    )
    db.add(sim_run)
    await db.flush()

    result = await run_simulation(request)

    if not result.success:
        sim_run.status    = SimStatus.FAILED
        sim_run.error_msg = result.error_msg
        _print(f"  Simulation failed: {result.error_msg}")
        return

    sim_run.status = SimStatus.DONE

    base = result.base
    metric = RiskMetric(
        simulation_run_id = sim_run.id,
        counterparty_id   = request.counterparty.id,
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

    await log_event(
        db,
        action        = "seed_simulate",
        user_id       = triggered_by,
        resource_type = "simulation_run",
        resource_id   = sim_run.id,
        detail        = {"label": label, "cva": base.cva},
    )

    _print(
        f"  Done — CVA={base.cva:.5f}  "
        f"PFE_max={max(base.pfe_profile, default=0):.4f}  "
        f"Arch={base.arch_used}  "
        f"Time={base.compute_time_us/1000:.1f}ms"
    )

    # Create a margin call if exposure exceeds collateral.
    if base.margin_required > request.counterparty.collateral:
        mc = MarginCall(
            counterparty_id   = request.counterparty.id,
            simulation_run_id = sim_run.id,
            amount            = base.margin_required,
            excess_exposure   = base.margin_required - request.counterparty.collateral,
            reason            = (
                f"Seed: margin required ({base.margin_required:.2f}) "
                f"exceeds collateral ({request.counterparty.collateral:.2f})"
            ),
        )
        db.add(mc)
        _print(f"  Margin call created: amount={base.margin_required:.2f}")


# ── Main ─────────────────────────────────────────────────────────────────────

async def main() -> None:
    print("\nCCR Seed Demo Data")
    print("=" * 50)

    async with AsyncSessionLocal() as db:
        # ── Users ─────────────────────────────────────────────────────────────
        print("\n[Users]")
        admin = await _get_or_create_user(
            db, "admin", "admin@ccr.local", "admin123", UserRole.ADMIN
        )
        auditor = await _get_or_create_user(
            db, "auditor", "auditor@ccr.local", "auditor123", UserRole.AUDITOR
        )
        risk_mgr = await _get_or_create_user(
            db, "risk", "risk@ccr.local", "risk123", UserRole.RISK_MANAGER
        )

        # ── Counterparties ────────────────────────────────────────────────────
        print("\n[Counterparties]")
        alpha = await _get_or_create_counterparty(
            db,
            external_id    = "CP-ALPHA",
            name           = "Alpha Bank",
            credit_rating  = "A",
            hazard_rate    = 0.015,
            recovery_rate  = 0.40,
            collateral     = 50_000.0,
            margin_threshold = 0.0,
            mpor_days      = 10,
            created_by     = admin.id,
        )
        beta = await _get_or_create_counterparty(
            db,
            external_id    = "CP-BETA",
            name           = "Beta Capital",
            credit_rating  = "BBB",
            hazard_rate    = 0.030,
            recovery_rate  = 0.35,
            collateral     = 10_000.0,
            margin_threshold = 0.0,
            mpor_days      = 10,
            created_by     = admin.id,
        )

        # ── Portfolios ────────────────────────────────────────────────────────
        print("\n[Portfolios & Derivatives]")
        port_alpha = await _get_or_create_portfolio(
            db,
            external_id    = "PORT-ALPHA-001",
            counterparty_id= alpha.id,
            collateral     = 0.0,
            net_value      = 0.0,
            auto_run       = False,
        )
        port_beta = await _get_or_create_portfolio(
            db,
            external_id    = "PORT-BETA-001",
            counterparty_id= beta.id,
            collateral     = 0.0,
            net_value      = 0.0,
            auto_run       = False,
        )

        # Derivatives for Alpha (IRS)
        result = await db.execute(
            select(Derivative).where(Derivative.portfolio_id == port_alpha.id)
        )
        if not result.scalars().first():
            db.add(Derivative(
                external_id     = "DERIV-ALPHA-IRS-1",
                portfolio_id    = port_alpha.id,
                deriv_type      = "IRS",
                notional        = 10_000_000.0,
                maturity_years  = 5.0,
                underlying_price= 0.05,
                strike          = 0.04,
                cash_flow_freq  = 2.0,
            ))
            db.add(Derivative(
                external_id     = "DERIV-ALPHA-IRS-2",
                portfolio_id    = port_alpha.id,
                deriv_type      = "IRS",
                notional        = 5_000_000.0,
                maturity_years  = 3.0,
                underlying_price= 0.045,
                strike          = 0.04,
                cash_flow_freq  = 4.0,
            ))
            await db.flush()
            _print("Created 2 IRS derivatives for Alpha Bank")

        # Derivatives for Beta (IRS + CDS)
        result = await db.execute(
            select(Derivative).where(Derivative.portfolio_id == port_beta.id)
        )
        if not result.scalars().first():
            db.add(Derivative(
                external_id     = "DERIV-BETA-IRS-1",
                portfolio_id    = port_beta.id,
                deriv_type      = "IRS",
                notional        = 8_000_000.0,
                maturity_years  = 7.0,
                underlying_price= 0.055,
                strike          = 0.045,
                cash_flow_freq  = 2.0,
            ))
            db.add(Derivative(
                external_id     = "DERIV-BETA-CDS-1",
                portfolio_id    = port_beta.id,
                deriv_type      = "CDS",
                notional        = 5_000_000.0,
                maturity_years  = 5.0,
                underlying_price= 0.03,
                strike          = 0.02,
                cash_flow_freq  = 4.0,
            ))
            await db.flush()
            _print("Created IRS + CDS derivatives for Beta Capital")

        # ── Simulations ───────────────────────────────────────────────────────
        print("\n[Simulations]")

        # Alpha Bank — conservative, no stress
        alpha_request = SimulationRequest(
            sim_params = SimParamsRequest(
                num_paths     = 5000,
                num_timesteps = 12,
                num_assets    = 1,
                mu            = 0.02,
                sigma         = 0.20,
                rho_wwr       = 0.0,
                recovery_rate = 0.40,
                horizon_years = 1.0,
                mode          = SimMode.STANDARD,
                grid_type     = GridType.MONTHLY,
            ),
            counterparty = CounterpartyRequest(
                id             = alpha.id,
                name           = "Alpha Bank",
                hazard_rate    = 0.015,
                recovery_rate  = 0.40,
                collateral     = 50_000.0,
                margin_threshold = 0.0,
                mpor_days      = 10,
            ),
            portfolio = PortfolioRequest(
                id              = port_alpha.id,
                counterparty_id = alpha.id,
                derivatives     = [
                    DerivativeSpecRequest(
                        id              = "DERIV-ALPHA-IRS-1",
                        type            = DerivativeType.IRS,
                        notional        = 10_000_000.0,
                        maturity_years  = 5.0,
                        underlying_price= 0.05,
                        strike          = 0.04,
                        cash_flow_freq  = 2.0,
                    ),
                ],
                collateral = 50_000.0,
                net_value  = 0.0,
            ),
            enable_wwr            = False,
            enable_jump_diffusion = False,
            enable_collateral     = True,
            deterministic_quantile= True,
            rng_seed              = 42,
        )
        await _run_demo_simulation(db, alpha_request, "Alpha Bank IRS", admin.id)

        # Beta Capital — higher risk, will likely trigger margin call
        beta_request = SimulationRequest(
            sim_params = SimParamsRequest(
                num_paths     = 5000,
                num_timesteps = 12,
                num_assets    = 1,
                mu            = 0.025,
                sigma         = 0.30,
                rho_wwr       = 0.3,
                recovery_rate = 0.35,
                horizon_years = 1.0,
                mode          = SimMode.STANDARD,
                grid_type     = GridType.MONTHLY,
            ),
            counterparty = CounterpartyRequest(
                id             = beta.id,
                name           = "Beta Capital",
                hazard_rate    = 0.030,
                recovery_rate  = 0.35,
                collateral     = 10_000.0,
                margin_threshold = 0.0,
                mpor_days      = 10,
            ),
            portfolio = PortfolioRequest(
                id              = port_beta.id,
                counterparty_id = beta.id,
                derivatives     = [
                    DerivativeSpecRequest(
                        id              = "DERIV-BETA-IRS-1",
                        type            = DerivativeType.IRS,
                        notional        = 8_000_000.0,
                        maturity_years  = 7.0,
                        underlying_price= 0.055,
                        strike          = 0.045,
                        cash_flow_freq  = 2.0,
                    ),
                    DerivativeSpecRequest(
                        id              = "DERIV-BETA-CDS-1",
                        type            = DerivativeType.CDS,
                        notional        = 5_000_000.0,
                        maturity_years  = 5.0,
                        underlying_price= 0.03,
                        strike          = 0.02,
                        cash_flow_freq  = 4.0,
                    ),
                ],
                collateral = 10_000.0,
                net_value  = 0.0,
            ),
            enable_wwr            = True,
            enable_jump_diffusion = False,
            enable_collateral     = True,
            deterministic_quantile= True,
            rng_seed              = 99,
        )
        await _run_demo_simulation(db, beta_request, "Beta Capital IRS+CDS", risk_mgr.id)

        await db.commit()

    print("\n" + "=" * 50)
    print("Seed complete.\n")
    print("Login credentials:")
    print("  admin    / admin123     [ADMIN]")
    print("  risk     / risk123      [RISK_MANAGER]")
    print("  auditor  / auditor123   [AUDITOR]")
    print("\nOpen http://localhost:8000/ (or :5173 for Vite dev server)")


if __name__ == "__main__":
    asyncio.run(main())
