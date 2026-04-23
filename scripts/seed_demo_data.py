#!/usr/bin/env python3
"""
Seed demo data for the CCR dashboard.

Creates:
  - 3 users  (admin, risk manager, auditor)
  - 6 counterparties across different sectors and credit ratings
  - 1–2 portfolios per counterparty with realistic derivatives
    (IRS, CDS, Equity, FX, Commodity)
  - Multiple simulation runs per counterparty (history charts look good)
  - Stressed runs for high-risk counterparties
  - Margin calls in various states (PENDING, ACKNOWLEDGED, SETTLED)
  - Audit log entries covering all major actions

Usage:
    python scripts/seed_demo_data.py

Requires:
    - DATABASE_URL env var (or .env file at project root)
    - Server dependencies installed  (uv sync)
    - The CCR engine .so present in server/bindings/
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# ── Path setup ───────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "server" / "bindings"))

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
    AuditLog,
    Counterparty,
    Derivative,
    MarginCall,
    MarginCallStatus,
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
    StressScenarioRequest as StressParams,
)
from server.models.db_models import PriceHistory
from server.notifications.audit import log_event


# ── Helpers ───────────────────────────────────────────────────────────────────

def _p(msg: str) -> None:
    print(f"  {msg}")


async def _upsert_user(db, username, email, password, role) -> User:
    r = await db.execute(select(User).where(User.username == username))
    u = r.scalar_one_or_none()
    if u:
        _p(f"user '{username}' exists — skip")
        return u
    u = User(username=username, email=email,
             hashed_pw=hash_password(password), role=role, is_active=True)
    db.add(u)
    await db.flush()
    _p(f"created user: {username} / {password}  [{role}]")
    return u


async def _upsert_cp(db, **kw) -> Counterparty:
    r = await db.execute(select(Counterparty).where(Counterparty.external_id == kw["external_id"]))
    cp = r.scalar_one_or_none()
    if cp:
        _p(f"counterparty '{kw['name']}' exists — skip")
        return cp
    cp = Counterparty(**kw)
    db.add(cp)
    await db.flush()
    _p(f"created counterparty: {kw['name']}")
    return cp


async def _upsert_portfolio(db, **kw) -> Portfolio:
    r = await db.execute(select(Portfolio).where(Portfolio.external_id == kw["external_id"]))
    p = r.scalar_one_or_none()
    if p:
        _p(f"portfolio '{kw['external_id']}' exists — skip")
        return p
    p = Portfolio(**kw)
    db.add(p)
    await db.flush()
    _p(f"created portfolio: {kw['external_id']}")
    return p


async def _add_deriv_if_empty(db, port_id, derivs: list[dict]) -> None:
    r = await db.execute(select(Derivative).where(Derivative.portfolio_id == port_id))
    if r.scalars().first():
        _p(f"  derivatives for {port_id[:8]}… exist — skip")
        return
    for d in derivs:
        db.add(Derivative(portfolio_id=port_id, **d))
    await db.flush()
    _p(f"  added {len(derivs)} derivative(s)")


async def _run_sim(db, request: SimulationRequest, label: str, user_id: str,
                   is_stressed: bool = False) -> str | None:
    """Run one simulation, persist metrics + optional margin call. Returns run_id."""
    _p(f"running: {label}…")
    sim_run = SimulationRun(
        triggered_by    = user_id,
        trigger_type    = TriggerType.MANUAL,
        counterparty_id = request.counterparty.id,
        sim_params_json = request.sim_params.model_dump(),
        stress_json     = request.stress.model_dump() if request.stress else None,
        status          = SimStatus.RUNNING,
    )
    db.add(sim_run)
    await db.flush()

    result = await run_simulation(request)

    if not result.success:
        sim_run.status    = SimStatus.FAILED
        sim_run.error_msg = result.error_msg
        _p(f"  FAILED: {result.error_msg}")
        return None

    sim_run.status       = SimStatus.DONE
    sim_run.completed_at = datetime.now(timezone.utc)

    for metrics, stressed_flag in [(result.base, False),
                                   *([(result.stressed, True)] if result.stressed else [])]:
        m = RiskMetric(
            simulation_run_id = sim_run.id,
            counterparty_id   = request.counterparty.id,
            cva               = metrics.cva,
            wwr_cva           = metrics.wwr_cva,
            epe_profile       = json.dumps(metrics.epe_profile),
            pfe_profile       = json.dumps(metrics.pfe_profile),
            time_grid_years   = json.dumps(metrics.time_grid_years),
            margin_required   = metrics.margin_required,
            compute_time_us   = metrics.compute_time_us,
            is_stressed       = stressed_flag,
        )
        db.add(m)

    base = result.base
    _p(f"  CVA={base.cva:.4f}  margin={base.margin_required:,.0f}  "
       f"PFEmax={max(base.pfe_profile, default=0):,.0f}  "
       f"arch={base.arch_used}  {base.compute_time_us/1000:.1f}ms")

    await log_event(db, action="seed_simulate", user_id=user_id,
                    resource_type="simulation_run", resource_id=sim_run.id,
                    detail={"label": label, "cva": base.cva})

    # Margin call if exposure > collateral
    collateral = request.counterparty.collateral
    if base.margin_required > collateral:
        mc = MarginCall(
            counterparty_id   = request.counterparty.id,
            simulation_run_id = sim_run.id,
            amount            = base.margin_required,
            excess_exposure   = base.margin_required - collateral,
            reason            = (
                f"{label}: margin required {base.margin_required:,.0f} "
                f"exceeds collateral {collateral:,.0f}"
            ),
        )
        db.add(mc)
        await db.flush()
        _p(f"  margin call: excess={base.margin_required - collateral:,.0f}")
        return mc.id   # return margin call id so caller can change status
    return sim_run.id


async def _seed_price_history(db) -> None:
    """Insert 90 days of synthetic GBM price paths for backtest data.

    Skips symbols that already have ≥ 80 rows so re-runs are idempotent.
    """
    import math
    import random

    TODAY    = datetime.now(timezone.utc).replace(hour=16, minute=0, second=0, microsecond=0)
    DAYS     = 90
    SYMBOLS  = [
        ("SPX",     4_500.0, 0.18),   # S&P 500 proxy
        ("EUR.USD",     1.08, 0.08),   # EUR/USD FX
        ("OIL.WTI",    80.0, 0.35),   # Crude oil
        ("SONIA",    0.052, 0.04),    # Short rate (IRS underlying proxy)
        ("IG.CDX",   0.032, 0.12),    # Investment-grade CDS spread proxy
    ]

    rng = random.Random(2024_04_23)

    for symbol, s0, annual_vol in SYMBOLS:
        # Check if already seeded
        from sqlalchemy import func as sqlfunc
        count_res = await db.execute(
            select(sqlfunc.count()).select_from(PriceHistory).where(PriceHistory.symbol == symbol)
        )
        if (count_res.scalar() or 0) >= 80:
            _p(f"price history '{symbol}' exists — skip")
            continue

        dt   = 1 / 252          # daily step
        vol  = annual_vol
        mu   = 0.0              # risk-neutral drift for mark-to-model
        price = s0
        rows  = []
        for d in range(DAYS, 0, -1):
            ts    = TODAY - timedelta(days=d)
            z     = rng.gauss(0, 1)
            price = price * math.exp((mu - 0.5 * vol**2) * dt + vol * math.sqrt(dt) * z)
            rows.append(PriceHistory(ts=ts, symbol=symbol, price=round(price, 6), source="seed"))

        db.add_all(rows)
        await db.flush()
        _p(f"price history '{symbol}': {len(rows)} rows  (s0={s0}, σ={annual_vol})")

    await db.commit()


# ── Main ─────────────────────────────────────────────────────────────────────

async def main() -> None:
    print("\nCCR Engine — Demo Seed")
    print("=" * 56)

    async with AsyncSessionLocal() as db:

        # ── Users ─────────────────────────────────────────────────────────────
        print("\n[Users]")
        admin    = await _upsert_user(db, "admin",   "admin@ccr.local",   "admin123",   UserRole.ADMIN)
        risk_mgr = await _upsert_user(db, "risk",    "risk@ccr.local",    "risk123",    UserRole.RISK_MANAGER)
        auditor  = await _upsert_user(db, "auditor", "auditor@ccr.local", "auditor123", UserRole.AUDITOR)

        # ── Counterparties ────────────────────────────────────────────────────
        print("\n[Counterparties]")
        # 6 counterparties across sectors / rating bands
        alpha = await _upsert_cp(db,
            external_id="CP-ALPHA", name="Alpha Bank",
            credit_rating="AA",
            hazard_rate=0.008, recovery_rate=0.42,
            collateral=200_000.0, margin_threshold=50_000.0, mpor_days=10,
            created_by=admin.id)

        beta = await _upsert_cp(db,
            external_id="CP-BETA", name="Beta Capital",
            credit_rating="BBB",
            hazard_rate=0.028, recovery_rate=0.38,
            collateral=80_000.0, margin_threshold=20_000.0, mpor_days=10,
            created_by=admin.id)

        gamma = await _upsert_cp(db,
            external_id="CP-GAMMA", name="Gamma Hedge Fund",
            credit_rating="BB",
            hazard_rate=0.055, recovery_rate=0.30,
            collateral=30_000.0, margin_threshold=0.0, mpor_days=5,
            created_by=risk_mgr.id)

        delta = await _upsert_cp(db,
            external_id="CP-DELTA", name="Delta Energy",
            credit_rating="A",
            hazard_rate=0.018, recovery_rate=0.40,
            collateral=120_000.0, margin_threshold=30_000.0, mpor_days=10,
            created_by=admin.id)

        epsilon = await _upsert_cp(db,
            external_id="CP-EPSILON", name="Epsilon Insurance",
            credit_rating="AA",
            hazard_rate=0.006, recovery_rate=0.45,
            collateral=500_000.0, margin_threshold=100_000.0, mpor_days=15,
            created_by=admin.id)

        zeta = await _upsert_cp(db,
            external_id="CP-ZETA", name="Zeta Corp",
            credit_rating="B",
            hazard_rate=0.090, recovery_rate=0.25,
            collateral=10_000.0, margin_threshold=0.0, mpor_days=5,
            created_by=risk_mgr.id)

        # ── Portfolios & Derivatives ──────────────────────────────────────────
        print("\n[Portfolios & Derivatives]")

        # Alpha Bank — two portfolios: rates desk + FX desk
        pa1 = await _upsert_portfolio(db, external_id="PORT-ALPHA-RATES",
            counterparty_id=alpha.id, collateral=100_000.0, net_value=0.0)
        await _add_deriv_if_empty(db, pa1.id, [
            dict(external_id="D-ALPHA-IRS-1", deriv_type="IRS",
                 notional=25_000_000, maturity_years=7.0,
                 underlying_price=0.052, strike=0.040, cash_flow_freq=2.0),
            dict(external_id="D-ALPHA-IRS-2", deriv_type="IRS",
                 notional=15_000_000, maturity_years=3.0,
                 underlying_price=0.048, strike=0.038, cash_flow_freq=4.0),
            dict(external_id="D-ALPHA-IRS-3", deriv_type="IRS",
                 notional=10_000_000, maturity_years=10.0,
                 underlying_price=0.058, strike=0.045, cash_flow_freq=2.0),
        ])
        pa2 = await _upsert_portfolio(db, external_id="PORT-ALPHA-FX",
            counterparty_id=alpha.id, collateral=50_000.0, net_value=0.0)
        await _add_deriv_if_empty(db, pa2.id, [
            dict(external_id="D-ALPHA-FX-1", deriv_type="FX",
                 notional=10_000_000, maturity_years=1.0,
                 underlying_price=1.085, strike=1.08, cash_flow_freq=4.0),
            dict(external_id="D-ALPHA-FX-2", deriv_type="FX",
                 notional=5_000_000, maturity_years=2.0,
                 underlying_price=1.092, strike=1.09, cash_flow_freq=2.0),
        ])

        # Beta Capital — mixed IRS + CDS
        pb1 = await _upsert_portfolio(db, external_id="PORT-BETA-MIXED",
            counterparty_id=beta.id, collateral=40_000.0, net_value=0.0)
        await _add_deriv_if_empty(db, pb1.id, [
            dict(external_id="D-BETA-IRS-1", deriv_type="IRS",
                 notional=20_000_000, maturity_years=5.0,
                 underlying_price=0.055, strike=0.042, cash_flow_freq=2.0),
            dict(external_id="D-BETA-CDS-1", deriv_type="CDS",
                 notional=8_000_000, maturity_years=5.0,
                 underlying_price=0.032, strike=0.020, cash_flow_freq=4.0),
        ])

        # Gamma Hedge Fund — equity derivatives
        # underlying_price is the normalised spot (1.0 = current level); strike is
        # a fraction of spot so V = notional × (S_t − strike × df) stays bounded.
        pg1 = await _upsert_portfolio(db, external_id="PORT-GAMMA-EQ",
            counterparty_id=gamma.id, collateral=15_000.0, net_value=0.0)
        await _add_deriv_if_empty(db, pg1.id, [
            dict(external_id="D-GAMMA-EQ-1", deriv_type="EQUITY",
                 notional=5_000_000, maturity_years=1.0,
                 underlying_price=1.0, strike=0.98, cash_flow_freq=1.0),
            dict(external_id="D-GAMMA-EQ-2", deriv_type="EQUITY",
                 notional=3_000_000, maturity_years=2.0,
                 underlying_price=1.0, strike=0.97, cash_flow_freq=1.0),
            dict(external_id="D-GAMMA-CDS-1", deriv_type="CDS",
                 notional=2_000_000, maturity_years=3.0,
                 underlying_price=0.045, strike=0.030, cash_flow_freq=4.0),
        ])

        # Delta Energy — commodity derivatives (normalised spot = 1.0)
        pd1 = await _upsert_portfolio(db, external_id="PORT-DELTA-COMM",
            counterparty_id=delta.id, collateral=60_000.0, net_value=0.0)
        await _add_deriv_if_empty(db, pd1.id, [
            dict(external_id="D-DELTA-COMM-1", deriv_type="COMMODITY",
                 notional=8_000_000, maturity_years=1.0,
                 underlying_price=1.0, strike=0.97, cash_flow_freq=4.0),
            dict(external_id="D-DELTA-COMM-2", deriv_type="COMMODITY",
                 notional=6_000_000, maturity_years=2.0,
                 underlying_price=1.0, strike=0.95, cash_flow_freq=2.0),
            dict(external_id="D-DELTA-IRS-1", deriv_type="IRS",
                 notional=12_000_000, maturity_years=4.0,
                 underlying_price=0.051, strike=0.043, cash_flow_freq=2.0),
        ])

        # Epsilon Insurance — long-dated IRS
        pe1 = await _upsert_portfolio(db, external_id="PORT-EPSILON-LDR",
            counterparty_id=epsilon.id, collateral=200_000.0, net_value=0.0)
        await _add_deriv_if_empty(db, pe1.id, [
            dict(external_id="D-EPSILON-IRS-1", deriv_type="IRS",
                 notional=50_000_000, maturity_years=10.0,
                 underlying_price=0.055, strike=0.040, cash_flow_freq=2.0),
            dict(external_id="D-EPSILON-IRS-2", deriv_type="IRS",
                 notional=30_000_000, maturity_years=7.0,
                 underlying_price=0.052, strike=0.038, cash_flow_freq=4.0),
        ])

        # Zeta Corp — high-yield with CDS
        pz1 = await _upsert_portfolio(db, external_id="PORT-ZETA-HY",
            counterparty_id=zeta.id, collateral=5_000.0, net_value=0.0)
        await _add_deriv_if_empty(db, pz1.id, [
            dict(external_id="D-ZETA-IRS-1", deriv_type="IRS",
                 notional=3_000_000, maturity_years=3.0,
                 underlying_price=0.065, strike=0.050, cash_flow_freq=4.0),
            dict(external_id="D-ZETA-CDS-1", deriv_type="CDS",
                 notional=2_000_000, maturity_years=5.0,
                 underlying_price=0.085, strike=0.060, cash_flow_freq=4.0),
        ])

        await db.commit()

        # ── Simulations ───────────────────────────────────────────────────────
        print("\n[Simulations]")

        def _cp_req(cp, coll=None):
            return CounterpartyRequest(
                id=cp.id, name=cp.name,
                hazard_rate=cp.hazard_rate, recovery_rate=cp.recovery_rate,
                credit_rating=2,
                collateral=coll if coll is not None else cp.collateral,
                margin_threshold=cp.margin_threshold, mpor_days=cp.mpor_days,
            )

        # ── Alpha Bank — 3 runs over time, low risk ───────────────────────────
        print("  Alpha Bank:")
        for seed, sigma, label in [(42, 0.18, "Alpha Q1"), (77, 0.20, "Alpha Q2"), (123, 0.22, "Alpha Q3")]:
            req = SimulationRequest(
                sim_params=SimParamsRequest(
                    num_paths=8000, num_timesteps=12, num_assets=1,
                    mu=0.02, sigma=sigma, rho_wwr=0.0, recovery_rate=0.42,
                    horizon_years=1.0, mode=SimMode.STANDARD, grid_type=GridType.MONTHLY,
                ),
                counterparty=_cp_req(alpha),
                portfolio=PortfolioRequest(
                    id=pa1.id, counterparty_id=alpha.id,
                    derivatives=[
                        DerivativeSpecRequest(id="DERIV-ALPHA-IRS-1", type=DerivativeType.IRS,
                            notional=25_000_000, maturity_years=7.0,
                            underlying_price=0.052, strike=0.040, cash_flow_freq=2.0),
                        DerivativeSpecRequest(id="DERIV-ALPHA-IRS-2", type=DerivativeType.IRS,
                            notional=15_000_000, maturity_years=3.0,
                            underlying_price=0.048, strike=0.038, cash_flow_freq=4.0),
                    ],
                    collateral=alpha.collateral, net_value=0.0,
                ),
                enable_wwr=False, enable_jump_diffusion=False,
                enable_collateral=True, deterministic_quantile=True, rng_seed=seed,
            )
            await _run_sim(db, req, label, admin.id)
            await db.commit()

        # ── Beta Capital — 2 runs, moderate risk, one margin call ─────────────
        print("  Beta Capital:")
        for seed, sigma, label in [(99, 0.28, "Beta Q1"), (200, 0.32, "Beta Q2")]:
            req = SimulationRequest(
                sim_params=SimParamsRequest(
                    num_paths=8000, num_timesteps=12, num_assets=1,
                    mu=0.025, sigma=sigma, rho_wwr=0.3, recovery_rate=0.38,
                    horizon_years=1.0, mode=SimMode.STANDARD, grid_type=GridType.MONTHLY,
                ),
                counterparty=_cp_req(beta),
                portfolio=PortfolioRequest(
                    id=pb1.id, counterparty_id=beta.id,
                    derivatives=[
                        DerivativeSpecRequest(id="DERIV-BETA-IRS-1", type=DerivativeType.IRS,
                            notional=20_000_000, maturity_years=5.0,
                            underlying_price=0.055, strike=0.042, cash_flow_freq=2.0),
                        DerivativeSpecRequest(id="DERIV-BETA-CDS-1", type=DerivativeType.CDS,
                            notional=8_000_000, maturity_years=5.0,
                            underlying_price=0.032, strike=0.020, cash_flow_freq=4.0),
                    ],
                    collateral=beta.collateral, net_value=0.0,
                ),
                enable_wwr=True, enable_jump_diffusion=False,
                enable_collateral=True, deterministic_quantile=True, rng_seed=seed,
            )
            await _run_sim(db, req, label, risk_mgr.id)
            await db.commit()

        # ── Gamma Hedge Fund — stressed simulation, high risk ─────────────────
        print("  Gamma Hedge Fund:")
        req = SimulationRequest(
            sim_params=SimParamsRequest(
                num_paths=8000, num_timesteps=12, num_assets=1,
                mu=0.03, sigma=0.45, rho_wwr=0.5, recovery_rate=0.30,
                horizon_years=1.0, mode=SimMode.STANDARD, grid_type=GridType.MONTHLY,
            ),
            counterparty=_cp_req(gamma),
            portfolio=PortfolioRequest(
                id=pg1.id, counterparty_id=gamma.id,
                derivatives=[
                    DerivativeSpecRequest(id="DERIV-GAMMA-EQ-1", type=DerivativeType.EQUITY,
                        notional=5_000_000, maturity_years=1.0,
                        underlying_price=1.0, strike=0.98, cash_flow_freq=1.0),
                    DerivativeSpecRequest(id="DERIV-GAMMA-EQ-2", type=DerivativeType.EQUITY,
                        notional=3_000_000, maturity_years=1.0,
                        underlying_price=1.0, strike=0.97, cash_flow_freq=1.0),
                ],
                collateral=gamma.collateral, net_value=0.0,
            ),
            stress=StressParams(
                vol_shock=0.15, equity_shock=0.20,
                interest_rate_shock=0.01, hazard_rate_shock=0.03,
                jump_amplitude=0.25,
            ),
            enable_wwr=True, enable_jump_diffusion=True,
            enable_collateral=True, deterministic_quantile=True, rng_seed=777,
        )
        await _run_sim(db, req, "Gamma stress (WWR+JD)", risk_mgr.id, is_stressed=True)
        await db.commit()

        # ── Delta Energy — commodity portfolio ────────────────────────────────
        print("  Delta Energy:")
        for seed, sigma, label in [(55, 0.25, "Delta Q1"), (66, 0.28, "Delta Q2")]:
            req = SimulationRequest(
                sim_params=SimParamsRequest(
                    num_paths=8000, num_timesteps=12, num_assets=1,
                    mu=0.02, sigma=sigma, rho_wwr=0.0, recovery_rate=0.40,
                    horizon_years=1.0, mode=SimMode.STANDARD, grid_type=GridType.MONTHLY,
                ),
                counterparty=_cp_req(delta),
                portfolio=PortfolioRequest(
                    id=pd1.id, counterparty_id=delta.id,
                    derivatives=[
                        DerivativeSpecRequest(id="DERIV-DELTA-COMM-1", type=DerivativeType.COMMODITY,
                            notional=8_000_000, maturity_years=1.0,
                            underlying_price=1.0, strike=0.97, cash_flow_freq=4.0),
                        DerivativeSpecRequest(id="DERIV-DELTA-IRS-1", type=DerivativeType.IRS,
                            notional=12_000_000, maturity_years=4.0,
                            underlying_price=0.051, strike=0.043, cash_flow_freq=2.0),
                    ],
                    collateral=delta.collateral, net_value=0.0,
                ),
                enable_wwr=False, enable_jump_diffusion=False,
                enable_collateral=True, deterministic_quantile=True, rng_seed=seed,
            )
            await _run_sim(db, req, label, risk_mgr.id)
            await db.commit()

        # ── Epsilon Insurance — large notional, low risk ───────────────────────
        print("  Epsilon Insurance:")
        req = SimulationRequest(
            sim_params=SimParamsRequest(
                num_paths=8000, num_timesteps=20, num_assets=1,
                mu=0.015, sigma=0.15, rho_wwr=0.0, recovery_rate=0.45,
                horizon_years=2.0, mode=SimMode.STANDARD, grid_type=GridType.MONTHLY,
            ),
            counterparty=_cp_req(epsilon),
            portfolio=PortfolioRequest(
                id=pe1.id, counterparty_id=epsilon.id,
                derivatives=[
                    DerivativeSpecRequest(id="DERIV-EPSILON-IRS-1", type=DerivativeType.IRS,
                        notional=50_000_000, maturity_years=10.0,
                        underlying_price=0.055, strike=0.040, cash_flow_freq=2.0),
                    DerivativeSpecRequest(id="DERIV-EPSILON-IRS-2", type=DerivativeType.IRS,
                        notional=30_000_000, maturity_years=7.0,
                        underlying_price=0.052, strike=0.038, cash_flow_freq=4.0),
                ],
                collateral=epsilon.collateral, net_value=0.0,
            ),
            enable_wwr=False, enable_jump_diffusion=False,
            enable_collateral=True, deterministic_quantile=True, rng_seed=11,
        )
        await _run_sim(db, req, "Epsilon long-dated IRS", admin.id)
        await db.commit()

        # ── Zeta Corp — high yield, definite margin call ───────────────────────
        print("  Zeta Corp (high risk):")
        req = SimulationRequest(
            sim_params=SimParamsRequest(
                num_paths=8000, num_timesteps=12, num_assets=1,
                mu=0.04, sigma=0.55, rho_wwr=0.6, recovery_rate=0.25,
                horizon_years=1.0, mode=SimMode.STANDARD, grid_type=GridType.MONTHLY,
            ),
            counterparty=_cp_req(zeta),
            portfolio=PortfolioRequest(
                id=pz1.id, counterparty_id=zeta.id,
                derivatives=[
                    DerivativeSpecRequest(id="DERIV-ZETA-IRS-1", type=DerivativeType.IRS,
                        notional=3_000_000, maturity_years=3.0,
                        underlying_price=0.065, strike=0.050, cash_flow_freq=4.0),
                    DerivativeSpecRequest(id="DERIV-ZETA-CDS-1", type=DerivativeType.CDS,
                        notional=2_000_000, maturity_years=5.0,
                        underlying_price=0.085, strike=0.060, cash_flow_freq=4.0),
                ],
                collateral=zeta.collateral, net_value=0.0,
            ),
            enable_wwr=True, enable_jump_diffusion=False,
            enable_collateral=True, deterministic_quantile=True, rng_seed=666,
        )
        await _run_sim(db, req, "Zeta high-yield", risk_mgr.id)
        await db.commit()

        # ── Set some margin calls to non-PENDING state ────────────────────────
        print("\n[Margin Call States]")
        r = await db.execute(select(MarginCall).order_by(MarginCall.issued_at))
        all_mcs = r.scalars().all()
        if len(all_mcs) >= 2:
            all_mcs[0].status         = MarginCallStatus.ACKNOWLEDGED
            all_mcs[0].acknowledged_at = datetime.now(timezone.utc) - timedelta(hours=2)
            _p(f"acknowledged: {all_mcs[0].id[:8]}…")
        if len(all_mcs) >= 3:
            all_mcs[1].status    = MarginCallStatus.SETTLED
            all_mcs[1].acknowledged_at = datetime.now(timezone.utc) - timedelta(hours=4)
            all_mcs[1].settled_at = datetime.now(timezone.utc) - timedelta(hours=1)
            _p(f"settled:      {all_mcs[1].id[:8]}…")
        await db.commit()

        # ── Audit log ─────────────────────────────────────────────────────────
        print("\n[Audit Log]")
        for action, resource_type, detail in [
            ("create_counterparty", "counterparty",  {"name": "Alpha Bank",        "external_id": "CP-ALPHA"}),
            ("create_counterparty", "counterparty",  {"name": "Beta Capital",       "external_id": "CP-BETA"}),
            ("create_counterparty", "counterparty",  {"name": "Gamma Hedge Fund",   "external_id": "CP-GAMMA"}),
            ("create_counterparty", "counterparty",  {"name": "Delta Energy",       "external_id": "CP-DELTA"}),
            ("create_counterparty", "counterparty",  {"name": "Epsilon Insurance",  "external_id": "CP-EPSILON"}),
            ("create_counterparty", "counterparty",  {"name": "Zeta Corp",          "external_id": "CP-ZETA"}),
            ("update_counterparty", "counterparty",  {"name": "Alpha Bank",         "field": "collateral", "value": 200000}),
            ("acknowledge_margin_call", "margin_call", {"reason": "Reviewing exposure"}),
            ("settle_margin_call",      "margin_call", {"amount": 45000}),
            ("login",                  "user",        {"username": "risk"}),
            ("login",                  "user",        {"username": "auditor"}),
        ]:
            db.add(AuditLog(
                user_id       = risk_mgr.id,
                action        = action,
                resource_type = resource_type,
                ip_address    = "10.0.0.1",
                detail        = detail,
            ))
        await db.commit()
        _p("added audit log entries")

        # ── Price history (backtest data) ─────────────────────────────────────
        print("\n[Price History]")
        await _seed_price_history(db)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 56)
    print("Seed complete.\n")
    print("Login credentials:")
    print("  admin    / admin123     [ADMIN]")
    print("  risk     / risk123      [RISK_MANAGER]")
    print("  auditor  / auditor123   [AUDITOR]")
    print()
    print("Counterparties seeded:")
    print("  Alpha Bank (AA)       — 2 portfolios, 5 derivatives, 3 sim runs")
    print("  Beta Capital (BBB)    — 1 portfolio,  2 derivatives, 2 sim runs")
    print("  Gamma Hedge Fund (BB) — 1 portfolio,  3 derivatives, 1 stressed run")
    print("  Delta Energy (A)      — 1 portfolio,  3 derivatives, 2 sim runs")
    print("  Epsilon Insurance (AA)— 1 portfolio,  2 derivatives, 1 sim run")
    print("  Zeta Corp (B)         — 1 portfolio,  2 derivatives, 1 sim run (high-risk)")
    print()
    print("Open http://localhost:5173  (or :8000 for prod build)")


if __name__ == "__main__":
    asyncio.run(main())
