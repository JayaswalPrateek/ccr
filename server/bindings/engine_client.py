"""
Glue layer between Pydantic request/response models and the pybind11 C++ types.

build_engine_config() — converts a SimulationRequest into _ccr_engine.EngineConfig
result_to_response()  — converts a _ccr_engine.CcrResult into SimulationResponse
"""

from __future__ import annotations

from typing import Optional

import _ccr_engine as _ccr

from server.models.schemas import (
    SimulationRequest,
    SimulationResponse,
    RiskMetricsResponse,
    MarginCallResponse,
    DerivativeType as PyDerivType,
    CreditRating   as PyCreditRating,
    SimMode        as PySimMode,
    GridType       as PyGridType,
)


def build_engine_config(req: SimulationRequest) -> _ccr.EngineConfig:
    """Map a SimulationRequest (Pydantic) to a _ccr_engine.EngineConfig (C++)."""
    cfg = _ccr.EngineConfig()

    # SimParams
    sp = _ccr.SimParams()
    sp.num_paths     = req.sim_params.num_paths
    sp.num_timesteps = req.sim_params.num_timesteps
    sp.num_assets    = req.sim_params.num_assets
    sp.mu            = req.sim_params.mu
    sp.sigma         = req.sim_params.sigma
    sp.rho_wwr       = req.sim_params.rho_wwr
    sp.recovery_rate = req.sim_params.recovery_rate
    sp.horizon_years = req.sim_params.horizon_years
    sp.mode          = _ccr.SimMode(int(req.sim_params.mode))
    sp.grid_type     = _ccr.GridType(int(req.sim_params.grid_type))
    cfg.sim_params   = sp

    # CounterpartyConfig
    cp = _ccr.CounterpartyConfig()
    cp.id               = req.counterparty.id
    cp.name             = req.counterparty.name
    cp.credit_rating    = _ccr.CreditRating(int(req.counterparty.credit_rating))
    cp.hazard_rate      = req.counterparty.hazard_rate
    cp.recovery_rate    = req.counterparty.recovery_rate
    cp.collateral       = req.counterparty.collateral
    cp.margin_threshold = req.counterparty.margin_threshold
    cp.mpor_days        = req.counterparty.mpor_days
    cfg.counterparty    = cp

    # PortfolioConfig
    port = _ccr.PortfolioConfig()
    port.id             = req.portfolio.id
    port.counterparty_id = req.portfolio.counterparty_id
    port.collateral     = req.portfolio.collateral
    port.net_value      = req.portfolio.net_value

    derivs = []
    for d in req.portfolio.derivatives:
        ds = _ccr.DerivativeSpec()
        ds.id               = d.id
        ds.type             = _ccr.DerivativeType(int(d.type))
        ds.notional         = d.notional
        ds.maturity_years   = d.maturity_years
        ds.underlying_price = d.underlying_price
        ds.strike           = d.strike
        ds.cash_flow_freq   = d.cash_flow_freq
        derivs.append(ds)
    port.derivatives = derivs
    cfg.portfolio = port

    # Optional stress scenario
    if req.stress is not None:
        sc = _ccr.StressScenario()
        sc.vol_shock           = req.stress.vol_shock
        sc.fx_shock            = req.stress.fx_shock
        sc.equity_shock        = req.stress.equity_shock
        sc.interest_rate_shock = req.stress.interest_rate_shock
        sc.credit_spread_shock = req.stress.credit_spread_shock
        sc.hazard_rate_shock   = req.stress.hazard_rate_shock
        sc.jump_amplitude      = req.stress.jump_amplitude
        sc.label               = req.stress.label
        cfg.stress = sc

    cfg.enable_wwr             = req.enable_wwr
    cfg.enable_jump_diffusion  = req.enable_jump_diffusion
    cfg.enable_collateral      = req.enable_collateral
    cfg.deterministic_quantile = req.deterministic_quantile
    cfg.log_overflow_warnings  = req.log_overflow_warnings
    cfg.rng_seed               = req.rng_seed

    return cfg


def _metrics_to_response(m: _ccr.RiskMetrics) -> RiskMetricsResponse:
    return RiskMetricsResponse(
        cva              = m.cva,
        wwr_cva          = m.wwr_cva,
        margin_required  = m.margin_required,
        pfe_profile      = list(m.pfe_profile),
        epe_profile      = list(m.epe_profile),
        time_grid_years  = list(m.time_grid_years),
        compute_time_us  = int(m.compute_time_us.microseconds
                               + m.compute_time_us.seconds * 1_000_000),
        overflow_detected = m.overflow_detected,
        arch_used        = m.arch_used,
        paths_used       = m.paths_used,
    )


def result_to_response(result: _ccr.CcrResult) -> SimulationResponse:
    """Convert a C++ CcrResult into a Pydantic SimulationResponse."""
    if not result.success:
        # Return a minimal failure response; base metrics will be zeroed.
        return SimulationResponse(
            base      = RiskMetricsResponse(
                cva=0, wwr_cva=0, margin_required=0,
                pfe_profile=[], epe_profile=[], time_grid_years=[],
                compute_time_us=0, overflow_detected=False,
                arch_used="", paths_used=0,
            ),
            success   = False,
            error_msg = result.error_msg,
        )

    stressed: Optional[RiskMetricsResponse] = None
    if result.stressed is not None:
        try:
            stressed = _metrics_to_response(result.stressed)
        except Exception:
            pass  # stressed is optional; ignore conversion failures

    return SimulationResponse(
        base      = _metrics_to_response(result.base),
        stressed  = stressed,
        success   = True,
    )
