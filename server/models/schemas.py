"""
Pydantic request/response schemas for the CCR API.
These mirror the C++ types.hpp structs; the glue layer in engine_client.py
maps between them and the pybind11 objects.
"""

from __future__ import annotations

from enum import IntEnum
from typing import List, Optional

from pydantic import BaseModel, Field


# ── Enumerations ────────────────────────────────────────────────────────────

class DerivativeType(IntEnum):
    IRS       = 0
    CDS       = 1
    FX        = 2
    EQUITY    = 3
    COMMODITY = 4

class CreditRating(IntEnum):
    AAA = 0
    AA  = 1
    A   = 2
    BBB = 3
    BB  = 4
    B   = 5
    CCC = 6
    D   = 7

class SimMode(IntEnum):
    REGULATORY  = 0
    STANDARD    = 1
    APPROX_FAST = 2

class GridType(IntEnum):
    MONTHLY     = 0
    WEEKLY      = 1
    DAILY       = 2
    PARSIMONIOUS = 3


# ── Simulation parameters ────────────────────────────────────────────────────

class SimParamsRequest(BaseModel):
    num_paths:     int   = Field(10_000, gt=0)
    num_timesteps: int   = Field(12,     gt=0)
    num_assets:    int   = Field(1,      gt=0)
    mu:            float = Field(0.02)
    sigma:         float = Field(0.20,   gt=0)
    rho_wwr:       float = Field(0.0,    ge=-1.0, le=1.0)
    recovery_rate: float = Field(0.40,   ge=0.0,  le=1.0)
    horizon_years: float = Field(1.0,    gt=0)
    mode:          SimMode  = SimMode.STANDARD
    grid_type:     GridType = GridType.MONTHLY


# ── Counterparty ─────────────────────────────────────────────────────────────

class CounterpartyRequest(BaseModel):
    id:               str         = "CP-001"
    name:             str         = "Counterparty"
    credit_rating:    CreditRating = CreditRating.BBB
    hazard_rate:      float       = Field(0.02, ge=0)
    recovery_rate:    float       = Field(0.40, ge=0, le=1)
    collateral:       float       = Field(0.0,  ge=0)
    margin_threshold: float       = Field(0.0,  ge=0)
    mpor_days:        int         = Field(10,   ge=0)


# ── Derivative specs ──────────────────────────────────────────────────────────

class DerivativeSpecRequest(BaseModel):
    id:               str           = "DERIV-001"
    type:             DerivativeType = DerivativeType.IRS
    notional:         float         = Field(1_000_000.0, gt=0)
    maturity_years:   float         = Field(5.0,         gt=0)
    underlying_price: float         = Field(0.05,        gt=0)  # initial rate/price
    strike:           float         = Field(0.05)               # fixed rate / strike
    cash_flow_freq:   float         = Field(2.0,         gt=0)  # payments/year


# ── Portfolio ─────────────────────────────────────────────────────────────────

class PortfolioRequest(BaseModel):
    id:             str                         = "PORT-001"
    counterparty_id: str                        = "CP-001"
    derivatives:    List[DerivativeSpecRequest] = Field(default_factory=list)
    collateral:     float                       = Field(0.0, ge=0)
    net_value:      float                       = 0.0


# ── Stress scenario ───────────────────────────────────────────────────────────

class StressScenarioRequest(BaseModel):
    vol_shock:           float = 0.0
    fx_shock:            float = 0.0
    equity_shock:        float = 0.0
    interest_rate_shock: float = 0.0
    credit_spread_shock: float = 0.0
    hazard_rate_shock:   float = 0.0
    jump_amplitude:      float = 0.0
    label:               str   = "stress"


# ── Top-level simulation request ──────────────────────────────────────────────

class SimulationRequest(BaseModel):
    sim_params:             SimParamsRequest   = Field(default_factory=SimParamsRequest)
    counterparty:           CounterpartyRequest = Field(default_factory=CounterpartyRequest)
    portfolio:              PortfolioRequest    = Field(default_factory=PortfolioRequest)
    stress:                 Optional[StressScenarioRequest] = None
    enable_wwr:             bool = False
    enable_jump_diffusion:  bool = False
    enable_collateral:      bool = False
    deterministic_quantile: bool = True
    log_overflow_warnings:  bool = False
    rng_seed:               int  = 42


# ── Result types ─────────────────────────────────────────────────────────────

class RiskMetricsResponse(BaseModel):
    cva:             float
    wwr_cva:         float
    margin_required: float
    pfe_profile:     List[float]
    epe_profile:     List[float]
    time_grid_years: List[float]
    compute_time_us: int
    overflow_detected: bool
    arch_used:       str
    paths_used:      int


class SimulationResponse(BaseModel):
    base:      RiskMetricsResponse
    stressed:  Optional[RiskMetricsResponse] = None
    success:   bool
    error_msg: str = ""


class MarginCallResponse(BaseModel):
    id:              str
    counterparty_id: str
    amount:          float
    excess_exposure: float
    status:          str
    reason:          str
