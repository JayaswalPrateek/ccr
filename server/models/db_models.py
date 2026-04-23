"""SQLAlchemy 2.0 ORM models for all CCR database tables."""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


# ── Base ──────────────────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


# ── Enums ─────────────────────────────────────────────────────────────────────

class UserRole(str, enum.Enum):
    ADMIN        = "ADMIN"
    RISK_MANAGER = "RISK_MANAGER"
    AUDITOR      = "AUDITOR"


class SimStatus(str, enum.Enum):
    RUNNING   = "RUNNING"
    DONE      = "DONE"
    FAILED    = "FAILED"


class MarginCallStatus(str, enum.Enum):
    PENDING      = "PENDING"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    SETTLED      = "SETTLED"
    DISPUTED     = "DISPUTED"


class TriggerType(str, enum.Enum):
    MANUAL    = "MANUAL"
    SCHEDULED = "SCHEDULED"
    AUTO_RERUN = "AUTO_RERUN"


class ParamType(str, enum.Enum):
    SPOT   = "SPOT"
    VOL    = "VOL"
    RATE   = "RATE"
    HAZARD = "HAZARD"


# ── Regular tables ─────────────────────────────────────────────────────────────

class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(
        String, primary_key=True, server_default=text("gen_random_uuid()::text")
    )
    username:    Mapped[str]      = mapped_column(String(64), unique=True, nullable=False)
    email:       Mapped[str]      = mapped_column(String(256), unique=True, nullable=False)
    hashed_pw:   Mapped[str]      = mapped_column(String(256), nullable=False)
    role:        Mapped[str]      = mapped_column(String(32), nullable=False, default=UserRole.AUDITOR)
    is_active:   Mapped[bool]     = mapped_column(Boolean, nullable=False, default=True)
    created_at:  Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    last_login:  Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)


class Counterparty(Base):
    __tablename__ = "counterparties"

    id: Mapped[str] = mapped_column(
        String, primary_key=True, server_default=text("gen_random_uuid()::text")
    )
    external_id:      Mapped[str]   = mapped_column(String(64), unique=True, nullable=False)
    name:             Mapped[str]   = mapped_column(String(256), nullable=False)
    credit_rating:    Mapped[str]   = mapped_column(String(8), nullable=False, default="BBB")
    hazard_rate:      Mapped[float] = mapped_column(Float, nullable=False, default=0.02)
    recovery_rate:    Mapped[float] = mapped_column(Float, nullable=False, default=0.40)
    collateral:       Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    margin_threshold: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    mpor_days:        Mapped[int]   = mapped_column(Integer, nullable=False, default=10)
    # Hazard rate term structure (optional — if set, overrides the flat hazard_rate for CVA)
    hz_1y:            Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    hz_3y:            Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    hz_5y:            Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    hz_10y:           Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    created_by:       Mapped[Optional[str]] = mapped_column(String, ForeignKey("users.id"), nullable=True)
    created_at:       Mapped[datetime]      = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at:       Mapped[datetime]      = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    portfolios:    Mapped[List["Portfolio"]]   = relationship("Portfolio", back_populates="counterparty", lazy="selectin")
    margin_calls:  Mapped[List["MarginCall"]]  = relationship("MarginCall", back_populates="counterparty")


class Portfolio(Base):
    __tablename__ = "portfolios"

    id: Mapped[str] = mapped_column(
        String, primary_key=True, server_default=text("gen_random_uuid()::text")
    )
    external_id:     Mapped[str]   = mapped_column(String(64), unique=True, nullable=False)
    counterparty_id: Mapped[str]   = mapped_column(String, ForeignKey("counterparties.id"), nullable=False)
    collateral:      Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    net_value:       Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    auto_run:        Mapped[bool]  = mapped_column(Boolean, nullable=False, default=False)
    created_at:      Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at:      Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    counterparty: Mapped["Counterparty"]     = relationship("Counterparty", back_populates="portfolios")
    derivatives:  Mapped[List["Derivative"]] = relationship("Derivative", back_populates="portfolio", lazy="selectin")


class Derivative(Base):
    __tablename__ = "derivatives"

    id: Mapped[str] = mapped_column(
        String, primary_key=True, server_default=text("gen_random_uuid()::text")
    )
    external_id:      Mapped[str]   = mapped_column(String(64), nullable=False)
    portfolio_id:     Mapped[str]   = mapped_column(String, ForeignKey("portfolios.id"), nullable=False)
    deriv_type:       Mapped[str]   = mapped_column(String(32), nullable=False, default="IRS")
    notional:         Mapped[float] = mapped_column(Float, nullable=False, default=1_000_000.0)
    maturity_years:   Mapped[float] = mapped_column(Float, nullable=False, default=5.0)
    underlying_price: Mapped[float] = mapped_column(Float, nullable=False, default=0.05)
    strike:           Mapped[float] = mapped_column(Float, nullable=False, default=0.05)
    cash_flow_freq:   Mapped[float] = mapped_column(Float, nullable=False, default=2.0)
    created_at:       Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    portfolio: Mapped["Portfolio"] = relationship("Portfolio", back_populates="derivatives")


class SimulationRun(Base):
    __tablename__ = "simulation_runs"

    id: Mapped[str] = mapped_column(
        String, primary_key=True, server_default=text("gen_random_uuid()::text")
    )
    portfolio_id:     Mapped[Optional[str]] = mapped_column(String, ForeignKey("portfolios.id"), nullable=True)
    counterparty_id:  Mapped[Optional[str]] = mapped_column(String, ForeignKey("counterparties.id"), nullable=True)
    triggered_by:     Mapped[Optional[str]] = mapped_column(String, ForeignKey("users.id"), nullable=True)
    trigger_type:     Mapped[str]           = mapped_column(String(32), nullable=False, default=TriggerType.MANUAL)
    sim_params_json:  Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB, nullable=True)
    stress_json:      Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB, nullable=True)
    status:           Mapped[str]           = mapped_column(String(16), nullable=False, default=SimStatus.RUNNING)
    error_msg:        Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    note:             Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    started_at:       Mapped[datetime]      = mapped_column(DateTime(timezone=True), server_default=func.now())
    completed_at:     Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)


class MarginCall(Base):
    __tablename__ = "margin_calls"

    id: Mapped[str] = mapped_column(
        String, primary_key=True, server_default=text("gen_random_uuid()::text")
    )
    counterparty_id:    Mapped[str]   = mapped_column(String, ForeignKey("counterparties.id"), nullable=False)
    simulation_run_id:  Mapped[Optional[str]] = mapped_column(String, ForeignKey("simulation_runs.id"), nullable=True)
    amount:             Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    excess_exposure:    Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    status:             Mapped[str]   = mapped_column(String(16), nullable=False, default=MarginCallStatus.PENDING)
    reason:             Mapped[str]   = mapped_column(Text, nullable=False, default="")
    issued_at:          Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    acknowledged_at:    Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    settled_at:         Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    issued_by:          Mapped[Optional[str]] = mapped_column(String, ForeignKey("users.id"), nullable=True)

    counterparty: Mapped["Counterparty"] = relationship("Counterparty", back_populates="margin_calls")


class MarketParam(Base):
    __tablename__ = "market_params"
    __table_args__ = (UniqueConstraint("symbol", "param_type"),)

    id: Mapped[str] = mapped_column(
        String, primary_key=True, server_default=text("gen_random_uuid()::text")
    )
    symbol:     Mapped[str]   = mapped_column(String(32), nullable=False)
    param_type: Mapped[str]   = mapped_column(String(16), nullable=False)  # SPOT/VOL/RATE/HAZARD
    value:      Mapped[float] = mapped_column(Float, nullable=False)
    source:     Mapped[str]   = mapped_column(String(64), nullable=False, default="")
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


# ── TimescaleDB hypertables (time-partitioned) ────────────────────────────────

class RiskMetric(Base):
    """One row per simulation run. Hypertable partitioned by time."""
    __tablename__ = "risk_metrics"

    id: Mapped[str] = mapped_column(
        String, primary_key=True, server_default=text("gen_random_uuid()::text")
    )
    # time is the second component of the composite PK required by TimescaleDB.
    time:              Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True, nullable=False, server_default=func.now())
    simulation_run_id: Mapped[Optional[str]] = mapped_column(String, ForeignKey("simulation_runs.id"), nullable=True)
    counterparty_id:   Mapped[Optional[str]] = mapped_column(String, ForeignKey("counterparties.id"), nullable=True)
    cva:               Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    wwr_cva:           Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    epe_profile:       Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON array
    pfe_profile:       Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON array
    time_grid_years:   Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON array
    margin_required:   Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    compute_time_us:   Mapped[int]   = mapped_column(BigInteger, nullable=False, default=0)
    is_stressed:       Mapped[bool]  = mapped_column(Boolean, nullable=False, default=False)


class AuditLog(Base):
    """Append-only audit trail. Hypertable partitioned by time."""
    __tablename__ = "audit_log"

    id: Mapped[str] = mapped_column(
        String, primary_key=True, server_default=text("gen_random_uuid()::text")
    )
    # time is the second component of the composite PK required by TimescaleDB.
    time:          Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True, nullable=False, server_default=func.now())
    user_id:       Mapped[Optional[str]] = mapped_column(String, nullable=True)
    action:        Mapped[str]           = mapped_column(String(128), nullable=False)
    resource_type: Mapped[str]           = mapped_column(String(64), nullable=False, default="")
    resource_id:   Mapped[Optional[str]] = mapped_column(String, nullable=True)
    detail:        Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB, nullable=True)
    ip_address:    Mapped[Optional[str]] = mapped_column(String(64), nullable=True)


class PriceHistory(Base):
    """Tick-level price history. Hypertable partitioned by ts."""
    __tablename__ = "price_history"

    id: Mapped[str] = mapped_column(
        String, primary_key=True, server_default=text("gen_random_uuid()::text")
    )
    # ts is the second component of the composite PK required by TimescaleDB.
    ts:     Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True, nullable=False, server_default=func.now())
    symbol: Mapped[str]      = mapped_column(String(32), nullable=False)
    price:  Mapped[float]    = mapped_column(Float, nullable=False)
    source: Mapped[str]      = mapped_column(String(64), nullable=False, default="")


# ── Simulation presets ────────────────────────────────────────────────────────

class SimPreset(Base):
    """Named simulation parameter presets, owned per user, optionally scoped to a counterparty."""
    __tablename__ = "sim_presets"

    id: Mapped[str] = mapped_column(
        String, primary_key=True, server_default=text("gen_random_uuid()::text")
    )
    name:             Mapped[str]                    = mapped_column(String(200), nullable=False)
    description:      Mapped[Optional[str]]          = mapped_column(Text, nullable=True)
    owner_id:         Mapped[Optional[str]]          = mapped_column(String, ForeignKey("users.id"),           nullable=True)
    counterparty_id:  Mapped[Optional[str]]          = mapped_column(String, ForeignKey("counterparties.id"), nullable=True)
    params_json:      Mapped[Dict[str, Any]]         = mapped_column(JSONB, nullable=False)
    stress_json:      Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB, nullable=True)
    is_shared:        Mapped[bool]                   = mapped_column(Boolean, nullable=False, default=False)
    use_count:        Mapped[int]                    = mapped_column(Integer, nullable=False, default=0)
    last_used_at:     Mapped[Optional[datetime]]     = mapped_column(DateTime(timezone=True), nullable=True)
    created_at:       Mapped[datetime]               = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at:       Mapped[datetime]               = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
