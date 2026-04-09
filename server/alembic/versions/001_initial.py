"""001 - Initial schema: all tables + TimescaleDB hypertables.

Revision ID: 001
Revises:
Create Date: 2026-03-30
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from alembic import op

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Ensure TimescaleDB extension is present before any hypertable creation.
    # The Docker image enables it by default; this is a no-op there but keeps
    # the migration self-contained for other setups.
    op.execute("CREATE EXTENSION IF NOT EXISTS timescaledb")

    # ── Regular tables ────────────────────────────────────────────────────────

    op.create_table(
        "users",
        sa.Column("id",          sa.String(),  server_default=sa.text("gen_random_uuid()::text"), nullable=False),
        sa.Column("username",    sa.String(64),  nullable=False),
        sa.Column("email",       sa.String(256), nullable=False),
        sa.Column("hashed_pw",   sa.String(256), nullable=False),
        sa.Column("role",        sa.String(32),  nullable=False, server_default="AUDITOR"),
        sa.Column("is_active",   sa.Boolean(),   nullable=False, server_default=sa.text("true")),
        sa.Column("created_at",  sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("last_login",  sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("username"),
        sa.UniqueConstraint("email"),
    )

    op.create_table(
        "counterparties",
        sa.Column("id",               sa.String(),  server_default=sa.text("gen_random_uuid()::text"), nullable=False),
        sa.Column("external_id",      sa.String(64),  nullable=False),
        sa.Column("name",             sa.String(256), nullable=False),
        sa.Column("credit_rating",    sa.String(8),   nullable=False, server_default="BBB"),
        sa.Column("hazard_rate",      sa.Float(),     nullable=False, server_default="0.02"),
        sa.Column("recovery_rate",    sa.Float(),     nullable=False, server_default="0.40"),
        sa.Column("collateral",       sa.Float(),     nullable=False, server_default="0.0"),
        sa.Column("margin_threshold", sa.Float(),     nullable=False, server_default="0.0"),
        sa.Column("mpor_days",        sa.Integer(),   nullable=False, server_default="10"),
        sa.Column("created_by",       sa.String(),    sa.ForeignKey("users.id"), nullable=True),
        sa.Column("created_at",       sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at",       sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("external_id"),
    )

    op.create_table(
        "portfolios",
        sa.Column("id",              sa.String(),  server_default=sa.text("gen_random_uuid()::text"), nullable=False),
        sa.Column("external_id",     sa.String(64),  nullable=False),
        sa.Column("counterparty_id", sa.String(),    sa.ForeignKey("counterparties.id"), nullable=False),
        sa.Column("collateral",      sa.Float(),     nullable=False, server_default="0.0"),
        sa.Column("net_value",       sa.Float(),     nullable=False, server_default="0.0"),
        sa.Column("auto_run",        sa.Boolean(),   nullable=False, server_default=sa.text("false")),
        sa.Column("created_at",      sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at",      sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("external_id"),
    )

    op.create_table(
        "derivatives",
        sa.Column("id",               sa.String(),  server_default=sa.text("gen_random_uuid()::text"), nullable=False),
        sa.Column("external_id",      sa.String(64),  nullable=False),
        sa.Column("portfolio_id",     sa.String(),    sa.ForeignKey("portfolios.id"), nullable=False),
        sa.Column("deriv_type",       sa.String(32),  nullable=False, server_default="IRS"),
        sa.Column("notional",         sa.Float(),     nullable=False, server_default="1000000.0"),
        sa.Column("maturity_years",   sa.Float(),     nullable=False, server_default="5.0"),
        sa.Column("underlying_price", sa.Float(),     nullable=False, server_default="0.05"),
        sa.Column("strike",           sa.Float(),     nullable=False, server_default="0.05"),
        sa.Column("cash_flow_freq",   sa.Float(),     nullable=False, server_default="2.0"),
        sa.Column("created_at",       sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "simulation_runs",
        sa.Column("id",              sa.String(),  server_default=sa.text("gen_random_uuid()::text"), nullable=False),
        sa.Column("portfolio_id",    sa.String(),  sa.ForeignKey("portfolios.id"), nullable=True),
        sa.Column("counterparty_id", sa.String(),  sa.ForeignKey("counterparties.id"), nullable=True),
        sa.Column("triggered_by",    sa.String(),  sa.ForeignKey("users.id"), nullable=True),
        sa.Column("trigger_type",    sa.String(32),  nullable=False, server_default="MANUAL"),
        sa.Column("sim_params_json", postgresql.JSONB(), nullable=True),
        sa.Column("stress_json",     postgresql.JSONB(), nullable=True),
        sa.Column("status",          sa.String(16),  nullable=False, server_default="RUNNING"),
        sa.Column("error_msg",       sa.Text(),      nullable=True),
        sa.Column("started_at",      sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("completed_at",    sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "margin_calls",
        sa.Column("id",                sa.String(),  server_default=sa.text("gen_random_uuid()::text"), nullable=False),
        sa.Column("counterparty_id",   sa.String(),  sa.ForeignKey("counterparties.id"), nullable=False),
        sa.Column("simulation_run_id", sa.String(),  sa.ForeignKey("simulation_runs.id"), nullable=True),
        sa.Column("amount",            sa.Float(),   nullable=False, server_default="0.0"),
        sa.Column("excess_exposure",   sa.Float(),   nullable=False, server_default="0.0"),
        sa.Column("status",            sa.String(16), nullable=False, server_default="PENDING"),
        sa.Column("reason",            sa.Text(),    nullable=False, server_default=""),
        sa.Column("issued_at",         sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("acknowledged_at",   sa.DateTime(timezone=True), nullable=True),
        sa.Column("settled_at",        sa.DateTime(timezone=True), nullable=True),
        sa.Column("issued_by",         sa.String(),  sa.ForeignKey("users.id"), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "market_params",
        sa.Column("id",         sa.String(),  server_default=sa.text("gen_random_uuid()::text"), nullable=False),
        sa.Column("symbol",     sa.String(32),  nullable=False),
        sa.Column("param_type", sa.String(16),  nullable=False),
        sa.Column("value",      sa.Float(),     nullable=False),
        sa.Column("source",     sa.String(64),  nullable=False, server_default=""),
        sa.Column("fetched_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("symbol", "param_type"),
    )

    # ── TimescaleDB hypertables ───────────────────────────────────────────────

    op.create_table(
        "risk_metrics",
        sa.Column("id",                sa.String(),  server_default=sa.text("gen_random_uuid()::text"), nullable=False),
        sa.Column("time",              sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("simulation_run_id", sa.String(),  sa.ForeignKey("simulation_runs.id"), nullable=True),
        sa.Column("counterparty_id",   sa.String(),  sa.ForeignKey("counterparties.id"), nullable=True),
        sa.Column("cva",               sa.Float(),   nullable=False, server_default="0.0"),
        sa.Column("wwr_cva",           sa.Float(),   nullable=False, server_default="0.0"),
        sa.Column("epe_profile",       sa.Text(),    nullable=True),
        sa.Column("pfe_profile",       sa.Text(),    nullable=True),
        sa.Column("time_grid_years",   sa.Text(),    nullable=True),
        sa.Column("margin_required",   sa.Float(),   nullable=False, server_default="0.0"),
        sa.Column("compute_time_us",   sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("is_stressed",       sa.Boolean(),   nullable=False, server_default=sa.text("false")),
        sa.PrimaryKeyConstraint("id", "time"),
    )
    op.execute(
        "SELECT create_hypertable('risk_metrics', 'time', if_not_exists => TRUE)"
    )
    op.create_index("ix_risk_metrics_cp_time", "risk_metrics", ["counterparty_id", sa.text("time DESC")])

    op.create_table(
        "audit_log",
        sa.Column("id",            sa.String(),  server_default=sa.text("gen_random_uuid()::text"), nullable=False),
        sa.Column("time",          sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("user_id",       sa.String(),    nullable=True),
        sa.Column("action",        sa.String(128), nullable=False),
        sa.Column("resource_type", sa.String(64),  nullable=False, server_default=""),
        sa.Column("resource_id",   sa.String(),    nullable=True),
        sa.Column("detail",        postgresql.JSONB(), nullable=True),
        sa.Column("ip_address",    sa.String(64),  nullable=True),
        sa.PrimaryKeyConstraint("id", "time"),
    )
    op.execute(
        "SELECT create_hypertable('audit_log', 'time', if_not_exists => TRUE)"
    )

    op.create_table(
        "price_history",
        sa.Column("id",     sa.String(),  server_default=sa.text("gen_random_uuid()::text"), nullable=False),
        sa.Column("ts",     sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("symbol", sa.String(32), nullable=False),
        sa.Column("price",  sa.Float(),    nullable=False),
        sa.Column("source", sa.String(64), nullable=False, server_default=""),
        sa.PrimaryKeyConstraint("id", "ts"),
    )
    op.execute(
        "SELECT create_hypertable('price_history', 'ts', if_not_exists => TRUE)"
    )
    op.create_index("ix_price_history_symbol_ts", "price_history", ["symbol", sa.text("ts DESC")])


def downgrade() -> None:
    op.drop_table("price_history")
    op.drop_table("audit_log")
    op.drop_table("risk_metrics")
    op.drop_table("market_params")
    op.drop_table("margin_calls")
    op.drop_table("simulation_runs")
    op.drop_table("derivatives")
    op.drop_table("portfolios")
    op.drop_table("counterparties")
    op.drop_table("users")
