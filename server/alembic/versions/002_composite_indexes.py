"""002 - Add composite indexes for common query patterns.

Indexes added:
  - margin_calls(counterparty_id, issued_at DESC)  — queries by counterparty
  - audit_log(user_id, time DESC)                  — admin panel user filter
  - simulation_runs(triggered_by, started_at DESC) — history queries by user
  - simulation_runs(counterparty_id, started_at DESC) — history queries by cp

Revision ID: 002
Revises: 001
Create Date: 2026-03-31
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_index(
        "ix_margin_calls_cp_issued",
        "margin_calls",
        ["counterparty_id", sa.text("issued_at DESC")],
    )
    op.create_index(
        "ix_audit_log_user_time",
        "audit_log",
        ["user_id", sa.text("time DESC")],
    )
    op.create_index(
        "ix_simulation_runs_triggered_by",
        "simulation_runs",
        ["triggered_by", sa.text("started_at DESC")],
    )
    op.create_index(
        "ix_simulation_runs_cp_started",
        "simulation_runs",
        ["counterparty_id", sa.text("started_at DESC")],
    )


def downgrade() -> None:
    op.drop_index("ix_simulation_runs_cp_started", table_name="simulation_runs")
    op.drop_index("ix_simulation_runs_triggered_by", table_name="simulation_runs")
    op.drop_index("ix_audit_log_user_time", table_name="audit_log")
    op.drop_index("ix_margin_calls_cp_issued", table_name="margin_calls")
