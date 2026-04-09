"""003 - Add sim_presets table for named scenario presets.

Revision ID: 003
Revises: 002
Create Date: 2026-04-01
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "003"
down_revision: Union[str, None] = "002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "sim_presets",
        sa.Column("id", sa.String(), server_default=sa.text("gen_random_uuid()::text"), nullable=False),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("owner_id", sa.String(), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column("counterparty_id", sa.String(), sa.ForeignKey("counterparties.id", ondelete="SET NULL"), nullable=True),
        sa.Column("params_json", postgresql.JSONB(), nullable=False),
        sa.Column("stress_json", postgresql.JSONB(), nullable=True),
        sa.Column("is_shared", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("use_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_presets_owner",        "sim_presets", ["owner_id"])
    op.create_index("ix_presets_counterparty", "sim_presets", ["counterparty_id"])
    op.create_index("ix_presets_shared",       "sim_presets", ["is_shared"])


def downgrade() -> None:
    op.drop_index("ix_presets_shared",       table_name="sim_presets")
    op.drop_index("ix_presets_counterparty", table_name="sim_presets")
    op.drop_index("ix_presets_owner",        table_name="sim_presets")
    op.drop_table("sim_presets")
