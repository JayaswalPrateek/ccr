"""005 - Add hazard rate term structure to counterparties.

Revision ID: 005
Revises: 004
Create Date: 2026-04-22
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "005"
down_revision: Union[str, None] = "004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("counterparties", sa.Column("hz_1y",  sa.Float(), nullable=True))
    op.add_column("counterparties", sa.Column("hz_3y",  sa.Float(), nullable=True))
    op.add_column("counterparties", sa.Column("hz_5y",  sa.Float(), nullable=True))
    op.add_column("counterparties", sa.Column("hz_10y", sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column("counterparties", "hz_10y")
    op.drop_column("counterparties", "hz_5y")
    op.drop_column("counterparties", "hz_3y")
    op.drop_column("counterparties", "hz_1y")
