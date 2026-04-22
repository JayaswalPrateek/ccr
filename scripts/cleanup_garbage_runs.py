#!/usr/bin/env python3
"""
Remove zero-CVA null-counterparty simulation runs left by empty-body test calls.

Usage:
    uv run python scripts/cleanup_garbage_runs.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "server" / "bindings"))

_env = PROJECT_ROOT / ".env"
if _env.exists():
    for _line in _env.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())

os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://ccr:ccr@localhost:5432/ccr")
os.environ.setdefault("JWT_SECRET", "dev-secret-change-me")


async def main() -> None:
    from server.core.database import AsyncSessionLocal
    from sqlalchemy import text

    async with AsyncSessionLocal() as db:
        # Count before
        r = await db.execute(
            text("SELECT COUNT(*) FROM risk_metrics WHERE counterparty_id IS NULL AND cva = 0 AND margin_required = 0")
        )
        n = r.scalar() or 0
        print(f"Found {n} garbage risk_metric rows (null counterparty, zero CVA)")

        # Delete garbage risk_metrics
        await db.execute(
            text("DELETE FROM risk_metrics WHERE counterparty_id IS NULL AND cva = 0 AND margin_required = 0")
        )

        # Delete orphan simulation_runs (null counterparty, no remaining risk_metrics, status=DONE)
        r2 = await db.execute(
            text("""
                DELETE FROM simulation_runs
                WHERE counterparty_id IS NULL
                  AND status = 'DONE'
                  AND id NOT IN (
                      SELECT DISTINCT simulation_run_id
                      FROM risk_metrics
                      WHERE simulation_run_id IS NOT NULL
                  )
            """)
        )
        print(f"Deleted {r2.rowcount} orphan simulation_run rows")

        await db.commit()
        print("Cleanup complete.")


if __name__ == "__main__":
    asyncio.run(main())
