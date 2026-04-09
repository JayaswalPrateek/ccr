"""Audit log helper — inserts into the audit_log hypertable."""

from __future__ import annotations

from typing import Any, Dict, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from server.models.db_models import AuditLog


async def log_event(
    db:            AsyncSession,
    action:        str,
    *,
    user_id:       Optional[str]            = None,
    resource_type: str                       = "",
    resource_id:   Optional[str]            = None,
    detail:        Optional[Dict[str, Any]] = None,
    ip_address:    Optional[str]            = None,
) -> None:
    """Append one row to audit_log.  Call after the main operation succeeds."""
    entry = AuditLog(
        user_id       = user_id,
        action        = action,
        resource_type = resource_type,
        resource_id   = resource_id,
        detail        = detail,
        ip_address    = ip_address,
    )
    db.add(entry)
    # Caller is responsible for commit — we just add to the session.
