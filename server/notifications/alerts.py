"""
Margin call alerts via email (aiosmtplib).

Falls back to logging if SMTP is not configured — the server never crashes
because email is unavailable.
"""

from __future__ import annotations

import logging
from typing import List

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from server.core.config import settings
from server.models.db_models import MarginCall, User, UserRole

logger = logging.getLogger(__name__)


async def send_margin_call_email(
    to:                List[str],
    counterparty_name: str,
    amount:            float,
    excess_exposure:   float,
) -> None:
    """Send an HTML margin call alert to a list of addresses.

    Silently falls back to a log warning when SMTP is unconfigured.
    """
    if not settings.smtp_host:
        logger.warning(
            "Margin call alert (SMTP unconfigured): %s — amount=%.2f excess=%.2f",
            counterparty_name, amount, excess_exposure,
        )
        return

    try:
        import aiosmtplib
        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText

        body = f"""
        <html><body>
        <h2>⚠ Margin Call Alert</h2>
        <p><strong>Counterparty:</strong> {counterparty_name}</p>
        <p><strong>Margin call amount:</strong> {amount:,.2f}</p>
        <p><strong>Excess exposure:</strong> {excess_exposure:,.2f}</p>
        <p>Please acknowledge this margin call in the CCR dashboard.</p>
        </body></html>
        """
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"[CCR] Margin Call — {counterparty_name}"
        msg["From"]    = settings.smtp_from
        msg["To"]      = ", ".join(to)
        msg.attach(MIMEText(body, "html"))

        await aiosmtplib.send(
            msg,
            hostname = settings.smtp_host,
            port     = settings.smtp_port,
            username = settings.smtp_user or None,
            password = settings.smtp_password or None,
            start_tls = True,
        )
        logger.info("Margin call email sent to %s for %s", to, counterparty_name)
    except Exception as exc:
        logger.error("Failed to send margin call email: %s", exc)


async def check_and_alert_margin_calls(
    db:              AsyncSession,
    counterparty_id: str,
    margin_required: float,
    collateral:      float,
    simulation_run_id: str | None = None,
) -> MarginCall | None:
    """Create a MarginCall record and send alerts when exposure exceeds collateral.

    Returns the created MarginCall or None if no breach.
    """
    if margin_required <= collateral:
        return None

    excess = margin_required - collateral

    mc = MarginCall(
        counterparty_id   = counterparty_id,
        simulation_run_id = simulation_run_id,
        amount            = margin_required,
        excess_exposure   = excess,
        reason            = (
            f"Margin required ({margin_required:.2f}) exceeds "
            f"posted collateral ({collateral:.2f})"
        ),
    )
    db.add(mc)
    # Caller commits after this returns.

    # Notify all active RISK_MANAGER users.
    result = await db.execute(
        select(User).where(
            User.role == UserRole.RISK_MANAGER,
            User.is_active == True,  # noqa: E712
        )
    )
    emails = [u.email for u in result.scalars().all() if u.email]

    if emails:
        from server.models.db_models import Counterparty
        cp_result = await db.execute(
            select(Counterparty).where(Counterparty.id == counterparty_id)
        )
        cp = cp_result.scalar_one_or_none()
        cp_name = cp.name if cp else counterparty_id

        # Fire-and-forget: don't block the response on email delivery.
        import asyncio
        asyncio.create_task(
            send_margin_call_email(emails, cp_name, margin_required, excess)
        )

    return mc
