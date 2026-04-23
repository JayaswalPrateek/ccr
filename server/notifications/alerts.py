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


def _fmt(v: float) -> str:
    """Abbreviate a financial number to K/M/B."""
    if v >= 1e9:  return f"{v/1e9:.2f}B"
    if v >= 1e6:  return f"{v/1e6:.2f}M"
    if v >= 1e3:  return f"{v/1e3:.1f}K"
    return f"{v:,.2f}"


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
        from datetime import datetime, timezone

        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        body = f"""
<!DOCTYPE html>
<html>
<head><meta charset="utf-8"></head>
<body style="margin:0;padding:0;background:#0f1117;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif">
  <table width="100%" cellpadding="0" cellspacing="0" style="background:#0f1117;padding:32px 0">
    <tr><td align="center">
      <table width="560" cellpadding="0" cellspacing="0" style="background:#1c1f26;border-radius:8px;overflow:hidden;border:1px solid #2d3142">

        <!-- Header -->
        <tr><td style="background:#1c3557;padding:24px 32px">
          <div style="font-size:11px;letter-spacing:2px;color:#7aa3c8;text-transform:uppercase;margin-bottom:6px">CCR Engine</div>
          <div style="font-size:22px;font-weight:700;color:#ffffff">Margin Call Alert</div>
        </td></tr>

        <!-- Body -->
        <tr><td style="padding:28px 32px">
          <p style="margin:0 0 20px;font-size:14px;color:#94a3b8">
            A margin breach has been detected. Immediate acknowledgement is required.
          </p>

          <!-- Metric boxes -->
          <table width="100%" cellpadding="0" cellspacing="0" style="margin-bottom:24px">
            <tr>
              <td width="50%" style="padding-right:8px">
                <div style="background:#0f1117;border:1px solid #2d3142;border-radius:6px;padding:16px">
                  <div style="font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px">Margin Call Amount</div>
                  <div style="font-size:24px;font-weight:700;color:#f87171">{_fmt(amount)}</div>
                </div>
              </td>
              <td width="50%" style="padding-left:8px">
                <div style="background:#0f1117;border:1px solid #2d3142;border-radius:6px;padding:16px">
                  <div style="font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px">Excess Exposure</div>
                  <div style="font-size:24px;font-weight:700;color:#fb923c">{_fmt(excess_exposure)}</div>
                </div>
              </td>
            </tr>
          </table>

          <!-- Details -->
          <table width="100%" cellpadding="0" cellspacing="0" style="border-collapse:collapse;margin-bottom:24px">
            <tr>
              <td style="padding:10px 0;border-bottom:1px solid #2d3142;font-size:13px;color:#64748b;width:40%">Counterparty</td>
              <td style="padding:10px 0;border-bottom:1px solid #2d3142;font-size:13px;color:#e2e8f0;font-weight:600">{counterparty_name}</td>
            </tr>
            <tr>
              <td style="padding:10px 0;border-bottom:1px solid #2d3142;font-size:13px;color:#64748b">Status</td>
              <td style="padding:10px 0;border-bottom:1px solid #2d3142;font-size:13px"><span style="background:#78350f;color:#fde68a;padding:2px 10px;border-radius:12px;font-size:12px;font-weight:600">PENDING</span></td>
            </tr>
            <tr>
              <td style="padding:10px 0;font-size:13px;color:#64748b">Generated</td>
              <td style="padding:10px 0;font-size:13px;color:#e2e8f0">{now_str}</td>
            </tr>
          </table>

          <a href="{settings.app_url}/margin-calls"
             style="display:inline-block;background:#3b82f6;color:#ffffff;font-size:13px;font-weight:600;padding:12px 24px;border-radius:6px;text-decoration:none">
            Acknowledge in Dashboard →
          </a>
        </td></tr>

        <!-- Footer -->
        <tr><td style="padding:16px 32px;border-top:1px solid #2d3142;font-size:11px;color:#475569">
          CCR Engine — Counterparty Credit Risk Platform &nbsp;|&nbsp; This alert was generated automatically.
        </td></tr>

      </table>
    </td></tr>
  </table>
</body>
</html>
"""
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"[CCR] Margin Call — {counterparty_name} ({_fmt(amount)})"
        msg["From"]    = settings.smtp_from
        msg["To"]      = ", ".join(to)
        msg.attach(MIMEText(body, "html"))

        # Mailtrap (port 2525) does not use STARTTLS; standard SMTP (587) does.
        use_tls = settings.smtp_port == 587
        await aiosmtplib.send(
            msg,
            hostname  = settings.smtp_host,
            port      = settings.smtp_port,
            username  = settings.smtp_user or None,
            password  = settings.smtp_password or None,
            start_tls = use_tls,
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
