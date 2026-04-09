"""
PDF and CSV export utilities for CCR simulation results.

PDF uses ReportLab's SimpleDocTemplate.  CSV uses the stdlib csv module.
Both functions return bytes ready to stream as a FastAPI response.
"""

from __future__ import annotations

import csv
import io
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


# ── CSV ───────────────────────────────────────────────────────────────────────

def export_simulation_csv(
    pfe_profile:     List[float],
    epe_profile:     List[float],
    time_grid_years: List[float],
) -> bytes:
    """Return UTF-8 CSV bytes: columns time_year, pfe, epe."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["time_year", "pfe", "epe"])
    for t, pfe, epe in zip(time_grid_years, pfe_profile, epe_profile):
        writer.writerow([round(t, 6), round(pfe, 6), round(epe, 6)])
    return buf.getvalue().encode("utf-8")


def export_margin_calls_csv(margin_calls: List[Dict[str, Any]]) -> bytes:
    """Return UTF-8 CSV bytes for a list of margin call dicts."""
    if not margin_calls:
        return b"id,counterparty_id,amount,excess_exposure,status,reason,issued_at\n"
    buf = io.StringIO()
    fields = ["id", "counterparty_id", "amount", "excess_exposure",
              "status", "reason", "issued_at"]
    writer = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(margin_calls)
    return buf.getvalue().encode("utf-8")


# ── PDF ───────────────────────────────────────────────────────────────────────

def export_simulation_pdf(
    *,
    run_id:          str,
    generated_by:    str,
    counterparty:    Dict[str, Any],
    base:            Dict[str, Any],
    stressed:        Optional[Dict[str, Any]] = None,
    margin_calls:    List[Dict[str, Any]]     = (),  # type: ignore[assignment]
    engine_info:     Optional[Dict[str, Any]] = None,
) -> bytes:
    """
    Build a PDF simulation report and return the raw bytes.

    Parameters
    ----------
    run_id          Database UUID of the simulation run
    generated_by    Username of the requesting user
    counterparty    Dict with keys: name, credit_rating, hazard_rate,
                    recovery_rate, collateral
    base            RiskMetrics dict (cva, wwr_cva, margin_required,
                    pfe_profile, epe_profile, time_grid_years,
                    compute_time_us, arch_used)
    stressed        Same shape as *base* or None
    margin_calls    List of margin call dicts (last N rows)
    engine_info     Dict from engine_info() — arch + simd_width
    """
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.platypus import (
            HRFlowable,
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )
    except ImportError as exc:
        raise RuntimeError("reportlab is required for PDF export") from exc

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )

    styles  = getSampleStyleSheet()
    h1      = styles["Heading1"]
    h2      = styles["Heading2"]
    normal  = styles["Normal"]
    small   = ParagraphStyle("small", parent=normal, fontSize=8, textColor=colors.grey)

    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # ── Palette ───────────────────────────────────────────────────────────────
    HEADER_BG  = colors.HexColor("#1c3557")
    ALT_ROW    = colors.HexColor("#f0f4f8")
    WHITE      = colors.white
    DARK_TEXT  = colors.HexColor("#0f1117")
    RED        = colors.HexColor("#c0392b")
    AMBER      = colors.HexColor("#e67e22")

    def _table_style(has_header: bool = True) -> TableStyle:
        cmds = [
            ("FONTNAME",      (0, 0), (-1, -1), "Helvetica"),
            ("FONTSIZE",      (0, 0), (-1, -1), 9),
            ("GRID",          (0, 0), (-1, -1), 0.25, colors.HexColor("#d0d7de")),
            ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
            ("ROWBACKGROUNDS",(0, 1), (-1, -1), [WHITE, ALT_ROW]),
        ]
        if has_header:
            cmds += [
                ("BACKGROUND",  (0, 0), (-1, 0), HEADER_BG),
                ("TEXTCOLOR",   (0, 0), (-1, 0), WHITE),
                ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE",    (0, 0), (-1, 0), 9),
            ]
        return TableStyle(cmds)

    story: list = []

    # ── Title ─────────────────────────────────────────────────────────────────
    story.append(Paragraph("CCR Simulation Report", h1))
    story.append(Paragraph(f"Run ID: {run_id}", normal))
    story.append(Paragraph(f"Generated: {now_str} &nbsp;&nbsp; By: {generated_by}", normal))
    story.append(HRFlowable(width="100%", thickness=1, color=HEADER_BG))
    story.append(Spacer(1, 0.4 * cm))

    # ── Counterparty summary ──────────────────────────────────────────────────
    story.append(Paragraph("Counterparty", h2))
    cp_data = [
        ["Name", "Credit Rating", "Hazard Rate", "Recovery Rate", "Collateral"],
        [
            str(counterparty.get("name", "—")),
            str(counterparty.get("credit_rating", "—")),
            f"{counterparty.get('hazard_rate', 0):.4f}",
            f"{counterparty.get('recovery_rate', 0):.2%}",
            f"{counterparty.get('collateral', 0):,.2f}",
        ],
    ]
    story.append(Table(cp_data, hAlign="LEFT", style=_table_style()))
    story.append(Spacer(1, 0.4 * cm))

    # ── Risk summary ──────────────────────────────────────────────────────────
    story.append(Paragraph("Risk Metrics", h2))
    headers = ["Metric", "Base"]
    if stressed:
        headers.append("Stressed")
    risk_rows = [headers]

    def _row(label: str, base_val: Any, stress_val: Any = None) -> list:
        row = [label, base_val]
        if stressed is not None:
            row.append(stress_val if stress_val is not None else "—")
        return row

    risk_rows.append(_row(
        "CVA",
        f"{base.get('cva', 0):,.4f}",
        f"{stressed.get('cva', 0):,.4f}" if stressed else None,
    ))
    risk_rows.append(_row(
        "WWR-CVA",
        f"{base.get('wwr_cva', 0):,.4f}",
        f"{stressed.get('wwr_cva', 0):,.4f}" if stressed else None,
    ))
    risk_rows.append(_row(
        "Margin Required",
        f"{base.get('margin_required', 0):,.2f}",
        f"{stressed.get('margin_required', 0):,.2f}" if stressed else None,
    ))
    risk_rows.append(_row(
        "Compute Time",
        f"{base.get('compute_time_us', 0):,} µs",
        f"{stressed.get('compute_time_us', 0):,} µs" if stressed else None,
    ))
    if "arch_used" in base:
        risk_rows.append(_row("Engine Arch", base["arch_used"]))

    risk_tbl = Table(risk_rows, hAlign="LEFT", style=_table_style())
    story.append(risk_tbl)
    story.append(Spacer(1, 0.4 * cm))

    # ── PFE/EPE profile ───────────────────────────────────────────────────────
    story.append(Paragraph("PFE / EPE Profile", h2))
    tg  = base.get("time_grid_years", [])
    pfe = base.get("pfe_profile", [])
    epe = base.get("epe_profile", [])

    profile_headers = ["T (years)", "PFE (base)", "EPE (base)"]
    if stressed:
        profile_headers += ["PFE (stress)", "EPE (stress)"]
        s_pfe = stressed.get("pfe_profile", [])
        s_epe = stressed.get("epe_profile", [])

    profile_data = [profile_headers]
    for i, t in enumerate(tg):
        row = [
            f"{t:.3f}",
            f"{pfe[i]:,.4f}" if i < len(pfe) else "—",
            f"{epe[i]:,.4f}" if i < len(epe) else "—",
        ]
        if stressed:
            row.append(f"{s_pfe[i]:,.4f}" if i < len(s_pfe) else "—")
            row.append(f"{s_epe[i]:,.4f}" if i < len(s_epe) else "—")
        profile_data.append(row)

    if len(profile_data) > 1:
        story.append(Table(profile_data, hAlign="LEFT", style=_table_style()))
    else:
        story.append(Paragraph("No profile data available.", normal))
    story.append(Spacer(1, 0.4 * cm))

    # ── Margin call history ───────────────────────────────────────────────────
    if margin_calls:
        story.append(Paragraph("Recent Margin Calls (last 10)", h2))
        mc_headers = ["Issued At", "Amount", "Excess", "Status", "Reason"]
        mc_data = [mc_headers]
        for mc in margin_calls[:10]:
            issued = mc.get("issued_at", "")
            if isinstance(issued, datetime):
                issued = issued.strftime("%Y-%m-%d %H:%M")
            mc_data.append([
                str(issued)[:16],
                f"{mc.get('amount', 0):,.2f}",
                f"{mc.get('excess_exposure', 0):,.2f}",
                str(mc.get("status", "—")),
                str(mc.get("reason", ""))[:60],
            ])
        mc_tbl = Table(mc_data, hAlign="LEFT", style=_table_style())
        story.append(mc_tbl)
        story.append(Spacer(1, 0.4 * cm))

    # ── Footer ────────────────────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.grey))
    arch = engine_info.get("arch", "unknown") if engine_info else "unknown"
    simd = engine_info.get("simd_width", "?") if engine_info else "?"
    story.append(Paragraph(
        f"CCR Engine v1.0.0 &nbsp;|&nbsp; Arch: {arch} &nbsp;|&nbsp; "
        f"SIMD width: {simd} &nbsp;|&nbsp; {now_str}",
        small,
    ))

    doc.build(story)
    return buf.getvalue()
