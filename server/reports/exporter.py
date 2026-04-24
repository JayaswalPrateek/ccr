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


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fmt(v: float, decimals: int = 2) -> str:
    """Abbreviate a number to K/M/B for display in tables."""
    if v >= 1e9:  return f"{v/1e9:.{decimals}f}B"
    if v >= 1e6:  return f"{v/1e6:.{decimals}f}M"
    if v >= 1e3:  return f"{v/1e3:.1f}K"
    return f"{v:,.{decimals}f}"


_RATING_COLOR_HEX = {
    "AAA": "#0f9960", "AA": "#15b371", "A": "#3dcc91",
    "BBB": "#f0b429", "BB": "#e67e22", "B": "#e05c31",
    "CCC": "#c0392b", "D": "#922b21",
}


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
    sa_ccr:          Optional[Dict[str, Any]] = None,
) -> bytes:
    """
    Build a polished PDF simulation report and return the raw bytes.

    Parameters
    ----------
    run_id          Database UUID of the simulation run
    generated_by    Username of the requesting user
    counterparty    Dict: name, credit_rating, hazard_rate, recovery_rate,
                    collateral, mpor_days
    base            RiskMetrics dict (cva, wwr_cva, margin_required,
                    pfe_profile, epe_profile, time_grid_years,
                    compute_time_us, arch_used)
    stressed        Same shape as *base* or None
    margin_calls    List of margin call dicts (last N rows)
    engine_info     Dict from engine_info() — arch + simd_width
    sa_ccr          Optional SA-CCR result dict: ead, rc, add_on_aggregate
    """
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.platypus import (
            HRFlowable,
            PageBreak,
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )
    except ImportError as exc:
        raise RuntimeError("reportlab is required for PDF export") from exc

    buf = io.BytesIO()
    A4_W, A4_H = A4
    MARGIN = 1.5 * cm
    doc = SimpleDocTemplate(
        buf,
        pagesize       = A4,
        leftMargin     = MARGIN,
        rightMargin    = MARGIN,
        topMargin      = MARGIN,
        bottomMargin   = MARGIN,
    )

    # ── Palette ───────────────────────────────────────────────────────────────
    HEADER_BG   = colors.HexColor("#1c3557")
    COVER_TEXT  = colors.white
    ALT_ROW     = colors.HexColor("#f0f4f8")
    WHITE       = colors.white
    RED         = colors.HexColor("#c0392b")
    AMBER       = colors.HexColor("#e67e22")
    GREEN       = colors.HexColor("#1e7e47")
    MUTED       = colors.HexColor("#64748b")
    BORDER      = colors.HexColor("#d0d7de")
    STRESSED_BG = colors.HexColor("#4a2c0a")

    styles = getSampleStyleSheet()
    normal = styles["Normal"]
    h2     = styles["Heading2"]
    small  = ParagraphStyle("small",  parent=normal, fontSize=8,  textColor=MUTED)
    body   = ParagraphStyle("body",   parent=normal, fontSize=9,  leading=14)
    bullet = ParagraphStyle("bullet", parent=normal, fontSize=9,  leading=14, leftIndent=12, bulletIndent=0)
    bold9  = ParagraphStyle("bold9",  parent=normal, fontSize=9,  fontName="Helvetica-Bold")
    white_title = ParagraphStyle("white_title", parent=normal, fontSize=22, fontName="Helvetica-Bold", textColor=COVER_TEXT)
    white_sub   = ParagraphStyle("white_sub",   parent=normal, fontSize=11, textColor=colors.HexColor("#7aa3c8"))
    white_label = ParagraphStyle("white_label", parent=normal, fontSize=8,  textColor=colors.HexColor("#94a3b8"), leading=12)
    white_val   = ParagraphStyle("white_val",   parent=normal, fontSize=18, fontName="Helvetica-Bold", textColor=COVER_TEXT, leading=22)
    white_small = ParagraphStyle("white_small", parent=normal, fontSize=8,  textColor=colors.HexColor("#94a3b8"))

    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    content_width = A4_W - 2 * MARGIN

    def _table_style(has_header: bool = True, stress_col: bool = False) -> TableStyle:
        cmds = [
            ("FONTNAME",       (0, 0), (-1, -1), "Helvetica"),
            ("FONTSIZE",       (0, 0), (-1, -1), 8),
            ("GRID",           (0, 0), (-1, -1), 0.25, BORDER),
            ("VALIGN",         (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING",     (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING",  (0, 0), (-1, -1), 4),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, ALT_ROW]),
        ]
        if has_header:
            cmds += [
                ("BACKGROUND",  (0, 0), (-1, 0), HEADER_BG),
                ("TEXTCOLOR",   (0, 0), (-1, 0), WHITE),
                ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE",    (0, 0), (-1, 0), 8),
            ]
            if stress_col:
                cmds += [
                    ("BACKGROUND", (-1, 0), (-1, 0), colors.HexColor("#7c3c0e")),
                ]
        return TableStyle(cmds)

    story: list = []

    # ─────────────────────────────────────────────────────────────────────────
    # COVER PAGE — dark header block filling the logical page width
    # ─────────────────────────────────────────────────────────────────────────
    rating     = counterparty.get("credit_rating", "—")
    rating_col = colors.HexColor(_RATING_COLOR_HEX.get(str(rating).upper(), "#64748b"))
    cva_val    = base.get("cva", 0.0)
    margin_val = base.get("margin_required", 0.0)

    # Top brand strip
    cover_brand = Table(
        [[
            Paragraph("CCR Engine", ParagraphStyle("brand", parent=normal, fontSize=10,
                       fontName="Helvetica", textColor=colors.HexColor("#7aa3c8"),
                       letterSpacing=2)),
            Paragraph(f"Run: {run_id[:8].upper()}", ParagraphStyle("runid", parent=normal,
                       fontSize=8, textColor=colors.HexColor("#94a3b8"))),
        ]],
        colWidths=[content_width * 0.7, content_width * 0.3],
    )
    cover_brand.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), HEADER_BG),
        ("TOPPADDING",    (0, 0), (-1, -1), 14),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING",   (0, 0), (-1, -1), 16),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 16),
        ("ALIGN", (1, 0), (1, 0), "RIGHT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))

    # Counterparty name + subtitle
    cover_name = Table(
        [[
            Paragraph("Counterparty Credit Risk Report", white_sub),
        ]],
        colWidths=[content_width],
    )
    cover_name.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), HEADER_BG),
        ("LEFTPADDING",   (0, 0), (-1, -1), 16),
        ("TOPPADDING",    (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))

    cover_cp_name = Table(
        [[Paragraph(str(counterparty.get("name", "—")), white_title)]],
        colWidths=[content_width],
    )
    cover_cp_name.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), HEADER_BG),
        ("LEFTPADDING",   (0, 0), (-1, -1), 16),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 20),
    ]))

    # Three metric boxes: CVA | Margin Required | Credit Rating
    cva_bg     = colors.HexColor("#7f1d1d") if cva_val > 0 else colors.HexColor("#14532d")
    margin_bg  = colors.HexColor("#78350f") if margin_val > 0 else colors.HexColor("#1c1f26")
    rating_bg  = colors.HexColor("#1c1f26")

    def _metric_cell(label: str, value: str, bg_hex: str) -> Table:
        bg = colors.HexColor(bg_hex) if isinstance(bg_hex, str) else bg_hex
        t = Table(
            [
                [Paragraph(label, white_label)],
                [Paragraph(value, white_val)],
            ],
            colWidths=[(content_width - 32) / 3],
        )
        t.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, -1), bg),
            ("TOPPADDING",    (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
            ("LEFTPADDING",   (0, 0), (-1, -1), 12),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 12),
            ("BOX",           (0, 0), (-1, -1), 0.5, colors.HexColor("#2d3142")),
        ]))
        return t

    m_cell_w = (content_width - 48) / 3
    metric_row = Table(
        [[
            _metric_cell("CVA", _fmt(cva_val), "#7f1d1d" if cva_val > 0 else "#14532d"),
            _metric_cell("Margin Required", _fmt(margin_val, 0), "#78350f" if margin_val > 0 else "#1c1f26"),
            _metric_cell("Credit Rating", str(rating), "#1c1f26"),
        ]],
        colWidths=[m_cell_w, m_cell_w, m_cell_w],
        hAlign="LEFT",
    )
    metric_row.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), HEADER_BG),
        ("TOPPADDING",    (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ("LEFTPADDING",   (0, 0), (-1, -1), 16),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 16),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ]))

    cover_metrics_wrap = Table([[metric_row]], colWidths=[content_width])
    cover_metrics_wrap.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), HEADER_BG),
        ("TOPPADDING",    (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 20),
        ("LEFTPADDING",   (0, 0), (-1, -1), 0),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 0),
    ]))

    # Footer strip with run metadata
    cover_footer = Table(
        [[
            Paragraph(
                f"Run ID: {run_id} &nbsp;|&nbsp; Generated: {now_str} &nbsp;|&nbsp; By: {generated_by}",
                white_small,
            ),
        ]],
        colWidths=[content_width],
    )
    cover_footer.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), colors.HexColor("#0f1117")),
        ("TOPPADDING",    (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING",   (0, 0), (-1, -1), 16),
    ]))

    story.extend([
        cover_brand,
        cover_name,
        cover_cp_name,
        cover_metrics_wrap,
        cover_footer,
        PageBreak(),
    ])

    # ─────────────────────────────────────────────────────────────────────────
    # PAGE 2 — Executive Summary
    # ─────────────────────────────────────────────────────────────────────────
    story.append(Paragraph("Executive Summary", h2))

    tg  = base.get("time_grid_years", [])
    pfe = base.get("pfe_profile",     [])
    epe = base.get("epe_profile",     [])

    peak_pfe = max(pfe) if pfe else 0.0
    peak_t   = tg[pfe.index(peak_pfe)] if (pfe and tg and peak_pfe > 0) else 0.0

    arch      = engine_info.get("arch", "unknown") if engine_info else "unknown"
    simd      = engine_info.get("simd_width", "?")  if engine_info else "?"
    compute_s = f"{base.get('compute_time_us', 0) / 1_000_000:.3f}s"

    cva_vs_stress = ""
    if stressed:
        delta = stressed.get("cva", 0.0) - cva_val
        cva_vs_stress = (
            f" Under stress, CVA increases by {_fmt(abs(delta))} ({'+' if delta > 0 else ''}{delta/cva_val*100:.1f}%)."
            if cva_val > 0 else " Stressed CVA: " + _fmt(stressed.get("cva", 0.0)) + "."
        )

    findings = [
        f"<b>CVA:</b> {_fmt(cva_val)} computed via Monte Carlo (GBM, Kahan summation).{cva_vs_stress}",
        f"<b>Margin:</b> {'Breach detected — required ' + _fmt(margin_val, 0) + ' in initial margin.' if margin_val > 0 else 'No margin breach — portfolio is within collateral limits.'}",
        f"<b>Peak PFE:</b> {_fmt(peak_pfe, 0)} at T={peak_t:.2f}y (99th percentile of simulated exposures).",
        f"<b>Engine:</b> {arch} · {base.get('paths_used', 0):,} paths simulated.",
    ]

    summary_rows = [[Paragraph("Key Findings", bold9)]] + [
        [Paragraph(f"• {f}", bullet)] for f in findings
    ]
    summary_tbl = Table(summary_rows, colWidths=[content_width])
    summary_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), ALT_ROW),
        ("BACKGROUND",    (0, 0), (0, 0),   HEADER_BG),
        ("TEXTCOLOR",     (0, 0), (0, 0),   WHITE),
        ("BOX",           (0, 0), (-1, -1), 0.5, BORDER),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
    ]))
    story.append(summary_tbl)
    story.append(Spacer(1, 0.4 * cm))

    # ── Counterparty detail ───────────────────────────────────────────────────
    story.append(Paragraph("Counterparty", h2))
    cp_data = [
        ["Name", "Credit Rating", "Hazard Rate", "Recovery Rate", "Collateral", "MPOR"],
        [
            str(counterparty.get("name", "—")),
            str(counterparty.get("credit_rating", "—")),
            f"{counterparty.get('hazard_rate', 0):.4f}",
            f"{counterparty.get('recovery_rate', 0):.1%}",
            _fmt(counterparty.get("collateral", 0.0), 0),
            f"{counterparty.get('mpor_days', 10)}d",
        ],
    ]
    cp_tbl = Table(cp_data, hAlign="LEFT", style=_table_style())
    story.append(cp_tbl)
    story.append(Spacer(1, 0.4 * cm))

    # ── Risk Metrics ──────────────────────────────────────────────────────────
    story.append(Paragraph("Risk Metrics", h2))
    has_stress = stressed is not None
    headers = ["Metric", "Base"]
    if has_stress:
        headers.append("Stressed")
    risk_rows = [headers]

    def _row(label: str, base_val: Any, stress_val: Any = None) -> list:
        row = [label, base_val]
        if has_stress:
            row.append(stress_val if stress_val is not None else "—")
        return row

    risk_rows.append(_row("CVA",      _fmt(cva_val),    _fmt(stressed.get("cva", 0.0)) if stressed else None))
    risk_rows.append(_row("WWR-CVA",  _fmt(base.get("wwr_cva", 0.0)),
                                       _fmt(stressed.get("wwr_cva", 0.0)) if stressed else None))
    risk_rows.append(_row("Margin Required",
                          _fmt(margin_val, 0),
                          _fmt(stressed.get("margin_required", 0.0), 0) if stressed else None))
    if sa_ccr:
        risk_rows.append(_row("SA-CCR EAD (Basel III)", _fmt(sa_ccr.get("ead", 0.0), 0)))
        risk_rows.append(_row("  → RC",                 _fmt(sa_ccr.get("rc", 0.0), 0)))
        risk_rows.append(_row("  → AddOn",              _fmt(sa_ccr.get("add_on_aggregate", 0.0), 0)))

    risk_tbl = Table(risk_rows, hAlign="LEFT", style=_table_style(has_header=True, stress_col=has_stress))
    # Colour CVA row
    cva_row_idx = 1
    if cva_val > 0:
        risk_tbl.setStyle(TableStyle([("BACKGROUND", (1, cva_row_idx), (1, cva_row_idx),
                                       colors.HexColor("#fde8e8"))]))
    else:
        risk_tbl.setStyle(TableStyle([("BACKGROUND", (1, cva_row_idx), (1, cva_row_idx),
                                       colors.HexColor("#d1fae5"))]))
    # Colour Margin row (row 3)
    if margin_val > 0:
        risk_tbl.setStyle(TableStyle([("BACKGROUND", (1, 3), (1, 3), colors.HexColor("#fff3cd"))]))
    story.append(risk_tbl)
    story.append(Spacer(1, 0.4 * cm))

    # ── PFE/EPE Profile ───────────────────────────────────────────────────────
    story.append(Paragraph("PFE / EPE Profile", h2))

    profile_headers = ["T (years)", "PFE", "EPE"]
    if has_stress:
        profile_headers += ["PFE (stress)", "EPE (stress)"]

    s_pfe = stressed.get("pfe_profile", []) if stressed else []
    s_epe = stressed.get("epe_profile", []) if stressed else []

    MAX_PROFILE_ROWS = 20
    tg_display = tg[:MAX_PROFILE_ROWS]
    truncated  = len(tg) > MAX_PROFILE_ROWS

    profile_data = [profile_headers]
    for i, t in enumerate(tg_display):
        row = [
            f"{t:.3f}",
            _fmt(pfe[i], 0) if i < len(pfe) else "—",
            _fmt(epe[i], 0) if i < len(epe) else "—",
        ]
        if has_stress:
            row.append(_fmt(s_pfe[i], 0) if i < len(s_pfe) else "—")
            row.append(_fmt(s_epe[i], 0) if i < len(s_epe) else "—")
        profile_data.append(row)

    if len(profile_data) > 1:
        story.append(Table(profile_data, hAlign="LEFT", style=_table_style(stress_col=has_stress)))
        if truncated:
            story.append(Paragraph(
                f"… {len(tg) - MAX_PROFILE_ROWS} additional rows not shown. Download CSV for the full profile.",
                small,
            ))
    else:
        story.append(Paragraph("No profile data available.", normal))
    story.append(Spacer(1, 0.4 * cm))

    # ── Margin call history ───────────────────────────────────────────────────
    if margin_calls:
        story.append(Paragraph("Recent Margin Calls", h2))
        mc_headers = ["Issued", "Amount", "Excess Exposure", "Status", "Reason (truncated)"]
        mc_data    = [mc_headers]
        for mc in list(margin_calls)[:10]:
            issued = mc.get("issued_at", "")
            if isinstance(issued, datetime):
                issued = issued.strftime("%Y-%m-%d %H:%M")
            status = str(mc.get("status", "—"))
            mc_data.append([
                str(issued)[:16],
                _fmt(mc.get("amount", 0.0), 0),
                _fmt(mc.get("excess_exposure", 0.0), 0),
                status,
                str(mc.get("reason", ""))[:55],
            ])
        mc_tbl = Table(mc_data, hAlign="LEFT", style=_table_style())
        # Highlight PENDING rows in amber
        for ri, mc in enumerate(list(margin_calls)[:10], start=1):
            if mc.get("status") == "PENDING":
                mc_tbl.setStyle(TableStyle([
                    ("BACKGROUND", (0, ri), (-1, ri), colors.HexColor("#fff8e1"))
                ]))
        story.append(mc_tbl)
        story.append(Spacer(1, 0.4 * cm))

    # ── Footer ────────────────────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5, color=BORDER))
    story.append(Paragraph(
        f"CCR Engine v1.0 &nbsp;|&nbsp; {now_str} &nbsp;|&nbsp; Confidential",
        small,
    ))

    doc.build(story)
    return buf.getvalue()
