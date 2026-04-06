# pdf_export.py  — v2.1 (Corrected Header & Table Layouts)
"""
Portfolio PDF Export Module
- Charts via matplotlib (donut, bar, benchmark, MC, optimizer, MCTR, attribution)
- AI Analysis as Section 2 (Portfolio Description) right after Overview
- HSTG@FT gold badge on every page
- KeepInFrame in every table cell to prevent text overflow
- Clean navy + white + accent colour scheme
"""

import io
import re
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    HRFlowable, Image, KeepInFrame,
    PageBreak, Paragraph, SimpleDocTemplate,
    Spacer, Table, TableStyle,
)

# ── Palette ───────────────────────────────────────────────────────────────────
NAVY   = colors.HexColor("#1E3A5F")
GREEN  = colors.HexColor("#16a34a")
RED    = colors.HexColor("#dc2626")
AMBER  = colors.HexColor("#d97706")
SILVER = colors.HexColor("#f8fafc")
LGRAY  = colors.HexColor("#e2e8f0")
MGRAY  = colors.HexColor("#94a3b8")
DGRAY  = colors.HexColor("#475569")
WHITE  = colors.white
BLACK  = colors.HexColor("#0f172a")

PAGE_W, PAGE_H = A4
MARGIN  = 1.6 * cm
CONTENT = PAGE_W - 2 * MARGIN

# matplotlib colours
_MN  = "#1E3A5F"
_GR  = "#16a34a"
_RD  = "#dc2626"
_AM  = "#d97706"
_GY  = "#94a3b8"
_PAL = ["#1E3A5F","#2563eb","#16a34a","#d97706","#dc2626",
        "#7c3aed","#0e7490","#b45309","#64748b","#be185d"]


# ─────────────────────────────────────────────────────────────────────────────
# CHART HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _fig_to_img(fig, w_cm, h_cm):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    plt.close(fig)
    return Image(buf, width=w_cm * cm, height=h_cm * cm)


def _chart_donut(tickers, weights_pct, cash_pct, names_dict):
    labels, sizes, clrs = [], [], []
    for i, (t, w) in enumerate(zip(tickers, weights_pct)):
        nm = (names_dict or {}).get(t, t)
        nm = nm[:16] + "…" if len(nm) > 16 else nm
        labels.append(f"{nm}\n{w:.1f}%")
        sizes.append(w)
        clrs.append(_PAL[i % len(_PAL)])
    if cash_pct and cash_pct > 0:
        labels.append(f"Cash\n{cash_pct*100:.1f}%")
        sizes.append(cash_pct * 100)
        clrs.append(_GY)

    fig, ax = plt.subplots(figsize=(6, 4.2), facecolor="white")
    wedges, _ = ax.pie(sizes, labels=None, colors=clrs, startangle=90,
                       counterclock=False,
                       wedgeprops=dict(width=0.52, edgecolor="white", linewidth=1.5))
    ax.text(0, 0, "Portfolio\nWeights", ha="center", va="center",
            fontsize=8, fontweight="bold", color=_MN)
    ncol = 2 if len(labels) > 6 else 1
    ax.legend(wedges, labels, loc="center left",
              bbox_to_anchor=(1.0, 0.5), fontsize=6, frameon=False, ncol=ncol)
    ax.set_title("Allocation", fontsize=9, fontweight="bold", color=_MN, pad=6)
    fig.tight_layout()
    return _fig_to_img(fig, 10, 6.2)


def _chart_risk_bar(metrics):
    so = not np.isnan(metrics.get("sortino", float("nan")))
    vals  = [metrics["volatility_pct"], metrics["sharpe"],
             metrics["sortino"] if so else 0, abs(metrics["max_drawdown_pct"])]
    labs  = ["Volatility (%)", "Sharpe Ratio", "Sortino Ratio", "Max Drawdown (%)"]
    disps = [f"{metrics['volatility_pct']:.1f}%", f"{metrics['sharpe']:.2f}",
             f"{metrics['sortino']:.2f}" if so else "N/A",
             f"{abs(metrics['max_drawdown_pct']):.1f}%"]
    clrs  = [_AM, _GR if metrics["sharpe"] >= 1 else _AM,
             _GR if so and metrics["sortino"] >= 1 else _AM, _RD]

    fig, ax = plt.subplots(figsize=(6.5, 3), facecolor="white")
    bars = ax.barh(range(4), vals, color=clrs, height=0.5, edgecolor="white")
    ax.set_yticks(range(4))
    ax.set_yticklabels(labs, fontsize=8, color=_MN, fontweight="bold")
    ax.xaxis.set_visible(False)
    ax.spines[:].set_visible(False)
    mx = max((v for v in vals if v > 0), default=1)
    for bar, dv in zip(bars, disps):
        ax.text(bar.get_width() + mx * 0.04,
                bar.get_y() + bar.get_height() / 2,
                dv, va="center", fontsize=8.5, fontweight="bold", color=_MN)
    ax.set_xlim(0, mx * 1.5)
    ax.set_title("Core Risk Metrics", fontsize=9, fontweight="bold",
                 color=_MN, loc="left", pad=5)
    fig.tight_layout()
    return _fig_to_img(fig, 9.5, 4.2)


def _chart_var(metrics):
    cats = ["VaR\n95%", "CVaR\n95%", "VaR\n99%", "CVaR\n99%"]
    vals = [abs(metrics["var_95_pct"]), abs(metrics["cvar_95_pct"]),
            abs(metrics["var_99_pct"]), abs(metrics["cvar_99_pct"])]
    clrs = [_AM, _RD, _AM, _RD]

    fig, ax = plt.subplots(figsize=(5, 3), facecolor="white")
    bars = ax.bar(range(4), vals, color=clrs, width=0.5, edgecolor="white")
    ax.set_xticks(range(4))
    ax.set_xticklabels(cats, fontsize=8)
    ax.set_ylabel("Daily Loss (%)", fontsize=7.5)
    ax.yaxis.set_tick_params(labelsize=7.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    mx = max(vals) if vals else 1
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + mx * 0.04,
                f"-{bar.get_height():.2f}%",
                ha="center", fontsize=8, fontweight="bold", color=_MN)
    ax.set_ylim(0, mx * 1.4)
    ax.set_title("VaR & CVaR", fontsize=9, fontweight="bold",
                 color=_MN, loc="left", pad=5)
    fig.tight_layout()
    return _fig_to_img(fig, 7.5, 4.2)


def _chart_bench(bench):
    cats = ["Portfolio", bench["label"], "Alpha"]
    vals = [bench["port_total"] * 100, bench["bench_total"] * 100, bench["alpha"]]
    clrs = [_MN, _GY, _GR if bench["alpha"] >= 0 else _RD]

    fig, ax = plt.subplots(figsize=(5.5, 3.5), facecolor="white")
    bars = ax.bar(cats, vals, color=clrs, width=0.42, edgecolor="white")
    ax.axhline(0, color=_GY, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylabel("Return (%)", fontsize=8)
    ax.tick_params(labelsize=8)
    mx = max(abs(v) for v in vals) if vals else 1
    for bar, v in zip(bars, vals):
        ypos = v + mx * 0.04 if v >= 0 else v - mx * 0.1
        ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                f"{v:+.1f}%", ha="center", fontsize=8.5, fontweight="bold", color=_MN)
    mn = min(v for v in vals)
    ax.set_ylim(mn * 1.25 - 5, max(v for v in vals) * 1.25 + 5)
    ax.set_title("Portfolio vs Benchmark", fontsize=9, fontweight="bold",
                 color=_MN, loc="left", pad=5)
    fig.tight_layout()
    return _fig_to_img(fig, 7.5, 4.5)


def _chart_mc(mc):
    cats = ["Bad\n(5th)", "Median\n(50th)", "Good\n(95th)"]
    vals = [mc["p5"], mc["p50"], mc["p95"]]
    clrs = [_RD, _AM, _GR]

    fig, ax = plt.subplots(figsize=(5.5, 3.5), facecolor="white")
    bars = ax.bar(cats, vals, color=clrs, width=0.42, edgecolor="white")
    ax.axhline(1.0, color=_GY, linewidth=1, linestyle="--", label="Break-even")
    ax.legend(fontsize=7, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylabel("Portfolio Value (×)", fontsize=8)
    ax.tick_params(labelsize=8)
    mx = max(vals) if vals else 2
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + mx * 0.025,
                f"{v:.2f}×", ha="center", fontsize=9, fontweight="bold", color=_MN)
    ax.set_ylim(0, mx * 1.22)
    ax.set_title(f"Monte Carlo ({mc['days']} days)", fontsize=9,
                 fontweight="bold", color=_MN, loc="left", pad=5)
    fig.tight_layout()
    return _fig_to_img(fig, 7.5, 4.5)


def _chart_opt(opt):
    labs = ["Return (%)", "Volatility (%)", "Sharpe"]
    man  = [opt["manual_ret"]*100, opt["manual_vol"]*100, opt["manual_sharpe"]]
    op   = [opt["opt_ret"]*100,    opt["opt_vol"]*100,    opt["opt_sharpe"]]
    x = np.arange(len(labs))

    fig, ax = plt.subplots(figsize=(6.5, 3.2), facecolor="white")
    w = 0.32
    b1 = ax.bar(x-w/2, man, width=w, color=_GY, label="Manual",    edgecolor="white")
    b2 = ax.bar(x+w/2, op,  width=w, color=_MN, label="Optimized", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(labs, fontsize=8.5)
    ax.legend(fontsize=8, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    mx = max(max(man), max(op))
    for grp in (b1, b2):
        for bar in grp:
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + mx*0.025,
                    f"{bar.get_height():.2f}",
                    ha="center", fontsize=7.5, fontweight="bold", color=_MN)
    ax.set_ylim(0, mx * 1.35)
    ax.set_title("Manual vs Optimized", fontsize=9, fontweight="bold",
                 color=_MN, loc="left", pad=5)
    fig.tight_layout()
    return _fig_to_img(fig, 9, 4.5)


def _chart_mctr(df):
    eq = df[df["Ticker"] != "CASH"].copy()
    if eq.empty:
        return None
    eq = eq.sort_values("% Risk Contribution", ascending=True)
    names = [(str(n)[:22]+"…" if len(str(n))>22 else str(n)) for n in eq["Company"]]
    vals  = eq["% Risk Contribution"].values.astype(float)
    norm  = plt.Normalize(vals.min(), vals.max())
    clrs  = plt.cm.YlOrRd(norm(vals))

    h = max(3, len(eq) * 0.58)
    fig, ax = plt.subplots(figsize=(8.5, h), facecolor="white")
    bars = ax.barh(range(len(eq)), vals, color=clrs, height=0.52, edgecolor="white")
    ax.set_yticks(range(len(eq)))
    ax.set_yticklabels(names, fontsize=8)
    ax.xaxis.set_visible(False)
    ax.spines[:].set_visible(False)
    mx = vals.max() if len(vals) else 1
    for bar, v in zip(bars, vals):
        ax.text(v + mx*0.015, bar.get_y() + bar.get_height()/2,
                f"{v:.1f}%", va="center", fontsize=8.5, fontweight="bold", color=_MN)
    ax.set_xlim(0, mx * 1.28)
    ax.set_title("% Risk Contribution", fontsize=9, fontweight="bold",
                 color=_MN, loc="left", pad=5)
    fig.tight_layout()
    return _fig_to_img(fig, 11, max(4, len(eq)*0.8))


def _chart_contrib(df):
    eq = df[(df["Ticker"] != "CASH") & (df["Ticker"] != "—")].copy()
    if eq.empty:
        return None
    eq = eq.sort_values("Active Contribution (%)", ascending=False)
    vals  = eq["Active Contribution (%)"].values.astype(float)
    names = [(str(n)[:13]+"…" if len(str(n))>13 else str(n)) for n in eq["Company"]]
    clrs  = [_GR if v >= 0 else _RD for v in vals]

    fig, ax = plt.subplots(figsize=(9.5, 3.8), facecolor="white")
    bars = ax.bar(range(len(eq)), vals, color=clrs, width=0.5, edgecolor="white")
    ax.axhline(0, color=_GY, linewidth=0.8)
    ax.set_xticks(range(len(eq)))
    ax.set_xticklabels(names, fontsize=7.5, rotation=30, ha="right")
    ax.set_ylabel("Active Contribution (%)", fontsize=7.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    mx = max(abs(v) for v in vals) if len(vals) else 1
    for bar, v in zip(bars, vals):
        ypos = v + mx*0.03 if v >= 0 else v - mx*0.09
        ax.text(bar.get_x() + bar.get_width()/2, ypos,
                f"{v:.1f}%", ha="center", fontsize=7.5, fontweight="bold", color=_MN)
    ax.set_title("Active Contribution by Stock", fontsize=9, fontweight="bold",
                 color=_MN, loc="left", pad=5)
    fig.tight_layout()
    return _fig_to_img(fig, 11, 5)


# ─────────────────────────────────────────────────────────────────────────────
# STYLES
# ─────────────────────────────────────────────────────────────────────────────

def _styles():
    return {
        # FIX: Increased leading for vertical spacing in header
        "cover_title": ParagraphStyle("CT", fontSize=24, textColor=WHITE,
            fontName="Helvetica-Bold", alignment=TA_LEFT, spaceAfter=8, leading=30),
        "cover_sub":   ParagraphStyle("CS", fontSize=9.5,
            textColor=colors.HexColor("#93c5fd"),
            fontName="Helvetica", alignment=TA_LEFT, leading=14),
        "h2": ParagraphStyle("H2", fontSize=12, textColor=NAVY,
            fontName="Helvetica-Bold", spaceBefore=0, spaceAfter=3),
        "h3": ParagraphStyle("H3", fontSize=9.5, textColor=NAVY,
            fontName="Helvetica-Bold", spaceBefore=8, spaceAfter=3),
        "body": ParagraphStyle("BD", fontSize=8.5, textColor=BLACK,
            fontName="Helvetica", leading=13, spaceAfter=3),
        "small": ParagraphStyle("SM", fontSize=7.5, textColor=DGRAY,
            fontName="Helvetica", leading=11, spaceAfter=3),
        "tbl_hdr": ParagraphStyle("TH", fontSize=7.5, textColor=WHITE,
            fontName="Helvetica-Bold", alignment=TA_CENTER),
        "tbl_cell": ParagraphStyle("TC", fontSize=7.5, textColor=BLACK,
            fontName="Helvetica", leading=10),
        "disclaimer": ParagraphStyle("DC", fontSize=6.5, textColor=MGRAY,
            fontName="Helvetica-Oblique", leading=9),
        "metric_val": ParagraphStyle("MV", fontSize=15, textColor=NAVY,
            fontName="Helvetica-Bold", alignment=TA_CENTER),
        "metric_lbl": ParagraphStyle("ML", fontSize=7, textColor=DGRAY,
            fontName="Helvetica", alignment=TA_CENTER),
        "ai_h": ParagraphStyle("AH", fontSize=10, textColor=NAVY,
            fontName="Helvetica-Bold", spaceBefore=10, spaceAfter=4),
        "ai_body": ParagraphStyle("AB", fontSize=8.5, textColor=BLACK,
            fontName="Helvetica", leading=13, spaceAfter=5),
    }


# ─────────────────────────────────────────────────────────────────────────────
# TABLE + CELL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _cell(text, style, max_h=0.8*cm):
    """Paragraph wrapped in KeepInFrame to prevent cell overflow."""
    p = Paragraph(str(text), style)
    return KeepInFrame(0, max_h, [p], mode="shrink")


def _base_ts(header_bg=NAVY, alt=SILVER):
    return TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  header_bg),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  WHITE),
        ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, 0),  8),
        ("ALIGN",         (0, 0), (-1, 0),  "CENTER"),
        ("TOPPADDING",    (0, 0), (-1, 0),  5),
        ("BOTTOMPADDING", (0, 0), (-1, 0),  5),
        ("FONTNAME",      (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",      (0, 1), (-1, -1), 7.5),
        ("TOPPADDING",    (0, 1), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 1), (-1, -1), 3),
        ("LEFTPADDING",   (0, 0), (-1, -1), 5),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 5),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [WHITE, alt]),
        ("GRID",          (0, 0), (-1, -1), 0.35, LGRAY),
        ("LINEBELOW",     (0, 0), (-1, 0),  1.2,  NAVY),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# LAYOUT HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _divider(story):
    story.append(HRFlowable(width="100%", thickness=0.4, color=LGRAY,
                             spaceBefore=4, spaceAfter=6))


def _sec_header(story, num, title, S):
    hdr = Table([[Paragraph(f"{num}. {title}", S["h2"])]],
                colWidths=[CONTENT],
                style=TableStyle([
                    ("BACKGROUND",    (0, 0), (-1, -1), SILVER),
                    ("LEFTPADDING",   (0, 0), (-1, -1), 10),
                    ("TOPPADDING",    (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                    ("LINEAFTER",     (0, 0), (0, 0), 3.5, colors.HexColor("#b8960c")),
                ]))
    story.append(hdr)
    story.append(Spacer(1, 0.22 * cm))


def _metric_tiles(story, tiles, S):
    n = len(tiles)
    w = CONTENT / n
    cells = []
    for label, value, clr in tiles:
        vstyle = ParagraphStyle(f"mv{label[:4]}", parent=S["metric_val"],
                                textColor=clr or NAVY)
        inner = Table(
            [[Paragraph(str(value), vstyle)],
             [Paragraph(label, S["metric_lbl"])]],
            colWidths=[w - 8],
            style=TableStyle([
                ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
                ("TOPPADDING",    (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                ("LEFTPADDING",   (0, 0), (-1, -1), 3),
                ("RIGHTPADDING",  (0, 0), (-1, -1), 3),
                ("BACKGROUND",    (0, 0), (-1, -1), WHITE),
                ("LINEBELOW",     (0, 1), (-1, 1), 2.5, clr or NAVY),
            ])
        )
        cells.append(inner)
    row = Table([cells], colWidths=[w]*n,
                style=TableStyle([
                    ("BACKGROUND",    (0, 0), (-1, -1), SILVER),
                    ("BOX",           (0, 0), (-1, -1), 0.4, LGRAY),
                    ("LEFTPADDING",   (0, 0), (-1, -1), 3),
                    ("RIGHTPADDING",  (0, 0), (-1, -1), 3),
                    ("TOPPADDING",    (0, 0), (-1, -1), 3),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ]))
    story.append(row)
    story.append(Spacer(1, 0.28 * cm))


# ─────────────────────────────────────────────────────────────────────────────
# PAGE TEMPLATE  (runs on every page — header rule + footer bar + HSTG badge)
# ─────────────────────────────────────────────────────────────────────────────

def _page_template(canvas, doc):
    canvas.saveState()

    # Top rule
    canvas.setStrokeColor(NAVY)
    canvas.setLineWidth(1.5)
    canvas.line(MARGIN, PAGE_H - MARGIN + 4*mm,
                PAGE_W - MARGIN, PAGE_H - MARGIN + 4*mm)

    # Footer bar
    canvas.setFillColor(NAVY)
    canvas.rect(0, 0, PAGE_W, 1.45*cm, fill=1, stroke=0)
    canvas.setFont("Helvetica", 6.5)
    canvas.setFillColor(WHITE)
    canvas.drawString(MARGIN, 0.55*cm,
        "Portfolio Analytics Dashboard  |  For educational purposes only. Not financial advice.")
    canvas.drawRightString(PAGE_W - MARGIN, 0.55*cm,
        f"Page {doc.page}  |  {datetime.now().strftime('%d %b %Y  %H:%M')}")

    # HSTG@FT gold badge
    bw, bh = 1.6*cm, 0.56*cm
    bx = PAGE_W - MARGIN - bw
    by = PAGE_H - MARGIN + 1.6*mm
    canvas.setFillColor(colors.HexColor("#b8960c"))
    canvas.roundRect(bx, by, bw, bh, 2.5*mm, fill=1, stroke=0)
    canvas.setFont("Helvetica-Bold", 7.5)
    canvas.setFillColor(WHITE)
    canvas.drawCentredString(bx + bw/2, by + 1.8*mm, "HSTG@FT")

    canvas.restoreState()


# ─────────────────────────────────────────────────────────────────────────────
# COVER
# ─────────────────────────────────────────────────────────────────────────────

def _cover(story, d, S):
    # FIX: Using specific cell padding and style leading to prevent overlap
    inner = Table(
        [[Paragraph("Portfolio Analytics Report", S["cover_title"])],
         [Paragraph(
             f"Period: {d['date_range']}  |  Assets: {d['num_assets']}  "
             f"|  Generated: {datetime.now().strftime('%d %b %Y')}",
             S["cover_sub"])]],
        colWidths=[CONTENT - 2*cm],
        style=TableStyle([("TOPPADDING",(0,0),(-1,0),2),
                          ("BOTTOMPADDING",(0,0),(-1,0),8), # Padding after title
                          ("TOPPADDING",(0,1),(-1,1),2),
                          ("LEFTPADDING",(0,0),(-1,-1),0)])
    )
    banner = Table([[inner]], colWidths=[CONTENT],
                   style=TableStyle([
                       ("BACKGROUND",    (0,0),(-1,-1), NAVY),
                       ("LEFTPADDING",   (0,0),(-1,-1), 16),
                       ("RIGHTPADDING",  (0,0),(-1,-1), 16),
                       ("TOPPADDING",    (0,0),(-1,-1), 18),
                       ("BOTTOMPADDING", (0,0),(-1,-1), 18),
                   ]))
    story.append(banner)
    story.append(Spacer(1, 0.5*cm))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Portfolio Overview
# ─────────────────────────────────────────────────────────────────────────────

def _sec_overview(story, d, S, num):
    _sec_header(story, num, "Portfolio Overview", S)
    S_th, S_tc = S["tbl_hdr"], S["tbl_cell"]

    # FIX: Rebalanced ratios to favor long stock names (e.g. Indian Stocks)
    tbl_w  = CONTENT * 0.52
    chrt_w = CONTENT * 0.48
    cws    = [tbl_w*0.22, tbl_w*0.60, tbl_w*0.18] # Ticker(22%), Company(60%), Weight(18%)

    rows = [[_cell("Ticker", S_th), _cell("Company", S_th), _cell("Weight", S_th)]]
    for t, nm, w in zip(d["tickers_list"], d["ticker_names_list"], d["weights_pct"]):
        # Truncation limit increased for readability
        nm_s = (nm[:32]+"…") if len(nm)>32 else nm
        rows.append([_cell(t, S_tc), _cell(nm_s, S_tc), _cell(f"{w:.2f}%", S_tc)])
    if d.get("cash_pct", 0) > 0:
        rows.append([_cell("CASH", S_tc), _cell("Cash (Risk-Free)", S_tc),
                     _cell(f"{d['cash_pct']*100:.2f}%", S_tc)])
    total = sum(d["weights_pct"]) + d.get("cash_pct", 0)*100
    rows.append([_cell("—", S_tc), _cell("TOTAL", S_tc),
                 _cell(f"{total:.2f}%", S_tc)])
    
    ts = _base_ts()
    ts.add("FONTNAME",   (0, len(rows)-1), (-1, len(rows)-1), "Helvetica-Bold")
    ts.add("BACKGROUND", (0, len(rows)-1), (-1, len(rows)-1), LGRAY)
    wt = Table(rows, colWidths=cws, style=ts, repeatRows=1)

    names_dict = {t: nm for t, nm in zip(d["tickers_list"], d["ticker_names_list"])}
    donut = _chart_donut(d["tickers_list"], d["weights_pct"],
                         d.get("cash_pct", 0), names_dict)

    # Padding added to separate table from chart
    side = Table([[wt, donut]], colWidths=[tbl_w, chrt_w],
                 style=TableStyle([("VALIGN",(0,0),(-1,-1),"TOP"),
                                   ("LEFTPADDING",(0,0),(-1,-1),0),
                                   ("RIGHTPADDING",(0,0),(0,0),10)])) 
    story.append(side)
    story.append(Spacer(1, 0.18*cm))

    facts = [f"<b>Date range:</b> {d['date_range']}",
             f"<b>Risk-free rate:</b> {d['risk_free']*100:.2f}% p.a."]
    if d.get("cash_pct", 0) > 0:
        facts.append(f"<b>Cash:</b> {d['cash_pct']*100:.1f}% at risk-free rate")
    story.append(Paragraph("  |  ".join(facts), S["small"]))
    story.append(Spacer(1, 0.35*cm))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — AI Portfolio Description  (immediately after Overview)
# ─────────────────────────────────────────────────────────────────────────────

def _sec_ai(story, d, S, num):
    _sec_header(story, num, "AI Portfolio Analysis", S)

    raw = d["ai_analysis"]
    raw = re.sub(r"^#{1,3}\s*\d*\.?\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", raw)
    raw = re.sub(r"\*(.+?)\*",     r"<i>\1</i>", raw)
    raw = re.sub(r"---+",          "",            raw)
    raw = re.sub(r"`([^`]+)`",     r"\1",         raw)

    for p in [x.strip() for x in raw.split("\n\n") if x.strip()]:
        clean = re.sub(r"<[^>]+>", "", p)
        if p.startswith("<b>") and len(clean) < 80:
            story.append(Paragraph(p, S["ai_h"]))
        else:
            story.append(Paragraph(p.replace("\n", " "), S["ai_body"]))
    story.append(Spacer(1, 0.35*cm))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION — Risk Metrics
# ─────────────────────────────────────────────────────────────────────────────

def _sec_risk(story, d, S, num):
    _sec_header(story, num, "Portfolio Risk Metrics", S)
    m = d["metrics"]
    so = not np.isnan(m.get("sortino", float("nan")))
    _metric_tiles(story, [
        ("Annualised Volatility", f"{m['volatility_pct']:.2f}%",  AMBER),
        ("Sharpe Ratio",          f"{m['sharpe']:.2f}",
         GREEN if m["sharpe"] >= 1 else AMBER),
        ("Sortino Ratio",
         f"{m['sortino']:.2f}" if so else "N/A",
         GREEN if so and m["sortino"] >= 1 else AMBER),
        ("Max Drawdown",          f"{m['max_drawdown_pct']:.2f}%", RED),
    ], S)

    if d.get("dd_peak") and d.get("dd_trough"):
        story.append(Paragraph(
            f"Max Drawdown:  <b>Peak {d['dd_peak']}</b>  →  <b>Trough {d['dd_trough']}</b>",
            S["small"]))
        story.append(Spacer(1, 0.12*cm))

    # Two charts side by side
    rc = _chart_risk_bar(m)
    vc = _chart_var(m)
    story.append(Table([[rc, vc]], colWidths=[CONTENT*0.55, CONTENT*0.45],
        style=TableStyle([("VALIGN",(0,0),(-1,-1),"TOP"),
                          ("LEFTPADDING",(0,0),(-1,-1),0),
                          ("RIGHTPADDING",(0,0),(-1,-1),0)])))
    story.append(Spacer(1, 0.2*cm))

    # Tail risk table
    story.append(Paragraph("Tail Risk Detail", S["h3"]))
    S_th, S_tc = S["tbl_hdr"], S["tbl_cell"]
    aw = CONTENT
    trows = [
        [_cell("Metric",S_th), _cell("Value",S_th), _cell("Interpretation",S_th)],
        [_cell("VaR (95%)",S_tc), _cell(f"{m['var_95_pct']:.2f}%",S_tc),
         _cell("Worst daily loss exceeded only 5% of trading days",S_tc)],
        [_cell("VaR (99%)",S_tc), _cell(f"{m['var_99_pct']:.2f}%",S_tc),
         _cell("Worst daily loss exceeded only 1% of trading days",S_tc)],
        [_cell("CVaR (95%)",S_tc), _cell(f"{m['cvar_95_pct']:.2f}%",S_tc),
         _cell("Avg loss on worst 5% of days (tail severity)",S_tc)],
        [_cell("CVaR (99%)",S_tc), _cell(f"{m['cvar_99_pct']:.2f}%",S_tc),
         _cell("Avg loss on worst 1% of days (extreme tail)",S_tc)],
    ]
    story.append(Table(trows, colWidths=[aw*0.20, aw*0.15, aw*0.65],
                       style=_base_ts(), repeatRows=1))
    story.append(Spacer(1, 0.35*cm))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION — Benchmark
# ─────────────────────────────────────────────────────────────────────────────

def _sec_benchmark(story, d, S, num):
    _sec_header(story, num, "Benchmark Comparison", S)
    b = d["benchmark"]
    ac = GREEN if b["alpha"] >= 0 else RED
    _metric_tiles(story, [
        ("Portfolio Return",       f"{b['port_total']*100:.2f}%",
         GREEN if b["port_total"] > 0 else RED),
        (f"{b['label']} Return",   f"{b['bench_total']*100:.2f}%", NAVY),
        ("Alpha",                   f"{b['alpha']:.2f}%", ac),
        ("Beta",                    f"{b['beta']:.2f}",
         AMBER if abs(b["beta"]-1) > 0.3 else GREEN),
    ], S)
    story.append(_chart_bench(b))
    story.append(Paragraph(
        f"<b>{b['label']}</b>  |  "
        f"{'Outperformed' if b['alpha']>=0 else 'Underperformed'} by "
        f"<b>{abs(b['alpha']):.2f}%</b>.  "
        f"Beta {b['beta']:.2f} → ~{b['beta']:.2f}× benchmark daily swing.",
        S["small"]))
    story.append(Spacer(1, 0.35*cm))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION — Monte Carlo
# ─────────────────────────────────────────────────────────────────────────────

def _sec_mc(story, d, S, num):
    _sec_header(story, num, "Monte Carlo Simulation", S)
    mc = d["monte_carlo"]
    _metric_tiles(story, [
        ("5th Pct. — Bad",     f"{mc['p5']:.2f}×",  RED),
        ("50th Pct. — Median", f"{mc['p50']:.2f}×", AMBER),
        ("95th Pct. — Good",   f"{mc['p95']:.2f}×", GREEN),
    ], S)
    story.append(_chart_mc(mc))
    story.append(Paragraph(
        f"Horizon: <b>{mc['days']} trading days</b>.  "
        f"90% confidence band: <b>{mc['p5']:.2f}×</b> → <b>{mc['p95']:.2f}×</b>.",
        S["small"]))
    story.append(Spacer(1, 0.35*cm))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION — Optimizer
# ─────────────────────────────────────────────────────────────────────────────

def _sec_optimizer(story, d, S, num):
    _sec_header(story, num, "Optimizer Results", S)
    opt = d["optimizer"]
    aw  = CONTENT
    S_th, S_tc = S["tbl_hdr"], S["tbl_cell"]

    rows = [
        [_cell("Metric",S_th),         _cell("Manual",S_th),  _cell("Optimized",S_th)],
        [_cell("Expected Return",S_tc), _cell(f"{opt['manual_ret']*100:.2f}%",S_tc),
         _cell(f"{opt['opt_ret']*100:.2f}%",S_tc)],
        [_cell("Volatility",S_tc),      _cell(f"{opt['manual_vol']*100:.2f}%",S_tc),
         _cell(f"{opt['opt_vol']*100:.2f}%",S_tc)],
        [_cell("Sharpe Ratio",S_tc),    _cell(f"{opt['manual_sharpe']:.3f}",S_tc),
         _cell(f"{opt['opt_sharpe']:.3f}",S_tc)],
    ]
    story.append(Table(rows, colWidths=[aw*0.36, aw*0.32, aw*0.32],
                       style=_base_ts(), repeatRows=1))
    story.append(Spacer(1, 0.15*cm))
    story.append(_chart_opt(opt))
    story.append(Spacer(1, 0.15*cm))

    story.append(Paragraph("Optimized Weights", S["h3"]))
    wrows = [[_cell("Ticker",S_th), _cell("Company",S_th), _cell("Opt. Weight",S_th)]]
    for tnm, nm, wv in zip(d["tickers_list"], d["ticker_names_list"],
                            opt["opt_weights_pct"]):
        nm_s = (nm[:24]+"…") if len(nm)>24 else nm
        wrows.append([_cell(tnm,S_tc), _cell(nm_s,S_tc), _cell(f"{wv:.2f}%",S_tc)])
    if d.get("cash_pct",0) > 0:
        wrows.append([_cell("CASH",S_tc), _cell("Cash (Risk-Free)",S_tc),
                      _cell(f"{d['cash_pct']*100:.2f}%",S_tc)])
    story.append(Table(wrows, colWidths=[aw*0.22, aw*0.57, aw*0.21],
                       style=_base_ts(), repeatRows=1))
    story.append(Spacer(1, 0.35*cm))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION — Efficient Frontier
# ─────────────────────────────────────────────────────────────────────────────

def _sec_frontier(story, d, S, num):
    _sec_header(story, num, "Efficient Frontier", S)
    fr  = d["frontier"]
    cur = fr["current"]
    pc  = GREEN if fr["position"]=="on" else (AMBER if fr["position"]=="above" else RED)
    _metric_tiles(story, [
        ("Current Return",     f"{cur['return']*100:.2f}%",     NAVY),
        ("Current Volatility", f"{cur['volatility']*100:.2f}%", AMBER),
        ("Current Sharpe",     f"{cur['sharpe']:.3f}",           NAVY),
        ("Frontier Position",  fr["position"].capitalize(),      pc),
    ], S)

    if fr.get("top5") is not None and not fr["top5"].empty:
        story.append(Paragraph("Top 5 Portfolios by Sharpe", S["h3"]))
        aw = CONTENT
        S_th, S_tc = S["tbl_hdr"], S["tbl_cell"]
        t5r = [[_cell("Return (%)",S_th), _cell("Volatility (%)",S_th),
                _cell("Sharpe",S_th)]]
        for _, row in fr["top5"].iterrows():
            t5r.append([_cell(f"{row['return']*100:.2f}%",S_tc),
                        _cell(f"{row['volatility']*100:.2f}%",S_tc),
                        _cell(f"{row['sharpe']:.4f}",S_tc)])
        story.append(Table(t5r, colWidths=[aw/3]*3,
                           style=_base_ts(), repeatRows=1))
    story.append(Spacer(1, 0.35*cm))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION — Risk Attribution (MCTR)
# ─────────────────────────────────────────────────────────────────────────────

def _sec_risk_attr(story, d, S, num):
    _sec_header(story, num, "Risk Attribution (MCTR)", S)
    mc = d["mctr"]
    aw = CONTENT
    _metric_tiles(story, [
        ("Equity Volatility",    f"{mc['equity_vol']:.2f}%", AMBER),
        ("Diversification Ratio", f"{mc['div_ratio']:.2f}×", GREEN),
    ], S)

    img = _chart_mctr(mc["df"])
    if img:
        story.append(img)
        story.append(Spacer(1, 0.15*cm))

    story.append(Paragraph("Full MCTR Table", S["h3"]))
    S_th, S_tc = S["tbl_hdr"], S["tbl_cell"]
    # FIX: Rebalanced - Company name gets 42% to prevent overlap
    cws = [aw*0.42, aw*0.12, aw*0.14, aw*0.15, aw*0.17]
    mrows = [[_cell("Company",S_th), _cell("Ticker",S_th), _cell("Weight",S_th),
              _cell("MCTR (%)",S_th), _cell("Risk Contrib.",S_th)]]
    for _, row in mc["df"].iterrows():
        mctr_v = f"{row['MCTR (%)']:.4f}" if row["Ticker"]!="CASH" else "—"
        nm = str(row["Company"])
        nm = (nm[:30]+"…") if len(nm)>30 else nm
        mrows.append([_cell(nm,S_tc), _cell(str(row["Ticker"]),S_tc),
                      _cell(f"{row['Weight (%)']:.2f}%",S_tc),
                      _cell(mctr_v,S_tc),
                      _cell(f"{row['% Risk Contribution']:.2f}%",S_tc)])
    story.append(Table(mrows, colWidths=cws, style=_base_ts(), repeatRows=1))
    story.append(Spacer(1, 0.35*cm))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION — Return Attribution
# ─────────────────────────────────────────────────────────────────────────────

def _sec_return_attr(story, d, S, num):
    _sec_header(story, num, "Active Return Attribution", S)
    ra = d["return_attr"]
    aw = CONTENT
    ac = GREEN if ra["active_return"] >= 0 else RED
    _metric_tiles(story, [
        ("Portfolio Return", f"{ra['portfolio_return']:.2f}%",
         GREEN if ra["portfolio_return"] >= 0 else RED),
        ("Benchmark Return", f"{ra['benchmark_return']:.2f}%", NAVY),
        ("Active Return",    f"{ra['active_return']:.2f}%",    ac),
        ("Boosters / Drags", f"{ra['num_boosters']} / {ra['num_drags']}", NAVY),
    ], S)

    img = _chart_contrib(ra["df"])
    if img:
        story.append(img)
        story.append(Spacer(1, 0.15*cm))

    story.append(Paragraph("Full Attribution Table", S["h3"]))
    S_th, S_tc = S["tbl_hdr"], S["tbl_cell"]
    # FIX: Rebalanced ratios for better spacing
    cws = [aw*0.40, aw*0.12, aw*0.15, aw*0.15, aw*0.18]
    arows = [[_cell("Company",S_th), _cell("Weight",S_th), _cell("Stock Ret.",S_th),
              _cell("Excess Ret.",S_th), _cell("Active Contrib.",S_th)]]
    ts_a = _base_ts()
    for i, row in enumerate(ra["df"].itertuples(), start=1):
        ticker = str(row.Ticker)
        nm = str(row.Company)
        nm = (nm[:30]+"…") if len(nm)>30 else nm
        # columns by position: index(0) Company(1) Ticker(2) Wt(3) SR(4) ExcR(5) AC(6)
        try:
            wt_v  = f"{float(row[3]):.2f}%"
        except Exception:
            wt_v  = str(row[3])
        try:
            sr_v  = f"{float(row[4]):.2f}%" if ticker not in ("—",) else "—"
        except Exception:
            sr_v  = str(row[4]) if ticker not in ("—",) else "—"
        try:
            ex_v  = f"{float(row[5]):.2f}%" if ticker not in ("—","CASH") else "—"
        except Exception:
            ex_v  = str(row[5]) if ticker not in ("—","CASH") else "—"
        try:
            ac_v  = float(row[6])
            ac_str = f"{ac_v:.4f}%"
            ac_clr = GREEN if ac_v >= 0 else RED
        except Exception:
            ac_str = str(row[6])
            ac_clr = BLACK

        ac_style = ParagraphStyle(f"ac{i}", parent=S_tc,
                                  textColor=ac_clr, alignment=TA_RIGHT)
        arows.append([_cell(nm,S_tc), _cell(wt_v,S_tc), _cell(sr_v,S_tc),
                      _cell(ex_v,S_tc),
                      KeepInFrame(0, 0.8*cm, [Paragraph(ac_str, ac_style)], mode="shrink")])
        if ticker == "—":
            ts_a.add("FONTNAME",   (0,i),(-1,i), "Helvetica-Bold")
            ts_a.add("BACKGROUND", (0,i),(-1,i), LGRAY)

    story.append(Table(arows, colWidths=cws, style=ts_a, repeatRows=1))
    story.append(Spacer(1, 0.35*cm))


# ─────────────────────────────────────────────────────────────────────────────
# DISCLAIMER
# ─────────────────────────────────────────────────────────────────────────────

def _disclaimer(story, S):
    story.append(HRFlowable(width="100%", thickness=0.4, color=LGRAY,
                             spaceBefore=8, spaceAfter=6))
    story.append(Paragraph(
        "DISCLAIMER: This report is for educational purposes only and does not "
        "constitute financial advice. All analysis is based on historical data. "
        "Past performance is not indicative of future results. Consult a qualified "
        "financial advisor before making investment decisions. "
        "AI-generated sections may contain errors and should be independently verified.",
        S["disclaimer"]))


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def generate_portfolio_pdf(data: dict) -> bytes:
    """
    Build the full PDF and return raw bytes.
    """
    buf = io.BytesIO()
    S   = _styles()

    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN,  bottomMargin=1.9*cm,
        title="Portfolio Analytics Report",
        author="Portfolio Analytics Dashboard — HSTG@FT",
    )

    story = []
    sec   = 1

    _cover(story, data, S)

    # 1. Overview
    _sec_overview(story, data, S, sec); sec += 1

    # 2. AI description (if available)
    if data.get("ai_analysis"):
        _sec_ai(story, data, S, sec); sec += 1

    # 3+ remaining sections
    _sec_risk(story, data, S, sec); sec += 1

    if data.get("benchmark"):
        _sec_benchmark(story, data, S, sec); sec += 1

    if data.get("monte_carlo"):
        _sec_mc(story, data, S, sec); sec += 1

    if data.get("optimizer"):
        _sec_optimizer(story, data, S, sec); sec += 1

    if data.get("frontier"):
        _sec_frontier(story, data, S, sec); sec += 1

    if data.get("mctr"):
        _sec_risk_attr(story, data, S, sec); sec += 1

    if data.get("return_attr"):
        _sec_return_attr(story, data, S, sec); sec += 1

    _disclaimer(story, S)

    doc.build(story, onFirstPage=_page_template, onLaterPages=_page_template)
    buf.seek(0)
    return buf.read()