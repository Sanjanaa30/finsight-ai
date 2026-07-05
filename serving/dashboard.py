"""FinSight AI -- Streamlit dashboard (fintech-terminal design, 4 tabs over the FastAPI backend).

Run:  streamlit run serving/dashboard.py     (API must be up: uvicorn serving.api:app --port 8000)

Design system
-------------
- Type:   Space Grotesk (display) / Inter (UI) / IBM Plex Mono (all numbers, tabular figures)
- Color:  cool near-white surfaces, deep ink, one indigo accent, disciplined green/red/amber semantics
- Signature: every number is monospaced and tabular, so columns align like a real terminal
"""

import html
from datetime import datetime, timezone

import httpx
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

API = "http://localhost:8000"

# ---- Design tokens (kept in sync with the CSS variables below) ----------------
ACCENT = "#3a57e8"
GREEN, RED, AMBER = "#1a8a55", "#cd3a4b", "#b5781c"
INK, MUTED, LINE = "#15171c", "#6a707c", "#e8eaee"
MONO = "IBM Plex Mono, ui-monospace, SFMono-Regular, Menlo, monospace"
SANS = "Inter, system-ui, -apple-system, Segoe UI, sans-serif"

st.set_page_config(page_title="FinSight AI", page_icon="📈", layout="wide",
                   initial_sidebar_state="expanded")

pio.templates["finsight"] = go.layout.Template(layout=dict(
    font=dict(family=SANS, color=INK, size=13),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    colorway=[ACCENT, "#0c8599", AMBER, GREEN, RED, "#6741d9"],
    xaxis=dict(gridcolor=LINE, zerolinecolor=LINE, linecolor=LINE,
               tickfont=dict(family=MONO, size=11, color=MUTED)),
    yaxis=dict(gridcolor=LINE, zerolinecolor=LINE, linecolor=LINE,
               tickfont=dict(family=MONO, size=11, color=MUTED)),
    legend=dict(font=dict(size=12), bgcolor="rgba(0,0,0,0)"),
    margin=dict(t=20, b=10, l=10, r=10),
))
pio.templates.default = "plotly_white+finsight"


def inject_css() -> None:
    st.markdown("""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Space+Grotesk:wght@500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

      :root {
        --bg:#f6f7f9; --surface:#ffffff; --ink:#15171c; --muted:#454b54; --faint:#666c77;
        --line:#e8eaee; --line-2:#dcdfe6; --accent:#3a57e8; --accent-soft:#eef1fe;
        --pos:#1a8a55; --pos-bg:#e7f4ec; --pos-line:#cfe8d8;
        --neg:#cd3a4b; --neg-bg:#fbecee; --neg-line:#f2d3d8;
        --amb:#b5781c; --amb-bg:#faf1e1; --amb-line:#eedfc2;
        --r:14px; --shadow:0 1px 2px rgba(20,23,28,.05), 0 1px 1px rgba(20,23,28,.03);
        --mono:'IBM Plex Mono', ui-monospace, SFMono-Regular, Menlo, monospace;
      }
      .stApp { background:var(--bg); }
      html, body, [class*="st-"], .stMarkdown, p, span, div, label, input, button {
        font-family:'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif; }
      .block-container { padding:1.5rem 2.6rem 2.4rem; max-width:1560px; }
      /* Sidebar: force it open + non-collapsible so it never hides behind the
         (removed) header toggle. */
      section[data-testid="stSidebar"] { width:236px !important; min-width:236px !important;
        transform:none !important; visibility:visible !important;
        background:var(--surface); border-right:1px solid var(--line); }
      [data-testid="stSidebarCollapseButton"], [data-testid="collapsedControl"] { display:none !important; }
      #MainMenu, footer, [data-testid="stToolbar"] { visibility:hidden; }
      [data-testid="stHeader"] { display:none; height:0; }
      [data-testid="stAppViewContainer"] > .main { padding-top:0; }

      h1,h2,h3 { font-family:'Space Grotesk', sans-serif; font-weight:600;
        letter-spacing:-0.015em; color:var(--ink); }
      h3 { font-size:1.04rem; margin:.4rem 0 .2rem; }
      .num { font-family:var(--mono); font-variant-numeric:tabular-nums; }
      .asset-hero { display:flex; align-items:baseline; gap:.55rem; flex-wrap:wrap;
        margin:2px 0 16px; padding-bottom:12px; border-bottom:1px solid var(--line); }
      .asset-hero .ah-tk { font-family:'Space Grotesk',sans-serif; font-size:1.6rem;
        font-weight:700; letter-spacing:-.02em; color:var(--ink); }
      .asset-hero .ah-nm { font-size:.98rem; color:var(--muted); }
      .asset-hero .ah-cls { font-size:.68rem; text-transform:uppercase; letter-spacing:.08em;
        color:var(--accent); background:var(--accent-soft); padding:2px 9px; border-radius:999px;
        align-self:center; }
      .chart-summary { font-size:1rem; color:var(--ink); line-height:1.55; margin:2px 0 14px; }
      /* Darken Streamlit's faint captions so explainer text is clearly readable */
      [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p,
      [data-testid="stCaptionContainer"] span, [data-testid="stCaptionContainer"] * {
        color:#1c2027 !important; font-size:.86rem !important; }
      .range-wrap { margin:14px 2px 2px; }
      .range-head { font-size:.96rem; color:var(--ink); line-height:1.5; margin-bottom:36px; }
      .range-track { position:relative; height:12px; border-radius:999px; background:var(--line-2);
        margin:0 8px; }
      .range-fill { position:absolute; left:0; top:0; height:100%; border-radius:999px;
        background:var(--accent); opacity:.32; }
      .range-dot { position:absolute; top:50%; width:18px; height:18px; border-radius:50%;
        background:var(--accent); border:3px solid #fff; box-shadow:0 1px 3px rgba(20,23,28,.28);
        transform:translate(-50%,-50%); }
      .range-now { position:absolute; top:-27px; transform:translateX(-50%); font-family:var(--mono);
        font-size:.8rem; color:var(--accent); white-space:nowrap; }
      .range-ends { display:flex; justify-content:space-between; margin-top:14px;
        font-size:.78rem; color:var(--muted); }
      .range-ends b { color:var(--ink); font-family:var(--mono); }

      /* header */
      .fs-head { display:flex; align-items:center; justify-content:space-between;
        padding:0 2px 16px; border-bottom:1px solid var(--line); margin-bottom:18px; }
      .fs-brand { display:flex; align-items:center; gap:11px; }
      .fs-brand .mark { width:30px; height:30px; border-radius:9px; background:var(--accent);
        display:flex; align-items:center; justify-content:center; }
      .fs-brand .nm { font-family:'Space Grotesk',sans-serif; font-size:1.32rem;
        font-weight:700; letter-spacing:-0.025em; color:var(--ink); }
      .fs-brand .tag { color:var(--faint); font-size:.78rem; margin-left:9px;
        padding-left:11px; border-left:1px solid var(--line-2); }
      .live { display:inline-flex; align-items:center; gap:7px; font-size:.78rem;
        color:var(--muted); }
      .live .dot { width:7px; height:7px; border-radius:50%; background:var(--pos);
        box-shadow:0 0 0 0 rgba(26,138,85,.45); animation:pulse 2.2s infinite; }
      @keyframes pulse { 0%{box-shadow:0 0 0 0 rgba(26,138,85,.40);}
        70%{box-shadow:0 0 0 6px rgba(26,138,85,0);} 100%{box-shadow:0 0 0 0 rgba(26,138,85,0);} }

      /* ticker tape */
      .tape { display:flex; flex-wrap:wrap; align-items:center; gap:0 6px;
        background:var(--surface); border:1px solid var(--line); border-radius:var(--r);
        padding:11px 8px; margin-bottom:18px; box-shadow:var(--shadow); }
      .tape .it { display:inline-flex; align-items:baseline; gap:8px; padding:2px 18px;
        font-family:var(--mono); font-variant-numeric:tabular-nums; font-size:.9rem;
        border-right:1px solid var(--line); }
      .tape .it:last-child { border-right:none; }
      .tape .sym { color:var(--ink); font-weight:500; }
      .tape .px { color:var(--muted); }

      /* KPI cards */
      .kpis { display:flex; flex-wrap:wrap; gap:14px; margin-bottom:20px; }
      .kpi { flex:1 1 210px; min-width:184px; background:var(--surface);
        border:1px solid var(--line); border-radius:16px; padding:15px 18px 16px;
        box-shadow:var(--shadow); }
      .kpi .lab { color:var(--muted); font-size:.82rem; }
      .kpi .val { font-family:var(--mono); font-variant-numeric:tabular-nums;
        font-size:1.78rem; font-weight:500; letter-spacing:-0.02em; color:var(--ink);
        margin:5px 0 3px; line-height:1.1; }
      .kpi .sub { font-size:.82rem; color:var(--muted); font-family:var(--mono); }
      .kpi .sub.pos{color:var(--pos);} .kpi .sub.neg{color:var(--neg);} .kpi .sub.amb{color:var(--amb);}
      .kpi .val.neg{color:var(--neg);} .kpi .val.amb{color:var(--amb);}

      /* chart card via st.container(border=True) */
      div[data-testid="stVerticalBlockBorderWrapper"] { background:var(--surface);
        border:1px solid var(--line); border-radius:18px; padding:10px 18px 16px;
        box-shadow:var(--shadow); }
      .card-head { display:flex; justify-content:space-between; align-items:center;
        margin:6px 2px 4px; font-size:1rem; font-weight:600; font-family:'Space Grotesk',sans-serif;
        letter-spacing:-0.01em; }
      .card-head .muted { color:var(--muted); font-weight:400; font-size:.8rem; font-family:var(--mono); }
      .axis-row { display:flex; justify-content:space-between; color:var(--ink);
        font-size:.82rem; font-weight:500; font-family:var(--mono); padding:2px 6px 0; }

      /* sentiment pills with mini gauge */
      .pills { display:flex; flex-wrap:wrap; gap:11px; margin:8px 0 20px; }
      .pill { flex:1 1 150px; min-width:132px; border-radius:14px; padding:13px 16px 14px;
        background:var(--surface); border:1px solid var(--line); box-shadow:var(--shadow); }
      .pill .pl { display:flex; align-items:center; justify-content:space-between;
        font-size:.84rem; color:var(--muted); }
      .pill .pi { position:relative; display:inline-flex; align-items:center; justify-content:center;
        width:18px; height:18px; border-radius:50%; border:1.5px solid var(--muted); color:var(--muted);
        font-size:.72rem; font-weight:700; font-style:normal; line-height:1; cursor:help; flex:none; }
      .pill .pi:hover { color:#fff; background:var(--ink); border-color:var(--ink); }
      .pill .pi:hover::after { content:attr(data-tip); position:absolute; top:138%; right:0; z-index:60;
        width:244px; background:#1c2027; color:#fff; font-size:.82rem; font-weight:400; line-height:1.55;
        padding:11px 13px; border-radius:9px; box-shadow:0 8px 22px rgba(15,17,20,.30); white-space:normal;
        text-align:left; letter-spacing:0; font-family:'Inter',sans-serif; }
      .pill .ps { font-family:var(--mono); font-variant-numeric:tabular-nums;
        font-size:1.5rem; font-weight:500; line-height:1.2; }
      .pill .bar { height:4px; border-radius:99px; background:var(--line); margin-top:9px; }
      .pill .bar i { display:block; height:100%; border-radius:99px; }
      .pill.pos .ps{color:var(--pos);} .pill.pos .bar i{background:var(--pos);}
      .pill.neg .ps{color:var(--neg);} .pill.neg .bar i{background:var(--neg);}
      .pill.neu .ps{color:var(--amb);} .pill.neu .bar i{background:var(--amb);}
      .pill .pn { font-size:.68rem; color:var(--muted); margin-top:8px; }
      .pill.low { opacity:.5; }
      .sent-overall { display:flex; flex-wrap:wrap; gap:4px 16px; align-items:baseline;
        font-size:.95rem; color:var(--ink); margin:2px 0 12px; }
      .sent-overall span { font-size:.82rem; color:var(--muted); }
      .pill .psrow { display:flex; align-items:baseline; gap:8px; }
      .tr { font-family:var(--mono); font-size:.68rem; font-weight:500; white-space:nowrap; }
      .tr.up { color:var(--pos); } .tr.dn { color:var(--neg); } .tr.flat { color:var(--faint); }
      .mood-box { padding:6px 2px; }
      .mood-lab { font-size:.82rem; color:var(--muted); }
      .mood-val { font-family:'Space Grotesk',sans-serif; font-size:1.7rem; font-weight:700;
        margin:2px 0 8px; }
      .mood-sub { font-size:.92rem; color:var(--ink); line-height:1.75; }
      .sec-banner { display:flex; align-items:center; gap:11px; margin:34px 0 16px; padding:14px 20px;
        border-radius:13px; font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:1.2rem;
        letter-spacing:-.015em; }
      .sec-banner .sb-ic { font-size:1.35rem; }
      .sb-asset { background:var(--accent-soft); color:var(--accent); border:1px solid #d7dffb;
        border-left:5px solid var(--accent); }
      .sb-market { background:#edf6f0; color:#157a44; border:1px solid #cfeadb;
        border-left:5px solid #1f9d57; }
      /* Keep Streamlit's Material icons (e.g. expander arrows) from inheriting Inter */
      [data-testid="stIconMaterial"], [data-testid="stExpanderToggleIcon"],
      .material-icons, .material-icons-outlined {
        font-family:'Material Symbols Rounded','Material Symbols Outlined','Material Icons' !important; }
      [data-testid="stExpander"] summary { color:var(--ink); font-weight:500; }

      /* alert + headlines */
      .alert { display:flex; gap:11px; align-items:flex-start; background:var(--amb-bg);
        border:1px solid var(--amb-line); color:#7a5410; border-radius:14px;
        padding:14px 18px; margin:16px 0 4px; font-size:.94rem; }
      .alert .ic { color:var(--amb); font-size:1.05rem; line-height:1.3; }
      .alert .num { color:inherit; font-weight:600; }
      .alert.strong { background:var(--neg-bg); border-color:var(--neg-line); color:#8a2533; }
      .alert.strong .ic { color:var(--neg); }
      .hl { border:1px solid var(--line); border-radius:16px; overflow:hidden;
        background:var(--surface); box-shadow:var(--shadow); }
      .hl-row { display:flex; align-items:flex-start; gap:13px; padding:13px 17px;
        border-bottom:1px solid var(--line); text-decoration:none; color:inherit; transition:background .12s; }
      .hl-row:last-child { border-bottom:none; }
      .hl-row:hover { background:#f4f6fb; }
      .hl-row.mention { background:var(--accent-soft); }
      .hl-row.mention:hover { background:#e6ebfe; }
      .badge { font-size:.72rem; font-weight:500; padding:3px 9px; border-radius:7px;
        white-space:nowrap; letter-spacing:.01em; font-family:var(--mono); margin-top:1px; }
      .badge.pos{background:var(--pos-bg);color:var(--pos);}
      .badge.neg{background:var(--neg-bg);color:var(--neg);}
      .badge.neu{background:var(--amb-bg);color:var(--amb);}
      .hl-main { display:flex; flex-direction:column; gap:3px; }
      .hl-title { font-size:.94rem; color:var(--ink); font-weight:500; line-height:1.35; }
      .hl-row:hover .hl-title { color:var(--accent); }
      .hl-meta { font-size:.76rem; color:var(--muted); font-family:var(--mono);
        display:flex; align-items:center; gap:7px; flex-wrap:wrap; }
      .hl-cat { background:var(--line); color:var(--muted); padding:1px 7px; border-radius:6px;
        text-transform:uppercase; font-size:.64rem; letter-spacing:.04em; }
      .hl-star { color:var(--accent); font-weight:600; }

      /* tabs */
      .stTabs [data-baseweb="tab-list"] { gap:2px; border-bottom:1px solid var(--line); }
      .stTabs [data-baseweb="tab"] { border-radius:9px 9px 0 0; padding:9px 18px;
        font-weight:500; color:var(--muted); }
      .stTabs [aria-selected="true"] { color:var(--accent); background:var(--accent-soft); }
      .stTabs [data-baseweb="tab-highlight"] { background:var(--accent); }

      /* controls */
      .stButton button { border-radius:10px; font-weight:500; border:1px solid var(--line-2); }
      .stButton button:hover { border-color:var(--accent); color:var(--accent); }
      .stButton button[kind="primary"], .stButton button[data-testid="baseButton-primary"] {
        background:var(--accent); border-color:var(--accent); color:#fff; }
      .stButton button[kind="primary"]:hover, .stButton button[data-testid="baseButton-primary"]:hover {
        background:var(--accent); border-color:var(--accent); color:#fff; opacity:.92; }
      [data-testid="stMetricValue"] { font-family:var(--mono); font-variant-numeric:tabular-nums; }

      /* stack Streamlit columns (Forecast/News tabs) on tablets & phones */
      @media (max-width:900px) {
        [data-testid="stHorizontalBlock"] { flex-wrap:wrap; }
        [data-testid="stHorizontalBlock"] > [data-testid="stColumn"],
        [data-testid="stHorizontalBlock"] > [data-testid="column"] {
          flex:1 1 100% !important; min-width:100% !important; width:100% !important; }
      }
      @media (max-width:520px) {
        .block-container{ padding:1.1rem 1.1rem 2rem; }
        .fs-brand .nm{ font-size:1.12rem; } .kpi .val{ font-size:1.45rem; }
        .fs-brand .tag, .live{ display:none; }
      }
    </style>
    """, unsafe_allow_html=True)


inject_css()


@st.cache_data(ttl=300)
def api_get(path: str, **params) -> dict:
    r = httpx.get(API + path, params=params, timeout=120)
    r.raise_for_status()
    return r.json()


def api_post(path: str, payload: dict) -> dict:
    r = httpx.post(API + path, json=payload, timeout=180)
    r.raise_for_status()
    return r.json()


def _mdsafe(s: str) -> str:
    """Escape $ so Streamlit markdown doesn't treat price pairs as LaTeX math."""
    return (s or "").replace("$", "\\$")


def report_html(title: str, md_text: str) -> str:
    """Wrap a markdown report in a clean, self-contained, print-ready HTML document."""
    import markdown as _md
    body = _md.markdown(md_text or "", extensions=["extra", "sane_lists", "nl2br"])
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>{html.escape(title)}</title>
<style>
  @page {{ margin: 20mm 18mm; }}
  body {{ font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; color:#1a1c22;
    line-height:1.55; max-width:820px; margin:24px auto; padding:0 18px; }}
  .fs-brand {{ font-weight:700; color:#3a57e8; letter-spacing:.02em; margin-bottom:2px; }}
  h1,h2,h3 {{ color:#111; line-height:1.25; }}
  h1 {{ font-size:1.6rem; border-bottom:2px solid #3a57e8; padding-bottom:6px; }}
  h2 {{ font-size:1.22rem; margin-top:1.4em; }}
  h3 {{ font-size:1.04rem; }}
  hr {{ border:none; border-top:1px solid #e2e2e2; margin:1.3em 0; }}
  blockquote {{ border-left:3px solid #3a57e8; margin:0; padding:2px 14px; color:#333; background:#f6f7fb; }}
  code {{ background:#f2f3f7; padding:1px 5px; border-radius:4px; }}
  .fs-foot {{ margin-top:2.4em; padding-top:10px; border-top:1px solid #eee; color:#8a8f99; font-size:.8rem; }}
  @media print {{ body {{ margin:0; }} }}
</style></head>
<body>
  <div class="fs-brand">FinSight AI</div>
  {body}
  <div class="fs-foot">Generated by FinSight AI · informational analysis, not investment advice.</div>
</body></html>"""


def report_pdf(title: str, md_text: str) -> bytes:
    """Render the markdown report straight to PDF bytes (via xhtml2pdf) for a one-click download."""
    import io
    from xhtml2pdf import pisa
    # Map a few glyphs the base PDF font can't draw to plain text.
    subs = {"↑": " (rising)", "↓": " (falling)", "→": " -> ", "↗": " (up)", "↘": " (down)"}
    for a, b in subs.items():
        md_text = (md_text or "").replace(a, b)
    buf = io.BytesIO()
    pisa.CreatePDF(src=report_html(title, md_text), dest=buf, encoding="utf-8")
    return buf.getvalue()


def _fmt(v: float) -> str:
    return f"{v / 1000:.1f}k" if abs(v) >= 10000 else f"{v:,.1f}"


def _to100(s: float) -> int:
    return int(round((s + 1) / 2 * 100))


# ---- HTML component builders --------------------------------------------------
def ticker_tape(quotes: list[dict]) -> str:
    items = []
    for q in quotes:
        ret = q["change_pct"]
        arrow, col = ("▲", GREEN) if ret >= 0 else ("▼", RED)
        items.append(f"<span class='it'><span class='sym'>{q['ticker']}</span>"
                     f"<span class='px'>{_fmt(q['price'])}</span>"
                     f"<span style='color:{col}'>{arrow} {abs(ret):.1f}%</span></span>")
    return f"<div class='tape'>{''.join(items)}</div>"


@st.fragment(run_every="3h")
def live_tape() -> None:
    """Ticker tape -- reruns itself every 3 hours to pull fresh quotes."""
    try:
        resp = httpx.get(API + "/live", timeout=45).json()
        quotes = resp.get("quotes", [])
        fetched = resp.get("fetched_at", "—")
    except Exception:  # noqa: BLE001
        quotes, fetched = [], "—"
    if not quotes:
        st.caption("Live quotes unavailable right now.")
        return
    # The API tracks the prior refresh time itself -> authoritative across viewers.
    prev = resp.get("previous_fetched_at")
    prev_txt = f"previous fetch {prev}" if prev else "(first refresh since the API started)"
    st.markdown(ticker_tape(quotes), unsafe_allow_html=True)
    st.caption(f"↻ Auto-refreshes every 3h  ·  data fetched {fetched}  ·  "
               f"{prev_txt}  ·  yfinance (~15-min delayed)")


def kpi_cards(ticker, latest, fc, n_anom, live_price=None, live_chg=None) -> str:
    if live_price is not None:  # live quote available -> price card shows real-time
        chg = live_chg or 0.0
        up = chg >= 0
        arrow = "▲" if up else "▼"
        rcls = "pos" if up else "neg"
        price_val, price_sub = (f"${live_price:,.2f}",
                                f"<span class='sub {rcls}'>{arrow} {abs(chg):.1f}% · live</span>")
    else:  # fall back to the stored last close
        ret = latest["daily_return"] * 100
        up = ret >= 0
        arrow = "▲" if up else "▼"
        rcls = "pos" if up else "neg"
        price_val, price_sub = (f"${latest['close']:,.2f}",
                                f"<span class='sub {rcls}'>{arrow} {abs(ret):.1f}% last close</span>")
    pf = fc["price_forecast"]
    central = pf[-1]["forecast"]
    lo, hi = min(p["low"] for p in pf), max(p["high"] for p in pf)
    # Annualized 20-day realized volatility -- a real, asset-specific risk number.
    vol = (latest.get("volatility_20d") or 0) * (252 ** 0.5) * 100
    vol_lab = "calm" if vol < 25 else ("moderate" if vol < 45 else "elevated")
    anom_cls = " neg" if n_anom else ""
    cards = [
        (f"{ticker} price", price_val, "", price_sub),
        ("Next 7-day price forecast predicted", f"${central:,.0f}", "",
         f"<span class='sub'>range {lo:,.0f}–{hi:,.0f}</span>"),
        ("Anomalies", f"{n_anom} flagged", anom_cls,
         "<span class='sub'>unusual days · last 1.5y</span>"),
        ("Volatility (20d)", f"{vol:.0f}%", " amb",
         f"<span class='sub amb'>{vol_lab} · annualized</span>"),
    ]
    body = "".join(f"<div class='kpi'><div class='lab'>{lab}</div>"
                   f"<div class='val{vc}'>{val}</div>{sub}</div>"
                   for lab, val, vc, sub in cards)
    return f"<div class='kpis'>{body}</div>"


@st.fragment(run_every="3h")
def live_kpis(ticker, latest, fc, n_anom) -> None:
    """KPI strip whose price card is LIVE (auto-refresh 3h). Other cards are as of last close."""
    try:
        q = httpx.get(API + f"/quote/{ticker}", timeout=45).json()
        lp, lc, ft = q.get("price"), q.get("change_pct"), q.get("fetched_at")
    except Exception:  # noqa: BLE001
        lp = lc = ft = None
    st.markdown(kpi_cards(ticker, latest, fc, n_anom, lp, lc), unsafe_allow_html=True)
    asof = str(latest.get("date", ""))[:10]
    if ft:
        st.caption(f"↻ Live · all metrics recomputed from fresh market data (close {asof}) · "
                   f"price fetched {ft} · auto-refreshes every 3h")
    else:
        st.caption(f"Live data unavailable — showing stored values (close {asof}).")


SENT_LABELS = {"market": "US Stocks", "geopolitical": "Geopolitical", "crypto": "Crypto",
               "macro": "Macro", "commodities": "Commodities", "country": "Country"}
SENT_ORDER = ["market", "geopolitical", "crypto", "macro", "commodities", "country"]
LOW_SAMPLE = 20  # fewer than this many articles -> score is noisy, so grey it out


def sentiment_pills(sent, deltas=None) -> str:
    deltas = deltas or {}
    out = []
    for k in SENT_ORDER:
        v = sent.get(k)
        if not v:
            continue
        score = _to100(v["avg_sentiment"])
        n = int(v.get("article_count", 0))
        cls = "pos" if score >= 58 else ("neg" if score <= 44 else "neu")
        conf = max(0.35, min(1.0, 0.35 + n / 60 * 0.65))     # bar opacity ~ how many articles
        d = deltas.get(k)                                    # change vs previous day (points)
        if n < LOW_SAMPLE or d is None or abs(d) < 0.5:      # too few articles -> don't trust a daily move
            arrow = "<span class='tr flat'>—</span>"
        elif d > 0:
            arrow = f"<span class='tr up'>▲ {abs(d):.0f}</span>"
        else:
            arrow = f"<span class='tr dn'>▼ {abs(d):.0f}</span>"
        note = f"{n} articles" + (" · low sample" if n < LOW_SAMPLE else "")
        mood_word = "upbeat" if score >= 58 else ("gloomy" if score <= 44 else "mixed")
        if n < LOW_SAMPLE:
            chg_txt = "Daily change hidden — too few articles to trust."
        elif d is None or abs(d) < 0.5:
            chg_txt = "No reliable daily change (news from a single day)."
        else:
            chg_txt = f"{'Up' if d > 0 else 'Down'} {abs(d):.0f} points vs yesterday."
        tip = (f"{SENT_LABELS[k]} news mood: {score}/100 ({mood_word}). "
               f"50 = neutral; higher = more positive news. {chg_txt} "
               f"Based on {n} headlines" + (" (low sample — less reliable)." if n < LOW_SAMPLE else "."))
        out.append(f"<div class='pill {cls}'>"
                   f"<div class='pl'>{SENT_LABELS[k]}<span class='pi' data-tip=\"{tip}\">i</span></div>"
                   f"<div class='psrow'><span class='ps'>{score}</span>{arrow}</div>"
                   f"<div class='bar'><i style='width:{score}%;opacity:{conf:.2f}'></i></div>"
                   f"<div class='pn'>{note}</div></div>")
    return f"<div class='pills'>{''.join(out)}</div>"


_MENTION_STOP = {"the", "and", "inc", "corp", "ltd", "group", "holdings", "platforms",
                 "index", "composite", "chase"}


def _mention_terms(ticker: str) -> set:
    """Distinctive words that signal a headline is about this asset (ticker + name)."""
    terms = set()
    base = ticker.lower().split("-")[0].split("=")[0].lstrip("^")
    if base.isalpha() and len(base) >= 2:
        terms.add(base)
    for w in ASSET_NAMES.get(ticker, "").lower().replace("(", " ").replace(")", " ").replace("·", " ").split():
        if w.isalpha() and len(w) >= 3 and w not in _MENTION_STOP:
            terms.add(w)
    return terms


def _ago(iso: str | None) -> str:
    """A coarse 'N h/d ago' from an ISO timestamp (falls back to '' if unparseable)."""
    if not iso:
        return ""
    try:
        dt = datetime.fromisoformat(str(iso).replace("Z", "+00:00"))
        secs = (datetime.now(timezone.utc) - dt).total_seconds()
    except Exception:  # noqa: BLE001
        return ""
    if secs < 3600:
        return f"{max(1, int(secs // 60))}m ago"
    if secs < 86400:
        return f"{int(secs // 3600)}h ago"
    return f"{int(secs // 86400)}d ago"


def pick_headlines(recent: list, ticker: str, n: int = 5) -> list:
    """The most *significant* headlines: asset mentions first, then strongest +/- sentiment."""
    terms = _mention_terms(ticker)
    mentions = [x for x in recent if any(t in x["title"].lower() for t in terms)][:2]
    by_score = sorted(recent, key=lambda x: x["sentiment_score"])
    strongest = by_score[:2] + list(reversed(by_score))[:2]   # 2 most negative + 2 most positive
    out, seen = [], set()
    for x in mentions + strongest:
        if x["title"] not in seen:
            seen.add(x["title"]); out.append(x)
    return out[:n]


def headlines(items, ticker: str) -> str:
    badge = {"positive": ("Pos", "pos"), "negative": ("Neg", "neg"), "neutral": ("Neu", "neu")}
    terms = _mention_terms(ticker)
    rows = []
    for n in items:
        txt, cls = badge.get(n["sentiment_label"], ("Neu", "neu"))
        title = html.escape(n["title"])
        url = html.escape(n.get("url") or "#", quote=True)
        src = html.escape(n.get("source_name") or "—")
        ago = _ago(n.get("published_at")) or (n.get("published_date") or "")
        cat = html.escape(n.get("category") or "")
        hit = any(t in n["title"].lower() for t in terms)
        star = f"<span class='hl-star'>★ mentions {ticker}</span>" if hit else ""
        rows.append(
            f"<a class='hl-row{' mention' if hit else ''}' href='{url}' target='_blank' rel='noopener'>"
            f"<span class='badge {cls}'>{txt} {n['sentiment_score']:+.2f}</span>"
            f"<span class='hl-main'><span class='hl-title'>{title}</span>"
            f"<span class='hl-meta'>{src} · {ago} · <span class='hl-cat'>{cat}</span>{star}</span>"
            f"</span></a>")
    return f"<div class='hl'>{''.join(rows)}</div>"


# ---- Overview tab -------------------------------------------------------------
ASSET_NAMES = {
    "NVDA": "NVIDIA", "AAPL": "Apple", "MSFT": "Microsoft", "AMZN": "Amazon",
    "GOOGL": "Alphabet (Google)", "META": "Meta Platforms", "TSLA": "Tesla",
    "JPM": "JPMorgan Chase", "XOM": "ExxonMobil", "NFLX": "Netflix", "AMD": "AMD",
    "BRK-B": "Berkshire Hathaway", "BTC-USD": "Bitcoin", "ETH-USD": "Ethereum",
    "SOL-USD": "Solana", "CL=F": "Crude Oil", "GC=F": "Gold", "SI=F": "Silver",
    "NG=F": "Natural Gas", "^NSEI": "Nifty 50 · India", "^FTSE": "FTSE 100 · UK",
    "^GDAXI": "DAX · Germany", "^N225": "Nikkei 225 · Japan", "^GSPC": "S&P 500",
    "^IXIC": "Nasdaq Composite",
}


def _asset_class(ticker: str) -> str:
    if ticker.endswith("-USD"):
        return "Crypto"
    if ticker.endswith("=F"):
        return "Commodity"
    if ticker.startswith("^"):
        return "Index"
    return "Equity"


def asset_header(ticker: str) -> None:
    """Prominent per-asset title so it's always obvious which asset a tab is showing."""
    st.markdown(
        f"<div class='asset-hero'><span class='ah-tk'>{ticker}</span>"
        f"<span class='ah-nm'>{ASSET_NAMES.get(ticker, '')}</span>"
        f"<span class='ah-cls'>{_asset_class(ticker)}</span></div>",
        unsafe_allow_html=True)


def section_banner(text: str, scope: str = "asset") -> None:
    """A coloured divider that flags whether a section is asset-specific or market-wide."""
    icon = "🌍" if scope == "market" else "📊"
    st.markdown(f"<div class='sec-banner sb-{scope}'><span class='sb-ic'>{icon}</span>"
                f"<span>{text}</span></div>", unsafe_allow_html=True)


def _sent_deltas() -> dict:
    """Per-category change in sentiment score vs the previous day (points on the 0-100 scale)."""
    try:
        trend = api_get("/sentiment/trend")["trend"]
    except Exception:  # noqa: BLE001
        return {}
    from collections import defaultdict
    series = defaultdict(list)
    for r in sorted(trend, key=lambda x: x["date"]):
        series[r["category"]].append(r["avg_sentiment"])
    return {c: (s[-1] - s[-2]) * 50 for c, s in series.items() if len(s) >= 2}


def render_overview(ticker: str) -> None:
    asset_header(ticker)
    live_tape()

    price = api_get(f"/prices/{ticker}", days=90)
    latest = price["latest"]
    fc = api_get(f"/forecast/{ticker}")
    anoms = api_get(f"/anomalies/{ticker}")["anomalies"]
    sent = api_get("/sentiment").get("categories", {})

    section_banner(f"THIS ASSET · {ticker} — price, forecast & anomalies", "asset")
    live_kpis(ticker, latest, fc, len(anoms))

    with st.container(border=True):
        st.markdown(f"<div class='card-head'>{ticker} — price &amp; 7-day forecast"
                    f"<span class='muted'>Prophet model · typically ±{fc['price_mape_pct']}% off</span></div>",
                    unsafe_allow_html=True)
        hist = pd.DataFrame(price["history"]); hist["date"] = pd.to_datetime(hist["date"])
        pf = pd.DataFrame(fc["price_forecast"]); pf["date"] = pd.to_datetime(pf["date"])
        now = hist["date"].iloc[-1]
        closes = hist["close"]
        last = float(closes.iloc[-1]); first = float(closes.iloc[0])
        hi90 = float(closes.max()); lo90 = float(closes.min())
        chg90 = (last / first - 1) * 100
        from_hi = (last / hi90 - 1) * 100
        fc_vs = (float(pf["forecast"].iloc[-1]) / last - 1) * 100
        money = lambda v: (f"${v:,.0f}" if abs(v) >= 100 else f"${v:,.2f}")  # noqa: E731

        # (summary + %) A one-line plain-English read of the same data.
        nm = ASSET_NAMES.get(ticker, ticker)
        cc = GREEN if chg90 >= 0 else RED
        peak_bit = (f", <span style='color:{RED}'><b>{abs(from_hi):.1f}%</b></span> below its recent peak"
                    if from_hi < -0.5 else "")
        fcword = ("roughly flat" if abs(fc_vs) < 1 else
                  (f"slightly higher (+{fc_vs:.1f}%)" if fc_vs > 0 else f"slightly lower ({fc_vs:.1f}%)"))
        st.markdown(
            f"<div class='chart-summary'><b>{nm}</b> is at <b>{money(last)}</b> — "
            f"<span style='color:{cc}'><b>{'up' if chg90 >= 0 else 'down'} {abs(chg90):.1f}%</b></span> "
            f"over 90 days{peak_bit}. The model expects it <b>{fcword}</b> next week.</div>",
            unsafe_allow_html=True)

        # (area chart) actual (filled) + forecast (dashed) + likely-range band, WITH price labels.
        band_hi = float(pf["high"].max()); band_lo = float(pf["low"].min())
        yhi = max(hi90, band_hi); ylo = min(lo90, band_lo); pad = (yhi - ylo) * 0.12 or 1
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist["date"], y=closes, mode="lines", name="Actual price",
                                 line=dict(color=ACCENT, width=2.4), fill="tozeroy",
                                 fillcolor="rgba(58,87,232,0.10)",
                                 hovertemplate="%{x|%b %d}: $%{y:,.0f}<extra></extra>"))
        fig.add_trace(go.Scatter(x=[now] + list(pf["date"]), y=[last] + list(pf["forecast"]),
                                 mode="lines", name="Forecast (next 7 days)",
                                 line=dict(color=ACCENT, width=2.4, dash="dash"),
                                 hovertemplate="%{x|%b %d}: ~$%{y:,.0f}<extra></extra>"))
        fig.add_trace(go.Scatter(x=pf["date"], y=pf["high"], mode="lines", line=dict(width=0),
                                 showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=pf["date"], y=pf["low"], mode="lines", fill="tonexty",
                                 fillcolor="rgba(58,87,232,0.16)", line=dict(width=0),
                                 name="Likely range", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=[now], y=[last], mode="markers", showlegend=False,
                                 marker=dict(color=ACCENT, size=10, line=dict(color="white", width=2)),
                                 hovertemplate="now: $%{y:,.0f}<extra></extra>"))
        fig.add_annotation(x=now, y=last, text=f"now {money(last)}", showarrow=False, yshift=15,
                           font=dict(size=12, color=ACCENT, family="IBM Plex Mono"))
        fig.update_layout(height=320, margin=dict(t=30, b=4, l=6, r=10),
                          legend=dict(orientation="h", y=1.15, x=0,
                                      font=dict(size=12, color=INK), bgcolor="rgba(0,0,0,0)"),
                          xaxis=dict(showgrid=False, showline=False, ticks="", showticklabels=False),
                          yaxis=dict(range=[ylo - pad, yhi + pad], showgrid=True, gridcolor="#e6e9ef",
                                     tickprefix="$", tickformat=",.0f", ticks="", nticks=5,
                                     tickfont=dict(size=12, color=INK)))
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        st.markdown("<div class='axis-row'><span>3 months ago</span><span>today</span>"
                    "<span>next week (forecast)</span></div>", unsafe_allow_html=True)

        # (range meter) a fuel-gauge read of where "now" sits between the 90-day low & high.
        span = (hi90 - lo90) or 1
        pct = max(0.0, min(100.0, (last - lo90) / span * 100))
        where = ("near the low — the cheaper end" if pct < 33
                 else "around the middle" if pct < 67
                 else "near the high — the pricier end")
        st.markdown(
            f"<div class='range-wrap'>"
            f"<div class='range-head'>Right now {ticker} is <b>{where}</b> of its 90-day range "
            f"— about <b>{pct:.0f}%</b> of the way up from its low.</div>"
            f"<div class='range-track'>"
            f"<div class='range-fill' style='width:{pct}%'></div>"
            f"<div class='range-now' style='left:{pct}%'>now {money(last)}</div>"
            f"<div class='range-dot' style='left:{pct}%'></div></div>"
            f"<div class='range-ends'><span>90-day low<br><b>{money(lo90)}</b></span>"
            f"<span style='text-align:right'>90-day high<br><b>{money(hi90)}</b></span></div>"
            f"</div>", unsafe_allow_html=True)

    # Asset-specific: most recent flagged unusual day (stays with the asset section).
    if anoms:
        a = anoms[0]
        sc = float(a.get("anomaly_score", 0.0))
        strength, scls = (("mild", "") if sc > -0.03
                          else ("notable", "") if sc > -0.08 else ("strong", "strong"))
        st.markdown(f"<div class='alert {scls}'><span class='ic'>&#9650;</span><div>"
                    f"<b>{ticker}</b> had a <b>{strength}</b> unusual day on "
                    f"<span class='num'>{a['date'][:10]}</span> — it moved "
                    f"<span class='num'>{a['daily_return'] * 100:+.1f}%</span>. "
                    f"<span style='opacity:.8'>{len(anoms)} unusual day(s) flagged in the last 1.5 years.</span>"
                    f"</div></div>", unsafe_allow_html=True)

    # ===== MARKET-WIDE (identical for every asset) =====
    section_banner(f"MARKET-WIDE · news & sentiment for the whole market (not specific to {ticker})", "market")
    st.markdown("#### Global market sentiment")
    st.caption("The mood of recent **news headlines** across the whole market. "
               "**50 = neutral**; higher = more positive news. 🟢 upbeat (≥58) · 🟠 mixed · 🔴 gloomy (≤44).")

    deltas = _sent_deltas()
    scored = {k: _to100(v["avg_sentiment"]) for k, v in sent.items() if v}
    if scored:
        avg = round(sum(scored.values()) / len(scored))
        mood = "Upbeat" if avg >= 58 else ("Gloomy" if avg <= 44 else "Mixed")
        mcol = GREEN if avg >= 58 else (RED if avg <= 44 else AMBER)
        hi = max(scored, key=scored.get); lo = min(scored, key=scored.get)
        g_col, s_col = st.columns([1, 1.5])
        with g_col:
            gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=avg, number={"suffix": "/100", "font": {"size": 26}},
                gauge={"axis": {"range": [0, 100], "tickvals": [0, 50, 100],
                                "tickfont": {"size": 10, "color": MUTED}},
                       "bar": {"color": mcol, "thickness": 0.28},
                       "steps": [{"range": [0, 44], "color": "#fbecee"},
                                 {"range": [44, 58], "color": "#faf1e1"},
                                 {"range": [58, 100], "color": "#e7f4ec"}],
                       "threshold": {"line": {"color": INK, "width": 2}, "value": 50}}))
            gauge.update_layout(height=180, margin=dict(t=8, b=6, l=26, r=26))
            st.plotly_chart(gauge, width="stretch", config={"displayModeBar": False})
        with s_col:
            st.markdown(
                f"<div class='mood-box'><div class='mood-lab'>Overall market mood</div>"
                f"<div class='mood-val' style='color:{mcol}'>{avg}/100 · {mood}</div>"
                f"<div class='mood-sub'>🟢 most upbeat: <b>{SENT_LABELS[hi]} ({scored[hi]})</b><br>"
                f"🔴 gloomiest: <b>{SENT_LABELS[lo]} ({scored[lo]})</b></div></div>",
                unsafe_allow_html=True)

    st.markdown(sentiment_pills(sent, deltas), unsafe_allow_html=True)
    st.caption("▲▼ = change vs yesterday (shown only when there are enough articles to be reliable) · "
               "faded bars = fewer articles.")

    # "What moved it" — the top +/- headline behind each category's score.
    drivers = api_get("/sentiment/drivers").get("drivers", {})
    if drivers:
        with st.expander("🔎 What's moving each category — the top headlines behind the scores"):
            for k in SENT_ORDER:
                dd = drivers.get(k)
                if not dd:
                    continue
                st.markdown(f"**{SENT_LABELS[k]}**")
                st.markdown(f"🟢 {dd['top_pos']['title']}  "
                            f"<span style='color:{MUTED};font-size:.8em'>"
                            f"({dd['top_pos']['source']} · score {dd['top_pos']['score']:+.2f})</span>",
                            unsafe_allow_html=True)
                st.markdown(f"🔴 {dd['top_neg']['title']}  "
                            f"<span style='color:{MUTED};font-size:.8em'>"
                            f"({dd['top_neg']['source']} · score {dd['top_neg']['score']:+.2f})</span>",
                            unsafe_allow_html=True)

    # Click-through: jump a category's headlines into the News tab.
    st.caption("👉 Click a category to open its exact headlines in the **News** tab:")
    bcols = st.columns(len(SENT_ORDER))
    clicked_cat = None
    for i, k in enumerate(SENT_ORDER):
        if bcols[i].button(SENT_LABELS[k], key=f"seecat_{k}", use_container_width=True):
            clicked_cat = k
            st.session_state["news_cat_select"] = k
    if clicked_cat:
        st.success(f"📰 The **News** tab is now filtered to **{SENT_LABELS[clicked_cat]}** — "
                   "click the **News** tab at the top to read those headlines.")

    st.markdown("#### Top market headlines")
    recent = api_get("/news", limit=120)["news"]
    picks = pick_headlines(recent, ticker)
    st.markdown(headlines(picks, ticker), unsafe_allow_html=True)
    starred = " · ★ = mentions " + ticker if any(
        any(t in p["title"].lower() for t in _mention_terms(ticker)) for p in picks) else ""
    st.caption(f"Strongest positive & negative headlines right now{starred} · "
               "open the **News** tab for the full feed →")

    section_banner(f"THIS ASSET · AI analysis of {ticker}", "asset")
    st.caption(f"A grounded, plain-English report on {ticker} from the AI agent — built from its "
               "**live** price, forecast, sentiment & SEC-filing context. ~10–20s.")
    if st.button(f"🧠 Analyze {ticker}", type="primary"):
        with st.spinner(f"Analyzing {ticker}…"):
            st.session_state[f"report_{ticker}"] = api_post("/agent", {"ticker": ticker})["report"]
    rep = st.session_state.get(f"report_{ticker}")
    if rep:
        with st.expander(f"Analyst report — {ticker}", expanded=True):
            st.markdown(_mdsafe(rep))
            st.download_button("📄 Download report (PDF)",
                               data=report_pdf(f"FinSight Analysis — {ticker}", rep),
                               file_name=f"finsight_{ticker}_report.pdf", mime="application/pdf")
    st.caption("Want a different ticker or follow-up questions? Use the **Analyst** tab →")


# ---- Forecast tab -------------------------------------------------------------
def render_forecast(ticker: str) -> None:
    asset_header(ticker)
    fc = api_get(f"/forecast/{ticker}")
    st.info("**The volatility forecast (HAR-RV) is the signal with real predictive edge** "
            "— it beats a naive baseline on ~88% of assets. The price forecast below is "
            "**illustrative only** (short-term prices are a random walk).")
    vf = fc.get("volatility_forecast")
    if vf:
        st.subheader("Volatility outlook (the real signal)")
        cur, pred = vf["current"], vf["predicted_next_week"]
        ann = 252 ** 0.5
        c1, c2 = st.columns(2)
        c1.metric("Current 5-day volatility", f"{cur * 100:.2f}% / day",
                  help=f"≈ {cur * ann * 100:.0f}% annualized")
        c2.metric("Predicted next week", f"{pred * 100:.2f}% / day",
                  f"{(pred / cur - 1) * 100:+.1f}% ({vf['direction']})",
                  help=f"≈ {pred * ann * 100:.0f}% annualized")
        st.caption(f"Shown as typical **daily** swings. Annualized: current ≈ **{cur * ann * 100:.0f}%**, "
                   f"predicted ≈ **{pred * ann * 100:.0f}%** — comparable to the Overview tab's annualized "
                   "volatility card (which uses a 20-day window, so it won't match exactly).")

    st.divider()
    st.subheader("7-day price forecast (illustrative)")
    if fc["price_high_uncertainty"]:
        st.warning(f"⚠️ **High uncertainty** — this asset's price forecast has "
                   f"{fc['price_mape_pct']}% historical error. Treat as noise, not a prediction.")
    hist = pd.DataFrame(api_get(f"/prices/{ticker}", days=60)["history"])
    hist["date"] = pd.to_datetime(hist["date"])
    pf = pd.DataFrame(fc["price_forecast"]); pf["date"] = pd.to_datetime(pf["date"])
    now = hist["date"].iloc[-1]
    last_close = float(hist["close"].iloc[-1])
    fc_end = float(pf["forecast"].iloc[-1])
    ylo = min(float(hist["close"].min()), float(pf["low"].min()))
    yhi = max(float(hist["close"].max()), float(pf["high"].max()))
    ypad = (yhi - ylo) * 0.08 or 1
    col1, col2 = st.columns([3, 1])
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist["date"], y=hist["close"], name="Actual price",
                                 line=dict(color=ACCENT, width=2.2),
                                 hovertemplate="%{x|%b %d}: $%{y:,.2f}<extra></extra>"))
        fig.add_trace(go.Scatter(x=[now] + list(pf["date"]), y=[last_close] + list(pf["forecast"]),
                                 name="Forecast (illustrative)",
                                 line=dict(color=AMBER, width=2.2, dash="dash"),
                                 hovertemplate="%{x|%b %d}: ~$%{y:,.2f}<extra></extra>"))
        fig.add_trace(go.Scatter(x=pf["date"], y=pf["high"], mode="lines", line=dict(width=0),
                                 showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=pf["date"], y=pf["low"], mode="lines", fill="tonexty",
                                 fillcolor="rgba(181,120,28,0.16)", line=dict(width=0),
                                 name="Likely range (80%)", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=[now, now], y=[ylo - ypad, yhi + ypad], mode="lines",
                                 line=dict(color="#c9ccd4", width=1, dash="dot"),
                                 showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=[now], y=[last_close], mode="markers", showlegend=False,
                                 marker=dict(color=ACCENT, size=8), hoverinfo="skip"))
        fig.add_annotation(x=now, y=yhi + ypad, text="↓ today · forecast starts", showarrow=False,
                           yshift=-2, xshift=6, xanchor="left", font=dict(size=11, color=MUTED))
        fig.add_annotation(x=pf["date"].iloc[-1], y=fc_end, text=f"~${fc_end:,.0f}", showarrow=False,
                           yshift=14, font=dict(size=11, color=AMBER))
        fig.update_layout(height=380, margin=dict(t=24, b=0, l=6, r=12), hovermode="x unified",
                          legend=dict(orientation="h", y=1.1, x=0, font=dict(size=11, color=INK)),
                          xaxis=dict(showgrid=False, tickfont=dict(size=11, color=INK)),
                          yaxis=dict(title=dict(text="Price (USD)", font=dict(size=12, color=MUTED)),
                                     range=[ylo - ypad, yhi + ypad], tickprefix="$", tickformat=",.0f",
                                     showgrid=True, gridcolor="#eef0f4", tickfont=dict(size=11, color=INK)))
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        st.caption("**Blue** = actual price · **amber dashed** = the model's forecast (illustrative) · "
                   "**shaded band** = the 80% range it could land in. Dotted line = today; the forecast "
                   "starts from today's price and the band widens as uncertainty grows.")
    with col2:
        mape = fc["price_mape_pct"]
        hi_unc = fc["price_high_uncertainty"]
        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=mape, number={"suffix": "%", "font": {"size": 34}},
            title={"text": "Forecast error (MAPE)", "font": {"size": 19, "color": INK}},
            gauge={"axis": {"range": [0, 40], "tickvals": [0, 15, 30], "ticksuffix": "%"},
                   "bar": {"color": RED if hi_unc else GREEN},
                   "steps": [{"range": [0, 15], "color": "#e7f4ec"},
                             {"range": [15, 40], "color": "#fbecee"}],
                   "threshold": {"line": {"color": INK, "width": 3}, "value": 15}}))
        gauge.update_layout(height=350, margin=dict(t=62, b=6, l=24, r=24))
        st.plotly_chart(gauge, width="stretch", config={"displayModeBar": False})
        st.caption(f"How far off this asset's past forecasts typically were. **{mape}%** — "
                   + ("⚠️ **high uncertainty**" if hi_unc else "✓ **within usable range**")
                   + ". Green (<15%) = usable · red (>15%) = treat as noise · black line = 15% cutoff.")
    st.caption("Day-by-day forecast with the 80% low–high range:")
    table = pf[["date", "forecast", "low", "high"]].copy()
    table["date"] = table["date"].dt.date
    st.dataframe(table, hide_index=True, width="stretch")

    _forecast_bts(fc)


def _forecast_bts(fc: dict) -> None:
    """Transparent 'how this was predicted' panel -- method, report card, drivers."""
    bt = fc.get("backtest", {})
    dr = fc.get("drivers", {})
    with st.expander("🔍 How this 7-day forecast is made — behind the scenes", expanded=True):
        st.markdown(
            f"**Method.** A Prophet model is fit on **{dr.get('history_days', '~750')} days** of this "
            "asset's daily closes. It separates the price into a long-term **trend** plus repeating "
            "**weekly & yearly seasonal** patterns, then projects those 7 days ahead. The shaded band "
            "is the model's 80% confidence interval.")

        st.markdown("**Report card** — scored on a held-out week the model never saw:")
        s1, s2, s3 = st.columns(3)
        s1.metric("Model error (MAPE)", f"{bt.get('model_mape_pct', '?')}%",
                  help="Average % the forecast was off on the held-out week. Lower is better.")
        s2.metric("Naive 'no-change' error", f"{bt.get('naive_mape_pct', '?')}%",
                  help="Error if you'd simply assumed: next week = today's price.")
        beats = bt.get("beats_naive")
        s3.metric("Verdict", "Beats naive ✅" if beats else "Naive wins ⚠️")
        if beats:
            st.success("On the backtest this model was **more accurate than doing nothing** for this "
                       "asset — a genuine (if modest) edge. The center line is worth reading here.")
        else:
            st.warning("On the backtest, simply assuming **no change** was more accurate than the model "
                       "— normal for short-term prices. So read the **band** (how far it could move), not "
                       "the exact center line, as the real signal.")

        st.markdown("**What's driving *this* forecast**")
        d1, d2, d3 = st.columns(3)
        d1.metric("Last close", f"${dr.get('last_close', '?')}")
        d2.metric("Model trend", f"${dr.get('trend_per_day', '?')}/day",
                  dr.get("trend_direction", ""), delta_color="off")
        vs = dr.get("forecast_vs_last_pct")
        d3.metric("7-day vs today", f"{vs:+.1f}%" if isinstance(vs, (int, float)) else "—")
        # Explain why the 7-day move can dwarf the gentle trend slope (reversion to trend).
        tpd, lc = dr.get("trend_per_day"), dr.get("last_close")
        if isinstance(vs, (int, float)) and isinstance(tpd, (int, float)) and lc:
            trend_wk = tpd * 7 / lc * 100
            if abs(vs) - abs(trend_wk) > 3:
                side = "below" if vs > 0 else "above"
                st.caption(f"ℹ️ The **{vs:+.1f}% 7-day move** is far bigger than the trend alone "
                           f"(~{trend_wk:+.1f}% over the week) because the last close sits **{side} the "
                           f"model's trend line**, so Prophet expects a reversion back toward it. That "
                           "reversion — not the gentle trend — drives most of the forecast, which is "
                           "another reason it's *illustrative*.")
        vf = fc.get("volatility_forecast")
        if vf:
            st.caption(f"The band's width tracks volatility: current 5-day swing is "
                       f"{vf['current'] * 100:.1f}%, forecast to be **{vf['direction']}** next week "
                       f"(≈{vf['predicted_next_week'] * 100:.1f}%). Wider band = more expected movement — "
                       "not a bug, the model being honest about risk.")


# ---- News tab -----------------------------------------------------------------
_LC = {"positive": GREEN, "negative": RED, "neutral": AMBER}


def _dot(label: str) -> str:
    return f"<span style='color:{_LC.get(label, MUTED)};font-size:.7em;vertical-align:.18em'>&#9679;</span>"


def render_news() -> None:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Sentiment by category")
        sent = api_get("/sentiment").get("categories", {})
        rows = [{"cat": k, "score": _to100(v["avg_sentiment"]), "n": int(v.get("article_count", 0))}
                for k, v in sent.items()]
        colf = lambda s: GREEN if s >= 58 else (RED if s <= 44 else AMBER)  # noqa: E731
        xl = [f"{r['cat']}<br>{r['n']} art" for r in rows]
        bar = go.Figure(go.Bar(
            x=xl, y=[r["score"] for r in rows],
            marker=dict(color=[colf(r["score"]) for r in rows],
                        opacity=[min(1.0, max(0.4, 0.4 + r["n"] / 60 * 0.6)) for r in rows]),
            text=[str(r["score"]) for r in rows], textposition="outside",
            hovertemplate="%{y}/100<extra></extra>"))
        bar.add_hline(y=50, line_dash="dot", line_color="#c9ccd4",
                      annotation_text="neutral (50)", annotation_position="top left")
        bar.update_layout(height=310, margin=dict(t=26, b=0),
                          yaxis=dict(title="sentiment (0–100)", range=[0, 112], tickvals=[0, 50, 100],
                                     tickfont=dict(color=INK)),
                          xaxis=dict(tickfont=dict(size=11, color=INK)))
        st.plotly_chart(bar, width="stretch", config={"displayModeBar": False})
        st.caption("0–100 scale (50 = neutral), matching the Overview pills. Bars are **faded when based "
                   "on few articles** (less reliable); each bar's article count is shown beneath it.")
    with c2:
        st.subheader("Daily sentiment trend")
        tdf = pd.DataFrame(api_get("/sentiment/trend")["trend"])
        tdf["date"] = pd.to_datetime(tdf["date"]); tdf["w"] = tdf["avg_sentiment"] * tdf["article_count"]
        daily = tdf.groupby("date").agg(w=("w", "sum"), n=("article_count", "sum")).reset_index()
        daily["s"] = ((daily["w"] / daily["n"]) + 1) / 2 * 100          # -> 0-100 scale
        nmax = daily["n"].max() or 1
        sizes = [7 + 13 * (n / nmax) for n in daily["n"]]              # bigger dot = more articles
        line = go.Figure(go.Scatter(
            x=daily["date"], y=daily["s"], mode="lines+markers", line=dict(color=ACCENT),
            marker=dict(size=sizes, color=ACCENT), customdata=daily["n"],
            hovertemplate="%{x|%b %d}: %{y:.0f}/100<br>%{customdata} articles<extra></extra>"))
        line.add_hline(y=50, line_dash="dot", line_color="#c9ccd4",
                       annotation_text="neutral (50)", annotation_position="top left")
        line.update_layout(height=310, margin=dict(t=26, b=0),
                           yaxis=dict(title="overall mood (0–100)", tickfont=dict(color=INK)),
                           xaxis=dict(tickfont=dict(color=INK)))
        st.plotly_chart(line, width="stretch", config={"displayModeBar": False})
        st.caption("Overall daily mood on the 0–100 scale (50 = neutral). **Bigger dots = more articles "
                   "that day** (more reliable); small dots are thin, noisy days.")

    st.divider()
    st.subheader("Sentiment analyzer (FinBERT)")
    with st.form("finbert_form"):
        text = st.text_input("Paste any financial headline to score it live:")
        submitted = st.form_submit_button("Score", type="primary")
    if submitted and text.strip():
        st.session_state["finbert_result"] = api_post("/sentiment/score", {"text": text})
    res = st.session_state.get("finbert_result")
    if res:
        a, b = st.columns([1, 2])
        a.metric(res["label"].title(), f"{res['score']:+.3f}")
        probs = res["probabilities"]
        pbar = go.Figure(go.Bar(x=list(probs.keys()), y=list(probs.values()),
                                marker_color=[_LC.get(k, "#ccc") for k in probs]))
        pbar.update_layout(height=200, yaxis_title="probability")
        b.plotly_chart(pbar, width="stretch")

    st.divider()
    st.subheader("News feed")
    f1, f2, f3 = st.columns(3)
    category = f1.selectbox("Category", ["all", "geopolitical", "commodities", "crypto",
                                         "country", "macro", "market"], key="news_cat_select")
    label = f2.selectbox("Sentiment", ["all", "positive", "negative", "neutral"])
    when = f3.selectbox("Time", ["Latest day", "All recent"], key="news_when")
    params = {"limit": 150}
    if category != "all":
        params["category"] = category
    if label != "all":
        params["label"] = label
    feed = api_get("/news", **params)["news"]
    if when == "Latest day" and feed:
        latest = max(n["published_date"] for n in feed)
        feed = [n for n in feed if n["published_date"] == latest]
        st.caption(f"Showing all **{len(feed)}** article(s) from **{latest}** — exactly the ones behind "
                   "that day's sentiment score.")
    else:
        st.caption(f"Showing **{len(feed)}** recent article(s).")
    fc1, fc2 = st.columns([1, 2])
    with fc1:
        if feed:
            counts = pd.Series([n["sentiment_label"] for n in feed]).value_counts()
            donut = go.Figure(go.Pie(labels=list(counts.index), values=[int(v) for v in counts.values],
                                     hole=0.6, marker_colors=[_LC.get(l, "#ccc") for l in counts.index]))
            donut.update_layout(height=300, title="Sentiment split")
            st.plotly_chart(donut, width="stretch")
    with fc2:
        if not feed:
            st.write("No articles match the filter.")
        else:
            with st.container(height=520):  # fixed-height, scrollable feed box
                for n in feed:
                    st.markdown(
                        f"{_dot(n['sentiment_label'])} **{n['title']}**  \n"
                        f"<span style='color:#666c77;font-size:0.82em;font-family:IBM Plex Mono,monospace'>"
                        f"{n['source_name']} · {n['published_date']} · {n['category']} · "
                        f"score {n['sentiment_score']:+.2f}</span>",
                        unsafe_allow_html=True)


# ---- Analyst tab --------------------------------------------------------------
def render_analyst(ticker: str) -> None:
    st.markdown("### 💬 Ask the analyst")
    st.caption("Ask **anything** — a general finance question, or about this dashboard's live data "
               "(any asset's price, forecast, volatility, anomalies, market sentiment, news, or SEC "
               "filings). The assistant pulls live data with tools when it needs to. Not financial advice.")

    if "chat" not in st.session_state:
        st.session_state.chat = []

    head = st.columns([6, 1])
    head[0].caption("Try one:")
    if st.session_state.chat and head[1].button("Clear", use_container_width=True):
        st.session_state.chat = []
        st.rerun()
    examples = [f"How is {ticker} doing?",
                "Which asset is most volatile right now?",
                "What's the market sentiment today?",
                "What does MAPE mean in plain English?",
                f"Give me {ticker}'s 7-day forecast",
                "Any recent unusual days for TSLA?"]
    picked = None
    ecols = st.columns(3)
    for i, ex in enumerate(examples):
        if ecols[i % 3].button(ex, key=f"ex_{i}", use_container_width=True):
            picked = ex

    # Render the whole conversation first; the input bar comes after it (typical chat layout).
    for m in st.session_state.chat:
        with st.chat_message(m["role"]):
            st.markdown(_mdsafe(m["content"]))

    typed = st.chat_input("Ask anything… e.g. 'Compare AAPL and MSFT volatility'")
    q = (typed or picked or "").strip()
    if q:
        st.session_state.chat.append({"role": "user", "content": q})
        with st.spinner("Thinking… (fetching live data if needed)"):
            try:
                ans = api_post("/ask", {"question": q,
                                        "history": st.session_state.chat[:-1]})["answer"]
            except Exception as exc:  # noqa: BLE001
                ans = f"⚠️ Error: {exc}"
        st.session_state.chat.append({"role": "assistant", "content": ans})
        st.rerun()  # re-render so the new turn appears above the input bar


# ---- App ----------------------------------------------------------------------
_MARK = ("<svg width='16' height='16' viewBox='0 0 16 16' fill='none'>"
         "<rect x='1' y='9' width='3' height='6' rx='1' fill='white'/>"
         "<rect x='6.5' y='5' width='3' height='10' rx='1' fill='white'/>"
         "<rect x='12' y='2' width='3' height='13' rx='1' fill='white'/></svg>")

_DEMO = bool(api_get("/").get("demo"))
_badge = "demo · historical data" if _DEMO else "live · real-time market data"
st.markdown(
    f"<div class='fs-head'><div class='fs-brand'>"
    f"<span class='mark'>{_MARK}</span>"
    f"<span class='nm'>FinSight AI</span>"
    f"<span class='tag'>global financial intelligence</span></div>"
    f"<span class='live'><span class='dot'></span>{_badge}</span></div>",
    unsafe_allow_html=True)

if _DEMO:
    st.info("🎬 **Demo mode** — everything below runs on **real historical data** (no live fetching, "
            "no API keys needed). The **AI report** on the Overview tab is pre-built; the free-form "
            "**chat** is off in the demo. Run locally with API keys for full live mode.")

tickers = api_get("/tickers")
ticker = st.sidebar.selectbox("Asset", tickers, index=0)
st.sidebar.caption("Historical snapshot (demo mode)." if _DEMO
                   else "Prices, forecasts & news are live — auto-refreshed every 3h.")

tab_overview, tab_forecast, tab_news, tab_analyst = st.tabs(
    ["Overview", "Forecast", "News", "Analyst"])
with tab_overview:
    render_overview(ticker)
with tab_forecast:
    render_forecast(ticker)
with tab_news:
    render_news()
with tab_analyst:
    render_analyst(ticker)

st.divider()
st.caption(
    "**Educational demo — not financial advice.** FinSight AI is a personal learning project; "
    "figures may be delayed or historical and must not be used for real trading decisions. "
    "**Data sources:** prices via Yahoo Finance (yfinance) · news via NewsAPI · filings via SEC EDGAR · "
    "macro via FRED (Federal Reserve Bank of St. Louis). Not affiliated with, or endorsed by, any data "
    "provider; all trademarks belong to their owners.")