"""HazeTrace web app.   Run:  streamlit run app.py"""
import base64
import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from hazetrace import config as C
from hazetrace import insights as I
from hazetrace.regions import assign_region

st.set_page_config(page_title="HazeTrace · Fire history for the haze season", page_icon="🔥", layout="wide")

# ---------------- look ----------------------------------------------------------------------------
INK, MUTED, LINE, CARD, PAGE = "#1F2328", "#5F6670", "#E5E0D8", "#FFFFFF", "#F7F5F2"
FIRE, MODIS, VIIRS, GRID = "#D9480F", "#2B6CB0", "#8A919A", "#EEE9E2"
FIRE_SCALE = ["#FFEDA0", "#FEB24C", "#FD8D3C", "#E31A1C", "#800026"]
CAL_SCALE = ["#FFF5EB", "#FDD0A2", "#FD8D3C", "#D94801", "#7F2704"]
SEL = "#FFB703"
LEVELS = {
    "Typical": ("#E7F0F7", "#2B6CB0", "Not unusual for this month."),
    "Above usual": ("#FFF1D6", "#8A5A00", "At least 1.5 times a typical month, or in the top quarter of years."),
    "Well above usual": ("#FFE4D1", "#B54708", "At least 3 times a typical month, or among the 4 highest years."),
    "Exceptional": ("#FDE2E1", "#B42318", "One of the 2 highest years for this month, or part of a top-3 haze season."),
}
FULL_MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
               "October", "November", "December"]
WHOLE = "Whole study area"

st.markdown(f"""
<style>
#MainMenu, footer {{visibility: hidden;}}
.block-container {{padding-top: 3.4rem; max-width: 1320px;}}
.ht-head {{display:flex; justify-content:space-between; align-items:flex-end; gap:1rem; flex-wrap:wrap; margin-bottom:.4rem;}}
.ht-title {{font-size:2.15rem; font-weight:800; letter-spacing:-.5px; color:{INK}; line-height:1.05;}}
.ht-title span {{color:{FIRE};}}
.ht-sub {{color:{MUTED}; font-size:1rem; margin-top:.3rem;}}
.ht-chip {{border:1px solid {LINE}; border-radius:999px; padding:.32rem .85rem; color:{MUTED}; font-size:.8rem;
  background:{CARD}; white-space:nowrap;}}
.ht-label {{color:{MUTED}; font-size:.74rem; font-weight:700; letter-spacing:.9px; text-transform:uppercase;
  margin:.6rem 0 .35rem 0;}}
.ht-card {{background:{CARD}; border:1px solid {LINE}; border-radius:12px; box-shadow:0 1px 2px rgba(0,0,0,.04);
  padding:.95rem 1.1rem; height:100%;}}
.ht-card h4 {{margin:0 0 .35rem 0; color:{INK}; font-size:1.02rem;}}
.ht-card p {{color:{MUTED}; margin:.25rem 0; font-size:.9rem;}}
.ht-kpi {{background:{CARD}; border:1px solid {LINE}; border-radius:12px; box-shadow:0 1px 2px rgba(0,0,0,.04);
  padding:.8rem 1rem; margin-bottom:.6rem;}}
.ht-kpi .v {{font-size:1.75rem; font-weight:800; color:{INK}; line-height:1.15; font-variant-numeric:tabular-nums;}}
.ht-kpi.first .v {{color:{FIRE};}}
.ht-kpi .l {{font-weight:700; color:{INK}; font-size:.92rem; margin-top:.1rem;}}
.ht-kpi .s {{color:{MUTED}; font-size:.8rem; margin-top:.25rem;}}
.ht-summary {{background:{CARD}; border:1px solid {LINE}; border-radius:12px; box-shadow:0 1px 2px rgba(0,0,0,.04);
  padding:1rem 1.2rem; margin:.6rem 0 1rem 0;}}
.ht-summary .ttl {{font-weight:700; color:{INK}; font-size:1.02rem;}}
.ht-summary .txt {{color:{INK}; margin-top:.45rem; font-size:1rem; line-height:1.6;}}
.ht-summary .ftr {{color:{MUTED}; font-size:.78rem; margin-top:.55rem;}}
.ht-pill {{display:inline-block; border-radius:999px; padding:.12rem .65rem; font-weight:700; font-size:.8rem;
  margin-right:.55rem; cursor:help;}}
.ht-explain {{background:#FFF8F3; border:1px solid #F6D9C6; border-radius:10px; padding:.7rem 1rem;
  color:{INK}; font-size:.93rem; margin:.3rem 0 .6rem 0;}}
.ht-badge {{display:inline-block; border-radius:6px; padding:.12rem .5rem; font-weight:700; font-size:.78rem;
  color:#fff; margin-bottom:.4rem;}}
.ht-credit {{color:{MUTED}; font-size:.78rem; margin-top:-.2rem;}}
button[data-baseweb="tab"] p {{font-size:.98rem; font-weight:600;}}
.ht-hint {{color:{MUTED}; font-size:.85rem; margin:.1rem 0 .2rem 0;}}
@media (max-width: 640px) {{ .block-container {{padding-top: 3.6rem;}} .ht-title {{font-size:1.8rem;}} }}
[data-testid="stMetricValue"] {{font-variant-numeric: tabular-nums;}}
</style>""", unsafe_allow_html=True)

CFG = {"displayModeBar": False}
LOG_TICKS = [1, 10, 100, 1e3, 1e4, 1e5, 1e6]
LOG_TEXT = ["1", "10", "100", "1k", "10k", "100k", "1M"]


def style(fig, h=380):
    fig.update_layout(height=h, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color=INK), margin=dict(l=10, r=10, t=10, b=10),
                      hoverlabel=dict(bgcolor="white", bordercolor=LINE, font=dict(color=INK)))
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=LINE, tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=LINE, tickfont=dict(color=MUTED))
    return fig


def log_y(fig, title=None):
    fig.update_yaxes(type="log", tickvals=LOG_TICKS, ticktext=LOG_TEXT, title=title)
    return fig


def pill(level):
    bg, fg, tip = LEVELS[level]
    return f'<span class="ht-pill" style="background:{bg};color:{fg}" title="{level}: {tip}">{level}</span>'


def ordinal(n):
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def fmt_lat(v):
    return f"{abs(v):.2f}°{'S' if v < 0 else 'N'}"


def hs(n):
    """'1 fire hotspot' / '2,345 fire hotspots'."""
    return f"{n:,.0f} fire hotspot{'' if round(n) == 1 else 's'}"


def fmt_n(v):
    return "n/a" if pd.isna(v) else f"{v:,.0f}"


GLOSSARY = """
- **Hotspot:** a satellite pixel where a fire was detected. Counts are an index of burning, not the number of fires.
- **MODIS** (since 2000, 1 km pixels) and **VIIRS** (since 2012, 375 m pixels): the two NASA fire sensors.
  VIIRS sees smaller fires, so it counts several times more hotspots.
- **Harmonized / MODIS scale:** VIIRS counts converted to what MODIS would have counted, so 2003 and 2026 compare fairly.
- **Raw:** the count straight from one sensor, not converted.
- **Likely range:** an approximate 10 to 90% range for the HazeTrace estimate.
- **× a typical month:** compared with the median of the same month in 2003 to 2025.
- **Rank 3 of 24:** 3rd highest of 24 years; 1 is the most fires.
- **Preliminary:** near-real-time data; NASA replaces it with final data about 5 months later.
- **Cell:** a 0.25° square, about 28 × 28 km.
- **Held-out year / R²:** a year hidden from the model and then predicted, to test it fairly; R² near 1 means
  predictions follow the real values closely.
"""


# ---------------- data ----------------------------------------------------------------------------
@st.cache_data
def load():
    cells = pd.read_parquet(C.OUTPUT_DIR / "harmonized_cells.parquet")
    u = cells[["cell_lat", "cell_lon"]].drop_duplicates()
    u["region"] = assign_region(u.cell_lat.values, u.cell_lon.values)
    cells = cells.drop(columns="region").merge(u, on=["cell_lat", "cell_lon"], how="left")
    cells["cid"] = I.cell_id(cells)
    meta = json.loads((C.OUTPUT_DIR / "metrics.json").read_text())
    by_year = pd.read_csv(C.OUTPUT_DIR / "validation_by_year.csv")
    by_group = pd.read_csv(C.OUTPUT_DIR / "validation_by_group.csv")
    imp = pd.read_csv(C.OUTPUT_DIR / "feature_importance.csv")
    return cells, meta, by_year, by_group, imp


@st.cache_data
def cell_gj():
    return I.cell_geojson(load()[0])


@st.cache_data
def satellite_uri():
    if not C.SATELLITE_IMAGE.exists():
        return None
    return "data:image/jpeg;base64," + base64.b64encode(C.SATELLITE_IMAGE.read_bytes()).decode()


@st.cache_data
def borders():
    return json.loads(C.BORDERS_FILE.read_text(encoding="utf-8")) if C.BORDERS_FILE.exists() else None


@st.cache_data
def provinces():
    return json.loads(C.PROVINCES_FILE.read_text(encoding="utf-8")) if C.PROVINCES_FILE.exists() else None


@st.cache_data
def malaysia_cells(strict=False):
    """Cell ids for Malaysia, the only country with 2026 MODIS files.

    strict=False: cells touching Malaysia (used for area totals such as Sarawak).
    strict=True: cells touching Malaysia and no other country, so their 2026 MODIS count is complete.
    """
    exact = C.OUTPUT_DIR / "malaysia_2026_modis_cells.csv"
    if strict and exact.exists():   # exact list from the FIRMS files, written by scripts/02_harmonize.py
        return set(I.cell_id(pd.read_csv(exact)))
    b = borders()
    cells_ = load()[0][["cid", "cell_lat", "cell_lon"]].drop_duplicates("cid")
    if not b:
        return set(load()[0].loc[load()[0].region == "Sarawak (Malaysia)", "cid"])
    import shapely
    from shapely.geometry import shape
    h = C.GRID_DEG / 2
    squares = shapely.box(cells_.cell_lon.values - h, cells_.cell_lat.values - h,
                          cells_.cell_lon.values + h, cells_.cell_lat.values + h)
    in_my = np.zeros(len(cells_), bool)
    in_other = np.zeros(len(cells_), bool)
    for f in b["features"]:
        hit = shapely.intersects(shape(f["geometry"]), squares)
        if f["properties"].get("name") == "Malaysia":
            in_my |= hit
        else:
            in_other |= hit
    return set(cells_.cid[in_my & ~in_other] if strict else cells_.cid[in_my])


@st.cache_data
def area_cells(region: str, box: tuple | None):
    cells = load()[0]
    if box:
        w, e, s, n = box
        return cells[cells.cell_lon.between(w, e) & cells.cell_lat.between(s, n)]
    return cells if region == WHOLE else cells[cells.region == region]


if not (C.OUTPUT_DIR / "metrics.json").exists():
    st.error("No results yet. Run scripts/01_aggregate.py and scripts/02_harmonize.py first.")
    st.stop()
cells, meta, by_year, by_group, imp = load()
GJ = cell_gj()
latest = I.latest_complete_month(cells)
yr_latest, mo_latest = int(latest[:4]), int(latest[5:7])
years = sorted({int(x[:4]) for x in cells.ym.unique()})

# ---------------- sidebar -------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🔥 HazeTrace")
    st.caption("Choose an area")
    mode = st.radio("Area", ["Named region", "My own box (lat/lon)"], label_visibility="collapsed")
    box = None
    if mode == "Named region":
        opts = [WHOLE] + C.REGION_ORDER
        region = st.selectbox("Region", opts, index=opts.index(st.session_state.get("last_region", WHOLE)))
        st.session_state["last_region"] = region     # remembered when switching to a box and back
        area_label = region
    else:
        region = "Custom"
        w0, e0, s0, n0 = C.DOMAIN
        c1, c2 = st.columns(2)
        w = c1.number_input("West (°E)", w0, e0, 100.0, 0.25)
        e = c2.number_input("East (°E)", w0, e0, 104.0, 0.25)
        s = c1.number_input("South (°N)", s0, n0, -1.0, 0.25, help="Latitude. Negative numbers are south of the equator.")
        n = c2.number_input("North (°N)", s0, n0, 2.5, 0.25, help="Latitude. Negative numbers are south of the equator.")
        box = (min(w, e), max(w, e), min(s, n), max(s, n))
        area_label = f"{fmt_lat(box[2])} to {fmt_lat(box[3])}, {box[0]:.2f} to {box[1]:.2f}°E"
    st.divider()
    live = st.toggle("Sharper live NASA imagery", value=True,
                     help="Loads NASA GIBS Blue Marble tiles online. Without internet the built-in Blue Marble "
                          "image shows instead, so the map is never blank.")
    st.caption(f"Data up to {C.DATA_END_DATE}. Latest complete month: {I.month_label(latest)}. Aug to Nov 2026 "
               "is preliminary (NASA updates it about 5 months later).")
    with st.expander("What the words mean"):
        st.markdown(GLOSSARY)

sel = area_cells(region, box)
if sel.empty:
    st.info("No fire detections in this area since 2003. Try a larger box or a named region.")
    st.stop()
series = I.monthly_series(sel, last=cells.ym.max())   # every area runs to the end of the data
stats = I.month_stats(series, latest)
seas = I.season_table(series)
season_rank = int(seas.loc[yr_latest, "rank"]) if yr_latest in seas.index else None
level = I.level_for(stats, season_rank)


def region_rows(ym):
    """Region table with share of ALL hotspots in the study area (including Other)."""
    rt = I.region_table(cells, ym)
    if int(ym[5:7]) in I.SEASON:   # same rule as the summary card: a top-3 haze season counts
        rt["Level"] = [I.level_for(I.month_stats(I.monthly_series(cells[cells.region == r], last=cells.ym.max()), ym),
                                   int(sr) if pd.notna(sr) else None)
                       for r, sr in zip(rt.Region, rt["Season rank"])]
    total = cells.loc[cells.ym == ym, "value"].sum()
    rt["Share"] = (rt.Hotspots / total * 100) if total else 0.0
    return rt


# ---------------- header --------------------------------------------------------------------------
st.markdown(f"""
<div class="ht-head">
  <div><div class="ht-title">Haze<span>Trace</span></div>
  <div class="ht-sub">One consistent fire history for Southeast Asia's haze season, from two NASA satellite sensors</div></div>
  <div class="ht-chip">Data up to {C.DATA_END_DATE} · Aug to Nov 2026 preliminary</div>
</div>
<div class="ht-hint">Showing <b>{area_label}</b>. Choose another area in the sidebar (on a phone, tap » at the top left).</div>""",
            unsafe_allow_html=True)

# ---------------- hero: why harmonize + how well it works -----------------------------------------
hl, hr = st.columns([2.1, 1], gap="large")
with hl:
    st.markdown('<div class="ht-label">Why harmonize?</div>', unsafe_allow_html=True)
    sz = series[series.month.isin(I.SEASON)].groupby("year").agg(
        value=("value", "sum"), modis=("modis", "sum"), viirs=("viirs", "sum")).reset_index()
    sz.loc[sz.year < 2012, "viirs"] = np.nan
    if region != "Sarawak (Malaysia)":
        sz.loc[sz.year >= int(C.MODIS_PARTIAL_FROM[:4]), "modis"] = np.nan
    sz = sz.replace(0, np.nan)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sz.year, y=sz.viirs, name="Raw VIIRS (sees smaller fires)", mode="lines+markers",
                             line=dict(color=VIIRS, dash="dot", width=2), marker=dict(size=5)))
    fig.add_trace(go.Scatter(x=sz.year, y=sz.modis, name="Raw MODIS", mode="lines+markers",
                             line=dict(color=MODIS, width=2), marker=dict(size=4)))
    fig.add_trace(go.Scatter(x=sz.year, y=sz.value, name="HazeTrace (one consistent record)", mode="lines+markers",
                             line=dict(color=FIRE, width=3.5), marker=dict(size=6)))
    fig.add_vline(x=2011.6, line=dict(color=VIIRS, dash="dash", width=1))
    fig.add_annotation(x=2011.6, y=0.02, yref="paper", text="VIIRS starts (2012)", showarrow=False,
                       xanchor="right", yanchor="bottom", xshift=-4, font=dict(color=MUTED, size=11))
    hover = [f"HazeTrace: {fmt_n(a)}<br>Raw MODIS: {fmt_n(m)}<br>Raw VIIRS: {fmt_n(v)}"
             for a, m, v in zip(sz.value, sz.modis, sz.viirs)]
    fig.update_traces(hoverinfo="skip", hovertemplate=None)
    fig.data[2].update(customdata=hover, hovertemplate="%{customdata}<extra></extra>", hoverinfo=None)
    fig.update_layout(hovermode="x unified", legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"))
    log_y(style(fig, 330), "Aug to Oct hotspots (log scale)")
    st.plotly_chart(fig, width="stretch", config=CFG)
    st.caption(f"Haze-season (Aug to Oct) hotspots, {area_label}. Raw VIIRS counts are several times higher than "
               "MODIS for the same fires, so joining the raw records would fake a jump in 2012. HazeTrace puts every "
               "year on the MODIS scale.")
    gap = sz[(sz.year >= 2012) & (sz.year <= int(C.PAIR_END[:4])) & (sz.modis >= 50) & (sz.value < 0.5 * sz.modis)]
    if len(gap):
        gy = [str(y) for y in gap.year]
        yrs = gy[0] if len(gy) == 1 else ", ".join(gy[:-1]) + " and " + gy[-1]
        st.warning(f"In {yrs}, HazeTrace is less than half of what MODIS observed in this area. In quiet years many "
                   "MODIS detections fall in cells where VIIRS saw no fire, and those cells count as 0. For these "
                   "years, compare with Raw MODIS (hover the chart or download the CSV).")
with hr:
    st.markdown('<div class="ht-label">How well it works</div>', unsafe_allow_html=True)
    o = meta["overall"]
    wins = int((by_year.model_MAE < by_year.base_MAE).sum())
    n20 = (meta.get("noaa20_transfer") or {}).get("model", {})
    n20b = (meta.get("noaa20_transfer") or {}).get("baseline", {})
    cards = [
        (f"{(1 - o['model_MAE'] / o['base_MAE']) * 100:.0f}% lower error", "than a fixed conversion ratio",
         f"Tested on years the model never saw (error {o['model_MAE']:.0f} vs {o['base_MAE']:.0f} hotspots "
         "per region-month)"),
        (f"{wins} of {len(by_year)} years", "won by the model",
         f"Each year {by_year.year.min()} to {by_year.year.max()} held out in turn"),
        (f"{(1 - n20.get('MAE', 0) / n20b.get('MAE', 1)) * 100:.0f}% lower error", "on NOAA-20, never trained on it",
         f"Error {n20.get('MAE', 0):.0f} vs {n20b.get('MAE', 0):.0f} for a fixed ratio (R² {n20.get('R2', 0):.2f} vs "
         f"{n20b.get('R2', 0):.2f}). NOAA-20 carries the record now that S-NPP has stopped"),
    ]
    for i, (v, l, s_) in enumerate(cards):
        st.markdown(f'<div class="ht-kpi{" first" if i == 0 else ""}"><div class="v">{v}</div>'
                    f'<div class="l">{l}</div><div class="s">{s_}</div></div>', unsafe_allow_html=True)

# ---------------- summary card --------------------------------------------------------------------
if stats:
    mname = FULL_MONTHS[mo_latest - 1]
    ratio_txt = f", {stats['ratio']:.1f}× a typical {mname}" if stats["ratio"] else ""
    seas_txt = (f" Fire hotspots in the {yr_latest} haze season (Aug to Oct) rank <b>{season_rank} of {len(seas)}"
                f"</b> since {years[0]}." if season_rank and seas.loc[yr_latest, "value"] > 0 else "")
    rank_txt = (f"Rank <b>{stats['rank']} of {stats['n']}</b> {mname}s since {years[0]}, where 1 is the most fires."
                if stats["value"] > 0 else f"No fire hotspots were detected here in {mname} {yr_latest}.")
    where = ""
    if region == WHOLE:
        top = region_rows(latest).iloc[0]
        where = f" Most of it was in <b>{top.Region}</b> ({top.Share:.0f}% of all hotspots)."
    st.markdown(f"""
<div class="ht-summary">
  {pill(level)}<span class="ttl">{area_label} · {I.month_label(latest)}</span>
  <div class="txt"><b>{hs(stats['value'])}</b> (likely range {stats['lo']:,.0f} to {stats['hi']:,.0f}, an approximate 10 to 90% range){ratio_txt}.
  {rank_txt}{seas_txt}{where}
  {' Preliminary data.' if stats['nrt'] else ''}</div>
  <div class="ftr">Hotspots are counted on the MODIS scale so every year since 2003 is comparable. Satellite fire
  statistics from NASA FIRMS, not an air-quality reading, forecast, official alert or health advice. For haze and
  health information see Malaysia DOE (APIMS), BMKG and ASMC.</div>
</div>""", unsafe_allow_html=True)

tab_map, tab_cal, tab_good, tab_where, tab_who, tab_data = st.tabs(
    ["🗺️ Map", "📅 Calendar & seasons", "✅ How good is it?", "🧭 Where it works", "👥 Who it's for",
     "⬇️ Data & method"])

# ---------------- map helpers ---------------------------------------------------------------------
METRICS = {"HazeTrace (harmonized)": "value", "Raw VIIRS": "viirs_raw", "Raw MODIS": "modis"}
LOG_MAX = float(np.log10(max(cells.viirs_raw.quantile(0.995), cells.value.quantile(0.999), 10)))
TICKS = [t for t in [1, 3, 10, 30, 100, 300] if np.log10(t) <= LOG_MAX + 0.01]
TICK_TEXT = [f"{t}" for t in TICKS[:-1]] + [f"{TICKS[-1]}+"]
MAP_STYLE = {"version": 8, "sources": {}, "layers": [
    {"id": "sea", "type": "background", "paint": {"background-color": "#0B2140"}}]}


def map_layers():
    W, E, S, N = C.MAP_EXTENT
    layers = []
    uri = satellite_uri()
    if uri:
        layers.append(dict(sourcetype="image", source=uri, coordinates=[[W, N], [E, N], [E, S], [W, S]],
                           below="traces"))
    if live:
        layers.append(dict(sourcetype="raster", source=[C.GIBS_TILES], below="traces",
                           sourceattribution="NASA GIBS Blue Marble"))
    b = borders()
    if b:
        layers.append(dict(sourcetype="geojson", source=b, type="line", below="traces",
                           color="rgba(255,255,255,0.65)", line=dict(width=0.8)))
    p = provinces()
    if p:
        others = {"type": "FeatureCollection",
                  "features": [f for f in p["features"] if f["properties"]["region"] != region]}
        layers.append(dict(sourcetype="geojson", source=others, type="line", below="traces",
                           color="rgba(255,255,255,0.45)", line=dict(width=1, dash=[2, 2])))
        chosen = [f for f in p["features"] if f["properties"]["region"] == region]
        if chosen:
            layers.append(dict(sourcetype="geojson", source={"type": "FeatureCollection", "features": chosen},
                               type="line", below="traces", color=SEL, line=dict(width=2.5)))
    return layers


def add_labels(fig, small=False):
    names = list(C.COUNTRY_LABELS)
    fig.add_trace(go.Scattermap(lat=[C.COUNTRY_LABELS[k][0] for k in names],
                                lon=[C.COUNTRY_LABELS[k][1] for k in names], mode="text", text=names,
                                textfont=dict(color="rgba(255,255,255,0.9)", size=11 if small else 13),
                                hoverinfo="skip", showlegend=False))
    cn = list(C.CITIES)
    fig.add_trace(go.Scattermap(lat=[C.CITIES[k][0] for k in cn], lon=[C.CITIES[k][1] for k in cn],
                                mode="markers+text", text=cn, textposition="top right",
                                textfont=dict(color="white", size=10 if small else 12),
                                marker=dict(size=6 if small else 7, color="white"), hoverinfo="text",
                                showlegend=False))
    if box:
        w, e, s, n = box
        fig.add_trace(go.Scattermap(lon=[w, e, e, w, w], lat=[s, s, n, n, s], mode="lines", showlegend=False,
                                    hoverinfo="skip", line=dict(color=SEL, width=2.5)))
    return fig


def area_view(small=False):
    """Centre and zoom for the selected area (whole study area: the default view)."""
    bounds = None
    if box:
        bounds = box
    elif region != WHOLE and provinces():
        from shapely.geometry import shape
        f = [f for f in provinces()["features"] if f["properties"]["region"] == region]
        if f:
            x0, y0, x1, y1 = shape(f[0]["geometry"]).bounds
            bounds = (x0, x1, y0, y1)
    base = (3.55 if small else 4.85)
    if not bounds:
        return {"lat": 0.6, "lon": 107.0}, base
    w, e, s_, n_ = bounds
    width_px, height_px = (430, 470) if small else (950, 600)
    ppd = min(width_px / max((e - w) * 1.4, 0.5), height_px / max((n_ - s_) * 1.4, 0.5))
    zoom = float(np.clip(np.log2(ppd * 360 / 512), base, 7.5))
    return {"lat": (s_ + n_) / 2, "lon": (w + e) / 2}, zoom


def fire_map(df, metric="value", animate=None, h=600, small=False, colorbar=True):
    df = df[df[metric] >= 0.5].copy()
    df["logv"] = np.log10(df[metric].clip(lower=1)).clip(upper=LOG_MAX)
    df["Hotspots"] = df[metric].round(0)
    kw = dict(geojson=GJ, locations="cid", color="logv", color_continuous_scale=FIRE_SCALE,
              range_color=[0, LOG_MAX], map_style=MAP_STYLE, zoom=area_view(small)[1],
              center=area_view(small)[0], opacity=0.85,
              hover_data={"cid": False, "logv": False, "Hotspots": ":,.0f", "region": True},
              labels={"region": "Region"})
    if animate:
        kw["animation_frame"] = animate
    fig = px.choropleth_map(df, **kw)
    fig.update_traces(marker_line_width=0.3, marker_line_color="rgba(128,0,38,0.55)")
    fig.update_layout(height=h, paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=70 if colorbar else 0, t=0, b=0),
                      map_layers=map_layers(), hoverlabel=dict(bgcolor="white", font=dict(color=INK)),
                      coloraxis_showscale=colorbar,
                      coloraxis_colorbar=dict(title=dict(text="Hotspots<br>per cell", font=dict(size=11, color=MUTED)),
                                              tickvals=[np.log10(t) for t in TICKS], ticktext=TICK_TEXT,
                                              tickfont=dict(color=MUTED, size=11), x=1.01, xanchor="left",
                                              thickness=12, len=0.7))
    if animate and fig.layout.updatemenus:
        fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 800
        fig.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 250
    return add_labels(fig, small)


def available(metric, ym):
    if metric == "modis" and ym >= C.MODIS_PARTIAL_FROM:
        return ("Raw MODIS is not available from Jan 2026 for the whole area (2026 MODIS was downloaded for "
                "Malaysia only). Showing HazeTrace.")
    if metric == "viirs_raw" and ym < C.SNPP_START:
        return "Raw VIIRS starts in Feb 2012 (VIIRS was launched in late 2011). Showing HazeTrace."
    return None


CREDIT = ("Basemap: NASA Blue Marble (via NASA GIBS). Fires: NASA FIRMS MODIS and VIIRS, harmonized by HazeTrace. "
          "Each square is a 0.25° cell (about 28 km). Yellow outline: selected region.")

# ---------------- map tab -------------------------------------------------------------------------
with tab_map:
    c1, c2, c3 = st.columns([2.2, 1, 1])
    view = c1.radio("View", ["One month", "Side by side", "Time-lapse"], horizontal=True)
    if view != "Time-lapse":
        yr = c2.selectbox("Year", years[::-1], index=0)
        months_in = sorted({int(x[5:7]) for x in cells.ym.unique() if x.startswith(str(yr))})
        default_mo = mo_latest if yr == yr_latest else (9 if 9 in months_in else months_in[-1])
        mo = c3.selectbox("Month", months_in, index=months_in.index(default_mo),
                          format_func=lambda m: I.MONTHS[m - 1])
        ym = f"{yr}-{mo:02d}"
        partial = ym == C.PARTIAL_MONTH
        month_all = cells[cells.ym == ym]
        msel = sel[sel.ym == ym]
        tot = {k: float(msel[k].sum()) for k in ["value", "viirs_raw", "modis"]}
        lbl = I.month_label(ym)
        ptxt = f" Partial month (data to {C.DATA_END_DATE}), preliminary, not ranked." if partial else ""

    if view == "One month":
        show = st.segmented_control("Show", list(METRICS), default="HazeTrace (harmonized)") or \
            "HazeTrace (harmonized)"
        metric = METRICS[show]
        why = available(metric, ym)
        if why:
            st.info(why)
            metric, show = "value", "HazeTrace (harmonized)"
        v, rv, rm = tot["value"], tot["viirs_raw"], tot["modis"]
        pre = ym < C.SNPP_START
        if metric == "value" and pre:
            txt = (f"<b>HazeTrace: {v:,.0f} hotspots</b> observed by MODIS. Before 2012 the record is MODIS "
                   "itself, so no conversion is needed.")
        elif metric == "value":
            txt = (f"<b>HazeTrace: {v:,.0f} hotspots</b> on the MODIS scale, comparable with every year back to "
                   f"2003. Raw VIIRS counted {rv:,.0f}.")
        elif metric == "viirs_raw":
            fac = f"about {rv / v:.1f}× more" if v >= 1 else "more"
            txt = (f"<b>Raw VIIRS: {rv:,.0f} hotspots.</b> VIIRS (375 m pixels) sees smaller fires than MODIS "
                   f"(1 km), so it counts {fac} for the same burning. HazeTrace converts this to {v:,.0f}.")
        elif pre:
            txt = f"<b>Raw MODIS: {rm:,.0f} hotspots.</b> Before 2012 this is the HazeTrace record itself."
        else:
            diff = f" ({(v / rm - 1) * 100:+.0f}%)" if rm >= 1 else ""
            txt = (f"<b>Raw MODIS: {rm:,.0f} hotspots</b> observed. HazeTrace estimate from VIIRS alone: "
                   f"{v:,.0f}{diff}. This is how the model is checked.")
        st.markdown(f'<div class="ht-explain">{lbl} · {area_label}: {txt}{ptxt}</div>', unsafe_allow_html=True)
        if month_all.empty:
            st.info("No hotspots in the study area this month.")
        else:
            st.plotly_chart(fire_map(month_all, metric), width="stretch", config=CFG)
    elif view == "Side by side":
        if partial:
            st.markdown(f'<div class="ht-explain">{lbl}.{ptxt}</div>', unsafe_allow_html=True)
        lm, ltitle = ("viirs_raw", "Raw VIIRS") if ym >= C.SNPP_START else ("modis", "Raw MODIS")
        a, b = st.columns(2)
        with a:
            st.markdown(f"**{ltitle}** · {tot[lm]:,.0f} hotspots")
            st.plotly_chart(fire_map(month_all, lm, h=470, small=True), width="stretch",
                            config=CFG, key="sbs_l")
        with b:
            st.markdown(f"**HazeTrace (harmonized)** · {tot['value']:,.0f} hotspots")
            st.plotly_chart(fire_map(month_all, "value", h=470, small=True), width="stretch", config=CFG,
                            key="sbs_r")
        st.caption("Same month, same colour scale. Raw VIIRS looks far worse because it sees smaller fires, not "
                   "because there is more burning. HazeTrace brings it onto the MODIS scale used since 2003."
                   if lm == "viirs_raw" else
                   "Before 2012 only MODIS exists, so HazeTrace is the MODIS record itself and both maps match.")
    else:
        play = c2.selectbox("Play", ["Haze seasons 2003 to 2026", "Months of one year"])
        if play.startswith("Months"):
            ly = c3.selectbox("Year", years[::-1], index=0, key="ly")
        show = st.segmented_control("Show", ["HazeTrace (harmonized)", "Raw VIIRS"],
                                    default="HazeTrace (harmonized)", key="lapse_show") or "HazeTrace (harmonized)"
        metric = METRICS[show]
        if play.startswith("Haze"):
            d = cells[cells.ym.str[5:7].isin(["08", "09", "10"])]
            if metric == "viirs_raw":
                d = d[d.ym >= C.SNPP_START]
            d = d.groupby(["cid", "region", d.ym.str[:4].rename("Season")])[metric].sum().reset_index()
            d = d.sort_values("Season")
            frame = "Season"
            shown = d[d.cid.isin(set(sel.cid))].groupby("Season")[metric].sum()
            top3 = ", ".join(shown.nlargest(3).index)
            first = "2012" if metric == "viirs_raw" else str(years[0])
            st.caption(f"Press ▶ to play one frame per haze season (Aug to Oct total). Most {show} hotspots in this "
                       f"area, {first} to 2026: {top3}. 2026 is preliminary."
                       + (" Raw VIIRS starts in 2012." if metric == "viirs_raw" else ""))
        else:
            d = cells[cells.ym.str.startswith(str(ly))].copy()
            d["Month"] = d.ym.map(I.month_label)
            d = d.sort_values("ym")
            frame = "Month"
            if metric == "viirs_raw" and ly < 2012:
                st.info("Raw VIIRS starts in 2012. Showing HazeTrace.")
                metric = "value"
            st.caption("Press ▶ to play. Watch the burning build from July and peak in September and October.")
        st.plotly_chart(fire_map(d, metric, animate=frame, h=640), width="stretch", config=CFG)
    st.markdown(f'<div class="ht-credit">{CREDIT}</div>', unsafe_allow_html=True)

    if view != "Time-lapse":
        st.markdown("#### Regions this month")
        rt = region_rows(ym)
        tbl = pd.DataFrame({
            "Region": rt.Region,
            "Hotspots": rt.Hotspots.round(0).astype(int),
            "Share of all hotspots": rt.Share.map(lambda x: f"{x:.0f}%"),
            "vs typical month": rt["x median"].map(lambda r: f"{r:.1f}×" if pd.notna(r) else "n/a")
            if not partial else "not ranked",
            "Rank this month": rt["Month rank"] if not partial else "not ranked",
            "Haze season rank": rt["Season rank"].map(lambda r: f"{int(r)}" if pd.notna(r) else "n/a")
            if not partial else "not ranked",
            "Level": rt.Level if not partial else "not ranked",
        })
        st.dataframe(tbl, hide_index=True, width="stretch")
        st.caption("Share of all hotspots in the study area, including areas outside the six named regions. Smoke "
                   "can cross borders with the wind; HazeTrace does not model smoke or attribute sources.")

# ---------------- calendar tab --------------------------------------------------------------------
with tab_cal:
    st.markdown(f"#### Fire calendar · {area_label}")
    grid = series.pivot(index="year", columns="month", values="value").reindex(columns=range(1, 13))
    last_full = pd.Period(latest, "M")
    for y in grid.index:
        for m in grid.columns:
            if pd.Period(f"{y}-{m:02d}", "M") > last_full:
                grid.loc[y, m] = np.nan
    z = np.log10(grid.clip(lower=1))
    fig = go.Figure(go.Heatmap(
        z=z.values, x=I.MONTHS, y=grid.index.astype(str), customdata=grid.values,
        colorscale=CAL_SCALE, zmin=0, zmax=max(float(np.nanmax(z.values)), 1), xgap=2, ygap=2,
        hovertemplate="%{x} %{y}: %{customdata:,.0f} hotspots<extra></extra>",
        colorbar=dict(title=dict(text="Hotspots", font=dict(color=MUTED, size=11)),
                      tickvals=[0, 1, 2, 3, 4, 5], ticktext=LOG_TEXT[:6], tickfont=dict(color=MUTED))))
    if C.PARTIAL_MONTH[:4] in grid.index.astype(str):
        fig.add_annotation(x=I.MONTHS[int(C.PARTIAL_MONTH[5:7]) - 1], y=C.PARTIAL_MONTH[:4], text="to 13 Nov",
                           showarrow=False, font=dict(size=9, color=MUTED))
    fig.update_layout(yaxis=dict(autorange="reversed", dtick=1))
    st.plotly_chart(style(fig, max(480, 22 * len(grid))), width="stretch", config=CFG)
    st.caption("Monthly hotspots on the MODIS scale. Before Feb 2012: MODIS observed. From Feb 2012: VIIRS converted "
               "by the model, so the sensor change does not show. Aug to Oct 2026 is preliminary; Nov 2026 is partial "
               "and left empty.")
    st.markdown("**Haze-season totals (Aug to Oct)**")
    s2 = seas.reset_index()
    fig = go.Figure(go.Bar(x=s2.year, y=s2.value, marker_color=[FIRE if y == yr_latest else "#C9C3BA" for y in s2.year],
                           hovertemplate="%{x}: %{y:,.0f} hotspots<extra></extra>"))
    fig.update_yaxes(title="Aug to Oct hotspots")
    st.plotly_chart(style(fig, 280), width="stretch", config=CFG)

# ---------------- how good is it? -----------------------------------------------------------------
with tab_good:
    o = meta["overall"]
    k = meta.get("check_malaysia_2026") or {}
    lost = ", ".join(str(y) for y in by_year.year[by_year.model_MAE >= by_year.base_MAE])
    a, b, c = st.columns(3)
    a.markdown(f'<div class="ht-card"><h4>Tested honestly</h4><p>Each year from {by_year.year.min()} to '
               f'{by_year.year.max()} was hidden from the model in turn and predicted from VIIRS alone. Average error '
               f'<b>{o["model_MAE"]:.0f}</b> hotspots per region-month, against <b>{o["base_MAE"]:.0f}</b> for a fixed '
               f'conversion ratio. R² {o["model_R2"]:.2f} vs {o["base_R2"]:.2f}. The model wins {wins} of '
               f'{len(by_year)} years' + (f'; the fixed ratio is as good or better in {lost}.' if lost else '.') +
               '</p></div>', unsafe_allow_html=True)
    if k:
        b.markdown(f'<div class="ht-card"><h4>Independent 2026 check, Malaysia</h4><p>MODIS 2026 data for Malaysia '
                   f'were never used for training. Observed MODIS: <b>{k["modis_observed"]:,}</b> hotspots '
                   f'({k["months"].replace(" to ", " to ")}). HazeTrace estimate from VIIRS: '
                   f'<b>{k["harmonized"]:,.0f}</b> ({k["total_error_pct"]:+.0f}%). Weakest where fires are few, as '
                   f'expected.</p></div>', unsafe_allow_html=True)
    if n20:
        c.markdown(f'<div class="ht-card"><h4>Transfer to NOAA-20</h4><p>Trained on S-NPP only, then applied to '
                   f'NOAA-20 (2018 to 2025) and compared with MODIS. Error <b>{n20["MAE"]:.0f}</b> vs '
                   f'<b>{n20b["MAE"]:.0f}</b> for a fixed ratio; R² {n20["R2"]:.2f} vs {n20b["R2"]:.2f}; bias '
                   f'{n20["bias_pct"]:+.1f}% vs {n20b["bias_pct"]:+.1f}%.</p></div>', unsafe_allow_html=True)
    st.write("")
    st.markdown(f"#### Raw sensors vs HazeTrace · {area_label}")
    tr = series.copy()
    if region != "Sarawak (Malaysia)":
        tr.loc[tr.ym >= C.MODIS_PARTIAL_FROM, "modis"] = np.nan
    tr.loc[tr.ym < C.SNPP_START, "viirs"] = np.nan
    for col in ["value", "lo", "hi", "modis", "viirs"]:
        tr[col] = tr[col].where(tr[col] > 0)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=tr.ym, y=tr.hi, line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=tr.ym, y=tr.lo, fill="tonexty", line=dict(width=0), fillcolor="rgba(217,72,15,0.15)",
                             name="Likely range (approx. 10 to 90%)", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=tr.ym, y=tr.viirs, name="Raw VIIRS", line=dict(color=VIIRS), visible="legendonly"))
    fig.add_trace(go.Scatter(x=tr.ym, y=tr.modis, name="Raw MODIS", line=dict(color=MODIS, width=1.6)))
    fig.add_trace(go.Scatter(x=tr.ym, y=tr.value, name="HazeTrace", line=dict(color=FIRE, width=2)))
    fig.update_traces(hoverinfo="skip", hovertemplate=None, selector=dict(type="scatter"))
    fig.data[4].update(customdata=[f"HazeTrace: {fmt_n(a)}<br>Raw MODIS: {fmt_n(m)}<br>Raw VIIRS: {fmt_n(v)}"
                                   for a, m, v in zip(tr.value, tr.modis, tr.viirs)],
                       hovertemplate="%{customdata}<extra></extra>", hoverinfo=None)
    fig.update_layout(hovermode="x unified", legend=dict(orientation="h", y=1.1))
    log_y(style(fig, 380), "Hotspots per month (log scale)")
    st.plotly_chart(fig, width="stretch", config=CFG)
    st.caption("From 2012 to 2025 the HazeTrace line (made from VIIRS only) should sit on the MODIS line. Click Raw "
               "VIIRS in the legend to see how far off the raw counts are.")
    c1, c2 = st.container(), st.columns([1.4, 1])[0]
    with c1:
        st.markdown("**Held-out years, all regions together** (region × month)")
        yt = by_year.rename(columns={
            "year": "Year", "model_MAE": "Model error (hotspots)", "base_MAE": "Fixed-ratio error (hotspots)",
            "model_R2": "Model R²", "base_R2": "Fixed-ratio R²", "model_bias_pct": "Model bias %",
            "base_bias_pct": "Fixed-ratio bias %", "model_n": "Region-months"})
        yt = yt[["Year", "Model error (hotspots)", "Fixed-ratio error (hotspots)", "Model R²", "Fixed-ratio R²",
                 "Model bias %", "Fixed-ratio bias %", "Region-months"]].round(2)
        st.dataframe(yt, hide_index=True, width="stretch",
                     column_config={"Year": st.column_config.NumberColumn(format="%d")})
    with c2:
        st.markdown("**What drives the conversion** (SHAP)")
        names = {"log_v_mean_frp": "Fire intensity (mean FRP)", "region_code": "Region",
                 "log_v_region_month": "Total fires in the region that month", "month_cos": "Time of year",
                 "month_sin": "Time of year", "v_frac_night": "Share of night detections",
                 "v_frac_high": "Share of high-confidence detections", "lc_code": "Land type"}
        im = imp.assign(feature=imp.feature.map(names).fillna(imp.feature)).groupby("feature", as_index=False)[
            "mean_abs_shap"].sum()
        im = im[im.mean_abs_shap > 0].sort_values("mean_abs_shap", ascending=False)
        fig = go.Figure(go.Bar(x=im.mean_abs_shap, y=im.feature, orientation="h", marker_color=FIRE,
                               hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
        fig.update_layout(yaxis=dict(autorange="reversed"))
        fig.update_xaxes(title="Average effect on the ratio (SHAP)")
        st.plotly_chart(style(fig, 280), width="stretch", config=CFG)
    with st.expander("Known limits"):
        st.markdown(
            "- Hotspots on the MODIS scale are a comparable index, not the true number of fires.\n"
            "- Clouds and thick smoke hide fires from both sensors.\n"
            "- The model is weakest where fires are few (Malaysia 2026 check, quiet years in Riau). South Sumatra "
            "has the lowest R².\n"
            "- A cell-month where VIIRS saw no fire counts as 0, even if MODIS saw one. In big seasons this hardly "
            "matters, but in quiet years HazeTrace can be well below Raw MODIS (Riau 2021: 57 vs 298 for Aug to Oct). "
            "A warning appears under the first chart when this happens in the selected area.\n"
            "- The regional activity feature was chosen after looking at all years, so the test is slightly "
            "optimistic.\n"
            "- Aug to Nov 2026 is near-real-time data. It has no fire type flag, so 500 m spots flagged as static "
            "sources (for example gas flares) in at least 2 years are removed. Tested: about 2% of real fires lost, "
            "net change about 0.2%. Terra MODIS 26 and 27 Mar 2026 removed.\n"
            "- 2026 MODIS covers Malaysia only, so training and testing stop at Dec 2025.\n"
            "- Regions use Natural Earth province boundaries, simplified for speed.")

# ---------------- where it works ------------------------------------------------------------------
with tab_where:
    st.markdown("#### The MODIS/VIIRS ratio is not constant")
    raw = cells[(cells.ym >= C.SNPP_START) & (cells.ym <= C.PAIR_END)].copy()
    raw["year"] = raw.ym.str[:4].astype(int)
    rr = raw.groupby(["region", "year"]).agg(m=("modis", "sum"), v=("viirs_raw", "sum")).reset_index()
    rr = rr[(rr.region != "Other") & (rr.v >= 500)]
    rr["ratio"] = rr.m / rr.v
    fig = px.line(rr, x="year", y="ratio", color="region", markers=True,
                  labels={"ratio": "MODIS hotspots per VIIRS hotspot", "year": "", "region": ""},
                  color_discrete_sequence=["#D9480F", "#2B6CB0", "#2F9E44", "#AE3EC9", "#F59F00", "#495057"])
    fig.update_traces(hovertemplate="%{x}: %{y:.2f}")
    fig.update_layout(legend=dict(orientation="h", y=1.12))
    st.plotly_chart(style(fig, 360), width="stretch", config=CFG)
    st.caption("The ratio changes from year to year and region to region, so one fixed ratio would distort trends. "
               "Low ratios in big seasons such as 2014 and 2015 are consistent with many small, low-intensity fires "
               "(often peat) that VIIRS (375 m) sees but MODIS (1 km) misses; this is an interpretation, hotspot counts "
               "cannot prove the fuel type. Riau's ratio rose to about 0.5 in the quiet years 2020 to 2024. Years with "
               "fewer than 500 VIIRS hotspots in a region are hidden.")
    c1, c2 = st.columns([2.4, 1], gap="large")
    with c1:
        st.markdown("**Accuracy by region** (held-out years)")
        g = by_group[by_group.kind == "region"].rename(columns={
            "group": "Region", "model_MAE": "Model error (hotspots)", "base_MAE": "Fixed-ratio error (hotspots)",
            "model_R2": "Model R²", "base_R2": "Fixed-ratio R²"})
        st.dataframe(g[["Region", "Model error (hotspots)", "Fixed-ratio error (hotspots)", "Model R²",
                        "Fixed-ratio R²"]].round(2), hide_index=True, width="stretch")
        better = [r for r, m_, b_ in zip(g.Region, g["Model error (hotspots)"], g["Fixed-ratio error (hotspots)"])
                  if m_ >= b_ and r != "All regions"]
        if better:
            st.caption("The fixed ratio is as good or slightly better in: " + ", ".join(better) + ". The model's gain "
                       "comes mainly from the big burning regions.")
        if meta.get("regions") != "provinces":
            st.caption("Computed with the earlier rectangular region boxes. Re-run scripts/02_harmonize.py to update "
                       "with the province boundaries used everywhere else in the app.")
        else:
            st.caption("Computed with the Natural Earth province boundaries used everywhere in the app.")
    with c2:
        lc = by_group[(by_group.kind == "landcover") & (by_group.group != "unknown")]
        if len(lc):
            st.markdown("**Accuracy by land type**")
            st.dataframe(lc.rename(columns={"group": "Land type", "model_MAE": "Model error (hotspots)",
                                            "model_R2": "Model R²"})[["Land type", "Model error (hotspots)",
                                                                       "Model R²"]].round(2),
                         hide_index=True, width="stretch")
        else:
            st.markdown('<div class="ht-card"><h4>Next: peat vs forest vs plantation</h4><p>The plan is to show where '
                        'harmonization works and where it fails by land type. The pipeline already supports it: add '
                        '<code>data/landcover_cells.csv</code> (MODIS MCD12Q1 land cover plus a peat map per 0.25° '
                        'cell) and re-run the model.</p><p>The low ratios in big seasons above are a first hint, not '
                        'proof, of why it matters.</p></div>', unsafe_allow_html=True)

# ---------------- who it's for --------------------------------------------------------------------
with tab_who:
    st.markdown("#### Who it is for")
    st.caption("HazeTrace counts fires seen from space. It does not measure haze or air quality and does not say "
               "whether it is safe to go outside.")
    srank_txt = (f"{yr_latest} has the {ordinal(season_rank)} most fire hotspots of any Aug to Oct season since "
                 f"{years[0]} in this area."
                 if season_rank else "See how this season compares with every year since 2003.")
    personas = [
        ("Air-quality analyst, Malaysia", "Is the burning upwind of us unusual this month, and where?",
         "Summary card and Map: level, rank and the regions table."),
        ("Provincial disaster officer, Indonesia", "How does this season in my province compare with 2015 and 2019?",
         "Pick the province, then Calendar & seasons."),
        ("Plantation or ESG no-burning analyst", "Did hotspots in our area rise compared with past years?",
         "My own box, then download the monthly series with its likely range."),
        ("Resident, teacher or journalist in KL", "Why is it hazy, and is this year bad?",
         f"Time-lapse and summary card: {srank_txt}"),
    ]
    cols = st.columns(4)
    for col, (who, q, ans) in zip(cols, personas):
        col.markdown(f'<div class="ht-card"><h4>{who}</h4><p><i>"{q}"</i></p><p>→ {ans}</p></div>',
                     unsafe_allow_html=True)
    st.write("")
    st.markdown("#### Sustainable Development Goals")
    sdgs = [
        ("#3F7E44", "SDG 13.1", "Resilience to climate hazards",
         "A consistent fire-season record since 2003 for planning and preparedness."),
        ("#3F7E44", "SDG 13.3", "Awareness",
         "Makes fire history readable for schools, media and the public. It informs; it does not warn."),
        ("#56C02B", "SDG 15.1", "Forests",
         "Trends for any area support forest and no-burning monitoring."),
        ("#56C02B", "SDG 15.3", "Land degradation",
         "Low MODIS/VIIRS ratios in big seasons are consistent with small peat fires (an interpretation)."),
        ("#19486A", "SDG 17.18", "Reliable open data",
         "Open NASA data, harmonized, with published tests and downloadable CSV."),
    ]
    cols = st.columns(5)
    for col, (c, code, name, why_) in zip(cols, sdgs):
        col.markdown(f'<div class="ht-card"><span class="ht-badge" style="background:{c}">{code}</span>'
                     f'<h4>{name}</h4><p>{why_}</p></div>', unsafe_allow_html=True)
    st.caption("Motivation only, not a claimed impact: haze harms health (SDG 3.9) and city air (SDG 11.6). HazeTrace "
               "has no health or PM2.5 data and makes no health claims.")
    st.write("")
    st.markdown("#### Why I built it")
    st.markdown("I live on the 31st floor in Kuala Lumpur. Every year from August to October I can't open my window "
                "because of the haze. That smoke starts as fire hotspots hundreds of kilometres away, but their "
                "20-year history is split across two satellite sensors. HazeTrace joins them into one record, so "
                "anyone can see whether this season is unusual.")

# ---------------- data & method -------------------------------------------------------------------
with tab_data:
    st.markdown(f"#### Download · {area_label}")
    y0, y1 = st.select_slider("Years", options=years, value=(years[0], years[-1]))
    exp = series.copy()
    exp.loc[exp.ym < C.SNPP_START, "viirs"] = np.nan                  # VIIRS did not exist yet
    if not set(sel.cid) <= malaysia_cells():
        exp.loc[exp.ym >= C.MODIS_PARTIAL_FROM, "modis"] = np.nan     # 2026 MODIS covers Malaysia only
    out = exp[exp.year.between(y0, y1)][["ym", "value", "lo", "hi", "modis", "viirs", "nrt"]].rename(columns={
        "ym": "month", "value": "hotspots_modis_scale", "lo": "likely_low", "hi": "likely_high",
        "modis": "modis_raw", "viirs": "viirs_raw", "nrt": "preliminary"}).round(1)
    out["partial_month"] = out.month == C.PARTIAL_MONTH
    safe = "".join(ch if ch.isalnum() else "_" for ch in area_label).strip("_")
    c1, c2 = st.columns(2)
    c1.download_button("⬇️ Monthly series (CSV)", out.to_csv(index=False), f"hazetrace_{safe}_{y0}-{y1}.csv",
                       "text/csv", width="stretch")
    cell_sel = sel[sel.ym.str[:4].astype(int).between(y0, y1)].drop(columns=["cid"]).copy()
    cell_sel.loc[cell_sel.ym < C.SNPP_START, "viirs_raw"] = np.nan
    cell_sel.loc[(cell_sel.ym >= C.MODIS_PARTIAL_FROM) & ~sel.loc[cell_sel.index, "cid"].isin(malaysia_cells(strict=True)),
                 "modis"] = np.nan
    if (cell_sel.landcover == "unknown").all():
        cell_sel = cell_sel.drop(columns="landcover")
    cell_sel = cell_sel.rename(columns={"ym": "month", "value": "hotspots_modis_scale", "lo": "likely_low",
                                        "hi": "likely_high", "modis": "modis_raw", "nrt": "preliminary"})
    cell_sel[["hotspots_modis_scale", "likely_low", "likely_high"]] = \
        cell_sel[["hotspots_modis_scale", "likely_low", "likely_high"]].round(2)
    cell_sel["partial_month"] = cell_sel.month == C.PARTIAL_MONTH
    c2.download_button("⬇️ Grid-cell data, 0.25° (CSV)", cell_sel.to_csv(index=False),
                       f"hazetrace_cells_{safe}_{y0}-{y1}.csv", "text/csv", width="stretch")
    st.caption("Blank = no observation: raw VIIRS before Feb 2012 (not launched yet) and raw MODIS from Jan 2026 "
               "outside Malaysia (only Malaysia was downloaded; in the grid-cell file also for border cells that include Indonesian land). 0 = the sensor looked and detected nothing.")
    with st.expander("What each column means"):
        st.markdown(f"""
- **month**: year and month (YYYY-MM). **cell_lat, cell_lon** (grid file): centre of the 0.25° cell, in degrees.
- **hotspots_modis_scale**: HazeTrace hotspot count on the MODIS scale. Use this one to compare years.
- **likely_low, likely_high**: approximate 10 to 90% range of that estimate.
- **modis_raw, viirs_raw**: counts straight from each sensor (VIIRS: S-NPP, or NOAA-20 after S-NPP stopped).
- **source** (grid file): whether the value is MODIS observed or converted from VIIRS.
- **preliminary**: near-real-time data, replaced by final data about 5 months later.
- **partial_month**: True for {I.month_label(C.PARTIAL_MONTH)}, which only runs to {C.DATA_END_DATE}.
- Units: number of fire hotspots per month (per cell in the grid file).
- The grid file lists only cell-months where at least one sensor detected a fire; a missing row means no detection.
- Area: {area_label}. Source: NASA FIRMS (https://earthdata.nasa.gov/firms), harmonized by HazeTrace.
""")
    st.dataframe(out, hide_index=True, width="stretch")
    st.markdown("#### Method")
    st.markdown(f"""
- **Problem:** MODIS (since 2000, now ending) and VIIRS (since 2012) detect fires differently, so their raw counts
  cannot be joined. S-NPP VIIRS stopped on 2 Nov 2026; NOAA-20 and NOAA-21 continue.
- **Model:** MODIS and VIIRS S-NPP counts are paired per 0.25° cell and month (2012 to 2025). A LightGBM Poisson model
  learns the MODIS/VIIRS ratio (VIIRS count as offset) and is compared with a fixed regional ratio by leaving one
  year out at a time. The better one ({meta['method_used']}) builds the record from 2003.
- **Not a forecast** and not an official alert. HazeTrace describes fire activity compared with history.
- **Data:** NASA FIRMS active fires, MODIS Collection 6.1 and VIIRS 375 m (S-NPP, NOAA-20), data up to
  {C.DATA_END_DATE}. Basemap: NASA Blue Marble via NASA GIBS. Borders and provinces: Natural Earth (public domain).
  Cite: NASA FIRMS, https://earthdata.nasa.gov/firms
- **Use of AI:** code and text drafted with help from Claude (Anthropic); every number comes from the code and NASA
  data.

**HazeTrace** · NASA Space Apps Challenge 2026 · Kuala Lumpur · Team Solo Orbit KL ·
Challenge: *Harmonization of MODIS and VIIRS Hot Spots*
""")
