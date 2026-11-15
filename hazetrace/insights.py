"""Derived numbers for the dashboard, KPIs and the data note. Pure pandas, no Streamlit."""
import numpy as np
import pandas as pd

from hazetrace import config as C

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
SEASON = (8, 9, 10)          # Aug-Oct haze season
CLIM_END_YEAR = 2025         # climatology (median, ranks) uses complete standard years only
MIN_FOR_LEVEL = 50           # below this many hotspots a month is never labelled above Typical


def month_label(ym: str) -> str:
    return f"{MONTHS[int(ym[5:7]) - 1]} {ym[:4]}"


def monthly_series(cells: pd.DataFrame, first="2003-01", last=None) -> pd.DataFrame:
    s = cells.groupby("ym").agg(value=("value", "sum"), lo=("lo", "sum"), hi=("hi", "sum"),
                                modis=("modis", "sum"), viirs=("viirs_raw", "sum"),
                                nrt=("nrt", "any")).reset_index()
    last = last or cells.ym.max()
    full = pd.period_range(first, last, freq="M").astype(str)
    s = s.set_index("ym").reindex(full).rename_axis("ym").reset_index()
    s[["value", "lo", "hi", "modis", "viirs"]] = s[["value", "lo", "hi", "modis", "viirs"]].fillna(0)
    s["nrt"] = s["nrt"].fillna(False).astype(bool)
    s["year"] = s.ym.str[:4].astype(int)
    s["month"] = s.ym.str[5:7].astype(int)
    return s


def month_stats(series: pd.DataFrame, ym: str) -> dict:
    """Value of one month vs the same calendar month in other years."""
    yr, mo = int(ym[:4]), int(ym[5:7])
    row = series[series.ym == ym]
    if row.empty:
        return {}
    v = float(row.value.iloc[0])
    pool = series[(series.month == mo) & series.value.notna()]
    clim = pool[(pool.year != yr) & (pool.year <= CLIM_END_YEAR)].value
    med = float(clim.median()) if len(clim) else 0.0
    rank = int((pool.value > v).sum()) + 1
    return {"ym": ym, "label": month_label(ym), "value": v, "lo": float(row.lo.iloc[0]), "hi": float(row.hi.iloc[0]),
            "median": med, "ratio": (v / med) if med >= 10 else None, "rank": rank, "n": int(len(pool)),
            "nrt": bool(row.nrt.iloc[0])}


def season_table(series: pd.DataFrame) -> pd.DataFrame:
    s = series[series.month.isin(SEASON)].groupby("year").agg(value=("value", "sum"), nrt=("nrt", "any"))
    s["rank"] = s.value.rank(ascending=False, method="min").astype(int)
    return s


def level_for(stats: dict, season_rank=None) -> str:
    """Statistical description of how unusual a month is. Not an alert level."""
    if not stats or stats["value"] < MIN_FOR_LEVEL:
        return "Typical"
    r, n, ratio = stats["rank"], stats["n"], stats["ratio"] or 0
    if r <= 2 or (season_rank is not None and season_rank <= 3):
        return "Exceptional"
    if ratio >= 3 or r <= 4:
        return "Well above usual"
    if ratio >= 1.5 or r <= max(1, round(n * 0.25)):
        return "Above usual"
    return "Typical"


def latest_complete_month(cells: pd.DataFrame) -> str:
    """Last month that is not the partial current month (data run to ~13 Nov 2026)."""
    last = cells.ym.max()
    return (pd.Period(last, "M") - 1).strftime("%Y-%m") if last == C.PARTIAL_MONTH else last


def region_table(cells: pd.DataFrame, ym: str) -> pd.DataFrame:
    rows = []
    for name in C.REGION_ORDER:
        s = monthly_series(cells[cells.region == name])
        st_ = month_stats(s, ym)
        seas = season_table(s)
        yr = int(ym[:4])
        rows.append({"Region": name, "Hotspots": st_.get("value", 0), "x median": st_.get("ratio"),
                     "Month rank": f"{st_.get('rank')} of {st_.get('n')}" if st_ else "-",
                     "Season rank": int(seas.loc[yr, "rank"]) if yr in seas.index else None,
                     "Level": level_for(st_)})
    t = pd.DataFrame(rows)
    total = t.Hotspots.sum()
    t["Share"] = (t.Hotspots / total * 100).round(0) if total else 0
    return t.sort_values("Hotspots", ascending=False)


def cell_geojson(cells: pd.DataFrame) -> dict:
    """True 0.25-degree squares for every grid cell that ever had a detection."""
    h = C.GRID_DEG / 2
    feats = []
    for lat, lon in cells[["cell_lat", "cell_lon"]].drop_duplicates().itertuples(index=False):
        cid = f"{lat:.3f}_{lon:.3f}"
        ring = [[lon - h, lat - h], [lon + h, lat - h], [lon + h, lat + h], [lon - h, lat + h], [lon - h, lat - h]]
        feats.append({"type": "Feature", "id": cid, "properties": {}, "geometry": {"type": "Polygon", "coordinates": [ring]}})
    return {"type": "FeatureCollection", "features": feats}


def cell_id(df: pd.DataFrame) -> pd.Series:
    return df.cell_lat.map("{:.3f}".format) + "_" + df.cell_lon.map("{:.3f}".format)
