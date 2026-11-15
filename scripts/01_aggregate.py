"""Step 1: FIRMS hotspot CSVs -> grid cell x month counts per sensor.

Put every CSV from the FIRMS archive download (fire_archive_*.csv and
fire_nrt_*.csv, MODIS and VIIRS) anywhere under data/raw/, then run:

    python scripts/01_aggregate.py

Output: data/processed/cell_month.parquet with one row per
(sensor, cell, month): count, frp_sum, n_high, n_night, n_nrt.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hazetrace import config as C  # noqa: E402

CHUNK = 500_000
SAT_MAP = {"N": "snpp", "N20": "noaa20", "1": "noaa20", "N21": "noaa21", "2": "noaa21"}


# Static-source mask settings, chosen on labeled 2026 NOAA-20 data held out from the mask:
# 500 m cells flagged in >= 2 different years catch ~78% of static sources while removing
# ~2% of real fires (net count error about -0.2%). A 2 km box with neighbours removed ~20%.
STATIC_DEG = 0.005
STATIC_MIN_YEARS = 2


def sensor_from_file(path: Path) -> str:
    """VIIRS names are checked first: some FIRMS VIIRS downloads use MODIS-style
    column names (brightness, bright_t31), so columns alone can mislead."""
    name = path.name.upper()
    if "J1V" in name or "JPSS1" in name or "NOAA20" in name or "NOAA-20" in name:
        return "noaa20"
    if "J2V" in name or "JPSS2" in name or "NOAA21" in name or "NOAA-21" in name:
        return "noaa21"
    if "SV-C2" in name or "SUOMI" in name or "SNPP" in name:
        return "snpp"
    if "M-C61" in name or "MODIS" in name:
        return "modis"
    first = pd.read_csv(path, nrows=5)
    first.columns = [c.lower() for c in first.columns]
    if "instrument" in first and first.instrument.astype(str).str.upper().eq("MODIS").all():
        return "modis"
    return "viirs_unknown"  # resolved row by row from the 'satellite' column


def static_key(lat, lon):
    return (np.floor(np.asarray(lat) / STATIC_DEG).astype(np.int64) * 1000000
            + np.floor(np.asarray(lon) / STATIC_DEG).astype(np.int64))


def build_static_mask(files):
    """Locations FIRMS flagged as volcano / static land source / offshore (type 1-3).
    NRT files carry no 'type' column, so their detections near these spots are dropped."""
    parts = []
    for f in files:
        cols = pd.read_csv(f, nrows=0).columns.str.lower()
        if "type" not in cols:
            continue
        use = {"latitude", "longitude", "type", "acq_date"}
        for ch in pd.read_csv(f, usecols=lambda c: c.lower() in use, chunksize=CHUNK):
            ch.columns = [c.lower() for c in ch.columns]
            st = ch[pd.to_numeric(ch["type"], errors="coerce").isin([1, 2, 3])]
            parts.append(pd.DataFrame({"k": static_key(st.latitude, st.longitude),
                                       "y": st.acq_date.astype(str).str[:4]}))
    if not parts:
        return pd.Index([])
    years = pd.concat(parts).drop_duplicates().groupby("k").y.size()
    return pd.Index(years[years >= STATIC_MIN_YEARS].index)


def near_static(df, mask):
    if len(mask) == 0 or df.empty:
        return np.zeros(len(df), bool)
    iy = np.floor(df.latitude.to_numpy() / STATIC_DEG).astype(np.int64)
    ix = np.floor(df.longitude.to_numpy() / STATIC_DEG).astype(np.int64)
    return np.isin(iy * 1000000 + ix, mask.to_numpy())


def clean_chunk(df: pd.DataFrame, sensor: str, is_nrt: bool, static_mask=None, stats=None) -> pd.DataFrame:
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.drop_duplicates(subset=["latitude", "longitude", "acq_date", "acq_time", "satellite"])
    if "type" in df.columns:
        df = df[pd.to_numeric(df["type"], errors="coerce").fillna(C.KEEP_TYPE) == C.KEEP_TYPE]
    elif static_mask is not None:
        hit = near_static(df, static_mask)
        if stats is not None:
            stats["static_removed"] = stats.get("static_removed", 0) + int(hit.sum())
        df = df[~hit]

    w, e, s, n = C.DOMAIN
    df = df[(df.longitude >= w) & (df.longitude < e) & (df.latitude >= s) & (df.latitude < n)]

    if sensor == "modis":
        conf = pd.to_numeric(df["confidence"], errors="coerce")
        df = df[conf >= C.MODIS_MIN_CONFIDENCE].copy()
        df = df[~df["acq_date"].astype(str).isin(C.MODIS_BAD_DATES)]
        df["is_high"] = pd.to_numeric(df["confidence"], errors="coerce") >= 80
        df["sensor"] = "modis"
    else:
        conf = df["confidence"].astype(str).str.strip().str.lower()
        df = df[conf.isin(C.VIIRS_KEEP_CONFIDENCE)].copy()
        df["is_high"] = df["confidence"].astype(str).str.strip().str.lower().isin({"h", "high"})
        if sensor == "viirs_unknown":
            df["sensor"] = df["satellite"].astype(str).str.strip().map(SAT_MAP).fillna("snpp")
        else:
            df["sensor"] = sensor

    if df.empty:
        return df
    df["is_night"] = df["daynight"].astype(str).str.upper().eq("N")
    df["is_nrt"] = is_nrt
    df["ym"] = pd.to_datetime(df["acq_date"]).dt.to_period("M").astype(str)
    g = C.GRID_DEG
    df["cell_lat"] = (np.floor(df.latitude / g) * g + g / 2).round(4)
    df["cell_lon"] = (np.floor(df.longitude / g) * g + g / 2).round(4)
    df["frp"] = pd.to_numeric(df["frp"], errors="coerce").fillna(0.0)
    return df[["sensor", "cell_lat", "cell_lon", "ym", "frp", "is_high", "is_night", "is_nrt"]]


def main():
    files = sorted(C.RAW_DIR.rglob("*.csv"))
    if not files:
        sys.exit(f"No CSV files found under {C.RAW_DIR}")
    static_mask = build_static_mask(files)
    print(f"Static-source mask: {len(static_mask):,} 500 m cells flagged (>= {STATIC_MIN_YEARS} years) "
          "as volcano/industrial/offshore")
    parts = []
    for f in files:
        sensor = sensor_from_file(f)
        is_nrt = bool(re.search(r"nrt", f.name, re.I))
        n_in, stats = 0, {}
        for chunk in pd.read_csv(f, chunksize=CHUNK, low_memory=False):
            n_in += len(chunk)
            c = clean_chunk(chunk, sensor, is_nrt, static_mask, stats)
            if c.empty:
                continue
            agg = c.groupby(["sensor", "cell_lat", "cell_lon", "ym"]).agg(
                count=("frp", "size"), frp_sum=("frp", "sum"),
                n_high=("is_high", "sum"), n_night=("is_night", "sum"),
                n_nrt=("is_nrt", "sum"))
            parts.append(agg)
        extra = f", {stats['static_removed']:,} near static sources removed" if stats.get("static_removed") else ""
        print(f"{f.name}: {n_in:,} rows read as {sensor} ({'NRT' if is_nrt else 'standard'}){extra}")

    out = pd.concat(parts).groupby(level=[0, 1, 2, 3]).sum().reset_index()
    C.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = C.PROCESSED_DIR / "cell_month.parquet"
    out.to_parquet(path, index=False)

    summary = out.groupby("sensor").agg(first=("ym", "min"), last=("ym", "max"),
                                        hotspots=("count", "sum"), cells=("cell_lat", "size"))
    print("\nCoverage by sensor:\n", summary.to_string())
    print(f"\nSaved {len(out):,} cell-month rows to {path}")


if __name__ == "__main__":
    main()
