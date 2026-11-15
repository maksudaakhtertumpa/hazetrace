"""Make FAKE FIRMS-shaped CSVs so the pipeline and app can be tested before the
real download arrives. Never use these numbers in the submission.

    python scripts/00_make_fake_data.py      # writes data/raw/fake/*.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hazetrace import config as C  # noqa: E402

rng = np.random.default_rng(42)
BIG_YEARS = {2006: 3.0, 2009: 1.8, 2015: 5.0, 2019: 3.5, 2023: 2.5, 2026: 2.8}
SEASON = {1: .15, 2: .25, 3: .35, 4: .2, 5: .25, 6: .5, 7: 1.0, 8: 2.2, 9: 2.8, 10: 1.6, 11: .4, 12: .15}

# fire "centres" with a hidden peat flag: VIIRS sees relatively more on peat
centres = []
for name, (w, e, s, n) in C.REGIONS.items():
    for _ in range(12):
        centres.append((rng.uniform(w, e), rng.uniform(s, n), rng.random() < 0.45))


FIRE_FILE = {}


def true_fires(year, month):
    """Same underlying fires for every sensor (the realistic part)."""
    key = (year, month)
    if key not in FIRE_FILE:
        rows = []
        for lon, lat, peat in centres:
            lam = 14 * SEASON[month] * BIG_YEARS.get(year, 1.0) * (1.5 if peat else 1.0)
            k = rng.poisson(lam)
            if k:
                size = rng.gamma(1.2, 1.0, k) * (0.5 if peat else 1.0)  # peat fires smaller
                rows.append(pd.DataFrame({"lon": lon + rng.normal(0, .3, k), "lat": lat + rng.normal(0, .3, k),
                                          "size": size, "peat": peat}))
        FIRE_FILE[key] = pd.concat(rows) if rows else pd.DataFrame(columns=["lon", "lat", "size", "peat"])
    return FIRE_FILE[key]


def write(sensor, start, end, path):
    recs = []
    for ym in pd.period_range(start, end, freq="M"):
        f = true_fires(ym.year, ym.month)
        if f.empty:
            continue
        if sensor == "modis":   # 1 km pixels: misses many small fires
            p_det = 1 - np.exp(-f["size"].to_numpy() / 1.2)
            npix = np.ones(len(f), int)
        else:                   # 375 m pixels: sees most fires, often as several pixels
            p_det = 1 - np.exp(-f["size"].to_numpy() / 0.25)
            npix = 1 + rng.poisson(1.5 * f["size"].to_numpy())
        det = rng.random(len(f)) < p_det
        f, npix = f[det], npix[det]
        if f.empty:
            continue
        idx = np.repeat(np.arange(len(f)), npix)
        g = f.iloc[idx]
        k = len(g)
        days = rng.integers(1, ym.days_in_month + 1, k)
        frp = (g["size"].to_numpy() * rng.gamma(2, 15 if sensor == "modis" else 6, k)).round(1)
        d = {"latitude": (g.lat.to_numpy() + rng.normal(0, .005, k)).round(4),
             "longitude": (g.lon.to_numpy() + rng.normal(0, .005, k)).round(4),
             "acq_date": [f"{ym}-{x:02d}" for x in days],
             "acq_time": rng.integers(0, 2400, k), "frp": frp,
             "daynight": np.where(g.peat.to_numpy() & (rng.random(k) < .4), "N", rng.choice(["D", "N"], k, p=[.8, .2])),
             "type": 0}
        if sensor == "modis":
            d.update(brightness=rng.normal(320, 10, k).round(1), satellite=rng.choice(["Terra", "Aqua"], k),
                     instrument="MODIS", confidence=np.clip(rng.normal(65, 20, k), 0, 100).astype(int),
                     version="6.1")
        else:
            d.update(bright_ti4=rng.normal(335, 10, k).round(1), satellite={"snpp": "N", "noaa20": "N20"}[sensor],
                     instrument="VIIRS", confidence=rng.choice(["l", "n", "h"], k, p=[.1, .75, .15]), version="2.0")
        recs.append(pd.DataFrame(d))
    df = pd.concat(recs)
    df.to_csv(path, index=False)
    print(f"{path.name}: {len(df):,} fake hotspots")


if __name__ == "__main__":
    out = C.RAW_DIR / "fake"
    out.mkdir(parents=True, exist_ok=True)
    write("modis", "2003-01", "2026-10", out / "fire_archive_M-C61_FAKE.csv")
    write("snpp", "2012-02", "2026-10", out / "fire_archive_SV-C2_FAKE.csv")
    write("noaa20", "2018-04", "2026-10", out / "fire_archive_J1V-C2_FAKE.csv")
