"""Assign 0.25-degree grid cells to study regions using real province boundaries.

Uses assets/provinces.geojson (Natural Earth admin-1, built by scripts/make_provinces.py).
Falls back to the approximate boxes in config.REGIONS if the file is missing.
"""
import json
from functools import lru_cache

import numpy as np
import shapely
from shapely.geometry import shape

from hazetrace import config as C


@lru_cache(maxsize=1)
def _polygons():
    if not C.PROVINCES_FILE.exists():
        return None
    feats = json.loads(C.PROVINCES_FILE.read_text(encoding="utf-8"))["features"]
    polys = {f["properties"]["region"]: shape(f["geometry"]) for f in feats}
    for g in polys.values():
        shapely.prepare(g)
    return polys


def _boxes(lat, lon):
    out = np.full(len(lat), "Other", dtype=object)
    for name in reversed(C.REGION_ORDER):  # reversed so the first listed wins
        w, e, s, n = C.REGIONS[name]
        out[(lon >= w) & (lon < e) & (lat >= s) & (lat < n)] = name
    return out


def assign_region(lat, lon):
    """Region name for each cell centre; 'Other' outside the six provinces."""
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    polys = _polygons()
    if polys is None:
        return _boxes(lat, lon)
    pts = shapely.points(lon, lat)
    out = np.full(len(lat), "Other", dtype=object)
    for name in reversed(C.REGION_ORDER):
        if name in polys:
            out[shapely.contains(polys[name], pts)] = name
    # Coastal cells whose centre falls just offshore: give them to a province within half a cell.
    left = out == "Other"
    for name in reversed(C.REGION_ORDER):
        if name in polys and left.any():
            near = left & shapely.dwithin(polys[name], pts, C.GRID_DEG / 2)
            out[near] = name
            left &= ~near
    return out
