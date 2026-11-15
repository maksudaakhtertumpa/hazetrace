"""Real province boundaries for the six study regions.

Usage: python scripts/make_provinces.py <ne_10m_admin_1_states_provinces.geojson> <assets/provinces.geojson>
Source: Natural Earth 1:10m admin-1 states and provinces (public domain).
"""
import json
import sys

from shapely.geometry import mapping, shape

KEEP = {("Indonesia", "Riau"): "Riau", ("Indonesia", "Jambi"): "Jambi",
        ("Indonesia", "Sumatera Selatan"): "South Sumatra",
        ("Indonesia", "Kalimantan Barat"): "West Kalimantan",
        ("Indonesia", "Kalimantan Tengah"): "Central Kalimantan",
        ("Malaysia", "Sarawak"): "Sarawak (Malaysia)"}


def rnd(o):
    if isinstance(o, float):
        return round(o, 4)
    if isinstance(o, (list, tuple)):
        return [rnd(x) for x in o]
    return o


def main(src, out):
    feats = {}
    for f in json.load(open(src, encoding="utf-8"))["features"]:
        p = f["properties"]
        key = (p.get("admin"), p.get("name"))
        if key in KEEP:
            g = shape(f["geometry"]).simplify(0.005, preserve_topology=True)
            m = mapping(g)
            feats[KEEP[key]] = {"type": "Feature", "properties": {"region": KEEP[key]},
                                "geometry": {"type": m["type"], "coordinates": rnd(m["coordinates"])}}
    missing = set(KEEP.values()) - set(feats)
    assert not missing, f"not found: {missing}"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"type": "FeatureCollection", "features": list(feats.values())}, fh, separators=(",", ":"))
    print(f"{len(feats)} provinces -> {out}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
