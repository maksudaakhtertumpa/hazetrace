"""Country borders for the map, clipped to Southeast Asia.

Usage: python scripts/make_basemap.py <ne_10m_admin_0_countries.geojson> <assets/basemap_sea.geojson>
Source: Natural Earth 1:10m admin-0 countries (public domain).
"""
import json
import sys

from shapely.geometry import box, mapping, shape

CLIP = box(86, -14, 130, 16)


def rnd(o):
    if isinstance(o, float):
        return round(o, 4)
    if isinstance(o, (list, tuple)):
        return [rnd(x) for x in o]
    return o


def main(src, out):
    feats = []
    for f in json.load(open(src, encoding="utf-8"))["features"]:
        g = shape(f["geometry"])
        if not g.intersects(CLIP):
            continue
        g = g.intersection(CLIP).simplify(0.01, preserve_topology=True)
        if g.is_empty:
            continue
        p = f["properties"]
        m = mapping(g)
        feats.append({"type": "Feature", "properties": {"name": p.get("NAME_EN") or p.get("NAME")},
                      "geometry": {"type": m["type"], "coordinates": rnd(m["coordinates"])}})
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"type": "FeatureCollection", "features": feats}, fh, separators=(",", ":"))
    print(f"{len(feats)} countries -> {out}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
