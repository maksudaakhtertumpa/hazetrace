"""Bundled satellite background for the map (so it is never blank offline).

Usage: python scripts/fetch_satellite.py [fallback_world_image.jpg]
1) Tries NASA GIBS WMS (Blue Marble) for C.MAP_EXTENT.
2) If GIBS is not reachable, crops a public-domain NASA Blue Marble equirectangular world image
   (downloaded, or the local file given as argument) to the same extent.
"""
import io
import sys
import urllib.request
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hazetrace import config as C  # noqa: E402

WMS = ("https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi?SERVICE=WMS&REQUEST=GetMap&VERSION=1.3.0"
       "&LAYERS={layer}&STYLES=&CRS=EPSG:4326&BBOX={S},{W},{N},{E}&WIDTH={w}&HEIGHT={h}&FORMAT=image/jpeg")
WORLD = "https://raw.githubusercontent.com/vasturiano/three-globe/master/example/img/earth-blue-marble.jpg"


def get(url, timeout=60):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()


def from_gibs():
    W, E, S, N = C.MAP_EXTENT
    w = 3600
    h = round(w * (N - S) / (E - W))
    for layer in C.GIBS_SNAPSHOT_LAYERS:
        try:
            data = get(WMS.format(layer=layer, S=S, W=W, N=N, E=E, w=w, h=h))
            if data[:2] == b"\xff\xd8":
                C.SATELLITE_IMAGE.write_bytes(data)
                return f"GIBS {layer}"
        except Exception as exc:  # noqa: BLE001
            print(f"GIBS {layer}: {exc}")
    return None


def from_world(local=None):
    raw = Path(local).read_bytes() if local else get(WORLD, 120)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    W, E, S, N = C.MAP_EXTENT
    px = lambda lon: round((lon + 180) / 360 * img.width)   # noqa: E731
    py = lambda lat: round((90 - lat) / 180 * img.height)   # noqa: E731
    crop = img.crop((px(W), py(N), px(E), py(S)))
    crop = crop.resize((2400, round(2400 * crop.height / crop.width)), Image.LANCZOS)
    crop.save(C.SATELLITE_IMAGE, quality=85)
    return "Blue Marble world image (cropped)"


if __name__ == "__main__":
    C.SATELLITE_IMAGE.parent.mkdir(parents=True, exist_ok=True)
    src = from_gibs() or from_world(sys.argv[1] if len(sys.argv) > 1 else None)
    print(f"{src} -> {C.SATELLITE_IMAGE} ({C.SATELLITE_IMAGE.stat().st_size // 1024} KB)")
