"""Shared settings for HazeTrace.

Region boxes are APPROXIMATE bounding boxes of each province/state, good enough
for a hackathon demo. Replace them with real boundaries (e.g. GADM) if time allows,
and say in the submission that they are approximate.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR = ROOT / "models"

# Study domain (matches the FIRMS download box): W, E, S, N
DOMAIN = (95.0, 119.0, -6.0, 7.0)

# Grid cell size in degrees (0.25 deg ~ 28 km at the equator)
GRID_DEG = 0.25

# Preset regions: name -> (lon_min, lon_max, lat_min, lat_max), approximate
REGIONS = {
    "Riau": (100.0, 104.0, -1.2, 2.5),
    "Jambi": (101.1, 104.5, -2.8, -0.7),
    "South Sumatra": (102.0, 106.2, -4.9, -1.6),
    "West Kalimantan": (108.8, 114.2, -3.1, 2.1),
    "Central Kalimantan": (110.7, 116.0, -3.6, 0.8),
    "Sarawak (Malaysia)": (109.6, 115.6, 0.8, 5.0),
}
# Order matters where boxes overlap: the first match wins.
REGION_ORDER = list(REGIONS)

# Detection filters
MODIS_MIN_CONFIDENCE = 30          # MODIS confidence is 0-100; >=30 = nominal + high
VIIRS_KEEP_CONFIDENCE = {"n", "h", "nominal", "high"}
KEEP_TYPE = 0                      # FIRMS 'type' 0 = presumed vegetation fire

# Sensor periods (from FIRMS archive page)
SNPP_START = "2012-02"            # first FULL S-NPP month (data begin 2012-01-20)
SNPP_END = "2026-11"               # S-NPP stopped 2026-11-02
NOAA20_START = "2018-04"

# The record starts when BOTH MODIS satellites fly (Aqua joined mid-2002). Before that,
# Terra alone sees fewer fires, which would look like a false upward trend.
RECORD_START = "2003-01"

# Known bad MODIS data (NASA flagged Terra standby 26-27 Mar 2026 as questionable)
MODIS_BAD_DATES = ["2026-03-26", "2026-03-27"]

# Years held out one at a time in validation are all full overlap years
FIRST_FULL_OVERLAP_YEAR = 2013

# Months with no data for a region (files not downloaded). Shown as blank "no data",
# never as zero fires. Here: 2026 FIRMS files were downloaded for Indonesia only.
COVERAGE_GAPS = {}

# MODIS and VIIRS are paired (training + validation) only where MODIS covers the whole
# domain. 2026 MODIS files exist for Malaysia only, so pairing stops at Dec 2025 and the
# Malaysian 2026 MODIS data serve as an independent out-of-time check instead.
PAIR_END = "2025-12"
MODIS_PARTIAL_FROM = "2026-01"   # MODIS observed values after this cover Malaysia only
# Files that define the Malaysia-only 2026 check: cells found in the Malaysia request files
# and NOT in the Indonesia request files (border cells mix both countries' data).
CHECK_INCLUDE_GLOB = "firms_2026/*/*84708[234].csv"   # Malaysia 2026 request (MODIS, NOAA-20, S-NPP)
CHECK_EXCLUDE_GLOB = "firms_2026/*/*84694[78].csv"    # Indonesia 2026 request (NOAA-20, S-NPP)

# The newest month in the data is partial (files end ~13 Nov 2026). KPIs and the data note
# use the latest COMPLETE month; the map can still show the partial month, labelled.
PARTIAL_MONTH = "2026-11"
DATA_END_DATE = "13 Nov 2026"
KL = (3.139, 101.687)   # Kuala Lumpur lat, lon

# ---- map assets (app) ----------------------------------------------------------------------------
MAP_EXTENT = (86.0, 130.0, -14.0, 16.0)          # W, E, S, N of the bundled satellite image
SATELLITE_IMAGE = ROOT / "assets" / "satellite_sea.jpg"
GIBS_SNAPSHOT_LAYERS = ["BlueMarble_ShadedRelief_Bathymetry", "BlueMarble_NextGeneration"]
GIBS_TILES = ("https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/BlueMarble_ShadedRelief_Bathymetry/default/"
              "GoogleMapsCompatible_Level8/{z}/{y}/{x}.jpeg")
BORDERS_FILE = ROOT / "assets" / "basemap_sea.geojson"
PROVINCES_FILE = ROOT / "assets" / "provinces.geojson"
CITIES = {"Kuala Lumpur": (3.139, 101.687), "Singapore": (1.290, 103.852),
          "Pekanbaru": (0.507, 101.448), "Jambi": (-1.610, 103.613), "Palembang": (-2.976, 104.775),
          "Pontianak": (-0.026, 109.343), "Palangka Raya": (-2.210, 113.920), "Kuching": (1.553, 110.359)}
COUNTRY_LABELS = {"MALAYSIA": (4.4, 102.0), "INDONESIA": (-4.6, 110.0),
                  "SINGAPORE": (1.15, 104.35), "BRUNEI": (4.7, 114.6), "THAILAND": (6.6, 100.8)}
