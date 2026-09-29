"""
export_nbr_2017_2025_30m.py — the four OBSERVATIONAL layers, exported programmatically.

Exports, on the exact validated 30 m inference grid:
    NBR_2017_mean        mean NBR over the VALID monthly observations of 2017
    n_valid_months_2017  how many monthly observations that mean rests on
    NBR_2025_mean        mean NBR over the VALID monthly observations of 2025
    n_valid_months_2025  how many monthly observations that mean rests on

NO imputation, NO normalisation, NO clipping, NO post-export resampling.
This is an OBSERVATIONAL ANALOGUE of the Recovery Gap numerator computed from valid observations
only. It is deliberately NOT the numerator itself: compute_targets() averaged an imputed series
(04_data_prep.ipynb cell 9 fills gaps with a global per-calendar-month mean), which Earth Engine
cannot and should not reproduce.

PROVENANCE — every choice traced to the original pipeline:
  collection    COPERNICUS/S2_SR_HARMONIZED, bands B4,B8,B8A,B11,B12   01_data_to_gee cell 15
  cloud mask    CLOUD_SCORE_PLUS/V1/S2_HARMONIZED, cs_cdf >= 0.60      01_data_to_gee cells 2, 15
  spatial mask  human infrastructure rasterised, inverted               01_data_to_gee cell 9
  region        gaza_envelope_for_decell_Dissolve                       01_data_to_gee cell 9
  reflectance   multiply(0.0001), toFloat                               01_data_to_gee cell 17
  compositing   MONTHLY MEAN of masked, rescaled bands                  01_data_to_gee cell 17
  NBR           normalizedDifference(B8,B12) ON the monthly composite   01_data_to_gee cell 19
  annual value  mean over the year's monthly NBR composites             matches 04_data_prep cell 12,
                                                                        which averages MONTHS
  grid          EPSG:32636, [30,0,620610,0,-30,3514560], 1625x2598      read from the inference rasters

Only 2017 and 2025 scenes are loaded. That is a filtering optimisation, not a methodological
change: the monthly composites for those years are identical either way.

AUTH. Uses gcloud Application Default Credentials explicitly, because the stale
~/.config/earthengine/credentials would otherwise take precedence and fail.

Run: python notebooks/export_nbr_2017_2025_30m.py
Writes data/raw/nbr_2017_2025_30m.tif (4 bands) + data/raw/nbr_export_provenance.json
"""
import io
import json
import time
import os
import sys
import zipfile
from datetime import datetime, timezone

import numpy as np
import rasterio
import requests
from rasterio.transform import Affine

import ee
import google.auth

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
OUT_TIF = os.path.join(RAW, "nbr_2017_2025_30m.tif")
PROJECT = os.environ.get("GEE_PROJECT", "YOUR-GEE-PROJECT")
ASSET_ROOT = "projects/%s/assets" % PROJECT
CRS = "EPSG:32636"
CRS_TRANSFORM = [30, 0, 620610, 0, -30, 3514560]
WIDTH, HEIGHT = 1625, 2598
CLEAR_THRESH = 0.60
BANDS = ["B4", "B8", "B8A", "B11", "B12"]
NODATA = -9999.0

# ---- the grid is READ from the inference rasters, never assumed -----------------------------
with rasterio.open(os.path.join(ROOT, "outputs", "inference_full_roi", "severity_30m.tif")) as s:
    ref_tr, ref_crs, ref_shape = s.transform, str(s.crs), (s.height, s.width)
assert ref_crs == CRS and ref_shape == (HEIGHT, WIDTH), (ref_crs, ref_shape)
assert ref_tr.almost_equals(Affine(*[CRS_TRANSFORM[0], CRS_TRANSFORM[1], CRS_TRANSFORM[2],
                                     CRS_TRANSFORM[3], CRS_TRANSFORM[4], CRS_TRANSFORM[5]])), ref_tr
print("grid read from severity_30m.tif:", ref_crs, ref_shape, ref_tr)

creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/earthengine",
                                       "https://www.googleapis.com/auth/cloud-platform"])
ee.Initialize(credentials=creds, project=PROJECT)
print("Earth Engine initialised via Application Default Credentials")

# ---- masked Sentinel-2, exactly as the original pipeline built it ----------------------------
region = ee.FeatureCollection(ASSET_ROOT + "/gaza_envelope_for_decell_Dissolve").geometry()
infra = ee.FeatureCollection(ASSET_ROOT + "/human_infrastructure_mask_utm36n")
final_mask = infra.reduceToImage(["Shape_Area"], ee.Reducer.count()).gt(0).Not()

YEARS = [2017, 2025]
s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
      .filterBounds(region)
      .filter(ee.Filter.Or(*[ee.Filter.calendarRange(y, y, "year") for y in YEARS]))
      .select(BANDS))
cs = (ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
      .filterBounds(region)
      .filter(ee.Filter.Or(*[ee.Filter.calendarRange(y, y, "year") for y in YEARS])))
s2_clean = ee.ImageCollection(s2.linkCollection(cs, ["cs_cdf"]).map(
    lambda img: img.updateMask(img.select("cs_cdf").gte(CLEAR_THRESH))
                   .select(BANDS).updateMask(final_mask).toFloat()))


def monthly_nbr(year, month):
    start = ee.Date.fromYMD(year, month, 1)
    ic = (s2_clean.filterDate(start, start.advance(1, "month"))
          .map(lambda img: img.select(BANDS).multiply(0.0001).toFloat()))
    comp = ic.select(BANDS).mean().toFloat()                    # monthly MEAN composite
    return comp.normalizedDifference(["B8", "B12"]).rename("NBR")   # NBR AFTER compositing


def annual(year):
    months = ee.ImageCollection([monthly_nbr(year, m) for m in range(1, 13)])
    return months.mean().toFloat(), months.count().toFloat()    # valid-only mean; valid-month count


# A composite has no fixed projection, so reduceResolution must be told what the input grid is.
# The original monthly composites were exported at scale=10 in EPSG:32636 (01_data_to_gee cell 17),
# so the input grid is 10 m in the same CRS. The origin is taken from the 30 m grid, which makes
# exactly 3x3 = 9 ten-metre cells nest inside every 30 m cell with no straddling.
S2_10M = dict(crs=CRS, crsTransform=[10, 0, 620610, 0, -10, 3514560])


def to_grid(img):
    return (img.setDefaultProjection(**S2_10M)
               .reduceResolution(reducer=ee.Reducer.mean(), maxPixels=16)
               .reproject(crs=CRS, crsTransform=CRS_TRANSFORM))


LAYERS = {}
for y in YEARS:
    m, c = annual(y)
    LAYERS["NBR_%d_mean" % y] = to_grid(m).unmask(NODATA)
    LAYERS["n_valid_months_%d" % y] = to_grid(c).unmask(0)
ORDER = ["NBR_2017_mean", "NBR_2025_mean", "n_valid_months_2017", "n_valid_months_2025"]


TILE = 512          # 512x512 computes in ~30 s and stays inside the Earth Engine memory limit;
                    # 1024x1024 returns "User memory limit exceeded". Tiles carry their own
                    # crs_transform origin, so the mosaic lands on the validated grid by
                    # construction and nothing is resampled afterwards.


def fetch_tile(img, col0, row0, w, h, nbands):
    tr = [30, 0, CRS_TRANSFORM[2] + col0 * 30, 0, -30, CRS_TRANSFORM[5] - row0 * 30]
    last = None
    for attempt in range(4):
        try:
            url = img.getDownloadURL({"name": "t", "crs": CRS, "crs_transform": tr,
                                      "dimensions": "%dx%d" % (w, h), "format": "GEO_TIFF"})
            r = requests.get(url, timeout=1800)
            if r.status_code != 200:
                last = "HTTP %s %s" % (r.status_code, r.text[:120].replace("\n", " "))
                time.sleep(5 * (attempt + 1))
                continue
            buf = r.content
            if buf[:2] == b"PK":
                z = zipfile.ZipFile(io.BytesIO(buf))
                buf = z.read([n for n in z.namelist() if n.endswith(".tif")][0])
            with rasterio.open(io.BytesIO(buf)) as src:
                assert (src.height, src.width) == (h, w), (src.height, src.width)
                assert abs(src.transform.c - tr[2]) < 1e-6 and abs(src.transform.f - tr[5]) < 1e-6, src.transform
                return np.stack([src.read(i + 1) for i in range(nbands)]).astype(np.float32)
        except Exception as e:
            last = "%s: %s" % (type(e).__name__, str(e)[:120])
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("tile (%d,%d) failed after retries: %s" % (row0, col0, last))


stack = ee.Image.cat([LAYERS[n].rename(n) for n in ORDER])
full = np.full((len(ORDER), HEIGHT, WIDTH), np.nan, dtype=np.float32)
tiles = [(c, r) for r in range(0, HEIGHT, TILE) for c in range(0, WIDTH, TILE)]
print("\ndownloading %d tiles of up to %dx%d, %d bands each" % (len(tiles), TILE, TILE, len(ORDER)))
t_start = time.time()
for k, (c0, r0) in enumerate(tiles, 1):
    w, h = min(TILE, WIDTH - c0), min(TILE, HEIGHT - r0)
    full[:, r0:r0 + h, c0:c0 + w] = fetch_tile(stack, c0, r0, w, h, len(ORDER))
    print("   tile %2d/%d  col %4d row %4d  %dx%d   %.0fs elapsed" % (k, len(tiles), c0, r0, w, h, time.time() - t_start), flush=True)
assert not np.isnan(full).any(), "mosaic has holes"
arrs = {n: full[i] for i, n in enumerate(ORDER)}
for n in ORDER:
    a = arrs[n]
    valid = a[a != NODATA] if n.startswith("NBR") else a
    print("   %-22s valid=%d  mean=%.4f" % (n, int((a != NODATA).sum()), float(valid.mean())))

prof = dict(driver="GTiff", height=HEIGHT, width=WIDTH, count=4, dtype="float32",
            crs=CRS, transform=ref_tr, nodata=NODATA, compress="deflate")
with rasterio.open(OUT_TIF, "w", **prof) as dst:
    for i, n in enumerate(ORDER, start=1):
        dst.write(arrs[n], i)
        dst.set_band_description(i, n)
print("\nwrote", OUT_TIF)

json.dump(dict(written=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
               project=PROJECT, bands=ORDER, crs=CRS, crs_transform=CRS_TRANSFORM,
               dimensions=[WIDTH, HEIGHT], nodata=NODATA, clear_thresh=CLEAR_THRESH,
               s2_collection="COPERNICUS/S2_SR_HARMONIZED",
               cloud_collection="GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED",
               roi_asset=ASSET_ROOT + "/gaza_envelope_for_decell_Dissolve",
               infra_asset=ASSET_ROOT + "/human_infrastructure_mask_utm36n",
               nbr="normalizedDifference(B8,B12) on the monthly mean composite",
               imputation="NONE — valid observations only, unlike the training series",
               native_grid_for_aggregation=S2_10M["crsTransform"],
               aggregation="reduceResolution(mean) from 10 m to 30 m, 9 nested cells",
               ee_version=ee.__version__),
          open(os.path.join(RAW, "nbr_export_provenance.json"), "w"), indent=1)
print("provenance written to data/raw/nbr_export_provenance.json")
