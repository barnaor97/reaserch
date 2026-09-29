"""Query the exported monthly composite assets for 2016.

Reports, per 2016 asset: n_images_in_month, unmasked fraction of the NBR band,
and mean pct_valid over the ROI. Also lists which 2016 assets exist at all.

Run:  python check_2016_assets.py
(first time:  earthengine authenticate  -- or  ee.Authenticate()  )
"""
import ee

PROJECT = os.environ.get("GEE_PROJECT", "YOUR-GEE-PROJECT")
ROOT    = f"projects/{PROJECT}/assets"
COLL    = f"{ROOT}/s2_monthly_image_collection_v3_10m"
ROI_ID  = f"{ROOT}/gaza_envelope_for_decell_Dissolve"
SCALE   = 100          # coarse: fractions are scale-invariant, keeps it fast
MAXPIX  = 1e13

ee.Initialize(project=PROJECT)
roi = ee.FeatureCollection(ROI_ID).geometry()
ic  = ee.ImageCollection(COLL)

print(f"collection size (all years): {ic.size().getInfo()}")
ids = ic.aggregate_array("system:index").getInfo()
present_2016 = sorted(i for i in ids if "2016" in i)
print(f"2016 assets present: {len(present_2016)} -> {present_2016}\n")

# total ROI pixels at SCALE, for the unmasked-fraction denominator
denom = (ee.Image.constant(1).rename("ones")
         .reduceRegion(ee.Reducer.count(), roi, SCALE, maxPixels=MAXPIX)
         .get("ones")).getInfo()
print(f"ROI pixel count at {SCALE} m: {denom:,}\n")

print(f"{'year-mo':>8} {'n_images_in_month':>18} {'NBR unmasked %':>15} "
      f"{'B8 unmasked %':>14} {'mean pct_valid':>15}")

for year in (2016, 2017):                       # 2017 included as a control
    for month in range(1, 13):
        idx = f"s2_monthly_mean_{year}_{month:02d}"
        sub = ic.filter(ee.Filter.eq("system:index", idx))
        if sub.size().getInfo() == 0:
            print(f"{year}-{month:02d} {'ASSET MISSING':>18}")
            continue
        img = ee.Image(sub.first())
        n_img = img.get("n_images_in_month").getInfo()
        nbr = img.normalizedDifference(["B8", "B12"]).rename("NBR")
        stats = (ee.Image.cat([nbr, img.select("B8"), img.select("pct_valid")])
                 .reduceRegion(
                     reducer=ee.Reducer.count().combine(
                         ee.Reducer.mean(), sharedInputs=True),
                     geometry=roi, scale=SCALE, maxPixels=MAXPIX)).getInfo()
        f_nbr = 100 * stats.get("NBR_count", 0) / denom
        f_b8  = 100 * stats.get("B8_count", 0) / denom
        pv    = stats.get("pct_valid_mean")
        pv_s  = f"{pv:.4f}" if pv is not None else "n/a"
        print(f"{year}-{month:02d} {str(n_img):>18} {f_nbr:>14.2f}% "
              f"{f_b8:>13.2f}% {pv_s:>15}")

# Does the underlying L2A collection itself hold 2016 scenes over this ROI?
print("\nRaw COPERNICUS/S2_SR_HARMONIZED scene counts over the ROI:")
for year in (2015, 2016, 2017):
    n = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
         .filterBounds(roi)
         .filterDate(f"{year}-01-01", f"{year+1}-01-01").size().getInfo())
    print(f"  {year}: {n} scenes")
print("\nEarliest S2_SR_HARMONIZED scene over the ROI:")
first = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(roi)
         .sort("system:time_start").first())
print(" ", ee.Date(first.get("system:time_start")).format("YYYY-MM-dd").getInfo())

# ---------------------------------------------------------------- sharpest test
# The GEE catalog lists COPERNICUS/S2_SR_HARMONIZED as starting 2017-03-28, yet the
# local arrays hold real per-pixel data for Jan and Feb 2017. Count raw scenes per
# month directly: if these are 0 while the asset holds data, the assets were NOT
# built from this collection.
print("\nRaw S2_SR_HARMONIZED scene counts over the ROI, by month (2016-01 .. 2017-06):")
src = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(roi)
for year in (2016, 2017):
    for month in range(1, 13):
        if year == 2017 and month > 6:
            break
        a = ee.Date.fromYMD(year, month, 1)
        b = a.advance(1, "month")
        print(f"  {year}-{month:02d}: {src.filterDate(a, b).size().getInfo():>4} scenes")

print("\nAsset collection composition:")
print(f"  total assets: {ic.size().getInfo()}   (120 = 2016-2025 complete, 108 = 2017-2025 only)")
yrs = ic.aggregate_array("year").getInfo()
from collections import Counter
for y, n in sorted(Counter(yrs).items()):
    print(f"    year {y}: {n} assets")
