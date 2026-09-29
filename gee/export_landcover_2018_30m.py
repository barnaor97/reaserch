"""
export_landcover_2018_30m.py — the one Earth Engine export Notebook 14 needs.

Exports the 2018 image of landcover_smoothed_yearly, aggregated to 30 m by
(area-weighted) mode, on exactly the grid of outputs/inference_full_roi/*_30m.tif:
EPSG:32636, 30 m pixels, upper-left corner (620610, 3514560), 1625 columns x 2598 rows.
Nothing else is built or exported.

Run:   python notebooks/export_landcover_2018_30m.py
Then:  download landcover_2018_30m.tif from Drive folder "thesisv4_exports" to
       data/raw/landcover_2018_30m.tif and re-run notebooks/14_landuse_presentation.ipynb
       (its §2 verifies the grid alignment before anything uses the layer).
"""
import math

import ee

PROJECT = os.environ.get("GEE_PROJECT", "YOUR-GEE-PROJECT")
ASSET = f"projects/{PROJECT}/assets/landcover_smoothed_yearly"

# The inference grid, as stored in outputs/inference_full_roi/severity_30m.tif
CRS = "EPSG:32636"
CRS_TRANSFORM = [30, 0, 620610, 0, -30, 3514560]
DIMENSIONS = "1625x2598"                     # width x height

ee.Initialize(project=PROJECT)

# Same image and band that Notebooks 02 and 07a used for the land-cover features
lc = ee.Image(ee.ImageCollection(ASSET)
              .filter(ee.Filter.eq("system:index", "2018")).first()).select("b1")

native_m = lc.projection().nominalScale().getInfo()
max_px = min(65536, int(math.ceil(30.0 / native_m + 2)) ** 2)
print(f"native resolution {native_m:.2f} m -> reduceResolution maxPixels = {max_px}")

lc_30m = (lc.reduceResolution(reducer=ee.Reducer.mode(), maxPixels=max_px)
            .reproject(crs=CRS, crsTransform=CRS_TRANSFORM)
            .unmask(0)                        # 0 = no land-cover data
            .toInt16())

task = ee.batch.Export.image.toDrive(
    image=lc_30m,
    description="landcover_2018_30m",
    folder="thesisv4_exports",
    fileNamePrefix="landcover_2018_30m",
    crs=CRS,
    crsTransform=CRS_TRANSFORM,
    dimensions=DIMENSIONS,
    maxPixels=1e9,
    fileFormat="GeoTIFF",
)
task.start()
print("export task started:", task.id, "- monitor at https://code.earthengine.google.com/tasks")
