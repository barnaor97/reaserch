# Data manifest

Three categories: **public** (retrievable by anyone, fetched by code), **restricted** (third-party research data, not redistributed here), and **generated** (produced by running the pipeline).

Expected local layout:

```
data/
  roi/                 fig_roi_polygon.*          included in this repository
  restricted/          <- you supply these; git-ignored
    fire_reference_inventory.shp (+ .shx .dbf .prj .cpg)
    landcover_2018.tif
  external/            <- auto-downloaded public rasters (git-ignored)
  raw/  processed/     <- generated intermediates (git-ignored)
```

---

## 1. Public datasets

### Sentinel-2 Level-2A surface reflectance
| | |
|---|---|
| Collection | `COPERNICUS/S2_SR_HARMONIZED` (Google Earth Engine) |
| Temporal range | 2016-01-01 – 2025-12-31 (2016 used **only** to build the seasonal baseline; the model input sequence is 2017-01 – 2025-12, 108 months) |
| Bands retained | B4 (red, 10 m), B8 (NIR, 10 m), B8A (red-edge, 20 m), B11 (SWIR-1, 20 m), B12 (SWIR-2, 20 m) |
| Grid | EPSG:32636, origin 620610 / 3514560, 10 m; 4875 × 7794 cells |
| Resampling | the three 20 m bands are **resampled** (not aggregated) to the 10 m grid using Earth Engine's default nearest-neighbour at a 10 m request scale; they carry 20 m native information |
| Compositing | monthly mean |
| Retrieved by | `gee/01_data_to_gee.ipynb`, `gee/02_sampling_computepixels.ipynb`, `gee/export_nbr_2017_2025_30m.py` |
| Redistribution | not redistributed; retrieved via GEE |

### Cloud Score+
| | |
|---|---|
| Collection | `GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED` |
| Use | cloud/shadow masking before compositing; pixels with `cs_cdf` < **0.60** are masked |
| Citation | Google Earth Engine (2023), Cloud Score+ S2_HARMONIZED |
| Redistribution | not redistributed; retrieved via GEE |

### WorldClim 2.1 — annual precipitation (BIO12)
| | |
|---|---|
| File | `data/external/ISR_wc2.1_30s_bio.tif`, band **12** (`wc2.1_30s_bio_12`) |
| URL | `https://geodata.ucdavis.edu/climate/worldclim/2_1/tiles/iso/ISR_wc2.1_30s_bio.tif` |
| SHA-256 | `7b755de4828ca25f5850474833171e197c131c6787edb17655d9c21a45a5570f` |
| CRS / resolution | EPSG:4326, 30 arc-seconds (0.00833333°) |
| Citation | Fick, S.E. & Hijmans, R.J. (2017). *Int. J. Climatology* 37(12), 4302–4315 (the citation specified by WorldClim for version 2.1) |
| Downloaded by | `figures/22_study_area_environment.py` (automatic) |
| Redistribution | not redistributed; download from source |

### SRTM GL1 digital elevation model
| | |
|---|---|
| File | `data/external/N31E034.hgt` (3601 × 3601, big-endian int16, 1 arc-second) |
| URL | `https://s3.amazonaws.com/elevation-tiles-prod/skadi/N31/N31E034.hgt.gz` |
| SHA-256 | `.hgt` `2875fdf4203100c2e0df5f168fd391d0c0a29704b8f7f8b90dea929298eddf68` · `.hgt.gz` `e96bb5e2f5ef6835f63ecddac119ca2ea92db4dea16b897b15cd396ee7ef90c2` |
| CRS / resolution | EPSG:4326, 1 arc-second (~30 m); tile SW corner N31 E034 |
| Citation | Farr, T.G. et al. (2007). *Reviews of Geophysics* 45, RG2004 |
| Downloaded by | `figures/22_study_area_environment.py` (automatic) |
| Redistribution | not redistributed; download from source |

### Region of interest
| | |
|---|---|
| File | `data/roi/fig_roi_polygon.shp` (+ `.shx .dbf .prj .cpg`) — **included** |
| CRS | EPSG:4326; 1 polygon, 2016 vertices; 1,936.2 km² (EPSG:32636) |
| SHA-256 (`.shp`) | `3aecc6da1a408c88f67d2a595d17365541d8cff1f2c9c9bdf5943476162cb738` |
| Consumed by | `figures/22_study_area_environment.py`, `pipeline_10m/*` |

---

## 2. Restricted datasets — not redistributed

### 2.1 Fire-reference inventory

Merged KKL/JNF fire records and Planet/Sentinel-2-derived burn scars (3,596 polygons in the thesis). **Third-party research data. Not included in this repository and not reconstructible from it.** It also contains operator names and internal comments, which is a further reason it is not published here.

Throughout the thesis and this repository it is called the **fire-reference inventory**. It is *not* independently validated ground truth: the Planet/Sentinel-2 polygons are themselves remote-sensing derived, and the KKL polygons supplied the classification training labels. (The label "Ground-truth fire polygons" appears once, inside a retained legacy appendix figure, and is explained as such in the thesis.)

**Expected file:** `data/restricted/fire_reference_inventory.shp` (+ `.shx .dbf .prj .cpg`), or any vector format readable by `geopandas`.

**Expected CRS:** EPSG:4326 (reprojected internally to EPSG:32636).

**Required schema** — only these fields are used:

| field | type | meaning |
|---|---|---|
| `geometry` | Polygon / MultiPolygon | burned area extent |
| `fire_year` | int | year of the fire event |
| `fire_month` | int (1–12) | month of the fire event |
| `fire_date` | int (epoch ms) or date | event date; used to assign the burned month |
| `Source` | str | provenance of the polygon, e.g. `Planet`, `Sentinel-2` |

Other attributes present in the original inventory are unused.

**How it is used:** polygons are rasterised onto the analysis grid per month to produce the per-pixel-per-month burned label `y_burned`; pixels are stratified by fire recurrence (number of months intersecting a polygon) into no-fires / low (1–2) / medium (3–4) / high (5+); and the reference-defined burned footprint is dissolved into the spatial units used in the regional analysis.

**Stages that cannot run without it:** burned-label construction; sample construction and stratification; the severity and persistence targets (which are defined over documented fire months); model training; all fire-detection evaluation; the reference-defined spatial units and every reference-population result.

### 2.2 Project land-cover product (2018 baseline)

A ten-class land-cover product developed within the research group by **Nitzan Hahamof** and provided as an input to this thesis. It combines an agricultural field classification (CropIL), the ESA WorldCover 2021 classification and a Sentinel-2 threshold-based gap-filling step. **It is not an ESA WorldCover product**, and the WorldCover accuracy figure quoted in the thesis (76.7 % ± 0.5 %) refers only to that one component, not to the combined layer. No independent validation of the combined layer was performed within the thesis.

**Third-party research data. Not included in this repository.**

**Expected file:** `data/restricted/landcover_2018.tif` (single band, integer).

**Expected CRS / grid:** EPSG:32636, aligned to the analysis grid. The final analysis uses it at 10 m (4875 × 7794); the supplementary 30 m workflow uses a 30 m version (1625 × 2598).

**Required class codes:**

| code | class | | code | class |
|---|---|---|---|---|
| 10 | forest | | 41 | covered agriculture |
| 11 | orchards | | 50 | built-up |
| 20 | shrubland | | 60 | bare |
| 30 | grassland | | 61 | fallow |
| 40 | cropland | | 80 | water |

`0` is treated as nodata/outside.

**Role:** the **fixed 2018 baseline** is used for all downstream spatial analysis, so that land-cover class is held constant across the study period. It is also supplied to the model as 20 one-hot features (current and previous year). Because no land-cover map exists for 2016–2017, the 2018 map is used as a proxy for those years — least reliable in croplands.

**Stages that cannot run without it:** the 20 land-cover features; spatial-unit construction (units are connected components of burned footprint × land-cover class); all per-land-cover results; within-class RIS prioritization.

---

## 3. Generated data (not committed)

| File | Size | Generated by | Note |
|---|---|---|---|
| `data/raw/full_pixels_10m.csv` | large | `gee/02_sampling_computepixels.ipynb` | per-pixel monthly time series for the sample; **restricted-derived** (the sample is defined by the fire inventory) |
| `data/processed/X_scaled.npy` | ~13.7 GB | `notebooks/04_data_prep.ipynb` | 608,922 × 108 × 52 scaled features; deterministic — regenerate, do not archive |
| `data/processed/y_burned.npy` | 263 MB | `notebooks/04_data_prep.ipynb` | burned labels; **restricted-derived** |
| `data/processed/targets_aux.npz` | 9.3 MB | `notebooks/04_data_prep.ipynb` | the four analytical indicators; **restricted-derived** |
| `data/processed/X_scaled_meta.csv` | 13 MB | `notebooks/04_data_prep.ipynb` | pixel_id, fire-history class, dominant land cover; **restricted-derived** |
| `data/raw/pixel_coords_10m.csv` | 32 MB | `gee/02_sampling_computepixels.ipynb` | coordinates + fire-history class; **restricted-derived** |
| `outputs/inference_10m/*.tif` | ~610 MB | `pipeline_10m/run_inference_10m.py` | final indicator surfaces |
| `outputs/**/units*.csv`, `*.gpkg` | — | `pipeline_10m/polygon_10m.py`, `geometries_10m.py` | per-unit tables/geometries; **restricted-derived** |

**Committed** are `configs/scaler.pkl`, `configs/splits.json`, `configs/feature_names.json` — the fitted scaler, the train/val/test index lists and the feature order. These carry no coordinates, labels or geometry.
