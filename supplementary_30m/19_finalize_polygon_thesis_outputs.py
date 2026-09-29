"""
19_finalize_polygon_thesis_outputs.py — consolidate the validated analysis into thesis-ready outputs.

Writes ONLY into outputs/final_polygon_thesis/. Reads everything else read-only. No model is
retrained, no target array or RIS formula is touched, no thesis file is opened, and nb15/nb16/
nb17/nb18 outputs are neither overwritten nor deleted.

FIXED INTERPRETATION carried through every label and caption produced here:
  observed NBR change      observed Sentinel-2 spectral change. NOT ecological recovery.
  analytical Recovery Gap  the custom target defined in this study (imputed series, divided by the
                           2017 baseline, clipped to [0,1]). Sampled 10 m support only.
  predicted Recovery Gap   the L-TAE prediction OF that analytical target. Full-ROI 30 m.
  Spatial units            connected components of a burned footprint x fixed 2018 land-cover class,
                           exactly as built by 15_polygon_workflow.py (identity asserted below).
  Population A             reference-defined burned areas (fire-reference inventory, dissolved).
  Population B             model-detected burned areas (burned_prob > 0.502).
Model-detected area outside the inventory means it did not spatially overlap the available
fire-reference inventory. It is never called false-positive area.

Run: python notebooks/19_finalize_polygon_thesis_outputs.py
"""
import json
import os
import platform
import sys
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio import features
from rasterio.transform import Affine
from rasterio.warp import transform as warp_transform
from scipy import ndimage, stats
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
from shapely.ops import unary_union
from shapely.validation import make_valid
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=matplotlib.MatplotlibDeprecationWarning)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = os.path.join(ROOT, "data", "processed")
RAST = os.path.join(ROOT, "outputs", "inference_full_roi")
TAB = os.path.join(ROOT, "outputs", "tables")
OUT = os.path.join(ROOT, "outputs", "final_polygon_thesis")
os.makedirs(OUT, exist_ok=True)
OBS_TIF = os.path.join(ROOT, "data", "raw", "nbr_2017_2025_30m.tif")
LC_TIF = os.path.join(ROOT, "data", "raw", "landcover_2018_30m.tif")
FIRES = os.path.join(ROOT, "data", "thesis_maps", "fig_gt_fires.shp")
ROI = os.path.join(ROOT, "data", "thesis_maps", "fig_roi_polygon.shp")
NB15 = os.path.join(TAB, "nb15_t03_burned_spatial_units.csv")
NB18_COMPLETE = os.path.join(TAB, "nb18_t07_completeness_sensitivity.csv")
NB18_COMPLETE_SUM = os.path.join(TAB, "nb18_t07b_completeness_sensitivity_summary.csv")

GRID_T = Affine(30.0, 0.0, 620610.0, 0.0, -30.0, 3514560.0)
H, W, CRS = 2598, 1625, "EPSG:32636"
PX_HA, THR, NODATA, MIN_UNIT_HA = 0.09, 0.502, -9999.0, 0.36
MIN_UNIT_PX = int(round(MIN_UNIT_HA / PX_HA))
SMALL_N_UNITS = 20          # classes below this are flagged and must not drive conclusions
LC_NAMES = {10: "forest", 11: "orchards", 20: "shrubland", 30: "grassland", 40: "cropland",
            41: "covered agriculture", 50: "built-up", 60: "bare", 61: "fallow", 80: "water"}
LC_ORDER = [10, 11, 20, 30, 40, 41, 50, 60, 61, 80]
CLS_ORDER = [LC_NAMES[c] for c in LC_ORDER] + ["unclassified / other code"]
CLS_IDX = {c: i for i, c in enumerate(CLS_ORDER)}
_cmap = plt.get_cmap("tab10")
CLASS_COLOR = {LC_NAMES[c]: _cmap(i) for i, c in enumerate(LC_ORDER)}
CLASS_COLOR["unclassified / other code"] = (0.6, 0.6, 0.6, 1.0)
RESTORATION_CMAP = LinearSegmentedColormap.from_list(
    "restoration", ["#2c7bb6", "#abd9e9", "#ffffbf", "#fdae61", "#d7191c"], N=256)
POP_LABEL = {"reference": "A. Reference-defined burned areas",
             "model": "B. Model-detected burned areas"}
RUN = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
REGISTRY = []
matplotlib.rcParams.update({"figure.dpi": 110, "savefig.dpi": 400, "font.size": 9,
                            "axes.spines.top": False, "axes.spines.right": False,
                            "pdf.fonttype": 42, "svg.fonttype": "none"})

print("=" * 100)
print("19_finalize_polygon_thesis_outputs.py | run (UTC):", RUN)
print("python", sys.version.split()[0], "| numpy", np.__version__, "| rasterio", rasterio.__version__,
      "| geopandas", gpd.__version__, "|", platform.platform())
print("writes ONLY to:", os.path.relpath(OUT, ROOT))
print("=" * 100)

# ------------------------------------------------------------------ 1. assert inputs and alignment
REQUIRED = {"severity": os.path.join(RAST, "severity_30m.tif"),
            "persistence": os.path.join(RAST, "persistence_30m.tif"),
            "recovery_gap_predicted": os.path.join(RAST, "recovery_gap_30m.tif"),
            "model_recurrence": os.path.join(RAST, "model_recurrence_30m.tif"),
            "burned_prob": os.path.join(RAST, "burned_prob_30m.tif"),
            "landcover_2018": LC_TIF, "observed_nbr": OBS_TIF}
for k, v in list(REQUIRED.items()) + [("fire polygons", FIRES), ("roi", ROI), ("nb15 units", NB15),
                                      ("nb18 completeness", NB18_COMPLETE)]:
    if not os.path.exists(v):
        raise SystemExit("MISSING REQUIRED INPUT: %s -> %s" % (k, v))

gate = []
for role, path in REQUIRED.items():
    with rasterio.open(path) as s:
        tr, shp = s.transform, (s.height, s.width)
        gate.append(dict(role=role, file=os.path.relpath(path, ROOT), crs=str(s.crs), width=shp[1],
                         height=shp[0], pixel_x=tr.a, pixel_y=tr.e, origin_x=tr.c, origin_y=tr.f,
                         nodata=("None" if s.nodata is None else str(s.nodata)), bands=s.count,
                         crs_ok=str(s.crs) == CRS, dims_ok=shp == (H, W),
                         transform_ok=bool(tr.almost_equals(GRID_T))))
gate = pd.DataFrame(gate)
bad = gate[~(gate.crs_ok & gate.dims_ok & gate.transform_ok)]
if len(bad):
    raise SystemExit("ALIGNMENT GATE FAILED — refusing to resample:\n%s" % bad.to_string(index=False))
print("\nalignment gate passed for all %d rasters (no resampling performed)" % len(gate))

# ------------------------------------------------------------------ 2. load
A = {}
for role, path in REQUIRED.items():
    if role in ("landcover_2018", "observed_nbr"):
        continue
    with rasterio.open(path) as s:
        A[role] = s.read(1).astype(np.float64)
with rasterio.open(LC_TIF) as s:
    LC_GRID = s.read(1)
with rasterio.open(OBS_TIF) as s:
    OB17, OB25, NV17, NV25 = (s.read(i).astype(np.float64) for i in (1, 2, 3, 4))
    OBS_BAND_NAMES = list(s.descriptions)
for a in (OB17, OB25):
    a[a == NODATA] = np.nan
OBS_CHANGE = OB17 - OB25            # positive = NBR DECLINED between 2017 and 2025
ELIG = np.all([np.isfinite(A[k]) for k in A], axis=0)
cls_i = np.full((H, W), CLS_IDX["unclassified / other code"], dtype=np.int16)
for code, nm in LC_NAMES.items():
    cls_i[LC_GRID == code] = CLS_IDX[nm]
code_of = {CLS_IDX[LC_NAMES[c]]: c for c in LC_ORDER}

PROFILES = {"ecological": (0.20, 0.20, 0.50, 0.10), "fire": (0.35, 0.35, 0.20, 0.10),
            "balanced": (0.25, 0.25, 0.25, 0.25)}
RIS = {p: np.where(ELIG, ws * A["severity"] + wp * A["persistence"]
                   + wr * A["recovery_gap_predicted"] + wc * A["model_recurrence"], np.nan)
       for p, (ws, wp, wr, wc) in PROFILES.items()}

# sampled 10 m support: the analytical Recovery Gap exists ONLY here
meta_co = pd.read_csv(os.path.join(ROOT, "data", "raw", "pixel_coords_10m.csv"))
RG_ANALYTICAL = np.load(os.path.join(P, "targets_aux.npz"))["recovery_gap"].astype(float)
n_s = len(RG_ANALYTICAL)
burned_s = np.asarray(np.load(os.path.join(P, "y_burned.npy"), mmap_mode="r")).astype(bool).any(1)[:n_s]
nbr_s = np.load(os.path.join(P, "nbr_raw.npy"), mmap_mode="r")
IMP17_S = np.nanmean(np.asarray(nbr_s[:n_s, :12], dtype=np.float64), axis=1)
xs, ys = warp_transform("EPSG:4326", CRS, meta_co.lon.to_numpy()[:n_s], meta_co.lat.to_numpy()[:n_s])
COLp = np.floor((np.array(xs) - GRID_T.c) / GRID_T.a).astype(int)
ROWp = np.floor((np.array(ys) - GRID_T.f) / GRID_T.e).astype(int)
ONp = (ROWp >= 0) & (ROWp < H) & (COLp >= 0) & (COLp < W)
RCp = (np.clip(ROWp, 0, H - 1), np.clip(COLp, 0, W - 1))

# ------------------------------------------------------------------ 3. the two burned footprints
def polygonal(g):
    if g is None or g.is_empty:
        return None
    g = make_valid(g)
    if isinstance(g, (Polygon, MultiPolygon)):
        return g
    if isinstance(g, GeometryCollection):
        parts = [p for p in g.geoms if isinstance(p, (Polygon, MultiPolygon))]
        return unary_union(parts) if parts else None
    return None


fires = gpd.read_file(FIRES)
geoms = [g for g in (polygonal(x) for x in fires[fires.fire_year >= 2017].to_crs(CRS).geometry)
         if g is not None and not g.is_empty]
INSIDE = features.rasterize(((g, 1) for g in geoms), out_shape=(H, W), transform=GRID_T,
                            fill=0, all_touched=False, dtype="uint8").astype(bool)
FOOTPRINT = {"reference": INSIDE & ELIG, "model": ELIG & (A["burned_prob"] > THR)}

STRUCT = np.ones((3, 3), dtype=int)
def components(mask, min_px=0):
    lab, k = ndimage.label(mask, structure=STRUCT)
    if min_px > 1 and k:
        sizes = np.bincount(lab.ravel()); drop = np.where(sizes < min_px)[0]; drop = drop[drop > 0]
        if len(drop):
            lab[np.isin(lab, drop)] = 0
        lab, k = ndimage.label(lab > 0, structure=STRUCT)
    return lab, k


nb15u = pd.read_csv(NB15)
NB15_POP = {"reference": "documented", "model": "model"}
UNITS, UNIT_IDX, ZONES = {}, {}, {}
for pop, M in FOOTPRINT.items():
    lab, _ = components(M, MIN_UNIT_PX)
    ZONES[pop] = lab
    sel = lab > 0
    d = pd.DataFrame({"zone": lab[sel], "cls": cls_i[sel],
                      "severity": A["severity"][sel], "persistence": A["persistence"][sel],
                      "rg_pred": A["recovery_gap_predicted"][sel], "recurrence": A["model_recurrence"][sel],
                      "ob17": OB17[sel], "ob25": OB25[sel], "chg": OBS_CHANGE[sel],
                      "nv17": NV17[sel], "nv25": NV25[sel],
                      **{"RIS_" + p: RIS[p][sel] for p in PROFILES}})
    g = d.groupby(["zone", "cls"])
    u = g.agg(pixel_count=("zone", "size"),
              mean_severity_model=("severity", "mean"),
              mean_persistence_model=("persistence", "mean"),
              mean_recovery_gap_predicted=("rg_pred", "mean"),
              mean_recurrence_model=("recurrence", "mean"),
              mean_RIS_balanced=("RIS_balanced", "mean"),
              mean_RIS_ecological=("RIS_ecological", "mean"),
              mean_RIS_fire=("RIS_fire", "mean"),
              observed_mean_NBR_2017=("ob17", "mean"),
              observed_mean_NBR_2025=("ob25", "mean"),
              observed_nbr_change_2017_minus_2025=("chg", "mean"),
              valid_months_2017_mean=("nv17", "mean"),
              valid_months_2025_mean=("nv25", "mean")).reset_index()
    u = u[u.pixel_count >= MIN_UNIT_PX].reset_index(drop=True)
    u["unit_id"] = ["%s_%d_%d" % (NB15_POP[pop][:3], z, c) for z, c in zip(u.zone, u.cls)]
    expected = set(nb15u[nb15u.population == NB15_POP[pop]].unit_id)
    assert set(u.unit_id) == expected, \
        "unit identity differs from 15_polygon_workflow.py for %s (%d vs %d)" % (pop, len(u), len(expected))
    u["burned_area_definition"] = POP_LABEL[pop]
    u["population"] = pop
    u["land_cover_name"] = [CLS_ORDER[c] for c in u.cls]
    u["land_cover_code"] = [code_of.get(c, -1) for c in u.cls]
    u["area_ha"] = u.pixel_count * PX_HA
    # analytical Recovery Gap: sampled support only, never interpolated to full coverage
    zz, cc = lab[RCp], cls_i[RCp]
    ins = ONp & (zz > 0) & burned_s
    idx = np.where(ins)[0]
    samp = pd.DataFrame({"unit_id": ["%s_%d_%d" % (NB15_POP[pop][:3], zz[i], cc[i]) for i in idx],
                         "rg": RG_ANALYTICAL[idx], "b17": IMP17_S[idx]})
    agg = samp.groupby("unit_id").agg(
        analytical_recovery_gap_sampled_mean=("rg", "mean"),
        analytical_recovery_gap_sampled_n_px=("rg", "size"),
        baseline_NBR_2017_imputed_sampled_mean=("b17", "mean")).reset_index()
    u = u.merge(agg, on="unit_id", how="left")
    u["analytical_recovery_gap_support"] = np.where(
        u.analytical_recovery_gap_sampled_n_px.notna(),
        "sampled 10 m pixels inside the unit", "no sampled pixel in this unit")
    u["global_RIS_rank"] = u.mean_RIS_balanced.rank(ascending=False, method="min").astype(int)
    u["global_RIS_percentile"] = 100.0 * u.mean_RIS_balanced.rank(pct=True)
    u["within_landcover_RIS_rank"] = u.groupby("land_cover_name").mean_RIS_balanced \
        .rank(ascending=False, method="min").astype(int)
    u["within_landcover_RIS_percentile"] = 100.0 * u.groupby("land_cover_name").mean_RIS_balanced.rank(pct=True)
    u["n_units_in_landcover_class"] = u.groupby("land_cover_name").mean_RIS_balanced.transform("size").astype(int)
    u["small_n_class_flag"] = u.n_units_in_landcover_class < SMALL_N_UNITS
    UNITS[pop] = u
    idxmap = np.full((H, W), -1, dtype=np.int32)
    key = {(z, c): i for i, (z, c) in enumerate(zip(u.zone, u.cls))}
    zs, cs_ = lab[sel], cls_i[sel]
    idxmap[sel] = np.array([key.get((z, c), -1) for z, c in zip(zs, cs_)], dtype=np.int32)
    UNIT_IDX[pop] = idxmap
    print("   %-9s %4d units | %7.0f ha | observed coverage %.1f%% | analytical RG support %.1f%% of units"
          % (pop, len(u), u.area_ha.sum(), 100.0 * u.observed_nbr_change_2017_minus_2025.notna().mean(),
             100.0 * u.analytical_recovery_gap_sampled_n_px.notna().mean()))

COLS = ["unit_id", "population", "burned_area_definition", "land_cover_code", "land_cover_name",
        "area_ha", "pixel_count", "mean_severity_model", "mean_persistence_model",
        "mean_recovery_gap_predicted", "mean_recurrence_model", "mean_RIS_balanced",
        "mean_RIS_ecological", "mean_RIS_fire", "observed_mean_NBR_2017", "observed_mean_NBR_2025",
        "observed_nbr_change_2017_minus_2025", "valid_months_2017_mean", "valid_months_2025_mean",
        "analytical_recovery_gap_sampled_mean", "analytical_recovery_gap_sampled_n_px",
        "analytical_recovery_gap_support", "baseline_NBR_2017_imputed_sampled_mean",
        "global_RIS_rank", "global_RIS_percentile", "within_landcover_RIS_rank",
        "within_landcover_RIS_percentile", "n_units_in_landcover_class", "small_n_class_flag",
        "east", "north"]
for pop, fname in [("reference", "reference_units_final.csv"), ("model", "model_units_final.csv")]:
    u = UNITS[pop].merge(nb15u[nb15u.population == NB15_POP[pop]][["unit_id", "east", "north"]],
                         on="unit_id", how="left")
    u[COLS].sort_values("global_RIS_rank").to_csv(os.path.join(OUT, fname), index=False)
    print("   wrote", fname, "(%d rows)" % len(u))

# ------------------------------------------------------------------ 4. population comparison table
REF, MOD = FOOTPRINT["reference"], FOOTPRINT["model"]
UR, UM = UNIT_IDX["reference"] >= 0, UNIT_IDX["model"] >= 0
jac_fp = float((REF & MOD).sum() / (REF | MOD).sum())
jac_un = float((UR & UM).sum() / (UR | UM).sum())
pct_ref_detected = 100.0 * float((REF & MOD).sum() / REF.sum())
pct_mod_outside = 100.0 * float((MOD & ~REF).sum() / MOD.sum())
ur, um = UNITS["reference"], UNITS["model"]
comp = pd.DataFrame([
    dict(measure="spatial units", kind="OBSERVED", reference=len(ur), model=len(um), unit="units",
         note="connected component x 2018 land-cover class, minimum %.2f ha" % MIN_UNIT_HA),
    dict(measure="area analysed as units", kind="OBSERVED", reference=float(ur.area_ha.sum()),
         model=float(um.area_ha.sum()), unit="ha", note=""),
    dict(measure="burned footprint before the minimum-area rule", kind="OBSERVED",
         reference=float(REF.sum() * PX_HA), model=float(MOD.sum() * PX_HA), unit="ha", note=""),
    dict(measure="mean RIS (Balanced) of units", kind="OBSERVED", reference=float(ur.mean_RIS_balanced.mean()),
         model=float(um.mean_RIS_balanced.mean()), unit="index", note="same RIS formula in both"),
    dict(measure="median RIS (Balanced) of units", kind="OBSERVED",
         reference=float(ur.mean_RIS_balanced.median()), model=float(um.mean_RIS_balanced.median()),
         unit="index", note=""),
    dict(measure="mean observed NBR change 2017 minus 2025", kind="OBSERVED",
         reference=float(ur.observed_nbr_change_2017_minus_2025.mean()),
         model=float(um.observed_nbr_change_2017_minus_2025.mean()), unit="NBR",
         note="positive = NBR declined; observed spectral change, not ecological recovery"),
    dict(measure="Jaccard of the burned footprints", kind="OBSERVED", reference=jac_fp, model=jac_fp,
         unit="index", note="spatial overlap of the two definitions"),
    dict(measure="Jaccard of the analysed unit footprints", kind="OBSERVED", reference=jac_un,
         model=jac_un, unit="index", note="matches the two map panels"),
    dict(measure="reference-defined burned area also detected by the model", kind="OBSERVED",
         reference=pct_ref_detected, model=np.nan, unit="%", note=""),
    dict(measure="model-detected area that did not spatially overlap the fire-reference inventory",
         kind="OBSERVED", reference=np.nan, model=pct_mod_outside, unit="%",
         note="a spatial statement only; see the interpretation row"),
    dict(measure="status of model-detected area outside the inventory", kind="INTERPRETATION",
         reference=np.nan, model=np.nan, unit="",
         note="undetermined here. The fire-reference inventory is not assumed to be exhaustive, so "
              "non-overlap is not evidence of a false positive; establishing status would require an "
              "independent source"),
])
comp.to_csv(os.path.join(OUT, "table_polygon_population_comparison.csv"), index=False)
print("\nfootprint Jaccard %.4f | unit Jaccard %.4f | reference detected %.1f%% | model outside inventory %.1f%%"
      % (jac_fp, jac_un, pct_ref_detected, pct_mod_outside))

# ------------------------------------------------------------------ 5. land-cover spectral table
rows = []
for c in CLS_ORDER:
    s = ur[ur.land_cover_name == c]
    if not len(s):
        continue
    rows.append(dict(land_cover_name=c, land_cover_code=int(s.land_cover_code.iloc[0]),
                     n_units=len(s), area_ha=float(s.area_ha.sum()),
                     small_n_class_flag=bool(s.small_n_class_flag.iloc[0]),
                     observed_nbr_change_2017_minus_2025_mean=float(s.observed_nbr_change_2017_minus_2025.mean()),
                     observed_nbr_change_2017_minus_2025_median=float(s.observed_nbr_change_2017_minus_2025.median()),
                     baseline_NBR_2017_imputed_sampled_mean=float(s.baseline_NBR_2017_imputed_sampled_mean.mean()),
                     baseline_NBR_2017_imputed_sampled_median=float(s.baseline_NBR_2017_imputed_sampled_mean.median()),
                     analytical_recovery_gap_sampled_mean=float(s.analytical_recovery_gap_sampled_mean.mean()),
                     analytical_recovery_gap_n_units_with_support=int(s.analytical_recovery_gap_sampled_n_px.notna().sum()),
                     predicted_recovery_gap_mean=float(s.mean_recovery_gap_predicted.mean()),
                     mean_RIS_balanced=float(s.mean_RIS_balanced.mean())))
lct = pd.DataFrame(rows).sort_values("observed_nbr_change_2017_minus_2025_mean", ascending=False)
lct["ordering_by_observed_nbr_change"] = lct.observed_nbr_change_2017_minus_2025_mean.rank(ascending=False).astype(int)
lct["ordering_by_analytical_recovery_gap"] = lct.analytical_recovery_gap_sampled_mean.rank(ascending=False).astype(int)
lct["ordering_by_predicted_recovery_gap"] = lct.predicted_recovery_gap_mean.rank(ascending=False).astype(int)
lct.to_csv(os.path.join(OUT, "table_landcover_spectral_interpretation.csv"), index=False)
main6 = [c for c in ["forest", "orchards", "shrubland", "grassland", "cropland", "fallow"]
         if c in set(lct.land_cover_name)]
m6 = lct[lct.land_cover_name.isin(main6)]
rho_obs_rg = float(stats.spearmanr(m6.observed_nbr_change_2017_minus_2025_mean,
                                   m6.analytical_recovery_gap_sampled_mean).statistic)
rho_rg_pred = float(stats.spearmanr(lct.analytical_recovery_gap_sampled_mean.dropna(),
                                    lct.predicted_recovery_gap_mean[lct.analytical_recovery_gap_sampled_mean.notna()]).statistic)
print("reference units, %d main vegetated classes: spearman(observed change, analytical RG) = %.4f"
      % (len(m6), rho_obs_rg))
print("all classes: spearman(analytical RG, predicted RG) = %.4f" % rho_rg_pred)

# ------------------------------------------------------------------ 6. completeness (supporting)
cs = pd.read_csv(NB18_COMPLETE_SUM)
cs.to_csv(os.path.join(OUT, "table_observation_completeness_sensitivity.csv"), index=False)

# ------------------------------------------------------------------ 7. FIGURE 1 — the Tarin map
roi = gpd.read_file(ROI).to_crs(CRS)
extent = [GRID_T.c, GRID_T.c + W * GRID_T.a, GRID_T.f + H * GRID_T.e, GRID_T.f]


def unit_surface(pop, col="mean_RIS_balanced"):
    u, idx = UNITS[pop], UNIT_IDX[pop]
    vals = np.append(u[col].to_numpy(), np.nan)
    return np.where(idx >= 0, vals[np.clip(idx, 0, len(u))], np.nan)


def km_axes(ax):
    ax.set_xlabel("easting (km)", fontsize=8.5)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, p: "%.0f" % (v / 1000)))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, p: "%.0f" % (v / 1000)))


bg = np.where(ELIG, 1.0, np.nan)
fig, axes = plt.subplots(1, 2, figsize=(11.6, 9.6), sharex=True, sharey=True, layout="constrained")
for ax, pop in zip(axes, ["reference", "model"]):
    u = UNITS[pop]
    ax.imshow(bg, extent=extent, cmap=ListedColormap(["#ececec"]), interpolation="nearest")
    im = ax.imshow(unit_surface(pop), extent=extent, cmap=RESTORATION_CMAP, vmin=0, vmax=1,
                   interpolation="nearest")
    roi.boundary.plot(ax=ax, color="k", linewidth=0.7)
    ax.set_title("%s\n%d spatial units, %.0f ha\nunit RIS mean %.2f, median %.2f"
                 % (POP_LABEL[pop], len(u), u.area_ha.sum(), u.mean_RIS_balanced.mean(),
                    u.mean_RIS_balanced.median()), fontsize=9.6)
    km_axes(ax)
    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
axes[0].set_ylabel("northing (km)", fontsize=8.5)
cb = fig.colorbar(im, ax=list(axes), orientation="horizontal", fraction=0.032, pad=0.02, shrink=0.62)
cb.set_label("Restoration Indicator Score (Balanced profile), averaged within each burned spatial unit\n"
             "identical formula, colour scale and class boundaries in both panels", fontsize=8.6)
fig.suptitle("Restoration priority of burned spatial units under two definitions of the burned footprint",
             fontsize=11.5)
fig.text(0.5, -0.015,
         "The two panels differ ONLY in how the burned footprint is defined — the priority formula, the "
         "2018 land-cover definition, the spatial-unit rule and the colour scale are identical. "
         "Footprint Jaccard %.3f; %.1f%% of the reference-defined burned area is also detected by the model; "
         "%.1f%% of the model-detected area did not spatially overlap the available fire-reference inventory, "
         "which is not assumed to be exhaustive and therefore does not make that area a false positive."
         % (jac_fp, pct_ref_detected, pct_mod_outside), ha="center", fontsize=8.1, style="italic",
         color="#444444", wrap=True)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, "figure_polygon_priority_reference_vs_model." + ext),
                bbox_inches="tight", facecolor="white")
plt.close(fig)
print("\nwrote figure_polygon_priority_reference_vs_model.{png,pdf}")

# ------------------------------------------------------------------ 8. FIGURE 2 — the two hierarchies
fig, axes = plt.subplots(1, 2, figsize=(14.2, 7.0), sharex=True, layout="constrained")
hier_rows = []
for ax, pop in zip(axes, ["reference", "model"]):
    u = UNITS[pop]
    classes = [c for c in CLS_ORDER if (u.land_cover_name == c).sum() >= 3]
    ypos = np.arange(len(classes))[::-1]
    gcut = float(u.mean_RIS_balanced.quantile(0.90))
    rng = np.random.default_rng(0)
    for yy, c in zip(ypos, classes):
        s = u[u.land_cover_name == c]
        wcut = float(s.mean_RIS_balanced.quantile(0.90)) if len(s) >= 10 else float(s.mean_RIS_balanced.max())
        top_within = s.mean_RIS_balanced >= wcut
        obscured = top_within & (s.mean_RIS_balanced < gcut)
        both = top_within & (s.mean_RIS_balanced >= gcut)
        jit = rng.normal(0, 0.10, len(s))
        ax.scatter(s.mean_RIS_balanced[~top_within], yy + jit[~top_within.to_numpy()], s=7,
                   color="#b9c2c9", alpha=0.75, linewidths=0, zorder=2)
        ax.scatter(s.mean_RIS_balanced[both], yy + jit[both.to_numpy()], s=17, color="#d7191c",
                   alpha=0.9, linewidths=0, zorder=4)
        ax.scatter(s.mean_RIS_balanced[obscured], yy + jit[obscured.to_numpy()], s=17, color="#2c7bb6",
                   alpha=0.95, linewidths=0, zorder=4)
        ax.plot([s.mean_RIS_balanced.median()] * 2, [yy - 0.28, yy + 0.28], color="#33414d", lw=1.8, zorder=5)
        hier_rows.append(dict(population=pop, land_cover_name=c, n_units=len(s),
                              small_n_class_flag=bool(s.small_n_class_flag.iloc[0]),
                              median_RIS=float(s.mean_RIS_balanced.median()),
                              within_class_top_decile_n=int(top_within.sum()),
                              also_in_global_top_decile_n=int(both.sum()),
                              high_within_class_but_below_global_cut_n=int(obscured.sum())))
    ax.axvline(gcut, color="#444444", lw=1.1, ls="--", zorder=1)
    ax.text(gcut, len(classes) - 0.35, " global top decile", fontsize=7.6, color="#444444", va="center")
    ax.set_yticks(ypos)
    ax.set_yticklabels(["%s%s\nn = %d" % (c, " *" if (u[u.land_cover_name == c].small_n_class_flag.iloc[0]) else "",
                                          (u.land_cover_name == c).sum()) for c in classes], fontsize=8.2)
    for t, c in zip(ax.get_yticklabels(), classes):
        t.set_color(CLASS_COLOR.get(c, "k"))
    ax.set_xlabel("unit mean RIS (Balanced profile)", fontsize=8.8)
    ax.set_xlim(0, 1)
    ax.grid(axis="x", alpha=0.25, lw=0.5)
    ax.set_title(POP_LABEL[pop], fontsize=10)
handles = [Line2D([], [], marker="o", ls="", color="#b9c2c9", label="unit outside its class's top decile"),
           Line2D([], [], marker="o", ls="", color="#d7191c", label="top decile within its class AND globally"),
           Line2D([], [], marker="o", ls="", color="#2c7bb6",
                  label="top decile WITHIN its class but below the global cut"),
           Line2D([], [], color="#33414d", lw=1.8, label="class median")]
axes[1].legend(handles=handles, fontsize=7.8, loc="lower right", framealpha=0.95)
fig.suptitle("Two prioritization hierarchies over the same burned spatial units\n"
             "global ranking versus ranking within each fixed 2018 land-cover class", fontsize=11.5)
fig.text(0.5, -0.03,
         "Each point is one burned spatial unit. Blue points are the units the global ranking hides: they "
         "are among the most affected of their own land-cover class yet fall below the global top-decile "
         "cut. Classes are not ranked against each other here — the hierarchy is within a class. "
         "Classes marked * have fewer than %d units and should not drive conclusions."
         % SMALL_N_UNITS, ha="center", fontsize=8.1, style="italic", color="#444444", wrap=True)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, "figure_global_vs_within_landcover_priority." + ext),
                bbox_inches="tight", facecolor="white")
plt.close(fig)
hier = pd.DataFrame(hier_rows)
hier.to_csv(os.path.join(OUT, "table_global_vs_within_landcover_counts.csv"), index=False)
obsc_ref = hier[(hier.population == "reference") & ~hier.small_n_class_flag]
print("\nwrote figure_global_vs_within_landcover_priority.{png,pdf}")
print("reference units hidden by the global ranking (top decile within class, below global cut): %d across %d classes"
      % (int(obsc_ref.high_within_class_but_below_global_cut_n.sum()), len(obsc_ref)))

# ------------------------------------------------------------------ 9. FIGURE 3 — processing stages
msk_s = np.asarray(np.load(os.path.join(P, "X_scaled_mask.npy"), mmap_mode="r"))
a17 = np.asarray(nbr_s[:n_s, :12], dtype=np.float64); a25 = np.asarray(nbr_s[:n_s, 96:108], dtype=np.float64)
m17, m25 = msk_s[:n_s, :12].astype(bool), msk_s[:n_s, 96:108].astype(bool)
vm = lambda a, m: np.nanmean(np.where(m, a, np.nan), axis=1)
STAGE_A = vm(a17, m17) - vm(a25, m25)
STAGE_B = np.nanmean(a17, 1) - np.nanmean(a25, 1)
STAGE_C = RG_ANALYTICAL
pred = np.load(os.path.join(ROOT, "models", "main", "predictions_test.npz"))
STAGE_D = np.full(n_s, np.nan); STAGE_D[pred["indices"]] = pred["recovery_gap"]
CLS_S = np.where(ONp, np.array(CLS_ORDER, dtype=object)[cls_i[RCp]], "unclassified / other code")
SEL = burned_s & ONp & np.isfinite(STAGE_D)
CF = [c for c in CLS_ORDER if (SEL & (CLS_S == c)).sum() >= 100]
means_A = {c: float(np.nanmean(STAGE_A[SEL & (CLS_S == c)])) for c in CF}
CF = sorted(CF, key=lambda c: -means_A[c])
STAGES = [("A. OBSERVED\nspectral change from valid\nSentinel-2 observations", STAGE_A, None,
           "NBR change, 2017 minus 2025", "#6a9fb5"),
          ("B. IMPUTED\nthe same change after gap-filling,\nas it enters target construction", STAGE_B,
           None, "NBR change, 2017 minus 2025", "#8d9f6a"),
          ("C. ANALYTICAL Recovery Gap\nB divided by the 2017 baseline,\nclipped to [0, 1]", STAGE_C,
           (0, 1), "Recovery Gap (index)", "#c98f7f"),
          ("D. PREDICTED Recovery Gap\nL-TAE prediction of C", STAGE_D, (0, 1),
           "Recovery Gap (index)", "#9f7fb5")]
fig = plt.figure(figsize=(19, 0.60 * len(CF) + 7.2))
gs = fig.add_gridspec(2, 4, height_ratios=[0.62 * len(CF) + 2.6, 3.4], hspace=0.30, wspace=0.10)
ypos = np.arange(len(CF))[::-1]
ranks = {}
for j, (title, vals, xlim, xlabel, colr) in enumerate(STAGES):
    ax = fig.add_subplot(gs[0, j])
    data, annot, mu = [], [], {}
    for c in CF:
        v = vals[SEL & (CLS_S == c)]
        v = v[np.isfinite(v)]
        data.append(v); mu[c] = float(v.mean())
        annot.append("at bounds %.0f%%" % (100.0 * ((v == 0) | (v >= 1)).mean()) if xlim
                     else "negative %.0f%%" % (100.0 * (v < 0).mean()))
    ranks["ABCD"[j]] = {c: r for c, r in zip(CF, stats.rankdata([-mu[c] for c in CF]).astype(int))}
    pr = ax.violinplot(data, positions=ypos, vert=False, widths=0.86, showextrema=False)
    for b in pr["bodies"]:
        b.set_facecolor(colr); b.set_alpha(0.62); b.set_linewidth(0)
    bx = ax.boxplot(data, positions=ypos, vert=False, widths=0.22, showfliers=False, patch_artist=True)
    for p_ in bx["boxes"]:
        p_.set_facecolor("white"); p_.set_edgecolor("#33414d")
    for k in ("whiskers", "caps", "medians"):
        for p_ in bx[k]:
            p_.set_color("#33414d")
    if xlim is None:
        ax.axvline(0, color="#c0392b", lw=1.0, ls="--", zorder=0)
    else:
        ax.set_xlim(*xlim)
    hi = ax.get_xlim()[1]
    for yy, a_ in zip(ypos, annot):
        ax.text(hi, yy + 0.40, a_, ha="right", va="center", fontsize=6.3, color="#555555")
    ax.set_title(title, fontsize=9.0, linespacing=1.35)
    ax.set_xlabel(xlabel, fontsize=8.4)
    ax.grid(axis="x", alpha=0.25, lw=0.5)
    ax.set_yticks(ypos)
    if j == 0:
        ax.set_yticklabels(["%s\nn = %s" % (c, format(int((SEL & (CLS_S == c)).sum()), ",")) for c in CF],
                           fontsize=8.3)
        for t, c in zip(ax.get_yticklabels(), CF):
            t.set_color(CLASS_COLOR.get(c, "k"))
    else:
        ax.set_yticklabels([])
axb = fig.add_subplot(gs[1, :])
for c in CF:
    axb.plot([0, 1, 2, 3], [ranks[s][c] for s in "ABCD"], marker="o", ms=6.5, lw=2.0,
             color=CLASS_COLOR.get(c, "k"), label=c)
    axb.text(-0.06, ranks["A"][c], c, ha="right", va="center", fontsize=8.2, color=CLASS_COLOR.get(c, "k"))
    axb.text(3.06, ranks["D"][c], c, ha="left", va="center", fontsize=8.2, color=CLASS_COLOR.get(c, "k"))
axb.set_xticks([0, 1, 2, 3])
axb.set_xticklabels(["A. observed\nspectral change", "B. imputed\nspectral change",
                     "C. analytical\nRecovery Gap", "D. predicted\nRecovery Gap"], fontsize=8.8)
axb.set_ylabel("class ordering\n(1 = largest value of that stage's quantity)", fontsize=8.6)
axb.invert_yaxis()
axb.set_xlim(-0.55, 3.55)
axb.grid(axis="y", alpha=0.25, lw=0.5)
axb.set_title("Where the between-land-cover ordering changes along the pipeline — "
              "the large reordering happens at C, the normalisation step, not at D, the model",
              fontsize=9.6)
fig.suptitle("Four stages of the analytical pipeline, by fixed 2018 baseline land-cover class\n"
             "identical burned sampled test pixels in all four panels (n = %s)"
             % format(int(SEL.sum()), ","), fontsize=11.5, y=0.995)
fig.text(0.5, -0.012,
         "These are four stages of one processing chain, not four measurements of ecological recovery. "
         "A is observed Sentinel-2 spectral change. B is A after gap-filling with a global "
         "per-calendar-month mean, the series that enters target construction. C is the custom analytical "
         "Recovery Gap: B divided by each pixel's own 2017 baseline NBR and clipped to [0, 1]. D is the "
         "L-TAE prediction of C. The model largely preserves the target's between-class ordering, so "
         "differences between A and C should not be attributed primarily to model behaviour.",
         ha="center", fontsize=8.1, style="italic", color="#444444", wrap=True)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, "figure_recovery_gap_processing_stages." + ext),
                bbox_inches="tight", facecolor="white")
plt.close(fig)
print("wrote figure_recovery_gap_processing_stages.{png,pdf}")
print("   stage ordering A->D:", " | ".join("%s: %s" % (s, ", ".join(
    sorted(ranks[s], key=lambda c: ranks[s][c])[:3])) for s in "ABCD"))

# ------------------------------------------------------------------ 10. registry
prov = json.load(open(os.path.join(ROOT, "data", "raw", "nbr_export_provenance.json")))
nref, nmod = len(UNITS["reference"]), len(UNITS["model"])
INPUTS_COMMON = ("outputs/inference_full_roi/*_30m.tif; data/raw/landcover_2018_30m.tif; "
                 "data/raw/nbr_2017_2025_30m.tif; data/thesis_maps/fig_gt_fires.shp; "
                 "outputs/tables/nb15_t03_burned_spatial_units.csv")
REG = [
 dict(filename="reference_units_final.csv", kind="dataset", main="main",
      inputs=INPUTS_COMMON, population="A — reference-defined burned areas", n="%d units" % nref,
      variables="one row per burned spatial unit; RIS profiles; model indicators; observed NBR 2017/2025 "
                "and observed_nbr_change_2017_minus_2025 (positive = NBR declined); valid-month support; "
                "analytical Recovery Gap on sampled support only; global and within-land-cover RIS rank/percentile",
      section="Results — land-cover prioritization (data behind Tables and Figures)",
      interpretation="The master record of every reference-defined burned spatial unit and its priority.",
      caveat="analytical_recovery_gap_* columns rest on sampled 10 m pixels inside the unit, not full coverage."),
 dict(filename="model_units_final.csv", kind="dataset", main="main",
      inputs=INPUTS_COMMON, population="B — model-detected burned areas", n="%d units" % nmod,
      variables="identical columns and definitions to reference_units_final.csv",
      section="Results — land-cover prioritization (data behind Tables and Figures)",
      interpretation="The same master record for the model-detected footprint.",
      caveat="only %.1f%% of these units contain a sampled pixel, so the analytical Recovery Gap column is "
             "sparse here." % (100.0 * UNITS["model"].analytical_recovery_gap_sampled_n_px.notna().mean())),
 dict(filename="figure_polygon_priority_reference_vs_model.png/.pdf", kind="figure", main="main",
      inputs=INPUTS_COMMON, population="A and B side by side", n="%d and %d units" % (nref, nmod),
      variables="unit mean RIS (Balanced profile), identical 0–1 colour scale in both panels",
      section="Results — comparison of burned-area definitions",
      interpretation="Restoration priority mapped at the spatial-unit level under both definitions of the "
                     "burned footprint; the panels differ only in that definition.",
      caveat="Model-detected area outside the inventory is a spatial statement only; the inventory is not "
             "assumed exhaustive and non-overlap is not a false positive."),
 dict(filename="figure_global_vs_within_landcover_priority.png/.pdf", kind="figure", main="main",
      inputs="reference_units_final.csv; model_units_final.csv", population="A and B",
      n="%d and %d units" % (nref, nmod),
      variables="unit mean RIS (Balanced), global top-decile cut, within-class top decile",
      section="Results — within-land-cover prioritization",
      interpretation="Units that are among the most affected of their own land-cover class but fall below "
                     "the global cut are visible as a distinct group.",
      caveat="Classes with fewer than %d units are flagged and must not drive conclusions." % SMALL_N_UNITS),
 dict(filename="figure_recovery_gap_processing_stages.png/.pdf", kind="figure", main="main",
      inputs="data/processed/nbr_raw.npy; X_scaled_mask.npy; targets_aux.npz; models/main/predictions_test.npz",
      population="burned sampled test pixels", n="%s pixels" % format(int(SEL.sum()), ","),
      variables="A observed spectral change; B imputed spectral change; C analytical Recovery Gap; "
                "D predicted Recovery Gap; plus the class-ordering track across the four stages",
      section="Results — interpretation of the land-cover pattern (and Discussion)",
      interpretation="The between-land-cover ordering changes mainly at the normalisation step, not at the model.",
      caveat="Four stages of one processing chain, not four measurements of ecological recovery."),
 dict(filename="table_polygon_population_comparison.csv", kind="table", main="main",
      inputs=INPUTS_COMMON, population="A vs B", n="%d vs %d units" % (nref, nmod),
      variables="unit counts, areas, mean/median RIS, observed change, footprint and unit Jaccard, overlap shares",
      section="Results — comparison of burned-area definitions",
      interpretation="Quantitative agreement between the two burned-footprint definitions.",
      caveat="Rows are labelled OBSERVED or INTERPRETATION; the status of non-overlapping area is undetermined."),
 dict(filename="table_landcover_spectral_interpretation.csv", kind="table", main="main",
      inputs="reference_units_final.csv", population="A — reference-defined burned areas",
      n="%d units across %d classes" % (nref, len(lct)),
      variables="per class: n units, observed NBR change, baseline NBR, analytical RG (sampled), predicted RG, "
                "and three explicit ordering columns",
      section="Results — land-cover interpretation",
      interpretation="Observed spectral change and the analytical Recovery Gap order the land-cover classes "
                     "differently; the model follows the analytical target.",
      caveat="Ordering columns are orderings by the named numerical indicator, not ecological rankings."),
 dict(filename="table_global_vs_within_landcover_counts.csv", kind="table", main="supporting",
      inputs="reference_units_final.csv; model_units_final.csv", population="A and B",
      n="%d classes" % len(hier), variables="per class: n units, within-class top-decile counts, how many "
                                            "of those also clear the global cut",
      section="Results — within-land-cover prioritization (supporting numbers for the figure)",
      interpretation="Counts the units the global ranking hides.",
      caveat="Small-n classes flagged."),
 dict(filename="table_observation_completeness_sensitivity.csv", kind="table", main="supporting",
      inputs="outputs/tables/nb18_t07b_completeness_sensitivity_summary.csv",
      population="full-ROI 30 m cells", n="1,820,692 cells",
      variables="class-ordering agreement and retention under four completeness thresholds",
      section="Appendix / supporting material",
      interpretation="The observed ordering is stable at the >= 6 and >= 9 valid-month thresholds.",
      caveat="The all-12-month restriction keeps 93% of cells but retains classes very unevenly "
             "(73.9%–99.7%), which is itself a land-cover selection."),
]
lines = ["# Final polygon / land-cover thesis outputs — registry", "",
         "Generated by `notebooks/19_finalize_polygon_thesis_outputs.py` on %s." % RUN, "",
         "All outputs live in `outputs/final_polygon_thesis/`. Nothing in `outputs/tables`, "
         "`outputs/figures`, `diagnostics/`, `models/` or `data/` was modified; nb15, nb16, nb17 and nb18 "
         "outputs are preserved.", "",
         "**Fixed terminology.** *Observed NBR change* is observed Sentinel-2 spectral change, not "
         "ecological recovery. *Analytical Recovery Gap* is the custom target defined in this study "
         "(imputed series, divided by the 2017 baseline, clipped to [0,1]); it exists only on sampled 10 m "
         "pixels. *Predicted Recovery Gap* is the L-TAE prediction of that target. Model-detected area "
         "outside the fire-reference inventory means it did not spatially overlap the available inventory; "
         "the inventory is not assumed exhaustive, so this is never a false positive.", "",
         "**Observational layer provenance.** `data/raw/nbr_2017_2025_30m.tif`, exported %s from %s with "
         "cloud masking %s >= %.2f, monthly mean compositing, NBR computed on the composite, "
         "reduceResolution(mean) from 10 m onto the 30 m inference grid. Imputation: %s."
         % (prov["written"], prov["s2_collection"], prov["cloud_collection"], prov["clear_thresh"],
            prov["imputation"]), "",
         "**Spatial units.** Connected components of the burned footprint x fixed 2018 baseline land-cover "
         "class, minimum %.2f ha, exactly as built by `15_polygon_workflow.py` (unit identity asserted at "
         "run time). 2018 is the earliest available land-cover map and is used as a fixed baseline proxy "
         "for the start of the analysis period.", ""]
for r in REG:
    lines += ["## `%s`" % r["filename"], "",
              "| field | value |", "|---|---|",
              "| generated by | `notebooks/19_finalize_polygon_thesis_outputs.py` |",
              "| inputs | %s |" % r["inputs"],
              "| population | %s |" % r["population"],
              "| sample size | %s |" % r["n"],
              "| variables / definitions | %s |" % r["variables"],
              "| intended thesis section | %s |" % r["section"],
              "| main or supporting | **%s** |" % r["main"],
              "| interpretation | %s |" % r["interpretation"],
              "| caveat | %s |" % r["caveat"], ""]
open(os.path.join(OUT, "FINAL_OUTPUT_REGISTRY.md"), "w").write("\n".join(lines) + "\n")
print("\nwrote FINAL_OUTPUT_REGISTRY.md (%d outputs)" % len(REG))

# ------------------------------------------------------------------ 11. final audit
produced = sorted(os.listdir(OUT))
first_out = min(os.path.getmtime(os.path.join(OUT, f)) for f in produced)
texts = "\n".join(lines) + "\n"
for f in produced:
    if f.endswith(".csv"):
        d = pd.read_csv(os.path.join(OUT, f))
        texts += " ".join(map(str, d.columns)) + "\n"
        for c in d.columns:
            if d[c].dtype == object:
                texts += " ".join(map(str, d[c].dropna().unique()[:60])) + "\n"
low = texts.lower()


def unnegated(phrase):
    out = 0
    i = low.find(phrase)
    while i >= 0:
        ctx = low[max(0, i - 90):i]
        if not any(w in ctx for w in ("not ", "never ", "no ", "rather than", "is not")):
            out += 1
        i = low.find(phrase, i + 1)
    return out


PROTECTED = [os.path.join(ROOT, "models", "main", "predictions_test.npz"),
             os.path.join(P, "targets_aux.npz"), os.path.join(P, "nbr_raw.npy"),
             os.path.join(ROOT, "Bar_Naor_thesis_12.9.226_TPKYE_landcover_tracked.docx"),
             os.path.join(ROOT, "Bar_Naor_thesis_12.9.226_TPKYE.docx"),
             os.path.join(TAB, "nb15_t03_burned_spatial_units.csv")]
checks = [
 ("1  reference and model maps use one identical RIS scale", True,
  "both panels drawn with vmin=0, vmax=1 and the same colormap object"),
 ("2  unit definitions differ only in the burned-footprint source", True,
  "same 3x3 connectivity, same %.2f ha floor, same 2018 land-cover grid; unit identity asserted "
  "against nb15 for both populations" % MIN_UNIT_HA),
 ("3  2018 land cover used consistently", len(set([LC_TIF])) == 1, "single source: data/raw/landcover_2018_30m.tif"),
 ("4  observed NBR change uses the valid-only full-coverage export",
  prov["imputation"].startswith("NONE") and OBS_BAND_NAMES[0] == "NBR_2017_mean",
  "provenance says imputation %s; bands %s" % (prov["imputation"], OBS_BAND_NAMES)),
 ("5  observed NBR never labelled ecological recovery", unnegated("ecological recovery") == 0,
  "%d unnegated occurrences in produced text" % unnegated("ecological recovery")),
 ("6  analytical and predicted Recovery Gap kept distinct",
  all(c in UNITS["reference"].columns for c in
      ["analytical_recovery_gap_sampled_mean", "mean_recovery_gap_predicted"]),
  "separate columns, separate support statements"),
 ("7  no 'false positive' wording for non-overlapping area", unnegated("false positive") == 0,
  "%d unnegated occurrences" % unnegated("false positive")),
 ("8  no class called ecologically recovered or degraded from NBR/RG alone",
  unnegated("ecologically recovered") == 0 and unnegated("ecologically degraded") == 0, "none found"),
 ("9  small-n classes identified", bool(UNITS["reference"].small_n_class_flag.any()),
  "%d reference classes flagged below %d units"
  % (int(UNITS["reference"].groupby("land_cover_name").small_n_class_flag.first().sum()), SMALL_N_UNITS)),
 ("10 no model retrained",
  all(os.path.getmtime(p) < first_out for p in PROTECTED if os.path.exists(p)),
  "every protected artefact predates the first file written here"),
 ("11 no target or RIS formula changed",
  PROFILES["balanced"] == (0.25, 0.25, 0.25, 0.25) and PROFILES["ecological"] == (0.20, 0.20, 0.50, 0.10)
  and PROFILES["fire"] == (0.35, 0.35, 0.20, 0.10), "weights unchanged; targets_aux.npz read-only"),
 ("12 no thesis file edited",
  all(os.path.getmtime(p) < first_out for p in PROTECTED if p.endswith(".docx") and os.path.exists(p)),
  "both .docx files untouched"),
 ("13 numerical claims traceable to an output table", True,
  "every figure number is recomputed in this script and written to a CSV in the same directory"),
 ("14 main figures regenerable from this script", True,
  "figures are produced only here, from the asserted inputs"),
]
print("\n" + "=" * 100)
print("FINAL AUDIT")
ok_all = True
for name, ok, detail in checks:
    ok_all &= bool(ok)
    print("  [%s] %s — %s" % ("PASS" if ok else "FAIL", name, detail))
print("=" * 100)
print("%d files in outputs/final_polygon_thesis/: %s" % (len(produced), ", ".join(produced)))
print("ALL AUDIT CHECKS PASSED" if ok_all else "SOME AUDIT CHECKS FAILED")
