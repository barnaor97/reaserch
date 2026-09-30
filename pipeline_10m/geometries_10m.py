"""
20_unit_geometries_and_replacement_maps.py - geometry materialization + replacement Figures 10 and 11.

VISUALISATION AND EXPORT ONLY. No scientific quantity is recomputed here:
  * the burned masks, connected components, 2018 land-cover intersection and 0.36 ha floor are
    reconstructed with exactly the rules of 15_polygon_workflow.py / 19_finalize_polygon_thesis_outputs.py,
    and the resulting unit_id sets are ASSERTED identical to the final CSVs before anything is drawn;
  * every plotted attribute is read from outputs/final_polygon_thesis/*_units_final.csv;
  * nothing is written to outputs/final_polygon_thesis/ and notebooks 15-19 are untouched.

Writes  outputs/final_polygon_maps/{reference_units.gpkg, model_units.gpkg, footprint_agreement.gpkg,
        figure10_replacement_unit_indicators.{png,pdf}, figure11_replacement_unit_prioritization.{png,pdf},
        GEOMETRY_VALIDATION.md}
Run: python notebooks/20_unit_geometries_and_replacement_maps.py
"""
import json, os, sys, warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio import features
from rasterio.transform import Affine
from scipy import ndimage
from shapely.geometry import shape, GeometryCollection, MultiPolygon, Polygon
from shapely.ops import unary_union
from shapely.validation import make_valid
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

warnings.filterwarnings("ignore", category=RuntimeWarning)
ROOT = os.environ.get("THESIS_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # scripts live one level deeper than nb20/nb21
RAST = os.path.join(ROOT, "outputs", "inference_10m_candidate")
FIN = os.path.join(ROOT, "outputs", "polygon_10m_candidate")
OUT = os.path.join(ROOT, "outputs", "polygon_10m_candidate", "maps")
os.makedirs(OUT, exist_ok=True)
LC_TIF = os.path.join(RAST, "landcover_2018_10m.tif")
FIRES = os.path.join(ROOT, "data", "thesis_maps", "fig_gt_fires.shp")
ROI = os.path.join(ROOT, "data", "thesis_maps", "fig_roi_polygon.shp")
GRID_T = Affine(10.0, 0.0, 620610.0, 0.0, -10.0, 3514560.0)
H, W, CRS = 7794, 4875, "EPSG:32636"
PX_HA, THR, MIN_UNIT_HA = 0.01, 0.502, 0.36
MIN_UNIT_PX = int(round(MIN_UNIT_HA / PX_HA))
LC_NAMES = {10: "forest", 11: "orchards", 20: "shrubland", 30: "grassland", 40: "cropland",
            41: "covered agriculture", 50: "built-up", 60: "bare", 61: "fallow", 80: "water"}
LC_ORDER = [10, 11, 20, 30, 40, 41, 50, 60, 61, 80]
CLS_ORDER = [LC_NAMES[c] for c in LC_ORDER] + ["unclassified / other code"]
CLS_IDX = {c: i for i, c in enumerate(CLS_ORDER)}
RESTORATION = LinearSegmentedColormap.from_list(
    "restoration", ["#2c7bb6", "#abd9e9", "#ffffbf", "#fdae61", "#d7191c"], N=256)
_cm = plt.get_cmap("tab10")
CLASS_COLOR = {LC_NAMES[c]: _cm(i) for i, c in enumerate(LC_ORDER)}
CLASS_COLOR["unclassified / other code"] = (0.62, 0.62, 0.62, 1.0)
RUN = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
matplotlib.rcParams.update({"figure.dpi": 110, "savefig.dpi": 400, "font.size": 8.6,
                            "axes.spines.top": False, "axes.spines.right": False,
                            "pdf.fonttype": 42})
LOG = []


def note(s):
    LOG.append(s)
    print(s)


# ---------------------------------------------------------------- 1. rebuild the authoritative masks
A = {}
for role, fn in [("severity", "severity_10m.tif"), ("persistence", "persistence_10m.tif"),
                 ("recovery_gap_predicted", "recovery_gap_10m.tif"),
                 ("model_recurrence", "model_recurrence_10m.tif"), ("burned_prob", "burned_prob_10m.tif")]:
    with rasterio.open(os.path.join(RAST, fn)) as s:
        assert str(s.crs) == CRS and (s.height, s.width) == (H, W) and s.transform.almost_equals(GRID_T), fn
        A[role] = s.read(1).astype(np.float64)
with rasterio.open(LC_TIF) as s:
    assert str(s.crs) == CRS and (s.height, s.width) == (H, W) and s.transform.almost_equals(GRID_T)
    LC_GRID = s.read(1)
ELIG = np.all([np.isfinite(A[k]) for k in A], axis=0)
cls_i = np.full((H, W), CLS_IDX["unclassified / other code"], dtype=np.int16)
for code, nm in LC_NAMES.items():
    cls_i[LC_GRID == code] = CLS_IDX[nm]


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
MASK = {"reference": INSIDE & ELIG, "model": ELIG & (A["burned_prob"] > THR)}
note("masks: reference %d px (%.0f ha) | model %d px (%.0f ha)"
     % (MASK["reference"].sum(), MASK["reference"].sum() * PX_HA,
        MASK["model"].sum(), MASK["model"].sum() * PX_HA))

STRUCT = np.ones((3, 3), dtype=int)


def components(mask, min_px=0):
    lab, k = ndimage.label(mask, structure=STRUCT)
    if min_px > 1 and k:
        sizes = np.bincount(lab.ravel())
        drop = np.where(sizes < min_px)[0]
        drop = drop[drop > 0]
        if len(drop):
            lab[np.isin(lab, drop)] = 0
        lab, k = ndimage.label(lab > 0, structure=STRUCT)
    return lab, k


# ---------------------------------------------------------------- 2. rebuild units, assert identity
CSV = {"reference": pd.read_csv(os.path.join(FIN, "reference_units_final.csv")),
       "model": pd.read_csv(os.path.join(FIN, "model_units_final.csv"))}
PREF = {"reference": "doc", "model": "mod"}
UIDX, UNITS = {}, {}
for pop, M in MASK.items():
    lab, _ = components(M, MIN_UNIT_PX)
    sel = lab > 0
    d = pd.DataFrame({"zone": lab[sel], "cls": cls_i[sel]})
    u = d.groupby(["zone", "cls"]).size().rename("pixel_count").reset_index()
    u = u[u.pixel_count >= MIN_UNIT_PX].reset_index(drop=True)
    u["unit_id"] = ["%s_%d_%d" % (PREF[pop], z, c) for z, c in zip(u.zone, u.cls)]
    want = set(CSV[pop].unit_id)
    assert set(u.unit_id) == want, ("unit identity differs for %s: rebuilt %d vs final %d"
                                    % (pop, len(u), len(want)))
    merged = u.merge(CSV[pop], on="unit_id", how="inner", suffixes=("", "_final"))
    assert (merged.pixel_count == merged.pixel_count_final).all(), "pixel counts differ from the final table"
    idx = np.full((H, W), -1, dtype=np.int32)
    key = {(z, c): i for i, (z, c) in enumerate(zip(u.zone, u.cls))}
    zz, cc = lab[sel], cls_i[sel]
    idx[sel] = np.array([key.get((z, c), -1) for z, c in zip(zz, cc)], dtype=np.int32)
    UIDX[pop] = idx
    UNITS[pop] = CSV[pop].set_index("unit_id").loc[u.unit_id].reset_index()
    note("%-9s rebuilt %d units, identity asserted against the final table; pixel counts match"
         % (pop, len(u)))

# ---------------------------------------------------------------- 3. materialise geometry
def vectorize(pop):
    idx, u = UIDX[pop], UNITS[pop]
    parts = {}
    for geom, val in features.shapes(idx, mask=idx >= 0, transform=GRID_T, connectivity=8):
        parts.setdefault(int(val), []).append(shape(geom))
    polys = [unary_union(parts[i]) if i in parts else None for i in range(len(u))]
    gdf = gpd.GeoDataFrame(u.copy(), geometry=polys, crs=CRS)
    # Vectorising an 8-connected raster mask produces pinch points where cells meet only at a corner,
    # which OGR reports as ring self-intersections. make_valid repairs the topology; the assertion
    # below confirms it changes no area, so the geometry still represents exactly the same cells.
    before = float(gdf.geometry.area.sum())
    gdf["geometry"] = gdf.geometry.apply(make_valid)
    assert abs(float(gdf.geometry.area.sum()) - before) < 1e-6, "make_valid altered the area"
    assert gdf.geometry.is_valid.all(), "geometry still invalid after repair"
    gdf["geometry_area_ha"] = gdf.geometry.area / 10000.0
    return gdf


GDF = {}
for pop in ("reference", "model"):
    g = vectorize(pop)
    ok = np.allclose(g.geometry_area_ha, g.pixel_count * PX_HA, atol=1e-6)
    note("%-9s geometry: %d features | area from geometry %.2f ha vs pixel count %.2f ha | exact match: %s"
         % (pop, len(g), g.geometry_area_ha.sum(), g.pixel_count.sum() * PX_HA, ok))
    assert ok, "vectorised area does not equal the pixel-count area"
    GDF[pop] = g
    g.to_file(os.path.join(OUT, "%s_units.gpkg" % pop), layer="%s_units" % pop, driver="GPKG")

# footprint agreement: the RAW masks, which are what the published agreement statistics describe
REF, DET = MASK["reference"], MASK["model"]
CAT = np.zeros((H, W), dtype=np.int16)
CAT[REF & ~DET] = 1
CAT[REF & DET] = 2
CAT[DET & ~REF] = 3
CATNAME = {1: "reference only", 2: "overlap", 3: "model only"}
rows = []
for v, nm in CATNAME.items():
    parts = [shape(g) for g, val in features.shapes((CAT == v).astype(np.uint8), mask=(CAT == v),
                                                    transform=GRID_T, connectivity=8)]
    rows.append(dict(category=nm, n_pixels=int((CAT == v).sum()), area_ha=float((CAT == v).sum() * PX_HA),
                     geometry=unary_union(parts) if parts else None))
AGREE = gpd.GeoDataFrame(rows, crs=CRS)
AGREE.to_file(os.path.join(OUT, "footprint_agreement.gpkg"), layer="footprint_agreement", driver="GPKG")

jac = float((REF & DET).sum() / (REF | DET).sum())
pct_ref_det = 100.0 * float((REF & DET).sum() / REF.sum())
pct_mod_out = 100.0 * float((DET & ~REF).sum() / DET.sum())
note("agreement: Jaccard %.4f | reference also detected %.1f%% | model outside inventory %.1f%%"
     % (jac, pct_ref_det, pct_mod_out))

# ---------------------------------------------------------------- 4. shared cartography
roi = gpd.read_file(ROI).to_crs(CRS)
EXT = [GRID_T.c, GRID_T.c + W * GRID_T.a, GRID_T.f + H * GRID_T.e, GRID_T.f]
BG = np.where(ELIG, 1.0, np.nan)


def paint(pop, col):
    """Paint a per-unit attribute onto the grid; cells outside units stay NaN."""
    u, idx = UNITS[pop], UIDX[pop]
    vals = np.append(u[col].to_numpy(dtype=float), np.nan)
    return np.where(idx >= 0, vals[np.clip(idx, 0, len(u))], np.nan)


def frame(ax, title):
    ax.imshow(BG, extent=EXT, cmap=ListedColormap(["#ededed"]), interpolation="nearest")
    roi.boundary.plot(ax=ax, color="k", linewidth=0.6)
    ax.set_xlim(EXT[0], EXT[1]); ax.set_ylim(EXT[2], EXT[3])
    ax.set_xticks([]); ax.set_yticks([]); ax.set_frame_on(False)
    ax.set_title(title, fontsize=9.0, linespacing=1.3)


def scalebar(ax, km=10):
    x0, y0 = EXT[0] + 1500, EXT[2] + 3000
    ax.plot([x0, x0 + km * 1000], [y0, y0], color="k", lw=2.2, solid_capstyle="butt")
    ax.text(x0 + km * 500, y0 + 1200, "%d km" % km, ha="center", fontsize=6.8)
    ax.annotate("N", xy=(EXT[1] - 3000, EXT[3] - 4000), xytext=(EXT[1] - 3000, EXT[3] - 12000),
                ha="center", fontsize=8.5, arrowprops=dict(arrowstyle="-|>", color="k", lw=1.0))


# ---------------------------------------------------------------- 5. FIGURE 10 replacement
IND = [("A", "mean_severity_model", "Burn severity\n(NBR-derived spectral severity indicator)"),
       ("B", "mean_persistence_model", "Persistence"),
       ("C", "mean_recovery_gap_predicted", "Recovery gap\n(analytical fixed-calendar spectral indicator)"),
       ("D", "mean_recurrence_model", "Recurrence")]
u_ref = UNITS["reference"]
note("\nFigure 10 attribute ranges over the %d reference-defined units:" % len(u_ref))
for lab, col, _ in IND:
    note("   %-28s min %.4f  median %.4f  max %.4f"
         % (col, u_ref[col].min(), u_ref[col].median(), u_ref[col].max()))

fig, axes = plt.subplots(1, 4, figsize=(16.4, 7.4), layout="constrained")
for ax, (lab, col, title) in zip(axes, IND):
    frame(ax, "(%s) %s" % (lab, title))
    im = ax.imshow(paint("reference", col), extent=EXT, cmap=RESTORATION, vmin=0, vmax=1,
                   interpolation="nearest")
scalebar(axes[0])
cb = fig.colorbar(im, ax=list(axes), orientation="horizontal", fraction=0.030, pad=0.02, shrink=0.45)
cb.set_label("Unit mean indicator value (all four indicators are defined on the interval 0 to 1)", fontsize=8.4)
fig.suptitle("Model-estimated disturbance and post-fire spectral indicators over reference-defined burned "
             "spatial units\n%s units, %s ha; each unit is one connected component of the reference-documented "
             "burned footprint intersected with the fixed 2018 land-cover baseline"
             % (format(len(u_ref), ","), format(int(round(u_ref.area_ha.sum())), ",")), fontsize=10.4)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, "figure10_replacement_unit_indicators." + ext),
                bbox_inches="tight", facecolor="white")
plt.close(fig)
note("wrote figure10_replacement_unit_indicators.{png,pdf}")

# ---------------------------------------------------------------- 6. FIGURE 11 replacement
# the "within-class top decile but below the global cut" selection, reproducing nb19 exactly
u = UNITS["reference"].copy()
gcut = float(u.mean_RIS_balanced.quantile(0.90))
hidden, per_class = [], {}
for c, s in u.groupby("land_cover_name"):
    wcut = float(s.mean_RIS_balanced.quantile(0.90)) if len(s) >= 10 else float(s.mean_RIS_balanced.max())
    top = s[s.mean_RIS_balanced >= wcut]
    hid = top[top.mean_RIS_balanced < gcut]
    per_class[c] = (len(hid), len(top))
    hidden += list(hid.unit_id)
MAIN = ["forest", "orchards", "shrubland", "grassland", "cropland", "fallow"]
n_hidden_main = sum(per_class.get(c, (0, 0))[0] for c in MAIN)
note("\nwithin-LC top decile but below the global cut: %d units across the six main vegetated classes"
     % n_hidden_main)
for c in MAIN:
    if c in per_class:
        note("   %-10s %d of %d" % (c, per_class[c][0], per_class[c][1]))
HID = GDF["reference"][GDF["reference"].unit_id.isin(hidden)]

fig, axes = plt.subplots(1, 4, figsize=(16.4, 7.8), layout="constrained")
# A - reference units by 2018 land cover
frame(axes[0], "(A) Reference-defined burned units\n%s units, %s ha, coloured by 2018 land cover"
      % (format(len(u_ref), ","), format(int(round(u_ref.area_ha.sum())), ",")))
cls_paint = np.full((H, W), np.nan)
present = [c for c in CLS_ORDER if (u_ref.land_cover_name == c).any()]
code_of = {c: i for i, c in enumerate(present)}
vals = np.append(u_ref.land_cover_name.map(code_of).to_numpy(dtype=float), np.nan)
cls_paint = np.where(UIDX["reference"] >= 0, vals[np.clip(UIDX["reference"], 0, len(u_ref))], np.nan)
axes[0].imshow(cls_paint, extent=EXT, cmap=ListedColormap([CLASS_COLOR[c] for c in present]),
               vmin=-0.5, vmax=len(present) - 0.5, interpolation="nearest")
axes[0].legend(handles=[Patch(facecolor=CLASS_COLOR[c], label="%s (%d)" % (c, int((u_ref.land_cover_name == c).sum())))
                        for c in present], fontsize=6.0, loc="lower right", framealpha=0.94, borderpad=0.4)
scalebar(axes[0])
# B - model units
u_mod = UNITS["model"]
frame(axes[1], "(B) Model-detected burned units\n%s units, %s ha, burned probability > %.3f"
      % (format(len(u_mod), ","), format(int(round(u_mod.area_ha.sum())), ","), THR))
vals = np.append(u_mod.land_cover_name.map(code_of).to_numpy(dtype=float), np.nan)
axes[1].imshow(np.where(UIDX["model"] >= 0, vals[np.clip(UIDX["model"], 0, len(u_mod))], np.nan),
               extent=EXT, cmap=ListedColormap([CLASS_COLOR[c] for c in present]),
               vmin=-0.5, vmax=len(present) - 0.5, interpolation="nearest")
# C - agreement
frame(axes[2], "(C) Reference and model footprints compared\nJaccard %.4f" % jac)
AGCOL = {1: "#2c7bb6", 2: "#6a51a3", 3: "#d7191c"}
axes[2].imshow(np.where(CAT > 0, CAT, np.nan), extent=EXT,
               cmap=ListedColormap([AGCOL[1], AGCOL[2], AGCOL[3]]), vmin=0.5, vmax=3.5,
               interpolation="nearest")
axes[2].legend(handles=[Patch(facecolor=AGCOL[1], label="reference only (%.0f ha)" % ((CAT == 1).sum() * PX_HA)),
                        Patch(facecolor=AGCOL[2], label="overlap (%.0f ha)" % ((CAT == 2).sum() * PX_HA)),
                        Patch(facecolor=AGCOL[3], label="model only (%.0f ha)" % ((CAT == 3).sum() * PX_HA))],
               fontsize=6.2, loc="lower right", framealpha=0.94, borderpad=0.4)
# D - within-land-cover prioritization
frame(axes[3], "(D) Within-land-cover prioritization\nunit RIS percentile inside its own 2018 class")
im = axes[3].imshow(paint("reference", "within_landcover_RIS_percentile"), extent=EXT,
                    cmap=RESTORATION, vmin=0, vmax=100, interpolation="nearest")
if len(HID):
    HID.boundary.plot(ax=axes[3], color="#111111", linewidth=0.9)
axes[3].legend(handles=[Line2D([], [], color="#111111", lw=1.2,
                               label="top decile within its class,\nbelow the global top decile (%d units)" % n_hidden_main)],
               fontsize=6.2, loc="lower right", framealpha=0.94, borderpad=0.4)
cb = fig.colorbar(im, ax=[axes[3]], orientation="horizontal", fraction=0.040, pad=0.02, shrink=0.85)
cb.set_label("within-class RIS percentile", fontsize=7.6)
fig.suptitle("Burned spatial units: reference-defined and model-detected footprints, their agreement, and "
             "land-cover-aware prioritization", fontsize=10.6)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, "figure11_replacement_unit_prioritization." + ext),
                bbox_inches="tight", facecolor="white")
plt.close(fig)
note("wrote figure11_replacement_unit_prioritization.{png,pdf}")

open(os.path.join(OUT, "GEOMETRY_VALIDATION.md"), "w").write(
    "# Geometry materialisation and replacement maps\n\nGenerated %s by "
    "`notebooks/20_unit_geometries_and_replacement_maps.py`.\n\n" % RUN + "\n".join("- " + l for l in LOG) + "\n")
print("\nfiles in outputs/final_polygon_maps:", ", ".join(sorted(os.listdir(OUT))))
