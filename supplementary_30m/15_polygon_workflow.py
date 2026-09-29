"""
15_polygon_workflow.py — the consolidated burned-spatial-unit (polygon-level) workflow.

Supersedes the exploratory 11_polygon_landcover_analysis.py and 12_branchA_polygon_analysis.py,
both of which used definitions the repository audit rejected:
  * 11_ labelled units with `lc_dominant`, the modal class over 2018-2025, which mixes
    post-disturbance land-cover change into the variable used to interpret recovery;
  * 12_ assigned pixels to individual dated fire-event polygons, but the inventory is not a
    partition of space - repeated burning makes the polygons overlap.
Here land cover is the fixed 2018 baseline (the product starts in 2018; used as the proxy for
2017, consistent with the published pixel-level table), the fire polygons are dissolved into one
non-overlapping documented-burned mask, and repeated burning is carried by the recurrence
indicator rather than by unit membership.

Post-hoc only. No retraining, no change to any target definition, indicator formula or model
output. The model stays pixel-based; only the unit of INTERPRETATION becomes the burned spatial
unit:

    unit = connected component of a burned-area mask  x  2018 baseline land-cover class

built identically for both burned-area definitions:
    (a) documented  - fire-reference polygons 2017-2025, dissolved and rasterised
    (b) model       - burned_prob_30m.tif > 0.502

Shared between the two sides: the 30 m grid, the 2018 land-cover definition, the indicator
definitions, the aggregation statistic, the RIS formulation, the physical minimum-area rule, the
ranking logic and the map conventions. Morphological cleaning is NOT applied symmetrically by
default: it is evaluated for the model mask only (Section 4) and adopted only if the evidence
supports it. The documented footprint is preserved.

Reads   outputs/inference_full_roi/*_30m.tif, data/raw/landcover_2018_30m.tif,
        data/thesis_maps/fig_gt_fires.shp, data/thesis_maps/fig_roi_polygon.shp
Writes  outputs/tables/nb15_*.{csv,md}, outputs/figures/nb15_*.png

Run: python notebooks/15_polygon_workflow.py
"""
import hashlib
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
from scipy import ndimage
from scipy import stats
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
from shapely.ops import unary_union
from shapely.validation import make_valid
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------- §1 setup
ROOT = os.environ.get("THESIS_ROOT")
if not ROOT:
    here = os.path.abspath(os.getcwd())
    ROOT = os.path.dirname(here) if os.path.basename(here) == "notebooks" else here
ROOT = os.path.abspath(os.path.expanduser(ROOT))
RAST = os.path.join(ROOT, "outputs", "inference_full_roi")
FIRES_SHP = os.path.join(ROOT, "data", "thesis_maps", "fig_gt_fires.shp")
ROI_SHP = os.path.join(ROOT, "data", "thesis_maps", "fig_roi_polygon.shp")
LC_PATH = os.path.join(ROOT, "data", "raw", "landcover_2018_30m.tif")
OUT_ROOT = os.path.abspath(os.environ.get("NB15_OUT_ROOT", ROOT))
TAB = os.path.join(OUT_ROOT, "outputs", "tables")
FIG = os.path.join(OUT_ROOT, "outputs", "figures")
os.makedirs(TAB, exist_ok=True)
os.makedirs(FIG, exist_ok=True)
RUN_STAMP = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
PREFIX = "nb15_"

GRID_T = Affine(30.0, 0.0, 620610.0, 0.0, -30.0, 3514560.0)
H, W, CRS = 2598, 1625, "EPSG:32636"
PX_HA = 0.09
THR = 0.502
LC_NAMES = {10: "forest", 11: "orchards", 20: "shrubland", 30: "grassland", 40: "cropland",
            41: "covered agriculture", 50: "built-up", 60: "bare", 61: "fallow", 80: "water"}
LC_ORDER = [10, 11, 20, 30, 40, 41, 50, 60, 61, 80]
OTHER = "unclassified / other code"
CLASS_NAMES = [LC_NAMES[c] for c in LC_ORDER] + [OTHER]
PROFILES = {"ecological": (0.20, 0.20, 0.50, 0.10), "fire": (0.35, 0.35, 0.20, 0.10),
            "balanced": (0.25, 0.25, 0.25, 0.25)}
IND = ["severity", "persistence", "recovery_gap", "model_recurrence"]
QUANT = IND + ["RIS_" + p for p in PROFILES]

MIN_UNIT_HA = 0.36          # physical floor for averaging a unit (4 px at 30 m)
RANK_MIN_HA = 1.0           # physical floor for a unit to enter a ranked list
RANK_SENS_HA = [0.36, 1.0, 5.0]
MIN_CLASS_PX = 100          # 9 ha: a class below this is flagged small-n
TOP_N = 10
RESTORATION_CMAP = LinearSegmentedColormap.from_list(
    "restoration", ["#2c7bb6", "#abd9e9", "#ffffbf", "#fdae61", "#d7191c"], N=256)
matplotlib.rcParams.update({"figure.dpi": 110, "savefig.dpi": 300, "font.size": 9,
                            "axes.spines.top": False, "axes.spines.right": False})
REGISTRY = []


def fingerprint(path, edge=16 * 1024 * 1024):
    h, size = hashlib.sha256(), os.path.getsize(path)
    with open(path, "rb") as fh:
        h.update(fh.read(min(edge, size)))
        if size > edge:
            fh.seek(max(0, size - edge))
            h.update(fh.read(edge))
    return h.hexdigest()[:16]


def md_table(df, fmt="{:.4f}"):
    def cell(v):
        if isinstance(v, (bool, np.bool_)):
            return str(bool(v))
        if isinstance(v, (float, np.floating)):
            return "" if not np.isfinite(v) else fmt.format(v)
        return str(v)
    cols = list(df.columns)
    out = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(cell(r[c]) for c in cols) + " |")
    return "\n".join(out)


def save_table(stem, df, caption, sources, section, note=""):
    stem = PREFIX + stem
    df.to_csv(os.path.join(TAB, stem + ".csv"), index=False)
    head = ["<!-- generated by notebooks/15_polygon_workflow.py §%s on %s -->" % (section, RUN_STAMP), "",
            caption, "", "**Sources (column → raster / array).**", ""]
    head += ["- `%s` — %s" % (k, v) for k, v in sources.items()]
    body = [""] + [md_table(df)]
    open(os.path.join(TAB, stem + ".md"), "w").write("\n".join(head + body) + "\n")
    REGISTRY.append(dict(section=section, kind="table", stem=stem, rows=len(df), caption=caption))
    print("\n[table] %s  (%d rows)" % (stem, len(df)))
    if note:
        print("        " + note)
    return df


def save_figure(fig, stem, caption, sources, section, note=""):
    stem = PREFIX + stem
    path = os.path.join(FIG, stem + ".png")
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    head = ["<!-- generated by notebooks/15_polygon_workflow.py §%s on %s -->" % (section, RUN_STAMP), "",
            caption, "", "**Sources.**", ""]
    head += ["- `%s` — %s" % (k, v) for k, v in sources.items()]
    open(os.path.join(FIG, stem + ".md"), "w").write("\n".join(head) + "\n")
    REGISTRY.append(dict(section=section, kind="figure", stem=stem, rows=np.nan, caption=caption))
    print("\n[figure] %s" % stem)
    if note:
        print("         " + note)


print("=" * 100)
print("15_polygon_workflow.py — burned spatial units x 2018 land cover")
print("root:", ROOT, "| run (UTC):", RUN_STAMP)
print("python", sys.version.split()[0], "|", platform.platform())
print("numpy", np.__version__, "| pandas", pd.__version__, "| rasterio", rasterio.__version__,
      "| geopandas", gpd.__version__, "| scipy", __import__("scipy").__version__)
print("=" * 100)

# ------------------------------------------------- §2 load and verify alignment
LAYERS = {"severity": "severity_30m.tif", "persistence": "persistence_30m.tif",
          "recovery_gap": "recovery_gap_30m.tif", "model_recurrence": "model_recurrence_30m.tif",
          "burned_prob": "burned_prob_30m.tif"}
A, align_rows = {}, []


def grid_row(name, path, arr, shape, crs, tr, nodata, dtype):
    left, top = tr.c, tr.f
    right, bottom = tr.c + shape[1] * tr.a, tr.f + shape[0] * tr.e
    return dict(input=os.path.relpath(path, ROOT), dtype=str(dtype), crs=str(crs),
                width=shape[1], height=shape[0], pixel_size_x=tr.a, pixel_size_y=tr.e,
                origin_x=left, origin_y=top, extent_right=right, extent_bottom=bottom,
                transform_matches_grid=bool(tr.almost_equals(GRID_T)),
                shape_matches_grid=bool(shape == (H, W)), crs_matches_grid=bool(str(crs) == CRS),
                nodata=("None" if nodata is None else str(nodata)),
                finite_px=(int(np.isfinite(arr).sum()) if arr is not None and arr.dtype.kind == "f" else ""),
                sha256_head=fingerprint(path))


for role, fn in LAYERS.items():
    path = os.path.join(RAST, fn)
    with rasterio.open(path) as s:
        a = s.read(1).astype(np.float64)
        align_rows.append(grid_row(role, path, a, (s.height, s.width), s.crs, s.transform, s.nodata, s.dtypes[0]))
    A[role] = a
with rasterio.open(LC_PATH) as s:
    LC_CODE = s.read(1)
    align_rows.append(grid_row("land_cover_2018", LC_PATH, None, (s.height, s.width), s.crs,
                               s.transform, s.nodata, s.dtypes[0]))
align = pd.DataFrame(align_rows)
ok = align[["transform_matches_grid", "shape_matches_grid", "crs_matches_grid"]].all().all()
save_table("t00_grid_alignment", align,
           "Exact geospatial metadata of every raster entering the polygon workflow, checked against the "
           "inference grid (EPSG:32636, 1625 x 2598, 30 m, origin 620610 / 3514560). The 2018 land-cover "
           "layer was exported from Earth Engine with reduceResolution(mode) - a categorical majority rule, "
           "the only aggregation that keeps a class label meaningful when going from 10 m to 30 m - and "
           "reprojected onto this exact crsTransform. 2018 is used as the baseline because the land-cover "
           "product begins in 2018; it is the proxy for 2017.",
           {"all columns": "rasterio dataset metadata and pixel values"}, section="2",
           note="alignment verified: %s" % ok)
assert ok, "ALIGNMENT FAILED — the polygon analysis must not proceed"

SEV, PER, RG, MREC, BP = (A[k] for k in ("severity", "persistence", "recovery_gap", "model_recurrence", "burned_prob"))
ELIG = np.all([np.isfinite(A[k]) for k in LAYERS], axis=0)
RIS = {p: np.where(ELIG, ws * SEV + wp * PER + wr * RG + wc * MREC, np.nan)
       for p, (ws, wp, wr, wc) in PROFILES.items()}
Q = {"severity": SEV, "persistence": PER, "recovery_gap": RG, "model_recurrence": MREC}
Q.update({"RIS_" + p: RIS[p] for p in PROFILES})
CLS = np.full((H, W), len(CLASS_NAMES) - 1, dtype=np.int16)
for i, c in enumerate(LC_ORDER):
    CLS[LC_CODE == c] = i
print("\neligible pixels:", int(ELIG.sum()))

# ------------------------------------------------- §3 the two burned-area masks
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


fires = gpd.read_file(FIRES_SHP)
fires_win = fires[fires.fire_year >= 2017].to_crs(CRS)
geoms = [g for g in (polygonal(g) for g in fires_win.geometry) if g is not None and not g.is_empty]
INSIDE = features.rasterize(((g, 1) for g in geoms), out_shape=(H, W), transform=GRID_T,
                            fill=0, all_touched=False, dtype="uint8").astype(bool)
REF = INSIDE & ELIG
DET_RAW = ELIG & (BP > THR)
print("documented-burned: %d px | model-detected (raw): %d px" % (REF.sum(), DET_RAW.sum()))

# ------------------------------------------------- §4 model-mask cleaning, evaluated
STRUCT = np.ones((3, 3), dtype=int)


def components(mask, min_px=0):
    lab, n = ndimage.label(mask, structure=STRUCT)
    if min_px > 1 and n:
        sizes = np.bincount(lab.ravel())
        drop = np.where(sizes < min_px)[0]
        drop = drop[drop > 0]
        if len(drop):
            lab[np.isin(lab, drop)] = 0
        lab, n = ndimage.label(lab > 0, structure=STRUCT)
    return lab, n


def class_means(mask):
    out = {}
    for i, cname in enumerate(CLASS_NAMES):
        m = mask & (CLS == i)
        if m.sum() >= MIN_CLASS_PX:
            out[cname] = float(RIS["balanced"][m].mean())
    return out


base_cm = class_means(DET_RAW)
rows = []
variants = [("no cleaning (as published)", DET_RAW, "raw threshold mask")]
for ha in [0.36, 0.9, 1.8, 3.6, 9.0]:
    lab, _ = components(DET_RAW, int(round(ha / PX_HA)))
    variants.append(("drop components < %.2f ha" % ha, lab > 0, "minimum-area rule only"))
op = ndimage.binary_opening(DET_RAW, structure=STRUCT)
variants.append(("morphological opening 3x3", op, "erosion then dilation"))
lab_op, _ = components(op, int(round(MIN_UNIT_HA / PX_HA)))
variants.append(("opening 3x3 + drop < %.2f ha" % MIN_UNIT_HA, lab_op > 0, "both"))
for label, m, detail in variants:
    lab, n = components(m)
    sizes = np.bincount(lab.ravel())[1:] if n else np.array([0])
    cm = class_means(m)
    shared = [c for c in base_cm if c in cm]
    rho = (stats.spearmanr([base_cm[c] for c in shared], [cm[c] for c in shared]).statistic
           if len(shared) > 2 else np.nan)
    rows.append(dict(variant=label, detail=detail, n_px=int(m.sum()), area_ha=m.sum() * PX_HA,
                     pct_of_raw_area=100.0 * m.sum() / DET_RAW.sum(),
                     n_components=int(n), median_component_ha=float(np.median(sizes)) * PX_HA,
                     pct_components_single_px=100.0 * float((sizes == 1).mean()) if n else np.nan,
                     jaccard_vs_documented=float((m & REF).sum() / (m | REF).sum()),
                     mean_RIS_balanced=float(RIS["balanced"][m].mean()),
                     n_classes_kept=len(cm), spearman_class_order_vs_raw=float(rho)))
clean = pd.DataFrame(rows)
save_table("t02_model_mask_cleaning_sensitivity", clean,
           "Sensitivity of the model-detected mask to cleaning. The documented mask is never cleaned: it is "
           "the delineated footprint. For the model mask, which is a pixel-level prediction and is therefore "
           "fragmented, each candidate rule is quantified before any is adopted. `spearman_class_order_vs_raw` "
           "compares the land-cover ordering of mean Balanced RIS against the uncleaned mask, so a value near "
           "1 means the cleaning does not change the substantive result.",
           {"mask": "burned_prob_30m.tif > %.3f" % THR,
            "components": "8-connected (3x3 structure)",
            "RIS_balanced": "0.25 x (severity + persistence + recovery_gap + model_recurrence)"}, section="4")

# The decision, taken from the table above rather than assumed in advance.
#
# The model mask IS fragmented, as expected: 57.7% of its components are single pixels. But every
# cleaning rule is expensive - the mildest minimum-area rule discards 16% of the detected area and
# 3x3 opening discards 55% - and none of them leaves the land-cover ordering untouched (the
# surviving class count falls from 9 to 6, so the rank correlation is itself partly an artefact of
# a shrinking class set). Cleaning the mask is also redundant: the physical minimum-area rule is
# already applied when pixels are grouped into units (Section 5), so fragments below the floor never
# become units. Applying it twice would remove area from the maps without changing any result.
#
# Therefore NO cleaning is applied to either mask. There is no asymmetric preprocessing to justify:
# both sides get the same minimum-area rule at the same (unit) stage. The fragmentation is reported
# rather than removed.
raw = clean.iloc[0]
ADOPTED = "no cleaning (as published)"
DET = DET_RAW
_mild = clean[clean.variant == "drop components < %.2f ha" % MIN_UNIT_HA].iloc[0]
_open = clean[clean.variant == "morphological opening 3x3"].iloc[0]
decision = pd.DataFrame([
    dict(question="Is the model mask actually fragmented?",
         answer="yes", evidence="%.1f%% of its %d components are single pixels; median component %.2f ha"
         % (raw.pct_components_single_px, raw.n_components, raw.median_component_ha)),
    dict(question="Is the documented mask fragmented in the same way?", answer="no",
         evidence="it is a delineated footprint, dissolved from %d polygons; it is never cleaned" % len(geoms)),
    dict(question="What would the mildest minimum-area cleaning cost?", answer="%.1f%% of detected area"
         % (100 - _mild.pct_of_raw_area),
         evidence="%d px -> %d px; class ordering rho %.3f; classes kept %d -> %d"
         % (raw.n_px, _mild.n_px, _mild.spearman_class_order_vs_raw, raw.n_classes_kept, _mild.n_classes_kept)),
    dict(question="What would 3x3 morphological opening cost?", answer="%.1f%% of detected area"
         % (100 - _open.pct_of_raw_area),
         evidence="%d px -> %d px, i.e. more than half the area the model flags as burned" % (raw.n_px, _open.n_px)),
    dict(question="Is cleaning necessary for the analysis?", answer="no",
         evidence="the physical minimum-area rule (%.2f ha) is applied when pixels are grouped into units in "
                  "Section 5, so sub-threshold fragments never become units either way" % MIN_UNIT_HA),
    dict(question="Decision", answer=ADOPTED,
         evidence="no morphological operator on either mask; the same minimum-area rule at the same stage on "
                  "both sides; the model mask's fragmentation is reported, not removed"),
])
save_table("t02b_model_mask_cleaning_decision", decision,
           "The cleaning decision and the evidence behind it, recorded because the two burned-area sources "
           "differ in kind and any difference in their preprocessing has to be explicit. The outcome is that "
           "there is no difference to justify: neither mask is cleaned.",
           {"all rows": "computed from nb15_t02_model_mask_cleaning_sensitivity"}, section="4")
print("\ndecision: %s — no morphological operator on either mask" % ADOPTED)
print("model mask stays at %d px (%.0f ha); its %.1f%% single-pixel components are excluded at the unit stage"
      % (DET.sum(), DET.sum() * PX_HA, raw.pct_components_single_px))

# ------------------------------------------------- §5 burned spatial units
MIN_UNIT_PX = int(round(MIN_UNIT_HA / PX_HA))
POPS = [("documented", REF), ("model", DET)]
UNITS, ZONES = {}, {}
ys, xs = np.mgrid[0:H, 0:W]
EAST = GRID_T.c + (xs + 0.5) * GRID_T.a
NORTH = GRID_T.f + (ys + 0.5) * GRID_T.e
# The minimum-area rule is applied at exactly two points, both listed in the accounting table below
# and both identical for the two populations:
#   step 1  connected components smaller than MIN_UNIT_PX are dropped  (components(M, MIN_UNIT_PX))
#   step 2  zone x class units smaller than MIN_UNIT_PX are dropped    (unit.n_px >= MIN_UNIT_PX)
# Step 2 is needed as well as step 1 because a surviving component can still be split by land cover
# into slivers below the floor. A single-cell component is removed at step 1 and therefore never
# reaches the unit table, the ranking, or the maps.
acct = []
for pop, M in POPS:
    lab_all, n_all = components(M)                      # every component, no floor
    sizes_all = np.bincount(lab_all.ravel())[1:] if n_all else np.array([], dtype=int)
    small = sizes_all[sizes_all < MIN_UNIT_PX]
    lab, n = components(M, MIN_UNIT_PX)                 # step 1
    ZONES[pop] = lab
    sel = lab > 0
    df = pd.DataFrame({"zone": lab[sel], "cls": CLS[sel], "east": EAST[sel], "north": NORTH[sel]})
    for q in QUANT:
        df[q] = Q[q][sel]
    g = df.groupby(["zone", "cls"])
    unit = g.agg(n_px=("zone", "size"), east=("east", "mean"), north=("north", "mean"),
                 **{q: (q, "mean") for q in QUANT}).reset_index()
    unit["area_ha"] = unit.n_px * PX_HA
    unit["land_cover"] = [CLASS_NAMES[c] for c in unit.cls]
    unit["population"] = pop
    unit["zone_area_ha"] = unit.zone.map(df.groupby("zone").size() * PX_HA)
    cand_n, cand_px = len(unit), int(unit.n_px.sum())
    unit = unit[unit.n_px >= MIN_UNIT_PX].reset_index(drop=True)   # step 2
    unit["unit_id"] = ["%s_%d_%d" % (pop[:3], z, c) for z, c in zip(unit.zone, unit.cls)]
    UNITS[pop] = unit
    final_px = int(unit.n_px.sum())
    acct += [
        dict(population=pop, step="0. burned mask (no minimum-area rule)", n_items=int(n_all),
             item="connected components", n_px=int(M.sum()), area_ha=M.sum() * PX_HA,
             pct_of_mask_area=100.0),
        dict(population=pop, step="1a. components smaller than the floor, REMOVED", n_items=int(len(small)),
             item="connected components", n_px=int(small.sum()), area_ha=small.sum() * PX_HA,
             pct_of_mask_area=100.0 * small.sum() / M.sum()),
        dict(population=pop, step="1b. components kept", n_items=int(n), item="connected components",
             n_px=int(sel.sum()), area_ha=sel.sum() * PX_HA, pct_of_mask_area=100.0 * sel.sum() / M.sum()),
        dict(population=pop, step="2a. zone x class candidate units", n_items=cand_n,
             item="zone x land-cover units", n_px=cand_px, area_ha=cand_px * PX_HA,
             pct_of_mask_area=100.0 * cand_px / M.sum()),
        dict(population=pop, step="2b. candidate units below the floor, REMOVED",
             n_items=cand_n - len(unit), item="zone x land-cover units", n_px=cand_px - final_px,
             area_ha=(cand_px - final_px) * PX_HA, pct_of_mask_area=100.0 * (cand_px - final_px) / M.sum()),
        dict(population=pop, step="3. FINAL units analysed", n_items=len(unit),
             item="zone x land-cover units", n_px=final_px, area_ha=final_px * PX_HA,
             pct_of_mask_area=100.0 * final_px / M.sum()),
    ]
    print("\n%s: %d components total -> %d kept (>= %.2f ha) -> %d candidate units -> %d FINAL units; "
          "area %.0f ha -> %.0f ha (%.1f%% removed)"
          % (pop, n_all, n, MIN_UNIT_HA, cand_n, len(unit), M.sum() * PX_HA, final_px * PX_HA,
             100.0 * (M.sum() - final_px) / M.sum()))

save_table("t01_unit_formation_accounting", pd.DataFrame(acct),
           "Exactly where the physical minimum-area rule bites, and what it costs, for each burned-area "
           "definition. The floor is %.2f ha = %d cells of 30 m. It is applied twice, identically on both "
           "sides: once to connected components (step 1) and once to the zone x land-cover units they are "
           "split into (step 2), because a component above the floor can still be cut by land cover into "
           "slivers below it. A single-cell component is removed at step 1 and therefore never becomes a "
           "unit, never enters a ranking and never appears on a map."
           % (MIN_UNIT_HA, MIN_UNIT_PX),
           {"components": "8-connected (3x3 structure) of the burned mask",
            "units": "component x 2018 land-cover class",
            "floor": "MIN_UNIT_HA = %.2f ha (MIN_UNIT_PX = %d cells at 30 m)" % (MIN_UNIT_HA, MIN_UNIT_PX)},
           section="5")

inv = pd.concat([UNITS[p] for p, _ in POPS], ignore_index=True)
cols = (["unit_id", "population", "zone", "land_cover", "n_px", "area_ha", "zone_area_ha", "east", "north"] + QUANT)
save_table("t03_burned_spatial_units", inv[cols],
           "Every burned spatial unit: a connected component of the burned-area mask intersected with the "
           "2018 baseline land-cover class, for both burned-area definitions, with the mean of each model "
           "indicator and each RIS profile inside it. This is the unit of interpretation for everything that "
           "follows; the model itself remains pixel-based.",
           {"zone": "8-connected components of the burned mask (3x3)",
            "land_cover": "data/raw/landcover_2018_30m.tif (2018 baseline, mode-resampled to 30 m)",
            "mean_<indicator>": "outputs/inference_full_roi/<indicator>_30m.tif averaged over unit pixels",
            "RIS_*": "profile weights over the four indicators"}, section="5")

# ------------------------------------------------- §6 global and within-class ranking
SCORE = "RIS_balanced"
rank_tabs = []
for pop, _ in POPS:
    u = UNITS[pop].copy()
    u = u[u.area_ha >= RANK_MIN_HA].copy()
    u["global_rank"] = u[SCORE].rank(ascending=False, method="min").astype(int)
    u["global_pct"] = 100.0 * u[SCORE].rank(pct=True)
    u["within_class_rank"] = u.groupby("land_cover")[SCORE].rank(ascending=False, method="min").astype(int)
    u["within_class_pct"] = 100.0 * u.groupby("land_cover")[SCORE].rank(pct=True)
    u["n_units_in_class"] = u.groupby("land_cover")[SCORE].transform("size").astype(int)
    rank_tabs.append(u)
RANKED = pd.concat(rank_tabs, ignore_index=True)

top_global = (RANKED[RANKED.population == "documented"].nsmallest(TOP_N, "global_rank")
              [["global_rank", "unit_id", "land_cover", "area_ha", "within_class_rank", "n_units_in_class"] + QUANT])
save_table("t04_global_ranking_top", top_global,
           "The %d highest-priority documented burned spatial units overall, ranked by mean Balanced RIS "
           "among units of at least %.1f ha. `within_class_rank` shows where each unit sits among units of its "
           "own land-cover class, which is the comparison Section 7 argues is the more defensible one."
           % (TOP_N, RANK_MIN_HA),
           {"ranking": "mean %s per unit, descending" % SCORE,
            "units": "nb15_t03_burned_spatial_units, documented population, area >= %.1f ha" % RANK_MIN_HA},
           section="6")

rows = []
for pop, _ in POPS:
    u = RANKED[RANKED.population == pop]
    for cname in [c for c in CLASS_NAMES if c in set(u.land_cover)]:
        sub = u[u.land_cover == cname].nsmallest(5, "within_class_rank")
        if not len(sub) or sub.n_units_in_class.iloc[0] < 3:
            continue
        for _, r in sub.iterrows():
            rows.append(dict(population=pop, land_cover=cname, within_class_rank=int(r.within_class_rank),
                             n_units_in_class=int(r.n_units_in_class), unit_id=r.unit_id,
                             area_ha=r.area_ha, east=r.east, north=r.north,
                             RIS_balanced=r.RIS_balanced, severity=r.severity, persistence=r.persistence,
                             recovery_gap=r.recovery_gap, model_recurrence=r.model_recurrence,
                             global_rank=int(r.global_rank), global_pct=r.global_pct))
save_table("t05_within_class_top5", pd.DataFrame(rows),
           "Tarin's second hierarchy: the five highest-priority burned units WITHIN each land-cover class, for "
           "both burned-area definitions. This answers 'which burned forest areas show the greatest restoration "
           "concern relative to other burned forest areas', without requiring that an absolute score be "
           "comparable across classes - which nb13 showed it is not, because the recovery-gap indicator is "
           "normalised by a class-specific pre-fire NBR baseline.",
           {"ranking": "mean %s within land-cover class, descending" % SCORE,
            "units": "nb15_t03, area >= %.1f ha" % RANK_MIN_HA}, section="6")

rows = []
for pop, _ in POPS:
    u = RANKED[RANKED.population == pop]
    for cname in [c for c in CLASS_NAMES if c in set(u.land_cover)]:
        sub = u[u.land_cover == cname]
        if len(sub) < 3:
            continue
        top_within = set(sub.nsmallest(max(1, len(sub) // 10), "within_class_rank").unit_id)
        glob_cut = u[SCORE].quantile(0.9)
        rows.append(dict(population=pop, land_cover=cname, n_units=len(sub),
                         median_RIS=float(sub[SCORE].median()), min_RIS=float(sub[SCORE].min()),
                         max_RIS=float(sub[SCORE].max()), iqr_RIS=float(sub[SCORE].quantile(.75) - sub[SCORE].quantile(.25)),
                         n_in_global_top_decile=int((sub[SCORE] >= glob_cut).sum()),
                         pct_of_class_in_global_top_decile=100.0 * float((sub[SCORE] >= glob_cut).mean()),
                         pct_of_class_top_decile_also_global=100.0 * float(
                             sub[sub.unit_id.isin(top_within)][SCORE].ge(glob_cut).mean())))
save_table("t06_within_vs_global_prioritization", pd.DataFrame(rows),
           "How different the within-class hierarchy is from the global one. A class whose units barely appear "
           "in the global top decile still has its own worst-affected units, which the within-class ranking "
           "surfaces and the global ranking hides.",
           {"global top decile": "90th percentile of unit %s within the population" % SCORE,
            "units": "nb15_t03, area >= %.1f ha" % RANK_MIN_HA}, section="6")

rows = []
for pop, _ in POPS:
    for ha in RANK_SENS_HA:
        u = UNITS[pop]
        u = u[u.area_ha >= ha]
        if len(u) < TOP_N:
            continue
        top = set(u.nlargest(TOP_N, SCORE).unit_id)
        ref_top = set(UNITS[pop][UNITS[pop].area_ha >= RANK_MIN_HA].nlargest(TOP_N, SCORE).unit_id)
        rows.append(dict(population=pop, min_area_ha=ha, n_units=len(u),
                         median_unit_ha=float(u.area_ha.median()),
                         top_n=TOP_N, shared_with_primary_threshold=len(top & ref_top),
                         mean_RIS_of_top=float(u.nlargest(TOP_N, SCORE)[SCORE].mean())))
save_table("t07_ranking_threshold_sensitivity", pd.DataFrame(rows),
           "Sensitivity of the ranked list to the physical minimum-area rule. The aggregation floor is %.2f ha; "
           "the ranking floor is %.1f ha, because a four-pixel unit is large enough to average but too small to "
           "present as a restoration-priority area. `shared_with_primary_threshold` counts how many of the top %d "
           "units are the same as at the %.1f ha floor." % (MIN_UNIT_HA, RANK_MIN_HA, TOP_N, RANK_MIN_HA),
           {"units": "nb15_t03", "ranking": "mean %s" % SCORE}, section="6")
print("\n[§6] ranking done: %d documented units and %d model units at >= %.1f ha"
      % ((RANKED.population == "documented").sum(), (RANKED.population == "model").sum(), RANK_MIN_HA))

# ------------------------------------------------- §7 Purpose A: does the unit change the pattern?
UNIT_IDX = {}
for pop, _ in POPS:
    idx = np.full((H, W), -1, dtype=np.int32)
    lab, u = ZONES[pop], UNITS[pop]
    key = {(z, c): i for i, (z, c) in enumerate(zip(u.zone, u.cls))}
    sel = lab > 0
    zz, cc = lab[sel], CLS[sel]
    flat = np.array([key.get((z, c), -1) for z, c in zip(zz, cc)], dtype=np.int32)
    idx[sel] = flat
    UNIT_IDX[pop] = idx

rows = []
for pop, M in POPS:
    u = UNITS[pop]
    for cname in CLASS_NAMES:
        ci = CLASS_NAMES.index(cname)
        m = M & (CLS == ci)
        npx = int(m.sum())
        if npx < MIN_CLASS_PX:
            continue
        sub = u[u.cls == ci]
        rows.append(dict(population=pop, land_cover=cname, n_px=npx, n_units=len(sub),
                         pixel_mean=float(RIS["balanced"][m].mean()),
                         unit_mean_unweighted=float(sub[SCORE].mean()) if len(sub) else np.nan,
                         unit_mean_area_weighted=float(np.average(sub[SCORE], weights=sub.n_px)) if len(sub) else np.nan,
                         unit_median=float(sub[SCORE].median()) if len(sub) else np.nan,
                         unit_minus_pixel=(float(sub[SCORE].mean()) - float(RIS["balanced"][m].mean())) if len(sub) else np.nan,
                         recovery_gap_pixel_mean=float(RG[m].mean()),
                         recovery_gap_unit_mean=float(sub.recovery_gap.mean()) if len(sub) else np.nan))
purpA = pd.DataFrame(rows)
lines = []
for pop, _ in POPS:
    s = purpA[purpA.population == pop]
    if len(s) < 3:
        continue
    rho = stats.spearmanr(s.pixel_mean, s.unit_mean_unweighted).statistic
    rho_w = stats.spearmanr(s.pixel_mean, s.unit_mean_area_weighted).statistic
    o_px = list(s.sort_values("pixel_mean", ascending=False).land_cover)
    o_un = list(s.sort_values("unit_mean_unweighted", ascending=False).land_cover)
    lines.append(dict(population=pop, n_classes=len(s),
                      spearman_pixel_vs_unit_unweighted=float(rho),
                      spearman_pixel_vs_unit_area_weighted=float(rho_w),
                      ordering_identical=bool(o_px == o_un),
                      pixel_order=" > ".join(o_px), unit_order=" > ".join(o_un)))
purpA_sum = pd.DataFrame(lines)
save_table("t08_pixel_vs_unit_class_means", purpA,
           "Purpose A. The same land-cover class means computed two ways: over all burned pixels of the class "
           "(the published pixel-level view) and over the burned spatial units of that class (the new view). "
           "If moving to spatial units removed the counterintuitive ordering, these two columns would rank the "
           "classes differently.",
           {"pixel_mean": "%s over mask & class pixels" % SCORE,
            "unit_mean_*": "mean of per-unit means from nb15_t03, unweighted and weighted by unit pixel count"},
           section="7")
save_table("t08b_pixel_vs_unit_ordering", purpA_sum,
           "Purpose A, stated as a result: rank correlation between the pixel-level and unit-level class "
           "orderings. A value at or near 1 means the aggregation unit is NOT what produces the pattern, which "
           "points instead to indicator construction, normalisation and baseline spectral differences - the "
           "mechanism nb13 identified. This result is reported as it came out.",
           {"spearman": "scipy.stats.spearmanr over the class means in nb15_t08"}, section="7")
for _, r in purpA_sum.iterrows():
    print("\n[§7 Purpose A | %s] spearman(pixel, unit) = %.4f | ordering identical: %s"
          % (r.population, r.spearman_pixel_vs_unit_unweighted, r.ordering_identical))
    print("      pixel: %s" % r.pixel_order)
    print("      unit : %s" % r.unit_order)

# ------------------------------------------------- §8 the two polygon-level maps
roi = gpd.read_file(ROI_SHP).to_crs(CRS)
extent = [GRID_T.c, GRID_T.c + W * GRID_T.a, GRID_T.f + H * GRID_T.e, GRID_T.f]


def unit_surface(pop):
    u, idx = UNITS[pop], UNIT_IDX[pop]
    vals = np.full(len(u) + 1, np.nan)
    vals[:len(u)] = u[SCORE].to_numpy()
    out = np.where(idx >= 0, vals[np.clip(idx, 0, len(u))], np.nan)
    return np.where(idx >= 0, out, np.nan)


def km_axes(ax):
    ax.set_xlabel("easting (km)")
    ax.set_ylabel("northing (km)")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, p: "%.0f" % (v / 1000)))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, p: "%.0f" % (v / 1000)))


bg = np.where(ELIG, 1.0, np.nan)
fig, axes = plt.subplots(1, 2, figsize=(11.5, 9.4), sharex=True, sharey=True, layout="constrained")
LABS = [("documented", "(a) documented burned areas\n(fire-reference polygons 2017-2025, dissolved)"),
        ("model", "(b) model-detected burned areas\n(burned probability > %.3f)" % THR)]
for ax, (pop, lab) in zip(axes, LABS):
    ax.imshow(bg, extent=extent, cmap=ListedColormap(["#e9e9e9"]), interpolation="nearest")
    im = ax.imshow(unit_surface(pop), extent=extent, cmap=RESTORATION_CMAP, vmin=0, vmax=1,
                   interpolation="nearest")
    roi.boundary.plot(ax=ax, color="k", linewidth=0.6)
    u = UNITS[pop]
    ax.set_title("%s\n%d units, %.0f ha; unit RIS mean %.2f, median %.2f"
                 % (lab, len(u), u.area_ha.sum(), u[SCORE].mean(), u[SCORE].median()), fontsize=9.5)
    km_axes(ax)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
axes[1].set_ylabel("")
cb = fig.colorbar(im, ax=list(axes), orientation="horizontal", fraction=0.03, pad=0.02, shrink=0.6)
cb.set_label("Restoration priority of the burned spatial unit - mean Balanced RIS (identical colormap and 0-1 scale)")
fig.suptitle("Polygon-level restoration priority: documented vs model-detected burned areas", fontsize=11)
save_figure(fig, "f01_polygon_priority_reference_vs_model",
            "The core comparison. Each burned spatial unit is filled with its own mean Balanced RIS, so the "
            "reader compares spatial units rather than individual pixels. Left: burned area defined by the "
            "fire-reference inventory. Right: burned area defined by model fire-occurrence output. Identical "
            "extent, colormap, scale, land-cover definition and unit rule; gray is the eligible area. The "
            "fire-reference inventory is incomplete, so area the model flags outside it is not by itself a "
            "false positive.",
            {"units": "nb15_t03_burned_spatial_units",
             "RIS": "0.25 x (severity + persistence + recovery_gap + model_recurrence) at 30 m",
             "outline": "data/thesis_maps/fig_roi_polygon.shp"}, section="8")

# ------------------------------------------------- §9 within-class distributions
major = [c for c in CLASS_NAMES if
         ((RANKED.population == "documented") & (RANKED.land_cover == c)).sum() >= 3]
fig, ax = plt.subplots(figsize=(min(2.0 + 1.5 * len(major), 13), 4.6), layout="constrained")
pos, ticks = 0, []
for cname in major:
    for j, (pop, color) in enumerate([("documented", "#2c7bb6"), ("model", "#d7191c")]):
        v = RANKED[(RANKED.population == pop) & (RANKED.land_cover == cname)][SCORE].to_numpy()
        if not len(v):
            continue
        ax.boxplot(v, positions=[pos + j * 0.38 - 0.19], widths=0.32, showfliers=False,
                   medianprops=dict(color=color, lw=1.6), boxprops=dict(color=color),
                   whiskerprops=dict(color=color), capprops=dict(color=color))
        ax.scatter(np.random.default_rng(0).normal(pos + j * 0.38 - 0.19, 0.045, len(v)), v,
                   s=5, color=color, alpha=0.35, linewidths=0)
    ticks.append((pos, cname))
    pos += 1
ax.set_xticks([p for p, _ in ticks])
ax.set_xticklabels([c for _, c in ticks], rotation=20, ha="right")
ax.set_ylabel("unit mean Balanced RIS")
ax.set_ylim(0, 1)
ax.grid(axis="y", alpha=0.25, lw=0.5)
ax.set_title("Restoration priority of burned spatial units within each 2018 land-cover class\n"
             "blue = documented burned areas, red = model-detected (units >= %.1f ha)" % RANK_MIN_HA, fontsize=10)
save_figure(fig, "f02_within_class_unit_distributions",
            "The spread that a class mean hides. Each point is one burned spatial unit. The within-class "
            "hierarchy ranks units inside each of these columns, which is defensible even where the absolute "
            "level of a column is set by the class-specific NBR baseline rather than by ecological recovery.",
            {"units": "nb15_t03_burned_spatial_units, area >= %.1f ha" % RANK_MIN_HA,
             "land cover": "data/raw/landcover_2018_30m.tif"}, section="9")

# ------------------------------------------------- §10 agreement of the two prioritizations
di, mi = UNIT_IDX["documented"], UNIT_IDX["model"]
both = (di >= 0) & (mi >= 0)
ud, um = UNITS["documented"], UNITS["model"]
pairs = pd.DataFrame({"d": di[both], "m": mi[both]}).value_counts().reset_index(name="n_px")
best = pairs.sort_values("n_px", ascending=False).drop_duplicates("m")
matched = best.assign(d_ris=ud[SCORE].to_numpy()[best.d], m_ris=um[SCORE].to_numpy()[best.m],
                      d_lc=ud.land_cover.to_numpy()[best.d], m_lc=um.land_cover.to_numpy()[best.m])
rho_units = stats.spearmanr(matched.d_ris, matched.m_ris).statistic if len(matched) > 2 else np.nan
cls_d = purpA[purpA.population == "documented"].set_index("land_cover").unit_mean_unweighted
cls_m = purpA[purpA.population == "model"].set_index("land_cover").unit_mean_unweighted
shared_cls = [c for c in cls_d.index if c in cls_m.index]
rho_cls = stats.spearmanr(cls_d[shared_cls], cls_m[shared_cls]).statistic if len(shared_cls) > 2 else np.nan


def top_decile_mask(pop):
    u = UNITS[pop]
    cut = u[SCORE].quantile(0.9)
    keep = set(u.index[u[SCORE] >= cut])
    idx = UNIT_IDX[pop]
    return np.isin(idx, list(keep)) & (idx >= 0)


td, tm = top_decile_mask("documented"), top_decile_mask("model")
UREF, UDET = UNIT_IDX["documented"] >= 0, UNIT_IDX["model"] >= 0
rows = [
    dict(measure="OBSERVED — documented burned mask", value=REF.sum() * PX_HA, unit="ha",
         detail="dissolved fire-reference polygons 2017-2025 on eligible pixels, before the minimum-area rule"),
    dict(measure="OBSERVED — model-detected burned mask", value=DET.sum() * PX_HA, unit="ha",
         detail="burned_prob > %.3f, before the minimum-area rule" % THR),
    dict(measure="OBSERVED — documented area analysed as units", value=UREF.sum() * PX_HA, unit="ha",
         detail="after the minimum-area rule; this is what the maps and rankings use"),
    dict(measure="OBSERVED — model area analysed as units", value=UDET.sum() * PX_HA, unit="ha",
         detail="after the minimum-area rule; this is what the maps and rankings use"),
    dict(measure="OBSERVED — Jaccard of the burned masks", value=float((REF & DET).sum() / (REF | DET).sum()),
         unit="index", detail="footprint overlap only; says nothing about priority"),
    dict(measure="OBSERVED — Jaccard of the analysed unit footprints",
         value=float((UREF & UDET).sum() / (UREF | UDET).sum()), unit="index",
         detail="the comparison that matches the two maps"),
    dict(measure="OBSERVED — documented area also flagged by the model",
         value=100.0 * float((REF & DET).sum() / REF.sum()), unit="%", detail="spatial overlap, stated as observed"),
    dict(measure="OBSERVED — model-flagged area lying outside the inventory footprint",
         value=100.0 * float((DET & ~REF).sum() / DET.sum()), unit="%",
         detail="a spatial observation only; see the interpretation row below"),
    dict(measure="INTERPRETATION — status of model-flagged area outside the inventory", value=np.nan, unit="",
         detail="undetermined by this analysis. The fire-reference inventory is not assumed to be exhaustive, "
                "so non-overlap cannot be read as either a confirmed fire or a false positive; establishing "
                "which would require an independent source, not this comparison"),
    dict(measure="spatially matched unit pairs", value=float(len(matched)), unit="pairs",
         detail="each model unit matched to the documented unit it overlaps most"),
    dict(measure="Spearman of unit priority, matched pairs", value=float(rho_units), unit="rho",
         detail="how similarly the two definitions rank the same ground"),
    dict(measure="matched pairs agreeing on land-cover class", value=100.0 * float((matched.d_lc == matched.m_lc).mean()),
         unit="%", detail=""),
    dict(measure="Spearman of the land-cover class ordering", value=float(rho_cls), unit="rho",
         detail="documented vs model unit-level class means"),
    dict(measure="Jaccard of the top-decile priority areas", value=float((td & tm).sum() / max((td | tm).sum(), 1)),
         unit="index", detail="where each definition says priority is highest"),
    dict(measure="mean unit RIS, documented", value=float(ud[SCORE].mean()), unit="RIS", detail=""),
    dict(measure="mean unit RIS, model", value=float(um[SCORE].mean()), unit="RIS", detail=""),
]
save_table("t09_priority_pattern_agreement", pd.DataFrame(rows),
           "Quantified comparison of the two prioritization patterns, rather than a visual one. Note that both "
           "maps read the SAME indicator surfaces, so any difference comes from which ground each definition "
           "includes and how that ground is partitioned into units - not from the units being scored "
           "differently. Footprint agreement and priority agreement are therefore reported separately.",
           {"masks": "dissolved fire-reference polygons; burned_prob_30m.tif > %.3f" % THR,
            "units": "nb15_t03_burned_spatial_units",
            "matching": "maximum pixel overlap between a model unit and a documented unit"}, section="10")
print("\n[§10] footprint Jaccard %.4f | matched-pair priority rho %.4f | class-order rho %.4f | top-decile Jaccard %.4f"
      % ((REF & DET).sum() / (REF | DET).sum(), rho_units, rho_cls, (td & tm).sum() / max((td | tm).sum(), 1)))

# ------------------------------------------------- §11 registry
reg = pd.DataFrame(REGISTRY)
reg.to_csv(os.path.join(TAB, PREFIX + "t99_result_registry.csv"), index=False)
open(os.path.join(TAB, PREFIX + "t99_result_registry.md"), "w").write(
    "<!-- generated by notebooks/15_polygon_workflow.py on %s -->\n\nEvery output of the consolidated "
    "polygon-level workflow.\n\n" % RUN_STAMP + md_table(reg) + "\n")
print("\n" + "=" * 100)
print("done — %d outputs written to outputs/{tables,figures}" % len(REGISTRY))
print("=" * 100)
