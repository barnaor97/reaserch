"""
21_final_figures_10_11.py — FINAL publication-quality replacement Figures 10 and 11.

Deterministic plotting from the verified research outputs only. No scientific quantity is
recomputed: unit construction is re-derived with the authoritative rules and ASSERTED identical to
outputs/final_polygon_thesis/*_units_final.csv, and every plotted value is read from those tables.
No generative or synthesized imagery is used anywhere.

Writes outputs/final_polygon_maps/{figure10_final,figure11_final}.{png,pdf} + FIGURE_PROVENANCE.md
Run: python notebooks/21_final_figures_10_11.py
"""
import hashlib, json, os, platform, sys, warnings
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
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAST = os.path.join(ROOT, "outputs", "inference_full_roi")
FIN = os.path.join(ROOT, "outputs", "final_polygon_thesis")
OUT = os.path.join(ROOT, "outputs", "final_polygon_maps")
os.makedirs(OUT, exist_ok=True)
LC_TIF = os.path.join(ROOT, "data", "raw", "landcover_2018_30m.tif")
FIRES = os.path.join(ROOT, "data", "thesis_maps", "fig_gt_fires.shp")
ROI_SHP = os.path.join(ROOT, "data", "thesis_maps", "fig_roi_polygon.shp")
GRID_T = Affine(30.0, 0.0, 620610.0, 0.0, -30.0, 3514560.0)
H, W, CRS = 2598, 1625, "EPSG:32636"
PX_HA, THR, MIN_UNIT_HA = 0.09, 0.502, 0.36
MIN_UNIT_PX = int(round(MIN_UNIT_HA / PX_HA))
LC_NAMES = {10: "forest", 11: "orchards", 20: "shrubland", 30: "grassland", 40: "cropland",
            41: "covered agriculture", 50: "built-up", 60: "bare", 61: "fallow", 80: "water"}
LC_ORDER = [10, 11, 20, 30, 40, 41, 50, 60, 61, 80]
CLS_ORDER = [LC_NAMES[c] for c in LC_ORDER] + ["unclassified / other code"]
CLS_IDX = {c: i for i, c in enumerate(CLS_ORDER)}
SEQ = LinearSegmentedColormap.from_list("seq", ["#f7fbff", "#c6dbef", "#6baed6", "#fdae61", "#d7191c"], N=256)
_cm = plt.get_cmap("tab10")
CLASS_COLOR = {LC_NAMES[c]: _cm(i) for i, c in enumerate(LC_ORDER)}
CLASS_COLOR["unclassified / other code"] = (0.62, 0.62, 0.62, 1.0)
RUN = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
matplotlib.rcParams.update({"font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42,
                            "axes.spines.top": False, "axes.spines.right": False})
QA, PROV = [], {}


def check(name, cond, detail=""):
    QA.append((name, bool(cond), detail))
    print("[%s] %-62s %s" % ("PASS" if cond else "FAIL", name, detail))
    if not cond:
        raise SystemExit("QA FAILED: %s — %s" % (name, detail))


# ================================================================ 0. freeze the source of truth
A = {}
for role, fn in [("severity", "severity_30m.tif"), ("persistence", "persistence_30m.tif"),
                 ("recovery_gap_predicted", "recovery_gap_30m.tif"),
                 ("model_recurrence", "model_recurrence_30m.tif"), ("burned_prob", "burned_prob_30m.tif")]:
    with rasterio.open(os.path.join(RAST, fn)) as s:
        assert str(s.crs) == CRS and (s.height, s.width) == (H, W) and s.transform.almost_equals(GRID_T), fn
        A[role] = s.read(1).astype(np.float64)
with rasterio.open(LC_TIF) as s:
    assert str(s.crs) == CRS and (s.height, s.width) == (H, W) and s.transform.almost_equals(GRID_T)
    LC_GRID = s.read(1)
check("all rasters on the validated 30 m grid", True, "EPSG:32636, 1625 x 2598, origin 620610/3514560")
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
REF_RAW, DET_RAW = INSIDE & ELIG, ELIG & (A["burned_prob"] > THR)
STRUCT = np.ones((3, 3), dtype=int)


def components(mask, min_px=0):
    lab, k = ndimage.label(mask, structure=STRUCT)
    if min_px > 1 and k:
        sizes = np.bincount(lab.ravel()); drop = np.where(sizes < min_px)[0]; drop = drop[drop > 0]
        if len(drop):
            lab[np.isin(lab, drop)] = 0
        lab, k = ndimage.label(lab > 0, structure=STRUCT)
    return lab, k


CSV = {"reference": pd.read_csv(os.path.join(FIN, "reference_units_final.csv")),
       "model": pd.read_csv(os.path.join(FIN, "model_units_final.csv"))}
PRE = {"reference": "doc", "model": "mod"}
UIDX, UNITS = {}, {}
for pop, M in (("reference", REF_RAW), ("model", DET_RAW)):
    lab, _ = components(M, MIN_UNIT_PX)
    sel = lab > 0
    d = pd.DataFrame({"zone": lab[sel], "cls": cls_i[sel]})
    u = d.groupby(["zone", "cls"]).size().rename("pixel_count").reset_index()
    u = u[u.pixel_count >= MIN_UNIT_PX].reset_index(drop=True)
    u["unit_id"] = ["%s_%d_%d" % (PRE[pop], z, c) for z, c in zip(u.zone, u.cls)]
    check("%s: unit_id set identical to the final table" % pop, set(u.unit_id) == set(CSV[pop].unit_id),
          "%d units" % len(u))
    check("%s: no duplicated unit_id" % pop, not u.unit_id.duplicated().any())
    m = u.merge(CSV[pop], on="unit_id", how="inner", suffixes=("", "_f"))
    check("%s: join is one-to-one by unit_id" % pop, len(m) == len(u) == len(CSV[pop]))
    check("%s: pixel counts match the final table" % pop, (m.pixel_count == m.pixel_count_f).all())
    idx = np.full((H, W), -1, dtype=np.int32)
    key = {(z, c): i for i, (z, c) in enumerate(zip(u.zone, u.cls))}
    idx[sel] = np.array([key.get((z, c), -1) for z, c in zip(lab[sel], cls_i[sel])], dtype=np.int32)
    UIDX[pop] = idx
    UNITS[pop] = CSV[pop].set_index("unit_id").loc[u.unit_id].reset_index()
u_ref, u_mod = UNITS["reference"], UNITS["model"]
check("reference units n = 1,064", len(u_ref) == 1064, str(len(u_ref)))
check("reference unit area = 2,819.70 ha", abs(u_ref.area_ha.sum() - 2819.70) < 0.01,
      "%.2f ha" % u_ref.area_ha.sum())
check("model units n = 2,209", len(u_mod) == 2209, str(len(u_mod)))
check("model unit area = 5,397.48 ha", abs(u_mod.area_ha.sum() - 5397.48) < 0.01,
      "%.2f ha" % u_mod.area_ha.sum())
check("plotted unit count equals expected (reference)", int((UIDX["reference"] >= 0).sum()) == int(u_ref.pixel_count.sum()))
check("plotted unit count equals expected (model)", int((UIDX["model"] >= 0).sum()) == int(u_mod.pixel_count.sum()))

CAT = np.zeros((H, W), dtype=np.int16)
CAT[REF_RAW & ~DET_RAW] = 1
CAT[REF_RAW & DET_RAW] = 2
CAT[DET_RAW & ~REF_RAW] = 3
a_ref_only, a_over, a_mod_only = [float((CAT == v).sum() * PX_HA) for v in (1, 2, 3)]
jac = float((REF_RAW & DET_RAW).sum() / (REF_RAW | DET_RAW).sum())
pct_det = 100.0 * float((REF_RAW & DET_RAW).sum() / REF_RAW.sum())
pct_out = 100.0 * float((DET_RAW & ~REF_RAW).sum() / DET_RAW.sum())
check("raw reference footprint = 2,984 ha", abs(a_ref_only + a_over - 2984.13) < 0.5, "%.2f ha" % (a_ref_only + a_over))
check("raw model footprint = 6,794 ha", abs(a_over + a_mod_only - 6794.28) < 0.5, "%.2f ha" % (a_over + a_mod_only))
check("reference only = 1,317 ha", abs(a_ref_only - 1317) < 2, "%.0f ha" % a_ref_only)
check("overlap = 1,667 ha", abs(a_over - 1667) < 2, "%.0f ha" % a_over)
check("model only = 5,127 ha", abs(a_mod_only - 5127) < 2, "%.0f ha" % a_mod_only)
check("footprint Jaccard = 0.2055", abs(jac - 0.2055) < 5e-4, "%.4f" % jac)
check("reference area also detected = 55.9%", abs(pct_det - 55.9) < 0.1, "%.1f%%" % pct_det)
check("model area outside inventory = 75.5%", abs(pct_out - 75.5) < 0.1, "%.1f%%" % pct_out)

# The published count of 37 covers the six main vegetated classes. In classes represented by very few
# units the "top decile" rule degenerates to the single highest-scoring unit, so those classes are
# reported per class but are NOT highlighted on the map; the caption states this.
MAIN = ["forest", "orchards", "shrubland", "grassland", "cropland", "fallow"]
gcut = float(u_ref.mean_RIS_balanced.quantile(0.90))
hidden, hidden_all, per_class = [], [], {}
for c, s in u_ref.groupby("land_cover_name"):
    wcut = float(s.mean_RIS_balanced.quantile(0.90)) if len(s) >= 10 else float(s.mean_RIS_balanced.max())
    top = s[s.mean_RIS_balanced >= wcut]
    hid = top[top.mean_RIS_balanced < gcut]
    per_class[c] = (len(hid), len(top))
    hidden_all += list(hid.unit_id)
    if c in MAIN:
        hidden += list(hid.unit_id)
n_hidden = sum(per_class.get(c, (0, 0))[0] for c in MAIN)
n_small = len(hidden_all) - len(hidden)
check("within-LC hidden units (six main vegetated classes) = 37", n_hidden == 37, str(n_hidden))
check("highlighted set equals the published 37", len(hidden) == 37, "%d highlighted; %d further units in small-n classes are reported but not highlighted" % (len(hidden), n_small))
check("cropland hidden = 25 of 27", per_class["cropland"] == (25, 27), str(per_class["cropland"]))
check("forest hidden = 8 of 16", per_class["forest"] == (8, 16), str(per_class["forest"]))
check("orchards hidden = 4 of 4", per_class["orchards"] == (4, 4), str(per_class["orchards"]))

ATTR = {"10a": "mean_severity_model", "10b": "mean_persistence_model",
        "10c": "mean_recovery_gap_predicted", "10d": "mean_recurrence_model",
        "11d": "within_landcover_RIS_percentile"}
for k, col in ATTR.items():
    v = u_ref[col].to_numpy(dtype=float)
    check("%s attribute '%s' present and finite" % (k, col), np.isfinite(v).all(), "n=%d" % len(v))

# ================================================================ cartographic helpers
roi = gpd.read_file(ROI_SHP).to_crs(CRS)
EXT = [GRID_T.c, GRID_T.c + W * GRID_T.a, GRID_T.f + H * GRID_T.e, GRID_T.f]
BG = np.where(ELIG, 1.0, np.nan)


def paint(pop, col):
    u, idx = UNITS[pop], UIDX[pop]
    vals = np.append(u[col].to_numpy(dtype=float), np.nan)
    return np.where(idx >= 0, vals[np.clip(idx, 0, len(u))], np.nan)


def basemap(ax, label, title):
    ax.imshow(BG, extent=EXT, cmap=ListedColormap(["#e8e8e8"]), interpolation="nearest", rasterized=True)
    roi.boundary.plot(ax=ax, color="#555555", linewidth=0.55)
    ax.set_xlim(EXT[0], EXT[1]); ax.set_ylim(EXT[2], EXT[3])
    ax.set_xticks([]); ax.set_yticks([]); ax.set_frame_on(False)
    ax.set_title(title, fontsize=9.5, pad=6)
    ax.text(0.015, 0.985, label, transform=ax.transAxes, fontsize=11, fontweight="bold",
            va="top", ha="left")


def decorate(ax, km=10):
    x0, y0 = EXT[0] + 2000, EXT[2] + 4000
    ax.plot([x0, x0 + km * 1000], [y0, y0], color="#222222", lw=2.4, solid_capstyle="butt")
    ax.text(x0 + km * 500, y0 + 2200, "%d km" % km, ha="center", fontsize=7.5)
    ax.annotate("N", xy=(EXT[1] - 4500, EXT[3] - 5000), xytext=(EXT[1] - 4500, EXT[3] - 15000),
                ha="center", fontsize=9, arrowprops=dict(arrowstyle="-|>", color="#222222", lw=1.1))


# ================================================================ FIGURE 10
PANELS10 = [("(a)", "Burn severity", ATTR["10a"], "full"),
            ("(b)", "Persistence", ATTR["10b"], "full"),
            ("(c)", "Recovery gap", ATTR["10c"], "full"),
            ("(d)", "Recurrence", ATTR["10d"], "robust")]
LIMITS = {}
fig, axes = plt.subplots(2, 2, figsize=(9.6, 11.4), layout="constrained")
for ax, (lab, title, col, mode) in zip(axes.ravel(), PANELS10):
    v = u_ref[col].to_numpy(dtype=float)
    if mode == "full":
        lo, hi, ext, clipped = float(v.min()), float(v.max()), "neither", 0
    else:
        lo, hi = 0.0, float(np.percentile(v, 98))
        clipped = int((v > hi).sum()); ext = "max"
    LIMITS[col] = dict(vmin=round(lo, 4), vmax=round(hi, 4), mode=mode, clipped_units=clipped,
                       observed_min=round(float(v.min()), 4), observed_max=round(float(v.max()), 4))
    basemap(ax, lab, title)
    im = ax.imshow(paint("reference", col), extent=EXT, cmap=SEQ, vmin=lo, vmax=hi,
                   interpolation="nearest", rasterized=True)
    cb = fig.colorbar(im, ax=ax, orientation="vertical", fraction=0.040, pad=0.015, extend=ext)
    cb.ax.tick_params(labelsize=7.5)
    cb.set_label("unit mean" + (" (display limit %.2f; %d units above)" % (hi, clipped) if clipped else ""),
                 fontsize=7.8)
decorate(axes[0, 0])
fig.suptitle("Disturbance and post-fire spectral indicators over reference-defined burned spatial units",
             fontsize=11.5)
for e in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, "figure10_final." + e), bbox_inches="tight", facecolor="white",
                dpi=400 if e == "png" else None)
plt.close(fig)
print("\nFigure 10 display limits:")
for k, d in LIMITS.items():
    print("   %-30s %s [%.4f, %.4f]  observed [%.4f, %.4f]  clipped %d"
          % (k, d["mode"], d["vmin"], d["vmax"], d["observed_min"], d["observed_max"], d["clipped_units"]))

# ================================================================ FIGURE 11
present = [c for c in CLS_ORDER if (u_ref.land_cover_name == c).any() or (u_mod.land_cover_name == c).any()]
code_of = {c: i for i, c in enumerate(present)}
LCCMAP = ListedColormap([CLASS_COLOR[c] for c in present])


def paint_class(pop):
    u, idx = UNITS[pop], UIDX[pop]
    vals = np.append(u.land_cover_name.map(code_of).to_numpy(dtype=float), np.nan)
    return np.where(idx >= 0, vals[np.clip(idx, 0, len(u))], np.nan)


fig, axes = plt.subplots(2, 2, figsize=(9.6, 11.8), layout="constrained")
basemap(axes[0, 0], "(a)", "Reference-defined burned units\n%s units, %s ha"
        % (format(len(u_ref), ","), format(int(round(u_ref.area_ha.sum())), ",")))
axes[0, 0].imshow(paint_class("reference"), extent=EXT, cmap=LCCMAP, vmin=-0.5, vmax=len(present) - 0.5,
                  interpolation="nearest", rasterized=True)
decorate(axes[0, 0])
basemap(axes[0, 1], "(b)", "Model-detected burned units\n%s units, %s ha (burned probability > %.3f)"
        % (format(len(u_mod), ","), format(int(round(u_mod.area_ha.sum())), ","), THR))
axes[0, 1].imshow(paint_class("model"), extent=EXT, cmap=LCCMAP, vmin=-0.5, vmax=len(present) - 0.5,
                  interpolation="nearest", rasterized=True)
basemap(axes[1, 0], "(c)", "Footprint agreement before the minimum-area rule\nJaccard %.4f" % jac)
AGC = {1: "#2166ac", 2: "#762a83", 3: "#d6604d"}
axes[1, 0].imshow(np.where(CAT > 0, CAT, np.nan), extent=EXT,
                  cmap=ListedColormap([AGC[1], AGC[2], AGC[3]]), vmin=0.5, vmax=3.5,
                  interpolation="nearest", rasterized=True)
axes[1, 0].legend(handles=[Patch(facecolor=AGC[1], label="reference only  %s ha" % format(int(round(a_ref_only)), ",")),
                           Patch(facecolor=AGC[2], label="overlap  %s ha" % format(int(round(a_over)), ",")),
                           Patch(facecolor=AGC[3], label="model only  %s ha" % format(int(round(a_mod_only)), ","))],
                  fontsize=7.6, loc="lower left", framealpha=0.95, borderpad=0.5, handlelength=1.4)
basemap(axes[1, 1], "(d)", "Prioritization within land-cover class\nRIS percentile among units of the same 2018 class")
im = axes[1, 1].imshow(paint("reference", ATTR["11d"]), extent=EXT, cmap=SEQ, vmin=0, vmax=100,
                       interpolation="nearest", rasterized=True)
HID = gpd.read_file(os.path.join(OUT, "reference_units.gpkg"), layer="reference_units")
HID = HID[HID.unit_id.isin(hidden)].copy()
check("outlined units = 37", len(HID) == 37, str(len(HID)))
_a = float(HID.geometry.area.sum())
HID["geometry"] = HID.geometry.apply(make_valid)
check("outline geometry repair preserves area", abs(float(HID.geometry.area.sum()) - _a) < 1e-6,
      "%.4f ha" % (_a / 1e4))
check("outlined geometries valid", bool(HID.geometry.is_valid.all()))
HID.boundary.plot(ax=axes[1, 1], color="#111111", linewidth=1.0)
cb = fig.colorbar(im, ax=axes[1, 1], orientation="vertical", fraction=0.040, pad=0.015)
cb.ax.tick_params(labelsize=7.5); cb.set_label("within-class RIS percentile", fontsize=7.8)
axes[1, 1].legend(handles=[Line2D([], [], color="#111111", lw=1.4,
                                  label="within-class top decile,\nbelow global top decile (%d units)" % n_hidden)],
                  fontsize=7.4, loc="lower left", framealpha=0.95, borderpad=0.5, handlelength=1.6)
fig.legend(handles=[Patch(facecolor=CLASS_COLOR[c], label=c) for c in present],
           loc="lower center", ncol=5, fontsize=8.0, frameon=False,
           bbox_to_anchor=(0.5, -0.060), title="2018 land-cover class (panels a and b)",
           title_fontsize=8.4)
fig.suptitle("Burned spatial units, footprint agreement and land-cover-aware prioritization", fontsize=11.5)
for e in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, "figure11_final." + e), bbox_inches="tight", facecolor="white",
                dpi=400 if e == "png" else None)
plt.close(fig)

# ================================================================ provenance
def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


files = ["figure10_final.png", "figure10_final.pdf", "figure11_final.png", "figure11_final.pdf"]
PROV = dict(generated=RUN, script="notebooks/21_final_figures_10_11.py",
            python=sys.version.split()[0], platform=platform.platform(),
            crs=CRS, grid="1625 x 2598 cells at 30 m, origin 620610 / 3514560 (EPSG:32636)",
            inputs=["outputs/inference_full_roi/{severity,persistence,recovery_gap,model_recurrence,burned_prob}_30m.tif",
                    "data/raw/landcover_2018_30m.tif", "data/thesis_maps/fig_gt_fires.shp",
                    "data/thesis_maps/fig_roi_polygon.shp",
                    "outputs/final_polygon_thesis/reference_units_final.csv",
                    "outputs/final_polygon_thesis/model_units_final.csv",
                    "outputs/final_polygon_maps/reference_units.gpkg"],
            unit_rules=dict(reference="fire_year >= 2017 polygons dissolved by rasterisation, intersected "
                                      "with the eligible model domain", model="burned probability > 0.502",
                            components="8-connected (3x3)", land_cover="fixed 2018 baseline, 30 m",
                            minimum_area="%.2f ha (%d cells)" % (MIN_UNIT_HA, MIN_UNIT_PX)),
            attributes=ATTR, display_limits=LIMITS,
            populations=dict(reference_units=len(u_ref), reference_area_ha=round(float(u_ref.area_ha.sum()), 2),
                             model_units=len(u_mod), model_area_ha=round(float(u_mod.area_ha.sum()), 2),
                             raw_reference_ha=round(a_ref_only + a_over, 2),
                             raw_model_ha=round(a_over + a_mod_only, 2), jaccard=round(jac, 4),
                             reference_detected_pct=round(pct_det, 1), model_outside_pct=round(pct_out, 1),
                             within_lc_hidden_units=n_hidden),
            outputs={f: sha(os.path.join(OUT, f)) for f in files},
            image_generation="none; all panels are matplotlib renderings of the research data")
json.dump(PROV, open(os.path.join(OUT, "FIGURE_PROVENANCE.json"), "w"), indent=1)
lines = ["# Final Figures 10 and 11 — provenance", "", "Generated %s by `%s`." % (RUN, PROV["script"]), "",
         "No generative or AI-synthesized imagery was used; every panel is a deterministic matplotlib "
         "rendering of the research data.", "", "## Files and SHA-256", "", "| file | sha256 |", "|---|---|"]
lines += ["| `%s` | `%s` |" % (f, h) for f, h in PROV["outputs"].items()]
lines += ["", "## Display limits (Figure 10)", "", "| panel attribute | mode | vmin | vmax | observed range | units clipped |", "|---|---|---|---|---|---|"]
for k, d in LIMITS.items():
    lines.append("| `%s` | %s | %.4f | %.4f | %.4f – %.4f | %d |"
                 % (k, d["mode"], d["vmin"], d["vmax"], d["observed_min"], d["observed_max"], d["clipped_units"]))
lines += ["", "## QA", "", "| check | result |", "|---|---|"] + \
         ["| %s | %s |" % (n, "PASS" if o else "FAIL") for n, o, _ in QA]
open(os.path.join(OUT, "FIGURE_PROVENANCE.md"), "w").write("\n".join(lines) + "\n")
print("\nwrote figure10_final.{png,pdf}, figure11_final.{png,pdf}, FIGURE_PROVENANCE.{md,json}")
print("all %d QA checks passed" % len(QA))
