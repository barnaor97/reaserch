import os
# -*- coding: utf-8 -*-
"""
Study-area environmental context figure for Section 3.2.1 (supervisor comment: stronger
map + environmental characterisation).

Two panels over the identical extent, both clipped to the thesis ROI polygon:
  (a) mean annual precipitation, WorldClim 2.1 bio_12   (Fick & Hijmans, 2017)
  (b) elevation, SRTM GL1 1 arc-second                  (Farr et al., 2007)

Reuses the ROI and data lineage of notebooks/09_revision_verification.py
(nb09_t04 bbox, nb09_t06 WorldClim 2.1, nb09_t07 SRTM 30 m). Nothing existing is modified;
this script only reads external rasters and writes a new figure + a stats table.

Reproduce:
    python notebooks/22_study_area_environment.py
(Source rasters are fetched once into data/external/ if absent; see FETCH below.)
"""
import json, os, subprocess, sys, warnings
warnings.filterwarnings('ignore', message='Mean of empty slice')
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.windows import from_bounds
from rasterio.features import geometry_mask
from rasterio.transform import from_origin
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPoly

ROOT = os.environ.get("THESIS_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXT_DIR = os.path.join(ROOT, "data", "external")
OUT = os.path.join(ROOT, "outputs", "figures")
TAB = os.path.join(ROOT, "outputs", "tables")
ROI_SHP = os.path.join(ROOT, "data", "thesis_maps", "fig_roi_polygon.shp")
DST_CRS = "EPSG:32636"                      # thesis analysis grid
BBOX_NB09 = (34.2672, 31.0574, 34.7820, 31.7495)   # nb09_t04, for cross-checking only

SRC = {
 "worldclim": dict(
    path=os.path.join(EXT_DIR, "ISR_wc2.1_30s_bio.tif"),
    url="https://geodata.ucdavis.edu/climate/worldclim/2_1/tiles/iso/ISR_wc2.1_30s_bio.tif",
    note="WorldClim 2.1, 30 arc-second bioclimatic variables, Israel tile; band 12 = bio_12"),
 "srtm": dict(
    path=os.path.join(EXT_DIR, "N31E034.hgt"),
    url="https://s3.amazonaws.com/elevation-tiles-prod/skadi/N31/N31E034.hgt.gz",
    note="SRTM GL1 void-filled 1 arc-second, AWS Open Data Terrain Tiles (skadi)"),
}

def fetch():
    os.makedirs(EXT_DIR, exist_ok=True)
    wc = SRC["worldclim"]
    if not os.path.exists(wc["path"]):
        subprocess.run(["curl", "-sL", "-o", wc["path"], wc["url"]], check=True)
    dem = SRC["srtm"]
    if not os.path.exists(dem["path"]):
        gz = dem["path"] + ".gz"
        subprocess.run(["curl", "-sL", "-o", gz, dem["url"]], check=True)
        subprocess.run(["gunzip", "-kf", gz], check=True)

def to_utm(src_arr, src_transform, src_crs, bounds_ll, res, method=Resampling.bilinear):
    """Reproject a WGS84 array to EPSG:32636 at the given metre resolution."""
    dst_transform, w, h = calculate_default_transform(
        src_crs, DST_CRS, src_arr.shape[1], src_arr.shape[0], *bounds_ll, resolution=res)
    dst = np.full((h, w), np.nan, dtype="float32")
    reproject(source=src_arr, destination=dst,
              src_transform=src_transform, src_crs=src_crs,
              dst_transform=dst_transform, dst_crs=DST_CRS,
              src_nodata=np.nan, dst_nodata=np.nan, resampling=method)
    return dst, dst_transform

def clip_to_roi(arr, transform, roi_utm, all_touched=False):
    """all_touched=True is used only for DISPLAY, so the coarse precipitation grid fills the
    detailed ROI outline without stair-step gaps. Reported statistics use the strict
    centre-in-polygon mask on the native grids (see *_native_*_roi below)."""
    m = geometry_mask(roi_utm.geometry, out_shape=arr.shape, transform=transform,
                      invert=True, all_touched=all_touched)
    out = arr.copy(); out[~m] = np.nan
    return out

# --------------------------------------------------------------------- load
fetch()
roi_ll = gpd.read_file(ROI_SHP)
roi = roi_ll.to_crs(DST_CRS)
minx, miny, maxx, maxy = roi.total_bounds
PAD = 1500.0
EXTENT = (minx - PAD, maxx + PAD, miny - PAD, maxy + PAD)

# (a) precipitation ------------------------------------------------------
wc = SRC["worldclim"]
with rasterio.open(wc["path"]) as s:
    assert s.count >= 12
    win = from_bounds(BBOX_NB09[0] - .05, BBOX_NB09[1] - .05,
                      BBOX_NB09[2] + .05, BBOX_NB09[3] + .05, s.transform)
    p_ll = s.read(12, window=win).astype("float32")
    p_tr = s.window_transform(win)
    p_bounds = rasterio.windows.bounds(win, s.transform)
    p_crs, p_native_res_deg = s.crs, s.res[0]
p_ll[p_ll < -1e30] = np.nan
# WorldClim is a land-only product: one sea-dominated cell on the Mediterranean coast inside the
# ROI is masked in the source (verified also absent from the PSE tile). It is left as no-data.
# Display grid is finer than the ~1 km native cell only so that the detailed ROI boundary is
# filled cleanly; NEAREST resampling keeps the native WorldClim cell structure intact and adds
# no spatial detail that is not in the source.
PREC_RES = 200.0
prec, prec_tr = to_utm(p_ll, p_tr, p_crs, p_bounds, PREC_RES, method=Resampling.nearest)
prec_c = clip_to_roi(prec, prec_tr, roi, all_touched=True)

# (b) elevation ----------------------------------------------------------
dem = SRC["srtm"]
h = np.fromfile(dem["path"], dtype=">i2").reshape(3601, 3601).astype("float32")
h[h == -32768] = np.nan                                   # SRTM void
dem_tr_full = from_origin(34.0 - 0.5/3600, 32.0 + 0.5/3600, 1/3600, 1/3600)
win = from_bounds(BBOX_NB09[0] - .02, BBOX_NB09[1] - .02,
                  BBOX_NB09[2] + .02, BBOX_NB09[3] + .02, dem_tr_full)
r0, c0 = int(win.row_off), int(win.col_off)
d_ll = h[r0:r0 + int(win.height), c0:c0 + int(win.width)]
d_tr = rasterio.windows.transform(win, dem_tr_full)
d_bounds = rasterio.windows.bounds(win, dem_tr_full)
DEM_RES = 30.0
elev, elev_tr = to_utm(d_ll, d_tr, "EPSG:4326", d_bounds, DEM_RES)
elev_c = clip_to_roi(elev, elev_tr, roi, all_touched=True)

# ------------------------------------------------------------------ stats
def stats(a):
    v = a[np.isfinite(a)]
    return dict(n_cells=int(v.size), minimum=float(v.min()), p25=float(np.percentile(v, 25)),
                median=float(np.median(v)), mean=float(v.mean()),
                p75=float(np.percentile(v, 75)), maximum=float(v.max()))
def edge_means(a):
    rows = [np.nanmean(r) for r in a]
    ok = [i for i, r in enumerate(rows) if np.isfinite(r)]
    return float(rows[ok[0]]), float(rows[ok[-1]])
def native_stats(arr, transform):
    m = geometry_mask(roi_ll.geometry, out_shape=arr.shape, transform=transform, invert=True)
    return stats(np.where(m, arr, np.nan))
S = {"precipitation_mm": stats(prec_c), "elevation_m": stats(elev_c),
     "precipitation_mm_native_30s_roi": native_stats(p_ll, p_tr),
     "elevation_m_native_1arcsec_roi": native_stats(d_ll, d_tr)}
S["precipitation_mm"]["north_edge"], S["precipitation_mm"]["south_edge"] = edge_means(prec_c)
S["elevation_m"]["north_edge"], S["elevation_m"]["south_edge"] = edge_means(elev_c)

# ------------------------------------------------------------------- plot
matplotlib.rcParams.update({"font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42,
                            "axes.linewidth": 0.7, "savefig.facecolor": "white"})
def decorate(ax, label, title):
    for geom in roi.geometry:
        ax.add_patch(MplPoly(np.asarray(geom.exterior.coords), closed=True,
                             fill=False, ec="#111111", lw=0.9, zorder=5))
    ax.set_xlim(EXTENT[0], EXTENT[1]); ax.set_ylim(EXTENT[2], EXTENT[3])
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9.5, pad=6)
    ax.text(0.02, 0.982, label, transform=ax.transAxes, fontsize=11, fontweight="bold",
            va="top", ha="left", zorder=6,
            bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.8))
    ax.set_xticks(np.arange(np.ceil(EXTENT[0]/5000)*5000, EXTENT[1], 5000))
    ax.set_yticks(np.arange(np.ceil(EXTENT[2]/5000)*5000, EXTENT[3], 10000))
    ax.set_xticklabels(["%d" % (v/1000) for v in ax.get_xticks()], fontsize=7)
    ax.set_yticklabels(["%d" % (v/1000) for v in ax.get_yticks()], fontsize=7)
    ax.set_xlabel("UTM 36N easting (km)", fontsize=8)
    ax.set_ylabel("UTM 36N northing (km)", fontsize=8)
    ax.tick_params(length=2.5, width=0.6)
    # scale bar, lower right where the ROI leaves clear space
    km = 10
    x1 = EXTENT[1] - 0.045*(EXTENT[1]-EXTENT[0]); x0 = x1 - km*1000
    y0 = EXTENT[2] + 0.030*(EXTENT[3]-EXTENT[2])
    ax.plot([x0, x1], [y0, y0], color="#111111", lw=2.6, solid_capstyle="butt", zorder=7)
    for xt in (x0, x1):
        ax.plot([xt, xt], [y0, y0 + 700], color="#111111", lw=1.0, zorder=7)
    ax.text((x0+x1)/2, y0 + 1100, "%d km" % km, ha="center", fontsize=7.5, zorder=7,
            bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.2))
    # north arrow
    ax.annotate("N", xy=(0.93, 0.955), xytext=(0.93, 0.855), xycoords="axes fraction",
                ha="center", fontsize=9, zorder=6,
                arrowprops=dict(arrowstyle="-|>", color="#111111", lw=1.1))

fig, axes = plt.subplots(1, 2, figsize=(7.2, 5.6), layout="constrained")
ex_p = (prec_tr.c, prec_tr.c + prec_c.shape[1]*prec_tr.a,
        prec_tr.f + prec_c.shape[0]*prec_tr.e, prec_tr.f)
im0 = axes[0].imshow(prec_c, extent=ex_p, cmap="YlGnBu", origin="upper",
                     interpolation="nearest", rasterized=True)
decorate(axes[0], "(a)", "Mean annual precipitation")
cb0 = fig.colorbar(im0, ax=axes[0], orientation="horizontal", pad=0.02, shrink=0.88, aspect=26)
cb0.set_label("mm yr$^{-1}$ (WorldClim 2.1 bio_12)", fontsize=8); cb0.ax.tick_params(labelsize=7.5)

ex_e = (elev_tr.c, elev_tr.c + elev_c.shape[1]*elev_tr.a,
        elev_tr.f + elev_c.shape[0]*elev_tr.e, elev_tr.f)
im1 = axes[1].imshow(elev_c, extent=ex_e, cmap="cividis", origin="upper",
                     interpolation="nearest", rasterized=True)
decorate(axes[1], "(b)", "Elevation")
cb1 = fig.colorbar(im1, ax=axes[1], orientation="horizontal", pad=0.02, shrink=0.88, aspect=26)
cb1.set_label("m above sea level (SRTM 30 m)", fontsize=8); cb1.ax.tick_params(labelsize=7.5)

os.makedirs(OUT, exist_ok=True)
base = os.path.join(OUT, "study_area_environment")
fig.savefig(base + ".png", dpi=400, bbox_inches="tight")
fig.savefig(base + ".pdf", bbox_inches="tight")
plt.close(fig)

prov = dict(
  roi_geometry=os.path.relpath(ROI_SHP, ROOT),
  roi_crs=str(roi_ll.crs), roi_area_km2=float(roi.area.sum()/1e6),
  target_crs=DST_CRS, display_res_m=dict(precipitation=PREC_RES, elevation=DEM_RES),
  worldclim=dict(url=wc["url"], file=os.path.relpath(wc["path"], ROOT), band=12,
                 variable="bio_12 annual precipitation (mm)", native_res_deg=float(p_native_res_deg),
                 citation="Fick & Hijmans (2017)"),
  srtm=dict(url=dem["url"], file=os.path.relpath(dem["path"], ROOT),
            native_res="1 arc-second (~30 m)", citation="Farr et al. (2007)"),
  nb09_bbox_wgs84=BBOX_NB09, statistics_over_roi_polygon=S,
  outputs=[os.path.relpath(base + e, ROOT) for e in (".png", ".pdf")])
os.makedirs(TAB, exist_ok=True)
with open(os.path.join(TAB, "nb22_study_area_environment_provenance.json"), "w") as f:
    json.dump(prov, f, indent=1)
print(json.dumps(S, indent=1))
print("\nwrote %s.png / .pdf" % base)
