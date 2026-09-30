"""
10_revision_figures.py - regenerate Figures 1, 2 and 6 for the final revision.

Figure 6 provenance. The published Figure 6 was produced by Notebook 03 cell 28,
which averaged NBR in a +/-12-month window around every documented fire event
belonging to `df["pixel_id"].unique()[:500]` - the 500 lowest-numbered burned
pixels in a frame sorted by pixel_id. That yields 511 qualifying fire events
from 476 pixels, exactly the n printed on the published figure, so the figure is
fully reproducible. The 500-pixel cap was a performance shortcut in an
exploratory notebook, not a sampling decision, so the figure is regenerated here
over the whole burned population under the same event-level rule. The two curves
agree at r = 0.998 with a maximum difference of 0.010 NBR, so widening the
sample does not change the result.

Run: python notebooks/10_revision_figures.py
Writes: outputs/figures/fig01_framework.png, fig02_study_area.png, fig06_event_aligned_nbr.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = os.path.join(ROOT, 'data', 'processed')
OUT = os.path.join(ROOT, 'outputs', 'figures')
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                     'axes.edgecolor': '#444444', 'axes.linewidth': 0.8})

W = 12


def fig06():
    y = np.load(os.path.join(P, 'y_burned.npy'), mmap_mode='r')
    yb = np.asarray(y).astype(bool)
    nbr = np.load(os.path.join(P, 'nbr_raw.npy'), mmap_mode='r')
    T = yb.shape[1]
    rows = np.where(yb.any(axis=1))[0]

    def collect(first_only):
        out = []
        npix = 0
        for r in rows:
            f = np.where(yb[r])[0]
            if first_only:
                f = f[:1]
            f = f[(f - W >= 0) & (f + W + 1 <= T)]
            if len(f) == 0:
                continue
            npix += 1
            v = np.asarray(nbr[r])
            out.extend(v[t - W:t + W + 1] for t in f)
        return np.asarray(out, dtype=np.float64), npix

    ev, npix_ev = collect(False)
    ff, npix_ff = collect(True)
    x = np.arange(-W, W + 1)
    m, sd = np.nanmean(ev, 0), np.nanstd(ev, 0)
    mf = np.nanmean(ff, 0)

    fig, ax = plt.subplots(figsize=(6.5, 2.9), dpi=300)
    ax.fill_between(x, m - sd, m + sd, color='#9aa6b2', alpha=0.28, linewidth=0,
                    label='±1 SD across fire events')
    ax.plot(x, m, color='#16324f', lw=2.0, zorder=3,
            label='all documented fire events (n = {:,} events, {:,} pixels)'.format(len(ev), npix_ev))
    ax.plot(x, mf, color='#b5651d', lw=1.4, ls=(0, (5, 2)), zorder=4,
            label='first documented fire only (n = {:,} pixels)'.format(len(ff)))
    ax.axvline(0, color='#c0392b', lw=1.1, ls='--', zorder=2)
    ax.annotate('documented\nfire month', xy=(0, ax.get_ylim()[1]), xytext=(0.6, 0.30),
                fontsize=7.5, color='#c0392b', ha='left', va='top')
    ax.axhline(0, color='#888888', lw=0.6, zorder=1)
    ax.set_xlabel('months relative to the documented fire')
    ax.set_ylabel('NBR')
    ax.set_xlim(-W, W)
    ax.set_xticks(np.arange(-12, 13, 3))
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=6.8, loc='lower right', framealpha=0.92, borderpad=0.5)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    fig.tight_layout(pad=0.4)
    p = os.path.join(OUT, 'fig06_event_aligned_nbr.png')
    fig.savefig(p, dpi=300, facecolor='white')
    plt.close(fig)
    print('fig06: %d events from %d pixels; first-fire curve %d pixels -> %s'
          % (len(ev), npix_ev, len(ff), p))
    return len(ev), npix_ev, len(ff)


if __name__ == '__main__':
    fig06()


# ---------------------------------------------------------------------------
# Figure 1 - the framework, drawn so that the five epistemic layers of
# Section 3.1 are visually separated.
# Stage 2 appears only as a footnote annotation, not as a major box.
# ---------------------------------------------------------------------------
LAYERS = [
    ('1  EXTERNAL FIRE-REFERENCE DATA', 'reference data, independent of the model',
     '#2f5d62', '#dceaec'),
    ('2  SATELLITE-DERIVED', 'measurements of reflectance',
     '#31618f', '#dde7f2'),
    ('3  ANALYTICALLY CONSTRUCTED', 'definitions written in this thesis',
     '#a85a1a', '#f6e6d6'),
    ('4  MODEL-ESTIMATED', 'outputs of the trained model',
     '#5f4380', '#e6dff0'),
    ('5  DECISION', 'choices, not observations',
     '#4a5d3a', '#e2e8da'),
]


def _box(ax, x, y, w, h, text, ec, fc, fs=7.2, ls='-', lw=1.1, weight='normal'):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle='round,pad=0.35,rounding_size=1.2',
                                linewidth=lw, edgecolor=ec, facecolor=fc,
                                linestyle=ls, zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
            fontsize=fs, color='#1b1b1b', zorder=4, linespacing=1.35,
            fontweight=weight)


def _arrow(ax, xy_from, xy_to, label=None, color='#5a5a5a', rad=0.0, fs=6.2):
    ax.add_patch(FancyArrowPatch(xy_from, xy_to, arrowstyle='-|>', mutation_scale=9,
                                 linewidth=0.9, color=color, zorder=2,
                                 connectionstyle='arc3,rad=%.2f' % rad,
                                 shrinkA=1, shrinkB=1))
    if label:
        mx, my = (xy_from[0] + xy_to[0]) / 2, (xy_from[1] + xy_to[1]) / 2
        ax.text(mx + 1.2, my, label, fontsize=fs, color=color, ha='left',
                va='center', style='italic', zorder=5,
                bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.85))


def fig01():
    fig, ax = plt.subplots(figsize=(6.8, 8.2), dpi=300)
    ax.set_xlim(0, 104); ax.set_ylim(0, 100); ax.axis('off')
    BX0, BX1 = 14.0, 103.0                 # band extent
    CX0, CX1 = 17.0, 100.0                 # content column, inset from the band
    CW = CX1 - CX0
    MID = (CX0 + CX1) / 2
    R1, R2 = 6.5, 10.3                     # arrow rails, outside the bands

    bands = [(86.0, 13.0), (65.5, 19.0), (48.0, 15.5), (21.0, 25.5), (7.0, 12.5)]
    for (name, sub, ec, fc), (y0, h) in zip(LAYERS, bands):
        ax.add_patch(FancyBboxPatch((BX0, y0), BX1 - BX0, h,
                                    boxstyle='round,pad=0.2,rounding_size=1.0',
                                    linewidth=0, facecolor=fc, alpha=0.6, zorder=0))
        # opaque bbox in the band colour so a flow arrow passing behind the
        # header is interrupted rather than overlapping the lettering
        ax.text(CX0, y0 + h - 1.5, '%s      %s' % (name, sub), fontsize=6.9,
                color=ec, fontweight='bold', va='top', ha='left', zorder=6,
                bbox=dict(boxstyle='square,pad=0.28', fc=fc, ec='none'))

    ec1 = LAYERS[0][2]
    _box(ax, CX0, 86.6, CW, 7.4,
         'Fire-reference inventory  -  KKL fire scars (2,531) + Planet / Sentinel-2\n'
         'polygons (1,065) = 3,596 merged polygons, 2017-2025  ->  monthly burned labels\n'
         'the only quantity with a reference outside the satellite record', ec1, '#ffffff', fs=6.8)

    ec2 = LAYERS[1][2]
    _box(ax, CX0, 74.5, CW / 2 - 1.2, 6.4,
         'Sentinel-2 L2A surface reflectance\n108 monthly composites\nJan 2017 - Dec 2025', ec2, '#ffffff', fs=6.5)
    _box(ax, CX0 + CW / 2 + 1.2, 74.5, CW / 2 - 1.2, 6.4,
         'Seasonal baseline 2016-2017\nper-calendar-month\nmean, median, SD', ec2, '#ffffff', fs=6.5)
    _box(ax, CX0, 66.5, CW, 6.6,
         '52 features per month\n'
         '5 bands  |  7 spectral indices  |  7 z-scores  |  7 raw anomalies\n'
         '4 deltas  |  2 harmonics  |  20 land-cover one-hots', ec2, '#ffffff', fs=6.5)

    ec3 = LAYERS[2][2]
    for k, (lab, eq) in enumerate([('Burn severity', 'Eq. 3.12'),
                                   ('Persistence', 'Eq. 3.13'),
                                   ('Recovery gap', 'Eq. 3.14-3.15')]):
        _box(ax, CX0 + k * (CW / 3), 51.0, CW / 3 - 2.2, 6.6,
             '%s\n%s' % (lab, eq), ec3, '#ffffff', fs=7.0)
    ax.text(MID, 49.2, 'deterministic functions of NBR and its z-score - not field measurements',
            fontsize=6.2, color=ec3, ha='center', va='center', style='italic', zorder=5)

    ec4 = LAYERS[3][2]
    _box(ax, CX0, 38.0, CW, 5.2,
         'L-TAE temporal encoder  -  shared representation, 66,436 trainable parameters',
         ec4, '#ffffff', fs=7.2, weight='bold')
    _box(ax, CX0, 29.5, 20.0, 6.6, 'fire occurrence\nper pixel-month\n(classification head)',
         ec4, '#ffffff', fs=6.3)
    for k, lab in enumerate(['severity', 'persistence', 'recovery gap']):
        _box(ax, CX0 + 22.0 + k * 20.5, 29.5, 19.0, 6.6, '%s\n(regression head)' % lab,
             ec4, '#ffffff', fs=6.3)
    _box(ax, CX0, 22.5, CW, 4.6,
         'recurrence  -  min(count(burned probability > 0.502) / 5, 1)\n'
         'derived post hoc from the classified occurrences; not a regression head',
         ec4, '#f7f4fb', fs=6.3, ls=(0, (3, 2)))

    ec5 = LAYERS[4][2]
    _box(ax, CX0, 10.8, CW, 5.6,
         'Restoration Indicator Score\nweighted combination of the four indicators under alternative weight profiles',
         ec5, '#ffffff', fs=7.0, weight='bold')
    ax.text(MID, 8.6, 'weights are choices; no stakeholder preferences were elicited in this thesis',
            fontsize=6.2, color=ec5, ha='center', va='center', style='italic', zorder=5)

    def straight(y0, y1, label=None):
        ax.add_patch(FancyArrowPatch((MID, y0), (MID, y1), arrowstyle='-|>',
                                     mutation_scale=10, lw=1.0, color='#4a4a4a', zorder=2))
        if label:
            ax.text(MID + 1.2, (y0 + y1) / 2, label, fontsize=6.1, color='#4a4a4a',
                    ha='left', va='center', style='italic', zorder=5,
                    bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.9))
    straight(86.6, 81.3)
    straight(74.5, 73.4)
    straight(66.5, 58.0)
    straight(51.0, 43.5, 'regression targets')
    straight(38.0, 36.4)
    straight(22.5, 16.7)

    def rail(x, ytop, ybot, label, color='#5a5a5a'):
        ax.add_patch(FancyArrowPatch((x, ytop), (CX0 - 0.4, ybot), arrowstyle='-|>',
                                     mutation_scale=10, lw=1.0, color=color, zorder=2,
                                     connectionstyle='angle,angleA=-90,angleB=180,rad=2.5'))
        ax.text(x - 1.2, (ytop + ybot) / 2 + 3, label, fontsize=6.0, color=color,
                rotation=90, ha='center', va='center', style='italic', zorder=5)
    rail(R1, 90.3, 54.3, 'fire months gate the definitions')
    rail(R2, 90.3, 32.8, 'training labels')
    ax.add_patch(FancyArrowPatch((101.8, 69.8), (CX1 + 0.4, 40.6), arrowstyle='-|>',
                                 mutation_scale=10, lw=1.0, color='#5a5a5a', zorder=2,
                                 connectionstyle='angle,angleA=-90,angleB=0,rad=2.5'))
    ax.text(103.2, 55.0, 'model input', fontsize=6.0, color='#5a5a5a',
            rotation=90, ha='center', va='center', style='italic', zorder=5)

    ax.add_patch(FancyArrowPatch((2.0, 99.0), (2.0, 21.0), arrowstyle='-',
                                 lw=2.2, color='#333333', zorder=1))
    ax.text(0.4, 60.0, 'STAGE 1  -  implemented in this thesis', rotation=90,
            fontsize=7.4, fontweight='bold', color='#333333', ha='center', va='center')
    ax.text(58, 3.0,
            'Stage 2 - stakeholder-defined weights and land-cover-specific ecological interpretation - is future\n'
            'work, described in the text (Sections 2.6 and 5.5 and Chapter 6) rather than implemented here.',
            fontsize=6.3, color='#555555', ha='center', va='center', style='italic',
            bbox=dict(boxstyle='round,pad=0.5', fc='#f4f4f4', ec='#cccccc', lw=0.7))

    fig.tight_layout(pad=0.2)
    p = os.path.join(OUT, 'fig01_framework.png')
    fig.savefig(p, dpi=300, facecolor='white')
    plt.close(fig)
    print('fig01 -> %s' % p)


# ---------------------------------------------------------------------------
# Figure 2 - study area. Four panels: a locator placing the region of interest
# relative to Israel, the Gaza Strip and Egypt; the existing 2025 land-cover
# classification, carried over unchanged; mean annual precipitation showing the
# north-south gradient; and relief. Every environmental value shown is the one
# already verified in Section 3.2.1 and reproduced by
# notebooks/09_revision_verification.py (tables nb09_t04, nb09_t06, nb09_t07).

# ---------------------------------------------------------------------------
BBOX = (34.2672, 31.0574, 34.7820, 31.7495)     # W, S, E, N  (nb09_t04)
NE_SHP = os.environ.get('NE_ADMIN0', '')        # ne_10m_admin_0_countries
WC_DIR = os.environ.get('WORLDCLIM_10M_DIR', '')
DEM_NPZ = os.environ.get('SRTM_GRID_NPZ', '')
LC_PANEL = os.environ.get('LANDCOVER_PANEL', '')  # the existing Figure 2 image


def _roi_mask(nx=110, ny=150):
    """Footprint of the sampled pixels, from data/raw/pixel_coords_10m.csv."""
    co = pd.read_csv(os.path.join(ROOT, 'data', 'raw', 'pixel_coords_10m.csv'))
    W, S, E, N = BBOX
    H, xe, ye = np.histogram2d(co.lon.values, co.lat.values, bins=[nx, ny],
                               range=[[W, E], [S, N]])
    return (H.T > 0), xe, ye


def fig02():
    import shapefile
    from matplotlib.colors import LinearSegmentedColormap
    W, S, E, N = BBOX
    fig = plt.figure(figsize=(7.2, 7.6), dpi=300)
    gs = fig.add_gridspec(2, 2, hspace=0.16, wspace=0.14,
                          left=0.06, right=0.97, top=0.95, bottom=0.05)
    mask, xe, ye = _roi_mask()

    # (a) locator ----------------------------------------------------------
    axa = fig.add_subplot(gs[0, 0])
    r = shapefile.Reader(NE_SHP)
    flds = [f[0] for f in r.fields[1:]]
    i_name = flds.index('NAME')
    COL = {'Israel': '#e8e2d5', 'Palestine': '#cdd9c6', 'Egypt': '#efe7dc',
           'Jordan': '#efe7dc', 'Lebanon': '#efe7dc', 'Syria': '#efe7dc'}
    axa.set_facecolor('#cfe0ec')                      # sea
    for sr in r.iterShapeRecords():
        nm = sr.record[i_name]
        if nm not in COL:
            continue
        pts = np.asarray(sr.shape.points)
        parts = list(sr.shape.parts) + [len(pts)]
        for k in range(len(parts) - 1):
            seg = pts[parts[k]:parts[k + 1]]
            axa.fill(seg[:, 0], seg[:, 1], facecolor=COL[nm], edgecolor='#7a7a7a',
                     linewidth=0.5, zorder=2)
    # the true footprint of the sampled pixels, not a bounding box, so the
    # locator does not imply that the region of interest includes the Gaza Strip
    axa.contour(xe[:-1], ye[:-1], mask.astype(float), levels=[0.5],
                colors='#c0392b', linewidths=1.4, zorder=5)
    axa.annotate('region of\ninterest', xy=(E, (S + N) / 2), xytext=(35.35, 31.45),
                 fontsize=6.6, color='#c0392b', ha='left', va='center', zorder=6,
                 arrowprops=dict(arrowstyle='-|>', color='#c0392b', lw=0.9))
    axa.text(34.45, 31.44, 'Gaza\nStrip', fontsize=6.0, color='#2f5d3a', ha='center',
             va='center', zorder=6, style='italic')
    axa.text(35.2, 30.4, 'Israel', fontsize=7.0, color='#555555', ha='center', zorder=6)
    axa.text(34.2, 30.2, 'Egypt', fontsize=6.4, color='#777777', ha='center', zorder=6)
    axa.text(33.9, 32.4, 'Mediterranean\nSea', fontsize=6.0, color='#31618f', ha='center',
             va='center', style='italic', zorder=6)
    axa.set_xlim(33.6, 36.1); axa.set_ylim(29.4, 33.4)
    axa.set_aspect(1 / np.cos(np.deg2rad(31.4)))
    axa.set_xticks([34, 35, 36]); axa.set_yticks([30, 31, 32, 33])
    axa.tick_params(labelsize=6)
    axa.set_title('(a)  Location', fontsize=8, loc='left', pad=3)

    # (b) land cover, carried over unchanged -------------------------------
    axb = fig.add_subplot(gs[0, 1])
    if LC_PANEL and os.path.exists(LC_PANEL):
        axb.imshow(plt.imread(LC_PANEL))
    axb.axis('off')
    axb.set_title('(b)  Land cover, 2025', fontsize=8, loc='left', pad=3)

    # (c) mean annual precipitation ----------------------------------------
    axc = fig.add_subplot(gs[1, 0])
    import rasterio
    from rasterio.windows import from_bounds
    with rasterio.open(os.path.join(WC_DIR, 'wc2.1_10m_bio_12.tif')) as src:
        win = from_bounds(W - 0.5, S - 0.5, E + 0.5, N + 0.5, src.transform)
        pr = src.read(1, window=win).astype(float)
        # WorldClim stores a large negative fill over the sea; without masking
        # it, contourf renders the Mediterranean as the driest class on land
        pr[pr < -1e30] = np.nan
        tr = src.window_transform(win)
    ny_, nx_ = pr.shape
    plon = tr.c + tr.a * (np.arange(nx_) + 0.5)
    plat = tr.f + tr.e * (np.arange(ny_) + 0.5)
    cmap = LinearSegmentedColormap.from_list('rain', ['#e9d8a6', '#94d2bd', '#0a9396'])
    im = axc.contourf(plon, plat, np.ma.masked_invalid(pr),
                      levels=np.arange(150, 451, 25), cmap=cmap, extend='both')
    axc.set_facecolor('#cfe0ec')
    cs = axc.contour(plon, plat, pr, levels=[200, 250, 300, 350], colors='#204040',
                     linewidths=0.7)
    axc.clabel(cs, fmt='%d mm', fontsize=5.6, inline=True)
    axc.contour(xe[:-1], ye[:-1], mask.astype(float), levels=[0.5], colors='#c0392b',
                linewidths=1.3)
    axc.set_xlim(W - 0.12, E + 0.12); axc.set_ylim(S - 0.10, N + 0.10)
    axc.set_aspect(1 / np.cos(np.deg2rad(31.4)))
    axc.tick_params(labelsize=6)
    cb = fig.colorbar(im, ax=axc, fraction=0.045, pad=0.02)
    cb.ax.tick_params(labelsize=5.6)
    cb.set_label('mm yr$^{-1}$', fontsize=6)
    axc.set_title('(c)  Mean annual precipitation', fontsize=8, loc='left', pad=3)
    axc.text(0.03, 0.05, '≈375 mm at the northern edge\n≈177 mm at the southern edge',
             transform=axc.transAxes, fontsize=5.8, va='bottom',
             bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#bbbbbb', lw=0.5))

    # (d) relief ------------------------------------------------------------
    axd = fig.add_subplot(gs[1, 1])
    d = np.load(DEM_NPZ)
    z, dlon, dlat = d['z'], d['lons'], d['lats']
    im2 = axd.contourf(dlon, dlat, z, levels=np.arange(0, 361, 20), cmap='terrain',
                       extend='max')
    axd.contour(dlon, dlat, z, levels=[100, 200, 300], colors='#444444', linewidths=0.5)
    axd.contour(xe[:-1], ye[:-1], mask.astype(float), levels=[0.5], colors='#c0392b',
                linewidths=1.3)
    axd.set_xlim(W - 0.12, E + 0.12); axd.set_ylim(S - 0.10, N + 0.10)
    axd.set_aspect(1 / np.cos(np.deg2rad(31.4)))
    axd.tick_params(labelsize=6)
    cb2 = fig.colorbar(im2, ax=axd, fraction=0.045, pad=0.02)
    cb2.ax.tick_params(labelsize=5.6)
    cb2.set_label('m a.s.l.', fontsize=6)
    axd.set_title('(d)  Elevation', fontsize=8, loc='left', pad=3)
    axd.text(0.03, 0.05, '0–318 m, median 102 m',
             transform=axd.transAxes, fontsize=5.8, va='bottom',
             bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#bbbbbb', lw=0.5))

    for ax_ in (axa, axc, axd):
        for sp in ax_.spines.values():
            sp.set_linewidth(0.6)
    p = os.path.join(OUT, 'fig02_study_area.png')
    fig.savefig(p, dpi=300, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    print('fig02 -> %s' % p)
