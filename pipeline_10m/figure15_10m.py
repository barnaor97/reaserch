# -*- coding: utf-8 -*-
"""
Figure 15 — two prioritization hierarchies over the same burned spatial units,
regenerated from the FINAL 10 m analysis.

The published Figure 15 image was rendered by the earlier 30 m implementation
(19_finalize_polygon_thesis_outputs.py) while the Section 4.13 prose reports the
10 m results. This script rebuilds the same figure from the 10 m outputs so that
figure and text describe one analysis.

Nothing scientific is redefined here. The hidden-unit rule is read straight from
polygon_10m.py: the within-class top decile is ceil(n/10) units by
within_landcover_RIS_rank, and a unit is "hidden" if it is in that set but falls
below the population's global top-decile cut (0.90 quantile of mean_RIS_balanced).
Classes with fewer than SMALL_N_UNITS units are flagged and excluded from the
headline hidden-unit total, exactly as in polygon_10m.py.

Inputs  (final 10 m only):
    outputs/polygon_10m_candidate/reference_units_final.csv
    outputs/polygon_10m_candidate/model_units_final.csv
    outputs/polygon_10m_candidate/tables/p10_t09_global_vs_within_landcover_counts.csv   (QA target)
    outputs/polygon_10m_candidate/AGREEMENT_10m.json                                     (QA target)

Outputs:
    outputs/polygon_10m_candidate/maps/figure15_10m.{png,pdf}
    outputs/polygon_10m_candidate/maps/FIGURE15_PROVENANCE.json

Run:
    python notebooks/prod10m/figure15_10m.py
"""
import hashlib
import json
import math
import os
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

def _find_root(start):
    """Walk up until a directory holding both outputs/ and data/ is found, so the
    script works whether it sits in notebooks/prod10m/ or in pipeline_10m/."""
    d = os.path.dirname(os.path.abspath(start))
    for _ in range(6):
        if os.path.isdir(os.path.join(d, "outputs")) and os.path.isdir(os.path.join(d, "data")):
            return d
        d = os.path.dirname(d)
    return os.path.dirname(os.path.dirname(os.path.abspath(start)))

ROOT = os.environ.get("THESIS_ROOT") or _find_root(__file__)
FIN = os.path.join(ROOT, "outputs", "polygon_10m_candidate")
TAB = os.path.join(FIN, "tables")
OUT = os.path.join(FIN, "maps")
os.makedirs(OUT, exist_ok=True)

# ---- constants, identical to polygon_10m.py -------------------------------
SMALL_N_UNITS = 20
LC_NAMES = {10: "forest", 11: "orchards", 20: "shrubland", 30: "grassland", 40: "cropland",
            41: "covered agriculture", 50: "built-up", 60: "bare", 61: "fallow", 80: "water"}
LC_ORDER = [10, 11, 20, 30, 40, 41, 50, 60, 61, 80]
CLS_ORDER = [LC_NAMES[c] for c in LC_ORDER] + ["unclassified / other code"]
POP_LABEL = {"reference": "A. Reference-defined burned areas",
             "model": "B. Model-detected burned areas"}
_cmap = plt.get_cmap("tab10")
CLASS_COLOR = {LC_NAMES[c]: _cmap(i) for i, c in enumerate(LC_ORDER)}
CLASS_COLOR["unclassified / other code"] = (0.6, 0.6, 0.6, 1.0)

matplotlib.rcParams.update({"figure.dpi": 110, "savefig.dpi": 400, "font.size": 9,
                            "axes.spines.top": False, "axes.spines.right": False,
                            "pdf.fonttype": 42, "svg.fonttype": "none"})


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


SRC = {"reference": os.path.join(FIN, "reference_units_final.csv"),
       "model": os.path.join(FIN, "model_units_final.csv")}
T09 = os.path.join(TAB, "p10_t09_global_vs_within_landcover_counts.csv")
AGJ = os.path.join(FIN, "AGREEMENT_10m.json")

UNITS = {p: pd.read_csv(SRC[p]) for p in ("reference", "model")}
T09_DF = pd.read_csv(T09)
AG = json.load(open(AGJ))

# ---- recompute the hierarchy exactly as polygon_10m.py does ---------------
computed_rows, hidden_ids, marks = [], {}, {}
for pop in ("reference", "model"):
    u = UNITS[pop]
    cut = u.mean_RIS_balanced.quantile(0.9)
    ids, per_class = [], {}
    for cname, sub in u.groupby("land_cover_name"):
        n = len(sub)
        small = n < SMALL_N_UNITS
        tw = sub.nsmallest(int(math.ceil(n / 10.0)), "within_landcover_RIS_rank")
        below = tw[tw.mean_RIS_balanced < cut]
        computed_rows.append(dict(population=pop, land_cover_name=cname, n_units=n,
                                  small_n_class_flag=small,
                                  median_RIS=float(sub.mean_RIS_balanced.median()),
                                  within_class_top_decile_n=len(tw),
                                  also_in_global_top_decile_n=int((tw.mean_RIS_balanced >= cut).sum()),
                                  high_within_class_but_below_global_cut_n=len(below)))
        per_class[cname] = dict(top_ids=set(tw.unit_id), hidden_ids=set(below.unit_id))
        if not small:
            ids += list(below.unit_id)
    hidden_ids[pop] = ids
    marks[pop] = dict(cut=float(cut), per_class=per_class)

COMP = pd.DataFrame(computed_rows)

# ---- QA gate: the figure must reproduce the canonical table ---------------
key = ["population", "land_cover_name"]
num = ["n_units", "within_class_top_decile_n", "also_in_global_top_decile_n",
       "high_within_class_but_below_global_cut_n"]
m = T09_DF.merge(COMP, on=key, suffixes=("_table", "_fig"))
assert len(m) == len(T09_DF) == len(COMP), "class/population sets differ from p10_t09"
qa = []
for c in num:
    bad = m[m[c + "_table"] != m[c + "_fig"]]
    qa.append(dict(quantity=c, rows=len(m), mismatches=int(len(bad))))
    assert bad.empty, "%s differs from p10_t09:\n%s" % (c, bad[key + [c + "_table", c + "_fig"]])
for c in ["median_RIS"]:
    d = float(np.nanmax(np.abs(m[c + "_table"].to_numpy() - m[c + "_fig"].to_numpy())))
    qa.append(dict(quantity=c, rows=len(m), max_abs_diff=d))
    assert d < 1e-9, "median_RIS differs from p10_t09 (max %g)" % d
for pop in ("reference", "model"):
    got, want = len(hidden_ids[pop]), int(AG["hidden_units_%s" % pop])
    qa.append(dict(quantity="hidden_units_%s" % pop, computed=got, agreement_json=want))
    assert got == want, "hidden_units_%s: %d vs AGREEMENT_10m %d" % (pop, got, want)
print("QA passed: figure quantities reproduce p10_t09 and AGREEMENT_10m exactly")

# ---- figure ---------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14.2, 7.0), sharex=True, layout="constrained")
displayed = []
for ax, pop in zip(axes, ["reference", "model"]):
    u = UNITS[pop]
    classes = [c for c in CLS_ORDER if (u.land_cover_name == c).sum() > 0]
    ypos = np.arange(len(classes))[::-1]
    gcut = marks[pop]["cut"]
    rng = np.random.default_rng(0)            # deterministic jitter
    for yy, c in zip(ypos, classes):
        s = u[u.land_cover_name == c]
        top_ids = marks[pop]["per_class"][c]["top_ids"]
        hid_ids = marks[pop]["per_class"][c]["hidden_ids"]
        top_within = s.unit_id.isin(top_ids)
        obscured = s.unit_id.isin(hid_ids)
        both = top_within & ~obscured
        jit = rng.normal(0, 0.10, len(s))
        ax.scatter(s.mean_RIS_balanced[~top_within], yy + jit[(~top_within).to_numpy()], s=7,
                   color="#b9c2c9", alpha=0.75, linewidths=0, zorder=2)
        ax.scatter(s.mean_RIS_balanced[both], yy + jit[both.to_numpy()], s=17, color="#d7191c",
                   alpha=0.9, linewidths=0, zorder=4)
        ax.scatter(s.mean_RIS_balanced[obscured], yy + jit[obscured.to_numpy()], s=17,
                   color="#2c7bb6", alpha=0.95, linewidths=0, zorder=4)
        ax.plot([s.mean_RIS_balanced.median()] * 2, [yy - 0.28, yy + 0.28],
                color="#33414d", lw=1.8, zorder=5)
        displayed.append(dict(population=pop, land_cover_name=c, n_units=int(len(s)),
                              plotted_grey=int((~top_within).sum()), plotted_red=int(both.sum()),
                              plotted_blue=int(obscured.sum()),
                              small_n_class_flag=bool(s.small_n_class_flag.iloc[0]),
                              median_RIS=float(s.mean_RIS_balanced.median())))
    ax.axvline(gcut, color="#444444", lw=1.1, ls="--", zorder=1)
    ax.text(gcut, len(classes) - 0.35, " global top decile", fontsize=7.6, color="#444444", va="center")
    ax.set_yticks(ypos)
    ax.set_yticklabels(["%s%s\nn = %d" % (c, " *" if bool(u[u.land_cover_name == c].small_n_class_flag.iloc[0]) else "",
                                          int((u.land_cover_name == c).sum())) for c in classes], fontsize=8.2)
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
         "cut. Classes are not ranked against each other here - the hierarchy is within a class. "
         "Classes marked * have fewer than %d units and should not drive conclusions."
         % SMALL_N_UNITS, ha="center", fontsize=8.1, style="italic", color="#444444", wrap=True)
base = os.path.join(OUT, "figure15_10m")
for ext in ("png", "pdf"):
    fig.savefig(base + "." + ext, bbox_inches="tight", facecolor="white")
plt.close(fig)

# ---- provenance -----------------------------------------------------------
DISP = pd.DataFrame(displayed)
ref_nonsmall = DISP[(DISP.population == "reference") & ~DISP.small_n_class_flag]
prov = dict(
    generated_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    generating_script="notebooks/prod10m/figure15_10m.py",
    purpose="Figure 15 regenerated from the final 10 m analysis so that figure and Section 4.13 "
            "prose describe the same analysis. Supersedes the 30 m rendering "
            "outputs/final_polygon_thesis/figure_global_vs_within_landcover_priority.png.",
    implementation_resolution="10 m",
    rule="within-class top decile = ceil(n/10) by within_landcover_RIS_rank; hidden if "
         "mean_RIS_balanced < population 0.90 quantile; classes with n < %d flagged and excluded "
         "from the headline total (identical to polygon_10m.py)" % SMALL_N_UNITS,
    inputs={os.path.relpath(p, ROOT): sha256(p) for p in
            [SRC["reference"], SRC["model"], T09, AGJ]},
    outputs={os.path.relpath(base + e, ROOT): sha256(base + e) for e in (".png", ".pdf")},
    global_cut={p: marks[p]["cut"] for p in ("reference", "model")},
    hidden_units={p: len(hidden_ids[p]) for p in ("reference", "model")},
    headline_values=dict(
        hidden_reference_units=int(len(hidden_ids["reference"])),
        cropland_reference="%d of %d" % (
            int(ref_nonsmall.loc[ref_nonsmall.land_cover_name == "cropland", "plotted_blue"].iloc[0]),
            int(ref_nonsmall.loc[ref_nonsmall.land_cover_name == "cropland", "plotted_blue"].iloc[0]
                + ref_nonsmall.loc[ref_nonsmall.land_cover_name == "cropland", "plotted_red"].iloc[0])),
        forest_reference="%d of %d" % (
            int(ref_nonsmall.loc[ref_nonsmall.land_cover_name == "forest", "plotted_blue"].iloc[0]),
            int(ref_nonsmall.loc[ref_nonsmall.land_cover_name == "forest", "plotted_blue"].iloc[0]
                + ref_nonsmall.loc[ref_nonsmall.land_cover_name == "forest", "plotted_red"].iloc[0]))),
    qa=qa,
    displayed_counts=displayed,
    ai_generated_content=False)
with open(os.path.join(OUT, "FIGURE15_PROVENANCE.json"), "w") as fh:
    json.dump(prov, fh, indent=1)

print("wrote %s.{png,pdf}" % os.path.relpath(base, ROOT))
print("hidden units  reference=%d  model=%d" % (len(hidden_ids["reference"]), len(hidden_ids["model"])))
print("headline: %s" % json.dumps(prov["headline_values"]))
