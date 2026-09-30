"""
extract_nbr_arrays.py - pull the raw spectral columns out of X_scaled.npy.

Run once on the machine that holds data/processed/X_scaled.npy
(the ~13.7 GB feature tensor). It writes three small arrays that let every
recovery-gap diagnostic run on a laptop:

    data/processed/nbr_raw.npy          (608922, 108) float32   ~263 MB
    data/processed/nbr_zscore_raw.npy   (608922, 108) float32   ~263 MB
    data/processed/ndvi_raw.npy         (608922, 108) float32   ~263 MB

Why these three:
  * NBR - the channel the recovery-gap and (via its z-score) the
                 severity and persistence targets are built from.
  * NBR_zscore - the channel severity and persistence are built from
                 directly; needed for the persistence-threshold sensitivity.
  * NDVI - an independent vegetation index. If the land-cover ordering
                 of the recovery gap appears in NBR but not in NDVI, the
                 ordering is a property of the index rather than of vegetation
                 recovery. This is the cheapest available cross-check on the
                 "spectral vs ecological" question.

Method. X_scaled holds per-feature standardized values for the 32 spectral
columns (the 20 land-cover one-hots were left at 0/1). StandardScaler is
exactly invertible, so

    raw = X_scaled[:, :, col] * scaler.scale_[pos] + scaler.mean_[pos]

recovers the pre-scaling values, which for NBR are the same imputed raw NBR
that Notebook 04 used to build the targets. The script proves this rather
than assuming it: it recomputes recovery_gap, severity and persistence from
the extracted arrays and compares them against the stored targets_aux.npz.
If the reconstruction were wrong, those checks would fail.

Reading is chunked over pixels so peak memory stays near 250 MB regardless
of how large X_scaled is; the file is opened read-only and never modified.

Usage:
    python gee/extract_nbr_arrays.py
"""

import json
import os
import pickle
import sys

import numpy as np

THESIS_ROOT = os.path.expanduser(os.environ.get("THESIS_ROOT", "~/thesisv4"))
sys.path.insert(0, THESIS_ROOT)
import config  # noqa: E402

# Feature -> output filename. Add entries here if another channel is needed.
WANTED = {
    "NBR":        "nbr_raw.npy",
    "NBR_zscore": "nbr_zscore_raw.npy",
    "NDVI":       "ndvi_raw.npy",
}

PIXEL_CHUNK = 10_000          # ~225 MB per read at (10000, 108, 52) float32


def _p(*parts):
    return os.path.join(THESIS_ROOT, *parts)


def load_scaler():
    with open(config.FILES["scaler"], "rb") as fh:
        bundle = pickle.load(fh)
    return bundle["scaler"], bundle["spectral_cols"]


def extract():
    with open(config.FILES["feature_names"]) as fh:
        feature_names = json.load(fh)
    scaler, spectral_cols = load_scaler()

    X = np.load(config.FILES["X_scaled"], mmap_mode="r")
    n_pixels, n_time, n_feat = X.shape
    print(f"X_scaled: {X.shape}  ({X.nbytes / 1e9:.1f} GB on disk, memory-mapped)")

    plan = {}
    for name in WANTED:
        col = feature_names.index(name)
        if col not in spectral_cols:
            raise ValueError(f"{name} is not a standardized column; nothing to invert")
        pos = spectral_cols.index(col)
        plan[name] = (col, float(scaler.mean_[pos]), float(scaler.scale_[pos]))
        print(f"  {name:<12} column {col:>2}  mean={plan[name][1]:+.6f}  scale={plan[name][2]:.6f}")

    out = {name: np.empty((n_pixels, n_time), dtype=np.float32) for name in plan}

    for start in range(0, n_pixels, PIXEL_CHUNK):
        end = min(start + PIXEL_CHUNK, n_pixels)
        block = np.asarray(X[start:end])                     # (chunk, T, C)
        for name, (col, mu, sd) in plan.items():
            out[name][start:end] = block[:, :, col] * sd + mu
        del block
        if (start // PIXEL_CHUNK) % 10 == 0:
            print(f"    {end:>7,} / {n_pixels:,} pixels", flush=True)

    return out, feature_names


def recompute_targets(nbr_raw, nbr_z, y, baseline_months=12, recent_months=12):
    """Vectorised restatement of Notebook 04 cell 12. Must match targets_aux.npz."""
    burned = y.sum(axis=1) > 0

    baseline = np.nanmean(nbr_raw[:, :baseline_months].astype(np.float64), axis=1)
    recent = np.nanmean(nbr_raw[:, -recent_months:].astype(np.float64), axis=1)
    gap = (baseline - recent) / (np.abs(baseline) + 1e-6)
    recovery_gap = np.zeros(len(y), dtype=np.float32)
    recovery_gap[burned] = np.clip(gap, 0, 1)[burned].astype(np.float32)

    severity = np.zeros(len(y), dtype=np.float32)
    persistence = np.zeros(len(y), dtype=np.float32)
    for i in np.flatnonzero(burned):
        months = np.flatnonzero(y[i] == 1)
        severity[i] = np.clip((-nbr_z[i, months]).mean() / 5.0, 0, 1)
        post = nbr_z[i, months[-1]:]
        if len(post) > 1:
            persistence[i] = (post < -0.5).mean()

    return {"recovery_gap": recovery_gap, "severity": severity,
            "persistence": persistence}


def verify(out):
    """Prove the reconstruction by rebuilding the stored targets from it."""
    y = np.load(config.FILES["y_burned"])
    stored = np.load(config.FILES["targets_aux"])
    rebuilt = recompute_targets(out["NBR"], out["NBR_zscore"], y)

    print("\nVerification — recomputing the stored targets from the extracted arrays:")
    ok = True
    for name, got in rebuilt.items():
        want = stored[name]
        diff = np.abs(got - want)
        agree = float(np.mean(diff < 1e-4))
        print(f"  {name:<14} max|diff| = {diff.max():.3e}   "
              f"within 1e-4 for {100 * agree:.4f}% of pixels   "
              f"corr = {np.corrcoef(got, want)[0, 1]:.8f}")
        if agree < 0.999:
            ok = False
    if not ok:
        print("\n  WARNING: reconstruction does not reproduce the stored targets.")
        print("  Do not use these arrays until the mismatch is understood.")
    else:
        print("\n  PASS — the extracted arrays reproduce targets_aux.npz.")
    return ok


def main():
    out, _ = extract()
    ok = verify(out)

    print("\nWriting:")
    for name, fname in WANTED.items():
        path = _p("data", "processed", fname)
        np.save(path, out[name])
        a = out[name]
        print(f"  {fname:<22} {a.shape}  {a.nbytes / 1e6:6.1f} MB  "
              f"range [{np.nanmin(a):+.4f}, {np.nanmax(a):+.4f}]  mean {np.nanmean(a):+.4f}")

    print("\nCopy these three files to the analysis machine at "
          "data/processed/ and run notebooks/08_target_diagnostics.py.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
