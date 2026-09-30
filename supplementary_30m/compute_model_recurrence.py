"""
compute_model_recurrence.py

Purpose:
    Compute a model-derived recurrence raster from the existing trained
    L-TAE model. Only recurrence is computed; the severity, persistence,
    and recovery_gap rasters produced by Notebook 07b are not modified.

Rationale:
    The original pipeline (Notebook 07b) stores per-pixel maximum burned
    probability across the 108-month window. This is sufficient for
    burned-vs-unburned classification but discards the per-month temporal
    information needed to count fire events per pixel.

    This script reruns inference using the same trained model but, for
    each block, accumulates the count of months in which the classification
    head exceeds the F1-optimized threshold. The per-pixel count is
    normalised as `min(count / 5, 1)` to match the analytical recurrence
    definition used at training time.

Output:
    outputs/inference_full_roi/model_recurrence_30m.tif

Prerequisites:
    - Trained model at models/main/model_main.pt
    - Feature stack at data/raw/feature_stack_30m/
    - Same environment as Notebook 07b (config.py, model_defs.py on path)

Estimated runtime:
    ~1 hour on RTX 4090 for the full ROI at 30 m.
"""

import os
import sys
import gc
import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
import torch
from tqdm import tqdm


# ------------------------------------------------------------
# 1. Environment setup
# ------------------------------------------------------------
IS_COLAB = "google.colab" in sys.modules
if IS_COLAB:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
    THESIS_ROOT = "/content/drive/MyDrive/thesisv4"
else:
    THESIS_ROOT = os.path.expanduser("~/thesisv4")

sys.path.insert(0, THESIS_ROOT)

import config
import model_defs

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------------------------------------
# 2. Load the trained main model
# ------------------------------------------------------------
MODEL_PATH = os.path.join(THESIS_ROOT, "models/main/model_main.pt")

model = model_defs.FireRecoveryModel(
    n_features=len(config.MODEL_FEATURES),
    d_model=config.MODEL["d_model"],
    n_head=config.MODEL["n_head"],
    d_k=config.MODEL["d_k"],
    max_len=config.EXPECTED_T,
    tau=config.MODEL["positional_tau"],
    dropout=0.0,
).to(DEVICE)

state = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state)
model.eval()

n_params = sum(p.numel() for p in model.parameters())
print(f"Loaded main model: {n_params:,} parameters")


# ------------------------------------------------------------
# 3. Threshold and paths
# ------------------------------------------------------------
BURN_THRESHOLD = 0.502   # F1-optimized threshold from results_main.json
BLOCK_SIZE = 512         # block dimensions for inference

FEATURE_DIR = Path(THESIS_ROOT) / "data/raw/feature_stack_v4_30m"
OUT_DIR = Path(THESIS_ROOT) / "outputs/inference_full_roi"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "model_recurrence_30m.tif"


# ------------------------------------------------------------
# 4. Read the ROI grid from an existing feature raster
# ------------------------------------------------------------
sample_tif = next(FEATURE_DIR.glob("features_m*.tif"))
with rasterio.open(sample_tif) as src:
    H, W = src.shape
    meta = src.meta.copy()
    transform = src.transform
    crs = src.crs

print(f"ROI grid: {H} rows x {W} cols")


# ------------------------------------------------------------
# 5. Precompute the feature indices and harmonic values
# ------------------------------------------------------------
GEE_FEATURES = config.GEE_FEATURES
n_features    = len(config.MODEL_FEATURES)
n_gee         = len(GEE_FEATURES)
spectral_cols = np.array(config.SPECTRAL_COLS, dtype=int)
lc_cols       = np.array(config.LC_COLS, dtype=int)

delta_source_indices = {
    b: GEE_FEATURES.index(b) for b in config.DELTA_SOURCE_INDICES
}
delta_target_indices = {
    f"delta_{b}": config.MODEL_FEATURES.index(f"delta_{b}")
    for b in config.DELTA_SOURCE_INDICES
}
harmonic_target_indices = {
    name: config.MODEL_FEATURES.index(name)
    for name in config.HARMONIC_FEATURES
}

month_list = config.get_month_list()
calendar_months  = np.array([m for _, m in month_list], dtype=np.float32)
month_cos_values = np.cos(2 * np.pi * calendar_months / 12.0).astype(np.float32)
month_sin_values = np.sin(2 * np.pi * calendar_months / 12.0).astype(np.float32)

T = config.EXPECTED_T


# ------------------------------------------------------------
# 6. Allocate output array
# ------------------------------------------------------------
# Full ROI recurrence, initialised to NaN so unmasked pixels are visible
model_recurrence = np.full((H, W), np.nan, dtype=np.float32)


# ------------------------------------------------------------
# 7. Block-wise inference - only recurrence is computed
# ------------------------------------------------------------
print("Loading 108 monthly feature rasters...")
feature_stack = []
for m_idx in range(T):
    fp = FEATURE_DIR / f"features_m{m_idx:03d}.tif"
    if fp.exists():
        with rasterio.open(fp) as src:
            feature_stack.append(src.read())    # (C, H, W)
    else:
        feature_stack.append(None)
present_m_idx = [i for i, x in enumerate(feature_stack) if x is not None]
print(f"Loaded {len(present_m_idx)} of {T} monthly rasters")

# Standardization scaler (must be identical to training)
SCALER_PATH = os.path.join(THESIS_ROOT, "artifacts/scaler_v4.npz")
if os.path.exists(SCALER_PATH):
    scaler = np.load(SCALER_PATH)
    scaler_mean = scaler["mean"]
    scaler_std  = scaler["std"]
    print(f"Loaded scaler from {SCALER_PATH}")
else:
    raise FileNotFoundError(
        f"Scaler not found at {SCALER_PATH}. "
        "Ensure the same standardization used at training is applied here."
    )


print("\nStarting block-wise inference (recurrence only)...")
start_time = time.time()

with torch.no_grad():
    for row_start in tqdm(range(0, H, BLOCK_SIZE), desc="Rows"):
        row_end = min(row_start + BLOCK_SIZE, H)
        block_h = row_end - row_start

        for col_start in range(0, W, BLOCK_SIZE):
            col_end = min(col_start + BLOCK_SIZE, W)
            block_w = col_end - col_start
            n_pixels = block_h * block_w

            # Assemble the GEE feature tensor for this block
            block_gee = np.full((T, n_gee, block_h, block_w),
                                 np.nan, dtype=np.float32)
            for m_idx in present_m_idx:
                block_gee[m_idx] = feature_stack[m_idx][:,
                                                         row_start:row_end,
                                                         col_start:col_end]

            # Compute delta features and harmonic features
            block_model = np.zeros((T, n_features, block_h, block_w),
                                    dtype=np.float32)

            for target_name, src_band in zip(
                delta_target_indices.keys(),
                config.DELTA_SOURCE_INDICES,
            ):
                src_idx = delta_source_indices[src_band]
                tgt_idx = delta_target_indices[target_name]
                # delta_t = value_t - value_{t-1}, first month = 0
                for t in range(1, T):
                    block_model[t, tgt_idx] = (
                        block_gee[t, src_idx] - block_gee[t - 1, src_idx]
                    )

            # Copy the direct GEE features to the model tensor
            for i, name in enumerate(GEE_FEATURES):
                if name in config.MODEL_FEATURES:
                    tgt_idx = config.MODEL_FEATURES.index(name)
                    block_model[:, tgt_idx] = block_gee[:, i]

            # Harmonic features are constant across pixels but vary with t
            for t in range(T):
                block_model[t, harmonic_target_indices["month_cos"]] = \
                    month_cos_values[t]
                block_model[t, harmonic_target_indices["month_sin"]] = \
                    month_sin_values[t]

            # Reshape to (n_pixels, T, C)
            X = block_model.transpose(2, 3, 0, 1).reshape(n_pixels, T, n_features)
            valid_mask = ~np.isnan(X).any(axis=(1, 2))    # pixels with any data

            if not valid_mask.any():
                continue

            # Standardize
            X_valid = X[valid_mask]
            X_scaled = (X_valid - scaler_mean) / scaler_std

            # Build the temporal presence mask (True where any feature exists)
            mask_temporal = np.zeros((X_valid.shape[0], T), dtype=bool)
            for t in range(T):
                if feature_stack[t] is not None:
                    mask_temporal[:, t] = True

            # Run model on this block
            Xb = torch.from_numpy(X_scaled).to(DEVICE)
            Mb = torch.from_numpy(mask_temporal).to(DEVICE)
            out = model(Xb, mask=Mb)

            # burned_logits shape: (n_valid_pixels, T)
            burned_probs = torch.sigmoid(out["burned_logits"]).cpu().numpy()

            # Count months where model predicted fire, normalize to [0, 1]
            fire_counts = (burned_probs > BURN_THRESHOLD).sum(axis=1)
            recurrence_block = np.clip(fire_counts / 5.0, 0, 1).astype(np.float32)

            # Scatter back into the full block, then into the full ROI array
            block_full = np.full(n_pixels, np.nan, dtype=np.float32)
            block_full[valid_mask] = recurrence_block
            block_2d = block_full.reshape(block_h, block_w)

            model_recurrence[row_start:row_end, col_start:col_end] = block_2d

            del Xb, Mb, out, burned_probs, X, X_valid, X_scaled, mask_temporal
            gc.collect()

elapsed = time.time() - start_time
print(f"\nInference complete in {elapsed / 60:.1f} minutes")


# ------------------------------------------------------------
# 8. Write output raster
# ------------------------------------------------------------
meta.update(dtype="float32", count=1, nodata=np.nan)
with rasterio.open(OUT_PATH, "w", **meta) as dst:
    dst.write(model_recurrence, 1)

n_valid = np.sum(~np.isnan(model_recurrence))
n_burned = np.sum(model_recurrence > 0)
n_high = np.sum(model_recurrence >= 0.4)   # 2 or more fires

print(f"\nWrote: {OUT_PATH}")
print(f"  Total valid pixels:      {n_valid:,}")
print(f"  Pixels with model_recurrence > 0:      {n_burned:,}")
print(f"  Pixels with 2+ predicted fires (>= 0.4): {n_high:,}")
print()
print("Next steps:")
print("  1. Upload the new raster as a GEE asset:")
print("     v4_model_recurrence_30m")
print("  2. Update the visualization script (gee_visualization_v4.js)")
print("     to use model_recurrence in place of GT-derived recurrence")
print("  3. Update the composition notebook to include")
print("     fig_model_recurrence.tif in Figure 4.2 panel (d)")
