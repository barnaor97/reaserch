"""
Central configuration for the Western Negev restoration prioritization study.

Every path, feature definition, target definition and hyperparameter used by the
thesis is declared here so that the notebooks and pipeline scripts share one
source of truth.

Sample design. The modelling sample contains 608,922 pixels: all burned pixels
of every recurrence stratum (300,032 low, 4,050 medium, 379 high) plus 304,461
pixels with no documented fire, drawn to match the burned class 1:1 at the pixel
level. Retaining all burned pixels preserves the rare high-recurrence cases; the
1:1 match balances the pixel-level classes but not the monthly classification
target, where burned pixel-months remain a small minority.

Targets. recovery_gap is defined only for pixels with at least one documented
fire month and is set to 0 elsewhere, so that pixels which never burned do not
contribute a spurious recovery deficit.

Environment. Colab and local Linux/GPU runs are detected automatically and the
paths and batch size adapt accordingly.
"""

import json
import os
import random
import sys
import numpy as np
import torch


# -----------------------------------------------------------
# Environment auto-detection
# -----------------------------------------------------------
IS_COLAB = "google.colab" in sys.modules

# Workspace root - picks Drive path in Colab and local disk otherwise.
if IS_COLAB:
    DRIVE_ROOT = "/content/drive/MyDrive/thesisv4"
else:
    DRIVE_ROOT = os.path.expanduser("~/thesisv4")


# -----------------------------------------------------------
# Project version & seed
# -----------------------------------------------------------
VERSION = "_10m"
RUN_TAG = "v4"
SEED    = 42


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------------------------------------
# Paths
# -----------------------------------------------------------
PATHS = {
    "drive_root":   DRIVE_ROOT,
    "data_raw":     f"{DRIVE_ROOT}/data/raw",
    "data_proc":    f"{DRIVE_ROOT}/data/processed",
    "data_eda":     f"{DRIVE_ROOT}/data/eda",
    "models_main":  f"{DRIVE_ROOT}/models/main",
    "models_exp":   f"{DRIVE_ROOT}/models/experiments",
    "outputs_fig":  f"{DRIVE_ROOT}/outputs/figures",
    "outputs_tab":  f"{DRIVE_ROOT}/outputs/tables",
    "outputs_map":  f"{DRIVE_ROOT}/outputs/maps",
}


def ensure_dirs():
    for p in PATHS.values():
        os.makedirs(p, exist_ok=True)


# -----------------------------------------------------------
# Filenames
# -----------------------------------------------------------
FILES = {
    "burn_count_tif":     f"{PATHS['data_raw']}/burn_count_with_mask{VERSION}.tif",
    "pixel_coords":       f"{PATHS['data_raw']}/pixel_coords{VERSION}.csv",
    "raw_timeseries":     f"{PATHS['data_raw']}/full_pixels{VERSION}.csv",
    "selected_features":  f"{PATHS['data_eda']}/selected_features.json",
    "X_scaled":           f"{PATHS['data_proc']}/X_scaled.npy",
    "y_burned":           f"{PATHS['data_proc']}/y_burned.npy",
    "targets_aux":        f"{PATHS['data_proc']}/targets_aux.npz",
    "splits":             f"{PATHS['data_proc']}/splits.json",
    "scaler":             f"{PATHS['data_proc']}/scaler.pkl",
    "feature_names":      f"{PATHS['data_proc']}/feature_names.json",
    "model_main":         f"{PATHS['models_main']}/model_main.pt",
    "history_main":       f"{PATHS['models_main']}/training_history.json",
    "preds_test":         f"{PATHS['models_main']}/predictions_test.npz",
}


# -----------------------------------------------------------
# Google Earth Engine asset IDs
# -----------------------------------------------------------
GEE_PROJECT    = os.environ.get("GEE_PROJECT", "YOUR-GEE-PROJECT")
GEE_ASSET_ROOT = f"projects/{GEE_PROJECT}/assets"

GEE = {
    "project":            GEE_PROJECT,
    "asset_root":         GEE_ASSET_ROOT,
    "roi":                f"{GEE_ASSET_ROOT}/gaza_envelope_for_decell_Dissolve",
    "infra_mask":         f"{GEE_ASSET_ROOT}/human_infrastructure_mask_utm36n",
    "fire_truth_l1":      f"{GEE_ASSET_ROOT}/fires_scars_otef_6991",
    "fire_truth_l2":      f"{GEE_ASSET_ROOT}/fire_damage",
    "fire_truth_merged":  f"{GEE_ASSET_ROOT}/fires_layer1_layer2{VERSION}",
    "landcover_yearly":   f"{GEE_ASSET_ROOT}/landcover_smoothed_yearly",
    "ic_s2":              f"{GEE_ASSET_ROOT}/s2_monthly_image_collection_v3{VERSION}",
    "ic_labels":          f"{GEE_ASSET_ROOT}/fire_monthly_labels_v3{VERSION}",
    "ic_baseline_mean":   f"{GEE_ASSET_ROOT}/seasonal_baseline_mean_2016_2017{VERSION}",
    "ic_baseline_median": f"{GEE_ASSET_ROOT}/seasonal_baseline_median_2016_2017{VERSION}",
    "ic_baseline_std":    f"{GEE_ASSET_ROOT}/seasonal_baseline_std_2016_2017{VERSION}",
}

# Names of the regional inference outputs written back to Earth Engine.
GEE_OUTPUT_ASSETS = {
    "severity":     f"{GEE_ASSET_ROOT}/severity_v4_30m",
    "persistence":  f"{GEE_ASSET_ROOT}/persistence_v4_30m",
    "recovery_gap": f"{GEE_ASSET_ROOT}/recovery_gap_v4_30m",
    "burned_prob":  f"{GEE_ASSET_ROOT}/burned_prob_v4_30m",
    "recurrence":   f"{GEE_ASSET_ROOT}/recurrence_v4_30m",
}


# -----------------------------------------------------------
# Analysis grid, projection and region of interest.
# -----------------------------------------------------------
SCALE      = 10
CRS        = "EPSG:32636"
MAX_PIXELS = 1e13


# -----------------------------------------------------------
# Monthly model input sequence. The 2016 imagery is used only to build the
# seasonal reference baseline and is not part of the input sequence.
# -----------------------------------------------------------
START_YEAR  = 2017
START_MONTH = 1
END_YEAR    = 2025
END_MONTH   = 12

BASELINE_YEARS = [2016, 2017]

S2_COLLECTION_START_YEAR = 2016
S2_COLLECTION_END_YEAR   = 2025


def get_month_list():
    months = []
    for y in range(START_YEAR, END_YEAR + 1):
        for m in range(1, 13):
            if y == START_YEAR and m < START_MONTH:
                continue
            if y == END_YEAR and m > END_MONTH:
                continue
            months.append((y, m))
    return months


def get_s2_month_list():
    months = []
    for y in range(S2_COLLECTION_START_YEAR, S2_COLLECTION_END_YEAR + 1):
        for m in range(1, 13):
            months.append((y, m))
    return months


EXPECTED_T = len(get_month_list())


# -----------------------------------------------------------
# Per-month feature schema: 52 values per pixel per month.
# -----------------------------------------------------------
RAW_BANDS = ["B4", "B8", "B8A", "B11", "B12"]

INDICES = ["NDVI", "EVI", "NDMI", "NBR", "MIRBI", "BSI", "BAIS2"]

QUALITY_BANDS = ["pct_valid"]

ANOMALY_SOURCE_INDICES = ["NDVI", "EVI", "NDMI", "NBR", "MIRBI", "BSI", "BAIS2"]
ANOMALY_BANDS = (
    [b + "_zscore"  for b in ANOMALY_SOURCE_INDICES] +
    [b + "_anomaly" for b in ANOMALY_SOURCE_INDICES]
)

DELTA_SOURCE_INDICES = ["NBR", "NDVI", "NDMI", "BAIS2"]
DELTA_FEATURES = [f"delta_{b}" for b in DELTA_SOURCE_INDICES]

HARMONIC_FEATURES = ["month_cos", "month_sin"]


# -----------------------------------------------------------
# Land-cover classes of the project land-cover product, used both as model
# features and as the fixed 2018 baseline for the spatial-unit analysis.
# -----------------------------------------------------------
LC_CATEGORIES = {
    10: "forest",
    11: "orchards",
    20: "shrubland",
    30: "grassland",
    40: "cropland",
    41: "covered_agriculture",
    50: "built_up",
    60: "bare",
    61: "fallow",
    80: "water",
}

LC_CODES = sorted(LC_CATEGORIES.keys())
LC_FEATURES_PREV = [f"lc_prev_{LC_CATEGORIES[c]}" for c in LC_CODES]
LC_FEATURES_CURR = [f"lc_curr_{LC_CATEGORIES[c]}" for c in LC_CODES]
LC_FEATURES      = LC_FEATURES_PREV + LC_FEATURES_CURR


# -----------------------------------------------------------
# Ordered feature list consumed by the model; the order defines the input axis.
# -----------------------------------------------------------
MODEL_FEATURES = (
    RAW_BANDS
    + INDICES
    + [b + "_zscore"  for b in ANOMALY_SOURCE_INDICES]
    + [b + "_anomaly" for b in ANOMALY_SOURCE_INDICES]
    + LC_FEATURES
    + DELTA_FEATURES
    + HARMONIC_FEATURES
)

ALL_FEATURES = (RAW_BANDS + INDICES + QUALITY_BANDS + ANOMALY_BANDS +
                LC_FEATURES + DELTA_FEATURES + HARMONIC_FEATURES)


# -----------------------------------------------------------
# Fire-recurrence strata used to build the sample.
# -----------------------------------------------------------
PIXEL_TYPES = [
    "no_fires",
    "low_recurrence",
    "medium_recurrence",
    "high_recurrence",
]

PIXEL_TYPE_LEGACY_MAP = {
    "never_burned":  "no_fires",
    "burned_low":    "low_recurrence",
    "burned_medium": "medium_recurrence",
    "burned_high":   "high_recurrence",
}


# -----------------------------------------------------------
# Sampling strategy: keep every burned pixel and match with an equal number
# of pixels that have no documented fire.
# -----------------------------------------------------------
SAMPLING = {
   "n_no_fires":        304461,
   "n_low_recurrence":  300032,
   "n_medium_recurrence": 4050,
   "n_high_recurrence":   379,
   "quality_threshold":   0.2,
   }

# -----------------------------------------------------------
# L-TAE architecture and optimisation settings.
# -----------------------------------------------------------
MODEL = {
    "d_model":        128,
    "n_head":         4,
    "d_k":            32,
    "positional_tau": 10000,
    "dropout":        0.1,
}


# -----------------------------------------------------------
# Relative weighting of the classification head and the three regression heads
# in the masked multi-task loss.
# -----------------------------------------------------------
LOSS = {
    "focal_alpha": 0.75,
    "focal_gamma": 2.0,
    "lambda_aux":  0.3,
}


# -----------------------------------------------------------
# Training
#
# batch_size defaults to 256 (RTX 4090). On Colab T4, falls back to 64.
# -----------------------------------------------------------
_BATCH_SIZE = 64 if IS_COLAB else 256

TRAINING = {
    "batch_size":   _BATCH_SIZE,
    "epochs":       50,
    "lr":           5e-4,
    "weight_decay": 1e-4,
    "grad_clip":    1.0,
}


# -----------------------------------------------------------
# Train / validation / test partition of the pixel sample.
# -----------------------------------------------------------
SPLITS = {
    "train_frac": 0.70,
    "val_frac":   0.15,
    "test_frac":  0.15,
}


def print_summary():
    print("=" * 60)
    print(f"CONFIG SUMMARY  (run_{RUN_TAG})")
    print("=" * 60)
    print(f"Environment:        {'Colab' if IS_COLAB else 'Local'}")
    print(f"Workspace root:     {DRIVE_ROOT}")
    print(f"GEE project:        {GEE_PROJECT}")
    print(f"Scale:              {SCALE} m")
    print(f"CRS:                {CRS}")
    print(f"Analysis period:    {START_YEAR}-{START_MONTH:02d} "
          f"to {END_YEAR}-{END_MONTH:02d}")
    print(f"Analysis months:    {len(get_month_list())}")
    print(f"Baseline years:     {BASELINE_YEARS}")
    print(f"Total features:     {len(MODEL_FEATURES)}")
    print(f"Sampling targets:")
    for k, v in SAMPLING.items():
        if k != "quality_threshold":
            print(f"  {k:<22} {v:>8,}")
        else:
            print(f"  {k:<22} {v:>8}")
    total = sum(v for k, v in SAMPLING.items() if k != "quality_threshold")
    print(f"  {'TOTAL':<22} {total:>8,}")
    print(f"Model:              d_model={MODEL['d_model']}, "
          f"n_head={MODEL['n_head']}, d_k={MODEL['d_k']}")
    print(f"Training:           batch_size={TRAINING['batch_size']}, "
          f"epochs={TRAINING['epochs']}")
    print("=" * 60)
