#!/usr/bin/env bash
# Final 10 m regional workflow - run from the repository root.
#
#   bash pipeline_10m/run_pipeline.sh
#
# Requires the restricted inputs to be present (see data/README.md):
#   data/thesis_maps/fig_gt_fires.*      fire-reference inventory
#   data/raw/landcover_2018_30m.tif      land-cover product
# and the 10 m inference tiles to have been produced by run_inference_10m.py.
set -euo pipefail
ROOT="${THESIS_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT"
PY="${PYTHON:-python}"
D=pipeline_10m
L=outputs/polygon_10m/pipeline.log
mkdir -p outputs/polygon_10m

stage () { echo "=== $1 ===" | tee -a "$L"; shift; "$@" >> "$L" 2>&1 || { echo "STAGE FAILED" | tee -a "$L"; exit 1; }; }

stage "STAGE 1  inference QA"        $PY -u $D/qa_10m.py
stage "STAGE 2  build 10 m sources"  $PY -u $D/build_10m_sources.py
stage "STAGE 3  polygon analysis"    $PY -u $D/polygon_10m.py
stage "STAGE 4  unit geometries"     $PY -u $D/geometries_10m.py
stage "STAGE 5  figures 11 and 12"   $PY -u $D/figures_10m.py
stage "STAGE 6  migration table"     $PY -u $D/migration_table.py
echo "=== PIPELINE COMPLETE ===" | tee -a "$L"
