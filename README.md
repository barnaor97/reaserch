# Post-conflict fire restoration prioritization in the Western Negev

Code accompanying the MSc thesis *Assessing Post-Fire Restoration Priorities in the Western Negev Using Sentinel-2 Time Series and Deep Learning* (Bar Naor).

The thesis develops the **first stage** of a two-stage decision-support framework: it derives per-pixel indicators of fire disturbance and post-fire spectral condition from a nine-year Sentinel-2 record, interprets them within a land-cover context, and demonstrates analytically how they can be combined into a Restoration Indicator Score (RIS). The operational, stakeholder-weighted second stage is future work and is **not** implemented here.

---

## Data availability and reproducibility

Publicly available satellite and environmental inputs can be retrieved programmatically using the provided code. The fire-reference inventory and project land-cover layer are third-party research inputs and are therefore not redistributed with this repository. Their required structure and role in the analytical workflow are documented. Consequently, reproduction of stages that depend on these inputs requires authorized access to the corresponding datasets.

### Reproducible from public sources alone

- Sentinel-2 L2A retrieval and Cloud Score+ cloud/shadow masking through Google Earth Engine (`gee/`).
- WorldClim 2.1 and SRTM retrieval and the Figure 3 environmental-setting map (`figures/22_study_area_environment.py`) — this runs end to end with no restricted input.
- The complete analytical workflow as source code, including the model architecture, training procedure, target definitions and the final 10 m pipeline.

### Requires authorized third-party input

- Construction of the per-pixel-per-month burned labels and the stratified sample.
- Reconstruction of the training dataset, and therefore model training.
- All evaluation of fire detection against the fire-reference inventory.
- Every land-cover-conditioned analysis, including the fixed 2018 baseline.
- Spatial-unit construction and all results that depend on those units, including the RIS prioritization.

Concretely:

| Input | Status | How to obtain |
|---|---|---|
| Sentinel-2 L2A, Cloud Score+ | public | retrieved via Google Earth Engine by `gee/` |
| WorldClim 2.1 (BIO12) | public | downloaded automatically by `figures/22_study_area_environment.py` |
| SRTM GL1 | public | downloaded automatically by `figures/22_study_area_environment.py` |
| Region-of-interest polygon | included | `data/roi/` |
| **Fire-reference inventory** (KKL/JNF + Planet/Sentinel-2 scars) | **restricted** | request from the provider; place in `data/restricted/` |
| **Project land-cover product** (2018 baseline) | **restricted** | request from the research group; place in `data/restricted/` |

See [`data/README.md`](data/README.md) for the full manifest, expected schemas and file paths.

This repository does not redistribute the fire-reference inventory or the land-cover product because they are third-party research data. Nothing in this repository attempts to reconstruct or approximate them.

---

## The canonical workflow

**The final regional analysis in the thesis is the 10 m workflow (`pipeline_10m/`).** An earlier 30 m implementation is retained in `supplementary_30m/` **only** because a small number of explicitly labelled supplementary diagnostics in the thesis appendix were produced with it. The 30 m pipeline is *not* the final regional analysis and should not be read as such.

```
  public Sentinel-2 / Cloud Score+  ─┐
                                     ├─→ preprocessing and sampling      gee/
  restricted fire-reference inventory┘        (burned labels, strata)
                                             ↓
                                     feature construction (52/month)     notebooks/04
                                             ↓
                         analytical target construction                  notebooks/04
                    (NBR-derived severity, persistence, recovery gap)
                                             ↓
                                     L-TAE training                      notebooks/05
                                             ↓
                    evaluation, robustness, baselines, sensitivity       notebooks/06, 06b
                                             ↓
                        FINAL 10 m regional inference                    pipeline_10m/
                                             ↓
   restricted land-cover (2018 baseline) ──→ burned spatial units        pipeline_10m/
                                             ↓
                            land-cover interpretation                    pipeline_10m/
                                             ↓
                        RIS analytical prioritization                    pipeline_10m/
                                             ↓
                              thesis figures and tables                  pipeline_10m/, figures/
```

Key parameters are preserved exactly as used in the thesis: fire-detection operating threshold **0.502**, 52 features per month, 108 monthly steps, 608,922 sampled pixels, seed 42, and the analytical target definitions in `notebooks/04_data_prep.ipynb`.

---

## Repository layout

| Path | Contents |
|---|---|
| `configs/` | `config.py`, feature names, fitted scaler, train/val/test split indices |
| `src/` | model definitions (L-TAE and baselines), evaluation metrics |
| `gee/` | Earth Engine assembly and the per-pixel time-series export |
| `notebooks/` | data preparation, training, experiments, feature importance, target diagnostics |
| `pipeline_10m/` | **final** 10 m inference, spatial units, land-cover analysis, RIS, figures/tables |
| `supplementary_30m/` | earlier 30 m workflow, retained for labelled appendix diagnostics only |
| `figures/` | Figure 3 environmental-setting map (fully public inputs) |
| `verification/` | independent recomputation of reported values |
| `models/` | training history, results JSON, per-experiment metrics (no weights — see below) |
| `outputs/` | aggregate result tables reproduced in the thesis, plus run provenance |
| `data/` | ROI polygon; README manifest; empty `restricted/` placeholder |

---

## Getting started

```bash
git clone <repository-url> && cd <repository>

# modelling stage
conda create -n thesis python=3.11 -y && conda activate thesis
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# geospatial stage (version-pinned)
python -m venv .venv-geo && source .venv-geo/bin/activate
pip install -r requirements-geo.txt

# Earth Engine
export GEE_PROJECT="your-ee-project-id"
earthengine authenticate
```

Scripts resolve the repository root automatically; override with `THESIS_ROOT` if needed.

Stages that need no restricted data — `figures/22_study_area_environment.py` and the public parts of `gee/` — run immediately. Everything downstream of burned-label construction requires the restricted inputs described in `data/README.md`.

---

## What this repository does not contain

- The fire-reference inventory and the project land-cover product (restricted third-party data).
- Any derived file from which those datasets could be reconstructed — notably per-pixel coordinates paired with fire-history labels, per-unit tables carrying coordinates, and the burned-label arrays. These were screened out; see `docs/table_screening.json`.
- The ~13.7 GB scaled feature matrix `X_scaled.npy`. It is a deterministic product of the documented pipeline and is regenerated by `notebooks/04_data_prep.ipynb`; it is not archived.
- Trained model weights. No pretrained checkpoints are distributed here.

## Licence

No software licence has yet been assigned to this repository.

## Open items

- **Trained model weights are not distributed.** This repository contains the model architecture, training code, configuration, results JSON and training histories, but no pretrained checkpoints.
- **Training-stage package versions are not pinned**; the geospatial stage is.

## Citation

Naor, B. (2026). *Assessing Post-Fire Restoration Priorities in the Western Negev Using Sentinel-2 Time Series and Deep Learning.* MSc thesis, Ben-Gurion University of the Negev.

Advisors: Prof. Tarin Paz-Kagan and Prof. Yael Edan.
