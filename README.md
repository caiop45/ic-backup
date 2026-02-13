# THT-TripGen Pipeline

This repository implements an end-to-end synthetic trip-data workflow focused on
`THT-TripGen`: topology-aware modeling, trip generation, metric computation, and
leakage-aware quality assessment.

The project is organized around one run directory per experiment and a small set of
canonical CLI commands.

## Table of Contents

1. [What this project does](#what-this-project-does)
1. [Who should use it](#who-should-use-it)
1. [Quick setup](#quick-setup)
1. [Repository layout](#repository-layout)
1. [Core workflow](#core-workflow)
1. [CLI command matrix](#cli-command-matrix)
1. [Recommended run recipes](#recommended-run-recipes)
1. [Configuration](#configuration)
1. [Output structure and file map](#output-structure-and-file-map)
1. [Exports and external analysis](#exports-and-external-analysis)
1. [Monitoring and troubleshooting](#monitoring-and-troubleshooting)
1. [Testing and validation](#testing-and-validation)
1. [Conventions and extension points](#conventions-and-extension-points)

## What This Project Does

THT-TripGen models trip OD, time, and residual fields with optional
passenger and fare attributes. The project supports:

1. Preparing data from raw taxi parquet and zone geometries.
1. Building spatial adjacency and zone embeddings.
1. Training the THT-TripGen model.
1. Sampling synthetic trips for validation or hold-out splits.
1. Recomputing quality/utility, privacy, and downstream evaluation metrics.
1. Exporting normalized run summaries for downstream dashboards and scripts.

## Who Should Use It

This is for analysts and researchers who need structured outputs to compare real and
synthetic transportation demand data.

No TVAE workflow is required anymore; only the THT-TripGen path is documented and
used for production-style runs.

## Quick Setup

### 1) Environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

### 2) Install project

```bash
pip install -e .[geo,viz,torch]
```

If your environment already has geopandas/contextily/torch pinned differently, install
them separately first and then install project extras as needed.

### 3) Verify CLI entrypoints

```bash
tvae-run-full-pipeline --help
tvae-train-tht-tripgen --help
tvae-recalc-metrics --help
```

Fallback when entrypoints are not available:

```bash
PYTHONPATH=src python -m tvae.tools.run_full_pipeline --help
PYTHONPATH=src python -m tvae.train_tht_tripgen --help
PYTHONPATH=src python -m tvae.recalc_metrics_tht_tripgen_hold --help
```

### 4) Install expectations

- Python `>=3.10`
- Torch-capable environment for full-size runs (CPU fallback possible)
- Memory and disk proportional to input parquet size and export verbosity
- Geospatial dependencies for topology tooling (`geopandas`, `shapely`, `pyproj`)

## Repository Layout

```text
.
├─ src/
│  └─ tvae/
│     ├─ config.py                 # Runtime and training configuration.
│     ├─ train_tht_tripgen.py      # THT-TripGen training.
│     ├─ sample_tht_tripgen.py     # Synthetic sampling.
│     ├─ recalc_metrics_tht_tripgen_hold.py
│     ├─ recalc_downstream_tht_tripgen.py
│     ├─ data_processing/          # Loaders and transformers.
│     ├─ models/                   # Model implementations.
│     ├─ topology/                 # Graph building and embeddings.
│     ├─ tools/                    # CLI-focused scripts.
│     └─ utils/                    # Metrics, privacy, serialization helpers.
├─ data/                          # Raw and derived input assets.
├─ outputs/
│  ├─ topology_cache/             # Efunc/Ephys/Ecomb node embeddings.
│  ├─ logs/                       # Training logs.
│  ├─ save_data/                  # Run artifacts.
│  ├─ graficos/                   # Generated figures.
│  └─ plots/                      # Optional training curves.
├─ tests/                         # Smoke and regression tests.
└─ README.md
```

### Notes on paths

`src/tvae/config.py` defines output roots and defaults:

1. `OUTPUT_BASE_DIR = outputs`
1. `SAVE_DATA_DIR = outputs/save_data/ryc`
1. `LOG_DIR = outputs/logs`
1. `PLOT_DIR = outputs/plots`
1. `THT_TOPOLOGY_CACHE_DIR = outputs/topology_cache`

## Core Workflow

The canonical production flow is:

1. Ingest and normalize raw trip inputs.
1. Build or verify physical adjacency graph.
1. Build zone embeddings.
1. Train THT-TripGen.
1. Recompute metrics on synthetic vs real splits.
1. Build consolidated metric and training exports.
1. Optionally create analysis outputs and plots.

## CLI Command Matrix

### Data preparation

`tvae-convert-latlon`

Purpose: convert raw taxi parquet input to centroidized/normalized schema used by all
subsequent steps.

Recommended flags: `--trip-parquet`, `--zones`, `--output`

```bash
tvae-convert-latlon \
  --trip-parquet data/yellow_tripdata_2024-04.parquet \
  --trip-parquet data/yellow_tripdata_2024-05.parquet \
  --zones data/taxi_zones.parquet \
  --output data/viagens_lat_long.parquet
```

### Topology setup

`tvae-build-phys-edges`

Purpose: generate physical adjacency for topology-aware embeddings.

Key flags: `--zones`, `--out`, `--split {S1,S2}`, `--adjacency`, `--gap-tol`,
`--bridge-max-dist`, `--fix-geoms`

```bash
tvae-build-phys-edges \
  --zones data/taxi_zones.parquet \
  --out data/topology/tht_phys_edges_S1.csv \
  --split S1 \
  --adjacency queen \
  --gap-tol 5.0 \
  --bridge-max-dist 2500 \
  --adjacency-crs EPSG:3857 \
  --fix-geoms make_valid_if_available
```

`tvae-build-zone-embeddings`

Purpose: train and cache zone embeddings used by THT-TripGen.

```bash
tvae-build-zone-embeddings --device cpu --alpha 1.0 --beta 1.0
```

Optional: `--force` to rebuild even when cache already exists.

### Graph inspection and visualization

`tvae-inspect-tht-phys`

Purpose: diagnose rook/queen adjacency and near-miss candidate pairs.

`tvae-viz-tht-phys`

Purpose: visualize physical adjacency overlay maps.

`tvae-viz-tht-func`

Purpose: visualize learned OD directionality and weighted top-k functional graph.

Common flags for both visualization scripts:

1. `--zones`
1. `--out-dir`
1. `--mappings`
1. `--basemap`
1. `--dpi`
1. `--hour` / `--hours` / `--hour-range` (functional graph only)

### Model and synthesis

`tvae-train-tht-tripgen`

Purpose: train a THT-TripGen model.

```bash
tvae-train-tht-tripgen --run-tag tht_tripgen_20260213 --device cuda
```

`tvae-sample-tht-tripgen`

Purpose: generate synthetic trip tables for `val` or `hold` split.

```bash
tvae-sample-tht-tripgen \
  --run-dir outputs/save_data/ryc/tht_tripgen_20260213 \
  --split hold \
  --rows 100000 \
  --seed 42
```

### Metrics and downstream evaluation

`tvae-recalc-metrics`

Purpose: recompute core distribution metrics for a trained run.

```bash
tvae-recalc-metrics \
  --run-dir outputs/save_data/ryc/tht_tripgen_20260213 \
  --split hold \
  --no-plots
```

`tvae-recalc-downstream`

Purpose: run fare-prediction downstream comparison on synthetic vs real hold/val split.

```bash
tvae-recalc-downstream \
  --run-dir outputs/save_data/ryc/tht_tripgen_20260213 \
  --split hold \
  --model gbr \
  --train-rows 50000 \
  --test-rows 20000
```

### Consolidated exports

`tvae-export-metrics`

Purpose: merge per-split metric files into one canonical structure.

`tvae-export-training`

Purpose: merge training checkpoints/logs/scalars into one canonical structure.

`tvae-analyze-exports`

Purpose: build a long-format analysis view plus plots and tooltips.

### Orchestrator

`tvae-run-full-pipeline`

Purpose: run topology check/build + training + sampling + metrics + downstream + exports.

Most used options:

1. `--tht-tripgen-run-dir` (or `--strategy-run-dir`) : explicit run folder.
1. `--build-embeddings`, `--skip-embeddings`
1. `--force-train`, `--force-sample`
1. `--downstream-model {gbr,hgbr,linear}`
1. `--downstream-split {val,hold}`
1. `--downstream-train-rows`, `--downstream-test-rows`
1. `--export-metrics` / `--no-export-metrics`
1. `--export-training-info` / `--no-export-training-info`
1. `--training-export-parse-logs`
1. `--analyze-exports`
1. `--analysis-no-plots`
1. `--skip-downstream`, `--skip-plots`
1. `--device`

## Recommended Run Recipes

### A. Fresh full run (recommended)

```bash
# Step 1: convert data
tvae-convert-latlon \
  --trip-parquet data/yellow_tripdata_2024-04.parquet \
  --trip-parquet data/yellow_tripdata_2024-05.parquet \
  --zones data/taxi_zones.parquet \
  --output data/viagens_lat_long.parquet

# Step 2: build physical adjacency

tvae-build-phys-edges \
  --zones data/taxi_zones.parquet \
  --out data/topology/tht_phys_edges_S1.csv \
  --split S1 \
  --adjacency queen \
  --gap-tol 5.0 \
  --bridge-max-dist 2500 \
  --adjacency-crs EPSG:3857

# Step 3: full pipeline

tvae-run-full-pipeline \
  --tht-tripgen-run-dir outputs/save_data/ryc/tht_tripgen_first \
  --build-embeddings \
  --downstream-model gbr \
  --downstream-train-rows 50000 \
  --downstream-test-rows 20000 \
  --analyze-exports \
  --analysis-dir outputs/save_data/ryc/tht_tripgen_first/analysis
```

### B. Fast smoke run

Use reduced configuration in code or environment, keep outputs to one folder, and validate
control flow only.

```bash
tvae-run-full-pipeline \
  --tht-tripgen-run-dir outputs/save_data/ryc/tht_tripgen_smoke \
  --build-embeddings \
  --force-train \
  --force-sample \
  --skip-plots \
  --analyze-exports \
  --analysis-no-plots
```

### C. Resume or reproduce an existing run

```bash
# Reuse model + cached artifacts, no retrain
tvae-run-full-pipeline \
  --tht-tripgen-run-dir outputs/save_data/ryc/tht_tripgen_first

# Force retrain or resample only
tvae-run-full-pipeline \
  --tht-tripgen-run-dir outputs/save_data/ryc/tht_tripgen_first \
  --force-train

tvae-run-full-pipeline \
  --tht-tripgen-run-dir outputs/save_data/ryc/tht_tripgen_first \
  --force-sample
```

## Configuration

All runtime behavior is controlled by `src/tvae/config.py`.

Core values to inspect before first run:

1. `REAL_DATA_PATH`, `DATETIME_COL`, `PICKUP_ID_COL`, `DROPOFF_ID_COL`
1. `FILTER_YEAR`, `FILTER_MONTHS`, `FILTER_DOW_MIN`, `FILTER_DOW_MAX`
1. `THT_SPLIT_STRATEGY` (`S1` or `S2`)
1. `THT_TRAIN_FRAC`, `THT_VAL_FRAC`
1. `THT_TIME_BINS_H`
1. `THT_PHYS_EDGES_CSV`
1. `THT_USE_PASSENGER_COUNT`, `THT_USE_TOTAL_AMOUNT`
1. `THT_NODE2VEC_*` and `THT_GFUNC_TOPK`
1. `THT_EPOCHS`, `THT_BATCH_SIZE`, `THT_LR`
1. `PRIVACY_MATCH_R_DECIMALS`, `PRIVACY_REJECTION_*`

## Output Structure and File Map

### Global outputs

- `outputs/topology_cache/` : `Efunc.pt`, `Ephys.pt`, `Ecomb.pt`, `metadata.json`
- `outputs/logs/<run_name>/` : text logs and training side outputs.
- `outputs/graficos/<run_name>/` : quick diagnostic charts.
- `outputs/plots/` : standalone plot directories used by tooling.
- `outputs/save_data/ryc/<run_name>/` : one folder per pipeline run.

### Per-run structure (`outputs/save_data/ryc/<run_name>/`)

- `tht_tripgen.pt` : model checkpoint.
- `mappings_tht_tripgen.json` : index mappings used for encoding/decoding.
- `loss_tht_tripgen.csv` : epoch-level loss history.
- `metrics/` : all split metric JSON/CSV outputs.
- `data/synthetic_tht_tripgen_<split>.csv` : generated synthetic samples.
- `logs/` : run-local scalar exports (for example `scalars.csv`).
- `training/` : `training_export.json` and `training_export.csv`.
- `plots/` : metric plots from recalc and training stages.
- `analysis/` : analysis report package.

### Important metric files in `metrics/`

- `metrics_tht_tripgen_val.json` : synthetic-to-real metrics for val split.
- `metrics_tht_tripgen_hold.json` : synthetic-to-real metrics for hold split.
- `metrics_hold_vs_val_real.json` : real hold-vs-real val comparison for drift diagnostics.
- `privacy_report_tht_tripgen_<split>.json` : privacy diagnostics.
- `downstream_tht_tripgen_<split>.json` : downstream fare prediction outputs.
- `topk_tht_tripgen_<split>_*.csv` : top-K mode tables.
- `metrics_export.json` and `metrics_export.csv` : consolidated metrics export.

## Exports and External Analysis

### `tvae-export-metrics`

Generates run-level consolidated files:

- `metrics_export.json`
- `metrics_export.csv`

Schema summary fields:

- `schema_version` (current: `3.0.0`)
- `pipeline_context`
- `summary` with section load state
- `sections` with one payload per strategy split
- `tables` for discovered top-k tables

### `tvae-export-training`

Generates run-level consolidated files:

- `training_export.json`
- `training_export.csv`

Schema summary fields:

- `schema_version` (current: `3.0.0`)
- `pipeline_context`
- `summary` with section counts and event counts
- `time_series` for loss/log/scalar timelines
- `sections` with model/checkpoint/mappings/log provenance

### `tvae-analyze-exports`

Builds a full analysis package for BI tools:

- `analysis/analysis_report.json`
- `analysis/analysis_summary.csv`
- `analysis/analysis_metrics_long.csv`
- `analysis/analysis_training_long.csv`
- `analysis/analysis_tooltips.csv`
- `analysis/analysis_report.html`
- `analysis/figs/comparison_metrics.png`
- `analysis/figs/comparison_leakage.png`
- `analysis/figs/training_curves.png` (if training timeline exists)

Use `jq` and `pandas` for quick checks:

```bash
jq '.summary, .metric_comparisons[:5]' outputs/save_data/ryc/tht_tripgen_first/analysis/analysis_report.json
python - <<'PY'
import pandas as pd
print(pd.read_csv('outputs/save_data/ryc/tht_tripgen_first/analysis/analysis_summary.csv').head())
PY
```

## Monitoring and Troubleshooting

Most frequent startup issues:

1. Entry point not found
   - Run commands with `PYTHONPATH=src python -m <module>` or install with `pip install -e .`.

2. Missing input parquet for splits
   - Verify `config.REAL_DATA_PATH` exists.
   - Re-run `tvae-convert-latlon` and set `--output` explicitly.

3. Missing topology cache
   - Run `tvae-build-zone-embeddings` before training or remove `--skip-embeddings`.

4. Empty validation/hold split
   - Review `FILTER_*` constraints and split fractions.

5. Geometry failures
   - Add `--fix-geoms make_valid_if_available`.
   - Confirm zone CRS matches your intended `--adjacency-crs`.

6. Permission or path warnings
   - Ensure write access to `outputs/`.
   - Verify the selected run folder exists and is writable.

## Testing and Validation

Smoke and regression checks:

1. `pytest tests/test_full_pipeline_smoke.py -q`
2. `pytest tests/test_export_run_metrics.py -q`
3. `pytest tests/test_export_training_info.py -q`
4. `pytest tests/test_analyze_run_exports.py -q`

For quick regression focused on training/output contracts:

```bash
pytest tests/test_train_tvae_outputs.py -q
pytest tests/test_train_tht_tripgen_smoke.py -q
```

Use targeted tests whenever you touch command plumbing.

## Conventions and Extension Points

1. Default export formats are JSON + CSV for machine consumption.
2. Run directories become deterministic when you pass `--tht-tripgen-run-dir`.
3. Scripts are documented through `--help` and support fallback execution via `PYTHONPATH=src`.
4. Legacy Portuguese field names in some outputs can remain for compatibility with historical artifacts.

For local utility scans and dependency tracing:

```bash
PYTHONPATH=src python -m tvae.tools.compare_models --help
PYTHONPATH=src python -m tvae.tools.list_pipeline_files src/tvae/tools/run_full_pipeline.py
PYTHONPATH=src python -m tvae.tools.collect_requirements
```
