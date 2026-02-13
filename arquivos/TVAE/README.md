# Synthetic Trip Generation Pipeline (THT-TripGen)

## Quickstart: Fresh Install to Results

Here is a fresh-install, end-to-end pipeline that takes you from raw taxi data to
THT-TripGen results and analysis exports. It is the minimal sequence that
works on a clean machine.

0) One-time setup (venv + deps)

```bash
cd /home/rcamargo/vscode-projects/ic-backup/arquivos/TVAE
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Core deps
python -m pip install torch numpy pandas pyarrow matplotlib

# Geo deps for physical graph (needed for build_tht_phys_edges_csv)
python -m pip install geopandas shapely pyproj

# Optional (basemaps + TensorBoard)
python -m pip install contextily tensorboard
```

If geopandas fails via pip, install it with conda/mamba instead.

1) Prepare data in place + set config

- Convert raw taxi trips to centroid coordinates first:

```bash
python tools/converter_lagitude_longitude.py \
  --trip-parquet /path/to/yellow_tripdata_2024-04.parquet \
  --trip-parquet /path/to/yellow_tripdata_2024-05.parquet \
  --zones /path/to/taxi-zones \
  --output data/viagens_lat_long.parquet
```

- Dependencies for this step: `geopandas`, `shapely`, `pyproj`, `pyarrow`, `pandas`.

- Output trip file expected by pipeline:
  - `data/viagens_lat_long.parquet` (can override via `--output`; default is `config.REAL_DATA_PATH`)
- Taxi zones polygons -> `data/taxi_zones.parquet` (or .shp/.zip) for graph tooling.
- Edit `config.py` to set:
  - `REAL_DATA_PATH`, `DATETIME_COL`, `PICKUP_ID_COL`, `DROPOFF_ID_COL`
  - time filters (`FILTER_YEAR`, `FILTER_MONTHS`, etc.)
  - `THT_SPLIT_STRATEGY` (S1 or S2)
  - `THT_PHYS_EDGES_CSV` path (matches the edge build below)

2) Build physical adjacency edges (required for Ephys/Ecomb)

```bash
PYTHONPATH=. python tools/build_tht_phys_edges_csv.py \
  --zones data/taxi_zones.parquet \
  --out data/topology/tht_phys_edges_S1.csv \
  --split S1 \
  --adjacency queen \
  --gap-tol 5.0 \
  --bridge-max-dist 2500 \
  --adjacency-crs EPSG:3857 \
  --fix-geoms make_valid_if_available
```

3) Run the full experiment pipeline (THT-TripGen + exports + analysis)

```bash
python tools/run_full_pipeline.py --build-embeddings --device cuda
```

That command will:

- build Node2Vec embeddings (Efunc/Ephys/Ecomb)
- train THT-TripGen
- recalc metrics on hold split and hold-vs-val split
- generate consolidated exports and analysis artifacts

Project for training, sampling, and evaluation of synthetic transportation data.

- **THT-TripGen**: models time -> origin -> destination, with zone embeddings (functional + physical).

The focus is to compare synthetic vs real data across marginal, spatial (OD), and spatiotemporal distributions, plus paper metrics (temporal W1, graph similarity, coverage, DCR).

## Pipeline (high-level)

1) **Data reading and filtering** (config + loaders): apply time filters and derive columns.
2) **Splits** (train/val/hold) with S1/S2 strategy (THT-TripGen).
3) **Graphs**: physical adjacency and functional (OD top-k) + **Node2Vec**.
4) **Visualization**: physical adjacency maps and directed functional graphs.
5) **Models and training**: THT-TripGen.
6) **Sampling / inference**: generate synthetic CSVs.
7) **Analysis and metrics**: marginal, OD, temporal, joint, and paper metrics.
8) **Experiments and comparisons**: multi-run comparisons and summary tables.

## Top-level structure

- `config.py`: central configuration (paths, filters, splits, hyperparameters).
- `data_processing/`: loaders and transformers.
- `models/`: THT-TripGen definitions.
- `topology/`: graphs, node2vec, and visualization.
- `tools/`: utilities (graphs, embeddings, visualization, pipeline).
- `tools/converter_lagitude_longitude.py`: raw trip converter to `viagens_lat_long.parquet`.
- `train_tht_tripgen.py`: training.
- `sample_tht_tripgen.py`: sampling.
- `recalc_metrics_tht_tripgen_hold.py`, `analyze_spatial.py`: evaluation.
- `compare_runs.py`, `tools/compare_models.py`, `tools/run_full_pipeline.py`: experiments.

## Detailed stages

### 1) Data and processing

**Key scripts**
- `data_processing/tht_tripgen_loader.py`: read + filters + S1/S2 split + time bins.

**Important config (config.py)**
- Paths: `REAL_DATA_PATH`, `DATETIME_COL`, `PICKUP_ID_COL`, `DROPOFF_ID_COL`.
- Filters: `FILTER_YEAR`, `FILTER_MONTHS`, `FILTER_START_DATE`, `FILTER_END_DATE`, `FILTER_DOW_MIN/MAX`.
- THT-TripGen: `THT_TIME_BINS_H`, `THT_SPLIT_STRATEGY`, `THT_TRAIN_FRAC`, `THT_VAL_FRAC`, `THT_FILTER_*`.

**Suggested command (split sanity check)**

```bash
python - <<'PY'
from data_processing.tht_tripgen_loader import load_and_split_tht_tripgen

tr, va, ho = load_and_split_tht_tripgen(split='S1')
print('[THT-TripGen S1] train/val/hold:', len(tr), len(va), len(ho))
PY
```

### 2) Physical graphs (adjacency) + Node2Vec

**Key scripts**
- `tools/build_tht_phys_edges_csv.py`: generate physical adjacency CSV (LocationID and indices).
- `tools/build_tht_zone_embeddings.py`: build Efunc/Ephys/Ecomb via Node2Vec.

**Important flags (physical)**
- `--adjacency` (rook/queen)
- `--gap-tol` (tolerance in meters when CRS is metric)
- `--bridge-max-dist` (connect disconnected components)
- `--adjacency-crs EPSG:3857` (distances in meters)
- `--mappings` (align indices with THT-TripGen training)

**Suggested command (physical + tolerance + bridge)**

```bash
python tools/build_tht_phys_edges_csv.py \
  --zones data/taxi_zones.parquet \
  --out outputs/tht_phys_edges.csv \
  --mappings outputs/save_data/tht_tripgen/mappings_tht_tripgen.json \
  --adjacency queen \
  --gap-tol 5.0 \
  --bridge-max-dist 2500 \
  --adjacency-crs EPSG:3857 \
  --fix-geoms make_valid_if_available
```

**Suggested command (node2vec)**

```bash
python tools/build_tht_zone_embeddings.py --device cpu --alpha 1.0 --beta 1.0
```

### 3) Graph visualization

**Physical adjacency**
- `tools/inspect_tht_phys_graph.py`: stats, near-miss and isolated nodes.
- `tools/viz_tht_phys_graph.py`: maps (rook/queen/diff), with tolerance and bridge.

**Suggested command (physical, exactly as training)**

```bash
python tools/viz_tht_phys_graph.py \
  --zones data/taxi_zones.parquet \
  --out-dir outputs/phys_viz \
  --edges outputs/tht_phys_edges.csv \
  --adjacency queen
```

**Suggested command (physical with gap-tol + bridge)**

```bash
python tools/viz_tht_phys_graph.py \
  --zones data/taxi_zones.parquet \
  --out-dir outputs/phys_viz \
  --adjacency-crs EPSG:3857 \
  --gap-tol 5.0 \
  --bridge-max-dist 2500 \
  --edge-style both \
  --atlas-mode none
```

**Directed weighted functional graph**
- `tools/viz_tht_func_graph.py`: arrows + weights + threshold + time filters.

**Suggested command (functional, weight threshold)**

```bash
python tools/viz_tht_func_graph.py \
  --zones data/taxi_zones.parquet \
  --out-dir outputs/func_viz \
  --min-weight 0.7
```

**Suggested command (functional by hour)**

```bash
python tools/viz_tht_func_graph.py \
  --zones data/taxi_zones.parquet \
  --out-dir outputs/func_viz_hour8 \
  --hour 8 \
  --min-weight 0.7
```

Notes:
- `hour_of_day` uses bins `0..THT_TIME_BINS_H-1` (default 24).
- `--hour`, `--hours`, and `--hour-range` allow time slices.
- `--min-weight` and `--max-edges` control graph density.

### 4) Model training

**THT-TripGen**
- Script: `train_tht_tripgen.py`.
- Flags: `--run-tag`, `--device`.
- Uses zone embeddings in `THT_TOPOLOGY_CACHE_DIR` (Ecomb/Efunc).

**Suggested command**

```bash
python train_tht_tripgen.py --run-tag tht_tripgen --device cuda
```

### 5) Sampling / inference

**THT-TripGen**
- Script: `sample_tht_tripgen.py`.
- Flags: `--run-dir`, `--split`, `--rows`, `--device`.

```bash
python sample_tht_tripgen.py \
  --run-dir outputs/save_data/tht_tripgen \
  --split hold \
  --rows 100000
```

### 6) Analysis and metrics

**Key scripts**
- `recalc_metrics_tht_tripgen_hold.py`: recompute THT-TripGen metrics.
- `analyze_spatial.py`: legacy TVAE spatial helper. THT-TripGen scripts already write route/pattern metrics in
  `metrics` and `privacy` JSON outputs.
- `tools/export_run_metrics.py`: exports all available run metrics into one JSON/CSV bundle.

**Suggested commands**

```bash
python recalc_metrics_tht_tripgen_hold.py \
  --run-dir outputs/save_data/tht_tripgen \
  --no-plots
```

**Key metrics**
- Marginals: `*_jsd`, `*_chi2`.
- OD, temporal, and joint (JSD/chi2/coverage/unique).
- Paper metrics: temporal W1, graph similarity, KNN coverage, DCR.

### Consolidated metrics export

For each run, use one consolidated summary file bundle:

- `.../<run-dir>/metrics/metrics_export.json` (full nested payload, include file provenance)
- `.../<run-dir>/metrics/metrics_export.csv` (flat key/value table for external tools)

Where to find the summary in `metrics_export.json`:

- `schema_version`: current export schema version (`3.0.0`)
- `pipeline_context`: run directories and optional labels used during export
- `summary`: compact counts and section discovery status
  - `n_sections`, `n_loaded`, `n_missing`
  - `loaded_sections`, `missing_sections`
- `n_topk_tables`, `n_topk_rows`
- `sections`: per-section payloads for each metric file (`strategy_*`)
- `tables.topk`: metadata entries for each discovered top-k table CSV

```bash
python tools/export_run_metrics.py \
  --strategy-run-dir /path/to/outputs/save_data/ryc/tht_tripgen_20260212_234749 \
  --output-json /path/to/outputs/save_data/ryc/tht_tripgen_20260212_234749/metrics/metrics_export.json \
  --output-csv /path/to/outputs/save_data/ryc/tht_tripgen_20260212_234749/metrics/metrics_export.csv
```

`run_full_pipeline.py` writes metrics summaries by default (`--no-export-metrics` to disable):

```bash
python tools/run_full_pipeline.py \
  --tht-tripgen-run-dir /path/to/outputs/save_data/ryc/tht_tripgen_20260212_234749 \
  --build-embeddings \
  --device cuda
```

Optional override flags for export:

- `--metrics-export-dir`: export directory (default: `<strategy-run-dir>/metrics`)
- `--metrics-export-json`: explicit JSON output path
- `--metrics-export-csv`: explicit CSV output path
- `--metrics-export-strict`: fail if expected files are missing
- `--metrics-export-label`: label embedded in bundle summary

### Consolidated training export

For provenance and training-trace analysis (checkpoints, mappings, losses, scalars, log parsing, and artifact hashes), use:

- `.../<strategy-run-dir>/training/training_export.json`
- `.../<strategy-run-dir>/training/training_export.csv`

Where to find the summary in `training_export.json`:

- `schema_version`: current export schema version (`3.0.0`)
- `pipeline_context`: run directories and optional labels used during export
- `summary`: compact counts and coverage for artifact availability
  - `n_sections`, `n_loaded`, `n_missing`
  - `n_loss_rows`, `n_scalar_points`, `n_log_events`
  - `loaded_sections`, `missing_sections`
- `time_series`: compact series metadata (`loss`, `scalars`, `log`) for each section
- `sections`: per-section payloads for strategy training artifacts

```bash
python tools/export_training_info.py \
  --strategy-run-dir /path/to/outputs/save_data/ryc/tht_tripgen_20260212_234749 \
  --training-export-json /path/to/outputs/save_data/ryc/tht_tripgen_20260212_234749/training/training_export.json \
  --training-export-csv /path/to/outputs/save_data/ryc/tht_tripgen_20260212_234749/training/training_export.csv
```

`run_full_pipeline.py` also writes this bundle by default (`--no-export-training-info` to disable). Optional flags:

- `--training-export-dir`: export directory (default: `<strategy-run-dir>/training`)
- `--training-export-json`: explicit JSON path
- `--training-export-csv`: explicit CSV path
- `--training-export-strict`: fail if required artifacts are missing
- `--training-export-parse-logs`: include epoch-level parsed events from text logs
- `--training-export-label`: label embedded in bundle summary

### Full run analysis export

`tools/analyze_run_exports.py` ingests the metrics and training export JSON files and produces a richer bundle for plotting and external analysis:

- `.../<strategy-run-dir>/analysis/analysis_report.json`
- `.../<strategy-run-dir>/analysis/analysis_summary.csv`
- `.../<strategy-run-dir>/analysis/analysis_metrics_long.csv`
- `.../<strategy-run-dir>/analysis/analysis_training_long.csv`
- `.../<strategy-run-dir>/analysis/analysis_tooltips.csv`
- `.../<strategy-run-dir>/analysis/analysis_report.html`
- `.../<strategy-run-dir>/analysis/figs/` (optional PNG plots)

```bash
python tools/analyze_run_exports.py \
  --run-dir /path/to/outputs/save_data/ryc/tht_tripgen_20260212_234749 \
  --metrics-export-json /path/to/outputs/save_data/ryc/tht_tripgen_20260212_234749/metrics/metrics_export.json \
  --training-export-json /path/to/outputs/save_data/ryc/tht_tripgen_20260212_234749/training/training_export.json \
  --output-dir /path/to/outputs/save_data/ryc/tht_tripgen_20260212_234749/analysis
```

You can also enable this step directly in the pipeline:

```bash
python tools/run_full_pipeline.py \
  --tht-tripgen-run-dir /path/to/outputs/save_data/ryc/tht_tripgen_20260212_234749 \
  --analyze-exports \
  --analysis-dir /path/to/outputs/save_data/ryc/tht_tripgen_20260212_234749/analysis
```

Relevant `analysis_report.json` sections:

- `metric_comparisons`: one row per structural metric per direction (`synth→val`, `synth→hold`, `hold→val`) plus derived deltas.
- `metric_comparisons` now also includes first-class synthetic-vs-real comparison rows for attribute metrics:
  - passenger metrics (`{PASSENGER_COL}_jsd`, `{PASSENGER_COL}_chi2`)
  - fare metrics (`{FARE_COL}_log1p_w1`, `_log1p_ks`, `_log1p_quantile_mae`, `_log1p_median_mae_by_*`)
  - residual/extended privacy metric row (`exact_match_rate_with_r_rounded`)
- `training_summary`: copied from `training_export.json["summary"]`.
- `summary.generated_rows`: counts of generated rows in CSV outputs.
- `generated_files`: absolute/relative paths for artifacts, useful for automation.

### 7) Experiments and comparisons

**Key scripts**
- `tools/run_full_pipeline.py`: run full THT-TripGen pipeline and emit exported artifacts.
- `tools/compare_models.py` and `compare_runs.py`: optional manual comparison scripts for multi-run scoreboards.

**Skip embeddings build/check (assumes Efunc/Ecomb exist)**

```bash
python tools/run_full_pipeline.py --skip-embeddings --device cuda
```

**All optional flags in one example (for reference)**

```bash
python tools/run_full_pipeline.py \
  --skip-embeddings \
  --skip-plots \
  --force-train \
  --force-sample \
  --device cuda
```

## Quick notes

- **Indices and mappings**: always use `mappings_tht_tripgen.json` from the run to align graphs and training.
- **Metric CRS**: `EPSG:3857` is recommended when using `--gap-tol` or `--bridge-max-dist`.
- **Viz atlas**: `viz_tht_phys_graph.py` defaults to `--atlas-mode none`.
- **Outputs**: check `SAVE_DATA_DIR`, `OUTPUT_BASE_DIR`, and `THT_TOPOLOGY_CACHE_DIR` in `config.py`.
