# TVAE

Project for training, sampling, and evaluation of synthetic transportation data. There are two main models:

- **Autoregressive TVAE**: models `pickup_id`, `dropoff_id`, `dia_da_semana`, `hora_do_dia`.
- **THT-TripGen**: models time -> origin -> destination, with zone embeddings (functional + physical).

The focus is to compare synthetic vs real data across marginal, spatial (OD), and spatiotemporal distributions, plus paper metrics (temporal W1, graph similarity, coverage, DCR).

## Pipeline (high-level)

1) **Data reading and filtering** (config + loaders): apply time filters and derive columns.
2) **Splits** (train/val/hold) with weekly strategy (TVAE) and S1/S2 (THT-TripGen).
3) **Graphs**: physical adjacency and functional (OD top-k) + **Node2Vec**.
4) **Visualization**: physical adjacency maps and directed functional graphs.
5) **Models and training**: TVAE and THT-TripGen.
6) **Sampling / inference**: generate synthetic CSVs.
7) **Analysis and metrics**: marginal, OD, temporal, joint, and paper metrics.
8) **Experiments and comparisons**: multi-run comparisons and summary tables.

## Top-level structure

- `config.py`: central configuration (paths, filters, splits, hyperparameters).
- `data_processing/`: loaders and transformers.
- `models/`: TVAE and THT-TripGen definitions.
- `topology/`: graphs, node2vec, and visualization.
- `tools/`: utilities (graphs, embeddings, visualization, pipeline).
- `train_tvae.py`, `train_tht_tripgen.py`: training.
- `sample_tvae.py`, `sample_tht_tripgen.py`: sampling.
- `recalc_metrics_hold.py`, `recalc_metrics_tht_tripgen_hold.py`, `analyze_spatial.py`: evaluation.
- `compare_runs.py`, `tools/compare_models.py`, `tools/run_full_pipeline.py`: experiments.

## Detailed stages

### 1) Data and processing

**Key scripts**
- `data_processing/loader.py` (TVAE): read + filters + weekly split.
- `data_processing/tht_tripgen_loader.py`: read + filters + S1/S2 split + time bins.

**Important config (config.py)**
- Paths: `REAL_DATA_PATH`, `DATETIME_COL`, `PICKUP_ID_COL`, `DROPOFF_ID_COL`.
- Filters: `FILTER_YEAR`, `FILTER_MONTHS`, `FILTER_START_DATE`, `FILTER_END_DATE`, `FILTER_DOW_MIN/MAX`.
- THT-TripGen: `THT_TIME_BINS_H`, `THT_SPLIT_STRATEGY`, `THT_TRAIN_FRAC`, `THT_VAL_FRAC`, `THT_FILTER_*`.

**Suggested command (split sanity check)**

```bash
python - <<'PY'
from data_processing.loader import load_and_split
from data_processing.tht_tripgen_loader import load_and_split_tht_tripgen

tr, va, ho = load_and_split()
print('[TVAE] train/val/hold:', len(tr), len(va), len(ho))

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
- `hora_do_dia` uses bins `0..THT_TIME_BINS_H-1` (default 24).
- `--hour`, `--hours`, and `--hour-range` allow time slices.
- `--min-weight` and `--max-edges` control graph density.

### 4) Model training

**TVAE (autoregressive)**
- Script: `train_tvae.py` (no CLI; uses `config.py`).
- Outputs: `SAVE_DATA_DIR/<run>/` with checkpoint, mappings, metrics.

**Suggested command**

```bash
python train_tvae.py
```

**THT-TripGen**
- Script: `train_tht_tripgen.py`.
- Flags: `--run-tag`, `--device`.
- Uses zone embeddings in `THT_TOPOLOGY_CACHE_DIR` (Ecomb/Efunc).

**Suggested command**

```bash
python train_tht_tripgen.py --run-tag tht_tripgen --device cuda
```

### 5) Sampling / inference

**TVAE**
- Script: `sample_tvae.py`.
- Flags: `--checkpoint`, `--mappings`, `--rows`, `--temperature`, `--batch-size`.

```bash
python sample_tvae.py \
  --checkpoint outputs/save_data/baseline/tvae_order_1.pt \
  --mappings outputs/save_data/baseline/mappings_order_1.json \
  --rows 100000 \
  --temperature 1.0
```

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
- `recalc_metrics_hold.py`: recompute TVAE metrics on hold.
- `recalc_metrics_tht_tripgen_hold.py`: recompute THT-TripGen metrics.
- `analyze_spatial.py`: deep spatial analysis (OD, degrees, conditionals).

**Suggested commands**

```bash
python recalc_metrics_hold.py \
  --run-dir outputs/save_data/baseline \
  --order-key order_1 \
  --no-plots
```

```bash
python recalc_metrics_tht_tripgen_hold.py \
  --run-dir outputs/save_data/tht_tripgen \
  --no-plots
```

```bash
python analyze_spatial.py \
  --mappings outputs/save_data/baseline/mappings_order_1.json \
  --checkpoint outputs/save_data/baseline/tvae_order_1.pt \
  --split val \
  --output-dir outputs/spatial
```

**Key metrics**
- Marginals: `*_jsd`, `*_chi2`.
- OD, temporal, and joint (JSD/chi2/coverage/unique).
- Paper metrics: temporal W1, graph similarity, KNN coverage, DCR.

### 7) Experiments and comparisons

**Key scripts**
- `tools/run_full_pipeline.py`: run baseline + THT-TripGen + metrics.
- `tools/compare_models.py`: compare baseline vs THT-TripGen.
- `compare_runs.py`: multi-run comparison (adjust the `runs` dict).

**Suggested command (full pipeline)**

```bash
python tools/run_full_pipeline.py --build-embeddings --device cuda
```

**Suggested command (compare two runs)**

```bash
python tools/compare_models.py \
  --baseline-run outputs/save_data/baseline \
  --tht-tripgen-run outputs/save_data/tht_tripgen \
  --out-dir outputs/comparison
```

## Quick notes

- **Indices and mappings**: always use `mappings_tht_tripgen.json` from the run to align graphs and training.
- **Metric CRS**: `EPSG:3857` is recommended when using `--gap-tol` or `--bridge-max-dist`.
- **Viz atlas**: `viz_tht_phys_graph.py` defaults to `--atlas-mode none`.
- **Outputs**: check `SAVE_DATA_DIR`, `OUTPUT_BASE_DIR`, and `THT_TOPOLOGY_CACHE_DIR` in `config.py`.
