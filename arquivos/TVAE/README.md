# TVAE

Projeto para treinamento, amostragem e avaliacao de dados sinteticos de transporte. Existem dois modelos principais:

- **TVAE autoregressivo**: modela `pickup_id`, `dropoff_id`, `dia_da_semana`, `hora_do_dia`.
- **Strategy A**: modela tempo -> origem -> destino, com embeddings de zona (funcional + fisico).

O foco e comparar dados sinteticos vs reais em distribuicoes marginais, espaciais (OD) e espaco-temporais, alem de metricas do paper (W1 temporal, graph similarity, coverage, DCR).

## Pipeline (visao macro)

1) **Leitura e filtragem do dado** (config + loaders): aplica filtros temporais e cria colunas derivadas.
2) **Splits** (train/val/hold) com estrategia semanal (TVAE) e S1/S2 (Strategy A).
3) **Grafos**: fisico (adjacencia) e funcional (OD top-k) + **Node2Vec**.
4) **Visualizacao**: mapas de adjacencia fisica e grafos funcionais direcionais.
5) **Modelos e treino**: TVAE e Strategy A.
6) **Amostragem / inferencia**: gera CSVs sinteticos.
7) **Analises e metricas**: marginais, OD, temporal, joint e metricas do paper.
8) **Experimentos e comparacoes**: multiplos runs e tabelas resumidas.

## Estrutura principal

- `config.py`: configuracao central (paths, filtros, splits, hiperparametros).
- `data_processing/`: loaders e transformadores.
- `models/`: definicoes do TVAE e Strategy A.
- `topology/`: grafos, node2vec e visualizacao.
- `tools/`: scripts utilitarios (grafos, embeddings, visualizacao, pipeline).
- `train_tvae.py`, `train_strategy_a.py`: treino.
- `sample_tvae.py`, `sample_strategy_a.py`: amostragem.
- `recalc_metrics_hold.py`, `recalc_metrics_strategy_a_hold.py`, `analyze_spatial.py`: avaliacao.
- `compare_runs.py`, `tools/compare_models.py`, `tools/run_full_pipeline.py`: experimentos.

## Etapas detalhadas

### 1) Dados e processamento

**Scripts chave**
- `data_processing/loader.py` (TVAE): leitura + filtros + split semanal.
- `data_processing/strategy_a_loader.py`: leitura + filtros + split S1/S2 + bins de tempo.

**Configuracoes importantes (config.py)**
- Paths: `REAL_DATA_PATH`, `DATETIME_COL`, `PICKUP_ID_COL`, `DROPOFF_ID_COL`.
- Filtros: `FILTER_YEAR`, `FILTER_MONTHS`, `FILTER_START_DATE`, `FILTER_END_DATE`, `FILTER_DOW_MIN/MAX`.
- Strategy A: `SA_TIME_BINS_H`, `SA_SPLIT_STRATEGY`, `SA_TRAIN_FRAC`, `SA_VAL_FRAC`, `SA_FILTER_*`.

**Comando sugerido (sanity check de split)**

```bash
python - <<'PY'
from data_processing.loader import load_and_split
from data_processing.strategy_a_loader import load_and_split_strategy_a

tr, va, ho = load_and_split()
print('[TVAE] train/val/hold:', len(tr), len(va), len(ho))

tr, va, ho = load_and_split_strategy_a(split='S1')
print('[Strategy A S1] train/val/hold:', len(tr), len(va), len(ho))
PY
```

### 2) Grafos fisicos (adjacencia) + Node2Vec

**Scripts chave**
- `tools/build_sa_phys_edges_csv.py`: gera CSV de adjacencia fisica (LocationID e indices).
- `tools/build_zone_embeddings.py`: cria Efunc/Ephys/Ecomb via Node2Vec.

**Flags importantes (fisico)**
- `--adjacency` (rook/queen)
- `--gap-tol` (tolerancia em metros quando CRS metrico)
- `--bridge-max-dist` (conectar componentes desconexos)
- `--adjacency-crs EPSG:3857` (distancias em metros)
- `--mappings` (garante indices iguais ao treino Strategy A)

**Comando sugerido (fisico + tolerancia + bridge)**

```bash
python tools/build_sa_phys_edges_csv.py \
  --zones data/taxi_zones.parquet \
  --out outputs/sa_phys_edges.csv \
  --mappings outputs/save_data/strategy_a/mappings_strategy_a.json \
  --adjacency queen \
  --gap-tol 5.0 \
  --bridge-max-dist 2500 \
  --adjacency-crs EPSG:3857 \
  --fix-geoms make_valid_if_available
```

**Comando sugerido (node2vec)**

```bash
python tools/build_zone_embeddings.py --device cpu --alpha 1.0 --beta 1.0
```

### 3) Visualizacao de grafos

**Adjacencia fisica**
- `tools/inspect_sa_phys_graph.py`: estatisticas, near-miss e isolados.
- `tools/viz_sa_phys_graph.py`: mapas (rook/queen/diff), com tolerancia e bridge.

**Comando sugerido (fisico exatamente como o treino)**

```bash
python tools/viz_sa_phys_graph.py \
  --zones data/taxi_zones.parquet \
  --out-dir outputs/phys_viz \
  --edges outputs/sa_phys_edges.csv \
  --adjacency queen
```

**Comando sugerido (fisico com gap-tol + bridge)**

```bash
python tools/viz_sa_phys_graph.py \
  --zones data/taxi_zones.parquet \
  --out-dir outputs/phys_viz \
  --adjacency-crs EPSG:3857 \
  --gap-tol 5.0 \
  --bridge-max-dist 2500 \
  --edge-style both \
  --atlas-mode none
```

**Grafo funcional direcionado e ponderado**
- `tools/viz_sa_func_graph.py`: setas com peso, threshold e filtro por hora.

**Comando sugerido (funcional, limiar de peso)**

```bash
python tools/viz_sa_func_graph.py \
  --zones data/taxi_zones.parquet \
  --out-dir outputs/func_viz \
  --min-weight 0.7
```

**Comando sugerido (funcional por hora)**

```bash
python tools/viz_sa_func_graph.py \
  --zones data/taxi_zones.parquet \
  --out-dir outputs/func_viz_hour8 \
  --hour 8 \
  --min-weight 0.7
```

Notas:
- `hora_do_dia` usa bins `0..SA_TIME_BINS_H-1` (por padrao 24).
- `--hour`, `--hours` e `--hour-range` permitem slices temporais.
- `--min-weight` e `--max-edges` controlam densidade do grafo.

### 4) Treino dos modelos

**TVAE (autoregressivo)**
- Script: `train_tvae.py` (sem CLI; usa `config.py`).
- Saidas: `SAVE_DATA_DIR/<run>/` com checkpoint, mappings e metricas.

**Comando sugerido**

```bash
python train_tvae.py
```

**Strategy A**
- Script: `train_strategy_a.py`.
- Flags: `--run-tag`, `--device`.
- Usa embeddings de zona em `SA_TOPOLOGY_CACHE_DIR` (Ecomb/Efunc).

**Comando sugerido**

```bash
python train_strategy_a.py --run-tag strategy_a --device cuda
```

### 5) Amostragem / inferencia

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

**Strategy A**
- Script: `sample_strategy_a.py`.
- Flags: `--run-dir`, `--split`, `--rows`, `--device`.

```bash
python sample_strategy_a.py \
  --run-dir outputs/save_data/strategy_a \
  --split hold \
  --rows 100000
```

### 6) Analise e metricas

**Scripts chave**
- `recalc_metrics_hold.py`: recalcula metricas TVAE no hold.
- `recalc_metrics_strategy_a_hold.py`: recalcula metricas Strategy A.
- `analyze_spatial.py`: analise espacial detalhada (OD, graus, condicionais).

**Comandos sugeridos**

```bash
python recalc_metrics_hold.py \
  --run-dir outputs/save_data/baseline \
  --order-key order_1 \
  --no-plots
```

```bash
python recalc_metrics_strategy_a_hold.py \
  --run-dir outputs/save_data/strategy_a \
  --no-plots
```

```bash
python analyze_spatial.py \
  --mappings outputs/save_data/baseline/mappings_order_1.json \
  --checkpoint outputs/save_data/baseline/tvae_order_1.pt \
  --split val \
  --output-dir outputs/spatial
```

**Metricas principais**
- Marginais: `*_jsd`, `*_chi2`.
- OD, temporal e joint (JSD/chi2/coverage/unique).
- Paper metrics: W1 temporal, graph similarity, coverage KNN, DCR.

### 7) Experimentos e comparacoes

**Scripts chave**
- `tools/run_full_pipeline.py`: executa baseline + Strategy A + metricas.
- `tools/compare_models.py`: compara baseline vs Strategy A.
- `compare_runs.py`: comparacao multi-run (ajuste o dicionario `runs`).

**Comando sugerido (pipeline completo)**

```bash
python tools/run_full_pipeline.py --build-embeddings --device cuda
```

**Comando sugerido (comparar dois runs)**

```bash
python tools/compare_models.py \
  --baseline-run outputs/save_data/baseline \
  --strategy-a-run outputs/save_data/strategy_a \
  --out-dir outputs/comparison
```

## Observacoes rapidas

- **Indices e mapeamentos**: use sempre o `mappings_strategy_a.json` do run para alinhar grafos e treino.
- **CRS metrico**: `EPSG:3857` e recomendado quando usar `--gap-tol` ou `--bridge-max-dist`.
- **Atlas de viz**: `viz_sa_phys_graph.py` agora usa `--atlas-mode none` por padrao.
- **Outputs**: verifique `SAVE_DATA_DIR`, `OUTPUT_BASE_DIR` e `SA_TOPOLOGY_CACHE_DIR` em `config.py`.
