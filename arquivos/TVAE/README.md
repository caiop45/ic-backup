# TVAE

Este diretorio contem o pipeline de treinamento, amostragem e avaliacao de um TVAE
autoregressivo aplicado a dados de transporte. O modelo trabalha com quatro colunas
categoricas: `pickup_id`, `dropoff_id`, `dia_da_semana`, `hora_do_dia`.

O objetivo principal e comparar o dado sintetico com o real nas distribuicoes
marginais, espaciais (OD) e espaco-temporais, alem das metricas do paper
(W1 temporal, graph similarity e coverage).

## Estrutura do diretorio (top-level)

- `config.py`: configuracao central (paths, filtros, splits, hiperparametros, pesos).
- `train_tvae.py`: entrypoint principal de treino e avaliacao.
- `train_strategy_a.py`: entrypoint de treino e avaliacao do Strategy A.
- `models/tvae_ar.py`: definicao do modelo TVAE autoregressivo.
- `models/strategy_a.py`: definicao do modelo Strategy A (factorizacao tempo->origem->destino).
- `data_processing/loader.py`: leitura do parquet, filtros temporais e split semanal.
- `data_processing/strategy_a_loader.py`: leitura do parquet, filtros temporais e splits Strategy A.
- `data_processing/transformer.py`: mapeamento categoria->indice, one-hot e decode.
- `data_processing/strategy_a_transformer.py`: vocabulario compartilhado + indices Strategy A.
- `utils/metrics.py`: metricas (JSD, chi2, OD/time/joint, W1, graph similarity, coverage) e plots.
- `utils/evaluation.py`: metricas compartilhadas (pipeline + paper metrics).
- `utils/serialization.py`: salvar/carregar checkpoints e mappings.
- `utils/helpers.py`: seeds e funcoes auxiliares.
- `sample_tvae.py`: gera amostras sinteticas a partir de um checkpoint.
- `sample_strategy_a.py`: gera amostras sintéticas do Strategy A condicionadas em u.
- `analyze_spatial.py`: avaliacao espacial detalhada (OD, graus, condicionais).
- `recalc_metrics_hold.py`: recalcula metricas usando hold como real.
- `recalc_metrics_strategy_a_hold.py`: recalcula metricas Strategy A (hold/val).
- `compare_runs.py`: compara runs, gera plots padronizados e tabela de metricas.
- `tools/compare_models.py`: compara baseline vs Strategy A em uma tabela unica.
- `tools/`: scripts utilitarios para inspecao de dependencias e arquivos do pipeline.
- `graficos/`: plots gerados (pode conter execucoes antigas).
- `logs/`: logs de treino (pode conter execucoes antigas).
- `save_data/`: artefatos locais (ex.: `tvae_order_1.pt`) e `save_data_ext/` com runs antigos.
- `comparacao_graficos/`: comparacoes multi-run geradas por `compare_runs.py`.
- `sample/` e `train/`: pastas vazias (reservadas).
- `2502.08856v1 (1).pdf`: paper de referencia.

Arquivos de debug/temporarios nao sao documentados aqui.

## Fluxo de dados (end-to-end)

1) **Leitura do dado bruto** (`data_processing/loader.py`)
   - Le `REAL_DATA_PATH` (parquet).
   - Converte `DATETIME_COL` para datetime.
   - Remove linhas sem datetime/pickup/dropoff.
   - Aplica filtros: ano/mes/dia-da-semana e janela de datas.
   - Cria colunas derivadas:
     - `hora_do_dia` = hora do timestamp.
     - `dia_da_semana` = dayofweek.
     - `pickup_id` e `dropoff_id` como inteiros.

2) **Split temporal** (`split_dataset_weekly`)
   - Ordena por datetime.
   - Agrupa por ISO week.
   - Usa `TRAIN_FRAC` e `VAL_FRAC` para alocar semanas em train/val.
   - O restante vai para hold.

3) **Transformacao categorica** (`data_processing/transformer.py`)
   - `CategoricalTransformer.fit()` ordena categorias observadas no train.
   - `transform()` mapeia ids reais para indices contiguos.
   - `one_hot_encode()` cria vetor de entrada do encoder.
   - `decode_indices()` converte indices para ids reais.

4) **Treino do VAE autoregressivo** (`train_tvae.py`)
   - `_prepare_dataloaders()` monta `TensorDataset` com:
     - `x_onehot`: entrada do encoder.
     - `y_indices`: indices reais por coluna (teacher forcing).
   - `TVAEAutoregressive` (encoder + decoder por cabecas).
   - `_epoch_pass()` calcula perdas e faz backprop.
   - `train_single_order()` executa loop de epocas com early stopping.

5) **Amostragem de sinteticos**
   - `_sample_synthetic()` gera `z ~ N(0, I)` e amostra sequencialmente:
     `pickup_id -> dropoff_id -> dia_da_semana -> hora_do_dia`.
   - Decodifica indices para ids reais.

6) **Metricas e artefatos**
   - `utils.evaluation.compute_metrics()` gera JSD/chi2 e metricas OD/time/joint.
   - `utils.evaluation.compute_paper_metrics()` gera W1, graph similarity e coverage.
   - Salva CSVs, JSONs e graficos no output do run.

## Arquitetura do modelo (models/tvae_ar.py)

**Encoder**
- Entrada: vetor one-hot concatenado das 4 colunas.
- MLP com `ENCODER_HIDDEN_DIMS`.
- Saidas: `mu` e `logvar` para latente `z`.

**Latente**
- Reparametrizacao: `z = mu + eps * exp(0.5 * logvar)`.

**Decoder autoregressivo**
- Uma cabeca MLP por coluna, na ordem definida em `config.ORDERS`.
- Cada cabeca recebe `z` concatenado com o contexto one-hot das colunas anteriores.
- Treino: `decode_teacher_forcing()` usa valores reais no contexto.
- Geracao: `sample()` usa as predicoes do proprio modelo.

## Perdas e regularizadores (train_tvae.py)

**Perda principal**
- Soma de cross-entropies por coluna (cada cabeca do decoder).
- Pesos por coluna:
  - `PICKUP_LOSS_WEIGHT`
  - `DROPOFF_LOSS_WEIGHT`

**KL do VAE**
- `kld = -0.5 * mean(1 + logvar - mu^2 - exp(logvar))`
- Annealing de `beta` via `_kl_beta()` (configura `KL_BETA_START/END`).

**Regularizador KL condicional**
- `PAIR_KL_WEIGHT`: penaliza divergencia entre
  `p(dropoff|pickup)` do modelo e do real.
- `PICKUP_KL_WEIGHT`: penaliza divergencia entre
  `p(pickup)` do modelo e do real.
- `PAIR_KL_EPS` e `PICKUP_KL_EPS` controlam suavizacao numerica.

## Metricas calculadas

**Marginais**
- `*_jsd` e `*_chi2` para cada coluna (`hora_do_dia`, `dia_da_semana`, `pickup_id`, `dropoff_id`).
- Graficos: `hist_<order_key>_<col>.png`.
- Top-k: `topk_<order_key>_<col>.csv` + `topk_<order_key>_<col>.png` (apenas pickup/dropoff).

**OD (pickup x dropoff)**
- `od_jsd`, `od_chi2`, `od_coverage_real`, `od_coverage_synth`,
  `od_unique_real`, `od_unique_synth`.

**Temporal (dia x hora)**
- `time_jsd`, `time_chi2`, `time_coverage_real`, `time_coverage_synth`,
  `time_unique_real`, `time_unique_synth`.

**Joint (OD x dia x hora)**
- `joint_jsd`, `joint_chi2`, `joint_coverage_real`, `joint_coverage_synth`,
  `joint_unique_real`, `joint_unique_synth`,
  `joint_mode_dropping_ratio`, `joint_invalid_ratio`.

**Metricas do paper**
- W1 temporal: `w1_tr_te`, `w1_tr_syn`, `w1_te_syn`.
- Graph similarity: `g_tr_te`, `g_tr_syn`, `g_te_syn` (valores * 100).
- Coverage KNN: `cov_tr_te`, `cov_tr_syn`, `cov_te_syn` (porcentagem).
- Privacidade (DCR): `dcr_tr_syn_p05`, `dcr_hold_syn_p05`, `rdcr_p05`.

## Scripts principais (o que fazem)

- `train_tvae.py`
  - Treina e avalia o TVAE.
  - Salva checkpoints, mappings, metricas e plots.
  - Gera baseline train_vs_val (`metrics_train_vs_val.json` e plots).

- `sample_tvae.py`
  - Amostra sinteticos a partir de um checkpoint e mappings.
  - Salva CSV com `hora_do_dia`, `dia_da_semana`, `pickup_id`, `dropoff_id`.

- `recalc_metrics_hold.py`
  - Recalcula metricas usando o hold como real.
  - Gera `metrics/metrics_tvae_order_1_hold.json` e plots no diretorio escolhido.
  - Recomendado: treine uma vez, depois ajuste métricas e rode o recalc reutilizando
    o sintético salvo (use `--force-sample` apenas quando quiser reamostrar).

- `analyze_spatial.py`
  - Analise espacial profunda:
    - JSD/chi2 para pickup, dropoff e OD.
    - Cobertura OD e massa invalida/ausente.
    - Distribuicao de graus (in/out).
    - Tabelas de top-k (over/under).
    - JSD condicional para top N origens/destinos.
  - Salva `spatial_metrics.json` e varios CSVs auxiliares.

- `compare_runs.py`
  - Padroniza plots e metricas para multiplos runs.
  - Gera tabela `metrics_table.csv` e `metrics_table.md`.
  - Inclui referencias do paper em `PAPER_TVAE`.

- `tools/list_pipeline_files.py`
  - Lista arquivos locais importados a partir de um entrypoint.

- `tools/collect_requirements.py`
  - Coleta imports externos e pode gerar um requirements basico.

## Logging de experimentos

O logger leve (`utils/experiment_logger.py`) grava sempre em CSV e pode opcionalmente
gerar logs do TensorBoard.

- CSV: `<run_dir>/logs/scalars.csv` (colunas: `step`, `tag`, `value`)
- TensorBoard (quando habilitado): `<run_dir>/logs/tensorboard/`

Tags usadas no Strategy A:
- `train/nll_total`, `train/nll_h`, `train/nll_o`, `train/nll_d`, `train/nll_r`
- `val/nll_total`, `val/nll_h`, `val/nll_o`, `val/nll_d`, `val/nll_r`
- `train/grad_norm`, `train/param_norm`
- `flow/layer_logdet_mean_<i>`, `flow/layer_logdet_std_<i>` (diagnosticos do flow)
- `metrics/*` (fast metrics opcionais em amostras pequenas)

Controles principais (em `config.py`):
- `SA_LOG_BATCH_EVERY` (0 desliga logs por batch)
- `SA_FLOW_DIAG_EVERY_EPOCHS`
- `SA_MONITOR_METRICS_EVERY_EPOCHS`
- `SA_MONITOR_MAX_SAMPLES`
- `SA_MONITOR_DO_COVERAGE`

Tags usadas no TVAE:
- `train/loss`, `train/recon`, `train/kl`, `train/pair_kl`, `train/pickup_kl`
- `val/loss`, `val/recon`, `val/kl`, `val/pair_kl`, `val/pickup_kl`
- `train/ce_<col>`, `train/acc_<col>` (por coluna em `OUTPUT_COLUMNS`)
- `val/ce_<col>`, `val/acc_<col>` (por coluna em `OUTPUT_COLUMNS`)
- `train/grad_norm`, `train/param_norm`
- `metrics/*` (fast metrics opcionais em amostras pequenas)

Controles principais (em `config.py`):
- `TVAE_MONITOR_METRICS_EVERY_EPOCHS`
- `TVAE_MONITOR_MAX_SAMPLES`

Exemplo de uso:

```
from utils.experiment_logger import ExperimentLogger

logger = ExperimentLogger(run_dir=output_dir, run_name="baseline", enable_tb=True)
logger.log_scalar("loss", value=1.23, step=1, split="train")
logger.log_scalars({"loss": 0.98, "acc": 0.45}, step=1, prefix="val")
logger.close()
```

Para habilitar/desabilitar TensorBoard:
- `LOG_ENABLE_TENSORBOARD = True` em `config.py`

Para visualizar:

```
tensorboard --logdir outputs/save_data/baseline/logs/tensorboard
```

## Plot de curvas de treino

O script `tools/plot_training_curves.py` lê `scalars.csv` e exporta PNGs.

```
python tools/plot_training_curves.py --run-dir outputs/save_data/baseline
```

Por default, salva em `<run_dir>/plots/`. Use `--out-dir` para customizar.

## Artefatos gerados por run

Para cada run (ex.: `baseline`), os principais arquivos salvos sao:
- Checkpoint: `tvae_order_1.pt`
- Mappings: `mappings_order_1.json`
- Perdas: `loss_order_1.csv`
- Metricas por split: `metrics/metrics_tvae_order_1_val.json` e
  `metrics/metrics_tvae_order_1_hold.json` (se hold existir)
- Sinteticos por split: `data/synthetic_tvae_order_1_val.csv` e
  `data/synthetic_tvae_order_1_hold.csv` (se hold existir)
- Alias legacy: `metrics_order_1.json` (hold se existir; senao val)
- Baseline train_vs_val: `metrics_train_vs_val.json`
- Estatisticas de split: `split_stats_train.json`, `split_stats_val.json`, `split_stats_hold.json`
- Contagens por split: `counts_<split>_<col>.csv`
- Top-k por coluna: `metrics/topk_<model_key>_<split>_<col>.csv`
- Graficos: `hist_<model_key>_<split>_<col>.png`, `topk_<model_key>_<split>_<col>.png`
- Logs: `logs/<run_tag>/train_order_1.log`

Os paths padrao desses artefatos sao controlados por `config.py`:
- `OUTPUT_BASE_DIR` (default: `/home-ext/caioloss/Dados/TVAE`)
- `SAVE_DATA_DIR`, `LOG_DIR`, `PLOT_DIR`

## Configuracao principal (config.py)

Pontos-chave:
- **Paths**: `REAL_DATA_PATH`, `OUTPUT_BASE_DIR`, `SAVE_DATA_SUBDIR`.
- **Filtros temporais**: `FILTER_YEAR`, `FILTER_MONTHS`, `FILTER_DOW_MIN/MAX`.
- **Split**:
  - `TRAIN_FRAC` e `VAL_FRAC` controlam train/val.
  - O `hold` e o restante (1 - TRAIN_FRAC - VAL_FRAC).
  - Observacao: manter `TRAIN_FRAC + VAL_FRAC <= 1.0` para ter hold.
- **Hiperparametros**: `LATENT_DIM`, `ENCODER_HIDDEN_DIMS`, `DECODER_HIDDEN_DIMS`,
  `BATCH_SIZE`, `EPOCHS`, `LEARNING_RATE`, `WEIGHT_DECAY`.
- **Pesos de loss**: `PICKUP_LOSS_WEIGHT`, `DROPOFF_LOSS_WEIGHT`,
  `PAIR_KL_WEIGHT`, `PICKUP_KL_WEIGHT`.
- **Sampling**: `SAMPLE_TEMPERATURE`, `SAMPLE_ROWS`.
- **Metricas**: `COVERAGE_K`, `COVERAGE_MAX_SAMPLES`, `COVERAGE_TIME_WEIGHT`,
  `COVERAGE_SPACE_WEIGHT`, `TIME_KEY_CARDINALITY`.
- **Experimentos**: `EXPERIMENTS` define os overrides por run.

## Observacoes sobre o fluxo de avaliacao

- `train_all_orders()` sempre calcula `train_vs_val` com dados reais, antes do treino.
- O treino usa `val_df` para early stopping.
- As metricas sao calculadas separadamente para `val` e `hold` (quando existir), e
  `compute_metrics` e `compute_paper_metrics` usam o mesmo split.
- Os artefatos de avaliacao ficam em `metrics/` (JSONs e tabelas top-k) e `data/`
  (sinteticos).
- `recalc_metrics_hold.py` padroniza a avaliacao usando hold para todos os runs.

## Strategy A pipeline

### Preparar embeddings de zona

```
python tools/build_zone_embeddings.py --device cpu
```

Arquivos esperados em `SA_TOPOLOGY_CACHE_DIR`:
- `Ecomb.pt` (preferido) ou `Efunc.pt`.

### Treinar Strategy A

```
python train_strategy_a.py --run-tag strategy_a
```

Saidas em `SAVE_DATA_DIR/strategy_a`:
- `strategy_a.pt`
- `mappings_strategy_a.json`
- `loss_strategy_a.csv`
- `metrics/metrics_strategy_a_val.json`
- `metrics/metrics_strategy_a_hold.json` (se hold existir)
- `data/synthetic_strategy_a_val.csv`
- `data/synthetic_strategy_a_hold.csv` (se hold existir)
- Aliases legacy: `metrics_strategy_a.json`, `metrics_strategy_a_hold.json`,
  `synthetic_strategy_a_hold.csv`

### Amostrar Strategy A (standalone)

```
python sample_strategy_a.py --run-dir outputs/save_data/strategy_a --split hold
```

Gera: `synthetic_strategy_a_hold.csv` no diretorio do run.

## Comparison protocol

Objetivo: comparar TVAE baseline vs Strategy A usando as mesmas metricas e splits.

1) **Treinar baseline (TVAE)**:

```
python train_tvae.py
```

2) **Recalcular métricas no hold (TVAE)**:

```
python recalc_metrics_hold.py --run-dir outputs/save_data/baseline
```

3) **Treinar Strategy A**:

```
python train_strategy_a.py --run-tag strategy_a
```

4) **Recalcular métricas Strategy A (hold)**:

```
python recalc_metrics_strategy_a_hold.py --run-dir outputs/save_data/strategy_a
```

5) **Comparar tabelas (baseline vs Strategy A)**:

```
python tools/compare_models.py \
  --baseline-run outputs/save_data/baseline \
  --strategy-a-run outputs/save_data/strategy_a \
  --out-dir outputs/compare_models
```

## One-shot pipeline

Para rodar tudo em uma tacada (treino, recalc, comparacao e plots):

```
python tools/run_full_pipeline.py --build-embeddings
```

Flags uteis:
- `--force-train` re-treina mesmo se o run ja existir
- `--force-sample` reamostra sinteticos no recalc
- `--baseline-run-dir` / `--strategy-run-dir` para reutilizar dirs existentes

## Reproducing paper metrics

As metricas do paper (W1 temporal, graph similarity, coverage) sao sempre geradas por
`utils.evaluation.compute_paper_metrics()`. Para reproducao consistente:

- Use os mesmos filtros temporais (`FILTER_YEAR`, `FILTER_MONTHS`, `FILTER_DOW_MIN/MAX`).
- Use o mesmo split base (`TRAIN_FRAC`, `VAL_FRAC`) e estrategia (`SA_SPLIT_STRATEGY`).
- Fixe `GLOBAL_SEED` quando amostrar/recalcular.

Comandos tipicos:

```
python train_tvae.py
python recalc_metrics_hold.py --run-dir outputs/save_data/baseline

python train_strategy_a.py --run-tag strategy_a
python recalc_metrics_strategy_a_hold.py --run-dir outputs/save_data/strategy_a
```

## Quick sanity run (<= 2 minutos em CPU)

Para um smoke test rapido, reduza epocas e amostras no `config.py`:
- `EPOCHS = 1`, `BATCH_SIZE = 256`, `MAX_EVAL_SAMPLES = 1000`
- `SA_EPOCHS = 1`, `SA_BATCH_SIZE = 256`, `SA_EVAL_SAMPLE_RATIO = 0.1`

Depois rode:

```
python train_tvae.py
python train_strategy_a.py --run-tag strategy_a
python tools/compare_models.py \
  --baseline-run outputs/save_data/baseline \
  --strategy-a-run outputs/save_data/strategy_a \
  --out-dir outputs/compare_models
```
