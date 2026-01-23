from __future__ import annotations

import argparse
import contextlib
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import config
from data_processing.loader import load_and_split
from data_processing.transformer import CategoricalTransformer
from models.tvae_ar import TVAEAutoregressive
from utils.helpers import set_seed
from utils.metrics import (
    chi2_counts,
    coverage_score,
    graph_similarity_score,
    joint_counts,
    joint_metrics,
    jsd_counts,
    marginal_counts,
    od_metrics,
    plot_marginal_hist,
    plot_topk,
    save_metrics_json,
    time_metrics,
    topk_table,
    wasserstein_time,
)
from utils.serialization import save_checkpoint, save_mappings

#Debug prints
class Tee:
    def __init__(self, *streams: Iterable):
        self.streams = streams

    def write(self, data: str) -> None:
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self) -> None:
        for s in self.streams:
            s.flush()

#Peso beta da loss
def _kl_beta(epoch: int) -> float:
    if config.KL_ANNEAL_EPOCHS <= 0:
        return config.KL_BETA_END
    progress = min(1.0, epoch / float(config.KL_ANNEAL_EPOCHS))
    return config.KL_BETA_START + (config.KL_BETA_END - config.KL_BETA_START) * progress

#Remove do dataset de teste as categorias que não estão no dataset de treino
#Isso aqui é legalizado?
def _filter_known(df: pd.DataFrame, transformer: CategoricalTransformer) -> pd.DataFrame:
    mask = np.ones(len(df), dtype=bool)
    for col in transformer.columns:
        allowed = set(transformer.categories[col])
        mask &= df[col].isin(allowed)
    return df.loc[mask].reset_index(drop=True)


def _prepare_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    transformer: CategoricalTransformer,
) -> Tuple[DataLoader, DataLoader, pd.DataFrame, pd.DataFrame]:
    #Faz um mapeamento para transformar em indices, meio que mapea os ids que temos atualmente em cada uma das counas
    # E adapta isso de forma que fique com ids continuos sem nenhum buraco, tipo, ao invés de ter 253, 255, 256
    # vai ter algo como 0, 1, 2, 3, 4, 5, ...
    #Precisa disso pro one hot encoding
    train_idx = transformer.transform(train_df, drop_unknown=True)
    val_idx = transformer.transform(val_df, drop_unknown=True)

    #onehot enconding treino e teste
    x_train = transformer.one_hot_encode(train_idx)
    x_val = transformer.one_hot_encode(val_idx)

    train_ds = TensorDataset(
        torch.from_numpy(x_train).float(),
        torch.from_numpy(train_idx.to_numpy(dtype=np.int64)),
    )
    val_ds = TensorDataset(
        torch.from_numpy(x_val).float(),
        torch.from_numpy(val_idx.to_numpy(dtype=np.int64)),
    )
    #Divide em batches
    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        drop_last=False,
        num_workers=config.NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=config.NUM_WORKERS,
    )

    return train_loader, val_loader, train_idx, val_idx

#Condição real p(dropoff | pickup) para regularizador KL por batch
def _build_dropoff_conditional_log_q(
    train_idx: pd.DataFrame,
    *,
    pickup_size: int,
    dropoff_size: int,
    eps: float,
) -> np.ndarray:
    counts = np.zeros((pickup_size, dropoff_size), dtype=np.float64)
    pickup = train_idx["pickup_id"].to_numpy(dtype=np.int64)
    dropoff = train_idx["dropoff_id"].to_numpy(dtype=np.int64)
    np.add.at(counts, (pickup, dropoff), 1)
    row_sum = counts.sum(axis=1, keepdims=True)
    probs = (counts + eps) / (row_sum + eps * dropoff_size)
    return np.log(probs)


def _build_pickup_log_q(
    train_idx: pd.DataFrame,
    *,
    pickup_size: int,
    eps: float,
) -> np.ndarray:
    counts = np.zeros((pickup_size,), dtype=np.float64)
    pickup = train_idx["pickup_id"].to_numpy(dtype=np.int64)
    np.add.at(counts, pickup, 1)
    total = counts.sum()
    probs = (counts + eps) / (total + eps * pickup_size)
    return np.log(probs)

#Aqui é só um pequeno truque no formato dos dados para que fique conforme o decoder do TVAE precisa.
def _build_y_indices(
    idx_batch: torch.Tensor, col_to_idx: Dict[str, int], order: List[str]
) -> Dict[str, torch.Tensor]:
    return {col: idx_batch[:, col_to_idx[col]] for col in order}

#aqui é onde realmente faz o treino do modelo.
def _epoch_pass(
    model: TVAEAutoregressive,
    loader: DataLoader,
    col_to_idx: Dict[str, int],
    order: List[str],
    device: torch.device,
    beta: float,
    *,
    pair_kl_log_q: torch.Tensor | None = None,
    pair_kl_weight: float = 0.0,
    pickup_kl_log_q: torch.Tensor | None = None,
    pickup_kl_weight: float = 0.0,
    train: bool,
    optimizer: torch.optim.Optimizer | None = None,
) -> Tuple[float, float, float, float, float]:
    total_loss = 0.0
    total_ce = 0.0
    total_kl = 0.0
    total_pair_kl = 0.0
    total_pickup_kl = 0.0
    total_batches = 0

    if train:
        model.train()
    else:
        model.eval()

    for x_batch, idx_batch in loader:
        x_batch = x_batch.to(device)
        idx_batch = idx_batch.to(device)
        y_indices = _build_y_indices(idx_batch, col_to_idx, order)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            logits, mu, logvar = model(x_batch, y_indices)
            ce_loss = 0.0
            for col in order:
                weight = 1.0
                if col == "pickup_id":
                    weight = config.PICKUP_LOSS_WEIGHT
                elif col == "dropoff_id":
                    weight = config.DROPOFF_LOSS_WEIGHT
                ce_loss = ce_loss + weight * F.cross_entropy(
                    logits[col],
                    y_indices[col],
                )
            kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            pair_kl = torch.tensor(0.0, device=device)
            if pair_kl_log_q is not None and pair_kl_weight > 0.0:
                log_p = F.log_softmax(logits["dropoff_id"], dim=1)
                p = log_p.exp()
                pickup_idx = y_indices["pickup_id"]
                log_q = pair_kl_log_q[pickup_idx]
                pair_kl = torch.sum(p * (log_p - log_q), dim=1).mean()
            pickup_kl = torch.tensor(0.0, device=device)
            if pickup_kl_log_q is not None and pickup_kl_weight > 0.0:
                log_p = F.log_softmax(logits["pickup_id"], dim=1)
                p = log_p.exp()
                pickup_kl = torch.sum(p * (log_p - pickup_kl_log_q), dim=1).mean()
            loss = (
                ce_loss
                + beta * kld
                + pair_kl_weight * pair_kl
                + pickup_kl_weight * pickup_kl
            )

        if train:
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item())
        total_ce += float(ce_loss.item())
        total_kl += float(kld.item())
        total_pair_kl += float(pair_kl.item())
        total_pickup_kl += float(pickup_kl.item())
        total_batches += 1

    if total_batches == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    return (
        total_loss / total_batches,
        total_ce / total_batches,
        total_kl / total_batches,
        total_pair_kl / total_batches,
        total_pickup_kl / total_batches,
    )

#Gera os dados sintéticos usando o decoder do TVAE. 
#Ele faz isso sequencialmente dessa forma:
# Primeiro ele pega algum ponto aleatório Z no espaço latente
#Ai ele pega o pickup_id que esse z representa 
# Então dado z, pickupid, ele gera o dropoff_id que esse z representa
#Na mesma lógica, dia_da_semana dado z, pickup_id, dropoff_id
# hora_do_dia dado z, pickup_id, dropoff_id, dia_da_semana
#Esse processo é repetido pra cada linha gerada 
def _sample_synthetic(
    model: TVAEAutoregressive,
    transformer: CategoricalTransformer,
    *,
    n_samples: int,
    temperature: float,
    device: torch.device,
    batch_size: int = 10000,
) -> pd.DataFrame:
    """Gera amostras sintéticas em batches para evitar estouro de memória GPU."""
    model.eval()
    all_samples: Dict[str, List[np.ndarray]] = {col: [] for col in transformer.columns}

    with torch.no_grad():
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            current_batch = end - start

            z = torch.randn(current_batch, model.latent_dim, device=device)
            samples = model.sample(z, temperature=temperature)

            for col in transformer.columns:
                all_samples[col].append(samples[col].cpu().numpy())

            # Liberar memória GPU
            del z, samples
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # Concatenar todos os batches
    data = {col: np.concatenate(all_samples[col]) for col in transformer.columns}
    df_idx = pd.DataFrame(data)
    df_decoded = transformer.decode_indices(df_idx)
    df_decoded = df_decoded[config.OUTPUT_COLUMNS]
    return df_decoded


def _eval_sample_size(df: pd.DataFrame) -> int:
    n_eval = int(np.ceil(len(df) * float(config.EVAL_SAMPLE_RATIO)))
    n_eval = max(1, n_eval)
    if config.MAX_EVAL_SAMPLES is not None:
        n_eval = min(n_eval, int(config.MAX_EVAL_SAMPLES))
    n_eval = min(n_eval, len(df))
    return n_eval

#Calcula as métricas de qualidade dos dados sintéticos. Nao são as métricas do paper!
#Marginal: JSD, chi² por coluna	Cada coluna isoladamente
#OD: od_jsd, od_coverage	Pares pickup-dropoff
#Temporal: time_jsd, time_coverage	Combinações dia-hora
#Joint: joint_jsd, mode_dropping	OD × dia × hora completo
def _compute_metrics(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    *,
    order_key: str,
    output_dir: Path,
    plot_dir: Path,
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    metrics["n_real"] = float(len(real_df))
    metrics["n_synth"] = float(len(synth_df))

    for col in config.OUTPUT_COLUMNS:
        real_counts = marginal_counts(real_df, col)
        synth_counts = marginal_counts(synth_df, col)
        metrics[f"{col}_jsd"] = jsd_counts(real_counts, synth_counts)
        metrics[f"{col}_chi2"] = chi2_counts(real_counts, synth_counts)

        plot_path = plot_dir / f"hist_{order_key}_{col}.png"
        plot_marginal_hist(real_counts, synth_counts, title=f"{order_key} {col}", path=plot_path)

        if col in ("pickup_id", "dropoff_id"):
            topk_df = topk_table(real_counts, synth_counts, k=20)
            topk_path = output_dir / f"topk_{order_key}_{col}.csv"
            topk_df.to_csv(topk_path, index=False)
            plot_topk(
                topk_df,
                title=f"{order_key} topk {col}",
                path=plot_dir / f"topk_{order_key}_{col}.png",
            )

    # Métricas diagnósticas separadas
    od = od_metrics(real_df, synth_df)
    metrics.update(od)

    time = time_metrics(real_df, synth_df)
    metrics.update(time)

    # Métricas da joint completa (OD × dia × hora)
    joint = joint_metrics(real_df, synth_df)
    metrics.update(joint)
    return metrics

#Aqui sim são as métricas do paper!
def _compute_paper_metrics(
    train_df: pd.DataFrame, test_df: pd.DataFrame, synth_df: pd.DataFrame
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}

    metrics["w1_tr_te"] = wasserstein_time(
        train_df, test_df, support_size=config.TIME_KEY_CARDINALITY
    )
    metrics["w1_tr_syn"] = wasserstein_time(
        train_df, synth_df, support_size=config.TIME_KEY_CARDINALITY
    )
    metrics["w1_te_syn"] = wasserstein_time(
        test_df, synth_df, support_size=config.TIME_KEY_CARDINALITY
    )

    metrics["g_tr_te"] = 100.0 * graph_similarity_score(train_df, test_df)
    metrics["g_tr_syn"] = 100.0 * graph_similarity_score(train_df, synth_df)
    metrics["g_te_syn"] = 100.0 * graph_similarity_score(test_df, synth_df)

    metrics["cov_tr_te"] = coverage_score(
        train_df,
        test_df,
        k=config.COVERAGE_K,
        max_samples=config.COVERAGE_MAX_SAMPLES,
        seed=config.GLOBAL_SEED,
        chunk_size=config.COVERAGE_CHUNK_SIZE,
        w_time=config.COVERAGE_TIME_WEIGHT,
        w_space=config.COVERAGE_SPACE_WEIGHT,
    )
    metrics["cov_tr_syn"] = coverage_score(
        train_df,
        synth_df,
        k=config.COVERAGE_K,
        max_samples=config.COVERAGE_MAX_SAMPLES,
        seed=config.GLOBAL_SEED,
        chunk_size=config.COVERAGE_CHUNK_SIZE,
        w_time=config.COVERAGE_TIME_WEIGHT,
        w_space=config.COVERAGE_SPACE_WEIGHT,
    )
    metrics["cov_te_syn"] = coverage_score(
        test_df,
        synth_df,
        k=config.COVERAGE_K,
        max_samples=config.COVERAGE_MAX_SAMPLES,
        seed=config.GLOBAL_SEED,
        chunk_size=config.COVERAGE_CHUNK_SIZE,
        w_time=config.COVERAGE_TIME_WEIGHT,
        w_space=config.COVERAGE_SPACE_WEIGHT,
    )

    metrics["coverage_k"] = float(config.COVERAGE_K)
    metrics["coverage_max_samples"] = float(config.COVERAGE_MAX_SAMPLES or 0)
    metrics["coverage_time_weight"] = float(config.COVERAGE_TIME_WEIGHT)
    metrics["coverage_space_weight"] = float(config.COVERAGE_SPACE_WEIGHT)

    return metrics

#salva valores unicos de cada coluna em cada split
def _save_value_counts(df: pd.DataFrame, col: str, path: Path) -> None:
    counts = df[col].value_counts().sort_index()
    out = counts.rename("count").reset_index().rename(columns={"index": col})
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)

#salva estatisticas de cada split
def _save_split_stats(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    hold_df: pd.DataFrame,
    *,
    output_dir: Path,
) -> None:
    def _summary(split_name: str, df: pd.DataFrame) -> Dict[str, float]:
        stats: Dict[str, float] = {"rows": float(len(df))}
        for col in config.OUTPUT_COLUMNS:
            _save_value_counts(df, col, output_dir / f"counts_{split_name}_{col}.csv")
            stats[f"unique_{col}"] = float(df[col].nunique())
            if split_name == "train":
                stats[f"coverage_{col}"] = 1.0
            else:
                train_unique = set(train_df[col].unique())
                split_unique = set(df[col].unique())
                denom = max(1, len(train_unique))
                stats[f"coverage_{col}"] = float(len(train_unique & split_unique) / denom)

        train_joint = set(joint_counts(train_df).index)
        split_joint = set(joint_counts(df).index)
        stats["joint_unique"] = float(len(split_joint))
        if split_name == "train":
            stats["joint_coverage"] = 1.0
        else:
            denom = max(1, len(train_joint))
            stats["joint_coverage"] = float(len(train_joint & split_joint) / denom)
        return stats

    for name, df in (("train", train_df), ("val", val_df), ("hold", hold_df)):
        stats = _summary(name, df)
        save_metrics_json(stats, output_dir / f"split_stats_{name}.json")

#Serve só pra rodar experimentos com parametros diferentes, daria pra usar o optuna pra isso
def _apply_overrides(overrides: Dict[str, object]) -> Dict[str, object]:
    backup: Dict[str, object] = {}
    for key, val in overrides.items():
        if not hasattr(config, key):
            raise KeyError(f"Unknown config key in experiment override: {key}")
        backup[key] = getattr(config, key)
        setattr(config, key, val)
    return backup


def _restore_overrides(backup: Dict[str, object]) -> None:
    for key, val in backup.items():
        setattr(config, key, val)


def _run_dirs(run_tag: str) -> Tuple[Path, Path, Path]:
    output_dir = Path(config.SAVE_DATA_DIR) / run_tag
    log_dir = Path(config.LOG_DIR) / run_tag
    plot_dir = Path(config.PLOT_DIR) / run_tag
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    return output_dir, log_dir, plot_dir

#Aqui ele junta o fluxo todo:

#_prepare_dataloaders: prepara os dados para o treino
#TVAEAutoregressive: instancia o modelo TVAE autoregressivo
# Ai entra no loop de treino:
# for epoch in 1...EPOCHS:
   #_epoch_pass(train=true) - treina o modelo
   #_epoch_pass(train=false) - avalia o modelo
   #se o modelo melhorar, salva o checkpoint
   #se o modelo nao melhorar, para o treino
   #depois de treinar, gera os dados sintéticos
#Depois do treino ele vai carregar o best_state
# Salva o modelo localmente, salva os mappings, salva as métricas
# Depois ele gera os dados sintéticos com _sample_synthetic()
# Ai calcula as métricas com _computer_metrics()
# Depois ele finaliza calculando as métricas do paper com _compute_paper_metrics()
def train_single_order(
    order_key: str,
    order: List[str],
    *,
    device: torch.device,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    hold_df: pd.DataFrame,
    transformer: CategoricalTransformer,
    output_dir: Path,
    log_dir: Path,
    plot_dir: Path,
) -> Dict[str, float]:
    set_seed(config.GLOBAL_SEED)

    log_path = log_dir / f"train_{order_key}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("w", encoding="utf-8") as fh, contextlib.redirect_stdout(
        Tee(sys.stdout, fh)
    ):
        print(f"[Train] order_key={order_key} order={order}")
        print(
            f"[Train] train_rows={len(train_df)} val_rows={len(val_df)} hold_rows={len(hold_df)}"
        )

        train_loader, val_loader, train_idx, val_idx = _prepare_dataloaders(
            train_df, val_df, transformer
        )
        col_to_idx = {col: i for i, col in enumerate(transformer.columns)}
        pair_kl_log_q = None
        pickup_kl_log_q = None
        if config.PAIR_KL_WEIGHT > 0.0:
            pickup_size = transformer.cardinalities["pickup_id"]
            dropoff_size = transformer.cardinalities["dropoff_id"]
            pair_kl_log_q = _build_dropoff_conditional_log_q(
                train_idx,
                pickup_size=pickup_size,
                dropoff_size=dropoff_size,
                eps=config.PAIR_KL_EPS,
            )
            pair_kl_log_q = torch.from_numpy(pair_kl_log_q).float().to(device)
        if config.PICKUP_KL_WEIGHT > 0.0:
            pickup_size = transformer.cardinalities["pickup_id"]
            pickup_kl_log_q = _build_pickup_log_q(
                train_idx,
                pickup_size=pickup_size,
                eps=config.PICKUP_KL_EPS,
            )
            pickup_kl_log_q = torch.from_numpy(pickup_kl_log_q).float().to(device)

        model = TVAEAutoregressive(
            column_sizes=transformer.cardinalities,
            order=order,
            encoder_hidden_dims=config.ENCODER_HIDDEN_DIMS,
            decoder_hidden_dims=config.DECODER_HIDDEN_DIMS,
            latent_dim=config.LATENT_DIM,
        ).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
        )

        best_val = float("inf")
        best_state = None
        epochs_no_improve = 0

        loss_rows = []
        train_start = time.monotonic()
        for epoch in range(1, config.EPOCHS + 1):
            beta = _kl_beta(epoch)
            train_loss, train_ce, train_kl, train_pair_kl, train_pickup_kl = _epoch_pass(
                model,
                train_loader,
                col_to_idx,
                order,
                device,
                beta,
                pair_kl_log_q=pair_kl_log_q,
                pair_kl_weight=config.PAIR_KL_WEIGHT,
                pickup_kl_log_q=pickup_kl_log_q,
                pickup_kl_weight=config.PICKUP_KL_WEIGHT,
                train=True,
                optimizer=optimizer,
            )
            val_loss, val_ce, val_kl, val_pair_kl, val_pickup_kl = _epoch_pass(
                model,
                val_loader,
                col_to_idx,
                order,
                device,
                beta,
                pair_kl_log_q=pair_kl_log_q,
                pair_kl_weight=config.PAIR_KL_WEIGHT,
                pickup_kl_log_q=pickup_kl_log_q,
                pickup_kl_weight=config.PICKUP_KL_WEIGHT,
                train=False,
            )

            loss_rows.append(
                {
                    "epoch": epoch,
                    "beta": beta,
                    "train_loss": train_loss,
                    "train_ce": train_ce,
                    "train_kl": train_kl,
                    "train_pair_kl": train_pair_kl,
                    "train_pickup_kl": train_pickup_kl,
                    "val_loss": val_loss,
                    "val_ce": val_ce,
                    "val_kl": val_kl,
                    "val_pair_kl": val_pair_kl,
                    "val_pickup_kl": val_pickup_kl,
                }
            )
            print(
                f"[Epoch {epoch:03d}] beta={beta:.4f} train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f}"
            )

            if best_val - val_loss > config.MIN_DELTA:
                best_val = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= config.PATIENCE:
                print("[Train] early stopping")
                break
        train_time_min = (time.monotonic() - train_start) / 60.0

        if best_state is not None:
            model.load_state_dict(best_state)

        loss_df = pd.DataFrame(loss_rows)
        loss_df.to_csv(output_dir / f"loss_{order_key}.csv", index=False)

        save_mappings(output_dir / f"mappings_{order_key}.json", transformer)

        meta = {
            "column_sizes": transformer.cardinalities,
            "order": order,
            "encoder_hidden_dims": list(config.ENCODER_HIDDEN_DIMS),
            "decoder_hidden_dims": list(config.DECODER_HIDDEN_DIMS),
            "latent_dim": config.LATENT_DIM,
            "columns": transformer.columns,
        }
        checkpoint_path = output_dir / f"tvae_{order_key}.pt"
        save_checkpoint(
            checkpoint_path,
            model_state=model.state_dict(),
            meta=meta,
            optimizer_state=optimizer.state_dict(),
            epoch=epoch,
            metrics={"best_val": best_val},
        )

        eval_df = hold_df if len(hold_df) > 0 else val_df
        n_eval = _eval_sample_size(eval_df)
        sample_start = time.monotonic()
        synth_df = _sample_synthetic(
            model,
            transformer,
            n_samples=n_eval,
            temperature=config.SAMPLE_TEMPERATURE,
            device=device,
        )
        sample_time_min = (time.monotonic() - sample_start) / 60.0

        metrics = _compute_metrics(
            val_df,
            synth_df,
            order_key=order_key,
            output_dir=output_dir,
            plot_dir=plot_dir,
        )
        paper_metrics = _compute_paper_metrics(train_df, eval_df, synth_df)
        metrics.update(paper_metrics)
        metrics["best_val"] = best_val
        metrics["train_time_min"] = train_time_min
        metrics["sample_time_min"] = sample_time_min
        metrics["total_time_min"] = train_time_min + sample_time_min
        save_metrics_json(metrics, output_dir / f"metrics_{order_key}.json")

        print(f"[Train] saved checkpoint={checkpoint_path}")
        print(f"[Train] metrics keys={sorted(metrics.keys())}")

    return metrics

#Aqui é só o orquestrador final, ele chama o train_single_order 
# A implementação tá desse jeito pq anteriormente eu tinha testado treinar ordens diferentes e comparar
# Mas dá pra simplificar isso aqui e unificar o train_single_order e train_all_orders
def train_all_orders() -> None:
    raw_train_df, raw_val_df, raw_hold_df = load_and_split()
    experiments = config.EXPERIMENTS or {"baseline": {}}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for exp_name, overrides in experiments.items():
        backup = _apply_overrides(overrides)
        output_dir, log_dir, plot_dir = _run_dirs(exp_name)

        _save_split_stats(raw_train_df, raw_val_df, raw_hold_df, output_dir=output_dir)
        baseline_metrics = _compute_metrics(
            raw_train_df,
            raw_val_df,
            order_key="train_vs_val",
            output_dir=output_dir,
            plot_dir=plot_dir,
        )
        save_metrics_json(baseline_metrics, output_dir / "metrics_train_vs_val.json")

        train_df = raw_train_df
        val_df = raw_val_df
        transformer = CategoricalTransformer(config.OUTPUT_COLUMNS)
        transformer.fit(train_df)

        val_df = _filter_known(val_df, transformer)
        train_df = _filter_known(train_df, transformer)
        hold_df = _filter_known(raw_hold_df, transformer)

        order_key = config.FIXED_ORDER_KEY
        order = config.ORDERS[order_key]
        metrics = train_single_order(
            order_key,
            order,
            device=device,
            train_df=train_df,
            val_df=val_df,
            hold_df=hold_df,
            transformer=transformer,
            output_dir=output_dir,
            log_dir=log_dir,
            plot_dir=plot_dir,
        )
        print(f"[Train] completed order={order_key} metrics_keys={len(metrics)}")

        _restore_overrides(backup)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    train_all_orders()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
