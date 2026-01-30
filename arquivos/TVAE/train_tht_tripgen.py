from __future__ import annotations

import argparse
import contextlib
import math
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

import config
from data_processing.tht_tripgen_loader import load_and_split_tht_tripgen
from data_processing.tht_tripgen_transformer import THTTripGenTransformer
from models.tht_tripgen import THTTripGenModel
from utils.evaluation import compute_metrics, compute_paper_metrics
from utils.experiment_logger import ExperimentLogger
from utils.metrics import compute_fast_metrics, save_metrics_json
from utils.serialization import save_checkpoint, save_tht_tripgen_mappings


class Tee:
    def __init__(self, *streams: Iterable):
        self.streams = streams

    def write(self, data: str) -> None:
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def _run_dirs(run_tag: str) -> Tuple[Path, Path, Path]:
    output_dir = Path(config.SAVE_DATA_DIR) / run_tag
    log_dir = Path(config.LOG_DIR) / run_tag
    plot_dir = Path(config.PLOT_DIR) / run_tag
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    return output_dir, log_dir, plot_dir


def _run_output_subdirs(output_dir: Path) -> Tuple[Path, Path]:
    """Create per-run subdirectories for metrics JSONs and synthetic samples."""
    metrics_dir = output_dir / "metrics"
    data_dir = output_dir / "data"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    return metrics_dir, data_dir


def _filter_known(df: pd.DataFrame, transformer: THTTripGenTransformer) -> pd.DataFrame:
    idx = transformer.transform(df, drop_unknown=False)
    mask = idx.notna().all(axis=1) & idx["r"].notna()
    return df.loc[mask].reset_index(drop=True)


def _tensor_dataset(
    df_idx: pd.DataFrame, conditional_cols: List[str]
) -> TensorDataset:
    h = torch.from_numpy(df_idx["h_idx"].to_numpy(dtype=np.int64, copy=True))
    o = torch.from_numpy(df_idx["o_idx"].to_numpy(dtype=np.int64, copy=True))
    d = torch.from_numpy(df_idx["d_idx"].to_numpy(dtype=np.int64, copy=True))
    r = torch.from_numpy(df_idx["r"].to_numpy(dtype=np.float32, copy=True))
    u_tensors = [
        torch.from_numpy(df_idx[col].to_numpy(dtype=np.int64, copy=True))
        for col in conditional_cols
    ]
    return TensorDataset(h, o, d, r, *u_tensors)


def _prepare_loader(
    df_idx: pd.DataFrame,
    conditional_cols: List[str],
    *,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = _tensor_dataset(df_idx, conditional_cols)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=0,
    )


def _epoch_pass(
    model: THTTripGenModel,
    loader: DataLoader,
    conditional_cols: List[str],
    device: torch.device,
    *,
    train: bool,
    optimizer: torch.optim.Optimizer | None = None,
    logger: ExperimentLogger | None = None,
    log_every: int = 0,
    epoch: int | None = None,
    split_name: str = "train",
) -> Dict[str, float]:
    totals = {"nll_h": 0.0, "nll_o": 0.0, "nll_d": 0.0, "nll_r": 0.0, "nll_total": 0.0}
    batches = 0

    if train:
        model.train()
    else:
        model.eval()

    for batch_idx, batch in enumerate(loader, start=1):
        h_idx, o_idx, d_idx, r, *u_vals = batch
        h_idx = h_idx.to(device)
        o_idx = o_idx.to(device)
        d_idx = d_idx.to(device)
        r = r.to(device)
        u = {col: u_vals[i].to(device) for i, col in enumerate(conditional_cols)}

        if train and optimizer is not None:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            nll = model.nll(u=u, h_idx=h_idx, o_idx=o_idx, d_idx=d_idx, r=r)
            loss = nll["nll_total"]

        if train and optimizer is not None:
            loss.backward()
            optimizer.step()

        for key in totals:
            totals[key] += float(nll[key].item())
        batches += 1

        if (
            logger is not None
            and log_every > 0
            and epoch is not None
            and batch_idx % log_every == 0
        ):
            batch_step = epoch * 1_000_000 + batch_idx
            logger.log_scalars(
                {
                    "nll_total": float(nll["nll_total"].item()),
                    "nll_h": float(nll["nll_h"].item()),
                    "nll_o": float(nll["nll_o"].item()),
                    "nll_d": float(nll["nll_d"].item()),
                    "nll_r": float(nll["nll_r"].item()),
                },
                step=batch_step,
                prefix=f"{split_name}/batch",
            )

    if batches == 0:
        return totals
    return {key: val / batches for key, val in totals.items()}


def _grad_norm(model: THTTripGenModel) -> float:
    total = 0.0
    for param in model.parameters():
        if param.grad is None:
            continue
        total += float(param.grad.detach().pow(2).sum().item())
    return float(math.sqrt(total))


def _param_norm(model: THTTripGenModel) -> float:
    total = 0.0
    for param in model.parameters():
        total += float(param.detach().pow(2).sum().item())
    return float(math.sqrt(total))


def _eval_sample_size(n_rows: int) -> int:
    ratio = float(getattr(config, "THT_EVAL_SAMPLE_RATIO", 1.0))
    n_eval = int(np.ceil(n_rows * ratio))
    n_eval = max(1, n_eval) if n_rows > 0 else 0
    return min(n_eval, n_rows)


def _sample_conditioned(
    model: THTTripGenModel,
    eval_idx: pd.DataFrame,
    conditional_cols: List[str],
    *,
    temperature: float,
    device: torch.device,
    batch_size: int = 10000,
) -> pd.DataFrame:
    model.eval()
    outputs: Dict[str, List[np.ndarray]] = {}

    with torch.no_grad():
        for start in range(0, len(eval_idx), batch_size):
            end = min(start + batch_size, len(eval_idx))
            batch_df = eval_idx.iloc[start:end]
            u = {
                col: torch.from_numpy(
                    batch_df[col].to_numpy(dtype=np.int64, copy=True)
                ).to(device)
                for col in conditional_cols
            }
            samples = model.sample(
                n=end - start,
                u=u,
                temperature=temperature,
            )

            for key, tensor in samples.items():
                outputs.setdefault(key, []).append(tensor.detach().cpu().numpy())

    data = {key: np.concatenate(chunks) for key, chunks in outputs.items()}
    return pd.DataFrame(data)


def _load_zone_embeddings() -> Tuple[torch.Tensor, Path, str]:
    cache_dir = Path(config.THT_TOPOLOGY_CACHE_DIR)
    comb_path = cache_dir / "Ecomb.pt"
    func_path = cache_dir / "Efunc.pt"
    if comb_path.exists():
        emb_path = comb_path
        source = "Ecomb"
    elif func_path.exists():
        emb_path = func_path
        source = "Efunc"
    else:
        raise FileNotFoundError(
            f"No frozen embeddings found in {cache_dir} (expected Ecomb.pt or Efunc.pt)"
        )

    embeddings = torch.load(emb_path, map_location="cpu").float()
    return embeddings, emb_path, source


def train_tht_tripgen(*, run_tag: str, device: torch.device) -> None:
    output_dir, log_dir, plot_dir = _run_dirs(run_tag)
    metrics_dir, data_dir = _run_output_subdirs(output_dir)
    log_path = log_dir / "train_tht_tripgen.log"

    with log_path.open("w", encoding="utf-8") as fh, contextlib.redirect_stdout(
        Tee(sys.stdout, fh)
    ), ExperimentLogger(output_dir, run_name=run_tag, enable_tb=None) as logger:
        print(f"[THT-TripGen] run_tag={run_tag}")

        raw_train_df, raw_val_df, raw_hold_df = load_and_split_tht_tripgen()
        print(
            f"[THT-TripGen] train_rows={len(raw_train_df)} val_rows={len(raw_val_df)} hold_rows={len(raw_hold_df)}"
        )

        transformer = THTTripGenTransformer().fit(raw_train_df)
        train_df = _filter_known(raw_train_df, transformer)
        val_df = _filter_known(raw_val_df, transformer)
        hold_df = _filter_known(raw_hold_df, transformer)

        train_idx = transformer.transform(train_df, drop_unknown=True)
        val_idx = transformer.transform(val_df, drop_unknown=True)
        hold_idx = transformer.transform(hold_df, drop_unknown=True)

        conditional_cols = list(transformer.conditional_columns)
        cond_card = transformer.conditional_idx_cardinalities

        zone_embeddings, emb_path, emb_source = _load_zone_embeddings()
        if zone_embeddings.shape[0] != transformer.num_zones:
            raise ValueError("Frozen embeddings num_zones mismatch with transformer")

        destination_head_type = getattr(
            config, "THT_DESTINATION_HEAD_TYPE", config.THT_DEST_HEAD_TYPE
        )

        model = THTTripGenModel(
            num_zones=transformer.num_zones,
            num_time_bins=transformer.num_time_bins,
            conditional_cardinalities=cond_card,
            frozen_zone_embeddings=zone_embeddings,
            cond_emb_dim=int(getattr(config, "THT_COND_EMB_DIM", config.THT_COND_EMB_DIM)),
            time_emb_dim=int(getattr(config, "THT_TIME_EMB_DIM", config.THT_TIME_EMB_DIM)),
            origin_emb_dim=int(getattr(config, "THT_ORIGIN_EMB_DIM", config.THT_ORIGIN_EMB_DIM)),
            context_mlp_hidden=int(getattr(config, "THT_MODEL_HIDDEN", config.THT_MODEL_HIDDEN)),
            context_mlp_layers=int(getattr(config, "THT_MODEL_LAYERS", config.THT_MODEL_LAYERS)),
            dropout=float(getattr(config, "THT_MODEL_DROPOUT", config.THT_MODEL_DROPOUT)),
            destination_head_type=str(destination_head_type),
            min_r_eps=float(getattr(config, "THT_MIN_R_EPS", config.THT_MIN_R_EPS)),
            residual_num_layers=int(getattr(config, "THT_RESIDUAL_NUM_LAYERS", config.THT_RESIDUAL_NUM_LAYERS)),
            residual_num_bins=int(getattr(config, "THT_RESIDUAL_NUM_BINS", config.THT_RESIDUAL_NUM_BINS)),
            residual_context_hidden=int(
                getattr(config, "THT_RESIDUAL_CONTEXT_HIDDEN", config.THT_RESIDUAL_CONTEXT_HIDDEN)
            ),
            residual_min_bin_width=float(
                getattr(config, "THT_RESIDUAL_MIN_BIN_WIDTH", config.THT_RESIDUAL_MIN_BIN_WIDTH)
            ),
            residual_min_bin_height=float(
                getattr(config, "THT_RESIDUAL_MIN_BIN_HEIGHT", config.THT_RESIDUAL_MIN_BIN_HEIGHT)
            ),
            residual_min_deriv=float(
                getattr(config, "THT_RESIDUAL_MIN_DERIV", config.THT_RESIDUAL_MIN_DERIV)
            ),
            residual_eps=float(getattr(config, "THT_RESIDUAL_EPS", config.THT_RESIDUAL_EPS)),
        ).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(getattr(config, "THT_LR", 1e-3)),
            weight_decay=float(getattr(config, "THT_WEIGHT_DECAY", 0.0)),
        )

        batch_size = int(getattr(config, "THT_BATCH_SIZE", 1024))
        train_loader = _prepare_loader(
            train_idx, conditional_cols, batch_size=batch_size, shuffle=True
        )
        val_loader = _prepare_loader(
            val_idx, conditional_cols, batch_size=batch_size, shuffle=False
        )

        best_val = float("inf")
        best_state = None
        epochs_no_improve = 0
        loss_rows: List[Dict[str, float]] = []

        train_start = time.monotonic()
        epochs = int(getattr(config, "THT_EPOCHS", 30))
        log_every = int(getattr(config, "THT_LOG_BATCH_EVERY", 0))
        flow_every = int(getattr(config, "THT_FLOW_DIAG_EVERY_EPOCHS", 0))
        monitor_every = int(getattr(config, "THT_MONITOR_METRICS_EVERY_EPOCHS", 0))
        monitor_max = int(getattr(config, "THT_MONITOR_MAX_SAMPLES", 0))

        diag_batch = None
        for batch in val_loader:
            diag_batch = batch
            break

        for epoch in range(1, epochs + 1):
            train_nll = _epoch_pass(
                model,
                train_loader,
                conditional_cols,
                device,
                train=True,
                optimizer=optimizer,
                logger=logger if log_every > 0 else None,
                log_every=log_every,
                epoch=epoch,
                split_name="train",
            )
            val_nll = _epoch_pass(
                model,
                val_loader,
                conditional_cols,
                device,
                train=False,
            )

            loss_rows.append(
                {
                    "epoch": epoch,
                    "train_nll_total": train_nll["nll_total"],
                    "train_nll_h": train_nll["nll_h"],
                    "train_nll_o": train_nll["nll_o"],
                    "train_nll_d": train_nll["nll_d"],
                    "train_nll_r": train_nll["nll_r"],
                    "val_nll_total": val_nll["nll_total"],
                    "val_nll_h": val_nll["nll_h"],
                    "val_nll_o": val_nll["nll_o"],
                    "val_nll_d": val_nll["nll_d"],
                    "val_nll_r": val_nll["nll_r"],
                }
            )

            logger.log_scalars(
                {
                    "nll_total": train_nll["nll_total"],
                    "nll_h": train_nll["nll_h"],
                    "nll_o": train_nll["nll_o"],
                    "nll_d": train_nll["nll_d"],
                    "nll_r": train_nll["nll_r"],
                },
                step=epoch,
                prefix="train",
            )
            logger.log_scalars(
                {
                    "nll_total": val_nll["nll_total"],
                    "nll_h": val_nll["nll_h"],
                    "nll_o": val_nll["nll_o"],
                    "nll_d": val_nll["nll_d"],
                    "nll_r": val_nll["nll_r"],
                },
                step=epoch,
                prefix="val",
            )
            logger.log_scalar("grad_norm", _grad_norm(model), step=epoch, split="train")
            logger.log_scalar("param_norm", _param_norm(model), step=epoch, split="train")

            if flow_every > 0 and epoch % flow_every == 0 and diag_batch is not None:
                h_idx, o_idx, d_idx, r, *u_vals = diag_batch
                u = {col: u_vals[i].to(device) for i, col in enumerate(conditional_cols)}
                h_idx = h_idx.to(device)
                o_idx = o_idx.to(device)
                d_idx = d_idx.to(device)
                r = r.to(device)
                was_training = model.training
                model.eval()
                with torch.no_grad():
                    _, diag = model.residual_log_prob(
                        u=u,
                        h_idx=h_idx,
                        o_idx=o_idx,
                        d_idx=d_idx,
                        r=r,
                        return_layer_logdet=True,
                    )
                if was_training:
                    model.train()
                for key, value in diag.items():
                    logger.log_scalar(f"flow/{key}", float(value), step=epoch, split=None)

            if monitor_every > 0 and epoch % monitor_every == 0 and len(val_idx) > 0:
                n_monitor = min(len(val_idx), monitor_max) if monitor_max > 0 else len(val_idx)
                if n_monitor > 0:
                    sample_idx = val_idx.sample(
                        n=n_monitor, random_state=config.GLOBAL_SEED
                    ).index
                    val_idx_sample = val_idx.loc[sample_idx].reset_index(drop=True)
                    val_df_sample = val_df.loc[sample_idx].reset_index(drop=True)

                    with torch.no_grad():
                        synth_idx = _sample_conditioned(
                            model,
                            val_idx_sample,
                            conditional_cols,
                            temperature=float(
                                getattr(config, "THT_TEMPERATURE", config.THT_SAMPLE_TEMPERATURE)
                            ),
                            device=device,
                        )

                    synth_decoded = transformer.decode(synth_idx)
                    synth_metrics_df = synth_decoded[config.OUTPUT_COLUMNS].copy()
                    eval_metrics_df = val_df_sample[config.OUTPUT_COLUMNS].copy()

                    fast_metrics = compute_fast_metrics(
                        eval_metrics_df,
                        synth_metrics_df,
                        do_coverage=bool(
                            getattr(config, "THT_MONITOR_DO_COVERAGE", False)
                        ),
                        coverage_max_samples=int(
                            getattr(config, "THT_MONITOR_COVERAGE_MAX_SAMPLES", 0) or 0
                        ),
                        seed=config.GLOBAL_SEED,
                    )
                    logger.log_scalars(fast_metrics, step=epoch, prefix="metrics")

            print(
                f"[Epoch {epoch:03d}] train_nll={train_nll['nll_total']:.4f} "
                f"val_nll={val_nll['nll_total']:.4f}"
            )

            if best_val - val_nll["nll_total"] > config.MIN_DELTA:
                best_val = val_nll["nll_total"]
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= config.PATIENCE:
                print("[THT-TripGen] early stopping")
                break

        train_time_min = (time.monotonic() - train_start) / 60.0

        if best_state is not None:
            model.load_state_dict(best_state)

        loss_df = pd.DataFrame(loss_rows)
        loss_df.to_csv(output_dir / "loss_tht_tripgen.csv", index=False)

        save_tht_tripgen_mappings(output_dir / "mappings_tht_tripgen.json", transformer)

        meta = {
            "num_zones": transformer.num_zones,
            "num_time_bins": transformer.num_time_bins,
            "conditional_cardinalities": cond_card,
            "zone_embeddings_source": emb_source,
            "zone_embeddings_dim": int(zone_embeddings.shape[1]),
            "zone_embeddings_path": str(emb_path),
            "cond_emb_dim": int(getattr(config, "THT_COND_EMB_DIM", config.THT_COND_EMB_DIM)),
            "time_emb_dim": int(getattr(config, "THT_TIME_EMB_DIM", config.THT_TIME_EMB_DIM)),
            "origin_emb_dim": int(getattr(config, "THT_ORIGIN_EMB_DIM", config.THT_ORIGIN_EMB_DIM)),
            "context_mlp_hidden": int(getattr(config, "THT_MODEL_HIDDEN", config.THT_MODEL_HIDDEN)),
            "context_mlp_layers": int(getattr(config, "THT_MODEL_LAYERS", config.THT_MODEL_LAYERS)),
            "dropout": float(getattr(config, "THT_MODEL_DROPOUT", config.THT_MODEL_DROPOUT)),
            "destination_head_type": str(destination_head_type),
            "min_r_eps": float(getattr(config, "THT_MIN_R_EPS", config.THT_MIN_R_EPS)),
            "residual_num_layers": int(getattr(config, "THT_RESIDUAL_NUM_LAYERS", config.THT_RESIDUAL_NUM_LAYERS)),
            "residual_num_bins": int(getattr(config, "THT_RESIDUAL_NUM_BINS", config.THT_RESIDUAL_NUM_BINS)),
            "residual_context_hidden": int(
                getattr(config, "THT_RESIDUAL_CONTEXT_HIDDEN", config.THT_RESIDUAL_CONTEXT_HIDDEN)
            ),
            "residual_min_bin_width": float(
                getattr(config, "THT_RESIDUAL_MIN_BIN_WIDTH", config.THT_RESIDUAL_MIN_BIN_WIDTH)
            ),
            "residual_min_bin_height": float(
                getattr(config, "THT_RESIDUAL_MIN_BIN_HEIGHT", config.THT_RESIDUAL_MIN_BIN_HEIGHT)
            ),
            "residual_min_deriv": float(
                getattr(config, "THT_RESIDUAL_MIN_DERIV", config.THT_RESIDUAL_MIN_DERIV)
            ),
            "residual_eps": float(getattr(config, "THT_RESIDUAL_EPS", config.THT_RESIDUAL_EPS)),
        }

        checkpoint_path = output_dir / "tht_tripgen.pt"
        save_checkpoint(
            checkpoint_path,
            model_state=model.state_dict(),
            meta=meta,
            optimizer_state=optimizer.state_dict(),
            epoch=epoch,
            metrics={"best_val": best_val},
        )

        def _evaluate_split(
            *,
            split_name: str,
            eval_df: pd.DataFrame,
            eval_idx: pd.DataFrame,
            save_synth: bool,
        ) -> None:
            if len(eval_idx) == 0:
                print(f"[THT-TripGen] skip {split_name}: empty split")
                return

            n_eval = _eval_sample_size(len(eval_idx))
            if n_eval < len(eval_idx):
                sample_idx = eval_idx.sample(n=n_eval, random_state=config.GLOBAL_SEED).index
                eval_idx_sample = eval_idx.loc[sample_idx].reset_index(drop=True)
                eval_df_sample = eval_df.loc[sample_idx].reset_index(drop=True)
            else:
                eval_idx_sample = eval_idx
                eval_df_sample = eval_df

            sample_start = time.monotonic()
            synth_idx = _sample_conditioned(
                model,
                eval_idx_sample,
                conditional_cols,
                temperature=float(getattr(config, "THT_TEMPERATURE", config.THT_SAMPLE_TEMPERATURE)),
                device=device,
            )
            sample_time_min = (time.monotonic() - sample_start) / 60.0

            synth_decoded = transformer.decode(synth_idx)
            synth_metrics_df = synth_decoded[config.OUTPUT_COLUMNS].copy()
            eval_metrics_df = eval_df_sample[config.OUTPUT_COLUMNS].copy()

            metrics: Dict[str, float | str] = compute_metrics(
                eval_metrics_df,
                synth_metrics_df,
                order_key=f"tht_tripgen_{split_name}",
                output_dir=metrics_dir,
                plot_dir=plot_dir,
            )
            paper_metrics = compute_paper_metrics(train_df, eval_metrics_df, synth_metrics_df)
            metrics.update(paper_metrics)
            metrics["eval_split"] = split_name
            metrics["n_train"] = float(len(train_df))
            metrics["n_eval"] = float(len(eval_metrics_df))
            metrics["n_synth"] = float(len(synth_metrics_df))
            metrics["best_val"] = float(best_val)
            metrics["train_time_min"] = float(train_time_min)
            metrics["sample_time_min"] = float(sample_time_min)
            metrics["total_time_min"] = float(train_time_min + sample_time_min)

            metrics_path = metrics_dir / f"metrics_tht_tripgen_{split_name}.json"
            save_metrics_json(metrics, metrics_path)

            if split_name == "val":
                save_metrics_json(metrics, output_dir / "metrics_tht_tripgen.json")
            elif split_name == "hold":
                save_metrics_json(metrics, output_dir / "metrics_tht_tripgen_hold.json")

            synth_path = data_dir / f"synthetic_tht_tripgen_{split_name}.csv"
            synth_decoded.to_csv(synth_path, index=False)
            if save_synth:
                synth_decoded.to_csv(
                    output_dir / "synthetic_tht_tripgen_hold.csv", index=False
                )

            print(f"[THT-TripGen] metrics saved: {metrics_path}")
            print(f"[THT-TripGen] synthetic saved: {synth_path}")

        # Keep val/hold evaluation separate (val for selection, hold for reporting).
        _evaluate_split(split_name="val", eval_df=val_df, eval_idx=val_idx, save_synth=False)
        if len(hold_df) > 0:
            _evaluate_split(
                split_name="hold",
                eval_df=hold_df,
                eval_idx=hold_idx,
                save_synth=True,
            )

        print(f"[THT-TripGen] saved checkpoint={checkpoint_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Train THT-TripGen model")
    parser.add_argument("--run-tag", default="tht_tripgen", help="output subdir name")
    parser.add_argument("--device", default=None, help="torch device override")
    args = parser.parse_args()

    device = (
        torch.device(args.device)
        if args.device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    train_tht_tripgen(run_tag=args.run_tag, device=device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
