from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from tvae import config
from tvae.data_processing.tht_tripgen_loader import load_and_split_tht_tripgen
from tvae.data_processing.tht_tripgen_transformer import THTTripGenTransformer
from tvae.utils.privacy import build_privacy_report, rejection_filter, save_privacy_report
from tvae.utils.serialization import (
    THT_TRIPGEN_CHECKPOINT_FILENAME,
    THT_TRIPGEN_MAPPINGS_FILENAME,
    build_tht_tripgen_model_from_checkpoint,
    load_checkpoint,
    load_tht_tripgen_mappings,
)


def _choose_eval_split(
    val_df: pd.DataFrame, hold_df: pd.DataFrame, *, split: str | None
) -> tuple[str, pd.DataFrame]:
    if split is None:
        return ("hold", hold_df) if len(hold_df) > 0 else ("val", val_df)
    if split not in ("val", "hold"):
        raise ValueError("split must be 'val' or 'hold'")
    if split == "hold" and len(hold_df) == 0:
        return "val", val_df
    return split, hold_df if split == "hold" else val_df


def _filter_known(df: pd.DataFrame, transformer: THTTripGenTransformer) -> pd.DataFrame:
    idx = transformer.transform(df, drop_unknown=False)
    mask = idx.notna().all(axis=1) & idx["r"].notna()
    return df.loc[mask].reset_index(drop=True)


def _sample_conditioned(
    model: torch.nn.Module,
    eval_idx: pd.DataFrame,
    conditional_cols: List[str],
    *,
    temperature: float,
    device: torch.device,
    seed: int | None = None,
    batch_size: int = 10000,
) -> pd.DataFrame:
    if len(eval_idx) == 0:
        return pd.DataFrame()

    outputs: Dict[str, List[np.ndarray]] = {}
    model.eval()
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
            batch_seed = None if seed is None else int(seed) + int(start)
            samples = model.sample(
                n=end - start,
                u=u,
                temperature=temperature,
                seed=batch_seed,
            )
            for key, tensor in samples.items():
                outputs.setdefault(key, []).append(tensor.detach().cpu().numpy())

    data = {key: np.concatenate(chunks) for key, chunks in outputs.items()}
    return pd.DataFrame(data)


def sample_tht_tripgen(
    *,
    run_dir: Path,
    split: str | None = None,
    rows: int | None = None,
    temperature: float | None = None,
    seed: int | None = None,
    device: torch.device | None = None,
    privacy_report: bool | None = None,
    reject_close: bool | None = None,
    tau: float | None = None,
    rejection_max_iters: int | None = None,
    privacy_report_dir: Path | None = None,
) -> Path:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_dir = Path(run_dir)
    checkpoint_path = run_dir / THT_TRIPGEN_CHECKPOINT_FILENAME
    mappings_path = run_dir / THT_TRIPGEN_MAPPINGS_FILENAME

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
    if not mappings_path.exists():
        raise FileNotFoundError(f"Missing mappings: {mappings_path}")

    transformer: THTTripGenTransformer = load_tht_tripgen_mappings(mappings_path)
    state = load_checkpoint(checkpoint_path, device=device)
    model = build_tht_tripgen_model_from_checkpoint(state).to(device)
    model.load_state_dict(state["model_state"])
    model.eval()

    train_df, val_df, hold_df = load_and_split_tht_tripgen()
    split_name, eval_df = _choose_eval_split(val_df, hold_df, split=split)

    eval_idx = transformer.transform(eval_df, drop_unknown=True)
    train_df = _filter_known(train_df, transformer)

    if rows is not None:
        rows = int(rows)
        rows = max(0, rows)
        if rows < len(eval_idx):
            eval_idx = eval_idx.sample(
                n=rows,
                random_state=seed if seed is not None else config.GLOBAL_SEED,
            ).reset_index(drop=True)

    conditional_cols = list(transformer.conditional_columns)
    temperature = (
        float(temperature)
        if temperature is not None
        else float(getattr(config, "THT_SAMPLE_TEMPERATURE", 1.0))
    )

    if privacy_report is None:
        privacy_report = bool(getattr(config, "PRIVACY_REPORT_ENABLE", True))
    if reject_close is None:
        reject_close = bool(getattr(config, "PRIVACY_REJECTION_ENABLE", False))
    if tau is None:
        tau = float(getattr(config, "PRIVACY_REJECTION_TAU", 0.05))
    if rejection_max_iters is None:
        rejection_max_iters = int(getattr(config, "PRIVACY_REJECTION_MAX_ITERS", 20))

    def _generate(n: int, batch_seed: int | None) -> pd.DataFrame:
        if n <= 0 or len(eval_idx) == 0:
            return pd.DataFrame()
        replace = n > len(eval_idx)
        eval_idx_sample = eval_idx.sample(
            n=n,
            replace=replace,
            random_state=batch_seed if batch_seed is not None else None,
        ).reset_index(drop=True)
        synth_idx_local = _sample_conditioned(
            model,
            eval_idx_sample,
            conditional_cols,
            temperature=temperature,
            device=device,
            seed=batch_seed,
            batch_size=int(getattr(config, "THT_BATCH_SIZE", 1024)),
        )
        return transformer.decode(synth_idx_local)

    if reject_close:
        decoded, stats = rejection_filter(
            train_df,
            generator_fn=_generate,
            target_rows=len(eval_idx),
            tau=float(tau),
            max_iters=int(rejection_max_iters),
            seed=seed if seed is not None else config.GLOBAL_SEED,
            max_samples_train=getattr(config, "DCR_MAX_SAMPLES", None),
            chunk_size=getattr(config, "DCR_CHUNK_SIZE", 1024),
            w_time=getattr(config, "COVERAGE_TIME_WEIGHT", 1.0),
            w_space=getattr(config, "COVERAGE_SPACE_WEIGHT", 1.0),
            w_residual=getattr(config, "DCR_RESIDUAL_WEIGHT", 1.0),
            use_residual=getattr(config, "DISTANCE_USE_RESIDUAL", True),
            residual_col=getattr(config, "DISTANCE_RESIDUAL_COL", "r"),
        )
        if len(decoded) < len(eval_idx):
            print(
                f"[THT-TripGen] rejection filter accepted {len(decoded)}/{len(eval_idx)} rows "
                f"(accept_rate={stats.get('accept_rate', 0.0):.3f})"
            )
    else:
        decoded = _generate(len(eval_idx), seed)

    output_cols = [col for col in config.THT_TRIPGEN_COLUMNS if col in decoded.columns]
    out_df = decoded[output_cols].copy()
    out_path = run_dir / f"synthetic_tht_tripgen_{split_name}.csv"
    out_df.to_csv(out_path, index=False)

    if privacy_report:
        report, match_metrics = build_privacy_report(
            train_df,
            decoded,
            discrete_cols=config.OUTPUT_COLUMNS,
            r_col=str(getattr(config, "DISTANCE_RESIDUAL_COL", "r")),
            r_decimals=int(getattr(config, "PRIVACY_MATCH_R_DECIMALS", 2)),
            max_samples=getattr(config, "DCR_MAX_SAMPLES", None),
            seed=seed if seed is not None else config.GLOBAL_SEED,
            chunk_size=getattr(config, "DCR_CHUNK_SIZE", 1024),
            w_time=getattr(config, "COVERAGE_TIME_WEIGHT", 1.0),
            w_space=getattr(config, "COVERAGE_SPACE_WEIGHT", 1.0),
            w_residual=getattr(config, "DCR_RESIDUAL_WEIGHT", 1.0),
            use_residual=getattr(config, "DISTANCE_USE_RESIDUAL", True),
            residual_col=getattr(config, "DISTANCE_RESIDUAL_COL", "r"),
        )
        report_dir = privacy_report_dir or (run_dir / "metrics")
        report_path = report_dir / f"privacy_report_tht_tripgen_{split_name}.json"
        save_privacy_report(
            report,
            report_path,
            split=split_name,
            n_train=float(len(train_df)),
            n_synth=float(len(decoded)),
            r_decimals=int(getattr(config, "PRIVACY_MATCH_R_DECIMALS", 2)),
            rejection_enabled=bool(reject_close),
            tau=float(tau),
            match_metrics=match_metrics,
        )
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Sample from a trained THT-TripGen run")
    parser.add_argument("--run-dir", required=True, help="Path to THT-TripGen run directory")
    parser.add_argument(
        "--split",
        choices=["val", "hold"],
        default=None,
        help="Split to sample (default: hold if available else val)",
    )
    parser.add_argument("--rows", type=int, default=None, help="Override number of rows")
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature (default: config.THT_SAMPLE_TEMPERATURE)",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for sampling")
    parser.add_argument("--device", default=None, help="torch device override")
    parser.add_argument("--privacy-report", action="store_true", help="Generate privacy report")
    parser.add_argument(
        "--reject-close",
        action="store_true",
        help="Reject samples with min distance < tau",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=None,
        help="Rejection threshold (default: config.PRIVACY_REJECTION_TAU)",
    )
    args = parser.parse_args()

    device = (
        torch.device(args.device)
        if args.device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    out_path = sample_tht_tripgen(
        run_dir=Path(args.run_dir),
        split=args.split,
        rows=args.rows,
        temperature=args.temperature,
        seed=args.seed,
        device=device,
        privacy_report=True if args.privacy_report else None,
        reject_close=True if args.reject_close else None,
        tau=args.tau,
    )
    print(f"[THT-TripGen] saved synthetic data: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
