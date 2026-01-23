from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

import config
from data_processing.strategy_a_loader import load_and_split_strategy_a
from data_processing.strategy_a_transformer import StrategyATransformer
from utils.serialization import (
    STRATEGY_A_CHECKPOINT_FILENAME,
    STRATEGY_A_MAPPINGS_FILENAME,
    build_strategy_a_model_from_checkpoint,
    load_checkpoint,
    load_strategy_a_mappings,
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
                col: torch.from_numpy(batch_df[col].to_numpy(dtype=np.int64)).to(device)
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


def sample_strategy_a(
    *,
    run_dir: Path,
    split: str | None = None,
    rows: int | None = None,
    temperature: float | None = None,
    seed: int | None = None,
    device: torch.device | None = None,
) -> Path:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_dir = Path(run_dir)
    checkpoint_path = run_dir / STRATEGY_A_CHECKPOINT_FILENAME
    mappings_path = run_dir / STRATEGY_A_MAPPINGS_FILENAME

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
    if not mappings_path.exists():
        raise FileNotFoundError(f"Missing mappings: {mappings_path}")

    transformer: StrategyATransformer = load_strategy_a_mappings(mappings_path)
    state = load_checkpoint(checkpoint_path, device=device)
    model = build_strategy_a_model_from_checkpoint(state).to(device)
    model.load_state_dict(state["model_state"])
    model.eval()

    _, val_df, hold_df = load_and_split_strategy_a()
    split_name, eval_df = _choose_eval_split(val_df, hold_df, split=split)

    eval_idx = transformer.transform(eval_df, drop_unknown=True)

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
        else float(getattr(config, "SA_SAMPLE_TEMPERATURE", 1.0))
    )

    synth_idx = _sample_conditioned(
        model,
        eval_idx,
        conditional_cols,
        temperature=temperature,
        device=device,
        seed=seed,
        batch_size=int(getattr(config, "SA_BATCH_SIZE", 1024)),
    )

    decoded = transformer.decode(synth_idx)
    out_df = decoded[config.OUTPUT_COLUMNS].copy()
    out_path = run_dir / f"synthetic_strategy_a_{split_name}.csv"
    out_df.to_csv(out_path, index=False)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Sample from a trained Strategy A run")
    parser.add_argument("--run-dir", required=True, help="Path to Strategy A run directory")
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
        help="Sampling temperature (default: config.SA_SAMPLE_TEMPERATURE)",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for sampling")
    parser.add_argument("--device", default=None, help="torch device override")
    args = parser.parse_args()

    device = (
        torch.device(args.device)
        if args.device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    out_path = sample_strategy_a(
        run_dir=Path(args.run_dir),
        split=args.split,
        rows=args.rows,
        temperature=args.temperature,
        seed=args.seed,
        device=device,
    )
    print(f"[Strategy A] saved synthetic data: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
