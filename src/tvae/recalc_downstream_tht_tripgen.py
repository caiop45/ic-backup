from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from tvae import config
from tvae.data_processing.tht_tripgen_loader import load_and_split_tht_tripgen
from tvae.utils.downstream_eval import FareEvalInputs, compute_downstream_fare_metrics
from tvae.utils.metrics import save_metrics_json


def _choose_split(
    val_df: pd.DataFrame, hold_df: pd.DataFrame, *, split: str | None
) -> tuple[str, pd.DataFrame]:
    if split is None:
        return ("hold", hold_df) if len(hold_df) > 0 else ("val", val_df)
    if split not in ("val", "hold"):
        raise ValueError("split must be 'val' or 'hold'")
    if split == "hold" and len(hold_df) == 0:
        return "val", val_df
    return split, hold_df if split == "hold" else val_df


def _resolve_synth_path(run_dir: Path, split_name: str) -> Path:
    direct = run_dir / f"synthetic_tht_tripgen_{split_name}.csv"
    if direct.exists():
        return direct
    data_dir = run_dir / "data" / f"synthetic_tht_tripgen_{split_name}.csv"
    if data_dir.exists():
        return data_dir
    raise FileNotFoundError(
        f"Synthetic file not found for split '{split_name}' in {run_dir}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path(config.SAVE_DATA_DIR) / "tht_tripgen",
        help="Run directory containing synthetic datasets.",
    )
    parser.add_argument("--split", type=str, choices=["val", "hold"], default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--train-rows", type=int, default=40000)
    parser.add_argument("--test-rows", type=int, default=20000)
    parser.add_argument("--model", type=str, default="gbr", choices=["gbr", "hgbr", "linear"])
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file (default: <run-dir>/metrics/downstream_tht_tripgen_<split>.json).",
    )
    args = parser.parse_args()

    train_df, val_df, hold_df = load_and_split_tht_tripgen()
    split_name, eval_df = _choose_split(val_df, hold_df, split=args.split)

    synth_path = _resolve_synth_path(args.run_dir, split_name)
    synth_df = pd.read_csv(synth_path)

    metrics_dir = args.run_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    seed = args.seed if args.seed is not None else config.GLOBAL_SEED
    inputs = FareEvalInputs(
        train_df=train_df, eval_df=eval_df, synth_df=synth_df, eval_split=split_name
    )
    metrics = compute_downstream_fare_metrics(
        inputs,
        train_rows=args.train_rows,
        test_rows=args.test_rows,
        model_name=args.model,
        seed=seed,
    )

    if metrics.get("skipped_fare"):
        reason = metrics.get("skip_reason", "unknown")
        print(f"[Downstream THT-TripGen] skipped fare eval: {reason}")

    out_path = args.output or (metrics_dir / f"downstream_tht_tripgen_{split_name}.json")
    save_metrics_json(metrics, out_path)
    print(f"[Downstream THT-TripGen] saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
