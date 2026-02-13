from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import config
from data_processing.tht_tripgen_loader import load_and_split_tht_tripgen
from data_processing.tht_tripgen_transformer import THTTripGenTransformer
from sample_tht_tripgen import sample_tht_tripgen
from utils.evaluation import compute_attribute_metrics, compute_metrics, compute_paper_metrics
from utils.metrics import save_metrics_json
from utils.privacy import build_privacy_report, save_privacy_report
from utils.serialization import (
    THT_TRIPGEN_CHECKPOINT_FILENAME,
    THT_TRIPGEN_MAPPINGS_FILENAME,
    load_tht_tripgen_mappings,
)


def _resolve_paths(run_dir: Path) -> tuple[Path, Path]:
    checkpoint = run_dir / THT_TRIPGEN_CHECKPOINT_FILENAME
    mappings = run_dir / THT_TRIPGEN_MAPPINGS_FILENAME
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if not mappings.exists():
        raise FileNotFoundError(f"Mappings not found: {mappings}")
    return checkpoint, mappings


def _filter_known(df: pd.DataFrame, transformer: THTTripGenTransformer) -> pd.DataFrame:
    idx = transformer.transform(df, drop_unknown=False)
    mask = idx.notna().all(axis=1) & idx["r"].notna()
    return df.loc[mask].reset_index(drop=True)


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


def _metrics_columns(*dfs: pd.DataFrame) -> list[str]:
    cols = list(config.OUTPUT_COLUMNS)
    residual_col = str(getattr(config, "DISTANCE_RESIDUAL_COL", "r"))
    use_residual = bool(getattr(config, "DISTANCE_USE_RESIDUAL", True))
    if use_residual and residual_col not in cols:
        if all(residual_col in df.columns for df in dfs):
            cols.append(residual_col)
    return cols


def _attribute_columns() -> tuple[list[str], list[str]]:
    passenger_cols = list(
        getattr(
            config,
            "THT_ATTRIBUTE_DISCRETE_COLUMNS",
            [str(getattr(config, "PASSENGER_COL", "passenger_count"))],
        )
    )
    fare_cols = list(
        getattr(
            config,
            "THT_ATTRIBUTE_CONTINUOUS_COLUMNS",
            [str(getattr(config, "FARE_COL", "total_amount"))],
        )
    )
    return passenger_cols, fare_cols


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path(config.SAVE_DATA_DIR) / "tht_tripgen",
        help="Diretorio do experimento (contendo checkpoint e mappings).",
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--mappings", type=Path, default=None)
    parser.add_argument("--rows", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=config.THT_SAMPLE_TEMPERATURE)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--split", type=str, choices=["val", "hold"], default=None)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--include-within-dcr", action="store_true")
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=None,
        help="Diretorio para salvar graficos (default: <run-dir>/plots).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Arquivo de metrics (default: <run-dir>/metrics_tht_tripgen_<split>.json).",
    )
    parser.add_argument("--force-sample", action="store_true")
    args = parser.parse_args()

    run_dir = args.run_dir
    checkpoint = args.checkpoint
    mappings = args.mappings
    if checkpoint is None or mappings is None:
        inferred_ckpt, inferred_map = _resolve_paths(run_dir)
        checkpoint = checkpoint or inferred_ckpt
        mappings = mappings or inferred_map

    if args.no_plots:
        import utils.evaluation as eval_mod

        eval_mod.plot_marginal_hist = lambda *_, **__: None
        eval_mod.plot_topk = lambda *_, **__: None

    train_df, val_df, hold_df = load_and_split_tht_tripgen()
    transformer = load_tht_tripgen_mappings(mappings)

    train_df = _filter_known(train_df, transformer)
    val_df = _filter_known(val_df, transformer)
    hold_df = _filter_known(hold_df, transformer)

    split_name, eval_df = _choose_split(val_df, hold_df, split=args.split)

    if len(eval_df) == 0:
        raise RuntimeError("eval_df vazio; ajuste split ou filtros.")

    if args.rows is None:
        ratio = float(getattr(config, "THT_EVAL_SAMPLE_RATIO", 1.0))
        n_eval = int(np.ceil(len(eval_df) * ratio))
        n_eval = max(1, n_eval) if len(eval_df) > 0 else 0
    else:
        n_eval = int(args.rows)
    n_eval = min(n_eval, len(eval_df))

    if n_eval < len(eval_df):
        eval_df = eval_df.sample(
            n=n_eval,
            random_state=args.seed if args.seed is not None else config.GLOBAL_SEED,
        ).reset_index(drop=True)

    synth_path = run_dir / f"synthetic_tht_tripgen_{split_name}.csv"
    if args.force_sample or not synth_path.exists():
        start = time.monotonic()
        synth_path = sample_tht_tripgen(
            run_dir=run_dir,
            split=split_name,
            rows=n_eval,
            temperature=args.temperature,
            seed=args.seed,
            privacy_report=False,
        )
        elapsed_min = (time.monotonic() - start) / 60.0
        print(f"[Recalc THT-TripGen] sampled {n_eval} rows in {elapsed_min:.2f} min")

    synth_df = pd.read_csv(synth_path)

    if len(synth_df) < len(eval_df):
        eval_df = eval_df.sample(
            n=len(synth_df),
            random_state=args.seed if args.seed is not None else config.GLOBAL_SEED,
        ).reset_index(drop=True)

    metrics_cols = _metrics_columns(train_df, eval_df, synth_df)
    eval_metrics_df = eval_df[metrics_cols].copy()
    synth_metrics_df = synth_df[metrics_cols].copy()

    plot_dir = args.plot_dir or (run_dir / "plots")
    metrics = compute_metrics(
        eval_metrics_df,
        synth_metrics_df,
        order_key=f"tht_tripgen_{split_name}",
        output_dir=run_dir,
        plot_dir=plot_dir,
    )

    paper_metrics = compute_paper_metrics(
        train_df,
        eval_df,
        synth_df,
        include_within=args.include_within_dcr,
    )
    metrics.update(paper_metrics)
    if bool(getattr(config, "EVAL_ENABLE_ATTRIBUTE_METRICS", True)):
        passenger_cols, fare_cols = _attribute_columns()
        metrics.update(
            compute_attribute_metrics(
                eval_df,
                synth_df,
                passenger_cols=passenger_cols,
                fare_cols=fare_cols,
                plot_dir=plot_dir,
                order_key=f"tht_tripgen_{split_name}",
            )
        )
    metrics["eval_split"] = split_name
    metrics["n_real"] = float(len(eval_metrics_df))
    metrics["n_synth"] = float(len(synth_metrics_df))

    if bool(getattr(config, "PRIVACY_REPORT_ENABLE", True)):
        report, match_metrics = build_privacy_report(
            train_df,
            synth_df,
            discrete_cols=config.OUTPUT_COLUMNS,
            r_col=str(getattr(config, "DISTANCE_RESIDUAL_COL", "r")),
            r_decimals=int(getattr(config, "PRIVACY_MATCH_R_DECIMALS", 2)),
            max_samples=getattr(config, "DCR_MAX_SAMPLES", None),
            seed=args.seed if args.seed is not None else config.GLOBAL_SEED,
            chunk_size=getattr(config, "DCR_CHUNK_SIZE", 1024),
            w_time=getattr(config, "COVERAGE_TIME_WEIGHT", 1.0),
            w_space=getattr(config, "COVERAGE_SPACE_WEIGHT", 1.0),
            w_residual=getattr(config, "DCR_RESIDUAL_WEIGHT", 1.0),
            use_residual=getattr(config, "DISTANCE_USE_RESIDUAL", True),
            residual_col=getattr(config, "DISTANCE_RESIDUAL_COL", "r"),
        )
        metrics.update(match_metrics)
        report_dir = run_dir / "metrics"
        report_path = report_dir / f"privacy_report_tht_tripgen_{split_name}.json"
        save_privacy_report(
            report,
            report_path,
            split=split_name,
            n_train=float(len(train_df)),
            n_synth=float(len(synth_df)),
            r_decimals=int(getattr(config, "PRIVACY_MATCH_R_DECIMALS", 2)),
        )

    if args.output is None:
        if split_name == "val":
            out_path = run_dir / "metrics_tht_tripgen.json"
        else:
            out_path = run_dir / "metrics_tht_tripgen_hold.json"
    else:
        out_path = args.output

    save_metrics_json(metrics, out_path)
    print(f"[Recalc THT-TripGen] saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
