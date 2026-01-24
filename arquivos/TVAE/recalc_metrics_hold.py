from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import torch

import config
import train_tvae
from data_processing.loader import load_and_split
from utils.evaluation import compute_metrics, compute_paper_metrics
from utils.helpers import set_seed
from utils.metrics import save_metrics_json
from utils.serialization import build_model_from_checkpoint, load_checkpoint, load_mappings


def _resolve_checkpoint(run_dir: Path) -> Path:
    checkpoint = run_dir / "tvae_order_1.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    return checkpoint


def _resolve_mappings(run_dir: Path) -> Path:
    mappings = run_dir / "mappings_order_1.json"
    if not mappings.exists():
        raise FileNotFoundError(f"Mappings not found: {mappings}")
    return mappings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path(config.SAVE_DATA_DIR) / "baseline",
        help="Diretorio do experimento (contendo checkpoint e mappings).",
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--mappings", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=100000)
    parser.add_argument("--temperature", type=float, default=config.SAMPLE_TEMPERATURE)
    parser.add_argument("--rows", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--order-key", type=str, default="order_1")
    parser.add_argument("--force-sample", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
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
        help="Arquivo de metrics (default: <run-dir>/metrics/metrics_tvae_order_1_hold.json).",
    )
    args = parser.parse_args()

    run_dir = args.run_dir
    checkpoint = args.checkpoint
    mappings = args.mappings
    if mappings is None:
        mappings = _resolve_mappings(run_dir)

    train_df, _, hold_df = load_and_split()
    transformer = load_mappings(mappings)

    train_df = train_tvae._filter_known(train_df, transformer)
    hold_df = train_tvae._filter_known(hold_df, transformer)
    if len(hold_df) == 0:
        raise RuntimeError("hold_df vazio; ajuste TRAIN_FRAC/VAL_FRAC ou use outro split.")

    model_key = args.order_key
    if not model_key.startswith("tvae_"):
        model_key = f"tvae_{model_key}"
    synth_path = run_dir / "data" / f"synthetic_{model_key}_hold.csv"

    if synth_path.exists() and not args.force_sample:
        synth_df = pd.read_csv(synth_path)
        print(f"[Recalc] loaded synth from {synth_path}")
    else:
        if checkpoint is None:
            checkpoint = _resolve_checkpoint(run_dir)

        if args.device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(args.device)

        state = load_checkpoint(checkpoint, device=device)
        model = build_model_from_checkpoint(state).to(device)
        model.load_state_dict(state["model_state"])
        model.eval()

        n_eval = len(hold_df) if args.rows is None else int(args.rows)
        set_seed(args.seed)
        start = time.monotonic()
        synth_df = train_tvae._sample_synthetic(
            model,
            transformer,
            n_samples=n_eval,
            temperature=args.temperature,
            device=device,
            batch_size=args.batch_size,
        )
        elapsed_min = (time.monotonic() - start) / 60.0
        synth_path.parent.mkdir(parents=True, exist_ok=True)
        synth_df.to_csv(synth_path, index=False)
        print(f"[Recalc] sampled {n_eval} rows in {elapsed_min:.2f} min on {device}")
        print(f"[Recalc] saved synth to {synth_path}")

    if args.no_plots:
        import utils.evaluation as eval_mod

        eval_mod.plot_marginal_hist = lambda *_, **__: None
        eval_mod.plot_topk = lambda *_, **__: None

    plot_dir = args.plot_dir or (run_dir / "plots")
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics = compute_metrics(
        hold_df,
        synth_df,
        order_key=f"{model_key}_hold",
        output_dir=metrics_dir,
        plot_dir=plot_dir,
    )

    paper_metrics = compute_paper_metrics(train_df, hold_df, synth_df)
    metrics.update(paper_metrics)
    metrics["eval_split"] = "hold"
    metrics["n_real"] = float(len(hold_df))
    metrics["n_synth"] = float(len(synth_df))

    out_path = args.output or (metrics_dir / f"metrics_{model_key}_hold.json")
    save_metrics_json(metrics, out_path)
    print(f"[Recalc] saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
