from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd
import torch

import config
import train_tvae
from data_processing.loader import load_and_split
from utils.metrics import marginal_counts, plot_marginal_hist, save_metrics_json
from utils.serialization import build_model_from_checkpoint, load_checkpoint, load_mappings


PAPER_TVAE = {
    "run": "paper_TVAE",
    "w1_tr_te": 0.1262,
    "w1_tr_syn": 0.9093,
    "w1_te_syn": 0.9057,
    "g_tr_te": 73.17,
    "g_tr_syn": 33.21,
    "g_te_syn": 32.67,
    "cov_tr_te": 74.78,
    "cov_tr_syn": 13.94,
    "cov_te_syn": 13.79,
}


def _write_markdown_table(df: pd.DataFrame, path: Path) -> None:
    headers = list(df.columns)
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    def fmt(val: object) -> str:
        if isinstance(val, float):
            return f"{val:.4f}"
        return str(val)

    for _, row in df.iterrows():
        lines.append("| " + " | ".join(fmt(row[h]) for h in headers) + " |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _resolve_device(device_str: str) -> torch.device:
    if device_str == "cuda" and not torch.cuda.is_available():
        print("[Warn] cuda nao disponivel; usando cpu.")
        return torch.device("cpu")
    return torch.device(device_str)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=100000)
    parser.add_argument("--rows", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=config.SAMPLE_TEMPERATURE)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "comparacao_graficos",
    )
    args = parser.parse_args()

    device = _resolve_device(args.device)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    shared_dir = out_dir / "_shared"
    shared_dir.mkdir(parents=True, exist_ok=True)

    save_root = Path("/home-ext/caioloss/Dados/TVAE/save_data")
    runs = {
        "baseline": save_root / "baseline",
        "weighted_pickup_dropoff": save_root / "weighted_pickup_dropoff" / "baseline",
        "pair_kl_conditional": save_root / "pair_kl_conditional" / "baseline",
        "pickup_pair_kl": save_root / "pickup_pair_kl" / "baseline",
    }

    print("[Info] carregando dados (split de acordo com config.py)")
    raw_train_df, raw_val_df, raw_hold_df = load_and_split()

    # train_vs_val dropoff plot (shared)
    tv_counts = marginal_counts(raw_train_df, "dropoff_id")
    vv_counts = marginal_counts(raw_val_df, "dropoff_id")
    train_vs_val_plot = shared_dir / "hist_train_vs_val_dropoff_id.png"
    plot_marginal_hist(
        tv_counts,
        vv_counts,
        title="train_vs_val dropoff_id",
        path=train_vs_val_plot,
    )

    metrics_rows = [PAPER_TVAE]

    for run_name, run_dir in runs.items():
        ckpt = run_dir / "tvae_order_1.pt"
        mappings = run_dir / "mappings_order_1.json"
        if not ckpt.exists() or not mappings.exists():
            print(f"[Warn] faltando checkpoint/mappings em {run_dir}, skip {run_name}")
            continue

        run_out = out_dir / run_name
        run_out.mkdir(parents=True, exist_ok=True)
        metrics_path = run_out / "metrics_order_1_hold.json"

        if args.skip_existing and metrics_path.exists():
            print(f"[Skip] {run_name} (metrics ja existe)")
        else:
            print(f"[Run] {run_name}")
            transformer = load_mappings(mappings)

            train_df = train_tvae._filter_known(raw_train_df, transformer)
            hold_df = train_tvae._filter_known(raw_hold_df, transformer)
            if len(hold_df) == 0:
                print(f"[Warn] hold vazio apos filtro, skip {run_name}")
                continue

            n_eval = len(hold_df) if args.rows is None else min(args.rows, len(hold_df))

            state = load_checkpoint(ckpt, device=device)
            model = build_model_from_checkpoint(state).to(device)
            model.load_state_dict(state["model_state"])
            model.eval()

            synth_df = train_tvae._sample_synthetic(
                model,
                transformer,
                n_samples=n_eval,
                temperature=args.temperature,
                device=device,
                batch_size=args.batch_size,
            )

            metrics = train_tvae._compute_metrics(
                hold_df,
                synth_df,
                order_key="order_1",
                output_dir=run_out,
                plot_dir=run_out,
            )
            paper_metrics = train_tvae._compute_paper_metrics(train_df, hold_df, synth_df)
            metrics.update(paper_metrics)
            metrics["eval_split"] = "hold"
            metrics["n_real"] = float(len(hold_df))
            metrics["n_synth"] = float(len(synth_df))
            save_metrics_json(metrics, metrics_path)

        # copiar train_vs_val dropoff plot para cada run
        shutil.copy2(train_vs_val_plot, run_out / "hist_train_vs_val_dropoff_id.png")

        # adicionar linha na tabela (se o json existir)
        if metrics_path.exists():
            with metrics_path.open("r", encoding="utf-8") as fh:
                m = json.load(fh)
            metrics_rows.append(
                {
                    "run": run_name,
                    "w1_tr_te": m.get("w1_tr_te"),
                    "w1_tr_syn": m.get("w1_tr_syn"),
                    "w1_te_syn": m.get("w1_te_syn"),
                    "g_tr_te": m.get("g_tr_te"),
                    "g_tr_syn": m.get("g_tr_syn"),
                    "g_te_syn": m.get("g_te_syn"),
                    "cov_tr_te": m.get("cov_tr_te"),
                    "cov_tr_syn": m.get("cov_tr_syn"),
                    "cov_te_syn": m.get("cov_te_syn"),
                    "n_real": m.get("n_real"),
                    "n_synth": m.get("n_synth"),
                }
            )

    df = pd.DataFrame(metrics_rows)
    df.to_csv(out_dir / "metrics_table.csv", index=False)
    _write_markdown_table(df, out_dir / "metrics_table.md")

    print(f"[Done] outputs em {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
