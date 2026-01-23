from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch

import config
from data_processing.loader import load_and_split
from utils.helpers import set_seed
from utils.metrics import chi2_counts, jsd_counts, od_counts
from utils.serialization import build_model_from_checkpoint, load_checkpoint, load_mappings


def _filter_known(df: pd.DataFrame, transformer) -> pd.DataFrame:
    mask = np.ones(len(df), dtype=bool)
    for col in transformer.columns:
        allowed = set(transformer.categories[col])
        mask &= df[col].isin(allowed)
    return df.loc[mask].reset_index(drop=True)


def _sample_synthetic(
    model,
    transformer,
    *,
    rows: int,
    temperature: float,
    device: torch.device,
    batch_size: int,
) -> pd.DataFrame:
    rows = max(1, int(rows))
    batch_size = int(batch_size)
    if batch_size <= 0:
        batch_size = rows

    all_samples: Dict[str, list[np.ndarray]] = {col: [] for col in transformer.columns}
    with torch.no_grad():
        for start in range(0, rows, batch_size):
            end = min(start + batch_size, rows)
            current_batch = end - start

            z = torch.randn(current_batch, model.latent_dim, device=device)
            samples = model.sample(z, temperature=temperature)
            for col in transformer.columns:
                all_samples[col].append(samples[col].cpu().numpy())

            del z, samples
            if device.type == "cuda":
                torch.cuda.empty_cache()

    data = {col: np.concatenate(all_samples[col]) for col in transformer.columns}
    df_idx = pd.DataFrame(data)
    df_decoded = transformer.decode_indices(df_idx)
    return df_decoded[config.OUTPUT_COLUMNS]


def _load_real(split: str) -> pd.DataFrame:
    train_df, val_df, hold_df = load_and_split()
    if split == "train":
        return train_df
    if split == "val":
        return val_df
    if split == "hold":
        return hold_df
    raise ValueError(f"Unknown split: {split}")


def _align_counts(
    real_counts: pd.Series, synth_counts: pd.Series
) -> Tuple[pd.Index, pd.Series, pd.Series]:
    idx = real_counts.index.union(synth_counts.index)
    real = real_counts.reindex(idx, fill_value=0)
    synth = synth_counts.reindex(idx, fill_value=0)
    return idx, real, synth


def _topk_diff_table(
    real_counts: pd.Series, synth_counts: pd.Series, k: int
) -> pd.DataFrame:
    idx, real, synth = _align_counts(real_counts, synth_counts)
    real_total = real.sum()
    synth_total = synth.sum()
    real_prob = real / max(1.0, real_total)
    synth_prob = synth / max(1.0, synth_total)
    diff = synth_prob - real_prob
    out = pd.DataFrame(
        {
            "key": idx.astype(str),
            "real_count": real.to_numpy(),
            "synth_count": synth.to_numpy(),
            "real_prob": real_prob.to_numpy(),
            "synth_prob": synth_prob.to_numpy(),
            "diff": diff.to_numpy(),
            "abs_diff": diff.abs().to_numpy(),
        }
    )
    return out.sort_values("abs_diff", ascending=False).head(k).reset_index(drop=True)


def _topk_signed_table(
    real_counts: pd.Series, synth_counts: pd.Series, k: int, *, descending: bool
) -> pd.DataFrame:
    idx, real, synth = _align_counts(real_counts, synth_counts)
    real_total = real.sum()
    synth_total = synth.sum()
    real_prob = real / max(1.0, real_total)
    synth_prob = synth / max(1.0, synth_total)
    diff = synth_prob - real_prob
    out = pd.DataFrame(
        {
            "key": idx.astype(str),
            "real_count": real.to_numpy(),
            "synth_count": synth.to_numpy(),
            "real_prob": real_prob.to_numpy(),
            "synth_prob": synth_prob.to_numpy(),
            "diff": diff.to_numpy(),
        }
    )
    out = out.sort_values("diff", ascending=not descending).head(k).reset_index(drop=True)
    return out


def _degree_counts(df: pd.DataFrame, *, source: str, target: str) -> pd.Series:
    deg = df.groupby(source)[target].nunique()
    return deg.value_counts().sort_index()


def _conditional_jsd(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    *,
    focus_col: str,
    target_col: str,
    top_n: int,
) -> pd.DataFrame:
    top_vals = real_df[focus_col].value_counts().head(top_n).index.tolist()
    rows = []
    for val in top_vals:
        real_sub = real_df[real_df[focus_col] == val][target_col].value_counts()
        synth_sub = synth_df[synth_df[focus_col] == val][target_col].value_counts()
        rows.append(
            {
                focus_col: int(val),
                "real_rows": int(real_sub.sum()),
                "synth_rows": int(synth_sub.sum()),
                "jsd": jsd_counts(real_sub, synth_sub),
                "chi2": chi2_counts(real_sub, synth_sub),
            }
        )
    return pd.DataFrame(rows).sort_values("jsd", ascending=False).reset_index(drop=True)


def _top_mass(counts: pd.Series, k: int) -> float:
    total = float(counts.sum())
    if total <= 0:
        return 0.0
    return float(counts.sort_values(ascending=False).head(k).sum() / total)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--mappings", type=Path, required=True)
    parser.add_argument("--synthetic", type=Path, default=None)
    parser.add_argument("--rows", type=int, default=2_000_000)
    parser.add_argument("--batch-size", type=int, default=10000)
    parser.add_argument("--temperature", type=float, default=config.SAMPLE_TEMPERATURE)
    parser.add_argument("--seed", type=int, default=config.GLOBAL_SEED)
    parser.add_argument("--split", choices=["train", "val", "hold"], default="val")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--cond-topk", type=int, default=30)
    parser.add_argument("--save-synth", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)

    transformer = load_mappings(args.mappings)

    if args.synthetic is None and args.checkpoint is None:
        raise ValueError("Provide --synthetic or --checkpoint for sampling.")

    if args.synthetic is not None:
        synth_df = pd.read_csv(args.synthetic)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state = load_checkpoint(args.checkpoint, device=device)
        model = build_model_from_checkpoint(state).to(device)
        model.load_state_dict(state["model_state"])
        model.eval()
        synth_df = _sample_synthetic(
            model,
            transformer,
            rows=args.rows,
            temperature=args.temperature,
            device=device,
            batch_size=args.batch_size,
        )
        if args.save_synth:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            synth_path = args.output_dir / "synthetic_samples.csv"
            synth_df.to_csv(synth_path, index=False)

    missing_cols = [c for c in config.OUTPUT_COLUMNS if c not in synth_df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in synthetic data: {missing_cols}")

    real_df = _load_real(args.split)
    real_df = _filter_known(real_df, transformer)
    if real_df.empty:
        raise ValueError("Real data empty after filtering known categories.")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    real_pickup = real_df["pickup_id"].value_counts().sort_index()
    synth_pickup = synth_df["pickup_id"].value_counts().sort_index()
    real_dropoff = real_df["dropoff_id"].value_counts().sort_index()
    synth_dropoff = synth_df["dropoff_id"].value_counts().sort_index()

    real_od = od_counts(real_df)
    synth_od = od_counts(synth_df)

    real_keys = set(real_od.index)
    synth_keys = set(synth_od.index)
    inter = real_keys & synth_keys

    real_total = float(real_od.sum())
    synth_total = float(synth_od.sum())
    real_prob = real_od / max(1.0, real_total)
    synth_prob = synth_od / max(1.0, synth_total)

    invalid_mass = float(synth_prob[~synth_prob.index.isin(real_keys)].sum())
    missing_mass = float(real_prob[~real_prob.index.isin(synth_keys)].sum())

    metrics = {
        "split": args.split,
        "n_real": float(len(real_df)),
        "n_synth": float(len(synth_df)),
        "pickup_jsd": jsd_counts(real_pickup, synth_pickup),
        "pickup_chi2": chi2_counts(real_pickup, synth_pickup),
        "dropoff_jsd": jsd_counts(real_dropoff, synth_dropoff),
        "dropoff_chi2": chi2_counts(real_dropoff, synth_dropoff),
        "od_jsd": jsd_counts(real_od, synth_od),
        "od_chi2": chi2_counts(real_od, synth_od),
        "od_coverage_real": float(len(inter) / max(1, len(real_keys))),
        "od_coverage_synth": float(len(inter) / max(1, len(synth_keys))),
        "od_unique_real": float(len(real_keys)),
        "od_unique_synth": float(len(synth_keys)),
        "od_invalid_mass": invalid_mass,
        "od_missing_mass": missing_mass,
        "od_top50_mass_real": _top_mass(real_od, 50),
        "od_top50_mass_synth": _top_mass(synth_od, 50),
        "od_top100_mass_real": _top_mass(real_od, 100),
        "od_top100_mass_synth": _top_mass(synth_od, 100),
        "od_top500_mass_real": _top_mass(real_od, 500),
        "od_top500_mass_synth": _top_mass(synth_od, 500),
    }

    out_degree_real = _degree_counts(real_df, source="pickup_id", target="dropoff_id")
    out_degree_synth = _degree_counts(synth_df, source="pickup_id", target="dropoff_id")
    in_degree_real = _degree_counts(real_df, source="dropoff_id", target="pickup_id")
    in_degree_synth = _degree_counts(synth_df, source="dropoff_id", target="pickup_id")

    metrics.update(
        {
            "out_degree_jsd": jsd_counts(out_degree_real, out_degree_synth),
            "out_degree_chi2": chi2_counts(out_degree_real, out_degree_synth),
            "in_degree_jsd": jsd_counts(in_degree_real, in_degree_synth),
            "in_degree_chi2": chi2_counts(in_degree_real, in_degree_synth),
        }
    )

    od_diff_topk = _topk_diff_table(real_od, synth_od, args.topk)
    od_over_topk = _topk_signed_table(real_od, synth_od, args.topk, descending=True)
    od_under_topk = _topk_signed_table(real_od, synth_od, args.topk, descending=False)

    pickup_diff_topk = _topk_diff_table(real_pickup, synth_pickup, args.topk)
    dropoff_diff_topk = _topk_diff_table(real_dropoff, synth_dropoff, args.topk)

    pickup_conditional = _conditional_jsd(
        real_df,
        synth_df,
        focus_col="pickup_id",
        target_col="dropoff_id",
        top_n=args.cond_topk,
    )
    dropoff_conditional = _conditional_jsd(
        real_df,
        synth_df,
        focus_col="dropoff_id",
        target_col="pickup_id",
        top_n=args.cond_topk,
    )

    (output_dir / "spatial_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    od_diff_topk.to_csv(output_dir / "od_diff_topk.csv", index=False)
    od_over_topk.to_csv(output_dir / "od_over_topk.csv", index=False)
    od_under_topk.to_csv(output_dir / "od_under_topk.csv", index=False)
    pickup_diff_topk.to_csv(output_dir / "pickup_diff_topk.csv", index=False)
    dropoff_diff_topk.to_csv(output_dir / "dropoff_diff_topk.csv", index=False)
    pickup_conditional.to_csv(output_dir / "pickup_conditional_jsd.csv", index=False)
    dropoff_conditional.to_csv(output_dir / "dropoff_conditional_jsd.csv", index=False)

    out_idx, out_real, out_synth = _align_counts(out_degree_real, out_degree_synth)
    out_degree = pd.DataFrame(
        {
            "degree": out_idx.astype(int),
            "real_count": out_real.to_numpy(),
            "synth_count": out_synth.to_numpy(),
        }
    )
    in_idx, in_real, in_synth = _align_counts(in_degree_real, in_degree_synth)
    in_degree = pd.DataFrame(
        {
            "degree": in_idx.astype(int),
            "real_count": in_real.to_numpy(),
            "synth_count": in_synth.to_numpy(),
        }
    )
    out_degree.to_csv(output_dir / "out_degree_counts.csv", index=False)
    in_degree.to_csv(output_dir / "in_degree_counts.csv", index=False)

    print(f"[Spatial] saved outputs to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
