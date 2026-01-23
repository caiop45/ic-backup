from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import config
from utils.helpers import set_seed
from utils.serialization import build_model_from_checkpoint, load_checkpoint, load_mappings


def _infer_output_path(checkpoint_path: Path, output_dir: Path) -> Path:
    stem = checkpoint_path.stem
    return output_dir / f"samples_{stem}.csv"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mappings", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--rows", type=int, default=config.SAMPLE_ROWS)
    parser.add_argument("--batch-size", type=int, default=10000)
    parser.add_argument("--temperature", type=float, default=config.SAMPLE_TEMPERATURE)
    parser.add_argument("--seed", type=int, default=config.GLOBAL_SEED)
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state = load_checkpoint(args.checkpoint, device=device)
    model = build_model_from_checkpoint(state).to(device)
    model.load_state_dict(state["model_state"])
    model.eval()

    transformer = load_mappings(args.mappings)

    rows = int(args.rows)
    rows = max(1, rows)
    temperature = float(args.temperature)

    batch_size = int(args.batch_size)
    if batch_size <= 0:
        batch_size = rows

    all_samples = {col: [] for col in transformer.columns}
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
    df_decoded = df_decoded[config.OUTPUT_COLUMNS]

    output_path = args.output
    if output_path is None:
        output_path = _infer_output_path(args.checkpoint, Path(config.SAVE_DATA_DIR))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_decoded.to_csv(output_path, index=False)

    print(f"[Sample] saved {len(df_decoded)} rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
