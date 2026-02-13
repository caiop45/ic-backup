from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _read_scalars(path: Path) -> Dict[str, List[Tuple[int, float]]]:
    data: Dict[str, List[Tuple[int, float]]] = {}
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row is None:
                continue
            tag = str(row.get("tag", "")).strip()
            if not tag:
                continue
            step = int(float(row.get("step", 0)))
            value = float(row.get("value", 0.0))
            data.setdefault(tag, []).append((step, value))
    return data


def _group_tags(tags: Iterable[str]) -> Dict[str, List[str]]:
    grouped: Dict[str, List[str]] = {}
    for tag in tags:
        prefix = tag.split("/", 1)[0] if "/" in tag else "misc"
        grouped.setdefault(prefix, []).append(tag)
    return grouped


def plot_training_curves(run_dir: Path, out_dir: Path | None = None) -> List[Path]:
    run_dir = Path(run_dir)
    out_dir = Path(out_dir) if out_dir is not None else run_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    scalars_path = run_dir / "logs" / "scalars.csv"
    if not scalars_path.exists():
        raise FileNotFoundError(f"scalars.csv not found: {scalars_path}")

    data = _read_scalars(scalars_path)
    grouped = _group_tags(data.keys())

    created: List[Path] = []
    group_order = ["train", "val", "metrics"]
    ordered_groups = group_order + sorted(k for k in grouped.keys() if k not in group_order)

    for prefix in ordered_groups:
        tags = grouped.get(prefix, [])
        for tag in sorted(tags):
            points = sorted(data.get(tag, []), key=lambda item: item[0])
            if not points:
                continue
            steps, values = zip(*points)
            plt.figure(figsize=(6, 4))
            plt.plot(steps, values)
            plt.title(tag)
            plt.xlabel("step")
            plt.ylabel("value")
            plt.tight_layout()

            filename = tag.replace("/", "__") + ".png"
            out_path = out_dir / filename
            plt.savefig(out_path)
            plt.close()
            created.append(out_path)

    return created


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot training curves from scalars.csv")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    outputs = plot_training_curves(args.run_dir, args.out_dir)
    print(f"[Plot] wrote {len(outputs)} PNGs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
