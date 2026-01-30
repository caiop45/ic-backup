from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path
from typing import Dict

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
import train_tht_tripgen
import train_tvae
from tools import compare_models, plot_training_curves
from utils.helpers import set_seed

import recalc_metrics_hold
import recalc_metrics_tht_tripgen_hold
import tools.build_tht_zone_embeddings as build_tht_zone_embeddings


def _timestamp_tag(prefix: str) -> str:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{stamp}"


def _resolve_run_dir(path_arg: str | None, default_prefix: str) -> Path:
    if path_arg is None:
        base = Path(config.SAVE_DATA_DIR)
        return base / _timestamp_tag(default_prefix)
    return Path(path_arg)


def _override_config(overrides: Dict[str, object]) -> Dict[str, object]:
    backup: Dict[str, object] = {}
    for key, value in overrides.items():
        if hasattr(config, key):
            backup[key] = getattr(config, key)
            setattr(config, key, value)
    return backup


def _restore_config(backup: Dict[str, object]) -> None:
    for key, value in backup.items():
        setattr(config, key, value)


def _call_main(module, argv: list[str]) -> None:
    original = sys.argv
    try:
        sys.argv = argv
        module.main()
    finally:
        sys.argv = original


def _ensure_embeddings(build: bool, device: str | None) -> None:
    cache_dir = Path(config.THT_TOPOLOGY_CACHE_DIR)
    comb_path = cache_dir / "Ecomb.pt"
    func_path = cache_dir / "Efunc.pt"
    if comb_path.exists() or func_path.exists():
        print(f"[Pipeline] embeddings found in {cache_dir}, reuse")
        return
    if not build:
        raise RuntimeError(
            "Zone embeddings missing. Re-run with --build-embeddings or create Efunc.pt/Ecomb.pt."
        )
    args = ["build_tht_zone_embeddings.py", "--device", device or "cpu"]
    _call_main(build_tht_zone_embeddings, args)


def _maybe_train_tvae(run_dir: Path, force_train: bool) -> None:
    checkpoint = run_dir / f"tvae_{config.FIXED_ORDER_KEY}.pt"
    if checkpoint.exists() and not force_train:
        print(f"[Pipeline] baseline reuse: {run_dir}")
        return
    print(f"[Pipeline] training baseline TVAE -> {run_dir}")
    backup = _override_config(
        {
            "SAVE_DATA_DIR": str(run_dir.parent),
            "EXPERIMENTS": {run_dir.name: {}},
        }
    )
    try:
        train_tvae.train_all_orders()
    finally:
        _restore_config(backup)


def _maybe_train_tht_tripgen(run_dir: Path, force_train: bool, device: torch.device) -> None:
    checkpoint = run_dir / "tht_tripgen.pt"
    if checkpoint.exists() and not force_train:
        print(f"[Pipeline] THT-TripGen reuse: {run_dir}")
        return
    print(f"[Pipeline] training THT-TripGen -> {run_dir}")
    backup = _override_config({"SAVE_DATA_DIR": str(run_dir.parent)})
    try:
        train_tht_tripgen.train_tht_tripgen(run_tag=run_dir.name, device=device)
    finally:
        _restore_config(backup)


def _recalc_baseline(run_dir: Path, force_sample: bool) -> None:
    args = [
        "recalc_metrics_hold.py",
        "--run-dir",
        str(run_dir),
        "--order-key",
        config.FIXED_ORDER_KEY,
        "--no-plots",
    ]
    if force_sample:
        args.append("--force-sample")
    _call_main(recalc_metrics_hold, args)


def _recalc_tht_tripgen(run_dir: Path, force_sample: bool) -> None:
    args = [
        "recalc_metrics_tht_tripgen_hold.py",
        "--run-dir",
        str(run_dir),
        "--no-plots",
    ]
    if force_sample:
        args.append("--force-sample")
    _call_main(recalc_metrics_tht_tripgen_hold, args)


def _compare_runs(baseline_dir: Path, tht_tripgen_dir: Path, out_dir: Path) -> None:
    args = [
        "compare_models.py",
        "--baseline-run",
        str(baseline_dir),
        "--tht-tripgen-run",
        str(tht_tripgen_dir),
        "--out-dir",
        str(out_dir),
    ]
    _call_main(compare_models, args)


def _plot_curves(run_dir: Path, out_dir: Path, label: str) -> None:
    tmp_dir = out_dir / f"tmp_{label}"
    created = plot_training_curves.plot_training_curves(run_dir, tmp_dir)
    for path in created:
        dest = out_dir / f"training_curves_{label}__{path.name}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), dest)
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run full TVAE + THT-TripGen pipeline")
    parser.add_argument("--baseline-run-dir", type=Path, default=None)
    parser.add_argument(
        "--tht-tripgen-run-dir",
        "--strategy-run-dir",
        dest="tht_tripgen_run_dir",
        type=Path,
        default=None,
        help="THT-TripGen run directory (alias: --strategy-run-dir)",
    )
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--force-sample", action="store_true")
    parser.add_argument("--build-embeddings", action="store_true")
    parser.add_argument("--comparison-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    set_seed(config.GLOBAL_SEED)

    baseline_dir = _resolve_run_dir(
        str(args.baseline_run_dir) if args.baseline_run_dir is not None else None,
        "baseline",
    )
    tht_tripgen_dir = _resolve_run_dir(
        str(args.tht_tripgen_run_dir) if args.tht_tripgen_run_dir is not None else None,
        "tht_tripgen",
    )

    comparison_dir = (
        args.comparison_dir
        if args.comparison_dir is not None
        else Path(config.OUTPUT_BASE_DIR) / "comparison"
    )
    comparison_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = comparison_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    device = (
        torch.device(args.device)
        if args.device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    _ensure_embeddings(args.build_embeddings, str(device))

    _maybe_train_tvae(baseline_dir, args.force_train)
    _maybe_train_tht_tripgen(tht_tripgen_dir, args.force_train, device)

    _recalc_baseline(baseline_dir, args.force_sample)
    _recalc_tht_tripgen(tht_tripgen_dir, args.force_sample)

    _compare_runs(baseline_dir, tht_tripgen_dir, comparison_dir)

    _plot_curves(baseline_dir, plots_dir, "baseline")
    _plot_curves(tht_tripgen_dir, plots_dir, "tht_tripgen")

    print(f"[Pipeline] comparison outputs -> {comparison_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
