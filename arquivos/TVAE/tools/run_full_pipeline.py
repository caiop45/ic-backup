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
import train_strategy_a
import train_tvae
from tools import compare_models, plot_training_curves
from utils.helpers import set_seed

import recalc_metrics_hold
import recalc_metrics_strategy_a_hold
import tools.build_zone_embeddings as build_zone_embeddings


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
    cache_dir = Path(config.SA_TOPOLOGY_CACHE_DIR)
    comb_path = cache_dir / "Ecomb.pt"
    func_path = cache_dir / "Efunc.pt"
    if comb_path.exists() or func_path.exists():
        print(f"[Pipeline] embeddings found in {cache_dir}, reuse")
        return
    if not build:
        raise RuntimeError(
            "Zone embeddings missing. Re-run with --build-embeddings or create Efunc.pt/Ecomb.pt."
        )
    args = ["build_zone_embeddings.py", "--device", device or "cpu"]
    _call_main(build_zone_embeddings, args)


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


def _maybe_train_strategy(run_dir: Path, force_train: bool, device: torch.device) -> None:
    checkpoint = run_dir / "strategy_a.pt"
    if checkpoint.exists() and not force_train:
        print(f"[Pipeline] strategy reuse: {run_dir}")
        return
    print(f"[Pipeline] training Strategy A -> {run_dir}")
    backup = _override_config({"SAVE_DATA_DIR": str(run_dir.parent)})
    try:
        train_strategy_a.train_strategy_a(run_tag=run_dir.name, device=device)
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


def _recalc_strategy(run_dir: Path, force_sample: bool) -> None:
    args = [
        "recalc_metrics_strategy_a_hold.py",
        "--run-dir",
        str(run_dir),
        "--no-plots",
    ]
    if force_sample:
        args.append("--force-sample")
    _call_main(recalc_metrics_strategy_a_hold, args)


def _compare_runs(baseline_dir: Path, strategy_dir: Path, out_dir: Path) -> None:
    args = [
        "compare_models.py",
        "--baseline-run",
        str(baseline_dir),
        "--strategy-a-run",
        str(strategy_dir),
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
    parser = argparse.ArgumentParser(description="Run full TVAE + Strategy A pipeline")
    parser.add_argument("--baseline-run-dir", type=Path, default=None)
    parser.add_argument("--strategy-run-dir", type=Path, default=None)
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
    strategy_dir = _resolve_run_dir(
        str(args.strategy_run_dir) if args.strategy_run_dir is not None else None,
        "strategy_a",
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
    _maybe_train_strategy(strategy_dir, args.force_train, device)

    _recalc_baseline(baseline_dir, args.force_sample)
    _recalc_strategy(strategy_dir, args.force_sample)

    _compare_runs(baseline_dir, strategy_dir, comparison_dir)

    _plot_curves(baseline_dir, plots_dir, "baseline")
    _plot_curves(strategy_dir, plots_dir, "strategy_a")

    print(f"[Pipeline] comparison outputs -> {comparison_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
