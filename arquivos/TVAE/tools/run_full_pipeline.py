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
from utils.metrics import joint_metrics, od_metrics, save_metrics_json

import recalc_downstream_tht_tripgen
import recalc_metrics_hold
import recalc_metrics_tht_tripgen_hold
import tools.build_tht_zone_embeddings as build_tht_zone_embeddings
from data_processing.tht_tripgen_loader import load_and_split_tht_tripgen
from data_processing.tht_tripgen_transformer import THTTripGenTransformer
from utils.serialization import THT_TRIPGEN_MAPPINGS_FILENAME, load_tht_tripgen_mappings


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


def _ensure_embeddings(build: bool, device: str | None, skip_check: bool) -> None:
    if skip_check:
        print("[Pipeline] skipping embeddings check/build (assumes Efunc/Ecomb already exist)")
        return
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


def _filter_known_real(df, transformer: THTTripGenTransformer):
    idx = transformer.transform(df, drop_unknown=False)
    mask = idx.notna().all(axis=1) & idx["r"].notna()
    return df.loc[mask].reset_index(drop=True)


def _recalc_hold_vs_val_real(run_dir: Path) -> None:
    """Recompute real split overlap metrics between hold and val (no synthetic sampling)."""
    _, val_df, hold_df = load_and_split_tht_tripgen()
    mappings_path = run_dir / THT_TRIPGEN_MAPPINGS_FILENAME
    if not mappings_path.exists():
        print(f"[Pipeline] skipping hold-vs-val real split metrics: missing mappings {mappings_path}")
        return

    transformer = load_tht_tripgen_mappings(mappings_path)
    val_df = _filter_known_real(val_df, transformer)
    hold_df = _filter_known_real(hold_df, transformer)

    if len(val_df) == 0 or len(hold_df) == 0:
        print("[Pipeline] skipping hold-vs-val real split metrics: empty split after filtering")
        return

    metrics = {}
    metrics.update(od_metrics(hold_df, val_df))
    metrics.update(joint_metrics(hold_df, val_df))

    out_dir = run_dir / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "metrics_hold_vs_val_real.json"
    save_metrics_json(metrics, out_path)
    print(f"[Pipeline] saved hold-vs-val real split metrics -> {out_path}")


def _recalc_downstream_tht_tripgen(
    run_dir: Path,
    *,
    split: str,
    train_rows: int,
    test_rows: int,
    model: str,
    seed: int | None,
) -> None:
    """Run downstream fare evaluation for a THT-TripGen run."""
    args = [
        "recalc_downstream_tht_tripgen.py",
        "--run-dir",
        str(run_dir),
        "--split",
        split,
        "--train-rows",
        str(train_rows),
        "--test-rows",
        str(test_rows),
        "--model",
        model,
    ]
    if seed is not None:
        args += ["--seed", str(seed)]
    _call_main(recalc_downstream_tht_tripgen, args)


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
    parser.add_argument(
        "--tht-only",
        action="store_true",
        help="Run only THT-TripGen (skips baseline + comparison).",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip baseline TVAE training + eval.",
    )
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--force-sample", action="store_true")
    parser.add_argument(
        "--build-embeddings",
        action="store_true",
        help="Build embeddings if missing (default behavior).",
    )
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Skip embeddings build/check (assumes Efunc/Ecomb already exist).",
    )
    parser.add_argument(
        "--skip-comparison",
        action="store_true",
        help="Skip comparison tables/plots (useful for THT-only runs).",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip training curve plots.",
    )
    parser.add_argument(
        "--skip-downstream",
        action="store_true",
        help="Skip downstream fare evaluation for THT-TripGen.",
    )
    parser.add_argument(
        "--downstream-split",
        type=str,
        choices=["val", "hold"],
        default="hold",
        help="Split to evaluate downstream fare metrics.",
    )
    parser.add_argument(
        "--downstream-train-rows",
        type=int,
        default=40000,
        help="Max rows for downstream training set.",
    )
    parser.add_argument(
        "--downstream-test-rows",
        type=int,
        default=20000,
        help="Max rows for downstream evaluation sets.",
    )
    parser.add_argument(
        "--downstream-model",
        type=str,
        choices=["gbr", "hgbr", "linear"],
        default="gbr",
        help="Model type for downstream fare prediction.",
    )
    parser.add_argument(
        "--downstream-seed",
        type=int,
        default=None,
        help="Seed for downstream sampling/training.",
    )
    parser.add_argument("--comparison-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    if args.tht_only:
        args.skip_baseline = True
        args.skip_comparison = True

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

    build_embeddings = True
    if args.skip_embeddings:
        build_embeddings = False
    elif args.build_embeddings:
        build_embeddings = True

    _ensure_embeddings(build_embeddings, str(device), args.skip_embeddings)

    if not args.skip_baseline:
        _maybe_train_tvae(baseline_dir, args.force_train)
    _maybe_train_tht_tripgen(tht_tripgen_dir, args.force_train, device)

    if not args.skip_baseline:
        _recalc_baseline(baseline_dir, args.force_sample)
    _recalc_tht_tripgen(tht_tripgen_dir, args.force_sample)
    _recalc_hold_vs_val_real(tht_tripgen_dir)
    if not args.skip_downstream:
        _recalc_downstream_tht_tripgen(
            tht_tripgen_dir,
            split=args.downstream_split,
            train_rows=args.downstream_train_rows,
            test_rows=args.downstream_test_rows,
            model=args.downstream_model,
            seed=args.downstream_seed,
        )

    if args.skip_baseline and not args.skip_comparison:
        print("[Pipeline] baseline skipped; comparison disabled")
        args.skip_comparison = True

    if not args.skip_comparison:
        _compare_runs(baseline_dir, tht_tripgen_dir, comparison_dir)

    if not args.skip_plots:
        if not args.skip_baseline:
            _plot_curves(baseline_dir, plots_dir, "baseline")
        _plot_curves(tht_tripgen_dir, plots_dir, "tht_tripgen")

    if not args.skip_comparison:
        print(f"[Pipeline] comparison outputs -> {comparison_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
