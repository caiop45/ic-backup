from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path
from typing import Dict

import torch
import config
import train_tht_tripgen
from tools import plot_training_curves
from utils.helpers import set_seed
from utils.evaluation import compute_attribute_metrics
from utils.metrics import joint_metrics, od_metrics, save_metrics_json
from tools.export_run_metrics import export_run_metrics
from tools.export_training_info import export_training_info

import recalc_downstream_tht_tripgen
import recalc_metrics_tht_tripgen_hold
import tools.build_tht_zone_embeddings as build_tht_zone_embeddings
from data_processing import tht_tripgen_loader
from data_processing.tht_tripgen_transformer import THTTripGenTransformer
from utils.serialization import THT_TRIPGEN_MAPPINGS_FILENAME, load_tht_tripgen_mappings

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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


def _recalc_hold_vs_val_real(run_dir: Path) -> None:
    """Recompute real split overlap metrics between hold and val (no synthetic sampling)."""
    try:
        _, val_df, hold_df = recalc_metrics_tht_tripgen_hold.load_and_split_tht_tripgen()
    except Exception:
        _, val_df, hold_df = tht_tripgen_loader.load_and_split_tht_tripgen()
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
    if bool(getattr(config, "EVAL_ENABLE_ATTRIBUTE_METRICS", True)):
        passenger_cols, fare_cols = _attribute_columns()
        metrics.update(
            compute_attribute_metrics(
                hold_df,
                val_df,
                passenger_cols=passenger_cols,
                fare_cols=fare_cols,
                order_key="tht_tripgen_hold_vs_val",
            )
        )

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


def _plot_curves(run_dir: Path, out_dir: Path, label: str) -> None:
    tmp_dir = out_dir / f"tmp_{label}"
    created = plot_training_curves.plot_training_curves(run_dir, tmp_dir)
    for path in created:
        dest = out_dir / f"training_curves_{label}__{path.name}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), dest)
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _resolve_export_output_path(
    *, args: argparse.Namespace, strategy_run_dir: Path
) -> tuple[Path | None, Path | None]:
    json_path = args.metrics_export_json
    csv_path = args.metrics_export_csv
    if json_path is not None and csv_path is not None:
        return json_path, csv_path
    if json_path is not None:
        return json_path, json_path.parent / "metrics_export.csv"
    if csv_path is not None:
        return csv_path.parent / "metrics_export.json", csv_path

    default_dir = args.metrics_export_dir or (strategy_run_dir / "metrics")
    return default_dir / "metrics_export.json", default_dir / "metrics_export.csv"


def _resolve_training_export_output_path(
    *, args: argparse.Namespace, strategy_run_dir: Path
) -> tuple[Path | None, Path | None]:
    json_path = args.training_export_json
    csv_path = args.training_export_csv
    if json_path is not None and csv_path is not None:
        return json_path, csv_path
    if json_path is not None:
        return json_path, json_path.parent / "training_export.csv"
    if csv_path is not None:
        return csv_path.parent / "training_export.json", csv_path
    default_dir = args.training_export_dir or (strategy_run_dir / "training")
    return default_dir / "training_export.json", default_dir / "training_export.csv"


def _export_run_metrics(
    strategy_dir: Path,
    *,
    args: argparse.Namespace,
) -> None:
    if not args.export_metrics:
        return

    output_json, output_csv = _resolve_export_output_path(
        args=args,
        strategy_run_dir=strategy_dir,
    )
    export_run_metrics(
        strategy_dir,
        output_json=output_json,
        output_csv=output_csv,
        strict=args.metrics_export_strict,
        run_label=args.metrics_export_label,
    )


def _export_training_info(
    strategy_dir: Path,
    *,
    args: argparse.Namespace,
) -> None:
    if not args.export_training_info:
        return

    output_json, output_csv = _resolve_training_export_output_path(
        args=args,
        strategy_run_dir=strategy_dir,
    )
    export_training_info(
        strategy_run_dir=strategy_dir,
        output_json=output_json,
        output_csv=output_csv,
        strict=args.training_export_strict,
        parse_logs=args.training_export_parse_logs,
        run_label=args.training_export_label,
    )


def _analyze_exports(
    *,
    strategy_dir: Path,
    args: argparse.Namespace,
    metrics_export_json: Path,
    training_export_json: Path,
) -> None:
    if not args.analyze_exports:
        return

    # Import lazily to avoid forcing matplotlib dependency unless analysis is requested.
    from tools import analyze_run_exports

    analyze_run_exports.analyze_run_exports(
        run_dir=strategy_dir,
        metrics_export_json=metrics_export_json if not args.analysis_no_metrics else None,
        training_export_json=training_export_json if not args.analysis_no_training else None,
        output_dir=args.analysis_dir,
        analyze_metrics=not args.analysis_no_metrics,
        analyze_training=not args.analysis_no_training,
        strict=args.analysis_strict,
        no_plots=args.analysis_no_plots,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run full THT-TripGen pipeline")
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
    parser.add_argument(
        "--export-metrics",
        action="store_true",
        default=True,
        help="Write consolidated metrics export files (JSON + CSV).",
    )
    parser.add_argument(
        "--no-export-metrics",
        dest="export_metrics",
        action="store_false",
        help="Do not write consolidated metrics export files.",
    )
    parser.add_argument(
        "--metrics-export-dir",
        type=Path,
        default=None,
        help="Directory for consolidated metrics export (defaults to <strategy-run-dir>/metrics).",
    )
    parser.add_argument(
        "--metrics-export-json",
        type=Path,
        default=None,
        help="Path for consolidated JSON export.",
    )
    parser.add_argument(
        "--metrics-export-csv",
        type=Path,
        default=None,
        help="Path for consolidated CSV export.",
    )
    parser.add_argument(
        "--metrics-export-strict",
        action="store_true",
        help="Fail if required metric files are missing.",
    )
    parser.add_argument(
        "--metrics-export-label",
        default=None,
        help="Optional label for the metrics export run.",
    )
    parser.add_argument(
        "--export-training-info",
        action="store_true",
        default=True,
        help="Write consolidated training export files (JSON + CSV).",
    )
    parser.add_argument(
        "--no-export-training-info",
        dest="export_training_info",
        action="store_false",
        help="Do not write consolidated training export files.",
    )
    parser.add_argument(
        "--training-export-dir",
        type=Path,
        default=None,
        help="Directory for consolidated training export (defaults to <strategy-run-dir>/training).",
    )
    parser.add_argument(
        "--training-export-json",
        type=Path,
        default=None,
        help="Path for consolidated training info JSON.",
    )
    parser.add_argument(
        "--training-export-csv",
        type=Path,
        default=None,
        help="Path for consolidated training info CSV.",
    )
    parser.add_argument(
        "--training-export-strict",
        action="store_true",
        help="Fail if required training artifacts are missing.",
    )
    parser.add_argument(
        "--training-export-parse-logs",
        action="store_true",
        dest="training_export_parse_logs",
        help="Parse epoch-level metrics from text logs.",
    )
    parser.add_argument(
        "--no-training-export-parse-logs",
        action="store_false",
        dest="training_export_parse_logs",
        help="Skip parsing epoch-level metrics from text logs.",
    )
    parser.add_argument(
        "--training-export-label",
        default=None,
        help="Optional label for the training export run.",
    )
    parser.add_argument(
        "--analyze-exports",
        action="store_true",
        default=False,
        help="Generate analysis artifacts from consolidated metric and training exports.",
    )
    parser.add_argument(
        "--no-analyze-exports",
        dest="analyze_exports",
        action="store_false",
        help="Skip analysis generation.",
    )
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=None,
        help="Directory for analyze_run_exports output (defaults to <strategy-run-dir>/analysis).",
    )
    parser.add_argument(
        "--analysis-no-plots",
        action="store_true",
        help="Skip analysis plots in analyze output.",
    )
    parser.add_argument(
        "--analysis-strict",
        action="store_true",
        help="Fail analysis if requested exports are missing or malformed.",
    )
    parser.add_argument(
        "--analysis-no-metrics",
        action="store_true",
        help="Skip metric comparison analysis while analyzing exports.",
    )
    parser.add_argument(
        "--analysis-no-training",
        action="store_true",
        help="Skip training timeline analysis while analyzing exports.",
    )
    parser.set_defaults(training_export_parse_logs=True)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    set_seed(config.GLOBAL_SEED)

    tht_tripgen_dir = _resolve_run_dir(
        str(args.tht_tripgen_run_dir) if args.tht_tripgen_run_dir is not None else None,
        "tht_tripgen",
    )

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

    _maybe_train_tht_tripgen(tht_tripgen_dir, args.force_train, device)
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
    _export_run_metrics(tht_tripgen_dir, args=args)
    _export_training_info(tht_tripgen_dir, args=args)
    metrics_output_json, _ = _resolve_export_output_path(
        args=args,
        strategy_run_dir=tht_tripgen_dir,
    )
    training_output_json, _ = _resolve_training_export_output_path(
        args=args,
        strategy_run_dir=tht_tripgen_dir,
    )
    _analyze_exports(
        strategy_dir=tht_tripgen_dir,
        args=args,
        metrics_export_json=metrics_output_json,
        training_export_json=training_output_json,
    )

    if not args.skip_plots:
        _plot_curves(tht_tripgen_dir, tht_tripgen_dir / "plots", "tht_tripgen")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
