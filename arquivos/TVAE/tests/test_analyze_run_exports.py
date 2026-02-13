from __future__ import annotations

import csv
import json
from pathlib import Path

from tools import analyze_run_exports


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, str | float | int]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow({str(k): str(v) for k, v in row.items()})


def test_analyze_run_exports_generates_outputs(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    metrics_dir = run_dir / "metrics"
    training_dir = run_dir / "training"
    output_dir = run_dir / "analysis"

    _write_json(
        metrics_dir / "metrics_export.json",
        {
            "schema_version": "2.0.0",
            "generated_at": "2026-01-01T00:00:00Z",
            "sections": {
                "strategy_val": {
                    "status": "ok",
                    "metrics": {
                        "od_jsd": 0.03,
                        "od_chi2": 10.0,
                        "od_coverage_real": 0.50,
                        "od_coverage_synth": 0.60,
                        "od_unique_real": 100,
                        "od_unique_synth": 110,
                        "joint_jsd": 0.20,
                        "joint_chi2": 20.0,
                        "joint_coverage_real": 0.40,
                        "joint_coverage_synth": 0.50,
                        "joint_unique_real": 200,
                        "joint_unique_synth": 210,
                        "joint_mode_dropping_ratio": 0.25,
                        "joint_invalid_ratio": 0.03,
                    },
                    "source_file": str(metrics_dir / "metrics_tht_tripgen_val.json"),
                },
                "strategy_hold": {
                    "status": "ok",
                    "metrics": {
                        "od_jsd": 0.01,
                        "od_chi2": 8.0,
                        "od_coverage_real": 0.55,
                        "od_coverage_synth": 0.65,
                        "od_unique_real": 95,
                        "od_unique_synth": 105,
                        "joint_jsd": 0.12,
                        "joint_chi2": 14.0,
                        "joint_coverage_real": 0.45,
                        "joint_coverage_synth": 0.56,
                        "joint_unique_real": 190,
                        "joint_unique_synth": 197,
                        "joint_mode_dropping_ratio": 0.33,
                        "joint_invalid_ratio": 0.02,
                    },
                    "source_file": str(metrics_dir / "metrics_tht_tripgen_hold.json"),
                },
                "strategy_hold_vs_val_real": {
                    "status": "ok",
                    "metrics": {
                        "od_jsd": 0.02,
                        "od_chi2": 9.0,
                        "od_coverage_real": 0.35,
                        "od_coverage_synth": 0.54,
                        "od_unique_real": 92,
                        "od_unique_synth": 112,
                        "joint_jsd": 0.13,
                        "joint_chi2": 16.0,
                        "joint_coverage_real": 0.30,
                        "joint_coverage_synth": 0.48,
                        "joint_unique_real": 190,
                        "joint_unique_synth": 210,
                        "joint_mode_dropping_ratio": 0.32,
                        "joint_invalid_ratio": 0.04,
                    },
                    "source_file": str(metrics_dir / "metrics_hold_vs_val_real.json"),
                },
            },
        },
    )

    _write_csv(
        training_dir / "loss_tht_tripgen.csv",
        [
            {"epoch": "1", "train_nll": "1.0", "val_nll": "1.2"},
            {"epoch": "2", "train_nll": "0.9", "val_nll": "1.0"},
        ],
    )
    _write_csv(
        training_dir / "logs" / "scalars.csv",
        [
            {"step": "1", "tag": "train/metric", "value": "1.2"},
            {"step": "2", "tag": "train/metric", "value": "0.9"},
            {"step": "2", "tag": "val/metric", "value": "1.0"},
        ],
    )
    _write_json(
        training_dir / "training_export.json",
        {
            "schema_version": "2.0.0",
            "summary": {
                "n_sections": 2,
                "n_loaded": 2,
                "n_missing": 0,
                "n_loss_rows": 2,
                "n_scalar_points": 3,
                "n_log_events": 1,
            },
            "sections": {
                "strategy_tht_loss": {
                    "status": "ok",
                    "artifact_name": "loss",
                    "source_file": str(training_dir / "loss_tht_tripgen.csv"),
                },
                "strategy_tht_scalars": {
                    "status": "ok",
                    "artifact_name": "scalars",
                    "source_file": str(training_dir / "logs" / "scalars.csv"),
                },
                "strategy_tht_log": {
                    "status": "ok",
                    "artifact_name": "log",
                    "source_file": str(training_dir / "train_tht_tripgen.log"),
                    "metrics": {
                        "events": [
                            {"epoch": 1, "train_metric": 1.0, "val_metric": 0.95},
                        ]
                    },
                },
            },
            "time_series": {
                "loss": [],
                "scalars": [],
                "log": [],
            },
        },
    )

    report = analyze_run_exports.analyze_run_exports(
        run_dir=run_dir,
        metrics_export_json=metrics_dir / "metrics_export.json",
        training_export_json=training_dir / "training_export.json",
        output_dir=output_dir,
        strict=True,
        no_plots=True,
    )

    assert (output_dir / "analysis_report.json").exists()
    assert (output_dir / "analysis_summary.csv").exists()
    assert (output_dir / "analysis_metrics_long.csv").exists()
    assert (output_dir / "analysis_training_long.csv").exists()
    assert (output_dir / "analysis_tooltips.csv").exists()
    assert report["schema_version"] == analyze_run_exports.ANALYSIS_SCHEMA_VERSION
    assert len(report["metric_comparisons"]) == 23
    assert report["summary"]["generated_rows"]["metrics_long"] == 69
    assert report["summary"]["generated_rows"]["tooltips"] > 10
    assert int(report["summary"]["generated_rows"]["training_long"]) > 0


def test_analyze_run_exports_includes_passenger_fare_and_r_metrics(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    metrics_dir = run_dir / "metrics"
    output_dir = run_dir / "analysis"

    metrics_payload = {
        "strategy_val": {
            "status": "ok",
            "metrics": {
                "passenger_count_jsd": 0.02,
                "passenger_count_chi2": 0.4,
                "total_amount_log1p_w1": 0.8,
                "total_amount_log1p_ks": 0.7,
                "total_amount_log1p_quantile_mae": 0.6,
                "exact_match_rate_with_r_rounded": 0.9,
            },
            "source_file": str(metrics_dir / "metrics_tht_tripgen_val.json"),
        },
        "strategy_hold": {
            "status": "ok",
            "metrics": {
                "passenger_count_jsd": 0.01,
                "passenger_count_chi2": 0.3,
                "total_amount_log1p_w1": 0.6,
                "total_amount_log1p_ks": 0.5,
                "total_amount_log1p_quantile_mae": 0.4,
                "exact_match_rate_with_r_rounded": 0.91,
            },
            "source_file": str(metrics_dir / "metrics_tht_tripgen_hold.json"),
        },
        "strategy_hold_vs_val_real": {
            "status": "ok",
            "metrics": {
                "od_jsd": 0.01,
                "od_chi2": 5.0,
            },
            "source_file": str(metrics_dir / "metrics_hold_vs_val_real.json"),
        },
    }
    _write_json(metrics_dir / "metrics_export.json", {
        "sections": metrics_payload,
    })
    _write_json(
        run_dir / "training" / "training_export.json",
        {"summary": {}, "sections": {}},
    )

    report = analyze_run_exports.analyze_run_exports(
        run_dir=run_dir,
        metrics_export_json=metrics_dir / "metrics_export.json",
        training_export_json=run_dir / "training" / "training_export.json",
        output_dir=output_dir,
        strict=True,
        no_plots=True,
        analyze_training=False,
    )

    comparison_map = {
        row["metric_key"]: row for row in report["metric_comparisons"] if row["record_type"] == "comparison_summary"
    }
    assert comparison_map["passenger_count_jsd"]["synth_to_hold"] == 0.01
    assert comparison_map["total_amount_log1p_w1"]["synth_to_val"] == 0.8
    assert comparison_map["exact_match_rate_with_r_rounded"]["synth_to_hold"] == 0.91
