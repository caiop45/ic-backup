import csv
import json
from pathlib import Path

from tvae.tools import export_run_metrics


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def test_collect_metric_sections_with_fallback_paths(tmp_path: Path) -> None:
    strategy_dir = tmp_path / "strategy"

    _write_json(strategy_dir / "metrics" / "metrics_tht_tripgen_hold.json", {"od_jsd": 0.02})
    _write_json(strategy_dir / "metrics" / "metrics_hold_vs_val_real.json", {"od_jsd": 0.009})
    # write fallback candidate path (without /metrics/) for downstream + privacy hold
    _write_json(
        strategy_dir / "downstream_tht_tripgen_hold.json",
        {
            "skipped_fare": False,
            "dwn_fare_tr_tr": {"r2": 0.1},
        },
    )
    _write_json(
        strategy_dir / "privacy_report_tht_tripgen_hold.json",
        {"exact_match_rate_discrete": 0.95},
    )

    bundle = export_run_metrics.build_export_bundle(
        strategy_run_dir=strategy_dir,
        run_label="smoke",
    )

    assert bundle["schema_version"] == export_run_metrics.SCHEMA_VERSION
    assert bundle["pipeline_context"]["strategy_run_dir"] == str(strategy_dir)

    sections = bundle["sections"]
    assert sections["strategy_hold"]["status"] == "ok"
    assert sections["strategy_hold"]["source_file"].endswith(
        "metrics/metrics_tht_tripgen_hold.json"
    )
    assert sections["strategy_downstream_hold"]["status"] == "ok"
    assert sections["strategy_downstream_hold"]["source_file"].endswith(
        "downstream_tht_tripgen_hold.json"
    )
    assert sections["strategy_privacy_hold"]["status"] == "ok"
    assert sections["strategy_privacy_hold"]["source_file"].endswith(
        "privacy_report_tht_tripgen_hold.json"
    )
    assert sections["strategy_hold"]["metrics"]["od_jsd"] == 0.02


def test_export_outputs_json_and_csv(tmp_path: Path) -> None:
    strategy_dir = tmp_path / "strategy"

    _write_json(
        strategy_dir / "metrics" / "metrics_tht_tripgen_val.json",
        {"od_jsd": 0.03, "od_chi2": 1.0, "eval_split": "val"},
    )

    out_json = tmp_path / "out" / "bundle.json"
    out_csv = tmp_path / "out" / "bundle.csv"
    bundle = export_run_metrics.export_run_metrics(
        strategy_run_dir=strategy_dir,
        output_json=out_json,
        output_csv=out_csv,
        run_label="smoke",
    )

    assert out_json.exists()
    assert out_csv.exists()

    parsed_bundle = json.loads(out_json.read_text(encoding="utf-8"))
    assert parsed_bundle["pipeline_context"]["strategy_run_dir"] == str(strategy_dir)
    assert parsed_bundle["sections"]["strategy_val"]["metrics"]["od_jsd"] == 0.03

    rows = list(csv.DictReader(out_csv.read_text(encoding="utf-8").splitlines()))
    keys = {(row["run_section"], row["metric_key"]): row["value"] for row in rows}
    assert ("strategy_val", "od_jsd") in keys
    assert keys[("strategy_val", "od_jsd")] == "0.03"
    assert bundle["sections"]["strategy_val"]["status"] == "ok"
