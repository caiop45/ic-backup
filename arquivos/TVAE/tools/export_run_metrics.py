from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "2.0.0"


SectionRecord = dict[str, Any]


TOPK_METRIC_RE = re.compile(r"^topk_(?:(?:tht_tripgen_)?(?P<split>hold|val|train|test)_(?P<feature>.+)\.csv)$")
CSV_HEADERS = [
    "record_type",
    "metric_table",
    "run_section",
    "metric_key",
    "value",
    "value_type",
    "source_file",
    "status",
    "section_type",
    "split",
    "row_index",
    "category",
    "feature",
    "real_prob",
    "synth_prob",
    "abs_diff",
]


def _resolve_metric_path(run_dir: Path, candidates: Iterable[str | Path]) -> Path | None:
    for candidate in candidates:
        path = run_dir / candidate
        if path.exists():
            return path
    return None


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if isinstance(payload, Mapping):
        return dict(payload)
    return {}


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (float, int)):
        if value == float("inf") or value == float("-inf"):
            return None
        return float(value)
    try:
        converted = float(str(value))
    except (TypeError, ValueError):
        return None
    if converted == float("inf") or converted == float("-inf"):
        return None
    return converted


def _file_status(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "path": str(path)}
    try:
        stat = path.stat()
    except OSError:
        return {"exists": False, "path": str(path)}
    try:
        digest = _sha256(path)
    except OSError:
        digest = None
    return {
        "exists": True,
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_iso": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        "sha256": digest,
    }


def _sha256(path: Path, chunk_size: int = 64 * 1024) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _read_csv_summary(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = 0
        first: dict[str, str | None] | None = None
        last: dict[str, str | None] | None = None
        for row in reader:
            rows += 1
            if first is None:
                first = dict(row)
            last = dict(row)
        return {
            "rows": rows,
            "columns": list(reader.fieldnames or []),
            "first": first or {},
            "last": last or {},
        }


def _read_topk_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append({
                "category": str(row.get("category", "")),
                "real_prob": row.get("real_prob"),
                "synth_prob": row.get("synth_prob"),
                "abs_diff": row.get("abs_diff"),
            })
    return rows, list(reader.fieldnames or [])


def _to_csv_value(value: Any) -> str | int | float | bool:
    if value is None:
        return ""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if isinstance(value, float) and (value != value or abs(value) == float("inf")):
            return ""
        return value
    return str(value)


def _flatten_for_csv(
    value: Any, prefix: str = "", sep: str = "."
) -> list[tuple[str, Any]]:
    out: list[tuple[str, Any]] = []
    if isinstance(value, Mapping):
        for key, nested in value.items():
            nested_key = f"{prefix}{sep}{key}" if prefix else str(key)
            out.extend(_flatten_for_csv(nested, nested_key, sep=sep))
        return out
    if isinstance(value, list):
        for idx, item in enumerate(value):
            nested_key = f"{prefix}[{idx}]"
            out.extend(_flatten_for_csv(item, nested_key, sep=sep))
        return out
    if isinstance(
        value,
        (str, int, float, bool) or value is None,
    ):
        return [(prefix, value)]
    return []


def _load_section(
    *,
    section_name: str,
    run_dir: Path,
    candidates: Iterable[str | Path],
    strict: bool,
) -> SectionRecord:
    source = _resolve_metric_path(run_dir, candidates)
    if source is None:
        if strict:
            raise FileNotFoundError(f"[{section_name}] missing metric file in {run_dir}")
        return {
            "status": "missing",
            "file_status": _file_status(Path("missing")),
            "source_file": None,
            "section_type": "metric_json",
            "metrics": {},
        }

    try:
        metrics = _read_json(source)
        status = "ok"
        file_status = _file_status(source)
    except Exception as exc:  # pragma: no cover - guard rail for malformed files
        if strict:
            raise
        return {
            "status": "invalid",
            "file_status": _file_status(source),
            "source_file": str(source),
            "section_type": "metric_json",
            "metrics": {},
            "error": str(exc),
        }
    return {
        "status": status,
        "source_file": str(source),
        "section_type": "metric_json",
        "file_status": file_status,
        "metrics": metrics,
    }


def _section_meta(section_name: str, run_dir: Path, split: str | None = None) -> SectionRecord:
    return {
        "section_name": section_name,
        "run_dir": str(run_dir),
        "split": split,
    }


def _collect_topk_tables(run_dir: Path) -> list[dict[str, Any]]:
    metrics_dir = run_dir / "metrics"
    if not metrics_dir.exists():
        return []

    tables: list[dict[str, Any]] = []
    for path in sorted(metrics_dir.glob("topk_*.csv")):
        m = TOPK_METRIC_RE.match(path.name)
        if m is None:
            continue

        split = m.group("split")
        feature = m.group("feature")
        rows, columns = _read_topk_rows(path)
        table_section = {
            "hold": "strategy_hold",
            "val": "strategy_val",
            "train": "strategy_train",
            "test": "strategy_test",
        }.get(split, "strategy_hold")

        table_metrics: dict[str, Any] = {
            "run_section": table_section,
            "source_file": str(path),
            "split": split,
            "feature": feature,
            "section_type": "metric_topk",
            "row_count": len(rows),
            "columns": columns,
        }
        table_payload = {
            "metadata": table_metrics,
            "rows": rows,
        }
        tables.append(table_payload)

    return tables


def collect_metric_sections(
    strategy_run_dir: Path,
    baseline_run_dir: Path | None = None,
    strict: bool = False,
) -> tuple[dict[str, SectionRecord], list[dict[str, Any]]]:
    sections: dict[str, SectionRecord] = {}

    sections["baseline_hold"] = _load_section(
        section_name="baseline_hold",
        run_dir=baseline_run_dir or strategy_run_dir,
        candidates=[
            Path("metrics") / "metrics_tvae_order_1_hold.json",
            Path("metrics_tvae_order_1_hold.json"),
            Path("metrics") / "metrics_tvae_order_1.json",
            Path("metrics_tvae_order_1.json"),
            Path("metrics") / "metrics_order_1_hold.json",
            Path("metrics_order_1_hold.json"),
        ],
        strict=False if baseline_run_dir is None else strict,
    )
    sections["baseline_hold"].update(_section_meta("baseline_hold", baseline_run_dir or strategy_run_dir, "hold"))

    sections["baseline_val"] = _load_section(
        section_name="baseline_val",
        run_dir=baseline_run_dir or strategy_run_dir,
        candidates=[
            Path("metrics") / "metrics_tvae_order_1_val.json",
            Path("metrics_tvae_order_1_val.json"),
            Path("metrics") / "metrics_tvae_val.json",
            Path("metrics_tvae_val.json"),
            Path("metrics") / "metrics_order_1_val.json",
            Path("metrics_order_1_val.json"),
        ],
        strict=False,
    )
    sections["baseline_val"].update(_section_meta("baseline_val", baseline_run_dir or strategy_run_dir, "val"))

    sections["strategy_hold"] = _load_section(
        section_name="strategy_hold",
        run_dir=strategy_run_dir,
        candidates=[
            Path("metrics") / "metrics_tht_tripgen_hold.json",
            Path("metrics_tht_tripgen_hold.json"),
            Path("metrics") / "metrics_tht_tripgen.json",
            Path("metrics_tht_tripgen.json"),
        ],
        strict=False,
    )
    sections["strategy_hold"].update(_section_meta("strategy_hold", strategy_run_dir, "hold"))

    sections["strategy_val"] = _load_section(
        section_name="strategy_val",
        run_dir=strategy_run_dir,
        candidates=[
            Path("metrics") / "metrics_tht_tripgen_val.json",
            Path("metrics") / "metrics_tht_tripgen.json",
            Path("metrics_tht_tripgen_val.json"),
            Path("metrics_tht_tripgen.json"),
        ],
        strict=False,
    )
    sections["strategy_val"].update(_section_meta("strategy_val", strategy_run_dir, "val"))

    sections["strategy_hold_vs_val_real"] = _load_section(
        section_name="strategy_hold_vs_val_real",
        run_dir=strategy_run_dir,
        candidates=[
            Path("metrics") / "metrics_hold_vs_val_real.json",
            Path("metrics_hold_vs_val_real.json"),
        ],
        strict=False,
    )
    sections["strategy_hold_vs_val_real"].update(_section_meta("strategy_hold_vs_val_real", strategy_run_dir))

    sections["strategy_downstream_hold"] = _load_section(
        section_name="strategy_downstream_hold",
        run_dir=strategy_run_dir,
        candidates=[
            Path("metrics") / "downstream_tht_tripgen_hold.json",
            Path("downstream_tht_tripgen_hold.json"),
        ],
        strict=False,
    )
    sections["strategy_downstream_hold"].update(_section_meta("strategy_downstream_hold", strategy_run_dir, "hold"))

    sections["strategy_downstream_val"] = _load_section(
        section_name="strategy_downstream_val",
        run_dir=strategy_run_dir,
        candidates=[
            Path("metrics") / "downstream_tht_tripgen_val.json",
            Path("downstream_tht_tripgen_val.json"),
        ],
        strict=False,
    )
    sections["strategy_downstream_val"].update(_section_meta("strategy_downstream_val", strategy_run_dir, "val"))

    sections["strategy_privacy_hold"] = _load_section(
        section_name="strategy_privacy_hold",
        run_dir=strategy_run_dir,
        candidates=[
            Path("metrics") / "privacy_report_tht_tripgen_hold.json",
            Path("privacy_report_tht_tripgen_hold.json"),
        ],
        strict=False,
    )
    sections["strategy_privacy_hold"].update(_section_meta("strategy_privacy_hold", strategy_run_dir, "hold"))

    sections["strategy_privacy_val"] = _load_section(
        section_name="strategy_privacy_val",
        run_dir=strategy_run_dir,
        candidates=[
            Path("metrics") / "privacy_report_tht_tripgen_val.json",
            Path("privacy_report_tht_tripgen_val.json"),
        ],
        strict=False,
    )
    sections["strategy_privacy_val"].update(_section_meta("strategy_privacy_val", strategy_run_dir, "val"))

    topk_tables = _collect_topk_tables(strategy_run_dir)
    return sections, topk_tables


def _build_summary(
    sections: Mapping[str, SectionRecord],
    topk_tables: Sequence[dict[str, Any]],
    run_label: str | None = None,
) -> dict[str, Any]:
    missing_sections = [name for name, payload in sections.items() if payload.get("status") != "ok"]
    loaded_sections = [name for name, payload in sections.items() if payload.get("status") == "ok"]
    n_topk_rows = sum(int(table.get("metadata", {}).get("row_count", 0)) for table in topk_tables)
    return {
        "run_label": run_label,
        "n_sections": len(sections),
        "n_loaded": len(loaded_sections),
        "n_missing": len(missing_sections),
        "loaded_sections": loaded_sections,
        "missing_sections": missing_sections,
        "n_topk_tables": len(topk_tables),
        "n_topk_rows": n_topk_rows,
    }


def _collect_topk_rows(topk_tables: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for table in topk_tables:
        metadata = table.get("metadata", {})
        rows_payload = table.get("rows", [])
        split = metadata.get("split")
        feature = metadata.get("feature")
        run_section = metadata.get("run_section", "strategy_hold")
        source_file = metadata.get("source_file")
        section_type = metadata.get("section_type")
        for idx, row in enumerate(rows_payload):
            rows.append(
                {
                    "record_type": "topk_row",
                    "metric_table": "topk",
                    "run_section": run_section,
                    "metric_key": f"topk.{feature}.{idx}",
                    "value": "",
                    "value_type": "null",
                    "source_file": source_file,
                    "status": "ok",
                    "section_type": section_type,
                    "split": split,
                    "row_index": idx,
                    "feature": feature,
                    "category": row.get("category"),
                    "real_prob": row.get("real_prob"),
                    "synth_prob": row.get("synth_prob"),
                    "abs_diff": row.get("abs_diff"),
                }
            )
    return rows


def _collect_metric_rows(sections: Mapping[str, SectionRecord]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for section_name, section_payload in sections.items():
        if not isinstance(section_payload, Mapping):
            continue

        source_file = section_payload.get("source_file")
        status = section_payload.get("status", "unknown")
        section_type = section_payload.get("section_type")
        metrics = section_payload.get("metrics", {})
        rows.append(
            {
                "record_type": "section_status",
                "metric_table": "section",
                "run_section": section_name,
                "metric_key": "status",
                "value": status,
                "value_type": type(status).__name__,
                "source_file": source_file if source_file is not None else "",
                "status": status,
                "section_type": section_type,
            }
        )
        if not isinstance(metrics, Mapping):
            continue

        if section_payload.get("file_status", {}).get("exists") and section_payload.get("file_status", {}).get("size_bytes") is not None:
            rows.append(
                {
                    "record_type": "section_file_size_bytes",
                    "metric_table": "section",
                    "run_section": section_name,
                    "metric_key": "file.size_bytes",
                    "value": section_payload["file_status"]["size_bytes"],
                    "value_type": "int",
                    "source_file": source_file if source_file is not None else "",
                    "status": status,
                    "section_type": section_type,
                }
            )

        for metric_key, metric_value in _flatten_for_csv(metrics):
            if metric_key == "":
                continue
            rows.append(
                {
                    "record_type": "section_metric",
                    "metric_table": "metrics",
                    "run_section": section_name,
                    "metric_key": metric_key,
                    "value": metric_value,
                    "value_type": type(metric_value).__name__,
                    "source_file": source_file if source_file is not None else "",
                    "status": status,
                    "section_type": section_type,
                }
            )
    return rows


def _read_topk_summary(topk_tables: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    summary_rows: list[dict[str, Any]] = []
    for table in topk_tables:
        metadata = table.get("metadata", {})
        rows_val = int(metadata.get("row_count", 0)) if isinstance(metadata, Mapping) else 0
        summary_rows.append(
            {
                "record_type": "topk_summary",
                "metric_table": "topk_summary",
                "run_section": metadata.get("run_section"),
                "metric_key": "summary.row_count",
                "value": metadata.get("row_count", 0),
                "value_type": "int",
                "source_file": metadata.get("source_file"),
                "status": "ok" if metadata else "missing",
                "section_type": metadata.get("section_type"),
                "split": metadata.get("split"),
                "feature": metadata.get("feature"),
            }
        )
        if metadata.get("columns"):
            columns = metadata.get("columns")
            summary_rows.append(
                {
                    "record_type": "topk_summary",
                    "metric_table": "topk_summary",
                    "run_section": metadata.get("run_section"),
                    "metric_key": "summary.columns",
                    "value": "|".join(columns),
                    "value_type": "str",
                    "source_file": metadata.get("source_file"),
                    "status": "ok" if metadata else "missing",
                    "section_type": metadata.get("section_type"),
                    "split": metadata.get("split"),
                    "feature": metadata.get("feature"),
                }
            )
        if metadata:
            summary_rows.append(
                {
                    "record_type": "topk_summary",
                    "metric_table": "topk_summary",
                    "run_section": metadata.get("run_section"),
                    "metric_key": "summary.n_rows",
                    "value": rows_val,
                    "value_type": "int",
                    "source_file": metadata.get("source_file"),
                    "status": "ok",
                    "section_type": metadata.get("section_type"),
                    "split": metadata.get("split"),
                    "feature": metadata.get("feature"),
                }
            )
    return summary_rows


def build_export_bundle(
    strategy_run_dir: Path,
    baseline_run_dir: Path | None = None,
    *,
    strict: bool = False,
    run_label: str | None = None,
) -> dict[str, Any]:
    sections, topk_tables = collect_metric_sections(
        strategy_run_dir=strategy_run_dir,
        baseline_run_dir=baseline_run_dir,
        strict=strict,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pipeline_context": {
            "strategy_run_dir": str(strategy_run_dir),
            "baseline_run_dir": str(baseline_run_dir) if baseline_run_dir else None,
            "run_label": run_label,
            "strict": strict,
        },
        "summary": _build_summary(sections, topk_tables, run_label=run_label),
        "sections": sections,
        "tables": {
            "topk": [table.get("metadata", {}) for table in topk_tables],
        },
    }


def _write_json(bundle: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(bundle, fh, indent=2, ensure_ascii=False, sort_keys=True)


def _write_csv(bundle: Mapping[str, Any], path: Path) -> None:
    sections = bundle.get("sections", {})
    if not isinstance(sections, Mapping):
        sections = {}
    topk_tables = []
    for table in bundle.get("tables", {}).get("topk", []):
        if isinstance(table, Mapping):
            metadata = dict(table)
            rows_payload = []
            table_path = Path(metadata.get("source_file", ""))
            if table_path.exists():
                try:
                    rows_payload, _ = _read_topk_rows(table_path)
                except Exception:
                    rows_payload = []
            topk_tables.append(
                {
                    "metadata": metadata,
                    "rows": rows_payload,
                }
            )

    rows = _collect_metric_rows(sections)
    rows.extend(_read_topk_summary(topk_tables))
    rows.extend(_collect_topk_rows(topk_tables))

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_HEADERS)
        writer.writeheader()
        for row in rows:
            normalized = {key: _to_csv_value(row.get(key)) for key in CSV_HEADERS}
            writer.writerow(normalized)


def export_run_metrics(
    strategy_run_dir: Path,
    baseline_run_dir: Path | None = None,
    *,
    output_json: Path | None = None,
    output_csv: Path | None = None,
    strict: bool = False,
    run_label: str | None = None,
) -> dict[str, Any]:
    bundle = build_export_bundle(
        strategy_run_dir=strategy_run_dir,
        baseline_run_dir=baseline_run_dir,
        strict=strict,
        run_label=run_label,
    )

    if output_json is not None:
        _write_json(bundle, output_json)
    if output_csv is not None:
        _write_csv(bundle, output_csv)
    return bundle


def _bool_or_none(raw: str | None) -> bool | None:
    if raw is None:
        return None
    if raw.lower() in {"1", "true", "yes", "y", "on"}:
        return True
    if raw.lower() in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {raw}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Consolidate run metric files into one JSON + CSV for external tooling."
    )
    parser.add_argument(
        "--strategy-run-dir",
        required=True,
        type=Path,
        help="Path to THT-TripGen experiment directory.",
    )
    parser.add_argument(
        "--baseline-run-dir",
        type=Path,
        default=None,
        help="Optional baseline run directory for comparison context.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Path to consolidated JSON output (default: <strategy-run-dir>/metrics/metrics_export.json)",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Path to flattened CSV output (default: <strategy-run-dir>/metrics/metrics_export.csv)",
    )
    parser.add_argument(
        "--export-json",
        type=_bool_or_none,
        default=True,
        help="Write consolidated JSON output (true/false).",
    )
    parser.add_argument(
        "--export-csv",
        type=_bool_or_none,
        default=True,
        help="Write flattened CSV output (true/false).",
    )
    parser.add_argument("--strict", action="store_true", help="Fail fast on missing required files.")
    parser.add_argument("--run-label", type=str, default=None, help="Optional label for the run.")

    args = parser.parse_args()

    strategy_run_dir = args.strategy_run_dir
    if not strategy_run_dir.exists():
        raise FileNotFoundError(f"strategy run dir does not exist: {strategy_run_dir}")

    output_json = args.output_json
    output_csv = args.output_csv
    if output_json is None and output_csv is None:
        output_dir = strategy_run_dir / "metrics"
        output_json = output_dir / "metrics_export.json"
        output_csv = output_dir / "metrics_export.csv"
    elif output_json is None and output_csv is not None:
        output_json = output_csv.parent / "metrics_export.json"
    elif output_csv is None and output_json is not None:
        output_csv = output_json.parent / "metrics_export.csv"

    export_run_metrics(
        strategy_run_dir=strategy_run_dir,
        baseline_run_dir=args.baseline_run_dir,
        output_json=output_json if args.export_json is not False else None,
        output_csv=output_csv if args.export_csv is not False else None,
        strict=args.strict,
        run_label=args.run_label,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
