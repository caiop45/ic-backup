from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch

from tvae import config
from tvae.utils.serialization import (
    THT_TRIPGEN_CHECKPOINT_FILENAME,
    THT_TRIPGEN_LOSS_FILENAME,
    THT_TRIPGEN_MAPPINGS_FILENAME,
)


SCHEMA_VERSION = "3.0.0"
SectionRecord = dict[str, Any]

CSV_FIELDS = [
    "record_type",
    "metric_table",
    "run_section",
    "artifact_name",
    "metric_key",
    "value",
    "value_type",
    "source_file",
    "status",
    "section_type",
    "epoch",
    "step",
    "tag",
    "metric_name",
]

_FLOAT_RE = r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?"
_THT_LOG_RE = re.compile(
    rf"^\[Epoch\s+(?P<epoch>{_FLOAT_RE})\].*train_nll=(?P<train>{_FLOAT_RE}).*val_nll=(?P<val>{_FLOAT_RE})"
)


def _resolve_file(run_dir: Path, candidates: Iterable[str | Path]) -> Path | None:
    for candidate in candidates:
        path = run_dir / candidate
        if path.exists():
            return path
    return None


def _resolve_log_path(
    run_dir: Path,
    log_name: str,
) -> Path | None:
    run_name = run_dir.name
    candidate_paths = [
        Path(config.LOG_DIR) / run_name / log_name,
        run_dir / "logs" / log_name,
    ]
    for candidate in candidate_paths:
        if candidate.exists():
            return candidate
    return None


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if isinstance(payload, Mapping):
        return dict(payload)
    return {}


def _safe_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (float, int)):
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    try:
        parsed = float(str(value))
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _sha256(path: Path, chunk_size: int = 64 * 1024) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _file_status(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "path": str(path), "size_bytes": 0, "mtime_iso": None, "sha256": None}
    stat = path.stat()
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


def _snapshot_config(keys: Iterable[str]) -> dict[str, Any]:
    return {key: getattr(config, key) for key in keys if hasattr(config, key)}


def _read_checkpoint(path: Path) -> tuple[dict[str, Any], str | None]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        return {"error": "checkpoint payload is not a dict"}, "invalid-checkpoint-payload"

    meta = payload.get("meta")
    metrics = payload.get("metrics")
    return {
        "epoch": payload.get("epoch"),
        "payload_keys": sorted(str(k) for k in payload.keys()),
        "has_model_state": "model_state" in payload,
        "has_optimizer_state": bool(payload.get("optimizer_state") is not None),
        "meta": dict(meta) if isinstance(meta, Mapping) else {},
        "metrics": metrics if isinstance(metrics, Mapping) else {},
        "state_keys": sorted(str(k) for k in payload.keys()),
    }, None


def _read_tht_mapping_summary(path: Path) -> tuple[dict[str, Any], str | None]:
    payload = _read_json(path)
    zone_categories = payload.get("zone_categories", {})
    time_categories = payload.get("time_categories", [])
    conditional_categories = payload.get("conditional_categories", {})
    return {
        "num_zones": len(zone_categories) if isinstance(zone_categories, Mapping) else 0,
        "num_time_bins": len(time_categories) if isinstance(time_categories, list) else int(time_categories or 0),
        "zone_categories": {
            str(k): len(v) if isinstance(v, list) else len(list(v))
            for k, v in zone_categories.items()
            if isinstance(zone_categories, Mapping)
        },
        "conditional_categories": {
            str(k): len(v) if isinstance(v, list) else 0
            for k, v in (
                conditional_categories.items()
                if isinstance(conditional_categories, Mapping)
                else {}
            )
        },
        "use_passenger_count": bool(payload.get("use_passenger_count", False)),
        "use_total_amount": bool(payload.get("use_total_amount", False)),
    }, None


def _read_csv_summary(path: Path) -> tuple[dict[str, Any], str | None]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            return {"rows": 0, "columns": []}, None

        rows = 0
        first: dict[str, float | str | int | None] | None = None
        last: dict[str, float | str | int | None] | None = None
        for row in reader:
            rows += 1
            if first is None:
                first = {k: _safe_float(v) for k, v in row.items()}
            last = {k: _safe_float(v) for k, v in row.items()}
        return {
            "rows": rows,
            "columns": reader.fieldnames,
            "first": first or {},
            "last": last or {},
        }, None


def _read_scalars_summary(path: Path) -> tuple[dict[str, Any], str | None]:
    tag_counts: dict[str, int] = {}
    first_step = None
    last_step = None
    rows = 0
    first = None
    last = None
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows += 1
            if first is None:
                first = row
            last = row
            tag = str(row.get("tag", "")).strip()
            if tag:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
            step = _safe_float(row.get("step"))
            if step is not None:
                step_int = int(step)
                if first_step is None:
                    first_step = step_int
                last_step = step_int

    return {
        "rows": rows,
        "n_tags": len(tag_counts),
        "first_step": first_step,
        "last_step": last_step,
        "tags": sorted(tag_counts.keys()),
        "tag_counts": tag_counts,
        "first": {k: _safe_float(v) if k in ("step", "value") else v for k, v in first.items()} if isinstance(first, dict) else {},
        "last": {k: _safe_float(v) if k in ("step", "value") else v for k, v in last.items()} if isinstance(last, dict) else {},
    }, None


def _read_log_events(path: Path, model: str) -> tuple[dict[str, Any], str | None]:
    if model == "tht":
        regex = _THT_LOG_RE
    else:
        return {"error": f"unsupported log model: {model}"}, "invalid-log-model"

    events: list[dict[str, Any]] = []
    first_checkpoint_line = None
    has_early_stop = False
    lines_read = 0
    last_lines: list[str] = []

    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            lines_read += 1
            line = raw.rstrip("\n")
            if len(last_lines) < 40:
                last_lines.append(line)
            else:
                last_lines = last_lines[-39:] + [line]

            low = line.lower()
            if "early stopping" in low:
                has_early_stop = True
            if first_checkpoint_line is None and "saved checkpoint" in low:
                first_checkpoint_line = line

            match = regex.match(line)
            if not match:
                continue
            epoch = _safe_float(match.group("epoch"))
            if epoch is None:
                continue
            events.append(
                {
                    "epoch": int(epoch),
                    "train_metric": _safe_float(match.group("train")),
                    "val_metric": _safe_float(match.group("val")),
                }
            )

    return {
        "lines": lines_read,
        "event_count": len(events),
        "events": events,
        "has_early_stopping": has_early_stop,
        "first_checkpoint_line": first_checkpoint_line,
        "last_lines": last_lines,
    }, None


def _add_artifact_section(
    name: str,
    path: Path | None,
    loader,
    *,
    required_file: bool,
    strict: bool,
    required: bool,
    section_type: str,
) -> SectionRecord:
    status = "missing"
    source_file: str | None = None
    metrics: dict[str, Any] = {}
    if path is not None:
        source_file = str(path)
        try:
            payload, err = loader(path)
            if err is not None:
                status = "invalid"
                metrics = {"error": err}
            else:
                status = "ok"
                metrics = payload
                file_meta = _file_status(path)
                metrics["file_status"] = file_meta
        except Exception as exc:
            status = "invalid"
            metrics = {"error": str(exc), "file_status": _file_status(path)}
    elif required_file and required and strict:
        raise FileNotFoundError(f"missing required artifact for {section_type}: {name}")

    return {
        "status": status,
        "source_file": source_file,
        "artifact_name": name,
        "section_type": section_type,
        "file_status": _file_status(path) if path is not None else {"exists": False, "path": str(path) if path is not None else None},
        "metrics": metrics,
    }


def _build_tht_sections(
    run_dir: Path,
    parse_logs: bool,
    *,
    strict: bool,
    required: bool,
) -> dict[str, SectionRecord]:
    sections: dict[str, SectionRecord] = {}
    checkpoint_candidates = [Path(THT_TRIPGEN_CHECKPOINT_FILENAME)]
    mappings_candidates = [Path(THT_TRIPGEN_MAPPINGS_FILENAME)]
    loss_candidates = [Path(THT_TRIPGEN_LOSS_FILENAME)]
    log_path = _resolve_log_path(run_dir, "train_tht_tripgen.log")
    scalars_path = _resolve_file(run_dir, [Path("logs") / "scalars.csv"])
    checkpoint_path = _resolve_file(run_dir, checkpoint_candidates)
    mappings_path = _resolve_file(run_dir, mappings_candidates)
    loss_path = _resolve_file(run_dir, loss_candidates)

    has_any_artifact = any(
        p is not None
        for p in (checkpoint_path, mappings_path, loss_path, scalars_path, log_path)
    )
    if not has_any_artifact and not required:
        return {}

    sections["run"] = {
        "status": "ok",
        "source_file": str(run_dir),
        "artifact_name": "run",
        "section_type": "run",
        "file_status": _file_status(run_dir),
        "metrics": {
            "run_dir": str(run_dir),
            "run_name": run_dir.name,
            "model_type": "tht_tripgen",
            "config": _snapshot_config(
                [
                    "THT_EPOCHS",
                    "THT_BATCH_SIZE",
                    "THT_LR",
                    "THT_WEIGHT_DECAY",
                    "THT_GRAD_CLIP_NORM",
                    "PATIENCE",
                    "MIN_DELTA",
                    "THT_USE_PASSENGER_COUNT",
                    "THT_USE_TOTAL_AMOUNT",
                    "THT_COND_EMB_DIM",
                    "THT_TIME_EMB_DIM",
                    "THT_ORIGIN_EMB_DIM",
                    "THT_MODEL_HIDDEN",
                    "THT_MODEL_LAYERS",
                    "THT_MODEL_DROPOUT",
                ]
            ),
        },
    }

    sections["checkpoint"] = _add_artifact_section(
        "checkpoint",
        checkpoint_path,
        _read_checkpoint,
        required_file=True,
        strict=strict,
        required=required,
        section_type="checkpoint",
    )
    sections["mappings"] = _add_artifact_section(
        "mappings",
        mappings_path,
        _read_tht_mapping_summary,
        required_file=True,
        strict=strict,
        required=required,
        section_type="mappings",
    )
    sections["loss"] = _add_artifact_section(
        "loss",
        loss_path,
        _read_csv_summary,
        required_file=True,
        strict=strict,
        required=required,
        section_type="loss",
    )
    sections["scalars"] = _add_artifact_section(
        "scalars",
        scalars_path,
        _read_scalars_summary,
        required_file=False,
        strict=strict,
        required=required,
        section_type="scalars",
    )

    if log_path is not None:
        log_metrics, log_err = (_read_log_events(log_path, "tht") if parse_logs else ({}, None))
        if log_err is not None:
            sections["log"] = {
                "status": "invalid",
                "source_file": str(log_path),
                "artifact_name": "log",
                "section_type": "log",
                "file_status": _file_status(log_path),
                "metrics": {"error": log_err},
            }
        else:
            log_metrics["file_status"] = _file_status(log_path)
            sections["log"] = {
                "status": "ok",
                "source_file": str(log_path),
                "artifact_name": "log",
                "section_type": "log",
                "file_status": _file_status(log_path),
                "metrics": log_metrics,
            }
    elif strict and required:
        raise FileNotFoundError(f"missing required THT log for {run_dir}")
    else:
        sections["log"] = {
            "status": "missing",
            "source_file": None,
            "artifact_name": "log",
            "section_type": "log",
            "file_status": {"exists": False, "path": None},
            "metrics": {},
        }

    sections["artifacts"] = {
        "status": "ok",
        "source_file": None,
        "artifact_name": "artifacts",
        "section_type": "artifacts",
        "file_status": {"exists": True, "path": str(run_dir)},
        "metrics": {
            "run_dir": str(run_dir),
            "artifacts": [
                _status_record(path)
                for path in [checkpoint_path, mappings_path, loss_path, scalars_path, log_path]
            ],
        },
    }
    return sections


def _status_record(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"exists": False, "path": None}
    if path.exists():
        return {"exists": True, "path": str(path), **_file_status(path)}
    return {"exists": False, "path": str(path)}


def _prefix_sections(prefix: str, sections: Mapping[str, SectionRecord]) -> dict[str, SectionRecord]:
    return {f"{prefix}_{name}": payload for name, payload in sections.items()}


def collect_training_sections(
    strategy_run_dir: Path,
    *,
    parse_logs: bool = True,
    strict: bool = False,
) -> dict[str, SectionRecord]:
    sections: dict[str, SectionRecord] = {}
    sections.update(
        _prefix_sections(
            "strategy_tht",
            _build_tht_sections(strategy_run_dir, parse_logs, strict=strict, required=True),
        )
    )

    return sections


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append({str(k): "" if v is None else str(v) for k, v in row.items()})
    return rows


def _collect_loss_rows(sections: Mapping[str, SectionRecord]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for section_name, section_payload in sections.items():
        if section_payload.get("artifact_name") != "loss":
            continue
        if section_payload.get("status") != "ok":
            continue
        source_file = section_payload.get("source_file")
        if source_file is None:
            continue
        try:
            csv_rows = _read_csv_rows(Path(source_file))
        except Exception:
            continue
        for row_idx, row in enumerate(csv_rows):
            epoch = row.get("epoch")
            for metric_name, metric_value in row.items():
                if metric_name == "epoch":
                    continue
                rows.append(
                    {
                        "record_type": "loss_point",
                        "metric_table": "loss",
                        "run_section": section_name,
                        "artifact_name": "loss",
                        "metric_key": f"loss[{row_idx}].{metric_name}",
                        "value": _safe_float(metric_value),
                        "value_type": type(_safe_float(metric_value)).__name__,
                        "source_file": source_file,
                        "status": section_payload.get("status", "unknown"),
                        "section_type": section_payload.get("section_type"),
                        "step": row_idx,
                        "epoch": epoch,
                        "metric_name": metric_name,
                    }
                )
    return rows


def _collect_scalars_rows(sections: Mapping[str, SectionRecord]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for section_name, section_payload in sections.items():
        if section_payload.get("artifact_name") != "scalars":
            continue
        if section_payload.get("status") != "ok":
            continue
        source_file = section_payload.get("source_file")
        if source_file is None:
            continue
        try:
            csv_rows = _read_csv_rows(Path(source_file))
        except Exception:
            continue
        for row in csv_rows:
            step = row.get("step", "")
            tag = row.get("tag", "")
            value = _safe_float(row.get("value"))
            rows.append(
                {
                    "record_type": "scalar_point",
                    "metric_table": "scalars",
                    "run_section": section_name,
                    "artifact_name": "scalars",
                    "metric_key": f"scalar[{tag}]",
                    "value": value,
                    "value_type": type(value).__name__,
                    "source_file": source_file,
                    "status": section_payload.get("status", "unknown"),
                    "section_type": section_payload.get("section_type"),
                    "step": step,
                    "tag": tag,
                    "metric_name": tag,
                }
            )
    return rows


def _collect_log_rows(sections: Mapping[str, SectionRecord]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for section_name, section_payload in sections.items():
        if section_payload.get("artifact_name") != "log":
            continue
        events = section_payload.get("metrics", {}).get("events")
        if not isinstance(events, list):
            continue
        for event in events:
            if not isinstance(event, Mapping):
                continue
            epoch = event.get("epoch")
            rows.append(
                {
                    "record_type": "log_event_train",
                    "metric_table": "log",
                    "run_section": section_name,
                    "artifact_name": "log",
                    "metric_key": "log.train_metric",
                    "value": event.get("train_metric"),
                    "value_type": type(event.get("train_metric")).__name__,
                    "source_file": section_payload.get("source_file", ""),
                    "status": section_payload.get("status", "unknown"),
                    "section_type": section_payload.get("section_type"),
                    "epoch": epoch,
                    "metric_name": "train_metric",
                }
            )
            rows.append(
                {
                    "record_type": "log_event_val",
                    "metric_table": "log",
                    "run_section": section_name,
                    "artifact_name": "log",
                    "metric_key": "log.val_metric",
                    "value": event.get("val_metric"),
                    "value_type": type(event.get("val_metric")).__name__,
                    "source_file": section_payload.get("source_file", ""),
                    "status": section_payload.get("status", "unknown"),
                    "section_type": section_payload.get("section_type"),
                    "epoch": epoch,
                    "metric_name": "val_metric",
                }
            )
    return rows


def _flatten_for_csv(value: Any, prefix: str = "", sep: str = ".") -> list[tuple[str, Any]]:
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
    if isinstance(value, (str, int, float, bool) or value is None):
        return [(prefix, value)]
    return []


def _collect_section_rows(sections: Mapping[str, SectionRecord]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for section_name, section_payload in sections.items():
        if not isinstance(section_payload, Mapping):
            continue
        source_file = section_payload.get("source_file")
        status = section_payload.get("status", "unknown")
        artifact_name = section_payload.get("artifact_name")
        section_type = section_payload.get("section_type")
        rows.append(
            {
                "record_type": "section_status",
                "metric_table": "section",
                "run_section": section_name,
                "artifact_name": artifact_name if artifact_name is not None else "",
                "metric_key": "status",
                "value": status,
                "value_type": type(status).__name__,
                "source_file": source_file if source_file is not None else "",
                "status": status,
                "section_type": section_type,
            }
        )

        metrics = section_payload.get("metrics", {})
        if not isinstance(metrics, Mapping):
            continue
        if section_payload.get("file_status") is not None:
            rows.append(
                {
                    "record_type": "section_file",
                    "metric_table": "section",
                    "run_section": section_name,
                    "artifact_name": artifact_name if artifact_name is not None else "",
                    "metric_key": "file.size_bytes",
                    "value": section_payload.get("file_status", {}).get("size_bytes", 0),
                    "value_type": "int",
                    "source_file": source_file if source_file is not None else "",
                    "status": status,
                    "section_type": section_type,
                }
            )
            rows.append(
                {
                    "record_type": "section_file",
                    "metric_table": "section",
                    "run_section": section_name,
                    "artifact_name": artifact_name if artifact_name is not None else "",
                    "metric_key": "file.sha256",
                    "value": section_payload.get("file_status", {}).get("sha256"),
                    "value_type": "str",
                    "source_file": source_file if source_file is not None else "",
                    "status": status,
                    "section_type": section_type,
                }
            )

        for metric_key, metric_value in _flatten_for_csv(metrics):
            if metric_key == "":
                continue
            if metric_key.startswith("events"):
                continue
            if metric_key == "events":
                continue
            if metric_key.startswith("file_status"):
                continue
            rows.append(
                {
                    "record_type": "section_metric",
                    "metric_table": "metrics",
                    "run_section": section_name,
                    "artifact_name": artifact_name if artifact_name is not None else "",
                    "metric_key": metric_key,
                    "value": metric_value,
                    "value_type": type(metric_value).__name__,
                    "source_file": source_file if source_file is not None else "",
                    "status": status,
                    "section_type": section_type,
                }
            )
    return rows


def _collect_series_meta(sections: Mapping[str, SectionRecord]) -> dict[str, list[dict[str, Any]]]:
    series: dict[str, list[dict[str, Any]]] = {
        "loss": [],
        "scalars": [],
        "log": [],
    }
    for section_name, section_payload in sections.items():
        artifact = section_payload.get("artifact_name")
        if artifact not in {"loss", "scalars", "log"}:
            continue
        metrics = section_payload.get("metrics", {})
        if not isinstance(metrics, Mapping):
            continue
        if artifact == "loss":
            series["loss"].append(
                {
                    "run_section": section_name,
                    "path": section_payload.get("source_file"),
                    "rows": metrics.get("rows"),
                }
            )
        elif artifact == "scalars":
            series["scalars"].append(
                {
                    "run_section": section_name,
                    "path": section_payload.get("source_file"),
                    "rows": metrics.get("rows"),
                }
            )
        elif artifact == "log":
            series["log"].append(
                {
                    "run_section": section_name,
                    "path": section_payload.get("source_file"),
                    "events": metrics.get("event_count"),
                }
            )
    return series


def _build_summary(
    sections: Mapping[str, SectionRecord], run_label: str | None = None
) -> dict[str, Any]:
    missing_sections = [name for name, payload in sections.items() if payload.get("status") != "ok"]
    loaded_sections = [name for name, payload in sections.items() if payload.get("status") == "ok"]
    series = _collect_series_meta(sections)
    return {
        "run_label": run_label,
        "n_sections": len(sections),
        "n_loaded": len(loaded_sections),
        "n_missing": len(missing_sections),
        "loaded_sections": loaded_sections,
        "missing_sections": missing_sections,
        "n_loss_rows": sum(item.get("rows", 0) for item in series["loss"] if isinstance(item.get("rows"), int)),
        "n_scalar_points": sum(
            item.get("rows", 0) for item in series["scalars"] if isinstance(item.get("rows"), int)
        ),
        "n_log_events": sum(item.get("events", 0) for item in series["log"] if isinstance(item.get("events"), int)),
    }


def build_training_export_bundle(
    strategy_run_dir: Path,
    *,
    parse_logs: bool = True,
    strict: bool = False,
    run_label: str | None = None,
) -> dict[str, Any]:
    sections = collect_training_sections(
        strategy_run_dir=strategy_run_dir,
        parse_logs=parse_logs,
        strict=strict,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pipeline_context": {
            "strategy_run_dir": str(strategy_run_dir),
            "run_label": run_label,
            "strict": strict,
            "parse_logs": parse_logs,
        },
        "summary": _build_summary(sections, run_label=run_label),
        "time_series": _collect_series_meta(sections),
        "sections": sections,
    }


def _to_csv_value(value: Any) -> str | int | float | bool:
    if value is None:
        return ""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return ""
        return value
    return str(value)


def _write_json(bundle: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(bundle, fh, indent=2, ensure_ascii=False, sort_keys=True)


def _write_csv(bundle: Mapping[str, Any], path: Path) -> None:
    sections = bundle.get("sections", {})
    rows: list[dict[str, str | float | int | bool | None]] = []
    if isinstance(sections, Mapping):
        rows.extend(_collect_section_rows(sections))
        rows.extend(_collect_loss_rows(sections))
        rows.extend(_collect_scalars_rows(sections))
        rows.extend(_collect_log_rows(sections))

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _to_csv_value(row.get(field)) for field in CSV_FIELDS})


def export_training_info(
    strategy_run_dir: Path,
    *,
    output_json: Path | None = None,
    output_csv: Path | None = None,
    strict: bool = False,
    parse_logs: bool = True,
    run_label: str | None = None,
) -> dict[str, Any]:
    bundle = build_training_export_bundle(
        strategy_run_dir=strategy_run_dir,
        strict=strict,
        parse_logs=parse_logs,
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


def _resolve_export_path(
    *, args: argparse.Namespace, run_dir: Path
) -> tuple[Path, Path]:
    if args.training_export_json is not None and args.training_export_csv is not None:
        return args.training_export_json, args.training_export_csv
    if args.training_export_json is not None:
        return (
            args.training_export_json,
            args.training_export_json.parent / "training_export.csv",
        )
    if args.training_export_csv is not None:
        return (
            args.training_export_csv.parent / "training_export.json",
            args.training_export_csv,
        )
    output_dir = args.training_export_dir or (run_dir / "training")
    return output_dir / "training_export.json", output_dir / "training_export.csv"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Consolidate THT-TripGen training artifacts into one JSON + CSV."
    )
    parser.add_argument(
        "--strategy-run-dir",
        required=True,
        type=Path,
        help="Path to strategy THT-TripGen run directory.",
    )
    parser.add_argument(
        "--training-export-dir",
        type=Path,
        default=None,
        help="Directory for consolidated training export (default: <strategy-run-dir>/training).",
    )
    parser.add_argument("--training-export-json", type=Path, default=None)
    parser.add_argument("--training-export-csv", type=Path, default=None)
    parser.add_argument(
        "--training-export-parse-logs",
        type=_bool_or_none,
        default=True,
        help="Parse log text for epoch-level summaries (true/false).",
    )
    parser.add_argument("--export-json", type=_bool_or_none, default=True)
    parser.add_argument("--export-csv", type=_bool_or_none, default=True)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--run-label", type=str, default=None)
    args = parser.parse_args()

    strategy_run_dir = args.strategy_run_dir
    if not strategy_run_dir.exists():
        raise FileNotFoundError(f"strategy run dir does not exist: {strategy_run_dir}")

    output_json, output_csv = _resolve_export_path(args=args, run_dir=strategy_run_dir)
    export_training_info(
        strategy_run_dir=strategy_run_dir,
        output_json=output_json if args.export_json is not False else None,
        output_csv=output_csv if args.export_csv is not False else None,
        strict=args.strict,
        parse_logs=args.training_export_parse_logs if args.training_export_parse_logs is not None else True,
        run_label=args.run_label,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
