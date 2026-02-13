import csv
import json

from pathlib import Path

import pytest
import torch

import config
from tools import export_training_info


torch = pytest.importorskip("torch")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        if not rows:
            return
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _touch_file(path: Path, text: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_tvae_run(path: Path, with_log: bool = True) -> None:
    _write_json(
        path / f"mappings_{config.FIXED_ORDER_KEY}.json",
        {"columns": ["pickup_id", "dropoff_id"], "categories": {"pickup_id": [1, 2], "dropoff_id": [3, 4]}},
    )
    _write_csv(
        path / f"loss_{config.FIXED_ORDER_KEY}.csv",
        [
            {
                "epoch": "1",
                "train_loss": "1.0",
                "val_loss": "1.2",
            }
        ],
    )
    checkpoint = {
        "meta": {
            "column_sizes": [2, 2],
            "order": ["pickup_id", "dropoff_id"],
            "encoder_hidden_dims": [16],
            "decoder_hidden_dims": [16],
            "latent_dim": 4,
        },
        "epoch": 1,
        "optimizer_state": {"lr": 0.001},
        "metrics": {"best_val": 1.2},
        "model_state": {"linear": torch.tensor([[1.0, 2.0]])},
    }
    torch.save(checkpoint, path / f"tvae_{config.FIXED_ORDER_KEY}.pt")
    _write_csv(
        path / "logs" / "scalars.csv",
        [
            {"step": "1", "tag": "train/loss", "value": "1.0"},
            {"step": "1", "tag": "val/loss", "value": "1.2"},
        ],
    )
    if with_log:
        _touch_file(
            Path(config.LOG_DIR) / path.name / f"train_{config.FIXED_ORDER_KEY}.log",
            "[Epoch 001] train_loss=1.0 val_loss=1.2\n",
        )


def _write_tht_run(path: Path, with_log: bool = True) -> None:
    _write_json(
        path / "mappings_tht_tripgen.json",
        {
            "zone_categories": {"a": [1, 2], "b": [3]},
            "time_categories": list(range(4)),
            "conditional_categories": {"passenger_count": [1, 2], "total_amount": [10, 20]},
            "use_passenger_count": True,
            "use_total_amount": False,
        },
    )
    _write_csv(
        path / "loss_tht_tripgen.csv",
        [
            {
                "epoch": "1",
                "train_nll": "2.0",
                "val_nll": "2.2",
                "best_val": "2.2",
            }
        ],
    )
    checkpoint = {
        "meta": {
            "num_zones": 3,
            "num_time_bins": 4,
            "conditional_cardinalities": {"a": 4, "b": 5},
            "zone_embeddings_source": "Efunc",
            "zone_embeddings_dim": 8,
            "cond_emb_dim": 2,
            "time_emb_dim": 2,
            "origin_emb_dim": 3,
            "context_mlp_hidden": 8,
            "context_mlp_layers": 2,
            "dropout": 0.1,
            "destination_head_type": "embedding_softmax",
            "hybrid_residual_hidden": 4,
            "hybrid_residual_weight_init": 1.0,
            "use_passenger_count": True,
            "passenger_cardinality": 4,
            "use_total_amount": False,
            "total_amount_sigma_floor": 0.1,
            "min_r_eps": 1e-6,
            "residual_num_layers": 1,
            "residual_num_bins": 8,
            "residual_context_hidden": 8,
            "residual_min_bin_width": 0.1,
            "residual_min_bin_height": 0.1,
            "residual_min_deriv": 0.1,
            "residual_eps": 1e-6,
        },
        "epoch": 2,
        "optimizer_state": {"lr": 0.001},
        "metrics": {"best_val": 2.2},
        "model_state": {"linear": torch.tensor([[1.0, 2.0]])},
    }
    torch.save(checkpoint, path / "tht_tripgen.pt")
    _write_csv(
        path / "logs" / "scalars.csv",
        [
            {"step": "1", "tag": "train/nll_total", "value": "2.0"},
            {"step": "1", "tag": "val/nll_total", "value": "2.2"},
        ],
    )
    if with_log:
        _touch_file(
            Path(config.LOG_DIR) / path.name / "train_tht_tripgen.log",
            "[Epoch 001] train_nll=2.0 val_nll=2.2\n",
        )


def test_collect_training_sections_with_missing_optional_tvae(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "LOG_DIR", str(tmp_path / "logs"))
    baseline = tmp_path / "baseline"
    strategy = tmp_path / "strategy"

    _write_tvae_run(baseline)
    _write_tht_run(strategy)
    Path(config.LOG_DIR).mkdir(parents=True, exist_ok=True)

    bundle = export_training_info.build_training_export_bundle(
        strategy_run_dir=strategy,
        baseline_run_dir=baseline,
        strict=False,
        parse_logs=True,
        run_label="smoke",
    )

    sections = bundle["sections"]
    assert sections["baseline_tvae_checkpoint"]["status"] == "ok"
    assert sections["baseline_tvae_mappings"]["status"] == "ok"
    assert sections["baseline_tvae_log"]["status"] == "ok"
    assert sections["strategy_tht_checkpoint"]["status"] == "ok"
    assert "strategy_tvae_checkpoint" not in sections
    assert "strategy_tvae_artifacts" not in sections


def test_export_training_info_writes_json_and_csv(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "LOG_DIR", str(tmp_path / "logs"))
    strategy = tmp_path / "strategy"
    _write_tht_run(strategy, with_log=True)
    out_json = tmp_path / "out" / "train.json"
    out_csv = tmp_path / "out" / "train.csv"

    export_training_info.export_training_info(
        strategy,
        output_json=out_json,
        output_csv=out_csv,
        strict=True,
        parse_logs=True,
    )
    assert out_json.exists()
    assert out_csv.exists()

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert "sections" in payload
    assert payload["pipeline_context"]["strategy_run_dir"] == str(strategy)

    rows = list(csv.DictReader(out_csv.open("r", encoding="utf-8")))
    keys = {(row["run_section"], row["metric_key"]) for row in rows}
    assert ("strategy_tht_checkpoint", "epoch") in keys
