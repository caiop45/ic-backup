"""Build and cache node2vec embeddings for THT-TripGen.

Usage:
    python tools/build_tht_zone_embeddings.py [--force] [--device cpu] [--alpha 1.0] [--beta 1.0]

This script:
    - Loads THT-TripGen train data.
    - Fits THTTripGenTransformer to build shared zone vocabulary.
    - Builds functional graph from OD counts (top-K pruning).
    - Optionally builds physical graph from adjacency CSV (index space).
    - Trains node2vec embeddings and saves to the topology cache directory.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import torch

import config
from data_processing.tht_tripgen_loader import load_and_split_tht_tripgen
from data_processing.tht_tripgen_transformer import THTTripGenTransformer
from topology.embeddings import combine_embeddings
from topology.graphs import build_functional_graph, build_physical_graph_from_edges
from topology.node2vec import Node2VecConfig, train_node2vec


def _save_tensor(path: Path, tensor: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor, path)


def _save_metadata(path: Path, meta: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2, ensure_ascii=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="recompute embeddings")
    parser.add_argument("--device", default="cpu", help="torch device")
    parser.add_argument("--alpha", type=float, default=1.0, help="scale for Ephys")
    parser.add_argument("--beta", type=float, default=1.0, help="scale for Efunc")
    args = parser.parse_args()

    cache_dir = Path(config.THT_TOPOLOGY_CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)

    func_path = cache_dir / "Efunc.pt"
    phys_path = cache_dir / "Ephys.pt"
    comb_path = cache_dir / "Ecomb.pt"
    meta_path = cache_dir / "metadata.json"

    train_df, _, _ = load_and_split_tht_tripgen()
    transformer = THTTripGenTransformer().fit(train_df)
    train_idx = transformer.transform(train_df)
    num_nodes = transformer.num_zones

    node2vec_cfg = Node2VecConfig(
        embedding_dim=config.THT_NODE2VEC_DIM,
        walk_length=config.THT_NODE2VEC_WALK_LENGTH,
        walks_per_node=config.THT_NODE2VEC_WALKS_PER_NODE,
        window_size=config.THT_NODE2VEC_WINDOW,
        p=config.THT_NODE2VEC_P,
        q=config.THT_NODE2VEC_Q,
        seed=config.GLOBAL_SEED,
    )

    embeddings: Dict[str, torch.Tensor] = {}

    if args.force or not func_path.exists():
        gfunc = build_functional_graph(
            train_idx,
            num_nodes=num_nodes,
            top_k=config.THT_GFUNC_TOPK,
            weight_mode="log1p",
        )
        efunc = train_node2vec(gfunc, node2vec_cfg, device=args.device)
        _save_tensor(func_path, efunc)
    embeddings["Efunc"] = torch.load(func_path, map_location="cpu")

    if config.THT_PHYS_EDGES_CSV:
        if args.force or not phys_path.exists():
            gphys = build_physical_graph_from_edges(
                config.THT_PHYS_EDGES_CSV, num_nodes=num_nodes
            )
            ephys = train_node2vec(gphys, node2vec_cfg, device=args.device)
            _save_tensor(phys_path, ephys)
        embeddings["Ephys"] = torch.load(phys_path, map_location="cpu")

    if "Efunc" in embeddings and "Ephys" in embeddings:
        ecomb = combine_embeddings(
            embeddings["Ephys"],
            embeddings["Efunc"],
            alpha=args.alpha,
            beta=args.beta,
            mode="concat",
        )
        _save_tensor(comb_path, ecomb)

    metadata = {
        "num_nodes": num_nodes,
        "node2vec": {
            "embedding_dim": node2vec_cfg.embedding_dim,
            "walk_length": node2vec_cfg.walk_length,
            "walks_per_node": node2vec_cfg.walks_per_node,
            "window_size": node2vec_cfg.window_size,
            "p": node2vec_cfg.p,
            "q": node2vec_cfg.q,
            "negative_samples": node2vec_cfg.negative_samples,
            "lr": node2vec_cfg.lr,
            "epochs": node2vec_cfg.epochs,
            "batch_size": node2vec_cfg.batch_size,
            "seed": node2vec_cfg.seed,
        },
        "gfunc_top_k": config.THT_GFUNC_TOPK,
        "phys_edges_csv": config.THT_PHYS_EDGES_CSV,
        "alpha": args.alpha,
        "beta": args.beta,
    }
    _save_metadata(meta_path, metadata)


if __name__ == "__main__":
    main()
