"""Graph builders for Strategy A topology.

Gfunc: directed weighted graph from OD counts (log1p weights, top-K pruning).
Gphys: undirected graph from physical adjacency edges.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


@dataclass
class WeightedGraph:
    num_nodes: int
    directed: bool
    neighbors: List[List[int]]
    weights: List[List[float]]
    neighbor_set: List[set[int]] | None = None

    def ensure_neighbor_set(self) -> List[set[int]]:
        if self.neighbor_set is None:
            self.neighbor_set = [set(nbrs) for nbrs in self.neighbors]
        return self.neighbor_set


def build_functional_graph(
    train_idx_df: pd.DataFrame,
    num_nodes: int,
    top_k: int = 100,
    weight_mode: str = "log1p",
) -> WeightedGraph:
    if "o_idx" not in train_idx_df.columns or "d_idx" not in train_idx_df.columns:
        raise ValueError("train_idx_df must contain o_idx and d_idx columns")
    if num_nodes <= 0:
        raise ValueError("num_nodes must be positive")

    counts = (
        train_idx_df.groupby(["o_idx", "d_idx"]).size().reset_index(name="count")
    )

    if weight_mode == "log1p":
        counts["weight"] = np.log1p(counts["count"].to_numpy(dtype=np.float64))
    elif weight_mode == "count":
        counts["weight"] = counts["count"].to_numpy(dtype=np.float64)
    else:
        raise ValueError(f"Unsupported weight_mode: {weight_mode}")

    neighbors: List[List[int]] = [[] for _ in range(num_nodes)]
    weights: List[List[float]] = [[] for _ in range(num_nodes)]

    for origin, group in counts.groupby("o_idx"):
        origin_int = int(origin)
        if origin_int < 0 or origin_int >= num_nodes:
            raise ValueError(f"Origin index out of range: {origin_int}")

        group_sorted = group.sort_values("weight", ascending=False)
        if top_k is not None and top_k > 0:
            group_sorted = group_sorted.head(int(top_k))

        neighbors[origin_int] = group_sorted["d_idx"].astype("int64").tolist()
        weights[origin_int] = group_sorted["weight"].astype("float64").tolist()

    graph = WeightedGraph(
        num_nodes=num_nodes,
        directed=True,
        neighbors=neighbors,
        weights=weights,
    )
    validate_graph(graph)
    return graph


def build_physical_graph_from_edges(
    csv_path: str | Path, num_nodes: int
) -> WeightedGraph:
    if num_nodes <= 0:
        raise ValueError("num_nodes must be positive")

    df = pd.read_csv(csv_path)
    if df.shape[1] < 2:
        raise ValueError("Physical edges CSV must contain at least two columns")

    edges = df.iloc[:, :2].dropna()
    edges = edges.astype("int64")

    adjacency: List[set[int]] = [set() for _ in range(num_nodes)]
    for u, v in edges.itertuples(index=False, name=None):
        if u < 0 or u >= num_nodes or v < 0 or v >= num_nodes:
            raise ValueError(f"Edge node out of range: ({u}, {v})")
        if u == v:
            continue
        adjacency[u].add(v)
        adjacency[v].add(u)

    neighbors = [sorted(list(nbrs)) for nbrs in adjacency]
    weights = [[1.0 for _ in nbrs] for nbrs in neighbors]

    graph = WeightedGraph(
        num_nodes=num_nodes,
        directed=False,
        neighbors=neighbors,
        weights=weights,
    )
    validate_graph(graph)
    return graph


def validate_graph(graph: WeightedGraph) -> None:
    if len(graph.neighbors) != graph.num_nodes:
        raise ValueError("neighbors length must match num_nodes")
    if len(graph.weights) != graph.num_nodes:
        raise ValueError("weights length must match num_nodes")

    for node in range(graph.num_nodes):
        nbrs = graph.neighbors[node]
        wts = graph.weights[node]
        if len(nbrs) != len(wts):
            raise ValueError(f"Weights length mismatch at node {node}")
        for nbr in nbrs:
            if nbr < 0 or nbr >= graph.num_nodes:
                raise ValueError(f"Neighbor index out of range: {nbr}")
