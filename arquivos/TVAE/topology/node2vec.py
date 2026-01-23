"""Node2Vec implementation without external dependencies."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import torch
import torch.nn.functional as F

import config
from topology.graphs import WeightedGraph


@dataclass
class Node2VecConfig:
    embedding_dim: int = config.SA_NODE2VEC_DIM
    walk_length: int = config.SA_NODE2VEC_WALK_LENGTH
    walks_per_node: int = config.SA_NODE2VEC_WALKS_PER_NODE
    window_size: int = config.SA_NODE2VEC_WINDOW
    p: float = config.SA_NODE2VEC_P
    q: float = config.SA_NODE2VEC_Q
    negative_samples: int = 5
    lr: float = 0.025
    epochs: int = 3
    batch_size: int = 1024
    seed: int = config.GLOBAL_SEED


def _is_connected(graph: WeightedGraph, src: int, dst: int) -> bool:
    neighbor_set = graph.ensure_neighbor_set()
    if graph.directed:
        return (dst in neighbor_set[src]) or (src in neighbor_set[dst])
    return dst in neighbor_set[src]


def generate_walks(graph: WeightedGraph, cfg: Node2VecConfig) -> List[List[int]]:
    if cfg.walk_length <= 0:
        raise ValueError("walk_length must be positive")
    if cfg.walks_per_node <= 0:
        raise ValueError("walks_per_node must be positive")
    if cfg.p <= 0 or cfg.q <= 0:
        raise ValueError("p and q must be positive")

    rng = np.random.default_rng(cfg.seed)
    walks: List[List[int]] = []

    for node in range(graph.num_nodes):
        for _ in range(cfg.walks_per_node):
            walk = [node]
            prev = None
            curr = node

            for _step in range(cfg.walk_length - 1):
                neighbors = graph.neighbors[curr]
                if not neighbors:
                    break

                weights = graph.weights[curr]
                if prev is None:
                    probs = np.array(weights, dtype=np.float64)
                else:
                    unnorm = np.empty(len(neighbors), dtype=np.float64)
                    for i, nbr in enumerate(neighbors):
                        base = float(weights[i])
                        if nbr == prev:
                            base = base / cfg.p
                        elif _is_connected(graph, prev, nbr):
                            base = base
                        else:
                            base = base / cfg.q
                        unnorm[i] = base
                    probs = unnorm

                total = probs.sum()
                if total <= 0:
                    probs = np.ones_like(probs) / float(len(probs))
                else:
                    probs = probs / total

                next_node = int(rng.choice(neighbors, p=probs))
                walk.append(next_node)
                prev, curr = curr, next_node

            walks.append(walk)

    return walks


def _build_pairs(walks: List[List[int]], window_size: int) -> tuple[np.ndarray, np.ndarray]:
    centers: List[int] = []
    contexts: List[int] = []

    for walk in walks:
        length = len(walk)
        for i, center in enumerate(walk):
            start = max(0, i - window_size)
            end = min(length, i + window_size + 1)
            for j in range(start, end):
                if j == i:
                    continue
                centers.append(center)
                contexts.append(walk[j])

    if not centers:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    return np.array(centers, dtype=np.int64), np.array(contexts, dtype=np.int64)


def train_skipgram(
    walks: List[List[int]],
    num_nodes: int,
    cfg: Node2VecConfig,
    *,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    if num_nodes <= 0:
        raise ValueError("num_nodes must be positive")

    centers_np, contexts_np = _build_pairs(walks, cfg.window_size)

    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)

    emb_in = torch.randn(num_nodes, cfg.embedding_dim, device=device) * 0.01
    emb_out = torch.randn(num_nodes, cfg.embedding_dim, device=device) * 0.01
    emb_in.requires_grad_(True)
    emb_out.requires_grad_(True)

    if len(centers_np) == 0:
        return emb_in.detach().cpu()

    counts = np.zeros(num_nodes, dtype=np.float64)
    for walk in walks:
        for node in walk:
            counts[node] += 1.0

    neg_dist = counts ** 0.75
    if neg_dist.sum() == 0:
        neg_dist = np.ones(num_nodes, dtype=np.float64) / float(num_nodes)
    else:
        neg_dist = neg_dist / neg_dist.sum()
    neg_dist_t = torch.tensor(neg_dist, dtype=torch.float32, device=device)

    optimizer = torch.optim.SGD([emb_in, emb_out], lr=cfg.lr)

    num_pairs = len(centers_np)
    batch_size = max(1, cfg.batch_size)

    for epoch in range(cfg.epochs):
        perm = rng.permutation(num_pairs)
        centers_shuf = centers_np[perm]
        contexts_shuf = contexts_np[perm]

        for start in range(0, num_pairs, batch_size):
            end = min(start + batch_size, num_pairs)
            center_idx = torch.tensor(
                centers_shuf[start:end], dtype=torch.long, device=device
            )
            pos_idx = torch.tensor(
                contexts_shuf[start:end], dtype=torch.long, device=device
            )

            center_vec = emb_in[center_idx]
            pos_vec = emb_out[pos_idx]
            pos_score = (center_vec * pos_vec).sum(dim=1)
            pos_loss = -F.logsigmoid(pos_score)

            neg_count = cfg.negative_samples
            neg_idx = torch.multinomial(
                neg_dist_t, num_samples=(end - start) * neg_count, replacement=True
            ).view(-1, neg_count)
            neg_vec = emb_out[neg_idx]
            neg_score = torch.bmm(neg_vec, center_vec.unsqueeze(2)).squeeze(2)
            neg_loss = -F.logsigmoid(-neg_score).sum(dim=1)

            loss = (pos_loss + neg_loss).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return emb_in.detach().cpu()


def train_node2vec(
    graph: WeightedGraph,
    cfg: Node2VecConfig,
    *,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    walks = generate_walks(graph, cfg)
    return train_skipgram(walks, graph.num_nodes, cfg, device=device)
