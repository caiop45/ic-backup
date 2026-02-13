import pytest

pytest.importorskip("torch")

from tvae.topology.graphs import WeightedGraph
from tvae.topology.node2vec import Node2VecConfig, generate_walks


def test_node2vec_walk_determinism():
    graph = WeightedGraph(
        num_nodes=3,
        directed=True,
        neighbors=[[1, 2], [2], [0]],
        weights=[[1.0, 1.0], [1.0], [1.0]],
    )
    cfg = Node2VecConfig(walk_length=5, walks_per_node=2, seed=123)

    walks_a = generate_walks(graph, cfg)
    walks_b = generate_walks(graph, cfg)

    assert walks_a == walks_b
