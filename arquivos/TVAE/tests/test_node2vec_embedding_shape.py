import pytest

torch = pytest.importorskip("torch")

from topology.graphs import WeightedGraph
from topology.node2vec import Node2VecConfig, train_node2vec


def test_node2vec_embedding_shape():
    graph = WeightedGraph(
        num_nodes=4,
        directed=True,
        neighbors=[[1, 2], [2], [3], [0]],
        weights=[[1.0, 1.0], [1.0], [1.0], [1.0]],
    )
    cfg = Node2VecConfig(
        embedding_dim=8,
        walk_length=6,
        walks_per_node=2,
        window_size=2,
        epochs=1,
        batch_size=16,
        negative_samples=2,
        seed=7,
    )

    emb = train_node2vec(graph, cfg, device="cpu")

    assert emb.shape == (4, 8)
    assert torch.isfinite(emb).all()
