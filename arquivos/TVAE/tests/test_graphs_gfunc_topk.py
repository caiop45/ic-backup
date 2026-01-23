import pandas as pd

from topology.graphs import build_functional_graph


def test_build_functional_graph_topk():
    df = pd.DataFrame(
        {
            "o_idx": [0, 0, 0, 0, 1, 1],
            "d_idx": [1, 1, 2, 3, 2, 3],
        }
    )
    graph = build_functional_graph(df, num_nodes=4, top_k=2)

    assert len(graph.neighbors[0]) == 2
    assert 1 in set(graph.neighbors[0])

    # origin 1 only has two outgoing, should keep both
    assert set(graph.neighbors[1]) == {2, 3}
