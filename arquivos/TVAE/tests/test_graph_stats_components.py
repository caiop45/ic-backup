from topology.physical_adjacency import compute_graph_stats


def test_graph_stats_components_and_isolated_nodes():
    edges = [(0, 1), (1, 2), (3, 4)]
    stats = compute_graph_stats(num_nodes=6, edges=edges)

    assert [len(c) for c in stats.components] == [3, 2, 1]
    assert stats.isolated_nodes == [5]
