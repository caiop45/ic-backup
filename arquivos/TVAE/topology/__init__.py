from .graphs import WeightedGraph, build_functional_graph, build_physical_graph_from_edges
__all__ = [
    "WeightedGraph",
    "build_functional_graph",
    "build_physical_graph_from_edges",
]

try:  # torch is optional for graph-only usage
    from .embeddings import combine_embeddings, l2_normalize
    from .node2vec import Node2VecConfig, generate_walks, train_node2vec, train_skipgram

    __all__ += [
        "Node2VecConfig",
        "generate_walks",
        "train_node2vec",
        "train_skipgram",
        "l2_normalize",
        "combine_embeddings",
    ]
except ModuleNotFoundError:
    pass
