import json

import pytest


gpd = pytest.importorskip("geopandas")
pytest.importorskip("shapely")

from shapely.geometry import Polygon

from tools.inspect_sa_phys_graph import inspect_physical_graphs


def test_inspect_sa_phys_graph_smoke(tmp_path):
    polygons = [
        Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
        Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
        Polygon([(0, 1), (1, 1), (1, 2), (0, 2)]),
        Polygon([(1, 1), (2, 1), (2, 2), (1, 2)]),
    ]
    gdf = gpd.GeoDataFrame(
        {"LocationID": [1, 2, 3, 4], "geometry": polygons}, crs="EPSG:4326"
    )

    out_dir = tmp_path / "outputs"

    inspect_physical_graphs(
        gdf,
        out_dir,
        mode="compute",
        tol=1e-9,
        prefilter="bbox",
        fix_geoms="none",
        crs_metric="none",
        rook_edges_path=None,
        queen_edges_path=None,
        mappings_path=None,
        top_k=5,
    )

    expected_files = [
        "stats_rook.json",
        "stats_queen.json",
        "stats_diff.json",
        "degrees_rook.csv",
        "degrees_queen.csv",
        "degrees_delta.csv",
        "edges_rook.csv",
        "edges_queen.csv",
        "edges_only_in_queen.csv",
        "edges_only_in_rook.csv",
        "neighbors_rook.json",
        "neighbors_queen.json",
    ]
    for name in expected_files:
        assert (out_dir / name).exists()

    stats = json.loads((out_dir / "stats_rook.json").read_text(encoding="utf-8"))
    for key in ["num_nodes", "num_edges", "isolated_nodes", "components_count"]:
        assert key in stats
