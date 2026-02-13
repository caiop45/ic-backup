import json

import pytest


gpd = pytest.importorskip("geopandas")
pytest.importorskip("shapely")

import pandas as pd

from shapely.geometry import Polygon

from tvae.tools.inspect_tht_phys_graph import inspect_physical_graphs


def test_inspect_tht_phys_graph_smoke(tmp_path):
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
        gap_tol=0.0,
        report_near_miss=False,
        nearest_k=3,
        rook_edges_path=None,
        queen_edges_path=None,
        mappings_path=None,
        top_k=5,
    )

    expected_files = [
        "stats_rook.json",
        "stats_queen_strict.json",
        "stats_queen_tol.json",
        "stats_diff_rook_vs_queen.json",
        "stats_diff_strict_vs_tol.json",
        "degrees_rook.csv",
        "degrees_queen_strict.csv",
        "degrees_queen_tol.csv",
        "degrees_delta_rook_vs_queen.csv",
        "degrees_delta_strict_vs_tol.csv",
        "edges_rook.csv",
        "edges_queen_strict.csv",
        "edges_queen_tol.csv",
        "edges_only_in_tolerant.csv",
        "edges_only_in_strict.csv",
        "neighbors_rook.json",
        "neighbors_queen_strict.json",
        "neighbors_queen_tol.json",
        "isolated_nearest.csv",
    ]
    for name in expected_files:
        assert (out_dir / name).exists()

    stats = json.loads((out_dir / "stats_rook.json").read_text(encoding="utf-8"))
    for key in ["num_nodes", "num_edges", "isolated_nodes", "components_count"]:
        assert key in stats


def test_inspect_tht_phys_graph_gap_tol_near_miss(tmp_path):
    polygons = [
        Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
        Polygon([(1.05, 0), (2.05, 0), (2.05, 1), (1.05, 1)]),
    ]
    gdf = gpd.GeoDataFrame(
        {"LocationID": [1, 2], "geometry": polygons}, crs="EPSG:3857"
    )

    out_dir = tmp_path / "outputs_gap"

    inspect_physical_graphs(
        gdf,
        out_dir,
        mode="compute",
        tol=1e-9,
        prefilter="bbox",
        fix_geoms="none",
        crs_metric="none",
        gap_tol=0.1,
        report_near_miss=True,
        nearest_k=2,
        rook_edges_path=None,
        queen_edges_path=None,
        mappings_path=None,
        top_k=5,
    )

    near_miss_path = out_dir / "near_miss_edges.csv"
    edges_tol_path = out_dir / "edges_queen_tol.csv"

    assert near_miss_path.exists()
    assert edges_tol_path.exists()

    near_df = pd.read_csv(near_miss_path)
    assert not near_df.empty
    assert (near_df.loc[0, "u_location_id"], near_df.loc[0, "v_location_id"]) == (1, 2)

    edges_tol_df = pd.read_csv(edges_tol_path)
    assert ((edges_tol_df["u_location_id"] == 1) & (edges_tol_df["v_location_id"] == 2)).any()
