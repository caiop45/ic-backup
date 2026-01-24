import pytest


gpd = pytest.importorskip("geopandas")
pytest.importorskip("shapely")

from shapely.geometry import Polygon

from topology.physical_adjacency import compute_edges_location_id


def test_compute_edges_rook_vs_queen():
    polygons = [
        Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
        Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
        Polygon([(0, 1), (1, 1), (1, 2), (0, 2)]),
        Polygon([(1, 1), (2, 1), (2, 2), (1, 2)]),
    ]
    gdf = gpd.GeoDataFrame(
        {"LocationID": [1, 2, 3, 4], "geometry": polygons}, crs="EPSG:4326"
    )

    edges_rook = compute_edges_location_id(
        gdf,
        "LocationID",
        mode="rook",
        tol=1e-9,
        prefilter="bbox",
    )
    edges_queen = compute_edges_location_id(
        gdf,
        "LocationID",
        mode="queen",
        tol=1e-9,
        prefilter="bbox",
    )

    expected_rook = [(1, 2), (1, 3), (2, 4), (3, 4)]
    expected_queen = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]

    assert edges_rook == expected_rook
    assert edges_queen == expected_queen
