import pytest


gpd = pytest.importorskip("geopandas")
pytest.importorskip("shapely")

from shapely.geometry import Polygon

from tvae.topology.physical_adjacency import (
    augment_connect_components_by_distance,
    compute_edges_location_id,
    compute_near_miss_edges_location_id,
)


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


def test_queen_gap_tolerance_adds_edge():
    polygons = [
        Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
        Polygon([(1.1, 0), (2.1, 0), (2.1, 1), (1.1, 1)]),
    ]
    gdf = gpd.GeoDataFrame(
        {"LocationID": [10, 20], "geometry": polygons}, crs="EPSG:3857"
    )

    edges_strict = compute_edges_location_id(
        gdf,
        "LocationID",
        mode="queen",
        tol=1e-9,
        prefilter="bbox",
        gap_tol=0.0,
    )
    edges_tolerant = compute_edges_location_id(
        gdf,
        "LocationID",
        mode="queen",
        tol=1e-9,
        prefilter="bbox",
        gap_tol=0.2,
    )

    assert edges_strict == []
    assert edges_tolerant == [(10, 20)]


def test_compute_near_miss_edges_location_id():
    polygons = [
        Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
        Polygon([(1.05, 0), (2.05, 0), (2.05, 1), (1.05, 1)]),
    ]
    gdf = gpd.GeoDataFrame(
        {"LocationID": [1, 2], "geometry": polygons}, crs="EPSG:3857"
    )

    near = compute_near_miss_edges_location_id(
        gdf,
        "LocationID",
        gap_tol=0.1,
        prefilter="bbox",
    )

    assert len(near) == 1
    u, v, dist = near[0]
    assert (u, v) == (1, 2)
    assert dist == pytest.approx(0.05, rel=1e-6)


def test_augment_connect_components_by_distance():
    polygons = [
        Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
        Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
        Polygon([(100, 0), (101, 0), (101, 1), (100, 1)]),
    ]
    gdf = gpd.GeoDataFrame(
        {"LocationID": [1, 2, 3], "geometry": polygons}, crs="EPSG:3857"
    )
    strict_edges = [(1, 2)]

    edges_aug, added = augment_connect_components_by_distance(
        gdf, "LocationID", strict_edges, max_dist=200.0
    )

    assert len(added) == 1
    assert (2, 3) in [(u, v) for u, v, _ in added]
    assert (2, 3) in edges_aug or (3, 2) in edges_aug

    _, _, dist = added[0]
    assert dist == pytest.approx(98.0, rel=1e-6)

    edges_no, added_no = augment_connect_components_by_distance(
        gdf, "LocationID", strict_edges, max_dist=50.0
    )
    assert added_no == []
    assert edges_no == sorted(set(strict_edges))
