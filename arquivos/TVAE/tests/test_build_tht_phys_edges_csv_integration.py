import json
import sys

import pytest


gpd = pytest.importorskip("geopandas")
pytest.importorskip("shapely")

import pandas as pd
from shapely.geometry import Polygon

import tools.build_tht_phys_edges_csv as builder


def test_build_tht_phys_edges_csv_with_bridge(tmp_path, monkeypatch):
    polygons = [
        Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
        Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
        Polygon([(100, 0), (101, 0), (101, 1), (100, 1)]),
    ]
    gdf = gpd.GeoDataFrame(
        {"LocationID": [1, 2, 3], "geometry": polygons}, crs="EPSG:3857"
    )

    zones_path = tmp_path / "zones.geojson"
    gdf.to_file(zones_path, driver="GeoJSON")

    mapping_path = tmp_path / "mapping.json"
    mapping_path.write_text(
        json.dumps({"location_id_to_idx": {"1": 0, "2": 1, "3": 2}}),
        encoding="utf-8",
    )

    out_csv = tmp_path / "edges.csv"

    argv = [
        "prog",
        "--zones",
        str(zones_path),
        "--out",
        str(out_csv),
        "--mappings",
        str(mapping_path),
        "--adjacency",
        "queen",
        "--gap-tol",
        "0.0",
        "--bridge-max-dist",
        "200.0",
        "--adjacency-crs",
        "EPSG:3857",
        "--fix-geoms",
        "none",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    builder.main()

    assert out_csv.exists()
    df = pd.read_csv(out_csv)
    expected_cols = {
        "u_idx",
        "v_idx",
        "u_location_id",
        "v_location_id",
        "edge_type",
        "distance_m",
    }
    assert expected_cols.issubset(df.columns)

    edge_types = set(df["edge_type"].tolist())
    assert "strict" in edge_types
    assert "bridge" in edge_types
