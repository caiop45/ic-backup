import pytest


gpd = pytest.importorskip("geopandas")
pytest.importorskip("shapely")
pytest.importorskip("matplotlib")

import matplotlib

matplotlib.use("Agg")

from shapely.geometry import Polygon

from tvae.topology.viz_phys_graph import (
    RenderConfig,
    compute_degrees_for_locations,
    render_debug_zone_map,
    render_degree_map,
)


def test_viz_render_smoke(tmp_path):
    polygons = [
        Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
        Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
        Polygon([(0, 1), (1, 1), (1, 2), (0, 2)]),
        Polygon([(1, 1), (2, 1), (2, 2), (1, 2)]),
    ]
    gdf = gpd.GeoDataFrame(
        {"LocationID": [1, 2, 3, 4], "geometry": polygons}, crs="EPSG:3857"
    )

    edges = [(1, 2), (1, 3), (2, 4), (3, 4)]
    location_ids = [1, 2, 3, 4]
    degrees = compute_degrees_for_locations(location_ids, edges)

    config = RenderConfig(
        figsize=(6.0, 6.0),
        dpi=200,
        poly_alpha=0.6,
        edge_alpha=0.4,
        edge_width=0.5,
        label=False,
        label_field="LocationID",
        label_fontsize=6,
        basemap="off",
        basemap_provider="CartoDB.Positron",
    )

    out_path = tmp_path / "map.png"
    render_degree_map(
        gdf,
        "LocationID",
        location_ids=location_ids,
        degrees=degrees,
        edges=edges,
        near_miss_edges=None,
        title="Smoke",
        out_path=out_path,
        config=config,
        edge_mode="all",
        stats_text="nodes=4 edges=4 avg_degree=2.00 isolated=0",
        edge_style="strict",
    )

    assert out_path.exists()
    assert out_path.stat().st_size > 10_000

    debug_path = tmp_path / "debug_zone_1.png"
    degrees_map = {loc: deg for loc, deg in zip(location_ids, degrees)}
    render_debug_zone_map(
        gdf,
        "LocationID",
        zone_id=1,
        candidate_ids=location_ids,
        location_ids=location_ids,
        degrees_map=degrees_map,
        strict_edges=edges,
        tolerant_edges=edges,
        near_miss_edges=[(1, 2, 0.05)],
        out_path=debug_path,
        config=config,
        edge_style="both",
        title="Debug zone 1",
        show_labels=True,
    )

    assert debug_path.exists()
    assert debug_path.stat().st_size > 10_000
