"""Polygon adjacency utilities (rook/queen) and graph statistics.

Rook adjacency: polygons share a boundary segment. Operationally, the
boundary intersection length must be > tol.
Queen adjacency: polygons touch at any boundary point (corner-touch allowed).

Tol guidance:
- Geographic CRS (EPSG:4326): use a tiny tol such as 1e-9.
- Projected/metric CRS: use a tol in meters (e.g., 5.0).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Literal, Sequence, Tuple

import pandas as pd

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import geopandas as gpd
    from shapely.geometry.base import BaseGeometry


DEFAULT_ADJACENCY_MODE: Literal["rook", "queen"] = "rook"
DEFAULT_PREFILTER = "touches"
DEFAULT_TOL_DEGREES = 1e-9
DEFAULT_TOL_METERS = 5.0


@dataclass(frozen=True)
class DegreesSummary:
    min_degree: int
    max_degree: int
    mean_degree: float
    median_degree: float
    average_degree: float


@dataclass(frozen=True)
class ComponentsSummary:
    num_components: int
    largest_component_size: int
    smallest_component_size: int
    mean_component_size: float
    num_isolated: int


@dataclass(frozen=True)
class GraphStats:
    num_nodes: int
    num_edges: int
    degrees: List[int]
    degrees_summary: DegreesSummary
    isolated_nodes: List[int]
    components: List[List[int]]
    components_summary: ComponentsSummary


@dataclass(frozen=True)
class EdgeDiff:
    num_nodes: int
    num_edges_a: int
    num_edges_b: int
    edges_only_in_a: List[Tuple[int, int]]
    edges_only_in_b: List[Tuple[int, int]]
    degree_delta_df: pd.DataFrame


def read_zones(path: Path | str) -> "gpd.GeoDataFrame":
    """Read a zones file into a GeoDataFrame.

    Supports .shp (unpacked), .zip (zipped shapefile), .parquet (GeoParquet),
    and geojson/json via geopandas.
    """
    try:
        import geopandas as gpd
    except Exception as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "geopandas is required to read zones. Install it (conda preferred) "
            "or pip install geopandas."
        ) from exc

    zones_path = Path(path)
    suf = zones_path.suffix.lower()
    if suf == ".zip":
        return gpd.read_file(f"zip://{zones_path}")
    if suf == ".parquet":
        return gpd.read_parquet(zones_path)
    return gpd.read_file(zones_path)


def find_location_id_col(gdf: "gpd.GeoDataFrame") -> str:
    candidates = [
        "LocationID",
        "locationid",
        "location_id",
        "LOCATIONID",
        "locationId",
        "LOCATION_ID",
    ]
    for col in candidates:
        if col in gdf.columns:
            return col
    raise ValueError(
        "Could not find a LocationID column. "
        f"Columns found: {list(gdf.columns)}"
    )


def _get_make_valid_fn():
    try:
        from shapely import make_valid

        return make_valid
    except Exception:
        pass
    try:
        from shapely.validation import make_valid

        return make_valid
    except Exception:
        return None


def fix_geometries(gdf: "gpd.GeoDataFrame", mode: str = "none") -> "gpd.GeoDataFrame":
    """Fix invalid geometries with the selected mode.

    mode:
      - "none" (default): no changes
      - "buffer0": apply buffer(0) to each geometry
      - "make_valid_if_available": use shapely.make_valid when available
    """
    if mode == "none":
        return gdf

    gdf = gdf.copy()

    if mode == "buffer0":
        def _buffer0(geom: "BaseGeometry | None") -> "BaseGeometry | None":
            if geom is None or geom.is_empty:
                return geom
            return geom.buffer(0)

        gdf["geometry"] = gdf.geometry.apply(_buffer0)
        return gdf

    if mode == "make_valid_if_available":
        make_valid = _get_make_valid_fn()
        if make_valid is None:
            return gdf

        def _make_valid(geom: "BaseGeometry | None") -> "BaseGeometry | None":
            if geom is None or geom.is_empty:
                return geom
            return make_valid(geom)

        gdf["geometry"] = gdf.geometry.apply(_make_valid)
        return gdf

    raise ValueError(f"Unsupported geometry fix mode: {mode}")


def _bbox_intersects(bounds_a: Sequence[float], bounds_b: Sequence[float]) -> bool:
    minx_a, miny_a, maxx_a, maxy_a = bounds_a
    minx_b, miny_b, maxx_b, maxy_b = bounds_b
    if maxx_a < minx_b or maxx_b < minx_a:
        return False
    if maxy_a < miny_b or maxy_b < miny_a:
        return False
    return True


def _normalize_edge(u: int, v: int) -> Tuple[int, int]:
    return (u, v) if u < v else (v, u)


def _normalize_edges(
    edges: Iterable[Tuple[int, int]],
    num_nodes: int | None = None,
) -> List[Tuple[int, int]]:
    edge_set: set[Tuple[int, int]] = set()
    for u_raw, v_raw in edges:
        u = int(u_raw)
        v = int(v_raw)
        if u == v:
            continue
        if num_nodes is not None:
            if u < 0 or v < 0 or u >= num_nodes or v >= num_nodes:
                raise ValueError(f"Edge node out of range: ({u}, {v})")
        edge_set.add(_normalize_edge(u, v))
    return sorted(edge_set)


def compute_edges_location_id(
    gdf: "gpd.GeoDataFrame",
    loc_col: str,
    mode: Literal["rook", "queen"] = DEFAULT_ADJACENCY_MODE,
    tol: float = DEFAULT_TOL_DEGREES,
    *,
    prefilter: str = DEFAULT_PREFILTER,
) -> List[Tuple[int, int]]:
    """Compute adjacency edges in LocationID space.

    Returns edges as (LocationID_u, LocationID_v) with u < v.

    prefilter (rook only):
      - "touches": use geom.touches() as a fast rejection
      - "boundary_intersects": use boundary.intersects() as a fast rejection
      - "bbox": use bounding box intersection as a fast rejection
      - "none": no prefilter
    """
    if loc_col not in gdf.columns:
        raise ValueError(f"LocationID column '{loc_col}' not found")

    gdf = gdf.reset_index(drop=True)
    loc_series = gdf[loc_col]
    if loc_series.isnull().any():
        raise ValueError(f"LocationID column '{loc_col}' contains nulls")

    loc_ids = [int(v) for v in loc_series.tolist()]
    geoms = gdf.geometry.tolist()

    edges: set[Tuple[int, int]] = set()
    n = len(geoms)

    for i in range(n):
        gi = geoms[i]
        if gi is None or gi.is_empty:
            continue
        for j in range(i + 1, n):
            gj = geoms[j]
            if gj is None or gj.is_empty:
                continue

            u_loc = int(loc_ids[i])
            v_loc = int(loc_ids[j])
            if u_loc == v_loc:
                continue

            if mode == "queen":
                if gi.touches(gj):
                    edges.add(_normalize_edge(u_loc, v_loc))
                continue

            if mode != "rook":
                raise ValueError(f"Unsupported adjacency mode: {mode}")

            if prefilter == "touches":
                if not gi.touches(gj):
                    continue
            elif prefilter == "boundary_intersects":
                if not gi.boundary.intersects(gj.boundary):
                    continue
            elif prefilter == "bbox":
                if not _bbox_intersects(gi.bounds, gj.bounds):
                    continue
            elif prefilter == "none":
                pass
            else:
                raise ValueError(f"Unsupported prefilter: {prefilter}")

            inter = gi.boundary.intersection(gj.boundary)
            if (not inter.is_empty) and float(inter.length) > tol:
                edges.add(_normalize_edge(u_loc, v_loc))

    return sorted(edges)


def edges_to_degrees(
    num_nodes: int, edges: Iterable[Tuple[int, int]]
) -> List[int]:
    if num_nodes <= 0:
        raise ValueError("num_nodes must be positive")

    deg = [0] * num_nodes
    for u, v in _normalize_edges(edges, num_nodes=num_nodes):
        if u == v:
            continue
        deg[u] += 1
        deg[v] += 1
    return deg


def connected_components(
    num_nodes: int, edges: Iterable[Tuple[int, int]]
) -> List[List[int]]:
    if num_nodes <= 0:
        raise ValueError("num_nodes must be positive")

    adjacency: List[set[int]] = [set() for _ in range(num_nodes)]
    for u, v in _normalize_edges(edges, num_nodes=num_nodes):
        if u == v:
            continue
        adjacency[u].add(v)
        adjacency[v].add(u)

    visited = [False] * num_nodes
    components: List[List[int]] = []

    for node in range(num_nodes):
        if visited[node]:
            continue
        stack = [node]
        visited[node] = True
        comp: List[int] = []
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nbr in adjacency[cur]:
                if not visited[nbr]:
                    visited[nbr] = True
                    stack.append(nbr)
        comp.sort()
        components.append(comp)

    components.sort(key=lambda c: (-len(c), c))
    return components


def _median(sorted_vals: Sequence[int]) -> float:
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2 == 1:
        return float(sorted_vals[mid])
    return (sorted_vals[mid - 1] + sorted_vals[mid]) / 2.0


def compute_graph_stats(
    num_nodes: int, edges: Iterable[Tuple[int, int]]
) -> GraphStats:
    if num_nodes <= 0:
        raise ValueError("num_nodes must be positive")

    edges_norm = _normalize_edges(edges, num_nodes=num_nodes)
    degrees = edges_to_degrees(num_nodes, edges_norm)
    num_edges = len(edges_norm)

    sorted_deg = sorted(degrees)
    mean_degree = sum(degrees) / float(num_nodes)
    average_degree = 0.0
    if num_nodes > 0:
        average_degree = (2.0 * float(num_edges)) / float(num_nodes)

    degrees_summary = DegreesSummary(
        min_degree=int(sorted_deg[0]) if sorted_deg else 0,
        max_degree=int(sorted_deg[-1]) if sorted_deg else 0,
        mean_degree=float(mean_degree),
        median_degree=_median(sorted_deg),
        average_degree=float(average_degree),
    )

    isolated_nodes = [i for i, d in enumerate(degrees) if d == 0]

    components = connected_components(num_nodes, edges_norm)
    sizes = [len(c) for c in components]

    if sizes:
        mean_size = sum(sizes) / float(len(sizes))
        largest = max(sizes)
        smallest = min(sizes)
    else:
        mean_size = 0.0
        largest = 0
        smallest = 0

    components_summary = ComponentsSummary(
        num_components=len(components),
        largest_component_size=int(largest),
        smallest_component_size=int(smallest),
        mean_component_size=float(mean_size),
        num_isolated=len(isolated_nodes),
    )

    return GraphStats(
        num_nodes=num_nodes,
        num_edges=num_edges,
        degrees=degrees,
        degrees_summary=degrees_summary,
        isolated_nodes=isolated_nodes,
        components=components,
        components_summary=components_summary,
    )


def diff_edges(
    edges_a: Iterable[Tuple[int, int]],
    edges_b: Iterable[Tuple[int, int]],
    num_nodes: int,
) -> EdgeDiff:
    if num_nodes <= 0:
        raise ValueError("num_nodes must be positive")

    edges_a_norm = _normalize_edges(edges_a, num_nodes=num_nodes)
    edges_b_norm = _normalize_edges(edges_b, num_nodes=num_nodes)

    set_a = set(edges_a_norm)
    set_b = set(edges_b_norm)

    edges_only_in_a = sorted(set_a - set_b)
    edges_only_in_b = sorted(set_b - set_a)

    deg_a = edges_to_degrees(num_nodes, edges_a_norm)
    deg_b = edges_to_degrees(num_nodes, edges_b_norm)

    degree_delta_df = pd.DataFrame(
        {
            "node": list(range(num_nodes)),
            "degree_a": deg_a,
            "degree_b": deg_b,
        }
    )
    degree_delta_df["delta"] = degree_delta_df["degree_b"] - degree_delta_df["degree_a"]

    return EdgeDiff(
        num_nodes=num_nodes,
        num_edges_a=len(edges_a_norm),
        num_edges_b=len(edges_b_norm),
        edges_only_in_a=edges_only_in_a,
        edges_only_in_b=edges_only_in_b,
        degree_delta_df=degree_delta_df,
    )
