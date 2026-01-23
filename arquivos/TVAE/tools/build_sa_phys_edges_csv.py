#!/usr/bin/env python3
"""
Build SA_PHYS_EDGES_CSV for Strategy A.

Output format:
    CSV with at least 2 columns: u_idx, v_idx
    (Optionally includes debug columns u_location_id, v_location_id)

Important:
    u_idx/v_idx are StrategyATransformer zone indices (NOT raw LocationID).
    So we:
      1) load Strategy A split
      2) fit StrategyATransformer on TRAIN
      3) compute polygon adjacency (rook by default)
      4) map LocationID -> idx
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

import pandas as pd

try:
    import geopandas as gpd
except Exception as e:
    raise RuntimeError(
        "geopandas is required for this script. Install it (preferably via conda) "
        "or pip install geopandas."
    ) from e

import config
from data_processing.strategy_a_loader import load_and_split_strategy_a
from data_processing.strategy_a_transformer import StrategyATransformer


def _read_zones(zones_path: Path) -> "gpd.GeoDataFrame":
    """
    Supports:
      - .shp (unpacked shapefile)
      - .zip (zipped shapefile)
      - .parquet (GeoParquet via geopandas)
      - .geojson/.json (geojson)
    """
    suf = zones_path.suffix.lower()
    if suf == ".zip":
        # geopandas/fiona can read zipped shapefiles via the zip:// URI.
        # Many zips contain exactly one shapefile; if not, unzip and point to .shp.
        return gpd.read_file(f"zip://{zones_path}")
    if suf == ".parquet":
        return gpd.read_parquet(zones_path)
    return gpd.read_file(zones_path)


def _get_location_id_column(gdf: "gpd.GeoDataFrame") -> str:
    for c in ["LocationID", "locationid", "location_id", "LOCATIONID"]:
        if c in gdf.columns:
            return c
    raise ValueError(
        "Could not find a LocationID column in the zones file. "
        f"Columns found: {list(gdf.columns)}"
    )


def _compute_adjacency_edges_location_id(
    gdf: "gpd.GeoDataFrame",
    loc_col: str,
    adjacency: str,
    tol: float,
) -> List[Tuple[int, int]]:
    """
    Returns edges as (LocationID_u, LocationID_v), with u < v.

    adjacency:
      - 'rook': share a boundary segment (intersection length > tol)
      - 'queen': touch at any boundary point (corner-touch allowed)
    """
    # Ensure clean indexing
    gdf = gdf.reset_index(drop=True)

    loc_ids = gdf[loc_col].astype("int64").tolist()
    geoms = gdf.geometry.tolist()

    n = len(geoms)
    edges: List[Tuple[int, int]] = []

    for i in range(n):
        gi = geoms[i]
        if gi is None or gi.is_empty:
            continue
        for j in range(i + 1, n):
            gj = geoms[j]
            if gj is None or gj.is_empty:
                continue

            # Fast rejection: if they don't even touch, skip.
            # touches() catches boundary-contact (including corner).
            if not gi.touches(gj):
                continue

            if adjacency == "queen":
                edges.append((int(loc_ids[i]), int(loc_ids[j])))
                continue

            # rook: require a shared boundary segment (not just a single point)
            inter = gi.boundary.intersection(gj.boundary)
            if (not inter.is_empty) and (float(inter.length) > tol):
                edges.append((int(loc_ids[i]), int(loc_ids[j])))

    # Deduplicate + sort
    edges = sorted(set(edges))
    return edges


def _map_location_id_edges_to_index(
    edges_loc: Iterable[Tuple[int, int]],
    zone_to_idx: Dict[int, int],
) -> pd.DataFrame:
    rows: List[Tuple[int, int, int, int]] = []
    missing: Set[int] = set()

    for u_loc, v_loc in edges_loc:
        if u_loc not in zone_to_idx:
            missing.add(u_loc)
            continue
        if v_loc not in zone_to_idx:
            missing.add(v_loc)
            continue
        u_idx = int(zone_to_idx[u_loc])
        v_idx = int(zone_to_idx[v_loc])
        if u_idx == v_idx:
            continue
        rows.append((u_idx, v_idx, u_loc, v_loc))

    if missing:
        print(
            f"[WARN] {len(missing)} LocationIDs were present in the polygons "
            f"but not in the Strategy A zone vocabulary (train split). "
            f"Examples: {sorted(list(missing))[:20]}"
        )

    df = pd.DataFrame(rows, columns=["u_idx", "v_idx", "u_location_id", "v_location_id"])
    df = df.drop_duplicates().sort_values(["u_idx", "v_idx"]).reset_index(drop=True)
    return df


def _print_graph_stats(edges_df: pd.DataFrame, num_nodes: int) -> None:
    deg = [0] * num_nodes
    for u, v in edges_df[["u_idx", "v_idx"]].itertuples(index=False, name=None):
        if 0 <= u < num_nodes and 0 <= v < num_nodes:
            deg[u] += 1
            deg[v] += 1
    iso = sum(1 for d in deg if d == 0)
    print(f"[INFO] num_nodes={num_nodes}")
    print(f"[INFO] undirected edges (unique pairs)={len(edges_df)}")
    print(f"[INFO] isolated nodes={iso}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--zones",
        type=str,
        required=True,
        help="Path to taxi zones polygons: taxi_zones.shp | taxi_zones.zip | taxi_zones.parquet | zones.geojson",
    )
    ap.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output CSV path for SA physical edges (index space).",
    )
    ap.add_argument(
        "--split",
        type=str,
        default="S1",
        choices=["S1", "S2"],
        help="Strategy A split (must match how you'll train/build embeddings).",
    )
    ap.add_argument(
        "--adjacency",
        type=str,
        default="rook",
        choices=["rook", "queen"],
        help="rook = shared boundary segment; queen = any touch (corner-touch allowed).",
    )
    ap.add_argument(
        "--tol",
        type=float,
        default=1e-9,
        help="Length threshold for rook adjacency boundary intersection.",
    )
    args = ap.parse_args()

    zones_path = Path(args.zones)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 1) Load Strategy A data & fit transformer on TRAIN
    train_df, _hold_df, _test_df = load_and_split_strategy_a(split=args.split)
    transformer = StrategyATransformer()
    transformer.fit(train_df)

    zone_to_idx = transformer.zone_to_idx
    zone_vocab = set(zone_to_idx.keys())
    num_nodes = transformer.num_zones

    # 2) Read polygons
    gdf = _read_zones(zones_path)
    loc_col = _get_location_id_column(gdf)

    # 3) Filter polygons to the Strategy A vocabulary (so indices line up)
    gdf = gdf[gdf[loc_col].astype("int64").isin(zone_vocab)].copy()
    if gdf.empty:
        raise RuntimeError(
            "After filtering polygons to the Strategy A zone vocabulary, "
            "no rows remained. Check that your dataset uses NYC TLC LocationIDs "
            "and that you loaded the correct polygons file."
        )

    # 4) Compute adjacency in LocationID space
    edges_loc = _compute_adjacency_edges_location_id(
        gdf=gdf,
        loc_col=loc_col,
        adjacency=args.adjacency,
        tol=float(args.tol),
    )
    if not edges_loc:
        raise RuntimeError("No adjacency edges found. Something is wrong with polygons or filtering.")

    # 5) Map to index space
    edges_df = _map_location_id_edges_to_index(edges_loc, zone_to_idx=zone_to_idx)

    # Validate index bounds
    if edges_df[["u_idx", "v_idx"]].max().max() >= num_nodes or edges_df[["u_idx", "v_idx"]].min().min() < 0:
        raise RuntimeError(
            "Index bounds check failed. This likely means you built edges with a different "
            "zone vocabulary than the transformer uses."
        )

    # 6) Save
    edges_df.to_csv(out_path, index=False)
    print(f"[OK] Wrote: {out_path}")
    _print_graph_stats(edges_df, num_nodes=num_nodes)


if __name__ == "__main__":
    main()