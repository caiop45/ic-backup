#!/usr/bin/env python3
"""
Build THT_PHYS_EDGES_CSV for THT-TripGen.

Output format:
    CSV with columns:
      u_idx, v_idx, u_location_id, v_location_id, edge_type, distance_m

Important:
    u_idx/v_idx are THTTripGenTransformer zone indices (NOT raw LocationID).
    You can either:
      1) load THT-TripGen split and fit THTTripGenTransformer on TRAIN, or
      2) pass a mappings JSON to map LocationID -> idx directly.

Edge types:
    - strict: adjacency from touches (queen) or boundary intersection (rook)
    - gap_tol: queen adjacency added due to distance <= gap_tol
    - bridge: edges added to connect disconnected components (distance-based)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Set, Tuple

import pandas as pd

try:
    import geopandas as gpd
except Exception as e:
    raise RuntimeError(
        "geopandas is required for this script. Install it (preferably via conda) "
        "or pip install geopandas."
    ) from e

import config
from data_processing.tht_tripgen_loader import load_and_split_tht_tripgen
from data_processing.tht_tripgen_transformer import THTTripGenTransformer

from topology.physical_adjacency import (
    augment_connect_components_by_distance,
    compute_edges_location_id,
    fix_geometries,
)


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

def _load_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _is_index_like(values: List[int]) -> bool:
    if not values:
        return False
    uniq = sorted(set(values))
    return uniq[0] == 0 and uniq[-1] == len(uniq) - 1 and len(uniq) == len(values)


def _parse_mapping_dict(data: Mapping[object, object]) -> Dict[int, int]:
    def _to_int_dict(obj: Mapping[object, object]) -> Dict[int, int]:
        out: Dict[int, int] = {}
        for k, v in obj.items():
            out[int(k)] = int(v)
        return out

    if "location_id_to_idx" in data:
        return _to_int_dict(data["location_id_to_idx"])
    if "zone_to_idx" in data:
        return _to_int_dict(data["zone_to_idx"])
    if "idx_to_location_id" in data:
        inv = _to_int_dict(data["idx_to_location_id"])
        return {v: k for k, v in inv.items()}
    if "idx_to_zone" in data:
        inv = _to_int_dict(data["idx_to_zone"])
        return {v: k for k, v in inv.items()}

    direct = _to_int_dict(data)
    keys = list(direct.keys())
    values = list(direct.values())

    keys_index_like = _is_index_like(keys)
    values_index_like = _is_index_like(values)

    if values_index_like and not keys_index_like:
        return {v: k for k, v in direct.items()}
    return direct


def _load_zone_to_idx(path: Path | None) -> Dict[int, int] | None:
    if path is None:
        return None
    data = _load_json(path)
    if isinstance(data, dict):
        return _parse_mapping_dict(data)
    raise ValueError("Unsupported mappings JSON format (expected dict)")


def _map_edges_to_index_with_metadata(
    edges_loc: Iterable[Tuple[int, int]],
    zone_to_idx: Dict[int, int],
    edge_type_map: Dict[Tuple[int, int], str],
    distance_map: Dict[Tuple[int, int], float | None],
) -> pd.DataFrame:
    rows: List[Tuple[int, int, int, int, str, float | None]] = []
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
        edge_key = (u_loc, v_loc) if u_loc < v_loc else (v_loc, u_loc)
        edge_type = edge_type_map.get(edge_key, "strict")
        dist = distance_map.get(edge_key)
        rows.append((u_idx, v_idx, u_loc, v_loc, edge_type, dist))

    if missing:
        print(
            f"[WARN] {len(missing)} LocationIDs were present in the polygons "
            f"but not in the THT-TripGen zone vocabulary (train split). "
            f"Examples: {sorted(list(missing))[:20]}"
        )

    df = pd.DataFrame(
        rows,
        columns=[
            "u_idx",
            "v_idx",
            "u_location_id",
            "v_location_id",
            "edge_type",
            "distance_m",
        ],
    )
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
        help="Output CSV path for THT-TripGen physical edges (index space).",
    )
    ap.add_argument(
        "--mappings",
        type=str,
        default=None,
        help="Optional JSON mapping for LocationID -> idx (skips THT-TripGen fit).",
    )
    ap.add_argument(
        "--split",
        type=str,
        default="S1",
        choices=["S1", "S2"],
        help="THT-TripGen split (must match how you'll train/build embeddings).",
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
    ap.add_argument(
        "--gap-tol",
        type=float,
        default=0.0,
        help="Gap tolerance for tolerant queen adjacency (CRS units).",
    )
    ap.add_argument(
        "--bridge-max-dist",
        type=float,
        default=0.0,
        help="Max distance for component-bridging augmentation (CRS units).",
    )
    ap.add_argument(
        "--adjacency-crs",
        choices=["none", "EPSG:3857"],
        default="none",
        help="CRS for adjacency computation (recommended EPSG:3857 when gap_tol/bridge enabled).",
    )
    ap.add_argument(
        "--fix-geoms",
        choices=["none", "buffer0", "make_valid_if_available"],
        default="make_valid_if_available",
        help="Geometry fixing mode.",
    )
    args = ap.parse_args()

    zones_path = Path(args.zones)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    zone_to_idx = _load_zone_to_idx(Path(args.mappings)) if args.mappings else None
    if zone_to_idx is None:
        # 1) Load THT-TripGen data & fit transformer on TRAIN
        train_df, _hold_df, _test_df = load_and_split_tht_tripgen(split=args.split)
        transformer = THTTripGenTransformer()
        transformer.fit(train_df)

        zone_to_idx = transformer.zone_to_idx
        zone_vocab = set(zone_to_idx.keys())
        num_nodes = transformer.num_zones
    else:
        zone_vocab = set(zone_to_idx.keys())
        num_nodes = max(zone_to_idx.values()) + 1 if zone_to_idx else 0

    # 2) Read polygons
    gdf = _read_zones(zones_path)
    loc_col = _get_location_id_column(gdf)

    # 3) Fix geometries and optionally reproject for adjacency distances
    gdf = fix_geometries(gdf, mode=args.fix_geoms)
    adjacency_crs = args.adjacency_crs
    if adjacency_crs == "none" and (args.gap_tol > 0 or args.bridge_max_dist > 0):
        adjacency_crs = "EPSG:3857"
        print("[INFO] adjacency_crs set to EPSG:3857 because gap_tol/bridge enabled")
    if adjacency_crs != "none":
        gdf = gdf.to_crs(adjacency_crs)

    # 4) Filter polygons to the THT-TripGen vocabulary (so indices line up)
    gdf = gdf[gdf[loc_col].astype("int64").isin(zone_vocab)].copy()
    if gdf.empty:
        raise RuntimeError(
            "After filtering polygons to the THT-TripGen zone vocabulary, "
            "no rows remained. Check that your dataset uses NYC TLC LocationIDs "
            "and that you loaded the correct polygons file."
        )
    # Some zone files contain multiple polygons per LocationID (e.g., islands).
    # Collapse to one geometry per LocationID so distance-based steps have a unique node.
    dup_mask = gdf[loc_col].duplicated()
    if dup_mask.any():
        dup_ids = sorted(gdf.loc[dup_mask, loc_col].astype("int64").unique().tolist())
        print(
            f"[WARN] Found {len(dup_ids)} duplicated LocationIDs in zones; "
            "dissolving to a single geometry per LocationID."
        )
        gdf = gdf[[loc_col, "geometry"]].dissolve(by=loc_col, as_index=False)

    # 5) Compute adjacency in LocationID space
    if args.adjacency == "rook":
        if args.gap_tol > 0:
            print("[WARN] gap_tol ignored for rook adjacency")
        edges_strict = compute_edges_location_id(
            gdf=gdf,
            loc_col=loc_col,
            mode="rook",
            tol=float(args.tol),
            prefilter="bbox",
            gap_tol=0.0,
        )
        edges_tolerant = edges_strict
        gap_edges: List[Tuple[int, int]] = []
    else:
        edges_strict = compute_edges_location_id(
            gdf=gdf,
            loc_col=loc_col,
            mode="queen",
            tol=float(args.tol),
            prefilter="bbox",
            gap_tol=0.0,
        )
        edges_tolerant = compute_edges_location_id(
            gdf=gdf,
            loc_col=loc_col,
            mode="queen",
            tol=float(args.tol),
            prefilter="bbox",
            gap_tol=float(args.gap_tol),
        )
        gap_edges = sorted(set(edges_tolerant) - set(edges_strict))

    if not edges_tolerant:
        raise RuntimeError(
            "No adjacency edges found. Something is wrong with polygons or filtering."
        )

    # 6) Optional component-bridging augmentation
    bridge_edges: List[Tuple[int, int, float]] = []
    edges_final = edges_tolerant
    if args.bridge_max_dist and args.bridge_max_dist > 0:
        edges_final, bridge_edges = augment_connect_components_by_distance(
            gdf=gdf,
            loc_col=loc_col,
            edges_loc=edges_tolerant,
            max_dist=float(args.bridge_max_dist),
        )

    # 7) Build edge metadata maps
    geom_map = {
        int(loc): geom for loc, geom in zip(gdf[loc_col].astype("int64").tolist(), gdf.geometry.tolist())
    }

    edge_type_map: Dict[Tuple[int, int], str] = {}
    distance_map: Dict[Tuple[int, int], float | None] = {}

    for u, v in edges_strict:
        edge_type_map[(u, v)] = "strict"
        distance_map[(u, v)] = None

    for u, v in gap_edges:
        edge_type_map[(u, v)] = "gap_tol"
        gu = geom_map.get(u)
        gv = geom_map.get(v)
        if gu is None or gv is None or gu.is_empty or gv.is_empty:
            distance_map[(u, v)] = None
        else:
            distance_map[(u, v)] = float(gu.distance(gv))

    for u, v, dist in bridge_edges:
        edge_type_map[(u, v)] = "bridge"
        distance_map[(u, v)] = float(dist)

    edges_df = _map_edges_to_index_with_metadata(
        edges_final,
        zone_to_idx=zone_to_idx,
        edge_type_map=edge_type_map,
        distance_map=distance_map,
    )

    # Validate index bounds
    if edges_df[["u_idx", "v_idx"]].max().max() >= num_nodes or edges_df[["u_idx", "v_idx"]].min().min() < 0:
        raise RuntimeError(
            "Index bounds check failed. This likely means you built edges with a different "
            "zone vocabulary than the transformer uses."
        )

    # 8) Save
    edges_df.to_csv(out_path, index=False)
    print(f"[OK] Wrote: {out_path}")
    _print_graph_stats(edges_df, num_nodes=num_nodes)


if __name__ == "__main__":
    main()
