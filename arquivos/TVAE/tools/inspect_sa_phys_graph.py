#!/usr/bin/env python3
"""Inspect physical adjacency graphs for taxi zones.

Examples:
  # Compute rook/queen from polygons (EPSG:4326)
  python tools/inspect_sa_phys_graph.py \
    --zones data/taxi_zones.zip \
    --out-dir outputs/phys_graph

  # Compute in a metric CRS with 5m tol (default)
  python tools/inspect_sa_phys_graph.py \
    --zones data/taxi_zones.zip \
    --out-dir outputs/phys_graph_metric \
    --crs-metric EPSG:3857

  # Load from precomputed edge CSVs
  python tools/inspect_sa_phys_graph.py \
    --zones data/taxi_zones.zip \
    --out-dir outputs/phys_graph_loaded \
    --mode load \
    --rook-edges data/sa_phys_edges_rook.csv \
    --queen-edges data/sa_phys_edges_queen.csv \
    --mappings data/zone_mappings.json
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Literal, Mapping, Sequence, TYPE_CHECKING

import pandas as pd

from topology.physical_adjacency import (
    GraphStats,
    EdgeDiff,
    compute_edges_location_id,
    compute_graph_stats,
    diff_edges,
    find_location_id_col,
    fix_geometries,
    read_zones,
)

if TYPE_CHECKING:
    import geopandas as gpd


Prefilter = Literal["touches", "boundary_intersects", "bbox", "none"]
Mode = Literal["compute", "load"]


def _ensure_out_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _degree_percentiles(degrees: Sequence[int]) -> dict[str, float]:
    series = pd.Series(degrees, dtype="float64")
    percentiles = [0, 5, 10, 25, 50, 75, 90, 95, 99, 100]
    return {f"p{p}": float(series.quantile(p / 100.0)) for p in percentiles}


def _graph_stats_to_dict(stats: GraphStats) -> dict[str, object]:
    return {
        "num_nodes": stats.num_nodes,
        "num_edges": stats.num_edges,
        "degrees_summary": asdict(stats.degrees_summary),
        "degree_percentiles": _degree_percentiles(stats.degrees),
        "isolated_nodes": stats.isolated_nodes,
        "components_count": stats.components_summary.num_components,
        "components_summary": asdict(stats.components_summary),
        "components": stats.components,
    }


def _diff_stats_to_dict(diff: EdgeDiff) -> dict[str, object]:
    delta_series = diff.degree_delta_df["delta"].astype("float64")
    return {
        "num_nodes": diff.num_nodes,
        "num_edges_rook": diff.num_edges_a,
        "num_edges_queen": diff.num_edges_b,
        "edges_only_in_rook": len(diff.edges_only_in_a),
        "edges_only_in_queen": len(diff.edges_only_in_b),
        "delta_degree_summary": {
            "min": float(delta_series.min()),
            "max": float(delta_series.max()),
            "mean": float(delta_series.mean()),
        },
        "delta_degree_percentiles": {
            f"p{p}": float(delta_series.quantile(p / 100.0))
            for p in [0, 5, 10, 25, 50, 75, 90, 95, 99, 100]
        },
    }


def _neighbors_from_edges(
    location_ids: Sequence[int],
    edges: Iterable[tuple[int, int]],
) -> dict[str, list[int]]:
    neighbors: dict[int, set[int]] = {int(loc): set() for loc in location_ids}
    for u, v in edges:
        u_int = int(u)
        v_int = int(v)
        if u_int == v_int:
            continue
        if u_int in neighbors:
            neighbors[u_int].add(v_int)
        if v_int in neighbors:
            neighbors[v_int].add(u_int)

    return {str(k): sorted(list(v)) for k, v in neighbors.items()}


def _edges_df(edges: Iterable[tuple[int, int]]) -> pd.DataFrame:
    rows = []
    for u, v in edges:
        u_int = int(u)
        v_int = int(v)
        if u_int == v_int:
            continue
        if u_int < v_int:
            rows.append((u_int, v_int))
        else:
            rows.append((v_int, u_int))
    df = pd.DataFrame(rows, columns=["u_location_id", "v_location_id"])
    return df.sort_values(["u_location_id", "v_location_id"]).reset_index(drop=True)


def _degrees_df(location_ids: Sequence[int], degrees: Sequence[int]) -> pd.DataFrame:
    rows = [(int(loc), int(deg)) for loc, deg in zip(location_ids, degrees)]
    df = pd.DataFrame(rows, columns=["location_id", "degree"])
    return df.sort_values("location_id").reset_index(drop=True)


def _load_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _is_index_like(values: Sequence[int]) -> bool:
    if not values:
        return False
    uniq = sorted(set(values))
    return uniq[0] == 0 and uniq[-1] == len(uniq) - 1 and len(uniq) == len(values)


def _parse_mapping_dict(data: Mapping[object, object]) -> dict[int, int]:
    def _to_int_dict(obj: Mapping[object, object]) -> dict[int, int]:
        out: dict[int, int] = {}
        for k, v in obj.items():
            out[int(k)] = int(v)
        return out

    if "idx_to_location_id" in data:
        return _to_int_dict(data["idx_to_location_id"])
    if "idx_to_zone" in data:
        return _to_int_dict(data["idx_to_zone"])
    if "location_id_to_idx" in data:
        inv = _to_int_dict(data["location_id_to_idx"])
        return {v: k for k, v in inv.items()}
    if "zone_to_idx" in data:
        inv = _to_int_dict(data["zone_to_idx"])
        return {v: k for k, v in inv.items()}

    direct = _to_int_dict(data)
    keys = list(direct.keys())
    values = list(direct.values())

    keys_index_like = _is_index_like(keys)
    values_index_like = _is_index_like(values)

    if values_index_like and not keys_index_like:
        return {v: k for k, v in direct.items()}
    return direct


def load_idx_to_location_id(path: Path | None) -> dict[int, int] | None:
    if path is None:
        return None
    data = _load_json(Path(path))
    if isinstance(data, list):
        return {i: int(v) for i, v in enumerate(data)}
    if isinstance(data, dict):
        return _parse_mapping_dict(data)
    raise ValueError("Unsupported mappings JSON format")


def _extract_edges_from_csv(
    path: Path,
    idx_to_location_id: dict[int, int] | None,
) -> list[tuple[int, int]]:
    df = pd.read_csv(path)
    if df.shape[1] < 2:
        raise ValueError(f"Edges CSV {path} must have at least two columns")

    if {"u_location_id", "v_location_id"}.issubset(df.columns):
        edge_df = df[["u_location_id", "v_location_id"]].dropna()
        edges = edge_df.astype("int64").itertuples(index=False, name=None)
        edges_loc: set[tuple[int, int]] = set()
        for u, v in edges:
            u_int = int(u)
            v_int = int(v)
            if u_int == v_int:
                continue
            if u_int < v_int:
                edges_loc.add((u_int, v_int))
            else:
                edges_loc.add((v_int, u_int))
        return sorted(edges_loc)

    if idx_to_location_id is None:
        raise ValueError(
            f"Edges CSV {path} lacks u_location_id/v_location_id columns. "
            "Provide --mappings to map idx to LocationID."
        )

    if {"u_idx", "v_idx"}.issubset(df.columns):
        edge_df = df[["u_idx", "v_idx"]].dropna()
    else:
        edge_df = df.iloc[:, :2].dropna()

    edges_idx = edge_df.astype("int64").itertuples(index=False, name=None)

    missing: set[int] = set()
    edges_loc: set[tuple[int, int]] = set()
    for u_idx, v_idx in edges_idx:
        if int(u_idx) not in idx_to_location_id:
            missing.add(int(u_idx))
            continue
        if int(v_idx) not in idx_to_location_id:
            missing.add(int(v_idx))
            continue
        u_loc = int(idx_to_location_id[int(u_idx)])
        v_loc = int(idx_to_location_id[int(v_idx)])
        if u_loc != v_loc:
            if u_loc < v_loc:
                edges_loc.add((u_loc, v_loc))
            else:
                edges_loc.add((v_loc, u_loc))

    if missing:
        print(
            f"[WARN] {len(missing)} idx values from {path} were missing in mappings. "
            f"Examples: {sorted(list(missing))[:10]}"
        )

    return sorted(edges_loc)


def _location_ids_from_gdf(gdf: "gpd.GeoDataFrame") -> list[int]:
    loc_col = find_location_id_col(gdf)
    loc_series = gdf[loc_col].dropna().astype("int64")
    return sorted(set(int(v) for v in loc_series.tolist()))


def _map_edges_to_indices(
    edges_loc: Iterable[tuple[int, int]],
    loc_to_idx: dict[int, int],
) -> tuple[list[tuple[int, int]], list[int]]:
    missing: set[int] = set()
    edges_idx: set[tuple[int, int]] = set()
    for u_loc, v_loc in edges_loc:
        if u_loc not in loc_to_idx:
            missing.add(int(u_loc))
            continue
        if v_loc not in loc_to_idx:
            missing.add(int(v_loc))
            continue
        u_idx = int(loc_to_idx[u_loc])
        v_idx = int(loc_to_idx[v_loc])
        if u_idx == v_idx:
            continue
        if u_idx < v_idx:
            edges_idx.add((u_idx, v_idx))
        else:
            edges_idx.add((v_idx, u_idx))
    return sorted(edges_idx), sorted(missing)


def _print_top_low_degree(
    degrees_df: pd.DataFrame,
    top_k: int,
) -> None:
    if top_k <= 0:
        return
    lowest = degrees_df.sort_values(["degree", "location_id"]).head(top_k)
    rows = ", ".join(
        f"{row.location_id}:{row.degree}" for row in lowest.itertuples(index=False)
    )
    print(f"[TOP] Low degree (location_id:degree): {rows}")


def _print_top_delta(degree_delta_df: pd.DataFrame, top_k: int) -> None:
    if top_k <= 0:
        return
    df = degree_delta_df.copy()
    df["abs_delta"] = df["delta"].abs()
    df = df.sort_values(
        ["abs_delta", "delta", "location_id"], ascending=[False, False, True]
    )
    top = df.head(top_k)
    rows = ", ".join(
        f"{row.location_id}:{row.delta}" for row in top.itertuples(index=False)
    )
    print(f"[TOP] Delta degree (location_id:delta): {rows}")


def inspect_physical_graphs(
    gdf: "gpd.GeoDataFrame",
    out_dir: Path,
    *,
    mode: Mode,
    tol: float,
    prefilter: Prefilter,
    fix_geoms: str,
    crs_metric: str,
    rook_edges_path: Path | None,
    queen_edges_path: Path | None,
    mappings_path: Path | None,
    top_k: int,
) -> None:
    gdf = fix_geometries(gdf, mode=fix_geoms)
    if crs_metric != "none":
        gdf = gdf.to_crs(crs_metric)

    loc_col = find_location_id_col(gdf)
    location_ids = _location_ids_from_gdf(gdf)
    if not location_ids:
        raise ValueError("No LocationIDs found in zones file")
    loc_to_idx = {loc: i for i, loc in enumerate(location_ids)}
    num_nodes = len(location_ids)

    idx_to_location_id = load_idx_to_location_id(mappings_path)

    if mode == "compute":
        edges_rook = compute_edges_location_id(
            gdf, loc_col, mode="rook", tol=tol, prefilter=prefilter
        )
        edges_queen = compute_edges_location_id(
            gdf, loc_col, mode="queen", tol=tol, prefilter="touches"
        )
    else:
        if rook_edges_path is None or queen_edges_path is None:
            raise ValueError("--rook-edges and --queen-edges are required in load mode")
        edges_rook = _extract_edges_from_csv(rook_edges_path, idx_to_location_id)
        edges_queen = _extract_edges_from_csv(queen_edges_path, idx_to_location_id)

    edges_rook_idx, missing_rook = _map_edges_to_indices(edges_rook, loc_to_idx)
    edges_queen_idx, missing_queen = _map_edges_to_indices(edges_queen, loc_to_idx)

    if missing_rook:
        print(
            f"[WARN] {len(missing_rook)} LocationIDs in rook edges not found in zones. "
            f"Examples: {missing_rook[:10]}"
        )
    if missing_queen:
        print(
            f"[WARN] {len(missing_queen)} LocationIDs in queen edges not found in zones. "
            f"Examples: {missing_queen[:10]}"
        )

    idx_to_loc = {idx: loc for loc, idx in loc_to_idx.items()}
    edges_rook_loc = sorted((idx_to_loc[u], idx_to_loc[v]) for u, v in edges_rook_idx)
    edges_queen_loc = sorted((idx_to_loc[u], idx_to_loc[v]) for u, v in edges_queen_idx)
    edges_only_in_queen = sorted(set(edges_queen_loc) - set(edges_rook_loc))
    edges_only_in_rook = sorted(set(edges_rook_loc) - set(edges_queen_loc))

    stats_rook = compute_graph_stats(num_nodes=num_nodes, edges=edges_rook_idx)
    stats_queen = compute_graph_stats(num_nodes=num_nodes, edges=edges_queen_idx)
    diff = diff_edges(edges_rook_idx, edges_queen_idx, num_nodes=num_nodes)

    _ensure_out_dir(out_dir)

    (out_dir / "stats_rook.json").write_text(
        json.dumps(_graph_stats_to_dict(stats_rook), indent=2),
        encoding="utf-8",
    )
    (out_dir / "stats_queen.json").write_text(
        json.dumps(_graph_stats_to_dict(stats_queen), indent=2),
        encoding="utf-8",
    )
    (out_dir / "stats_diff.json").write_text(
        json.dumps(_diff_stats_to_dict(diff), indent=2),
        encoding="utf-8",
    )

    degrees_rook = _degrees_df(location_ids, stats_rook.degrees)
    degrees_queen = _degrees_df(location_ids, stats_queen.degrees)

    degrees_rook.to_csv(out_dir / "degrees_rook.csv", index=False)
    degrees_queen.to_csv(out_dir / "degrees_queen.csv", index=False)

    degree_delta_df = diff.degree_delta_df.copy()
    degree_delta_df["location_id"] = [
        location_ids[int(i)] for i in degree_delta_df["node"]
    ]
    degree_delta_df = degree_delta_df[[
        "location_id",
        "degree_a",
        "degree_b",
        "delta",
    ]].rename(
        columns={
            "degree_a": "degree_rook",
            "degree_b": "degree_queen",
        }
    )
    degree_delta_df.to_csv(out_dir / "degrees_delta.csv", index=False)

    _edges_df(edges_rook_loc).to_csv(out_dir / "edges_rook.csv", index=False)
    _edges_df(edges_queen_loc).to_csv(out_dir / "edges_queen.csv", index=False)
    _edges_df(edges_only_in_queen).to_csv(
        out_dir / "edges_only_in_queen.csv", index=False
    )
    _edges_df(edges_only_in_rook).to_csv(
        out_dir / "edges_only_in_rook.csv", index=False
    )

    (out_dir / "neighbors_rook.json").write_text(
        json.dumps(_neighbors_from_edges(location_ids, edges_rook_loc), indent=2),
        encoding="utf-8",
    )
    (out_dir / "neighbors_queen.json").write_text(
        json.dumps(_neighbors_from_edges(location_ids, edges_queen_loc), indent=2),
        encoding="utf-8",
    )

    print(f"[INFO] num_nodes={num_nodes}")
    print(f"[INFO] rook edges={stats_rook.num_edges}, queen edges={stats_queen.num_edges}")
    print(f"[INFO] rook isolated={len(stats_rook.isolated_nodes)}, queen isolated={len(stats_queen.isolated_nodes)}")
    print(
        f"[INFO] diff edges only in queen={len(edges_only_in_queen)}, only in rook={len(edges_only_in_rook)}"
    )

    _print_top_low_degree(degrees_rook, top_k)
    _print_top_delta(degree_delta_df, top_k)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--zones", type=str, required=True, help="Path to zones polygons file")
    ap.add_argument("--out-dir", type=str, required=True, help="Output directory")
    ap.add_argument("--mode", choices=["compute", "load"], default="compute")
    ap.add_argument("--rook-edges", type=str, default=None, help="Path to rook edges CSV")
    ap.add_argument("--queen-edges", type=str, default=None, help="Path to queen edges CSV")
    ap.add_argument(
        "--mappings",
        type=str,
        default=None,
        help="JSON mapping for idx<->LocationID (required if CSV lacks LocationID columns)",
    )
    ap.add_argument(
        "--tol",
        type=float,
        default=None,
        help="Length threshold for rook adjacency. Default 1e-9 for geographic, 5.0 for metric CRS.",
    )
    ap.add_argument(
        "--prefilter",
        choices=["touches", "boundary_intersects", "bbox", "none"],
        default="touches",
        help="Prefilter used for rook adjacency",
    )
    ap.add_argument(
        "--fix-geoms",
        choices=["none", "buffer0", "make_valid_if_available"],
        default="none",
        help="Geometry fixing mode",
    )
    ap.add_argument(
        "--crs-metric",
        choices=["none", "EPSG:3857"],
        default="none",
        help="If set, compute adjacency in metric CRS (tol interpreted in meters)",
    )
    ap.add_argument(
        "--top-k-problems",
        type=int,
        default=25,
        help="Show top zones by low degree and delta degree",
    )
    return ap.parse_args()


def main() -> None:
    args = _parse_args()

    tol = args.tol
    if tol is None:
        tol = 5.0 if args.crs_metric != "none" else 1e-9

    gdf = read_zones(args.zones)
    inspect_physical_graphs(
        gdf,
        Path(args.out_dir),
        mode=args.mode,
        tol=float(tol),
        prefilter=args.prefilter,
        fix_geoms=args.fix_geoms,
        crs_metric=args.crs_metric,
        rook_edges_path=Path(args.rook_edges) if args.rook_edges else None,
        queen_edges_path=Path(args.queen_edges) if args.queen_edges else None,
        mappings_path=Path(args.mappings) if args.mappings else None,
        top_k=int(args.top_k_problems),
    )


if __name__ == "__main__":
    main()
