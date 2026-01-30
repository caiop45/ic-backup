#!/usr/bin/env python3
"""Visualize physical adjacency graphs for taxi zones.

Examples:
  # (a) Compute rook + queen + diff from polygons
  python tools/viz_tht_phys_graph.py \
    --zones data/taxi_zones.zip \
    --out-dir outputs/phys_viz

  # (b) Load rook/queen edges from CSVs
  python tools/viz_tht_phys_graph.py \
    --zones data/taxi_zones.zip \
    --out-dir outputs/phys_viz_loaded \
    --mode load \
    --rook-edges data/tht_phys_edges_rook.csv \
    --queen-edges data/tht_phys_edges_queen.csv \
    --mappings data/zone_mappings.json

  # (c) Render a single graph from an edges CSV
  python tools/viz_tht_phys_graph.py \
    --zones data/taxi_zones.zip \
    --out-dir outputs/phys_viz_single \
    --edges data/tht_phys_edges_rook.csv \
    --adjacency rook \
    --mappings data/zone_mappings.json

  # (d) Debug strict vs near-miss adjacency with gap tolerance
  python tools/viz_tht_phys_graph.py \
    --zones data/taxi_zones.parquet \
    --out-dir outputs/phys_viz_debug \
    --adjacency-crs EPSG:3857 \
    --gap-tol 5.0 \
    --bridge-max-dist 2500 \
    --edge-style both \
    --debug-zone-ids "5,84" \
    --debug-radius-m 2500
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Literal, Mapping, Sequence

import pandas as pd

from topology.physical_adjacency import (
    augment_connect_components_by_distance,
    compute_edges_location_id,
    compute_near_miss_edges_location_id,
    find_location_id_col,
    fix_geometries,
    read_zones,
)
from topology.viz_phys_graph import (
    RenderConfig,
    compute_degrees_for_locations,
    render_degree_map,
    render_delta_map,
    render_debug_zone_map,
    render_zone_atlas,
    select_problem_zones,
)

Mode = Literal["compute", "load"]


def _parse_figsize(raw: str) -> tuple[float, float]:
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != 2:
        raise ValueError("--figsize must be in the form WIDTH,HEIGHT")
    return float(parts[0]), float(parts[1])


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


def _normalize_edge(u: int, v: int) -> tuple[int, int]:
    if u < v:
        return u, v
    return v, u


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
        return sorted({_normalize_edge(int(u), int(v)) for u, v in edges if int(u) != int(v)})

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
            edges_loc.add(_normalize_edge(u_loc, v_loc))

    if missing:
        print(
            f"[WARN] {len(missing)} idx values from {path} were missing in mappings. "
            f"Examples: {sorted(list(missing))[:10]}"
        )

    return sorted(edges_loc)


def _location_ids_from_gdf(gdf, loc_col: str) -> list[int]:
    loc_series = gdf[loc_col].dropna().astype("int64")
    return sorted(set(int(v) for v in loc_series.tolist()))


def _stats_text(location_ids: Sequence[int], edges: Sequence[tuple[int, int]], degrees: Sequence[int]) -> str:
    num_nodes = len(location_ids)
    num_edges = len(edges)
    avg_degree = float(sum(degrees)) / float(num_nodes) if num_nodes else 0.0
    isolated = sum(1 for d in degrees if d == 0)
    return f"nodes={num_nodes} edges={num_edges} avg_degree={avg_degree:.2f} isolated={isolated}"


def _prepare_plot_gdf(gdf, *, plot_crs: str) -> object:
    if plot_crs == "none":
        return gdf
    return gdf.to_crs(plot_crs)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--zones", type=str, required=True, help="Path to zones polygons file")
    ap.add_argument("--out-dir", type=str, required=True, help="Output directory")
    ap.add_argument("--mode", choices=["compute", "load"], default="compute")
    ap.add_argument("--rook-edges", type=str, default=None, help="Path to rook edges CSV")
    ap.add_argument("--queen-edges", type=str, default=None, help="Path to queen edges CSV")
    ap.add_argument("--edges", type=str, default=None, help="Single edges CSV to visualize")
    ap.add_argument(
        "--adjacency",
        choices=["rook", "queen"],
        default="rook",
        help="Adjacency label for single-graph visualization",
    )
    ap.add_argument(
        "--tol",
        type=float,
        default=None,
        help="Rook boundary length threshold. Defaults to 1e-9 for geographic or 5.0 for metric adjacency.",
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
        "--crs-metric",
        choices=["none", "EPSG:3857"],
        default="EPSG:3857",
        help="Plotting CRS (EPSG:3857 recommended for basemap)",
    )
    ap.add_argument(
        "--adjacency-crs",
        choices=["none", "EPSG:3857"],
        default="none",
        help="Optional CRS for adjacency computation (separate from plotting)",
    )
    ap.add_argument(
        "--basemap",
        choices=["auto", "on", "off"],
        default="auto",
        help="Basemap mode (auto uses contextily if available)",
    )
    ap.add_argument(
        "--basemap-provider",
        choices=["CartoDB.Positron", "OpenStreetMap.Mapnik"],
        default="CartoDB.Positron",
        help="Basemap provider (contextily)",
    )
    ap.add_argument("--dpi", type=int, default=400, help="Output DPI")
    ap.add_argument(
        "--figsize",
        type=str,
        default="20,20",
        help="Figure size as WIDTH,HEIGHT",
    )
    ap.add_argument("--poly-alpha", type=float, default=0.55, help="Polygon alpha")
    ap.add_argument("--edge-alpha", type=float, default=0.35, help="Edge alpha")
    ap.add_argument("--edge-width", type=float, default=0.4, help="Edge linewidth")
    ap.add_argument("--label", choices=["on", "off"], default="on")
    ap.add_argument(
        "--label-field",
        choices=["LocationID", "idx"],
        default="LocationID",
    )
    ap.add_argument("--label-fontsize", type=int, default=5)
    ap.add_argument(
        "--edge-mode",
        choices=["none", "all", "diff_only"],
        default="all",
        help="Edge overlay mode (diff_only used for diff map)",
    )
    ap.add_argument(
        "--edge-style",
        choices=["strict", "tolerant", "both"],
        default=None,
        help="Edge style for queen visualization (default both when gap_tol>0).",
    )
    ap.add_argument(
        "--debug-zone-ids",
        type=str,
        default="",
        help="Comma-separated LocationIDs for debug zoom views.",
    )
    ap.add_argument(
        "--debug-radius-m",
        type=float,
        default=2500.0,
        help="Radius in meters for debug zone context.",
    )
    ap.add_argument(
        "--debug-max-candidates",
        type=int,
        default=40,
        help="Max candidate zones in debug view (nearest by distance).",
    )
    ap.add_argument("--atlas-topk", type=int, default=25)
    ap.add_argument(
        "--atlas-mode",
        choices=["none", "low_degree", "high_delta", "both"],
        default="none",
    )
    ap.add_argument(
        "--fix-geoms",
        choices=["none", "buffer0", "make_valid_if_available"],
        default="none",
        help="Geometry fixing mode",
    )
    ap.add_argument(
        "--mappings",
        type=str,
        default=None,
        help="JSON mapping for idx<->LocationID (required if CSV lacks LocationID columns)",
    )

    args = ap.parse_args()

    tol = args.tol
    if tol is None:
        tol = 5.0 if args.adjacency_crs != "none" else 1e-9
    gap_tol = float(args.gap_tol)
    if gap_tol > 0 and args.adjacency_crs == "none":
        print("[WARN] gap_tol set but adjacency_crs is 'none'. Consider EPSG:3857.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gdf = read_zones(args.zones)
    loc_col = find_location_id_col(gdf)
    gdf = fix_geometries(gdf, mode=args.fix_geoms)
    dup_mask = gdf[loc_col].duplicated()
    if dup_mask.any():
        dup_ids = sorted(gdf.loc[dup_mask, loc_col].astype("int64").unique().tolist())
        print(
            f"[WARN] Found {len(dup_ids)} duplicated LocationIDs in zones; "
            "dissolving to a single geometry per LocationID."
        )
        gdf = gdf[[loc_col, "geometry"]].dissolve(by=loc_col, as_index=False)

    gdf_adj = gdf if args.adjacency_crs == "none" else gdf.to_crs(args.adjacency_crs)
    gdf_plot = _prepare_plot_gdf(gdf, plot_crs=args.crs_metric)

    location_ids = _location_ids_from_gdf(gdf, loc_col)
    if not location_ids:
        raise ValueError("No LocationIDs found in zones file")

    mappings_path = Path(args.mappings) if args.mappings else None
    if mappings_path is not None and not mappings_path.exists():
        print(f"[ERROR] mappings file not found: {mappings_path}")
        print(
            "[HINT] If your edges CSV includes u_location_id/v_location_id, you can omit --mappings. "
            "Otherwise, use outputs/save_data/tht_tripgen/mappings_tht_tripgen.json."
        )
        return
    idx_to_location_id = load_idx_to_location_id(mappings_path)

    config = RenderConfig(
        figsize=_parse_figsize(args.figsize),
        dpi=int(args.dpi),
        poly_alpha=float(args.poly_alpha),
        edge_alpha=float(args.edge_alpha),
        edge_width=float(args.edge_width),
        label=args.label == "on",
        label_field=args.label_field,
        label_fontsize=int(args.label_fontsize),
        basemap=args.basemap,
        basemap_provider=args.basemap_provider,
    )
    edge_style = args.edge_style
    if edge_style is None:
        if gap_tol > 0:
            edge_style = "both"
        elif args.bridge_max_dist and args.bridge_max_dist > 0:
            edge_style = "tolerant"
        else:
            edge_style = "strict"

    if args.edges:
        edges_single = _extract_edges_from_csv(Path(args.edges), idx_to_location_id)
        degrees_single = compute_degrees_for_locations(location_ids, edges_single)
        stats_text = _stats_text(location_ids, edges_single, degrees_single)
        edges_to_draw = edges_single if args.edge_mode != "none" else None

        render_degree_map(
            gdf_plot,
            loc_col,
            location_ids=location_ids,
            degrees=degrees_single,
            edges=edges_to_draw,
            near_miss_edges=None,
            title=f"Adjacency ({args.adjacency})",
            out_path=out_dir / f"map_{args.adjacency}.png",
            config=config,
            edge_mode=args.edge_mode,
            stats_text=stats_text,
            edge_style="strict",
        )

        if args.atlas_mode in ("low_degree", "both"):
            zones = select_problem_zones(
                location_ids,
                degrees_single,
                delta=None,
                mode="low_degree",
                top_k=int(args.atlas_topk),
            )
            if zones:
                degrees_map = {int(loc): int(deg) for loc, deg in zip(location_ids, degrees_single)}
                render_zone_atlas(
                    gdf_plot,
                    loc_col,
                    location_ids=location_ids,
                    degrees_map=degrees_map,
                    edges=edges_single,
                    zones=zones,
                    out_dir=out_dir / "atlas" / "low_degree",
                    config=config,
                    title_prefix="Low degree",
                )
        if args.atlas_mode in ("high_delta", "both"):
            print("[WARN] High-delta atlas requires rook+queen; skipping in single mode")
        return

    if args.mode == "compute":
        edges_rook = compute_edges_location_id(
            gdf_adj,
            loc_col,
            mode="rook",
            tol=float(tol),
            prefilter="touches",
        )
        edges_queen_strict = compute_edges_location_id(
            gdf_adj,
            loc_col,
            mode="queen",
            tol=float(tol),
            prefilter="bbox",
            gap_tol=0.0,
        )
        edges_queen_tol = compute_edges_location_id(
            gdf_adj,
            loc_col,
            mode="queen",
            tol=float(tol),
            prefilter="bbox",
            gap_tol=gap_tol,
        )
        bridge_edges: list[tuple[int, int, float]] = []
        edges_queen_final = edges_queen_tol
        if args.bridge_max_dist and args.bridge_max_dist > 0:
            edges_queen_final, bridge_edges = augment_connect_components_by_distance(
                gdf=gdf_adj,
                loc_col=loc_col,
                edges_loc=edges_queen_tol,
                max_dist=float(args.bridge_max_dist),
            )
    else:
        if args.rook_edges is None or args.queen_edges is None:
            raise ValueError("--rook-edges and --queen-edges are required in load mode")
        edges_rook = _extract_edges_from_csv(Path(args.rook_edges), idx_to_location_id)
        edges_queen_strict = _extract_edges_from_csv(
            Path(args.queen_edges), idx_to_location_id
        )
        edges_queen_tol = (
            compute_edges_location_id(
                gdf_adj,
                loc_col,
                mode="queen",
                tol=float(tol),
                prefilter="bbox",
                gap_tol=gap_tol,
            )
            if gap_tol > 0
            else edges_queen_strict
        )
        bridge_edges = []
        edges_queen_final = edges_queen_tol

    degrees_rook = compute_degrees_for_locations(location_ids, edges_rook)
    degrees_queen_strict = compute_degrees_for_locations(location_ids, edges_queen_strict)
    degrees_queen_tol = compute_degrees_for_locations(location_ids, edges_queen_tol)
    degrees_queen_final = compute_degrees_for_locations(location_ids, edges_queen_final)
    delta = [int(q - r) for q, r in zip(degrees_queen_strict, degrees_rook)]

    edges_only_in_queen = sorted(set(edges_queen_strict) - set(edges_rook))
    near_miss = (
        compute_near_miss_edges_location_id(
            gdf_adj,
            loc_col,
            gap_tol=gap_tol,
            prefilter="bbox",
        )
        if gap_tol > 0
        else []
    )
    near_miss_pairs = [(u, v) for u, v, _ in near_miss]
    edges_only_in_tolerant = sorted(set(edges_queen_tol) - set(edges_queen_strict))

    stats_rook = _stats_text(location_ids, edges_rook, degrees_rook)
    stats_queen = _stats_text(location_ids, edges_queen_strict, degrees_queen_strict)
    stats_diff = _stats_text(location_ids, edges_only_in_queen, [abs(d) for d in delta])
    stats_queen_tol = _stats_text(location_ids, edges_queen_tol, degrees_queen_tol)
    stats_queen_final = _stats_text(location_ids, edges_queen_final, degrees_queen_final)

    edges_to_draw = edges_rook if args.edge_mode != "none" else None
    render_degree_map(
        gdf_plot,
        loc_col,
        location_ids=location_ids,
        degrees=degrees_rook,
        edges=edges_to_draw,
        near_miss_edges=None,
        title="Rook adjacency",
        out_path=out_dir / "map_rook.png",
        config=config,
        edge_mode=args.edge_mode,
        stats_text=stats_rook,
        edge_style="strict",
    )

    if edge_style == "tolerant":
        degree_mode = degrees_queen_final
        edges_to_draw = edges_queen_final if args.edge_mode != "none" else None
        near_to_draw = None
        title = (
            "Queen adjacency (tolerant + bridge)"
            if bridge_edges
            else "Queen adjacency (tolerant)"
        )
        stats_text = stats_queen_final
        bridge_to_draw = None
    elif edge_style == "both":
        degree_mode = degrees_queen_final
        edges_to_draw = edges_queen_strict if args.edge_mode != "none" else None
        near_to_draw = near_miss_pairs if gap_tol > 0 else None
        title = (
            "Queen adjacency (strict + near-miss + bridge)"
            if bridge_edges
            else "Queen adjacency (strict + near-miss)"
        )
        stats_text = stats_queen_final
        bridge_to_draw = bridge_edges if bridge_edges else None
    else:
        degree_mode = degrees_queen_strict
        edges_to_draw = edges_queen_strict if args.edge_mode != "none" else None
        near_to_draw = near_miss_pairs if gap_tol > 0 else None
        title = "Queen adjacency (strict)"
        stats_text = stats_queen
        bridge_to_draw = None

    render_degree_map(
        gdf_plot,
        loc_col,
        location_ids=location_ids,
        degrees=degree_mode,
        edges=edges_to_draw,
        near_miss_edges=near_to_draw,
        bridge_edges=bridge_to_draw,
        title=title,
        out_path=out_dir / "map_queen.png",
        config=config,
        edge_mode=args.edge_mode,
        stats_text=stats_text,
        edge_style=edge_style,
    )

    diff_edges_to_draw = edges_only_in_queen if args.edge_mode != "none" else None
    render_delta_map(
        gdf_plot,
        loc_col,
        location_ids=location_ids,
        delta=delta,
        edges_only_in_queen=diff_edges_to_draw,
        title="Delta (queen - rook)",
        out_path=out_dir / "map_diff.png",
        config=config,
        stats_text=stats_diff,
    )

    debug_ids = [int(x) for x in args.debug_zone_ids.split(",") if x.strip()] if args.debug_zone_ids else []
    if debug_ids:
        debug_rows = []
        loc_to_idx = {loc: i for i, loc in enumerate(location_ids)}
        geoms_adj = gdf_adj.set_index(loc_col)
        for zone_id in debug_ids:
            if zone_id not in geoms_adj.index:
                print(f"[WARN] debug zone {zone_id} not found in zones")
                continue
            zone_geom = geoms_adj.loc[zone_id].geometry
            if zone_geom is None or zone_geom.is_empty:
                continue
            buffer_geom = zone_geom.buffer(float(args.debug_radius_m))
            candidates = gdf_adj[gdf_adj.geometry.intersects(buffer_geom)][loc_col].astype("int64").tolist()
            if len(candidates) > int(args.debug_max_candidates):
                dist_pairs = []
                for cand in candidates:
                    if cand == zone_id:
                        continue
                    cand_geom = geoms_adj.loc[cand].geometry
                    if cand_geom is None or cand_geom.is_empty:
                        continue
                    dist_pairs.append((cand, float(zone_geom.distance(cand_geom))))
                dist_pairs.sort(key=lambda x: (x[1], x[0]))
                keep = [zone_id] + [c for c, _ in dist_pairs[: int(args.debug_max_candidates) - 1]]
                candidates = keep

            strict_degree = degrees_queen_strict[loc_to_idx[zone_id]]
            tolerant_degree = degrees_queen_tol[loc_to_idx[zone_id]]
            num_near = sum(1 for u, v, _ in near_miss if u == zone_id or v == zone_id)
            nearest_distance = None
            for other_loc in location_ids:
                if other_loc == zone_id:
                    continue
                other_geom = geoms_adj.loc[other_loc].geometry
                if other_geom is None or other_geom.is_empty:
                    continue
                dist = float(zone_geom.distance(other_geom))
                if nearest_distance is None or dist < nearest_distance:
                    nearest_distance = dist

            debug_rows.append(
                {
                    "debug_zone_id": zone_id,
                    "strict_degree": strict_degree,
                    "tolerant_degree": tolerant_degree,
                    "num_near_miss": num_near,
                    "nearest_distance": nearest_distance,
                }
            )

            near_for_zone = [
                (u, v, dist)
                for u, v, dist in near_miss
                if (u == zone_id or v == zone_id) and (u in candidates and v in candidates)
            ]

            degree_map = {
                int(loc): int(deg)
                for loc, deg in zip(
                    location_ids,
                    degrees_queen_tol if edge_style in ("tolerant", "both") else degrees_queen_strict,
                )
            }

            render_debug_zone_map(
                gdf_plot,
                loc_col,
                zone_id=zone_id,
                candidate_ids=candidates,
                location_ids=location_ids,
                degrees_map=degree_map,
                strict_edges=edges_queen_strict,
                tolerant_edges=edges_queen_tol,
                near_miss_edges=near_for_zone,
                out_path=out_dir / f"debug_zone_{zone_id}.png",
                config=config,
                edge_style=edge_style,
                title=f"Debug zone {zone_id}",
                show_labels=True,
            )

        if debug_rows:
            pd.DataFrame(debug_rows).to_csv(out_dir / "debug_summary.csv", index=False)

    if args.atlas_mode in ("low_degree", "both"):
        zones = select_problem_zones(
            location_ids,
            degrees_rook,
            delta=delta,
            mode="low_degree",
            top_k=int(args.atlas_topk),
        )
        if zones:
            degrees_map = {int(loc): int(deg) for loc, deg in zip(location_ids, degrees_rook)}
            render_zone_atlas(
                gdf_plot,
                loc_col,
                location_ids=location_ids,
                degrees_map=degrees_map,
                edges=edges_rook,
                zones=zones,
                out_dir=out_dir / "atlas" / "low_degree",
                config=config,
                title_prefix="Low degree",
            )

    if args.atlas_mode in ("high_delta", "both"):
        zones = select_problem_zones(
            location_ids,
            degrees_queen_strict,
            delta=delta,
            mode="high_delta",
            top_k=int(args.atlas_topk),
        )
        if zones:
            degrees_map = {int(loc): int(deg) for loc, deg in zip(location_ids, degrees_queen_strict)}
            render_zone_atlas(
                gdf_plot,
                loc_col,
                location_ids=location_ids,
                degrees_map=degrees_map,
                edges=edges_only_in_queen,
                zones=zones,
                out_dir=out_dir / "atlas" / "high_delta",
                config=config,
                title_prefix="High delta",
            )


if __name__ == "__main__":
    main()
