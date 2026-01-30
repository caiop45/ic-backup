#!/usr/bin/env python3
"""Visualize Strategy A functional graph (directed + weighted).

Examples:
  # Basic functional graph for full train split (top-k per origin)
  python tools/viz_sa_func_graph.py \
    --zones data/taxi_zones.parquet \
    --out-dir outputs/func_viz

  # Time-of-day slice (hour bin 8) with weight threshold
  python tools/viz_sa_func_graph.py \
    --zones data/taxi_zones.parquet \
    --out-dir outputs/func_viz_hour8 \
    --hour 8 \
    --min-weight 0.7

  # Use mappings from trained Strategy A run
  python tools/viz_sa_func_graph.py \
    --zones data/taxi_zones.parquet \
    --out-dir outputs/func_viz \
    --mappings outputs/save_data/strategy_a/mappings_strategy_a.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pandas as pd

import config
from data_processing.strategy_a_loader import load_and_split_strategy_a
from data_processing.strategy_a_transformer import StrategyATransformer
from topology.graphs import build_functional_graph
from topology.physical_adjacency import find_location_id_col, fix_geometries, read_zones
from topology.viz_phys_graph import RenderConfig, add_basemap


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


def _load_zone_to_idx(path: Path | None) -> dict[int, int] | None:
    if path is None:
        return None
    data = _load_json(path)
    if isinstance(data, dict):
        return _parse_mapping_dict(data)
    raise ValueError("Unsupported mappings JSON format (expected dict)")


def _representative_points(gdf, loc_col: str) -> dict[int, tuple[float, float]]:
    reps = gdf.geometry.representative_point()
    coords: dict[int, tuple[float, float]] = {}
    for loc, geom in zip(gdf[loc_col].tolist(), reps):
        if geom is None or geom.is_empty:
            continue
        coords[int(loc)] = (float(geom.x), float(geom.y))
    return coords


def _plot_polygons(gdf, value_col: str, *, cmap, norm, alpha: float, ax, linewidth: float) -> None:
    gdf.plot(
        column=value_col,
        cmap=cmap,
        norm=norm,
        alpha=alpha,
        linewidth=linewidth,
        edgecolor="black",
        ax=ax,
    )


def _add_colorbar(ax, cmap, norm, label: str) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label(label)


def _add_labels(ax, gdf, loc_col: str, label_field: str, label_fontsize: int, idx_map: Mapping[int, int]) -> None:
    import matplotlib.patheffects as patheffects

    reps = gdf.geometry.representative_point()
    for loc, geom in zip(gdf[loc_col].tolist(), reps):
        if geom is None or geom.is_empty:
            continue
        loc_int = int(loc)
        if label_field == "idx":
            label = str(idx_map.get(loc_int, ""))
        else:
            label = str(loc_int)
        if label == "":
            continue
        txt = ax.text(
            geom.x,
            geom.y,
            label,
            fontsize=label_fontsize,
            ha="center",
            va="center",
            color="black",
        )
        txt.set_path_effects([patheffects.withStroke(linewidth=1.0, foreground="white")])


def _add_stats_box(ax, text: str) -> None:
    props = dict(boxstyle="round", facecolor="white", alpha=0.7)
    ax.text(
        0.01,
        0.01,
        text,
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="bottom",
        bbox=props,
    )


def _parse_hours(hour: int | None, hours: str | None, hour_range: str | None, max_h: int) -> list[int] | None:
    selected: set[int] = set()

    if hour is not None:
        selected.add(int(hour))
    if hours:
        parts = [p.strip() for p in hours.split(",") if p.strip()]
        selected.update(int(p) for p in parts)
    if hour_range:
        parts = [p.strip() for p in hour_range.split(",")]
        if len(parts) != 2:
            raise ValueError("--hour-range must be START,END")
        start = int(parts[0])
        end = int(parts[1])
        if start <= end:
            selected.update(range(start, end + 1))
        else:
            selected.update(range(start, max_h))
            selected.update(range(0, end + 1))

    if not selected:
        return None

    out = sorted(selected)
    for h in out:
        if h < 0 or h >= max_h:
            raise ValueError(f"hour out of range: {h} (expected 0..{max_h - 1})")
    return out


def _compute_degrees(
    location_ids: Sequence[int],
    edges: Iterable[tuple[int, int, float]],
    mode: str,
) -> list[float]:
    loc_to_idx = {int(loc): i for i, loc in enumerate(location_ids)}
    deg = [0.0] * len(location_ids)

    weighted = "weighted" in mode
    use_in = mode in ("in", "total", "weighted_in", "weighted_total")
    use_out = mode in ("out", "total", "weighted_out", "weighted_total")

    for u, v, w in edges:
        u_int = int(u)
        v_int = int(v)
        if u_int not in loc_to_idx or v_int not in loc_to_idx:
            continue
        val = float(w) if weighted else 1.0
        if use_out:
            deg[loc_to_idx[u_int]] += val
        if use_in:
            deg[loc_to_idx[v_int]] += val

    return deg


def _edges_from_graph(
    neighbors: Sequence[Sequence[int]],
    weights: Sequence[Sequence[float]],
    idx_to_loc: Mapping[int, int],
    *,
    min_weight: float,
    max_edges: int | None,
) -> list[tuple[int, int, float]]:
    rows: list[tuple[int, int, float]] = []
    for u_idx, nbrs in enumerate(neighbors):
        for v_idx, w in zip(nbrs, weights[u_idx]):
            if float(w) < min_weight:
                continue
            rows.append((int(u_idx), int(v_idx), float(w)))

    if max_edges is not None and max_edges > 0 and len(rows) > max_edges:
        rows.sort(key=lambda x: x[2], reverse=True)
        rows = rows[:max_edges]

    edges_loc: list[tuple[int, int, float]] = []
    for u_idx, v_idx, w in rows:
        if u_idx not in idx_to_loc or v_idx not in idx_to_loc:
            continue
        edges_loc.append((int(idx_to_loc[u_idx]), int(idx_to_loc[v_idx]), float(w)))
    return edges_loc


def _stats_text(num_nodes: int, num_edges: int, avg_out: float, avg_in: float) -> str:
    return (
        f"nodes={num_nodes} edges={num_edges} "
        f"avg_out={avg_out:.2f} avg_in={avg_in:.2f}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--zones", type=str, required=True, help="Path to zones polygons file")
    ap.add_argument("--out-dir", type=str, required=True, help="Output directory")
    ap.add_argument(
        "--split",
        choices=["S1", "S2"],
        default=None,
        help="Strategy A split (defaults to config.SA_SPLIT_STRATEGY)",
    )
    ap.add_argument(
        "--mappings",
        type=str,
        default=None,
        help="Optional JSON mapping for LocationID -> idx",
    )
    ap.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Top-K destinations per origin (defaults to config.SA_GFUNC_TOPK)",
    )
    ap.add_argument(
        "--weight-mode",
        choices=["log1p", "count"],
        default="log1p",
        help="Edge weight mode (log1p is the training default)",
    )
    ap.add_argument(
        "--min-weight",
        type=float,
        default=0.0,
        help="Drop edges with weight < min-weight (after weight_mode)",
    )
    ap.add_argument(
        "--max-edges",
        type=int,
        default=None,
        help="Optional cap on total edges drawn (keeps highest-weight edges)",
    )
    ap.add_argument("--hour", type=int, default=None, help="Single hour bin (0..H-1)")
    ap.add_argument(
        "--hours",
        type=str,
        default=None,
        help="Comma-separated hour bins (e.g., 8,9,10)",
    )
    ap.add_argument(
        "--hour-range",
        type=str,
        default=None,
        help="Hour range START,END (inclusive, wraps if START > END)",
    )
    ap.add_argument(
        "--crs-metric",
        choices=["none", "EPSG:3857"],
        default="EPSG:3857",
        help="Plotting CRS (EPSG:3857 recommended for basemap)",
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
    ap.add_argument("--figsize", type=str, default="20,20", help="Figure size as WIDTH,HEIGHT")
    ap.add_argument("--poly-alpha", type=float, default=0.55, help="Polygon alpha")
    ap.add_argument("--edge-alpha", type=float, default=0.5, help="Base edge alpha")
    ap.add_argument("--edge-width", type=float, default=0.4, help="Base edge linewidth")
    ap.add_argument("--arrow-size", type=float, default=8.0, help="Base arrowhead size")
    ap.add_argument("--label", choices=["on", "off"], default="on")
    ap.add_argument(
        "--label-field",
        choices=["LocationID", "idx"],
        default="LocationID",
    )
    ap.add_argument("--label-fontsize", type=int, default=5)
    ap.add_argument(
        "--degree-mode",
        choices=[
            "out",
            "in",
            "total",
            "weighted_out",
            "weighted_in",
            "weighted_total",
        ],
        default="weighted_out",
        help="Zone coloring metric for degrees",
    )
    ap.add_argument(
        "--fix-geoms",
        choices=["none", "buffer0", "make_valid_if_available"],
        default="none",
        help="Geometry fixing mode",
    )
    args = ap.parse_args()

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

    gdf_plot = gdf if args.crs_metric == "none" else gdf.to_crs(args.crs_metric)
    location_ids = sorted(set(int(v) for v in gdf[loc_col].astype("int64").tolist()))
    if not location_ids:
        raise ValueError("No LocationIDs found in zones file")

    mappings_path = Path(args.mappings) if args.mappings else None
    if mappings_path is not None and not mappings_path.exists():
        print(f"[ERROR] mappings file not found: {mappings_path}")
        print(
            "[HINT] Use outputs/save_data/strategy_a/mappings_strategy_a.json or omit --mappings "
            "to fit the transformer on the train split."
        )
        return

    zone_to_idx = _load_zone_to_idx(mappings_path) if mappings_path else None

    train_df, _val_df, _hold_df = load_and_split_strategy_a(
        split=args.split
    )
    transformer = None
    if zone_to_idx is None:
        transformer = StrategyATransformer().fit(train_df)
        zone_to_idx = transformer.zone_to_idx

    idx_to_loc = {idx: loc for loc, idx in zone_to_idx.items()}

    max_h = int(config.SA_TIME_BINS_H)
    hours = _parse_hours(args.hour, args.hours, args.hour_range, max_h=max_h)
    if hours is not None:
        before = len(train_df)
        train_df = train_df[train_df["hora_do_dia"].isin(hours)].reset_index(drop=True)
        print(f"[INFO] time filter hours={hours} rows={len(train_df)} (from {before})")

    if transformer is None:
        # Map pickup/dropoff directly using zone_to_idx
        train_idx = pd.DataFrame()
        train_idx["o_idx"] = train_df["pickup_id"].map(zone_to_idx)
        train_idx["d_idx"] = train_df["dropoff_id"].map(zone_to_idx)
        train_idx = train_idx.dropna().astype("int64").reset_index(drop=True)
    else:
        train_idx = transformer.transform(train_df)

    num_nodes = max(zone_to_idx.values()) + 1 if zone_to_idx else 0
    if num_nodes <= 0:
        raise ValueError("No zones found in mapping")

    top_k = int(args.top_k) if args.top_k is not None else int(config.SA_GFUNC_TOPK)
    gfunc = build_functional_graph(
        train_idx,
        num_nodes=num_nodes,
        top_k=top_k,
        weight_mode=args.weight_mode,
    )

    edges = _edges_from_graph(
        gfunc.neighbors,
        gfunc.weights,
        idx_to_loc,
        min_weight=float(args.min_weight),
        max_edges=int(args.max_edges) if args.max_edges is not None else None,
    )

    if not edges:
        raise RuntimeError("No edges to visualize (check time filter / min-weight / top-k)")

    degrees = _compute_degrees(location_ids, edges, mode=args.degree_mode)
    degree_map = {int(loc): float(deg) for loc, deg in zip(location_ids, degrees)}
    gdf_plot["degree"] = gdf_plot[loc_col].map(degree_map).fillna(0.0)

    rep_points = _representative_points(gdf_plot, loc_col)
    weights = [w for _u, _v, w in edges]
    w_min = min(weights)
    w_max = max(weights)
    w_span = w_max - w_min

    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import matplotlib.colors as colors
    from matplotlib.patches import FancyArrowPatch

    fig, ax = plt.subplots(figsize=_parse_figsize(args.figsize), dpi=int(args.dpi))

    cmap = mpl.colormaps.get_cmap("RdYlGn")
    norm = colors.Normalize(vmin=float(gdf_plot["degree"].min()), vmax=float(gdf_plot["degree"].max()))
    _plot_polygons(
        gdf_plot,
        "degree",
        cmap=cmap,
        norm=norm,
        alpha=float(args.poly_alpha),
        ax=ax,
        linewidth=0.2,
    )

    for u_loc, v_loc, w in edges:
        if u_loc not in rep_points or v_loc not in rep_points:
            continue
        x1, y1 = rep_points[u_loc]
        x2, y2 = rep_points[v_loc]
        if w_span > 0:
            w_norm = (w - w_min) / w_span
        else:
            w_norm = 1.0
        lw = float(args.edge_width) * (0.3 + 0.7 * w_norm)
        alpha = min(1.0, float(args.edge_alpha) * (0.3 + 0.7 * w_norm))
        mscale = float(args.arrow_size) * (0.5 + 1.5 * w_norm)
        arrow = FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=mscale,
            linewidth=lw,
            color="black",
            alpha=alpha,
            shrinkA=0.0,
            shrinkB=0.0,
        )
        ax.add_patch(arrow)

    add_basemap(ax, gdf_plot.crs, mode=args.basemap, provider=args.basemap_provider)

    if args.label == "on":
        idx_map = {int(loc): i for i, loc in enumerate(location_ids)}
        _add_labels(ax, gdf_plot, loc_col, args.label_field, int(args.label_fontsize), idx_map)

    out_path = out_dir / "map_func.png"
    avg_out = sum(_compute_degrees(location_ids, edges, "out")) / float(len(location_ids))
    avg_in = sum(_compute_degrees(location_ids, edges, "in")) / float(len(location_ids))
    _add_colorbar(ax, cmap, norm, args.degree_mode)
    _add_stats_box(ax, _stats_text(len(location_ids), len(edges), avg_out, avg_in))
    ax.set_axis_off()
    ax.set_title("Functional graph (directed, weighted)")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    edges_df = pd.DataFrame(edges, columns=["u_location_id", "v_location_id", "weight"])
    edges_df.to_csv(out_dir / "func_edges_used.csv", index=False)
    print(f"[OK] wrote {out_path}")
    print(f"[OK] wrote {out_dir / 'func_edges_used.csv'}")


if __name__ == "__main__":
    main()
