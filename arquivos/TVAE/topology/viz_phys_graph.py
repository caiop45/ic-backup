"""Visualization helpers for physical adjacency graphs.

This module renders taxi zone polygons colored by degree or delta-degree,
with optional edge overlays, labels, and per-zone atlas zooms.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Mapping, Sequence, TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    import geopandas as gpd


EdgeMode = Literal["none", "all", "diff_only"]
BasemapMode = Literal["auto", "on", "off"]
LabelField = Literal["LocationID", "idx"]


@dataclass(frozen=True)
class RenderConfig:
    figsize: tuple[float, float]
    dpi: int
    poly_alpha: float
    edge_alpha: float
    edge_width: float
    label: bool
    label_field: LabelField
    label_fontsize: int
    basemap: BasemapMode
    basemap_provider: str


def compute_degrees_for_locations(
    location_ids: Sequence[int],
    edges: Iterable[tuple[int, int]],
) -> list[int]:
    loc_to_idx = {int(loc): i for i, loc in enumerate(location_ids)}
    degrees = [0] * len(location_ids)
    for u, v in edges:
        u_int = int(u)
        v_int = int(v)
        if u_int == v_int:
            continue
        if u_int not in loc_to_idx or v_int not in loc_to_idx:
            continue
        i = loc_to_idx[u_int]
        j = loc_to_idx[v_int]
        degrees[i] += 1
        degrees[j] += 1
    return degrees


def neighbors_from_edges(
    location_ids: Sequence[int],
    edges: Iterable[tuple[int, int]],
) -> dict[int, set[int]]:
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
    return neighbors


def add_basemap(
    ax,
    crs,
    *,
    mode: BasemapMode,
    provider: str,
) -> None:
    if mode == "off":
        return
    try:
        import contextily as ctx
    except Exception:
        if mode in ("auto", "on"):
            print("[WARN] contextily not available; skipping basemap")
        return

    if crs is None:
        print("[WARN] Missing CRS; skipping basemap")
        return

    source = None
    if provider:
        try:
            source = _resolve_provider(ctx, provider)
        except Exception:
            print(f"[WARN] Unable to resolve basemap provider: {provider}")
            source = None

    try:
        ctx.add_basemap(ax, crs=crs, source=source, attribution_size=6)
    except Exception as exc:
        print(f"[WARN] Failed to add basemap: {exc}")


def _resolve_provider(ctx, provider: str):
    parts = provider.split(".")
    src = ctx.providers
    for part in parts:
        src = getattr(src, part)
    return src


def _representative_points(gdf: "gpd.GeoDataFrame", loc_col: str) -> dict[int, tuple[float, float]]:
    reps = gdf.geometry.representative_point()
    coords: dict[int, tuple[float, float]] = {}
    for loc, geom in zip(gdf[loc_col].tolist(), reps):
        if geom is None or geom.is_empty:
            continue
        coords[int(loc)] = (float(geom.x), float(geom.y))
    return coords


def _build_edge_lines(
    edges: Iterable[tuple[int, int]],
    rep_points: Mapping[int, tuple[float, float]],
):
    try:
        from shapely.geometry import LineString
    except Exception as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("shapely is required for edge rendering") from exc

    lines = []
    for u, v in edges:
        u_int = int(u)
        v_int = int(v)
        if u_int not in rep_points or v_int not in rep_points:
            continue
        lines.append(LineString([rep_points[u_int], rep_points[v_int]]))
    return lines


def _plot_polygons(
    gdf: "gpd.GeoDataFrame",
    value_col: str,
    *,
    cmap,
    norm,
    alpha: float,
    ax,
    linewidth: float,
) -> None:
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


def _add_labels(
    ax,
    gdf: "gpd.GeoDataFrame",
    loc_col: str,
    label_field: LabelField,
    label_fontsize: int,
    idx_map: Mapping[int, int],
) -> None:
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
        txt.set_path_effects(
            [patheffects.withStroke(linewidth=1.0, foreground="white")]
        )


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


def render_degree_map(
    gdf: "gpd.GeoDataFrame",
    loc_col: str,
    *,
    location_ids: Sequence[int],
    degrees: Sequence[int],
    edges: Iterable[tuple[int, int]] | None,
    title: str,
    out_path: Path,
    config: RenderConfig,
    edge_mode: EdgeMode,
    stats_text: str,
) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.colors as colors

    gdf_plot = gdf.copy()
    degree_map = {int(loc): int(deg) for loc, deg in zip(location_ids, degrees)}
    gdf_plot["degree"] = gdf_plot[loc_col].map(degree_map).fillna(0).astype("int64")

    fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)

    cmap = cm.get_cmap("RdYlGn")
    norm = colors.Normalize(
        vmin=float(gdf_plot["degree"].min()),
        vmax=float(gdf_plot["degree"].max()),
    )
    _plot_polygons(
        gdf_plot,
        "degree",
        cmap=cmap,
        norm=norm,
        alpha=config.poly_alpha,
        ax=ax,
        linewidth=0.2,
    )

    if edges is not None and edge_mode != "none":
        rep_points = _representative_points(gdf_plot, loc_col)
        lines = _build_edge_lines(edges, rep_points)
        if lines:
            try:
                import geopandas as gpd
            except Exception as exc:  # pragma: no cover - dependency guard
                raise RuntimeError("geopandas is required for edge rendering") from exc
            gpd.GeoSeries(lines, crs=gdf_plot.crs).plot(
                ax=ax,
                color="black",
                linewidth=config.edge_width,
                alpha=config.edge_alpha,
            )

    add_basemap(
        ax,
        gdf_plot.crs,
        mode=config.basemap,
        provider=config.basemap_provider,
    )

    if config.label:
        idx_map = {int(loc): i for i, loc in enumerate(location_ids)}
        _add_labels(
            ax,
            gdf_plot,
            loc_col,
            config.label_field,
            config.label_fontsize,
            idx_map,
        )

    _add_colorbar(ax, cmap, norm, "degree")
    _add_stats_box(ax, stats_text)
    ax.set_axis_off()
    ax.set_title(title)

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def render_delta_map(
    gdf: "gpd.GeoDataFrame",
    loc_col: str,
    *,
    location_ids: Sequence[int],
    delta: Sequence[int],
    edges_only_in_queen: Iterable[tuple[int, int]] | None,
    title: str,
    out_path: Path,
    config: RenderConfig,
    stats_text: str,
) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.colors as colors

    gdf_plot = gdf.copy()
    delta_map = {int(loc): int(val) for loc, val in zip(location_ids, delta)}
    gdf_plot["delta"] = gdf_plot[loc_col].map(delta_map).fillna(0).astype("int64")

    fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)

    cmap = cm.get_cmap("RdYlGn")
    max_abs = max(abs(gdf_plot["delta"].min()), abs(gdf_plot["delta"].max()))
    if max_abs == 0:
        max_abs = 1
    norm = colors.TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)

    _plot_polygons(
        gdf_plot,
        "delta",
        cmap=cmap,
        norm=norm,
        alpha=config.poly_alpha,
        ax=ax,
        linewidth=0.2,
    )

    if edges_only_in_queen is not None:
        rep_points = _representative_points(gdf_plot, loc_col)
        lines = _build_edge_lines(edges_only_in_queen, rep_points)
        if lines:
            try:
                import geopandas as gpd
            except Exception as exc:  # pragma: no cover - dependency guard
                raise RuntimeError("geopandas is required for edge rendering") from exc
            gpd.GeoSeries(lines, crs=gdf_plot.crs).plot(
                ax=ax,
                color="dodgerblue",
                linewidth=config.edge_width,
                alpha=config.edge_alpha,
            )

    add_basemap(
        ax,
        gdf_plot.crs,
        mode=config.basemap,
        provider=config.basemap_provider,
    )

    if config.label:
        idx_map = {int(loc): i for i, loc in enumerate(location_ids)}
        _add_labels(
            ax,
            gdf_plot,
            loc_col,
            config.label_field,
            config.label_fontsize,
            idx_map,
        )

    _add_colorbar(ax, cmap, norm, "delta_degree")
    _add_stats_box(ax, stats_text)
    ax.set_axis_off()
    ax.set_title(title)

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def render_zone_atlas(
    gdf: "gpd.GeoDataFrame",
    loc_col: str,
    *,
    location_ids: Sequence[int],
    degrees_map: Mapping[int, int],
    edges: Iterable[tuple[int, int]],
    zones: Sequence[int],
    out_dir: Path,
    config: RenderConfig,
    title_prefix: str,
) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.colors as colors

    out_dir.mkdir(parents=True, exist_ok=True)
    idx_map = {int(loc): i for i, loc in enumerate(location_ids)}
    neighbors = neighbors_from_edges(location_ids, edges)

    for zone_id in zones:
        zone_int = int(zone_id)
        zone_neighbors = neighbors.get(zone_int, set())
        subset_ids = {zone_int} | set(zone_neighbors)
        subset = gdf[gdf[loc_col].isin(subset_ids)].copy()
        if subset.empty:
            continue

        subset["degree"] = subset[loc_col].map(degrees_map).fillna(0).astype("int64")
        bounds = subset.total_bounds
        minx, miny, maxx, maxy = bounds
        pad_x = (maxx - minx) * 0.1 if maxx > minx else 1.0
        pad_y = (maxy - miny) * 0.1 if maxy > miny else 1.0

        fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)

        cmap = cm.get_cmap("RdYlGn")
        norm = colors.Normalize(
            vmin=float(subset["degree"].min()),
            vmax=float(subset["degree"].max()),
        )
        _plot_polygons(
            subset,
            "degree",
            cmap=cmap,
            norm=norm,
            alpha=config.poly_alpha,
            ax=ax,
            linewidth=0.2,
        )

        zone_geom = subset[subset[loc_col] == zone_int]
        if not zone_geom.empty:
            zone_geom.plot(
                ax=ax,
                facecolor="none",
                edgecolor="black",
                linewidth=1.2,
            )

        rep_points = _representative_points(subset, loc_col)
        incident_edges = [
            (u, v)
            for u, v in edges
            if int(u) in subset_ids and int(v) in subset_ids and (u == zone_int or v == zone_int)
        ]
        lines = _build_edge_lines(incident_edges, rep_points)
        if lines:
            try:
                import geopandas as gpd
            except Exception as exc:  # pragma: no cover - dependency guard
                raise RuntimeError("geopandas is required for edge rendering") from exc
            gpd.GeoSeries(lines, crs=subset.crs).plot(
                ax=ax,
                color="black",
                linewidth=max(config.edge_width, 0.8),
                alpha=min(config.edge_alpha + 0.2, 1.0),
            )

        add_basemap(
            ax,
            subset.crs,
            mode=config.basemap,
            provider=config.basemap_provider,
        )

        if config.label:
            _add_labels(
                ax,
                subset,
                loc_col,
                config.label_field,
                config.label_fontsize,
                idx_map,
            )

        ax.set_xlim(minx - pad_x, maxx + pad_x)
        ax.set_ylim(miny - pad_y, maxy + pad_y)
        ax.set_axis_off()
        ax.set_title(f"{title_prefix} zone {zone_int}")
        fig.savefig(out_dir / f"zone_{zone_int}.png", bbox_inches="tight")
        plt.close(fig)


def select_problem_zones(
    location_ids: Sequence[int],
    degrees: Sequence[int],
    delta: Sequence[int] | None,
    *,
    mode: Literal["low_degree", "high_delta", "both"],
    top_k: int,
) -> list[int]:
    if top_k <= 0:
        return []

    loc_deg = list(zip(location_ids, degrees))
    low_degree = [loc for loc, _deg in sorted(loc_deg, key=lambda x: (x[1], x[0]))[:top_k]]

    high_delta: list[int] = []
    if delta is not None:
        loc_delta = list(zip(location_ids, delta))
        loc_delta.sort(key=lambda x: (-abs(x[1]), -x[1], x[0]))
        high_delta = [loc for loc, _ in loc_delta[:top_k]]

    if mode == "low_degree":
        return low_degree
    if mode == "high_delta":
        return high_delta
    combined = list(dict.fromkeys(low_degree + high_delta))
    return combined
