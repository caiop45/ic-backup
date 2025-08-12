# ──────────────────── main.py ────────────────────
from __future__ import annotations

from pathlib import Path
from typing import Iterable
import unicodedata

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# (demais imports de ML/torch/optuna removidos – não usados aqui)

_TAXI_ZONES_CACHE: gpd.GeoDataFrame | None = None

# ───────────────────────────────── utilidades de zonas ────────────────────────────────
def _load_zones(taxi_zones_path: str) -> gpd.GeoDataFrame:
    global _TAXI_ZONES_CACHE
    if _TAXI_ZONES_CACHE is not None:
        return _TAXI_ZONES_CACHE

    gdf = gpd.read_file(Path(taxi_zones_path))
    rename_map = {}
    if "LocationID" in gdf and "zone_id" not in gdf:
        rename_map["LocationID"] = "zone_id"
    if "zone" in gdf and "zone_name" not in gdf:
        rename_map["zone"] = "zone_name"
    if rename_map:
        gdf = gdf.rename(columns=rename_map)

    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)

    _TAXI_ZONES_CACHE = gdf
    return gdf


def _norm(text: str | None) -> str | None:
    if text is None or pd.isna(text):
        return None
    return (
        unicodedata.normalize("NFKD", text)
        .encode("ascii", "ignore")
        .decode()
        .lower()
    )


def assign_zone_names(
    df: pd.DataFrame,
    taxi_zones_path: str = "/home-ext/caioloss/Dados/taxi-zones",
    pu_id: int | Iterable[int] | None = None,
    do_id: int | Iterable[int] | None = None,
) -> pd.DataFrame:
    gdf = _load_zones(taxi_zones_path)

    pu_points = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df["PU_longitude"], df["PU_latitude"]), crs=4326
    )
    pu_join = gpd.sjoin(
        pu_points, gdf[["zone_id", "zone_name", "geometry"]], how="left", predicate="within"
    )  # <<< fix >>>
    df["PU_zone_id"] = pu_join["zone_id"].values
    df["PU_zone_name"] = pu_join["zone_name"].values

    do_points = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df["DO_longitude"], df["DO_latitude"]), crs=4326
    )
    do_join = gpd.sjoin(
        do_points, gdf[["zone_id", "zone_name", "geometry"]], how="left", predicate="within"
    )  # <<< fix >>>
    df["DO_zone_id"] = do_join["zone_id"].values
    df["DO_zone_name"] = do_join["zone_name"].values

    if pu_id is not None:
        df = df[df["PU_zone_id"].isin(pu_id if isinstance(pu_id, Iterable) else [pu_id])]
    if do_id is not None:
        df = df[df["DO_zone_id"].isin(do_id if isinstance(do_id, Iterable) else [do_id])]

    if "geometry" in df:
        df = df.drop(columns="geometry")

    df["OD"] = df.apply(lambda r: [_norm(r["PU_zone_name"]), _norm(r["DO_zone_name"])], axis=1)
    return df


# ───────────────────────────────────────── main ──────────────────────────────────────
if __name__ == "__main__":
    # ---------- SINTÉTICOS ----------
    df_synth = pd.read_parquet("/home-ext/caioloss/Dados/viagens_synth_vae_od.parquet")
    df_synth = assign_zone_names(df_synth).dropna(subset=["PU_zone_id", "DO_zone_id"])
    df_synth = df_synth.drop(
        columns=[
            "PU_longitude",
            "PU_latitude",
            "DO_longitude",
            "DO_latitude",
            "sin_hr",
            "cos_hr",
        ],
        errors="ignore",
    )
    df_synth["OD_key"] = df_synth["OD"].apply(tuple)
    total_synth = len(df_synth)                             # <<< fix >>>
    top_synth = (
        df_synth.groupby("OD_key").size().sort_values(ascending=False).head(10)
    )
    dist = top_synth.to_frame("num_viagens_synth")
    dist["pct_total_synth"] = (dist["num_viagens_synth"] / total_synth * 100).round(2)  # <<< fix >>>

    # ---------- REAIS ----------
    df_real = pd.read_parquet("/home-ext/caioloss/Dados/viagens_lat_long.parquet")
    df_real = df_real[
        (df_real["tpep_pickup_datetime"].dt.year == 2024)
        & (df_real["tpep_pickup_datetime"].dt.month == 1)
        & (df_real["tpep_pickup_datetime"].dt.dayofweek.between(0, 3))
    ]
    df_real = assign_zone_names(df_real).dropna(subset=["PU_zone_id", "DO_zone_id"])
    df_real = df_real.drop(columns=df_synth.columns.intersection(["sin_hr", "cos_hr"]), errors="ignore")

    df_real["OD_key"] = df_real["OD"].apply(tuple)
    total_real = len(df_real)
    counts_real_aligned = df_real.groupby("OD_key").size().reindex(dist.index, fill_value=0)
    dist["num_viagens_real"] = counts_real_aligned
    dist["pct_total_real"] = (counts_real_aligned / total_real * 100).round(2)

    # Top-10 reais independente dos sintéticos
    top_real = df_real.groupby("OD_key").size().sort_values(ascending=False).head(10)
    top_real_df = top_real.to_frame("num_viagens_real")
    top_real_df["pct_total_real"] = (top_real_df["num_viagens_real"] / total_real * 100).round(2)

    # ---------- Prints ----------
    print(f"\n[SANITY] total_synth={total_synth:,} | total_real={total_real:,}")  # <<< fix >>>
    print("\n===== Comparação por OD (top-10 sintéticos) =====")
    print(dist.to_string())

    print("\n===== Top-10 ODs dos dados REAIS =====")
    print(top_real_df.to_string())
