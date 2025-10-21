# ─────────────────── utils/zone_id.py ───────────────────
from __future__ import annotations
import unidecode
import warnings
from collections.abc import Iterable
from pathlib import Path
import unicodedata
import cupy as cp
import geopandas as gpd
import pandas as pd

# Caminho padrão para o GeoJSON das zonas TLC
DEFAULT_TAXI_ZONES_PATH = "/home-ext/caioloss/Dados/taxi-zones"


def _load_zones(taxi_zones_path: str) -> gpd.GeoDataFrame:
    """Lê shapefile de zonas, padroniza CRS e nomes de colunas."""

    path = Path(taxi_zones_path)
    if not path.exists():
        raise FileNotFoundError(f"Taxi zones path not found: {taxi_zones_path}")

    gdf = gpd.read_file(path)
    print(f"[DEBUG] _load_zones | loaded {len(gdf)} polygons | CRS original: {gdf.crs}")

    rename_map = {}
    if "LocationID" in gdf.columns and "zone_id" not in gdf.columns:
        rename_map["LocationID"] = "zone_id"
    if "zone" in gdf.columns and "zone_name" not in gdf.columns:
        rename_map["zone"] = "zone_name"
    if rename_map:
        gdf = gdf.rename(columns=rename_map)
        print(f"[DEBUG] _load_zones | renomeou colunas {rename_map}")

    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
        print("[DEBUG] _load_zones | CRS definido para EPSG:4326")
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)
        print("[DEBUG] _load_zones | reprojetado para EPSG:4326")

    _TAXI_ZONES_CACHE = gdf
    return gdf


def _norm(text: str | None) -> str | None:
    """Normaliza texto (lowercase, sem acento)."""
    if text is None or pd.isna(text):
        return None
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()


def assign_zone_names(
    df: pd.DataFrame,
    taxi_zones_path: str = "/home-ext/caioloss/Dados/taxi-zones",
    pu_id: int | Iterable[int] | None = None,
    do_id: int | Iterable[int] | None = None,
) -> pd.DataFrame:
    """Anexa zone_id/zone_name a PU/DO, cria coluna OD e aplica filtros."""
    print(f"[DEBUG] assign_zone_names | linhas de entrada: {len(df)}")
    gdf = _load_zones(taxi_zones_path)

    # Pickup -----------------------------------------------------------------
    pu_points = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df["PU_longitude"], df["PU_latitude"]), crs=4326
    )
    # Usa 'within' explicitamente para evitar matches em bordas indevidas; fallback para versões antigas
    try:
        pu_join = gpd.sjoin(
            pu_points, gdf[["zone_id", "zone_name", "geometry"]], how="left", predicate="within"
        )
    except TypeError:
        pu_join = gpd.sjoin(
            pu_points, gdf[["zone_id", "zone_name", "geometry"]], how="left", op="within"
        )
    df["PU_zone_id"] = pu_join["zone_id"].values
    df["PU_zone_name"] = pu_join["zone_name"].values
    print(f"[DEBUG] PU join | sem zona: {df['PU_zone_id'].isna().sum()}/{len(df)}")

    # Drop-off ----------------------------------------------------------------
    do_points = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df["DO_longitude"], df["DO_latitude"]), crs=4326
    )
    try:
        do_join = gpd.sjoin(
            do_points, gdf[["zone_id", "zone_name", "geometry"]], how="left", predicate="within"
        )
    except TypeError:
        do_join = gpd.sjoin(
            do_points, gdf[["zone_id", "zone_name", "geometry"]], how="left", op="within"
        )
    df["DO_zone_id"] = do_join["zone_id"].values
    df["DO_zone_name"] = do_join["zone_name"].values
    print(f"[DEBUG] DO join | sem zona: {df['DO_zone_id'].isna().sum()}/{len(df)}")

    # Filtros opcionais -------------------------------------------------------
    if pu_id is not None:
        before = len(df)
        df = df[df["PU_zone_id"].isin(pu_id if isinstance(pu_id, Iterable) else [pu_id])]
        print(f"[DEBUG] filtro PU_id | {before} ➞ {len(df)} linhas")
    if do_id is not None:
        before = len(df)
        df = df[df["DO_zone_id"].isin(do_id if isinstance(do_id, Iterable) else [do_id])]
        print(f"[DEBUG] filtro DO_id | {before} ➞ {len(df)} linhas")

    # Remove geometry temporária
    if "geometry" in df.columns:
        df = df.drop(columns="geometry")

    # Coluna OD
    df["OD"] = df.apply(lambda r: [_norm(r["PU_zone_name"]), _norm(r["DO_zone_name"])], axis=1)

    print(f"[DEBUG] assign_zone_names | linhas de saída: {len(df)}\n")
    return df



def _normalize_ids(ids: int | Iterable[int] | None) -> list[int] | None:
    """Normaliza ids para lista de ints; aceita None, int, iterável ou numpy scalar."""
    if ids is None:
        return None
    # Trata numpy scalars e similares
    try:
        if not isinstance(ids, Iterable) or isinstance(ids, (str, bytes)):
            return [int(ids)]
    except Exception:
        return [int(ids)]
    out: list[int] = []
    for v in ids:
        try:
            out.append(int(v))
        except Exception:
            continue
    return out if out else None


def filter_by_zone(
    df: pd.DataFrame,
    pu_id: int | Iterable[int] | None = None,
    taxi_zones_path: str = DEFAULT_TAXI_ZONES_PATH,
) -> pd.DataFrame:
    """
    Mantém apenas as linhas cujas coordenadas de embarque pertençam às zonas
    listadas em `pu_id`. Retorna um DataFrame **sem colunas extras**.

    Parameters
    ----------
    df : pd.DataFrame
        Deve conter 'PU_latitude' e 'PU_longitude' em graus decimais (WGS-84).
    pu_id : int | Iterable[int] | None
        IDs de zona a manter. Se None, devolve df sem filtragem.
    taxi_zones_path : str
        Caminho para o GeoJSON das zonas TLC.
    """
    pu_ids = _normalize_ids(pu_id)
    if pu_ids is None:
        return df.copy()

    # 1. Carrega zonas e garante CRS/nomes com helper
    gdf = _load_zones(taxi_zones_path)
    gdf = gdf[gdf["zone_id"].isin(pu_ids)][["zone_id", "geometry"]]

    zone_ids    = gdf["zone_id"].values
    zone_bounds = gdf.geometry.bounds.apply(
        lambda r: (r.minx, r.miny, r.maxx, r.maxy), axis=1
    ).tolist()

    # 2. Bounding-box na GPU
    pu_lon = cp.asarray(df["PU_longitude"].values)
    pu_lat = cp.asarray(df["PU_latitude"].values)

    mask    = cp.zeros(pu_lon.shape, dtype=bool)
    tmp_ids = cp.full (pu_lon.shape, -1, dtype=cp.int32)

    for zid, (minx, miny, maxx, maxy) in zip(zone_ids, zone_bounds):
        in_box  = (
            (pu_lon >= minx) & (pu_lon <= maxx)
            & (pu_lat >= miny) & (pu_lat <= maxy)
        )
        mask   |= in_box
        tmp_ids = cp.where(in_box, zid, tmp_ids)  # opcional: diagnosticar

    # 3. Diagnóstico rápido
    counts = {int(z): int((tmp_ids == z).sum()) for z in pu_ids}
    print("Contagem de linhas por PULocationID:", counts)

    # 4. Retorno sem colunas extras
    return df[cp.asnumpy(mask)].copy()
