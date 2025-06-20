import cupy as cp
import pandas as pd
import geopandas as gpd
import warnings
import cupy as cp
import pandas as pd
import geopandas as gpd
from collections.abc import Iterable  # ⇦ novo


def assign_zone_names(
    df: pd.DataFrame,
    taxi_zones_path: str = "/home-ext/caioloss/Dados/taxi-zones",
    pu_id: int | Iterable[int] | None = None,
    do_id: int | Iterable[int] | None = None,
) -> pd.DataFrame:
    """
    Atribui IDs/nome das zonas aos pontos de PU/DO e, opcionalmente,
    filtra pelos IDs fornecidos.
    """
    # ─────────────────────────────── 1. Polígonos das zonas
    gdf = (
    gpd.read_file(taxi_zones_path)[["LocationID", "zone", "geometry"]]
    .to_crs(epsg=4326)
    .assign(_area=lambda df: df.geometry.area)   # ① calcula área
    .sort_values("_area")                        # ② pequenos → grandes
    )
    id2zone = gdf.set_index("LocationID")["zone"].to_dict()
    gdf["bounds"] = gdf.geometry.bounds.apply(
        lambda r: (r.minx, r.miny, r.maxx, r.maxy), axis=1
    )

    zone_ids    = gdf["LocationID"].to_numpy()
    zone_bounds = gdf["bounds"].tolist()

    # ─────────────────────────────── 2. Pontos na GPU
    pu_lon = cp.asarray(df["PU_longitude"].to_numpy())
    pu_lat = cp.asarray(df["PU_latitude"].to_numpy())
    do_lon = cp.asarray(df["DO_longitude"].to_numpy())
    do_lat = cp.asarray(df["DO_latitude"].to_numpy())

    pu_ids = cp.full(pu_lon.shape, -1, dtype=cp.int32)
    do_ids = cp.full(do_lon.shape, -1, dtype=cp.int32)

    # ── loop: só preenche quem ainda está −1  ← ALTERAÇÃO CRÍTICA
    for zid, (minx, miny, maxx, maxy) in zip(zone_ids, zone_bounds):
        pu_mask = (
            (pu_ids == -1)
            & (pu_lon >= minx) & (pu_lon <= maxx)
            & (pu_lat >= miny) & (pu_lat <= maxy)
        )
        do_mask = (
            (do_ids == -1)
            & (do_lon >= minx) & (do_lon <= maxx)
            & (do_lat >= miny) & (do_lat <= maxy)
        )
        pu_ids = cp.where(pu_mask, zid, pu_ids)
        do_ids = cp.where(do_mask, zid, do_ids)

        # break antecipado: todos atribuídos
        if not (pu_ids == -1).any() and not (do_ids == -1).any():
            break

    # volta para CPU
    df["PULocationID"] = cp.asnumpy(pu_ids)
    df["DOLocationID"] = cp.asnumpy(do_ids)

    # ─────────────────────────────── 3. Filtro opcional
    def _normalize(x):
        if x is None:
            return None
        if isinstance(x, int):
            return [x]
        if isinstance(x, Iterable):
            return list(x)
        raise TypeError("pu_id/do_id devem ser int ou iterável de int")

    pu_ids_filter = _normalize(pu_id)
    do_ids_filter = _normalize(do_id)

    if pu_ids_filter is not None:
        df = df[df["PULocationID"].isin(pu_ids_filter)].copy()
        if df.empty:
            warnings.warn(f"Nenhum ponto encontrado para PU {pu_ids_filter}.")
    elif do_ids_filter is not None:
        df = df[df["DOLocationID"].isin(do_ids_filter)].copy()
        if df.empty:
            warnings.warn(f"Nenhum ponto encontrado para DO {do_ids_filter}.")

    # ─────────────────────────────── 4. IDs → nomes
    df["PULocationID"] = df["PULocationID"].map(id2zone)
    df["DOLocationID"] = df["DOLocationID"].map(id2zone)

    return df

def filter_by_zone(
    df: pd.DataFrame,
    pu_id: int | Iterable[int] | None = None,
    taxi_zones_path: str = "/home-ext/caioloss/Dados/taxi-zones",
) -> pd.DataFrame:
    """
    Mantém apenas as linhas cujas coordenadas de embarque pertençam às zonas
    listadas em ``pu_id``. Não adiciona nem remove colunas: devolve exatamente
    o mesmo DataFrame (filtrado) que entrou.

    Parameters
    ----------
    df : pd.DataFrame
        Deve ter 'PU_latitude' e 'PU_longitude' em graus decimais (WGS-84).
    pu_id : int | Iterable[int] | None
        IDs de zona a manter. Se None, devolve df sem filtragem.
    taxi_zones_path : str
        Caminho para o GeoJSON das zonas TLC.
    """
    # ─────────────── 1. Normalização da entrada ────────────────
    def _normalize(x):
        if x is None:
            return None
        if isinstance(x, int):
            return [x]
        if isinstance(x, Iterable):
            return list(x)
        raise TypeError("`pu_id` deve ser int ou iterável de int")

    pu_ids = _normalize(pu_id)
    if pu_ids is None:
        return df.copy()

    # ─────────────── 2. Carrega apenas as zonas desejadas ──────
    gdf = (
        gpd.read_file(taxi_zones_path)[["LocationID", "geometry"]]
        .to_crs(epsg=4326)
        .query("LocationID in @pu_ids")
    )

    zone_ids    = gdf["LocationID"].values
    zone_bounds = gdf.geometry.bounds.apply(
        lambda r: (r.minx, r.miny, r.maxx, r.maxy), axis=1
    ).tolist()

    # ─────────────── 3. Busca espacial na GPU ──────────────────
    pu_lon = cp.asarray(df["PU_longitude"].values)
    pu_lat = cp.asarray(df["PU_latitude"].values)

    mask    = cp.zeros(pu_lon.shape, dtype=bool)
    tmp_ids = cp.full (pu_lon.shape, -1, dtype=cp.int32)

    for zid, (minx, miny, maxx, maxy) in zip(zone_ids, zone_bounds):
        in_box  = (pu_lon >= minx) & (pu_lon <= maxx) & (pu_lat >= miny) & (pu_lat <= maxy)
        mask   |= in_box                      # acumula qualquer acerto
        tmp_ids = cp.where(in_box, zid, tmp_ids)

    #Printa o número de viagens por ID filtrado
    counts = {z: int((tmp_ids == z).sum()) for z in pu_ids}
    print("Contagem de linhas por PULocationID:", counts)

    # ─────────────── 5. Retorno sem colunas extras ─────────────
    return df[cp.asnumpy(mask)].copy()