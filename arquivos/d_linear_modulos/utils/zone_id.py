import cupy as cp
import pandas as pd
import geopandas as gpd

import cupy as cp
import pandas as pd
import geopandas as gpd
from collections.abc import Iterable  # ⇦ novo

def assign_zone_names(
    df: pd.DataFrame,
    taxi_zones_path: str = '/home-ext/caioloss/Dados/taxi-zones',
    pu_id: int | Iterable[int] | None = None,   # ⇦ aceita int ou iterável
    do_id: int | Iterable[int] | None = None,   # ⇦ idem
) -> pd.DataFrame:
    """
    Adiciona as colunas PULocationID/DOLocationID (nomes das zonas) e,
    opcionalmente, filtra o DataFrame pelos IDs fornecidos.

    Parameters
    ----------
    df : DataFrame
        Dados com colunas de latitude/longitude de PU e DO.
    taxi_zones_path : str
        Caminho para o shapefile GeoJSON das zonas de táxi da TLC.
    pu_id, do_id : int ou iterável de int, opcional
        IDs a serem mantidos. Se pu_id for passado filtra por PU; caso
        contrário, se do_id for passado filtra por DO.
    """
    # ──────────────────────────────────────────────────────────────
    # 1. GeoDataFrame + mapeamento ID → nome de zona
    # ──────────────────────────────────────────────────────────────
    gdf = gpd.read_file(taxi_zones_path)[['LocationID', 'zone', 'geometry']]
    gdf = gdf.to_crs(epsg=4326)

    id2zone = gdf.set_index('LocationID')['zone'].to_dict()

    gdf['bounds'] = gdf.geometry.bounds.apply(
        lambda row: (row.minx, row.miny, row.maxx, row.maxy), axis=1
    )
    zone_ids    = gdf['LocationID'].values
    zone_bounds = gdf['bounds'].tolist()

    # ──────────────────────────────────────────────────────────────
    # 2. Busca espacial na GPU
    # ──────────────────────────────────────────────────────────────
    pu_lon = cp.asarray(df['PU_longitude'].values)
    pu_lat = cp.asarray(df['PU_latitude'].values)
    do_lon = cp.asarray(df['DO_longitude'].values)
    do_lat = cp.asarray(df['DO_latitude'].values)

    pu_ids = cp.full(pu_lon.shape, -1, dtype=cp.int32)
    do_ids = cp.full(do_lon.shape, -1, dtype=cp.int32)

    for idx, (minx, miny, maxx, maxy) in enumerate(zone_bounds):
        zid = zone_ids[idx]
        pu_ids = cp.where(
            (pu_lon >= minx) & (pu_lon <= maxx) & (pu_lat >= miny) & (pu_lat <= maxy),
            zid,
            pu_ids,
        )
        do_ids = cp.where(
            (do_lon >= minx) & (do_lon <= maxx) & (do_lat >= miny) & (do_lat <= maxy),
            zid,
            do_ids,
        )

    # volta para CPU
    df['PULocationID'] = cp.asnumpy(pu_ids)
    df['DOLocationID'] = cp.asnumpy(do_ids)

    # ──────────────────────────────────────────────────────────────
    # 3. Filtro flexível por lista ou valor único  ← ALTERAÇÃO
    # ──────────────────────────────────────────────────────────────
    def _normalize(x):
        """Converte None → None, int → [int], iterável → list(iterável)."""
        if x is None:
            return None
        # string não é aceita; precisamos de ints
        if isinstance(x, int):
            return [x]
        if isinstance(x, Iterable):
            return list(x)
        raise TypeError("pu_id/do_id devem ser int ou iterável de int")

    pu_ids_filter = _normalize(pu_id)
    do_ids_filter = _normalize(do_id)

    if pu_ids_filter is not None:
        df = df[df['PULocationID'].isin(pu_ids_filter)].copy()
    elif do_ids_filter is not None:
        df = df[df['DOLocationID'].isin(do_ids_filter)].copy()

    # ──────────────────────────────────────────────────────────────
    # 4. Converte IDs numéricos → nomes das zonas
    # ──────────────────────────────────────────────────────────────
    df['PULocationID'] = df['PULocationID'].map(id2zone)
    df['DOLocationID'] = df['DOLocationID'].map(id2zone)

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