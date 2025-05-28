import cupy as cp
import pandas as pd
import geopandas as gpd

def add_location_ids_cupy(
    df,
    taxi_zones_path='/home-ext/caioloss/Dados/taxi-zones',
    pu_id=None,
    do_id=None,
):
    # ──────────────────────────────────────────────────────────────
    # 1. GeoDataFrame: agora também carregamos 'zone'
    # ──────────────────────────────────────────────────────────────
    gdf = gpd.read_file(taxi_zones_path)[['LocationID', 'zone', 'geometry']]
    gdf = gdf.to_crs(epsg=4326)

    # dicionário para mapear ID → nome da zona
    id2zone = gdf.set_index('LocationID')['zone'].to_dict()

    # bounding-boxes e IDs (sem mudanças)
    gdf['bounds'] = gdf.geometry.bounds.apply(
        lambda row: (row.minx, row.miny, row.maxx, row.maxy), axis=1
    )
    zone_ids   = gdf['LocationID'].values
    zone_bounds = gdf['bounds'].tolist()

    # ──────────────────────────────────────────────────────────────
    # 2. Parte pesada na GPU (sem mudanças)
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
    # 3. Filtros ainda funcionam com IDs numéricos (sem mudanças)
    # ──────────────────────────────────────────────────────────────
    if pu_id is not None:
        df = df[df['PULocationID'] == pu_id]
    elif do_id is not None:
        df = df[df['DOLocationID'] == do_id]

    # ──────────────────────────────────────────────────────────────
    # 4. **Única alteração funcional**:
    #    convertemos IDs → nomes da zona antes de retornar
    # ──────────────────────────────────────────────────────────────
    df['PULocationID'] = df['PULocationID'].map(id2zone)
    df['DOLocationID'] = df['DOLocationID'].map(id2zone)

    return df
