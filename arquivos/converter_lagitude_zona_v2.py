import cupy as cp
import pandas as pd
import geopandas as gpd

def add_location_ids_cupy(df, taxi_zones_path='/home-ext/caioloss/Dados/taxi-zones', pu_id=None, do_id=None):
    """
    Versão otimizada utilizando CuPy para realizar os cálculos na GPU.
    
    Parâmetros:
      df: pandas.DataFrame com as colunas de coordenadas:
          - PU_longitude, PU_latitude
          - DO_longitude, DO_latitude
      taxi_zones_path: caminho para o shapefile dos taxi zones.
      pu_id: (opcional) Filtro para retornar somente as linhas cujo PULocationID seja igual ao valor fornecido.
      do_id: (opcional) Filtro para retornar somente as linhas cujo DOLocationID seja igual ao valor fornecido.
      
      Observação: Nunca serão passados pu_id e do_id simultaneamente, apenas um ou outro.
      
    Retorna:
      DataFrame com as colunas 'PULocationID' e 'DOLocationID' preenchidas e, se solicitado, filtrado.
    """
    # Carrega o shapefile e converte o CRS para EPSG:4326
    gdf = gpd.read_file(taxi_zones_path)[['LocationID', 'geometry']]
    gdf = gdf.to_crs(epsg=4326)
    
    # Calcula a bounding box para cada polígono (minx, miny, maxx, maxy)
    gdf['bounds'] = gdf['geometry'].bounds.apply(
        lambda row: (row.minx, row.miny, row.maxx, row.maxy), axis=1
    )
    
    # Extraindo IDs e as bounding boxes
    zone_ids = gdf['LocationID'].values
    zone_bounds = gdf['bounds'].tolist()  # lista de tuplas
    
    # Converte as colunas de coordenadas para arrays do CuPy (GPU)
    pu_lon = cp.asarray(df['PU_longitude'].values)
    pu_lat = cp.asarray(df['PU_latitude'].values)
    do_lon = cp.asarray(df['DO_longitude'].values)
    do_lat = cp.asarray(df['DO_latitude'].values)
    
    # Inicializa os arrays para armazenar os IDs; -1 indica ausência
    pu_ids = cp.full(pu_lon.shape, -1, dtype=cp.int32)
    do_ids = cp.full(do_lon.shape, -1, dtype=cp.int32)
    
    # Para cada zona, cria as máscaras vetorizadas e atualiza os resultados
    for idx, (minx, miny, maxx, maxy) in enumerate(zone_bounds):
        zone_id = zone_ids[idx]
        # Máscara para pontos de pickup
        mask_pu = (pu_lon >= minx) & (pu_lon <= maxx) & (pu_lat >= miny) & (pu_lat <= maxy)
        pu_ids = cp.where(mask_pu, zone_id, pu_ids)
        
        # Máscara para pontos de dropoff
        mask_do = (do_lon >= minx) & (do_lon <= maxx) & (do_lat >= miny) & (do_lat <= maxy)
        do_ids = cp.where(mask_do, zone_id, do_ids)
    
    # Transfere os resultados da GPU para a CPU e atribui as novas colunas ao DataFrame
    df['PULocationID'] = cp.asnumpy(pu_ids)
    df['DOLocationID'] = cp.asnumpy(do_ids)
    
    # Realiza a filtragem do DataFrame, se necessário
    if pu_id is not None:
        df = df[df['PULocationID'] == pu_id]
    elif do_id is not None:
        df = df[df['DOLocationID'] == do_id]
    
    return df
