import pandas as pd
import geopandas as gpd

def add_location_ids(df, taxi_zones_path='/home-ext/caioloss/Dados/taxi-zones'):
    """
    Esta função recebe um DataFrame com as colunas:
      - hora_do_dia
      - PU_longitude
      - PU_latitude
      - DO_longitude
      - DO_latitude
    e adiciona duas colunas:
      - PULocationID
      - DOLocationID
    que correspondem aos IDs das zonas de pickup e dropoff, respectivamente,
    determinados com base na interseção dos pontos com as bounding boxes dos taxi zones.
    
    Parâmetros:
      df: pandas.DataFrame
          DataFrame de entrada com as colunas de coordenadas.
      taxi_zones_path: str, opcional
          Caminho para o shapefile dos taxi zones (padrão: '/home-ext/caioloss/Dados/taxi-zones').
    
    Retorno:
      pandas.DataFrame com as colunas 'PULocationID' e 'DOLocationID' adicionadas.
    """
    # Carrega o shapefile dos taxi zones e mantém apenas as colunas necessárias
    gdf = gpd.read_file(taxi_zones_path)[['LocationID', 'geometry']]
    gdf = gdf.to_crs(epsg=4326)  # Garante que o CRS seja o EPSG:4326

    # Calcula a bounding box para cada polígono (minx, miny, maxx, maxy)
    gdf['bounds'] = gdf['geometry'].bounds.apply(lambda row: (row.minx, row.miny, row.maxx, row.maxy), axis=1)

    # Cria um dicionário: chave = LocationID e valor = bounding box
    zone_bounds_dict = dict(zip(gdf['LocationID'], gdf['bounds']))

    # Função auxiliar para identificar o LocationID a partir de uma coordenada (lon, lat)
    def get_zone_id_from_point(lon, lat):
        for zone_id, (minx, miny, maxx, maxy) in zone_bounds_dict.items():
            if (minx <= lon <= maxx) and (miny <= lat <= maxy):
                return zone_id
        return None

    # Aplica a função para identificar a zona de pickup e dropoff para cada linha do DataFrame
    df['PULocationID'] = df.apply(
        lambda row: get_zone_id_from_point(row['PU_longitude'], row['PU_latitude']), axis=1
    )
    df['DOLocationID'] = df.apply(
        lambda row: get_zone_id_from_point(row['DO_longitude'], row['DO_latitude']), axis=1
    )
    
    return df
