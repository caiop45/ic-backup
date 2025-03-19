import time
import geopandas as gpd
import pandas as pd
import random
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Inicia contagem de tempo total (opcional, se quiser medir o total)
tempo_inicio_total = time.time()

# ===============================
# 1. Leitura dos dados de viagens
# ===============================
tempo_inicio = time.time()
parquet_files = [
    '/home-ext/caioloss/Dados/yellow_tripdata_2024-01.parquet'
    # Dá pra adicionar a viagem do resto dos meses aqui
]
df = pd.concat([pd.read_parquet(file) for file in parquet_files], ignore_index=True)
df = df[['tpep_pickup_datetime', 'PULocationID', 'DOLocationID']]
print(f"[1] Tempo de leitura dos dados de viagens: {time.time() - tempo_inicio:.2f} segundos")

# =================================
# 2. Leitura dos dados de taxi-zones
# =================================
tempo_inicio = time.time()
gdf = gpd.read_file('/home-ext/caioloss/Dados/taxi-zones')
gdf = gdf[['LocationID', 'geometry']]
print(gdf.crs)
gdf = gdf.to_crs(epsg=4326)
print(gdf.crs)  # Agora deve exibir EPSG:4326
print(f"[2] Tempo de leitura dos dados de taxi-zones: {time.time() - tempo_inicio:.2f} segundos")

# =================================
# 3. Merge das geometrias de embarque e desembarque
# =================================
tempo_inicio = time.time()
# Pega as coordenadas da zona de embarque
df = df.merge(
    gdf, 
    how='left', 
    left_on='PULocationID', 
    right_on='LocationID'
).rename(columns={'geometry': 'PU_Geometry'})

df.drop(columns=['LocationID'], inplace=True)

# Pega as coordenadas da zona de desembarque
df = df.merge(
    gdf, 
    how='left', 
    left_on='DOLocationID', 
    right_on='LocationID'
).rename(columns={'geometry': 'DO_Geometry'})
df.drop(columns=['LocationID'], inplace=True)

viagem_por_geometry = gpd.GeoDataFrame(df, geometry='PU_Geometry')
print(f"[3] Tempo de merge das geometrias: {time.time() - tempo_inicio:.2f} s")

# ===========================================
# 4. Pré-processar e armazenar bounds em RAM
# ===========================================
tempo_inicio = time.time()

zones_df = gdf[['LocationID', 'geometry']].copy()

# Calcula bounding box (minx, miny, maxx, maxy) para cada polígono
zones_df['bounds'] = zones_df['geometry'].bounds.apply(lambda row: (row.minx, row.miny, row.maxx, row.maxy), axis=1)

# Transforma num dicionário: zone_id -> (minx, miny, maxx, maxy)
zone_bounds_dict = dict(zip(zones_df['LocationID'], zones_df['bounds']))

print(f"[4] Tempo para pré-processar zones_df: {time.time() - tempo_inicio:.2f} s")

# ===========================================
# 4. Pré-processar e armazenar centroides
# ===========================================
tempo_inicio = time.time()

zones_df = gdf[['LocationID', 'geometry']].copy()
zones_df['centroid'] = zones_df['geometry'].centroid

# Cria um dicionário: zone_id -> centroide (objeto shapely Point)
zone_centroid_dict = dict(zip(zones_df['LocationID'], zones_df['centroid']))

print(f"[4] Tempo para pré-processar zones_df (centroides): {time.time() - tempo_inicio:.2f} s")

# ===========================================
# 5. Gerar coordenadas a partir dos centroides
# ===========================================
tempo_inicio = time.time()

final_data = []
num_rows = len(df)
for idx, row in df.iterrows():
    pu_id = row['PULocationID']
    do_id = row['DOLocationID']
    
    # Usa o centroide para PickUp
    if pu_id in zone_centroid_dict:
        pu_lon = zone_centroid_dict[pu_id].x
        pu_lat = zone_centroid_dict[pu_id].y
    else:
        pu_lon = None
        pu_lat = None
        
    # Usa o centroide para DropOff
    if do_id in zone_centroid_dict:
        do_lon = zone_centroid_dict[do_id].x
        do_lat = zone_centroid_dict[do_id].y
    else:
        do_lon = None
        do_lat = None

    final_data.append({
        "tpep_pickup_datetime": row["tpep_pickup_datetime"],
        "PULocationID": pu_id,
        "DOLocationID": do_id,
        "PU_longitude": pu_lon,
        "PU_latitude":  pu_lat,
        "DO_longitude": do_lon,
        "DO_latitude":  do_lat
    })

dados_taxi = pd.DataFrame(final_data)
print(f"[5] Tempo para gerar as coordenadas (centroides): {time.time() - tempo_inicio:.2f} s")
dados_taxi.to_parquet('viagens_lat_long.parquet', index=False)