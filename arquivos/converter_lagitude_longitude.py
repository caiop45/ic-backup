import time
import geopandas as gpd
import pandas as pd
import random
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pyproj import Transformer

# Inicia contagem de tempo total (opcional, se quiser medir o total)
tempo_inicio_total = time.time()

# ===============================
# 1. Leitura dos dados de viagens
# ===============================
tempo_inicio = time.time()
parquet_files = [
    '/home-ext/caioloss/Dados/yellow_tripdata_2024-04.parquet',
    '/home-ext/caioloss/Dados/yellow_tripdata_2024-05.parquet'
    # Dá pra adicionar a viagem do resto dos meses aqui
]
df = pd.concat([pd.read_parquet(file) for file in parquet_files], ignore_index=True)
required_cols = [
    'tpep_pickup_datetime',
    'PULocationID',
    'DOLocationID',
    'passenger_count',
    'total_amount',
]
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    raise ValueError(f"Columns missing in source parquet files: {missing_cols}")
df = df[required_cols]
print(f"[1] Tempo de leitura dos dados de viagens: {time.time() - tempo_inicio:.2f} segundos")

df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])
mes_counts = (
    df["tpep_pickup_datetime"]
      .dt.to_period("M")
      .value_counts()
      .sort_index()
      .reindex(pd.period_range("2024-01", "2024-12", freq="M"), fill_value=0)
    )

print("\n# Linhas por mês (2024)")
for mes, n in mes_counts.items():
    print(f"{mes}: {n}")

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

# Drop geometry helper columns used only for merge; keep only plain columns for parquet output.
df = df.drop(columns=[col for col in ("PU_Geometry", "DO_Geometry") if col in df.columns])

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

# Use projected centroids, then convert to WGS84 lon/lat with pyproj.
zones_df = zones_df.to_crs("EPSG:3857")
zones_df["centroid"] = zones_df.geometry.centroid
mercator_to_wgs84 = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)

# Cria dicionários: zone_id -> centroid lon/lat in graus.
zone_lon_dict = {}
zone_lat_dict = {}
for zone_id, centroid in zip(zones_df["LocationID"], zones_df["centroid"]):
    if centroid is None or pd.isna(centroid.x) or pd.isna(centroid.y):
        zone_lon_dict[int(zone_id)] = float("nan")
        zone_lat_dict[int(zone_id)] = float("nan")
    else:
        lon, lat = mercator_to_wgs84.transform(float(centroid.x), float(centroid.y))
        zone_lon_dict[int(zone_id)] = float(lon)
        zone_lat_dict[int(zone_id)] = float(lat)

print(f"[4] Tempo para pré-processar zones_df (centroides): {time.time() - tempo_inicio:.2f} s")

# ===========================================
# 5. Gerar coordenadas a partir dos centroides
# ===========================================
tempo_inicio = time.time()

dados_taxi = df.copy()
dados_taxi["PU_longitude"] = dados_taxi["PULocationID"].map(zone_lon_dict)
dados_taxi["PU_latitude"] = dados_taxi["PULocationID"].map(zone_lat_dict)
dados_taxi["DO_longitude"] = dados_taxi["DOLocationID"].map(zone_lon_dict)
dados_taxi["DO_latitude"] = dados_taxi["DOLocationID"].map(zone_lat_dict)

print(f"[5] Tempo para gerar as coordenadas (centroides): {time.time() - tempo_inicio:.2f} s")
dados_taxi.to_parquet('data/viagens_lat_long.parquet', index=False)
