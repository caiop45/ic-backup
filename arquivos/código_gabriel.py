import osmnx as ox
import pandas as pd
import geopandas as gpd
from pathlib import Path
from matplotlib import pyplot as plt
from tqdm import trange, tqdm
from multiprocessing import Pool
from itertools import repeat
from shapely.geometry import Point, LineString
import networkx as nx

# Verificando a versão do osmnx
ox.__version__

# Carrega o arquivo das zonas
# Esse é meio estranho porque o arquivo das zonas é um zip, o que eu fiz foi extrair tudo dentro dele pra essa pasta (taxi_zones) e carregar ela inteira como se fosse um arquivo, por algum motivo funciona
taxi_zones_df = gpd.read_file(Path("input_data") / Path("taxi_zones"))
print(taxi_zones_df.crs)
taxi_zones_df
taxi_zones_df.plot()
plt.show()

place_name = "Manhattan, New York City, New York, USA"

# Download dos dados da rede viária
G_source = ox.graph_from_place(place_name, network_type='drive')
G_source.graph
ox.projection.is_projected(G_source.graph["crs"])
ox.plot.plot_graph(G_source)
plt.show()

street_widths = {
    "footway": 0.05,
    "steps": 0.05,
    "pedestrian": 0.05,
    "path": 0.05,
    "track": 0.05,
    "service": 0.2,
    "residential": 0.3,
    "primary": 0.5,
    "motorway": 0.6,
}

# Isso aqui é relevante, é o sistema de coordenadas https://epsg.io/2263
# https://en.wikipedia.org/wiki/Spatial_reference_system

# Plotagem da figura base
base = ox.plot.plot_figure_ground(G_source, street_widths=street_widths, default_width=0.1, dpi=300, dist=30000)
base
plt.show()

G = ox.projection.project_graph(G_source, to_crs="EPSG:2263")
G

# Aqui pegamos as velocidades das ruas e os tempos de deslocamento
G = ox.routing.add_edge_speeds(G)
G = ox.routing.add_edge_travel_times(G)
G

# Converte o grafo em GeoDataFrames (nodos e arestas)
nodes, edges = ox.graph_to_gdfs(G)
nodes.loc[42421728]
nodes

# Por conveniência, vamos resetar os índices dos nodos para começarem em 0
node_ids = nodes.index.values
new_node_ids = list(range(len(node_ids)))
id_mapping = dict(zip(list(node_ids), new_node_ids))
nodes = nodes.reset_index()
nodes["node_index"] = nodes["osmid"].map(id_mapping)
nodes

# Junção espacial para associar cada nodo com a zona à qual ele pertence
# Também tem uns renomes para ficar com os nomes que o FleetPy esperava, mas o importante mesmo é o sjoin
nodes = nodes.sjoin_nearest(
    taxi_zones_df[["OBJECTID", "zone", "borough", "geometry"]],
    how="left"
).drop(columns=["index_right"]).rename(columns={"OBJECTID": "ZoneID"})

# Agora tem uma coluna ZoneID, uma zone e uma borough aqui também
nodes
edges

# Agora vamos arrumar as arestas para apontarem para os índices novos e para ficarem mais legíveis
edges = edges.rename(index=id_mapping).reset_index()
edges = edges.rename(columns={
    "length": "distance",
    "u": "from_node",
    "v": "to_node",
    "highway": "road_type",
})
edges

# Carregamos os dados de demanda
demand_df = pd.read_parquet(Path("input_data") / Path("yellow_tripdata_2023-10.parquet"))
demand_df

# Estrutura de diretórios esperada (comentários informativos)
# data/zones/
# data/zones/{zone_system_name}/
# data/zones/{zone_system_name}/general_information.csv
# data/zones/{zone_system_name}/polygon_definition.geojson
# data/zones/{zone_system_name}/crs.info
# data/zones/{zone_system_name}/{network_name}/
# data/zones/{zone_system_name}/{network_name}/node_zone_info.csv
# data/zones/{zone_system_name}/{network_name}/edge_zone_info.csv

# Função para gerar uma viagem possível a partir de um registro de viagem zona-a-zona. Usa dados de viagens de táxi de NYC.
def generate_possible_trip(data=None, nodes_with_zones=None, record=None):
    if data:
        record, nodes_with_zones = data
        record = record[1]
    req_time = record["tpep_pickup_datetime"]
    from_ZoneID = record["PULocationID"]
    to_ZoneID = record["DOLocationID"]
    # Obtém possíveis nodos de início para a viagem (qualquer nodo na zona de partida)
    possible_nodes = nodes_with_zones[nodes_with_zones["ZoneID"] == from_ZoneID]
    # Retorna None se a zona estiver vazia ou inválida
    if from_ZoneID > 263 or to_ZoneID > 263 or len(possible_nodes) == 0:
        return None
    # Amostra aleatoriamente um nodo inicial
    random_start_node = possible_nodes.sample(1).reset_index()
    # Verifica se os dados de distância da viagem são válidos. Se sim, usamos para escolher o nodo final.
    if not pd.isna(record["trip_distance"]) and record["trip_distance"] > 0.0001:
        # As distâncias de viagem estão em milhas e o CRS está em metros: usamos 1.609344 * 1000 como fator de conversão para obter metros.
        # Calcula as distâncias entre o nodo inicial e cada nodo na zona de destino e
        # subtrai o comprimento da viagem para estimar nodos de destino razoáveis.
        distances = (nodes_with_zones[nodes_with_zones["ZoneID"] == to_ZoneID].distance(
            random_start_node.iloc[0]["geometry"]) - record["trip_distance"] * 1.609344 * 1000
                     ).abs()
        # Obtém os 5 nodos mais prováveis
        closest_nodes = nodes_with_zones.loc[distances[distances > 0].nsmallest(5).index]
        # Verifica se o resultado faz sentido
        if len(closest_nodes) > 0:
            # Amostra aleatoriamente um dos 5 nodos prováveis como nodo final e retorna a viagem gerada nodo-a-nodo
            random_end_node = closest_nodes.sample(1).reset_index()
            return {
                        "start": random_start_node.iloc[0]["node_index"],
                        "end": random_end_node.iloc[0]["node_index"],
                        "rq_time": req_time,
                   }
    # Opção totalmente aleatória que retorna um nodo aleatório da zona de destino, desconsiderando a distância
    # A função chega aqui se alguma das verificações anteriores falhou
    possible_end_nodes = nodes_with_zones[nodes_with_zones["ZoneID"] == to_ZoneID]
    random_end_node = random_start_node
    # Evita escolher o mesmo nodo de início e fim caso as zonas de partida e destino sejam idênticas
    while random_end_node.iloc[0]["osmid"] == random_start_node.iloc[0]["osmid"]:
        random_end_node = possible_end_nodes.sample(1).reset_index()
    return {
                "start": random_start_node.iloc[0]["node_index"],
                "end": random_end_node.iloc[0]["node_index"],
                "rq_time": req_time,
            }

n_trips_orig = demand_df.shape[0]

print("Filtrando viagens para a área da rede...")
# Filtra viagens: exclui viagens que têm pickups ou dropoffs fora das zonas vistas nos nodos da rede.
seen_ZoneIDs = nodes["ZoneID"].unique()
demand_df = demand_df[
    demand_df["PULocationID"].isin(seen_ZoneIDs) & demand_df["DOLocationID"].isin(seen_ZoneIDs)]
n_trips_first_filter = demand_df.shape[0]
print(f"Viagens filtradas. Novo total de viagens é {n_trips_first_filter}. ({n_trips_orig - n_trips_first_filter} viagens removidas)")

# Filtra tempos de pickup para apenas dentro do mês esperado
year, month = 2023, 10
print(f"Removendo viagens fora do mês esperado {month:>02}/{year}")
demand_df = demand_df[demand_df["tpep_pickup_datetime"] >= pd.Timestamp(year=year, month=month, day=1, hour=0)]
demand_df = demand_df[demand_df["tpep_pickup_datetime"] < pd.Timestamp(year=(year if month != 12 else year + 1), month=month % 12 + 1, day=1, hour=0)]
demand_df = demand_df.reset_index(drop=True)
print(f"Feito. Novo total de viagens é {demand_df.shape[0]}. ({n_trips_first_filter - demand_df.shape[0]} viagens removidas)")

generate_possible_trip(record=demand_df.iloc[0], nodes_with_zones=nodes)

# Gerador de arquivos de dia
# Produz um arquivo por dia, com uma viagem gerada para cada linha do demand_df daquele dia
# Só estava fazendo desse jeito para gerar um arquivo com todas as viagens separadas por dia para facilitar nas simulações,
# se você quiser gerar viagens arbitrárias é só chamar a generate_possible_trip com data = (a linha do demand_df que corresponde à viagem, nodes)

print("Preparando dados para geração de viagens...")
# Ordena as viagens por data e hora de pickup
sorted_demand = demand_df.sort_values(by="tpep_pickup_datetime").reset_index(drop=True)

# Obtém o número de viagens por dia para a barra de progresso do TQDM
per_day_counts = sorted_demand['tpep_pickup_datetime'].dt.day.value_counts().sort_index()
per_day = [per_day_counts.get(day, 0) for day in range(1, 32)]

# Pode ignorar que aqui está usando multiprocessing, só estava aproveitando o Beluga que tem tipo 100 cores para rodar mais rápido

out_path = Path("generated_demand_data2")
out_path.mkdir(exist_ok=True)

print(f"Iniciando geração de viagens. Arquivos serão salvos em {out_path}")
for day in trange(0, 30):
    with Pool(3) as p:
        to_process = sorted_demand.iloc[sum(per_day[:day]):sum(per_day[:day+1])].iterrows()
        generated_trips = list(tqdm(p.imap(
            generate_possible_trip, zip(to_process, repeat(nodes)), chunksize=200,
        ), total=per_day[day]))
    trips = pd.DataFrame(generated_trips).sort_values(by="rq_time").reset_index(drop=True)
    trips.to_pickle(out_path / Path(f"generated_trips_{year}-{month}-{day + 1}.pkl"))
print("Feito.")
print("IMPORTANTE: Estes não são arquivos de demanda do FleetPy! Execute o gerador de arquivos de demanda em seguida para criar arquivos do FleetPy.")
