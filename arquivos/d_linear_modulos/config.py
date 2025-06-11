import os

# --------------------------------------------------
# CONFIGURAÇÕES GERAIS
# --------------------------------------------------
WINDOW               = 4           # nº de horas usadas como entrada do DLinear
SYNTHETIC_MULTIPLIER = 1           # fator de oversample p/ dados sintéticos
SAVE_DIR             = "/home/caioloss/arquivos/d_linear_modulos/save_data"
os.makedirs(SAVE_DIR, exist_ok=True)

NUM_EXECUCOES        = 10
DATA_SAMPLER_SEED    = 35          # seed para o date-sampler

# Ruídos (Código 1)
COUNT_NOISE_FRAC     = 0.05        # ±5 % na contagem/hora
COORD_SIGMA_DEG      = 0.0005      # ~55 m de σ em lat/lon

# Caminho dos dados reais parquet
REAL_DATA_PATH       = "/home-ext/caioloss/Dados/viagens_lat_long.parquet"
ZONE_CRR = '/home/caioloss/arquivos/d_linear_modulos/utils/correlacao_zone_locationid.csv'
