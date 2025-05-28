import os

# --------------------------------------------------
# CONFIGURAÇÕES GERAIS
# --------------------------------------------------
WINDOW               = 4           # nº de horas usadas como entrada do DLinear
SYNTHETIC_MULTIPLIER = 2           # fator de oversample p/ dados sintéticos
SAVE_DIR             = "/home/caioloss/gráficos/dlinear/"
os.makedirs(SAVE_DIR, exist_ok=True)

NUM_EXECUCOES        = 5
DATA_SAMPLER_SEED    = 35          # seed para o date-sampler

# Ruídos (Código 1)
COUNT_NOISE_FRAC     = 0.05        # ±5 % na contagem/hora
COORD_SIGMA_DEG      = 0.0005      # ~55 m de σ em lat/lon

# Caminho dos dados reais parquet
REAL_DATA_PATH       = "/home-ext/caioloss/Dados/viagens_lat_long.parquet"
