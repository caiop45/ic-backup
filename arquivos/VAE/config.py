import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# --------------------------------------------------
# CONFIGURAÇÕES GERAIS
# --------------------------------------------------
INPUT_WINDOW         = 4           # number of hours used as DLinear input
SYNTHETIC_MULTIPLIER = 1          # oversampling factor for synthetic data
SAVE_DIR             = str(BASE_DIR / "save_data")
os.makedirs(SAVE_DIR, exist_ok=True)

NUM_RUNS             = 10
DATE_SAMPLER_SEED    = 23          # seed for the date sampler

# Ruídos (Código 1)
COUNT_NOISE_FRAC     = 0.05        # ±5 % na contagem/hora
COORD_SIGMA_DEG      = 0.0005      # ~55 m de σ em lat/lon

# Caminho dos dados reais parquet
REAL_DATA_PATH       = "/home-ext/caioloss/Dados/viagens_lat_long.parquet"
ZONE_CORRELATION_CSV = str(BASE_DIR / "utils" / "correlacao_zone_locationid.csv")
