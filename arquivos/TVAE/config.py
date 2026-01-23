import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Data paths
REAL_DATA_PATH = "data/viagens_lat_long.parquet"
DATETIME_COL = "tpep_pickup_datetime"
PICKUP_ID_COL = "PULocationID"
DROPOFF_ID_COL = "DOLocationID"

# Time filters
FILTER_YEAR = 2024
FILTER_MONTHS = [4, 5]
FILTER_DOW_MIN = 0
FILTER_DOW_MAX = 5
FILTER_START_DATE = None
FILTER_END_DATE = None

# Split fractions (70% treino, 30% teste)
# Com TRAIN_FRAC + VAL_FRAC = 1.0, o hold_df fica vazio e val_df é usado como teste
# O modelo usa val_df tanto para early stopping quanto para avaliação final
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15; TRAIN_FRAC = 0.35

# Model hyperparameters
ENCODER_HIDDEN_DIMS = (256, 128)
DECODER_HIDDEN_DIMS = (128, 128)
LATENT_DIM = 32
BATCH_SIZE = 1024
EPOCHS = 1 #30
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0
PICKUP_LOSS_WEIGHT = 1.2
DROPOFF_LOSS_WEIGHT = 1.2
PAIR_KL_WEIGHT = 1.0
PAIR_KL_EPS = 1e-6
PICKUP_KL_WEIGHT = 1.0
PICKUP_KL_EPS = 1e-6

# KL annealing
KL_BETA_START = 0.0
KL_BETA_END = 1.0
KL_ANNEAL_EPOCHS = 10

# Early stopping
PATIENCE = 8
MIN_DELTA = 1e-4

# Order for autoregressive decoder (fixed to order_1)
ORDERS = {
    "order_1": ["pickup_id", "dropoff_id", "dia_da_semana", "hora_do_dia"],
}
FIXED_ORDER_KEY = "order_1"

OUTPUT_COLUMNS = ["hora_do_dia", "dia_da_semana", "pickup_id", "dropoff_id"]

# Sampling
SAMPLE_TEMPERATURE = 1.0
SAMPLE_ROWS = 20000
EVAL_SAMPLE_RATIO = 1.0  # 100% do teste para comparação equivalente
MAX_EVAL_SAMPLES = None

# Paper metrics
# TIME_KEY_CARDINALITY = 6 * 24 porque domingo (dia 6) está excluído pelo filtro FILTER_DOW_MAX=5
TIME_KEY_CARDINALITY = 6 * 24
TIME_PERIOD_DOW = 6  # Dias 0-5 (seg-sáb)
TIME_PERIOD_HOUR = 24
COVERAGE_K = 5
COVERAGE_MAX_SAMPLES = 100000
COVERAGE_CHUNK_SIZE = 512
COVERAGE_TIME_WEIGHT = 1.0
COVERAGE_SPACE_WEIGHT = 1.0

# Misc
GLOBAL_SEED = 42
NUM_WORKERS = 0

# Output dirs (all analyses/csvs under /home-ext/caioloss/Dados)
OUTPUT_BASE_DIR = Path("outputs")
SAVE_DATA_SUBDIR = "ryc"
SAVE_DATA_DIR = str(OUTPUT_BASE_DIR / "save_data" / SAVE_DATA_SUBDIR)
LOG_DIR = str(OUTPUT_BASE_DIR / "logs")
PLOT_DIR = str(OUTPUT_BASE_DIR / "graficos")

for _d in (SAVE_DATA_DIR, LOG_DIR, PLOT_DIR):
    os.makedirs(_d, exist_ok=True)

# Experiment overrides (train only order_1 with different params)
EXPERIMENTS = {
    "baseline": {},
}
