import os

# --- Caminhos ---
# !! IMPORTANTE: Ajuste este caminho para o local correto no seu sistema !!
BASE_DIR = "/home/caioloss/" # Ou o diretório base do seu projeto
DATA_FILE_PATH = r"/home-ext/caioloss/Dados/viagens_lat_long.parquet"
SAVE_DIR = os.path.join(BASE_DIR, "gráficos/linear_modular/") # Diretório para salvar resultados

# --- Parâmetros do Experimento ---
SYNTHETIC_MULTIPLIER = 5
NUM_EXECUCOES = 10
NCS_LIST = [25, 30, 35]  # Lista de número de componentes GMM a testar
EPOCHS_LIST = [50, 100, 150, 200] # Lista de épocas a testar

# --- Parâmetros do Modelo e Treinamento ---
WINDOW_SIZE = 4           # Janela para criar sequências
LEARNING_RATE = 0.001
TRAIN_SPLIT_RATIO = 0.7   # Proporção para divisão treino/validação

# --- Colunas a serem usadas ---
FEATURES_GMM = [
    'data_do_dia',
    'hora_do_dia',
    'PU_longitude',
    'PU_latitude',
    'DO_longitude',
    'DO_latitude'
]

# --- Cria o diretório de salvamento se não existir ---
os.makedirs(SAVE_DIR, exist_ok=True)

print(f"Diretório de salvamento: {SAVE_DIR}")
print(f"Arquivo de dados: {DATA_FILE_PATH}")
if not os.path.exists(DATA_FILE_PATH):
    print(f"AVISO: Arquivo de dados não encontrado em {DATA_FILE_PATH}")