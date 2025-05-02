import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pycave.bayes import GaussianMixture
from sklearn.preprocessing import StandardScaler
from converter_lagitude_zona_v2 import add_location_ids_cupy
import geopandas as gpd

# --------------------------------------------------
# CONFIGURAÇÕES INICIAIS
# --------------------------------------------------
SYNTHETIC_MULTIPLIER = 2
SAVE_DIR = "/home/caioloss/gráficos/linear/"
os.makedirs(SAVE_DIR, exist_ok=True)
NUM_EXECUCOES = 5
data_sampler_seed = 30 

# --------------------------------------------------
# 1) LEITURA E PREPARAÇÃO INICIAL DOS dados_reais REAIS 
# --------------------------------------------------
dados_reais = pd.read_parquet('/home-ext/caioloss/Dados/viagens_lat_long.parquet')
#dados_reais = dados_reais.sample(frac = 0.1)
dados_reais['tpep_pickup_datetime'] = pd.to_datetime(dados_reais['tpep_pickup_datetime'])
dados_reais['hora_do_dia'] = dados_reais['tpep_pickup_datetime'].dt.hour
dados_reais['dia_da_semana'] = dados_reais['tpep_pickup_datetime'].dt.dayofweek
dados_reais['num_viagens'] = 1
dados_reais = dados_reais[dados_reais['dia_da_semana'].between(0, 2)]

features = [
    'hora_do_dia',
    'PU_longitude',
    'PU_latitude',
    'DO_longitude',
    'DO_latitude'
]

features2 = [
    'tpep_pickup_datetime',
    'hora_do_dia',
    'num_viagens',
    'PU_longitude',
    'PU_latitude',
    'DO_longitude',
    'DO_latitude'
]
dados_reais_gmm = dados_reais[features].dropna()
dados_reais = dados_reais[features2].dropna()
# --------------------------------------------------
# FUNÇÕES AUXILIARES
# --------------------------------------------------



def make_date_sampler(df_real: pd.DataFrame,
                      ts_col: str = "tpep_pickup_datetime",
                      seed: int | None = None):
    """
    Cria um *sampler* que devolve uma data (YYYY-MM-DD) de acordo com a
    distribuição empírica P(data | hora) observada nos dados reais.

    Parameters
    ----------
    df_real : DataFrame
        Contém a coluna `ts_col` em formato datetime64[ns].
    ts_col : str, default="tpep_pickup_datetime"
        Nome da coluna de timestamp completo.
    seed : int | None
        Semente para reprodutibilidade (torch.manual_seed já cuida da rede;
        aqui controlamos apenas a amostragem de datas).

    Returns
    -------
    sample_date : callable
        Função que recebe `hora_do_dia` (int 0-23) e devolve uma
        `datetime.date` sorteada segundo P(data | hora).
    """
    # 1) Quebra o timestamp em partes
    df = df_real.copy()
    df["hora"]  = df[ts_col].dt.hour
    df["data"]  = df[ts_col].dt.normalize()        # zera HH:MM:SS
    rng = np.random.default_rng(12345)

    # 2) Constrói as tabelas de probabilidade por hora
    prob_table: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for h, sub in df.groupby("hora"):
        counts = sub["data"].value_counts().sort_index()
        dates  = counts.index.to_numpy()           # array de datas
        probs  = counts.values / counts.values.sum()
        prob_table[h] = (dates, probs)

    # 3) Closure que amostra usando a tabela pronta
    def sample_date(hora_do_dia: int):
        """
        Devolve uma data coerente com a distribuição P(data|hora).
        Se a hora não existir na prob_table, retorna 0.
        """
        try:
            dates, probs = prob_table[hora_do_dia]     # KeyError se não existir
            return rng.choice(dates, p=probs)
        except KeyError:
            return 0           #Retorna 0 para entrar como 1970-01-01 e ser filtrado

    return sample_date

#Considera as 4 horas anteriores para poder prever o número de viagens da próxima hora
def create_sequence_dataset(series_values, window_size=4):
    X, y = [], []
    for i in range(len(series_values) - window_size):
        print(i)
        X.append(series_values[i:i+window_size])
        y.append(series_values[i+window_size])
    print(np.array(X))
    print("---------")
    print(np.array(y))    
    return np.array(X), np.array(y)

class DLinear(nn.Module):
    def __init__(self, input_len=4, output_dim=1):
        super(DLinear, self).__init__()
        self.linear = nn.Linear(input_len, output_dim)
    
    def forward(self, x):
        return self.linear(x)

def train_model(
    model,
    X_train, y_train,
    X_val,   y_val,
    epochs,
    lr=1e-3,
    patience=10,
    min_delta=1e-4
):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val   = float("inf")   # menor perda já vista
    best_state = None           # checkpoint dos pesos
    wait       = 0              # épocas sem melhora

    for epoch in range(epochs):
        # ---------- treino ----------
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_train), y_train)
        loss.backward()
        optimizer.step()

        # ---------- validação ----------
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val), y_val)

        # ---------- early stopping ----------
        if val_loss.item() < best_val - min_delta:
            best_val   = val_loss.item()
            best_state = model.state_dict()  # guarda pesos atuais
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"⏹️  Early stopping na época {epoch+1} – "
                      f"val_loss não melhora há {patience} épocas.")
                break

        # log a cada 10 épocas (opcional)
        if (epoch + 1) % 10 == 0:
            #print(f"Epoch {epoch+1:03}/{epochs}  • "
                  #f"train={loss.item():.4f}  •  val={val_loss.item():.4f}")
            1 == 1

    # restaura melhores pesos
    if best_state is not None:
        model.load_state_dict(best_state)


# Dataframe para armazenar resultados
df_resultados = pd.DataFrame(columns=[
    'tipo_dado', 'metrica', 'valor', 'nc', 'epochs', 'seed'
])

# --------------------------------------------------
# LOOP PRINCIPAL DE EXPERIMENTOS
# --------------------------------------------------
sample_date = make_date_sampler(dados_reais, seed = data_sampler_seed)  
for nc in [25, 30, 35]:
    dados_reais_copy = dados_reais.copy()

    # 1) Treinar GMM e gerar dados sintéticos
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(dados_reais_gmm).astype(np.float32)

    gmm = GaussianMixture(
        num_components=nc,
        covariance_type='full',
        covariance_regularization=1e-5, 
        trainer_params={'max_epochs': 100, 'accelerator': 'gpu', 'devices': 1}
    )
    gmm.fit(train_scaled)

    synthetic_scaled = gmm.sample(int(len(dados_reais_gmm) * SYNTHETIC_MULTIPLIER)).cpu().numpy()
    synthetic_df = pd.DataFrame(
        scaler.inverse_transform(synthetic_scaled),
        columns=features
    )
    synthetic_df['num_viagens'] = 1

    ##Tratamentos dados sintéticos
    synthetic_df = (
    synthetic_df
        .assign(hora_do_dia = lambda df: np.floor(pd.to_numeric(df['hora_do_dia'],
                                                                errors='coerce')).astype(int))
        .query("hora_do_dia > 0")          # filtra 1–23
        .reset_index(drop=True)
)
    # Gera as datas sintéticas a partir da hora do dia
    rng = np.random.default_rng(seed = 12345)
    dates = synthetic_df['hora_do_dia'].astype(int).apply(sample_date)     

    # 1. cria máscara de linhas válidas (date ≠ 0)
    mask_valid = dates != 0

    # 2. aplica a máscara ao DataFrame e ao próprio vetor
    synthetic_df = synthetic_df.loc[mask_valid].reset_index(drop=True)
    dates = dates[mask_valid]

    # 3. monta o timestamp completo (YYYY-MM-DD)
    synthetic_df['tpep_pickup_datetime'] = pd.to_datetime(dates) 
    synthetic_df['tpep_pickup_datetime'] =  synthetic_df['tpep_pickup_datetime'].dt.date
    dados_reais_copy['tpep_pickup_datetime'] = dados_reais_copy['tpep_pickup_datetime'].dt.date

    #Combinar dados_reais_copy reais + sintéticos
    df_real_plus_sint = pd.concat([
        dados_reais_copy,
        synthetic_df])
    
    #Aqui converte a lagitude e longitude pra ID de zona dnv e filtra de acordo com a zona passada na função
    dados_reais_copy = add_location_ids_cupy(df=dados_reais_copy)
    synthetic_df = add_location_ids_cupy(df=synthetic_df)

    #Deixa só as colunas importantes pra treinar o dlinear 
    dados_reais_copy = dados_reais_copy[['hora_do_dia', 'tpep_pickup_datetime', 'num_viagens']]
    synthetic_df = synthetic_df[['hora_do_dia', 'tpep_pickup_datetime', 'num_viagens']]
    df_real_plus_sint = df_real_plus_sint[['hora_do_dia', 'tpep_pickup_datetime', 'num_viagens']]
    dados_reais_copy= dados_reais_copy.sort_values(['tpep_pickup_datetime','hora_do_dia']).reset_index(drop=True)
    synthetic_df= synthetic_df.sort_values(['tpep_pickup_datetime','hora_do_dia']).reset_index(drop=True)
    df_real_plus_sint =df_real_plus_sint.sort_values(['tpep_pickup_datetime','hora_do_dia']).reset_index(drop=True)
    # Transformar as duas colunas em string
    dados_reais_copy['hora_do_dia'] = dados_reais_copy['hora_do_dia'].astype(str)
    dados_reais_copy['tpep_pickup_datetime'] = dados_reais_copy['tpep_pickup_datetime'].astype(str)

    synthetic_df['hora_do_dia'] = synthetic_df['hora_do_dia'].astype(str)
    synthetic_df['tpep_pickup_datetime'] = synthetic_df['tpep_pickup_datetime'].astype(str)

    df_real_plus_sint['hora_do_dia'] = df_real_plus_sint['hora_do_dia'].astype(str)
    df_real_plus_sint['tpep_pickup_datetime'] = df_real_plus_sint['tpep_pickup_datetime'].astype(str)

    # Agrupar e ordenar para dados_reais_copy
    dados_reais_copy = (
        dados_reais_copy
        .groupby(['hora_do_dia', 'tpep_pickup_datetime'], as_index=False)['num_viagens'].sum()
        .reset_index(drop=True)
    )
    # Faz o groupby hora e data
    synthetic_df = (
        synthetic_df
        .groupby(['hora_do_dia', 'tpep_pickup_datetime'], as_index=False)['num_viagens'].sum()
        .reset_index(drop=True)
    )
    # Faz o groupby hora e data
    df_real_plus_sint = (
        df_real_plus_sint
        .groupby(['hora_do_dia', 'tpep_pickup_datetime'], as_index=False)['num_viagens'].sum()
        .reset_index(drop=True)
    )


    #Criar sequências para treinar o dlinear
    X_real, y_real = create_sequence_dataset(
        dados_reais_copy['num_viagens'].values.astype(float)
    )
    X_real_sint, y_real_sint = create_sequence_dataset(
        df_real_plus_sint['num_viagens'].values.astype(float)
    )
    X_sint, y_sint = create_sequence_dataset(
        synthetic_df['num_viagens'].values.astype(float)
    )

    # 6) Transformar em tensores
    X_real_t = torch.tensor(X_real, dtype=torch.float32)
    y_real_t = torch.tensor(y_real, dtype=torch.float32).unsqueeze(-1)

    X_real_sint_t = torch.tensor(X_real_sint, dtype=torch.float32)
    y_real_sint_t = torch.tensor(y_real_sint, dtype=torch.float32).unsqueeze(-1)

    X_sint_t = torch.tensor(X_sint, dtype=torch.float32)
    y_sint_t = torch.tensor(y_sint, dtype=torch.float32).unsqueeze(-1)

    # --------------------------------------------------
    # Loop de epochs e seeds para cada valor de nc
    # --------------------------------------------------
    for epochs in [100, 150, 200, 300]:
        execucao = 0 
        while execucao < NUM_EXECUCOES:
            seed = torch.seed() 
            torch.manual_seed(seed)
            # Vamos treinar 3 tipos de dados_reais_copy: real, synthetic, real+synthetic
            for data_type, (X, y) in zip(
                ['real', 'synthetic', 'real+synthetic'],
                [(X_real_t, y_real_t), (X_sint_t, y_sint_t), (X_real_sint_t, y_real_sint_t)]
            ):
                torch.manual_seed(seed)

                # treino/validação 70/30
                split_idx = int(0.7 * len(X))
                X_train, X_val = X[:split_idx], X[split_idx:]
                y_train, y_val = y[:split_idx], y[split_idx:]

                model = DLinear()
                train_model(model, X_train, y_train, X_val, y_val, epochs=epochs, lr=0.001)

                #Preve o número de viagens da próxima hora tendo como base o número de viagens das 4 horas anteriores
                with torch.no_grad():
                    pred = model(X_val).numpy().flatten()
                   # print("pred =", pred)
                true = y_val.numpy().flatten()
               # print("true =", true)
                metrics = {
                    'R²': r2_score(true, pred),
                    'MAE': mean_absolute_error(true, pred),
                    'RMSE': np.sqrt(mean_squared_error(true, pred))
                }

                # salva o resultado do treinamento 
                for metric, value in metrics.items():
                    new_row = pd.DataFrame([{
                        'tipo_dado': data_type,
                        'metrica': metric,
                        'valor': value,
                        'nc': nc,
                        'epochs': epochs,
                        'seed': seed
                    }])
                    df_resultados = pd.concat([df_resultados, new_row], ignore_index=True)
                    
                execucao += 1

# --------------------------------------------------
# GERAÇÃO DOS BOXPLOTS
# --------------------------------------------------
sns.set_style("whitegrid")
plt.style.use('seaborn-v0_8-white')

df_resultados.to_csv(f"{SAVE_DIR}resultados_metricas.csv", index=False)

metricas = ['R²', 'MAE', 'RMSE']

ncs_unicos = df_resultados['nc'].unique()

for metrica in metricas:
    df_metrica = df_resultados[df_resultados['metrica'] == metrica]

    # Loop para cada valor de nc
    for nc_atual in ncs_unicos:
        df_nc = df_metrica[df_metrica['nc'] == nc_atual]

        # Identifica os valores de epochs presentes para gerar subplots
        lista_epochs = sorted(df_nc['epochs'].unique())

        # Cálculo de colunas e linhas para subplots
        num_param = len(lista_epochs)
        cols = 3
        rows = (num_param + cols - 1) // cols

        fig, axs = plt.subplots(rows, cols, figsize=(cols*6.5, rows*5.5))
        axs = axs.flatten()

        for i, ep in enumerate(lista_epochs):
            ax = axs[i]
            df_sub = df_nc[df_nc['epochs'] == ep]

            # Agora temos 3 tipos de dado: 'real', 'synthetic', 'real+synthetic'
            sns.boxplot(
                x='tipo_dado',
                y='valor',
                data=df_sub,
                ax=ax,
                order=['real', 'synthetic', 'real+synthetic']
            )

            ax.set_title(f'nc={nc_atual}, epochs={ep}', fontsize=12, fontweight='bold')
            ax.set_xlabel('')
            ax.set_ylabel(metrica, fontsize=11)

            ymin, ymax = ax.get_ylim()

            # Inserir as estatísticas de média e mediana para cada tipo
            for tipo_idx, tipo in enumerate(['real', 'synthetic', 'real+synthetic']):
                dados_reais_tipo = df_sub[df_sub['tipo_dado'] == tipo]['valor']
                if not dados_reais_tipo.empty:
                    mediana = dados_reais_tipo.median()
                    media = dados_reais_tipo.mean()

                    ax.text(
                        tipo_idx, ymin,
                        f'Med: {mediana:.2f}',
                        ha='center', va='bottom',
                        color='black',
                        fontsize=12, fontweight='bold',
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2')
                    )

                    ax.text(
                        tipo_idx, ymax,
                        f'Média: {media:.2f}',
                        ha='center', va='top',
                        color='darkred',
                        fontsize=12, fontweight='bold',
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2')
                    )

        # Desativar eixos extras se sobrar espaço
        for j in range(i+1, len(axs)):
            axs[j].axis('off')

        plt.suptitle(
            f'Variação da métrica {metrica} - nc={nc_atual}',
            fontsize=16,
            fontweight='bold',
            y=1.02
        )
        plt.tight_layout()

        plt.savefig(f"{SAVE_DIR}nc_{nc_atual}_boxplot_{metrica}.png", bbox_inches='tight')
        plt.close()