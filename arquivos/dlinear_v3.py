# -*- coding: utf-8 -*-
"""
Pipeline DLinear (versão condensada)
────────────────────────────────────
• Gera dados sintéticos via GMM (PyCave)
• Constrói janelas dia-a-dia com build_pairs_df
• Aplica ponderação de crescimento (apply_growth_weighting)
• Treina DLinear e calcula R², MAE, RMSE
• Gera boxplots por métrica
"""
import os, math, warnings, datetime
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from pycave.bayes import GaussianMixture
from converter_lagitude_zona_v2 import add_location_ids_cupy
import geopandas as gpd

# --------------------------------------------------
# CONFIGURAÇÕES INICIAIS
# --------------------------------------------------
WINDOW               = 4          # tamanho da janela (nº horas ➜ input DLinear)
SYNTHETIC_MULTIPLIER = 1
SAVE_DIR             = "/home/caioloss/gráficos/linear/"
os.makedirs(SAVE_DIR, exist_ok=True)

NUM_EXECUCOES  = 5
data_sampler_seed = 30

torch.manual_seed(42)
np.random.seed(42)

# --------------------------------------------------
# 1) LEITURA E PREPARAÇÃO INICIAL DOS DADOS REAIS
# --------------------------------------------------
dados_reais = pd.read_parquet('/home-ext/caioloss/Dados/viagens_lat_long.parquet')
dados_reais['tpep_pickup_datetime'] = pd.to_datetime(dados_reais['tpep_pickup_datetime'])
dados_reais['hora_do_dia']          = dados_reais['tpep_pickup_datetime'].dt.hour
dados_reais['dia_da_semana']        = dados_reais['tpep_pickup_datetime'].dt.dayofweek
dados_reais['num_viagens']          = 1
dados_reais = dados_reais[
    (dados_reais["tpep_pickup_datetime"].dt.year  == 2024) &
    (dados_reais["tpep_pickup_datetime"].dt.month.isin([1, 2]))
]
dados_reais = dados_reais[dados_reais['dia_da_semana'].between(0, 2)]

features  = ['hora_do_dia','PU_longitude','PU_latitude','DO_longitude','DO_latitude']
features2 = ['tpep_pickup_datetime','hora_do_dia','num_viagens']

dados_reais_gmm = dados_reais[features].dropna()
dados_reais     = dados_reais[features2].dropna()

# --------------------------------------------------
# FUNÇÕES AUXILIARES (NOVAS)
# --------------------------------------------------
def make_date_sampler(df_real: pd.DataFrame,
                      ts_col: str = "tpep_pickup_datetime",
                      seed: int | None = None):
    df   = df_real.copy()
    df["hora"] = df[ts_col].dt.hour
    df["data"] = df[ts_col].dt.normalize()
    rng  = np.random.default_rng(seed or 12345)

    prob_table = {h: (sub["data"].value_counts().sort_index().index.to_numpy(),
                      sub["data"].value_counts(normalize=True).sort_index().values)
                  for h, sub in df.groupby("hora")}

    def sample_date(hora_do_dia: int):
        if hora_do_dia in prob_table:
            d, p = prob_table[hora_do_dia]
            return rng.choice(d, p=p)
        return 0
    return sample_date

def _prep(df0: pd.DataFrame):
    """Agrupa em nível (data, hora) somando #viagens."""
    df = df0[['tpep_pickup_datetime','hora_do_dia','num_viagens']].copy()
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])

    # Normaliza o timestamp para o início da hora para o groupby
    # Ou, crie uma coluna de data normalizada e agrupe por ela e hora_do_dia
    df['data_normalizada'] = df['tpep_pickup_datetime'].dt.normalize() # Obtém YYYY-MM-DD 00:00:00

    # Agrupa pela data normalizada e pela hora_do_dia
    df_agrupado = (df.groupby(['data_normalizada', 'hora_do_dia'], as_index=False)
                     .agg(num_viagens=('num_viagens', 'sum'))
                  )

    # Reconstrói tpep_pickup_datetime para ser YYYY-MM-DD HH:00:00
    # Isso é importante para a função build_pairs_df que espera um timestamp por hora
    df_agrupado['tpep_pickup_datetime'] = df_agrupado['data_normalizada'] + \
                                          pd.to_timedelta(df_agrupado['hora_do_dia'], unit='h')

    # Retorna apenas as colunas necessárias na ordem esperada
    return df_agrupado[['tpep_pickup_datetime', 'hora_do_dia', 'num_viagens']]

def build_pairs_df(group_df: pd.DataFrame, window: int = WINDOW):
    """Cria pares (dia t ➜ dia t+1) com janelas deslizantes."""
    df = group_df.copy()
    df['date'] = pd.to_datetime(df['tpep_pickup_datetime']).dt.normalize()
    mat = (df.pivot_table(index='date', columns='hora_do_dia',
                          values='num_viagens', aggfunc='sum',
                          fill_value=np.nan).sort_index())
    min_h = int(mat.columns.min())
    max_h = int(mat.columns.max()) - (window + 1)

    rows = []
    for i in range(len(mat.index)-1):
        d_train, d_val = mat.index[i].date(), mat.index[i+1].date()
        dia_t, dia_t1  = mat.iloc[i], mat.iloc[i+1]
        for h in range(min_h, max_h+1):
            hs = list(range(h, h+window))
            h_tgt = h + window
            if (dia_t[hs+[h_tgt]].notna().all() and
                dia_t1[hs+[h_tgt]].notna().all()):
                row = {
                    'date_train': d_train,
                    'date_val'  : d_val,
                    'window_start_hour': h,
                    'hours_used': ",".join(map(str, hs))
                }
                for k, hr in enumerate(hs):
                    row[f'h{k}_train'] = int(dia_t[hr])
                    row[f'h{k}_val']   = int(dia_t1[hr])
                row['target_train'] = int(dia_t[h_tgt])
                row['target_val']   = int(dia_t1[h_tgt])
                rows.append(row)
    return (pd.DataFrame(rows)
              .sort_values(['date_train','window_start_hour'])
              .reset_index(drop=True))

def apply_growth_weighting(X: np.ndarray):
    """Dá mais peso aos valores mais recentes da janela."""
    tiny = 1e-3
    Xw   = X.copy()
    for i in range(len(Xw)):
        w = [1.0]
        for j in range(1, Xw.shape[1]):
            prev  = Xw[i, j-1]
            ratio = Xw[i, j] / (prev if abs(prev) > tiny else tiny)
            w.append(w[-1] * ratio)
        Xw[i] = Xw[i] * np.array(w, dtype=np.float32)
    return Xw

# --------------------------------------------------
# MODELO E TREINO
# --------------------------------------------------
class DLinear(nn.Module):
    def __init__(self, input_len: int = WINDOW, output_dim: int = 1):
        super().__init__()
        self.linear = nn.Linear(input_len, output_dim)

    def forward(self, x):
        return self.linear(x)

def train_model(model, X_train, y_train, X_val, y_val,
                epochs, lr=1e-3, patience=10, min_delta=1e-4):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val, wait, best_state = float('inf'), 0, None
    for epoch in range(epochs):
        model.train(); optimizer.zero_grad()
        criterion(model(X_train), y_train).backward(); optimizer.step()
        model.eval()
        with torch.no_grad():
            v = criterion(model(X_val), y_val).item()
        if v < best_val - min_delta:
            best_val, best_state, wait = v, model.state_dict(), 0
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)

# --------------------------------------------------
# DATAFRAME PARA RESULTADOS
# --------------------------------------------------
df_resultados = pd.DataFrame(columns=['tipo_dado','metrica','valor',
                                      'nc','epochs','seed'])

# --------------------------------------------------
# LOOP PRINCIPAL
# --------------------------------------------------
sample_date = make_date_sampler(dados_reais, seed=data_sampler_seed)

for nc in [20, 22, 25, 27, 29]:
    # ---------- 1) GMM & DADOS SINTÉTICOS ----------
    scaler       = StandardScaler()
    train_scaled = scaler.fit_transform(dados_reais_gmm).astype(np.float32)

    gmm = GaussianMixture(
        num_components=nc,
        covariance_type='full',
        covariance_regularization=2.3681576970132,
        trainer_params={'max_epochs': 200, 'accelerator': 'gpu', 'devices': 1}
    )
    gmm.fit(train_scaled)

    synthetic_scaled = gmm.sample(int(len(dados_reais_gmm) * SYNTHETIC_MULTIPLIER)).cpu().numpy()
    synthetic_df = pd.DataFrame(scaler.inverse_transform(synthetic_scaled), columns=features)
    synthetic_df['num_viagens']  = 1
    synthetic_df['hora_do_dia']  = np.floor(pd.to_numeric(synthetic_df['hora_do_dia'],
                                                          errors='coerce')).astype(int)
    synthetic_df = synthetic_df.query("hora_do_dia >= 0").reset_index(drop=True)
    dates = synthetic_df['hora_do_dia'].astype(int).apply(sample_date)
    mask_valid = dates != 0
    synthetic_df = synthetic_df.loc[mask_valid].reset_index(drop=True)
    dates = dates[mask_valid]
    synthetic_df['tpep_pickup_datetime'] = (
        pd.to_datetime(dates) + pd.to_timedelta(synthetic_df['hora_do_dia'], unit='h')
    )

    # ---------- 2) COMBINAÇÃO E AJUSTES ----------
    #dados_reais_copy = add_location_ids_cupy(df=dados_reais.copy())
    dados_reais_copy = dados_reais.copy()
   # synthetic_df     = add_location_ids_cupy(df=synthetic_df)

    df_real_plus_sint = pd.concat([dados_reais_copy, synthetic_df], ignore_index=True)

    # Agrupa por data (normalizada) e hora, somando num_viagens
    df_real_plus_sint['data'] = df_real_plus_sint['tpep_pickup_datetime'].dt.normalize()

    df_real_plus_sint = (
        df_real_plus_sint
        .groupby(['data', 'hora_do_dia'], as_index=False)
        ['num_viagens'].sum()
        .sort_values(['data', 'hora_do_dia'])
        .reset_index(drop=True)
    )
    df_real_plus_sint['tpep_pickup_datetime'] = (
    df_real_plus_sint['data'] + pd.to_timedelta(df_real_plus_sint['hora_do_dia'], unit='h')
    )


    # Mantém apenas colunas necessárias
    for df in (dados_reais_copy, synthetic_df, df_real_plus_sint):
        df.drop(columns=[c for c in df.columns
                         if c not in ['tpep_pickup_datetime','hora_do_dia','num_viagens']],
                inplace=True, errors='ignore')
        df.sort_values(['tpep_pickup_datetime','hora_do_dia'], inplace=True)
        df.reset_index(drop=True, inplace=True)
        df['hora_do_dia'] = df['hora_do_dia'].astype(int)
    # ---------- 3) PIVOT / PARES DIA-A-DIA ----------
    groups = {
        'real'           : _prep(dados_reais_copy),
        'synthetic'      : _prep(synthetic_df),
        'real+synthetic' : _prep(df_real_plus_sint)
    }

    # Prints de quantidade de linhas e viagens totais
    print("--- Contagens e Viagens Totais (após _prep) ---")
    for name, df_group in groups.items():
        total_viagens = df_group['num_viagens'].sum() if 'num_viagens' in df_group.columns and not df_group.empty else 0
        print(f"➡️  {name:<15}: {len(df_group):>7} linhas | {total_viagens:>10} viagens totais")
    print("-------------------------------------------------")

    # Print do describe() para cada dataset
    print("\n--- Estatísticas Descritivas (após _prep) ---")
    for name, df_group in groups.items():
        print(f"\n📜 Describe para o dataset: {name}")
        if not df_group.empty:
            print(df_group.describe())
        else:
            print("   DataFrame está vazio.")
    print("-------------------------------------------------")

    pairs_dfs = {k: build_pairs_df(v, WINDOW) for k, v in groups.items()}

    # ---------- 4) TREINOS, MÉTRICAS & LOOP DE EPOCHS ----------
    
    for epochs in [100, 150, 200, 300]:
        execucao = 0
        while execucao < NUM_EXECUCOES:
            seed = torch.seed()
            torch.manual_seed(seed)

            for data_type in ['real','synthetic','real+synthetic']:
                # escolhe treino e validação
                if data_type == 'synthetic':
                    dfp_train = pairs_dfs['synthetic']
                    dfp_val   = pairs_dfs['real']
                else:
                    dfp_train = pairs_dfs[data_type]
                    dfp_val   = dfp_train

                if dfp_train.empty or dfp_val.empty:
                    continue

                # monta tensores a partir dos dataframes corretos
                X_tr_np = dfp_train[[f'h{k}_train' for k in range(WINDOW)]].values.astype(np.float32)
                y_tr_np = dfp_train['target_train'].values.astype(np.float32)

                X_va_np = dfp_val  [[f'h{k}_val'   for k in range(WINDOW)]].values.astype(np.float32)
                y_va_np = dfp_val  ['target_val'  ].values.astype(np.float32)

                X_train = torch.tensor(apply_growth_weighting(X_tr_np))
                X_val   = torch.tensor(apply_growth_weighting(X_va_np))
                y_train = torch.tensor(y_tr_np).unsqueeze(1)
                y_val   = torch.tensor(y_va_np).unsqueeze(1)

                # treina e avalia
                model = DLinear()
                train_model(model, X_train, y_train, X_val, y_val,
                            epochs=epochs, lr=0.01)

                with torch.no_grad():
                    pred = model(X_val).squeeze().numpy()
                true = y_val.squeeze().numpy()

                # calcula métricas e registra
                metrics = {
                    'R²'  : r2_score(true, pred),
                    'MAE' : mean_absolute_error(true, pred),
                    'RMSE': math.sqrt(mean_squared_error(true, pred))
                }
                for met, val in metrics.items():
                    df_resultados.loc[len(df_resultados)] = {
                        'tipo_dado': data_type,
                        'metrica'  : met,
                        'valor'    : val,
                        'nc'       : nc,
                        'epochs'   : epochs,
                        'seed'     : seed
                    }

                execucao += 1


# --------------------------------------------------
# GERAÇÃO DOS BOXPLOTS
# --------------------------------------------------
sns.set_style("whitegrid"); plt.style.use('seaborn-v0_8-white')
df_resultados.to_csv(f"{SAVE_DIR}resultados_metricas.csv", index=False)

metricas   = ['R²','MAE','RMSE']
ncs_unicos = df_resultados['nc'].unique()

for metrica in metricas:
    dfm = df_resultados[df_resultados['metrica'] == metrica]
    for nc_atual in ncs_unicos:
        df_nc   = dfm[dfm['nc'] == nc_atual]
        ep_list = sorted(df_nc['epochs'].unique())
        cols, rows = 3, (len(ep_list)+2)//3
        fig, axs = plt.subplots(rows, cols, figsize=(cols*6.5, rows*5.5))
        axs = axs.flatten()

        for i, ep in enumerate(ep_list):
            df_sub = df_nc[df_nc['epochs'] == ep]
            sns.boxplot(x='tipo_dado', y='valor', data=df_sub,
                        order=['real','synthetic','real+synthetic'],
                        ax=axs[i])
            axs[i].set_title(f'nc={nc_atual}, epochs={ep}', fontweight='bold')
            axs[i].set_xlabel(''); axs[i].set_ylabel(metrica)

        for j in range(i+1, len(axs)):
            axs[j].axis('off')

        plt.suptitle(f'Variação da métrica {metrica} - nc={nc_atual}',
                     fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(f"{SAVE_DIR}nc_{nc_atual}_boxplot_{metrica}.png",
                    bbox_inches='tight')
        plt.close()

# --------------------------------------------------
# GRÁFICOS DE MAE vs R²
# --------------------------------------------------
# Pivot para colocar R² e MAE em colunas
df_pivot = df_resultados.pivot_table(
    index=['tipo_dado','nc','epochs','seed'],
    columns='metrica',
    values='valor'
).reset_index()

# Agrupa por tipo_dado, epochs e nc, calculando média sobre as seeds
df_avg = df_pivot.groupby(['tipo_dado','epochs','nc'], as_index=False).mean()

# Para cada tipo de dado, plota uma figura com 4 curvas (cada epoch)
for data_type in ['real', 'synthetic', 'real+synthetic']:
    df_type = df_avg[df_avg['tipo_dado'] == data_type]
    plt.figure(figsize=(10, 6))
    for ep in [100, 150, 200, 300]:
        df_ep = df_type[df_type['epochs'] == ep].sort_values('nc')
        plt.plot(df_ep['R²'], df_ep['MAE'],
                 marker='o', linestyle='-',
                 label=f'epochs={ep}')
    plt.xlabel('R²')
    plt.ylabel('MAE')
    plt.title(f'MAE vs R² – {data_type}')
    plt.legend(title='Epócas')
    plt.tight_layout()
    # salva num arquivo separado para cada tipo de dado
    filename = f"{SAVE_DIR}{data_type.replace('+','plus')}_MAE_vs_R2.png"
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

print("🏁 Processo concluído – arquivos em:", SAVE_DIR)
