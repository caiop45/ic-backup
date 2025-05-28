import os, math, warnings, datetime, json
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from pycave.bayes import GaussianMixture
from converter_lagitude_zona_v2 import add_location_ids_cupy # Comentado se não for usado
import geopandas as gpd # Comentado se não for usado

# --------------------------------------------------
# CONFIGURAÇÕES INICIAIS
# --------------------------------------------------
WINDOW               = 4          # tamanho da janela (nº horas ➜ input DLinear)
SYNTHETIC_MULTIPLIER = 2          # Equivalente a SYNTH_OVERSAMPLE do Código (1)
SAVE_DIR             = "/home/caioloss/gráficos/dlinear/" # Diretório atualizado
os.makedirs(SAVE_DIR, exist_ok=True)

NUM_EXECUCOES  = 5
data_sampler_seed = 35 # Seed para make_date_sampler

# Configurações de ruído do Código (1)
COUNT_NOISE_FRAC = 0.05     # até ±5 % na contagem/hora
COORD_SIGMA_DEG  = 0.0005   # ~55 m de σ em lat/lon

torch.manual_seed(42) # Seed global inicial para PyTorch
np.random.seed(42)    # Seed global inicial para NumPy

# --- NOVA FUNÇÃO PARA MAPE ---
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Evitar divisão por zero: remover entradas onde y_true é 0
    # Ou adicionar um epsilon pequeno. Vamos remover por ora.
    mask = y_true != 0
    if not np.any(mask): # Todos os y_true são 0
        return np.nan 
    
    y_true_masked = y_true[mask]
    y_pred_masked = y_pred[mask]
    
    if len(y_true_masked) == 0: # Caso após a máscara não sobrem dados
        return np.nan
        
    mape = np.mean(np.abs((y_true_masked - y_pred_masked) / y_true_masked)) * 100
    
    # Lidar com MAPE potencialmente infinito se y_true_masked for muito pequeno e a diferença grande
    if np.isinf(mape) or mape > 1e9: # Um valor muito grande pode ser tratado como NaN ou capado
        return np.nan # Ou um valor capado, ex: 10000
    return mape

# ─── FUNÇÕES AUXILIARES DO CÓDIGO (1) ───────────────────────────────────
def decode_hour(sin_s, cos_s):
    ang = np.mod(np.arctan2(sin_s, cos_s), 2*np.pi)
    return np.rint(ang * 24/(2*np.pi)).astype(int) % 24

def synth_samples_cod1(gmm, n_samples, scaler_obj, feature_names):
    """ Adaptado de synth_samples do Código (1) """
    synth_scaled = gmm.sample(n_samples).cpu().numpy()
    s_df = pd.DataFrame(
        scaler_obj.inverse_transform(synth_scaled), columns=feature_names
    )
    s_df["hora_do_dia"] = decode_hour(s_df["sin_hr"], s_df["cos_hr"])
    s_df["num_viagens"] = 1
    return s_df

def equal_freq(s_df, hour_counts_dict_real, rng_param):
    """ Adaptado de equal_freq do Código (1) """
    parts=[]
    for hr, n_real in hour_counts_dict_real.items():
        sub = s_df[s_df["hora_do_dia"]==hr]
        if sub.empty: continue
        parts.append(
            sub.sample(n=n_real, replace=len(sub)<n_real,
                       random_state=rng_param.integers(1e6))
        )
    if not parts:
        return pd.DataFrame(columns=s_df.columns)
    return pd.concat(parts, ignore_index=True)


def qmap(col_s, col_r):
    """ Do Código (1) """
    if col_s.empty: # Se a coluna sintética está vazia, não há o que mapear
        return pd.Series(dtype=col_s.dtype)
    if col_r.empty: # Se a coluna real de referência está vazia, retorna NaNs
        return pd.Series(np.nan, index=col_s.index, dtype=np.float64)

    idx  = col_s.rank(method="first").astype(int) - 1
    real_sorted = np.sort(col_r.values) # col_r já foi .dropna() em apply_qmap

    # Assegurar que idx não cause index out of bounds
    max_idx_real = len(real_sorted) - 1
    if max_idx_real < 0 : # col_r (após dropna) resultou vazia
        return pd.Series(np.nan, index=col_s.index, dtype=np.float64)

    idx_values = idx.values
    if len(col_s) == 0: return pd.Series(np.nan, index=col_s.index, dtype=np.float64)

    scaled_indices = np.floor(idx_values * len(real_sorted) / len(col_s)).astype(int)
    final_indices = np.clip(scaled_indices, 0, max_idx_real)
    
    tgt  = real_sorted[final_indices]
    return pd.Series(tgt, index=col_s.index)


def apply_qmap(s_df, r_df_gmm_features):
    """ Do Código (1), r_df_gmm_features são os dados reais com as GMM_FEATURES """
    s_df_mapped = s_df.copy()
    coord_cols = ["PU_longitude","PU_latitude","DO_longitude","DO_latitude"]

    if r_df_gmm_features.empty:
        for c in coord_cols:
            if c in s_df_mapped.columns:
                 s_df_mapped[c] = np.nan
        return s_df_mapped

    for c in coord_cols:
        if c in s_df_mapped.columns:
            if c in r_df_gmm_features.columns and not r_df_gmm_features[c].dropna().empty:
                s_df_mapped[c] = qmap(s_df_mapped[c], r_df_gmm_features[c].dropna())
            else: # Coluna não existe em r_df ou está toda NaN
                s_df_mapped[c] = np.nan
    return s_df_mapped


def perturb_counts(df, rng, frac=COUNT_NOISE_FRAC):
    """ Do Código (1) """
    parts=[]
    for hr in range(24):
        sub = df[df["hora_do_dia"]==hr]
        if sub.empty: continue
        n_current_synth = len(sub)
        
        delta  = int(rng.normal(0, n_current_synth * frac))
        target_count = max(1, n_current_synth + delta)

        if target_count < n_current_synth:
            sub = sub.sample(n=target_count, replace=False,
                             random_state=rng.integers(1e6))
        elif target_count > n_current_synth:
            if not sub.empty:
                extra = sub.sample(n=target_count-n_current_synth, replace=True,
                                   random_state=rng.integers(1e6))
                sub   = pd.concat([sub, extra], ignore_index=True)
        parts.append(sub)
    if not parts:
        return pd.DataFrame(columns=df.columns)
    return pd.concat(parts, ignore_index=True)


def jitter_coords(df, rng, sigma=COORD_SIGMA_DEG):
    """ Do Código (1) """
    df_jittered = df.copy()
    for c in ["PU_latitude","PU_longitude","DO_latitude","DO_longitude"]:
        if c in df_jittered.columns:
            df_jittered[c] += rng.normal(0, sigma, size=len(df_jittered))
    return df_jittered
# ------------------------------------------------------------------------

# --------------------------------------------------
# 1) LEITURA E PREPARAÇÃO INICIAL DOS DADOS REAIS
# --------------------------------------------------
print("▶️  Carregando dados reais …")
dados_reais_orig = pd.read_parquet('/home-ext/caioloss/Dados/viagens_lat_long.parquet')
dados_reais_orig['tpep_pickup_datetime'] = pd.to_datetime(dados_reais_orig['tpep_pickup_datetime'])

dados_reais_orig = dados_reais_orig[
    (dados_reais_orig["tpep_pickup_datetime"].dt.year  == 2024) &
    (dados_reais_orig["tpep_pickup_datetime"].dt.month.isin([1, 2]))
]
dados_reais_orig = dados_reais_orig[dados_reais_orig['tpep_pickup_datetime'].dt.dayofweek.between(0, 2)] # Seg, Ter, Qua

dados_reais_orig['hora_do_dia']     = dados_reais_orig['tpep_pickup_datetime'].dt.hour
dados_reais_orig['num_viagens']     = 1

real_hour_counts_series = dados_reais_orig.groupby("hora_do_dia")["num_viagens"].sum().sort_index()
hour_counts_dict_real   = real_hour_counts_series.to_dict()

dados_reais_orig["sin_hr"] = np.sin(2*np.pi*dados_reais_orig["hora_do_dia"]/24)
dados_reais_orig["cos_hr"] = np.cos(2*np.pi*dados_reais_orig["hora_do_dia"]/24)

GMM_FEATURES = [
    "sin_hr", "cos_hr",
    "PU_longitude", "PU_latitude",
    "DO_longitude", "DO_latitude",
]
dados_reais_gmm = dados_reais_orig[GMM_FEATURES].dropna().astype(np.float32)

dados_reais_dlinear_input = dados_reais_orig[['tpep_pickup_datetime', 'hora_do_dia', 'num_viagens']].copy()


# --------------------------------------------------
# FUNÇÕES AUXILIARES (DO CÓDIGO 2 - Mantidas/Ajustadas)
# --------------------------------------------------
def make_date_sampler(df_real_for_sampler: pd.DataFrame,
                      ts_col: str = "tpep_pickup_datetime",
                      seed: int | None = None):
    df   = df_real_for_sampler.copy()
    df["hora"] = df[ts_col].dt.hour
    df["data"] = df[ts_col].dt.normalize()
    rng  = np.random.default_rng(seed or 12345)

    prob_table = {h: (sub["data"].value_counts().sort_index().index.to_numpy(),
                      sub["data"].value_counts(normalize=True).sort_index().values)
                  for h, sub in df.groupby("hora")}

    def sample_date(hora_do_dia: int):
        if hora_do_dia in prob_table:
            dates_for_hour, probs_for_hour = prob_table[hora_do_dia]
            if len(dates_for_hour) > 0:
                return rng.choice(dates_for_hour, p=probs_for_hour)
        
        available_hours_with_data = [h for h in prob_table if len(prob_table[h][0]) > 0]
        if available_hours_with_data:
            fallback_hour = rng.choice(available_hours_with_data)
            dates_fallback, probs_fallback = prob_table[fallback_hour]
            return rng.choice(dates_fallback, p=probs_fallback)
        
        return pd.NaT
    return sample_date

def _prep(df0: pd.DataFrame):
    if df0.empty or not {'tpep_pickup_datetime', 'hora_do_dia', 'num_viagens'}.issubset(df0.columns):
        return pd.DataFrame(columns=['tpep_pickup_datetime', 'hora_do_dia', 'num_viagens'])

    df = df0[['tpep_pickup_datetime','hora_do_dia','num_viagens']].copy()
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
    df['data_normalizada'] = df['tpep_pickup_datetime'].dt.normalize()
    df_agrupado = (df.groupby(['data_normalizada', 'hora_do_dia'], as_index=False)
                     .agg(num_viagens=('num_viagens', 'sum'))
                  )
    df_agrupado['tpep_pickup_datetime'] = df_agrupado['data_normalizada'] + \
                                          pd.to_timedelta(df_agrupado['hora_do_dia'], unit='h')
    return df_agrupado[['tpep_pickup_datetime', 'hora_do_dia', 'num_viagens']]

def build_pairs_df(group_df: pd.DataFrame, window: int = WINDOW):
    if group_df.empty or group_df['num_viagens'].sum() == 0:
        return pd.DataFrame()

    df = group_df.copy()
    df['date'] = pd.to_datetime(df['tpep_pickup_datetime']).dt.normalize()
    
    try:
        mat = (df.pivot_table(index='date', columns='hora_do_dia',
                              values='num_viagens', aggfunc='sum',
                              fill_value=np.nan).sort_index()) # fill_value=0 pode ser problemático para growth_weighting
    except Exception:
        return pd.DataFrame()

    if mat.empty or mat.columns.empty:
        return pd.DataFrame()

    min_h_col = mat.columns.min()
    max_h_col = mat.columns.max()

    if pd.isna(min_h_col) or pd.isna(max_h_col):
        return pd.DataFrame()
        
    min_h = int(min_h_col)
    max_h = int(max_h_col) - window 

    rows = []
    if len(mat.index) < 2: 
        return pd.DataFrame()

    for i in range(len(mat.index)-1): 
        d_train, d_val = mat.index[i].date(), mat.index[i+1].date()
        dia_t, dia_t1  = mat.iloc[i], mat.iloc[i+1]
        for h_start in range(min_h, max_h + 1): 
            hs_input = list(range(h_start, h_start+window))
            h_target = h_start + window 
            
            required_hours_check = hs_input + [h_target]
            if not all(hr_check in mat.columns for hr_check in required_hours_check):
                continue

            if dia_t[required_hours_check].notna().all() and \
               dia_t1[required_hours_check].notna().all():
                row = {
                    'date_train': d_train,
                    'date_val'  : d_val,
                    'window_start_hour': h_start,
                    'hours_used': ",".join(map(str, hs_input))
                }
                for k, hr_val in enumerate(hs_input): 
                    row[f'h{k}_train'] = int(dia_t[hr_val])
                    row[f'h{k}_val']   = int(dia_t1[hr_val])
                row['target_train'] = int(dia_t[h_target]) 
                row['target_val']   = int(dia_t1[h_target]) 
                rows.append(row)
    
    if not rows:
        return pd.DataFrame()
        
    return (pd.DataFrame(rows)
              .sort_values(['date_train','window_start_hour'])
              .reset_index(drop=True))


def apply_growth_weighting(X: np.ndarray):
    tiny = 1e-3
    Xw   = X.copy()
    for i in range(len(Xw)):
        w = [1.0]
        if Xw.shape[1] > 1: 
            for j in range(1, Xw.shape[1]):
                prev  = Xw[i, j-1]
                ratio = Xw[i, j] / (prev if abs(prev) > tiny else tiny)
                w.append(w[-1] * ratio)
            Xw[i] = Xw[i] * np.array(w, dtype=np.float32)
    return Xw

# --------------------------------------------------
# MODELO E TREINO (DO CÓDIGO 2 - Mantidos)
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
    best_val_loss, wait_count, best_model_state = float('inf'), 0, None
    
    if X_train.nelement() == 0 or X_val.nelement() == 0 or y_train.nelement() == 0 or y_val.nelement() == 0:
        return

    for epoch in range(epochs):
        model.train(); optimizer.zero_grad()
        
        y_train_expanded = y_train.unsqueeze(1) if y_train.ndim == 1 else y_train
        loss = criterion(model(X_train), y_train_expanded)
        loss.backward(); optimizer.step()
        
        model.eval()
        with torch.no_grad():
            y_val_expanded = y_val.unsqueeze(1) if y_val.ndim == 1 else y_val
            val_loss = criterion(model(X_val), y_val_expanded).item()
            
        if val_loss < best_val_loss - min_delta:
            best_val_loss, best_model_state, wait_count = val_loss, model.state_dict(), 0
        else:
            wait_count += 1
            if wait_count >= patience:
                break
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

# --------------------------------------------------
# DATAFRAME PARA RESULTADOS
# --------------------------------------------------
df_resultados = pd.DataFrame(columns=['tipo_dado','metrica','valor',
                                      'nc','epochs','seed'])

# --------------------------------------------------
# LOOP PRINCIPAL
# --------------------------------------------------
sample_date_func = make_date_sampler(dados_reais_dlinear_input, seed=data_sampler_seed)

gmm_scaler = StandardScaler()
if dados_reais_gmm.empty:
    raise ValueError("dados_reais_gmm está vazio após pré-processamento. Verifique os filtros e a fonte de dados.")
    
train_scaled_gmm = gmm_scaler.fit_transform(dados_reais_gmm)


for nc_gmm_components in [40, 45, 50]: # Reduzido para teste mais rápido
    print(f"\n🔄 Iniciando para GMM nc={nc_gmm_components}")
    
    gmm = GaussianMixture(
        num_components=nc_gmm_components,
        covariance_type='full',
        covariance_regularization=1e-4, 
        trainer_params={'max_epochs': 200, 'accelerator': 'auto', 'devices': 1}
    )
    gmm.fit(train_scaled_gmm)

    n_synth_oversampled = int(len(dados_reais_gmm) * SYNTHETIC_MULTIPLIER)
    if n_synth_oversampled == 0:
        print(f"  Aviso: n_synth_oversampled é 0 para nc={nc_gmm_components}. Pulando este nc.")
        continue

    for exec_idx in range(NUM_EXECUCOES):
        current_seed = torch.initial_seed() + nc_gmm_components + exec_idx 
        torch.manual_seed(current_seed)
        np.random.seed(current_seed)
        rng_execucao = np.random.default_rng(current_seed)
        
        print(f"  Execução {exec_idx+1}/{NUM_EXECUCOES} com seed {current_seed} para nc={nc_gmm_components}")

        synthetic_df_raw = synth_samples_cod1(gmm, n_synth_oversampled, gmm_scaler, GMM_FEATURES)
        if synthetic_df_raw.empty: continue

        synthetic_df_eq = equal_freq(synthetic_df_raw, hour_counts_dict_real, rng_execucao)
        if synthetic_df_eq.empty: continue

        synthetic_df_pert = perturb_counts(synthetic_df_eq, rng_execucao, frac=COUNT_NOISE_FRAC)
        if synthetic_df_pert.empty: continue

        synthetic_df_final_pre_date = synthetic_df_pert.copy()
       # synthetic_df_qmap = apply_qmap(synthetic_df_pert, dados_reais_gmm)
       # if synthetic_df_qmap.empty or synthetic_df_qmap[GMM_FEATURES[2:]].isnull().all().all(): continue

       # synthetic_df_final_pre_date = jitter_coords(synthetic_df_qmap, rng_execucao, sigma=COORD_SIGMA_DEG)
       # if synthetic_df_final_pre_date.empty: continue
        
        synthetic_df_pert['hora_do_dia'] = synthetic_df_final_pre_date['hora_do_dia'].astype(int)
        
        sampled_dates = synthetic_df_final_pre_date['hora_do_dia'].apply(sample_date_func)
        
        valid_dates_mask = sampled_dates.notna()
        synthetic_df_dated = synthetic_df_final_pre_date[valid_dates_mask].copy() 
        if synthetic_df_dated.empty: continue
            
        synthetic_df_dated['tpep_pickup_datetime'] = pd.to_datetime(sampled_dates[valid_dates_mask]) + \
                                                      pd.to_timedelta(synthetic_df_dated['hora_do_dia'], unit='h')
        
        synthetic_df_dlinear_input = synthetic_df_dated[['tpep_pickup_datetime', 'hora_do_dia', 'num_viagens']].copy()
        
        df_real_plus_sint_input = pd.concat([dados_reais_dlinear_input, synthetic_df_dlinear_input], ignore_index=True)

        groups_for_prep = {
            'real'           : dados_reais_dlinear_input,
            'synthetic'      : synthetic_df_dlinear_input,
            'real+synthetic' : df_real_plus_sint_input
        }
        
        prepared_groups = {}
        for name, df_to_prep in groups_for_prep.items():
            prepared_groups[name] = _prep(df_to_prep)

        pairs_dfs = {}
        for k, prep_df in prepared_groups.items():
            pairs_dfs[k] = build_pairs_df(prep_df, WINDOW)

        for epochs_dlinear in [50, 80, 100, 150, 200, 250]: # Reduzido para teste mais rápido
            for data_type in ['real','synthetic','real+synthetic']:
                dfp_train = pairs_dfs.get(data_type, pd.DataFrame())
                dfp_val   = pairs_dfs.get('real', pd.DataFrame())
                
                if dfp_train.empty or dfp_val.empty:
                    print(f"    Skipping DLinear: Treino '{data_type}', Validação 'real'. nc={nc_gmm_components}, ep={epochs_dlinear}, seed={current_seed} (pares vazios para treino ou validação real)")
                    continue

                X_tr_np = dfp_train[[f'h{k}_train' for k in range(WINDOW)]].values.astype(np.float32)
                y_tr_np = dfp_train['target_train'].values.astype(np.float32)
                
                X_va_np = dfp_val[[f'h{k}_val' for k in range(WINDOW)]].values.astype(np.float32)
                y_va_np = dfp_val['target_val'].values.astype(np.float32)


                if X_tr_np.shape[0] == 0 or X_va_np.shape[0] == 0 or \
                   y_tr_np.shape[0] == 0 or y_va_np.shape[0] == 0:
                    print(f"    Skipping DLinear: Treino '{data_type}', Validação 'real'. nc={nc_gmm_components}, ep={epochs_dlinear}, seed={current_seed} (arrays np vazios)")
                    continue

                X_train = torch.tensor(apply_growth_weighting(X_tr_np))
                X_val   = torch.tensor(apply_growth_weighting(X_va_np)) 
                y_train = torch.tensor(y_tr_np)
                y_val   = torch.tensor(y_va_np) 

                model = DLinear(input_len=WINDOW)
                train_model(model, X_train, y_train, X_val, y_val,
                            epochs=epochs_dlinear, lr=0.01, patience=10)

                with torch.no_grad():
                    pred_tensor = model(X_val) 
                    if pred_tensor.nelement() == 0:
                        continue
                    pred = pred_tensor.squeeze().cpu().numpy()
                true = y_val.squeeze().cpu().numpy() 

                if isinstance(pred, float): pred = np.array([pred])
                if isinstance(true, float): true = np.array([true])
                if pred.ndim == 0: pred = pred.reshape(1)
                if true.ndim == 0: true = true.reshape(1)
                
                if len(pred) == 0 or len(true) == 0 or len(pred) != len(true):
                     continue

                try:
                    r2 = r2_score(true, pred) if len(true) > 1 else np.nan
                    mae = mean_absolute_error(true, pred)
                    
                    # --- CORREÇÃO PARA RMSE ---
                    # Calcular MSE primeiro
                    mse = mean_squared_error(true, pred)
                    # RMSE é a raiz quadrada do MSE. MSE é sempre não-negativo.
                    # np.sqrt(np.nan) resulta em np.nan, o que é o comportamento desejado se mse for nan.
                    rmse_val = np.sqrt(mse)
                    # --- FIM DA CORREÇÃO ---
                    
                    # --- CÁLCULO DO MAPE ---
                    mape_val = mean_absolute_percentage_error(true, pred)
                    
                    metrics = {
                        'R²'  : r2,
                        'MAE' : mae,
                        'RMSE': rmse_val,
                        'MAPE': mape_val # Adicionado MAPE
                    }
                except ValueError as e: # Captura erros como input contendo NaN ou inf
                    print(f"Erro no cálculo de métricas: {e}. Pred shape: {pred.shape}, True shape: {true.shape}. True: {true[:5]}, Pred: {pred[:5]}")
                    metrics = {'R²': np.nan, 'MAE': np.nan, 'RMSE': np.nan, 'MAPE': np.nan}
                except Exception as e_gen: # Captura outros erros inesperados
                    print(f"Erro geral no cálculo de métricas: {e_gen}. Pred shape: {pred.shape}, True shape: {true.shape}. True: {true[:5]}, Pred: {pred[:5]}")
                    metrics = {'R²': np.nan, 'MAE': np.nan, 'RMSE': np.nan, 'MAPE': np.nan}


                for met, val in metrics.items():
                    new_row = pd.DataFrame([{
                        'tipo_dado': data_type, 'metrica': met, 'valor': val,
                        'nc': nc_gmm_components, 'epochs': epochs_dlinear, 'seed': current_seed
                    }])
                    df_resultados = pd.concat([df_resultados, new_row], ignore_index=True)
    print(f"🏁 Concluído para nc={nc_gmm_components}")

# --------------------------------------------------
# GERAÇÃO DOS BOXPLOTS E GRÁFICOS
# --------------------------------------------------
if df_resultados.empty:
    print("DataFrame de resultados está vazio. Nenhum gráfico será gerado.")
else:
    sns.set_style("whitegrid"); plt.style.use('seaborn-v0_8-whitegrid')
    df_resultados.to_csv(f"{SAVE_DIR}resultados_metricas_cod1_logic_valid_real_MAPE_15clusters_ate_22.csv", index=False)

    metricas_plot   = df_resultados['metrica'].unique()
    ncs_unicos_plot = sorted(df_resultados['nc'].unique())
    
    plot_order = ['real', 'synthetic', 'real+synthetic'] # Definir a ordem para os boxplots

    for metrica_val in metricas_plot:
        dfm = df_resultados[df_resultados['metrica'] == metrica_val]
        for nc_val in ncs_unicos_plot:
            df_nc_plot = dfm[dfm['nc'] == nc_val]
            if df_nc_plot.empty: continue

            ep_list_plot = sorted(df_nc_plot['epochs'].unique())
            if not ep_list_plot: continue

            cols_plot = 3
            rows_plot = math.ceil(len(ep_list_plot) / cols_plot)
            fig, axs = plt.subplots(rows_plot, cols_plot, figsize=(cols_plot*6.5, rows_plot*5.5), squeeze=False)
            axs = axs.flatten()

            for i, ep_val in enumerate(ep_list_plot):
                ax_current = axs[i] # Eixo atual
                df_sub_plot = df_nc_plot[df_nc_plot['epochs'] == ep_val]
                
                if df_sub_plot.empty:
                    ax_current.text(0.5, 0.5, 'Sem dados', ha='center', va='center', transform=ax_current.transAxes)
                else:
                    sns.boxplot(x='tipo_dado', y='valor', data=df_sub_plot,
                                order=plot_order, ax=ax_current) # Usar plot_order
                    
                    # --- ANOTAÇÃO DOS VALORES (MEDIANA) NOS BOXPLOTS ---
                    for patch_idx, patch in enumerate(ax_current.artists):
                        # patch_idx corresponde à ordem em plot_order (0: real, 1: synthetic, 2: real+synthetic)
                        if patch_idx < len(plot_order):
                            category_name = plot_order[patch_idx]
                            
                            # Filtrar dados para a categoria e subplot atuais
                            category_data = df_sub_plot[df_sub_plot['tipo_dado'] == category_name]['valor']
                            
                            if not category_data.empty:
                                # Usar mediana, conforme solicitado implicitamente pelo boxplot
                                value_to_display = category_data.median() 
                                
                                if pd.notna(value_to_display):
                                    # Posição X: centro da caixa
                                    x_pos = patch.get_x() + patch.get_width() / 2
                                    
                                    # Posição Y: um pouco acima do topo da caixa (Q3)
                                    # Q3 é patch.get_y() + patch.get_height()
                                    y_box_top = patch.get_y() + patch.get_height()
                                    
                                    # Pequeno offset para o texto não colar na caixa/whisker
                                    # O offset pode ser relativo ao range do eixo Y
                                    current_ylim = ax_current.get_ylim()
                                    y_offset = (current_ylim[1] - current_ylim[0]) * 0.02 # 2% do range do eixo Y
                                    
                                    y_text_pos = y_box_top + y_offset
                                    
                                    # Para MAPE, R², etc., ajustar a formatação pode ser útil
                                    format_str = '{:.2f}' if metrica_val not in ['MAPE'] else '{:.1f}%'
                                    if metrica_val == 'MAPE' and pd.notna(value_to_display) :
                                         text_to_show = format_str.format(value_to_display)
                                    elif pd.notna(value_to_display) :
                                         text_to_show = f'{value_to_display:.2f}' # Padrão 2 casas decimais
                                    else:
                                         text_to_show = "N/A"

                                    ax_current.text(x_pos, y_text_pos, text_to_show, 
                                                    ha='center', va='bottom', fontweight='normal', 
                                                    color='dimgray', fontsize=8)
                
                ax_current.set_title(f'nc={nc_val}, epochs={ep_val}', fontweight='bold')
                ax_current.set_xlabel('Tipo de Dado (Treino)')
                ax_current.set_ylabel(metrica_val)

            for j_ax in range(i + 1, len(axs)):
                axs[j_ax].axis('off')

            fig.suptitle(f'Variação da métrica {metrica_val} - nc={nc_val} (Validação em Dados Reais)',
                         fontsize=16, fontweight='bold')
            fig.tight_layout(rect=[0, 0, 1, 0.95]) 
            plt.savefig(f"{SAVE_DIR}nc_{nc_val}_boxplot_{metrica_val}_cod1_logic_valid_real_MAPE_annot.png",
                        bbox_inches='tight')
            plt.close(fig)

    # Gráficos MAE vs R² (já incluía R², MAE, RMSE, agora MAPE também estará no df_pivot se calculado)
    try:
        df_pivot = df_resultados.pivot_table(
            index=['tipo_dado','nc','epochs','seed'],
            columns='metrica', values='valor'
        ).reset_index()
    except Exception as e_pivot:
        print(f"Erro ao pivotar df_resultados para gráficos comparativos: {e_pivot}")
        df_pivot = pd.DataFrame()

    if not df_pivot.empty and {'R²', 'MAE'}.issubset(df_pivot.columns): # Mantém foco em R² vs MAE
        df_avg = df_pivot.groupby(['tipo_dado','epochs','nc'], as_index=False)[['R²','MAE']].mean()

        for data_type_plot in df_avg['tipo_dado'].unique():
            df_type_plot = df_avg[df_avg['tipo_dado'] == data_type_plot]
            if df_type_plot.empty: continue

            plt.figure(figsize=(10, 6))
            epochs_mae_r2_list = sorted(df_type_plot['epochs'].unique())
            for ep_mae_r2 in epochs_mae_r2_list:
                df_ep_plot = df_type_plot[df_type_plot['epochs'] == ep_mae_r2].sort_values('nc')
                if not df_ep_plot.empty:
                    plt.plot(df_ep_plot['R²'], df_ep_plot['MAE'],
                             marker='o', linestyle='-', label=f'epochs={ep_mae_r2}')
            plt.xlabel('R² (médio por seed)')
            plt.ylabel('MAE (médio por seed)')
            plt.title(f'MAE vs R² – Treino em {data_type_plot} (Lógica Código 1, Validação em Dados Reais)')
            plt.legend(title='Epócas'); plt.grid(True)
            plt.tight_layout()
            filename_mae_r2 = f"{SAVE_DIR}{data_type_plot.replace('+','plus')}_MAE_vs_R2_cod1_logic_valid_real_MAPE_annot.png"
            plt.savefig(filename_mae_r2, bbox_inches='tight')
            plt.close()
    else:
        print("Não foi possível gerar gráficos MAE vs R² devido a dados pivotados ausentes ou colunas R²/MAE faltando.")

print(f"🏁 Processo concluído – arquivos em: {SAVE_DIR}")