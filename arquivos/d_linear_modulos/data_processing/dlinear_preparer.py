import numpy as np
import pandas as pd
from config import WINDOW
import torch
from utils.helpers import agrupar_viagens_por_local, quarter_hour_slot

# ---------- pré-agregação ---------- #
def _prep(df0: pd.DataFrame):
    if df0.empty or not {"tpep_pickup_datetime", "hora_do_dia", "num_viagens"}.issubset(df0.columns):
        return pd.DataFrame(columns=["tpep_pickup_datetime", "hora_do_dia", "num_viagens"])

    df = df0[["tpep_pickup_datetime", "hora_do_dia", "num_viagens"]].copy()
    df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])
    df["data_normalizada"]     = df["tpep_pickup_datetime"].dt.normalize()
    df["hora_do_dia"] = quarter_hour_slot(df["tpep_pickup_datetime"]) 

    grp = (
        df.groupby(["data_normalizada", "hora_do_dia"], as_index=False)
          .agg(num_viagens=("num_viagens", "sum"))
    )

    grp["tpep_pickup_datetime"] = grp["data_normalizada"] + pd.to_timedelta(grp["hora_do_dia"] * 15, unit="m")

    return grp[["tpep_pickup_datetime", "hora_do_dia", "num_viagens"]]

# ---------- pares entrada-alvo ---------- #
def build_pairs_df(group_df: pd.DataFrame, window: int = WINDOW):
    if group_df.empty or group_df["num_viagens"].sum() == 0:
        return pd.DataFrame()

    df = group_df.copy()
    df["date"] = pd.to_datetime(df["tpep_pickup_datetime"]).dt.normalize()

    mat = (df.pivot_table(index="date",
                          columns="hora_do_dia",
                          values="num_viagens",
                          aggfunc="sum",
                          fill_value=np.nan)
             .sort_index())

    if mat.empty or mat.columns.empty:
        return pd.DataFrame()

    min_h, max_h = int(mat.columns.min()), int(mat.columns.max()) - window
    rows = []

    for i in range(len(mat) - 1):
        d_train, d_val = mat.index[i].date(), mat.index[i + 1].date()
        dia_t, dia_t1  = mat.iloc[i], mat.iloc[i + 1]

        for h_start in range(min_h, max_h + 1):
            hs_input  = list(range(h_start, h_start + window))
            h_target  = h_start + window
            required  = hs_input + [h_target]

            if not all(hr in mat.columns for hr in required):
                continue
            if dia_t[required].notna().all() and dia_t1[required].notna().all():
                row = {
                    "date_train"      : d_train,
                    "date_val"        : d_val,
                    "window_start_hour": h_start,
                    "hours_used"      : ",".join(map(str, hs_input)),
                }
                for k, hr in enumerate(hs_input):
                    row[f"h{k}_train"] = int(dia_t[hr])
                    row[f"h{k}_val"]   = int(dia_t1[hr])
                row["target_train"] = int(dia_t[h_target])
                row["target_val"]   = int(dia_t1[h_target])
                rows.append(row)

    if not rows:
        return pd.DataFrame()

    return (pd.DataFrame(rows)
              .sort_values(["date_train", "window_start_hour"])
              .reset_index(drop=True))

# ---------- growth-weighting ---------- #
def apply_growth_weighting(X: np.ndarray):
    tiny = 1e-3
    Xw   = X.copy()
    for i in range(len(Xw)):
        w = [1.0]
        for j in range(1, Xw.shape[1]):
            prev  = Xw[i, j - 1]
            ratio = Xw[i, j] / (prev if abs(prev) > tiny else tiny)
            w.append(w[-1] * ratio)
        Xw[i] *= np.array(w, dtype=np.float32)
    return Xw

def _create_windows(df: pd.DataFrame, input_window_size: int, prediction_horizon: int, device: torch.device):
    """
    Função auxiliar que transforma um DataFrame em janelas (X, y) já como tensores do PyTorch.
    """
    target_cols = [col for col in df.columns if col != 'hora_do_dia']
    
    X_list, y_list = [], []
    
    df_values = df.values
    df_target_values = df[target_cols].values

    num_samples = len(df) - input_window_size - prediction_horizon + 1
    
    for i in range(num_samples):
        input_end = i + input_window_size
        output_end = input_end + prediction_horizon
        
        window_x = df_values[i:input_end, :]
        X_list.append(window_x)
        
        window_y = df_target_values[input_end:output_end, :]
        y_list.append(window_y)
    
    # Converte as listas de arrays para tensores do PyTorch
    X_tensor = torch.from_numpy(np.array(X_list)).float()
    y_tensor = torch.from_numpy(np.array(y_list)).float()
    
    # Move os tensores para o dispositivo especificado (CPU ou GPU)
    return X_tensor.to(device), y_tensor.to(device)


def prepare_all_data_for_dlinear(
    training_groups: dict, 
    validation_df: pd.DataFrame, 
    input_window_size: int = 3, 
    prediction_horizon: int = 1,
    device: torch.device = torch.device('cpu')
    ):
    """
    Prepara todos os dados para o modelo Dlinear, retornando tensores do PyTorch.

    Args:
        training_groups (dict): Dicionário com os DataFrames de treino.
        validation_df (pd.DataFrame): DataFrame de validação.
        input_window_size (int): Passos de tempo na janela de entrada.
        prediction_horizon (int): Passos de tempo a serem previstos.
        device (torch.device): Dispositivo para alocar os tensores ('cpu' ou 'cuda').

    Returns:
        dict: Dicionário contendo os pares de tensores (X_train, y_train) e (X_val, y_val).
    """
    
    windowed_data = {}

    print(f"Preparando dados e movendo tensores para o dispositivo: '{device}'")

    for name, train_df in training_groups.items():
        X_train, y_train = _create_windows(train_df, input_window_size, prediction_horizon, device)
        windowed_data[name] = {
            'X_train': X_train,
            'y_train': y_train
        }
        print(f"-> Treino '{name}' processado. Shape X_train: {X_train.shape}, Shape y_train: {y_train.shape}")

    X_val, y_val = _create_windows(validation_df, input_window_size, prediction_horizon, device)
    windowed_data['validation'] = {
        'X_val': X_val,
        'y_val': y_val
    }
    print(f"-> Validação processada. Shape X_val: {X_val.shape}, Shape y_val: {y_val.shape}")
    
    return windowed_data

def preparar_e_agrupar_datasets(
    dados_reais: pd.DataFrame, 
    dados_sinteticos: pd.DataFrame,
    dados_reais_eval: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Recebe DataFrames de dados reais e sintéticos, cria um conjunto de dados híbrido,
    e agrupa os três conjuntos (real, sintético e híbrido) por hora e local,
    contando o número de viagens.

    Args:
        dados_reais (pd.DataFrame): DataFrame contendo os dados de viagens reais.
                                    Deve ter a coluna 'tpep_pickup_datetime'.
        dados_sinteticos (pd.DataFrame): DataFrame contendo os dados de viagens sintéticos.
                                         Deve ter a coluna 'tpep_pickup_datetime'.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Uma tupla contendo os 
        DataFrames agrupados na seguinte ordem:
        1. Dados Híbridos Agrupados (hybrid_grouped)
        2. Dados Reais Agrupados (real_grouped)
        3. Dados Sintéticos Agrupados (synth_grouped)
    """
    # É uma boa prática trabalhar com cópias para evitar efeitos colaterais
    # nos DataFrames originais fora da função.
    dados_reais_copy = dados_reais.copy()
    dados_sinteticos_copy = dados_sinteticos.copy()

    dados_reais_copy['tpep_pickup_datetime'] = pd.to_datetime(
        dados_reais_copy['tpep_pickup_datetime']
    ).dt.floor('15min')
    dados_sinteticos_copy['tpep_pickup_datetime'] = pd.to_datetime(
        dados_sinteticos_copy['tpep_pickup_datetime']
    ).dt.floor('15min')
    dados_reais_eval['tpep_pickup_datetime'] = pd.to_datetime(
        dados_reais_eval['tpep_pickup_datetime']
    ).dt.floor('15min')

    # 1. Cria o DataFrame híbrido combinando os dados reais e sintéticos
    dados_hibridos = pd.concat([dados_reais_copy, dados_sinteticos_copy], ignore_index=True)

    # 2. Agrupa cada um dos três DataFrames usando a função auxiliar
    real_grouped = agrupar_viagens_por_local(dados_reais_copy)
    synth_grouped = agrupar_viagens_por_local(dados_sinteticos_copy)
    hybrid_grouped = agrupar_viagens_por_local(dados_hibridos)
    dados_eval_grouped = agrupar_viagens_por_local(dados_reais_eval)

    def transformar_coluna_data(df: pd.DataFrame) -> pd.DataFrame:
         """Converte o timestamp em índice de 15 minutos (0-95) e remove o timestamp."""
         return df.assign(
            hora_do_dia=lambda df_interno: quarter_hour_slot(df_interno["tpep_pickup_datetime"])
        ).drop(columns=["tpep_pickup_datetime"])

    # 4. Aplica a transformação aos três DataFrames agrupados
    hybrid_grouped = transformar_coluna_data(hybrid_grouped)
    real_grouped = transformar_coluna_data(real_grouped)
    synth_grouped = transformar_coluna_data(synth_grouped)
    dados_eval_grouped = transformar_coluna_data(dados_eval_grouped)

    # 3. Retorna os três DataFrames agrupados na ordem especificada
    return hybrid_grouped, real_grouped, synth_grouped, dados_eval_grouped