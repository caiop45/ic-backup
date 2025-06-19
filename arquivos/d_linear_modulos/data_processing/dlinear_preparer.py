import numpy as np
import pandas as pd
from config import INPUT_WINDOW, ZONE_CORRELATION_CSV
import torch
from utils.helpers import group_trips_by_zone, quarter_hour_index

# ---------- pré-agregação ---------- #
def preprocess_trip_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate trip counts by 15 minute interval."""
    if df.empty or not {"tpep_pickup_datetime", "hour_of_day", "trip_count"}.issubset(df.columns):
        return pd.DataFrame(columns=["tpep_pickup_datetime", "hour_of_day", "trip_count"])

    df = df[["tpep_pickup_datetime", "hour_of_day", "trip_count"]].copy()
    df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])
    df["normalized_date"] = df["tpep_pickup_datetime"].dt.normalize()
    df["hour_of_day"] = quarter_hour_index(df["tpep_pickup_datetime"])

    grp = (
        df.groupby(["normalized_date", "hour_of_day"], as_index=False)
          .agg(trip_count=("trip_count", "sum"))
    )

    grp["tpep_pickup_datetime"] = grp["normalized_date"] + pd.to_timedelta(grp["hour_of_day"] * 15, unit="m")

    return grp[["tpep_pickup_datetime", "hour_of_day", "trip_count"]]

# ---------- pares entrada-alvo ---------- #
def build_input_target_pairs(group_df: pd.DataFrame, window: int = INPUT_WINDOW):
    if group_df.empty or group_df["trip_count"].sum() == 0:
        return pd.DataFrame()

    df = group_df.copy()
    df["date"] = pd.to_datetime(df["tpep_pickup_datetime"]).dt.normalize()

    mat = (df.pivot_table(index="date",
                          columns="hour_of_day",
                          values="trip_count",
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

def create_windows(
    df: pd.DataFrame,
    input_window_size: int,
    prediction_horizon: int,
    device: torch.device,
):
    """
    Função auxiliar que transforma um DataFrame em janelas (X, y) já como tensores do PyTorch.
    """
    device = torch.device('cuda:0')
    target_cols = [col for col in df.columns if col != 'hour_of_day']
    
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


def prepare_dlinear_tensors(
    training_groups: dict,
    validation_df: pd.DataFrame,
    input_window_size: int = 3, 
    prediction_horizon: int = 1,
    device: torch.device = torch.device('cuda:0')
    ):
    """Prepare PyTorch tensors for the DLinear model."""
    
    windowed_data = {}

    print(f"Preparando dados e movendo tensores para o dispositivo: '{device}'")

    for name, train_df in training_groups.items():
        X_train, y_train = create_windows(train_df, input_window_size, prediction_horizon, device)
        windowed_data[name] = {
            'X_train': X_train,
            'y_train': y_train
        }
        print(f"-> Treino '{name}' processado. Shape X_train: {X_train.shape}, Shape y_train: {y_train.shape}")

    X_val, y_val = create_windows(validation_df, input_window_size, prediction_horizon, device)
    windowed_data['validation'] = {
        'X_val': X_val,
        'y_val': y_val
    }
    print(f"-> Validação processada. Shape X_val: {X_val.shape}, Shape y_val: {y_val.shape}")
    
    return windowed_data

def prepare_and_group_datasets(
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    eval_real_data: pd.DataFrame,
    *,
    only_existing_zones: bool = False,
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
    real_copy = real_data.copy()
    synthetic_copy = synthetic_data.copy()

    real_copy['tpep_pickup_datetime'] = pd.to_datetime(
        real_copy['tpep_pickup_datetime']
    ).dt.floor('15min')
    synthetic_copy['tpep_pickup_datetime'] = pd.to_datetime(
        synthetic_copy['tpep_pickup_datetime']
    ).dt.floor('15min')
    eval_real_data['tpep_pickup_datetime'] = pd.to_datetime(
        eval_real_data['tpep_pickup_datetime']
    ).dt.floor('15min')

    # 1. Cria o DataFrame híbrido combinando os dados reais e sintéticos
    hybrid_data = pd.concat([real_copy, synthetic_copy], ignore_index=True)

    # Lista de zonas utilizadas em todas as tabelas
    if only_existing_zones:
        cor_df = pd.read_csv(
            ZONE_CORRELATION_CSV, dtype={"LocationID": "int32"}
        )
        all_zones = cor_df["zone"].tolist()
        existing_set = set(hybrid_data["PULocationID"].dropna().unique())
        zones_for_group = [z for z in all_zones if z in existing_set]
    else:
        zones_for_group = None

    # 2. Agrupa cada um dos três DataFrames usando a função auxiliar
    real_grouped = group_trips_by_zone(real_copy, expected_zones=zones_for_group)
    synth_grouped = group_trips_by_zone(
        synthetic_copy, expected_zones=zones_for_group
    )
    hybrid_grouped = group_trips_by_zone(hybrid_data, expected_zones=zones_for_group)
    eval_grouped = group_trips_by_zone(
        eval_real_data, expected_zones=zones_for_group
    )

    def transform_timestamp_column(df: pd.DataFrame) -> pd.DataFrame:
        """Convert timestamp to quarter-hour index and drop the original column."""
        return (
            df.assign(
                hour_of_day=lambda inner: quarter_hour_index(inner["tpep_pickup_datetime"])
            )
            .drop(columns=["tpep_pickup_datetime"])
        )

    # 4. Aplica a transformação aos três DataFrames agrupados
    hybrid_grouped = transform_timestamp_column(hybrid_grouped)
    real_grouped = transform_timestamp_column(real_grouped)
    synth_grouped = transform_timestamp_column(synth_grouped)
    eval_grouped = transform_timestamp_column(eval_grouped)

    # 3. Retorna os três DataFrames agrupados na ordem especificada
    return hybrid_grouped, real_grouped, synth_grouped, eval_grouped
