import numpy as np
import pandas as pd
from config import WINDOW

# ---------- pré-agregação ---------- #
def _prep(df0: pd.DataFrame):
    if df0.empty or not {"tpep_pickup_datetime", "hora_do_dia", "num_viagens"}.issubset(df0.columns):
        return pd.DataFrame(columns=["tpep_pickup_datetime", "hora_do_dia", "num_viagens"])

    df = df0[["tpep_pickup_datetime", "hora_do_dia", "num_viagens"]].copy()
    df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])
    df["data_normalizada"]     = df["tpep_pickup_datetime"].dt.normalize()

    grp = (df.groupby(["data_normalizada", "hora_do_dia"], as_index=False)
             .agg(num_viagens=("num_viagens", "sum")))

    grp["tpep_pickup_datetime"] = grp["data_normalizada"] + pd.to_timedelta(grp["hora_do_dia"], unit="h")
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
