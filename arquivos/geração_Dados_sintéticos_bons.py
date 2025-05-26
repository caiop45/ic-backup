#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Geração de dados sintéticos com GMM + Optuna + ruído controlado
e avaliação de robustez em vários seeds.

Mudanças em relação à versão anterior
─────────────────────────────────────
1.  Ruído leve pós-geração
    • jitter em coordenadas   (gaussiano, ~55 m σ)
    • jitter em contagens/h   (±5 % do total real por hora)
2.  Loop sobre múltiplos seeds para testar estabilidade

Salva, para cada seed:
    – melhores hiperparâmetros do Optuna
    – MAE horário (após ruído)
    – resumo final em CSV
"""
# ─── IMPORTS ─────────────────────────────────────────────────────────
import os, json, math, warnings
import numpy as np
import pandas as pd
import torch, optuna
from sklearn.preprocessing import StandardScaler
from pycave.bayes import GaussianMixture
# --------------------------------------------------------------------

# ─── CONFIG GERAIS ───────────────────────────────────────────────────
PARQUET_PATH   = "/home-ext/caioloss/Dados/viagens_lat_long.parquet"
SAVE_DIR       = "arquivos/comparação_distribuição_sintética_v4"
os.makedirs(SAVE_DIR, exist_ok=True)

SYNTH_OVERSAMPLE = 3        # fator base de amostra sintética
N_TRIALS         = 20       # Optuna
SEEDS            = [42, 73, 101, 999, 2025]

# Ruído controlado
COUNT_NOISE_FRAC = 0.05     # até ±5 % na contagem/hora
COORD_SIGMA_DEG  = 0.0005   # ~55 m de σ em lat/lon
# --------------------------------------------------------------------

# ─── CARREGAR DADOS REAIS (1×) ───────────────────────────────────────
print("▶️  Carregando dados reais …")
df = pd.read_parquet(PARQUET_PATH)
df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])

# mantém seg–qua, todas as horas
df = df[df["tpep_pickup_datetime"].dt.dayofweek.between(0, 2)]

df["hora_do_dia"] = df["tpep_pickup_datetime"].dt.hour
df["num_viagens"] = 1
real_hour_counts  = df.groupby("hora_do_dia")["num_viagens"].sum().sort_index()
hour_counts_dict  = real_hour_counts.to_dict()

# encode circular da hora
df["sin_hr"] = np.sin(2*np.pi*df["hora_do_dia"]/24)
df["cos_hr"] = np.cos(2*np.pi*df["hora_do_dia"]/24)

FEATURES = [
    "sin_hr", "cos_hr",
    "PU_longitude", "PU_latitude",
    "DO_longitude", "DO_latitude",
]
df_gmm = df[FEATURES].dropna()
scaler = StandardScaler().fit(df_gmm.astype(np.float32))

# ─── FUNÇÕES AUXILIARES ──────────────────────────────────────────────
def build_gmm(n_components:int, cov_type:str, cov_reg:float)->GaussianMixture:
    return GaussianMixture(
        num_components=n_components,
        covariance_type=cov_type,
        covariance_regularization=cov_reg,
        trainer_params={
            "max_epochs":200,
            "accelerator":"auto",
            "devices":1,
        },
    )

def decode_hour(sin_s, cos_s):
    ang = np.mod(np.arctan2(sin_s, cos_s), 2*np.pi)
    return np.rint(ang * 24/(2*np.pi)).astype(int) % 24

def synth_samples(gmm, n):
    synth_scaled = gmm.sample(n).cpu().numpy()
    s_df = pd.DataFrame(
        scaler.inverse_transform(synth_scaled), columns=FEATURES
    )
    s_df["hora_do_dia"] = decode_hour(s_df["sin_hr"], s_df["cos_hr"])
    s_df["num_viagens"] = 1
    return s_df

def equal_freq(s_df):
    rng = np.random.default_rng()
    parts=[]
    for hr, n_real in hour_counts_dict.items():
        sub = s_df[s_df["hora_do_dia"]==hr]
        if sub.empty: continue
        parts.append(
            sub.sample(n=n_real, replace=len(sub)<n_real,
                       random_state=rng.integers(1e6))
        )
    return pd.concat(parts, ignore_index=True)

def qmap(col_s, col_r):
    idx  = col_s.rank(method="first").astype(int) - 1
    real = np.sort(col_r.values)
    tgt  = real[np.clip((idx*len(real)//len(col_s)).values,0,len(real)-1)]
    return pd.Series(tgt, index=col_s.index)

def apply_qmap(s_df, r_df):
    for c in ["PU_longitude","PU_latitude","DO_longitude","DO_latitude"]:
        s_df[c] = qmap(s_df[c], r_df[c])
    return s_df

def perturb_counts(df, rng, frac=COUNT_NOISE_FRAC):
    parts=[]
    for hr in range(24):
        sub = df[df["hora_do_dia"]==hr]
        if sub.empty: continue
        n_real = len(sub)
        delta  = int(rng.normal(0, n_real*frac))
        target = max(1, n_real+delta)

        if target < n_real:
            sub = sub.sample(n=target, replace=False,
                             random_state=rng.integers(1e6))
        elif target > n_real:
            extra = sub.sample(n=target-n_real, replace=True,
                               random_state=rng.integers(1e6))
            sub   = pd.concat([sub, extra], ignore_index=True)
        parts.append(sub)
    return pd.concat(parts, ignore_index=True)

def jitter_coords(df, rng, sigma=COORD_SIGMA_DEG):
    for c in ["PU_latitude","PU_longitude","DO_latitude","DO_longitude"]:
        df[c] += rng.normal(0, sigma, size=len(df))
    return df
# --------------------------------------------------------------------

# Registrar resultados de todas as seeds
all_results = []

# ─── LOOP POR SEED ──────────────────────────────────────────────────
for seed in SEEDS:
    print(f"\n🔄  Seed {seed}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    # ─── Optuna ------------------------------------------------------
    def objective(trial: optuna.Trial) -> float:
        params = dict(
            n_components = trial.suggest_int("n_components", 15, 35, step=1),
            cov_type     = trial.suggest_categorical("cov_type", ["full","diag"]),
            cov_reg      = trial.suggest_float("cov_reg", 1e-5, 1e-2, log=True),
        )
        gmm = build_gmm(**params)
        gmm.fit(scaler.transform(df_gmm).astype(np.float32))

        if torch.isnan(gmm.model_.component_probs).any():
            return np.inf

        try:
            s_df = synth_samples(gmm, int(len(df_gmm)*SYNTH_OVERSAMPLE))
        except ValueError:
            return np.inf

        # Sem equal_freq/ruído aqui – objetivo puro
        s_counts = s_df.groupby("hora_do_dia")["num_viagens"].sum()
        diff = (s_counts.reindex(real_hour_counts.index, fill_value=0)
                - real_hour_counts).abs()
        mae = (diff/real_hour_counts).mean()
        return mae

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=N_TRIALS, n_jobs=1)
    best_params = study.best_params

    # ─── GMM final + geração ----------------------------------------
    best_gmm = build_gmm(**best_params)
    best_gmm.fit(scaler.transform(df_gmm).astype(np.float32))

    synth_df = synth_samples(best_gmm, int(len(df_gmm)*SYNTH_OVERSAMPLE))
    synth_df = equal_freq(synth_df)
    synth_df = perturb_counts(synth_df, rng)
    synth_df = apply_qmap(synth_df, df)
    synth_df = jitter_coords(synth_df, rng)

    # ─── Métrica pós-ruído ------------------------------------------
    synth_hour = synth_df.groupby("hora_do_dia")["num_viagens"].sum()
    diff = (synth_hour.reindex(real_hour_counts.index, fill_value=0)
            - real_hour_counts).abs()
    mae_hour = (diff/real_hour_counts).mean()

    all_results.append({
        "seed": seed,
        "mae_hour": float(mae_hour),
        **best_params
    })

    # Salvar describe de cada seed (opcional)
    descr_path = os.path.join(SAVE_DIR, f"describe_seed_{seed}.json")
    with open(descr_path, "w", encoding="utf-8") as f:
        json.dump({
            "best_params": best_params,
            "real_hour_describe": real_hour_counts.describe().to_dict(),
            "synth_hour_describe": synth_hour.describe().to_dict()
        }, f, indent=2, ensure_ascii=False)

# ─── Resumo geral ----------------------------------------------------
df_res = pd.DataFrame(all_results)

print("\n=========== Robustez (MAE horário) ===========")
print(df_res[["seed","mae_hour"]].to_string(index=False))
print("\nMAE médio  :", df_res["mae_hour"].mean())
print("MAE desvio :", df_res["mae_hour"].std())

# Salva CSV final
csv_path = os.path.join(SAVE_DIR, "robustez_seeds.csv")
df_res.to_csv(csv_path, index=False)
print(f"\n📝  Tabela de robustez salva em {csv_path}")
