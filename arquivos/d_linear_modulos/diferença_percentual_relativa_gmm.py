"""
main.py  –  Geração de dados sintéticos e avaliação
Refatorado para melhor legibilidade + prints de depuração
"""

from __future__ import annotations

# ───────────────────────── Imports ──────────────────────────
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from joblib import dump, load
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt

from config import (
    INPUT_WINDOW,
    SYNTHETIC_MULTIPLIER,
    NUM_RUNS,
    DATE_SAMPLER_SEED,
    SAVE_DIR,
)
from data_processing.loader import load_real_data, split_dataset_weekly
from data_processing.gmm_preparer import scale_features
from data_processing.dlinear_preparer import (
    prepare_dlinear_tensors,
    build_input_target_pairs,
    apply_growth_weighting,
    prepare_and_group_datasets,
)
from synthetic_data.date_sampler import make_date_sampler
from synthetic_data.generator import equal_freq, perturb_counts
from models.gmm_model import multiple_optuna_runs
from models import DLinearModel, train_model, optimize_dlinear
from evaluation.metrics import compute_metrics
from evaluation.plotting import (
    generate_plots,
    plot_hourly_trip_comparison,
    plot_random_pair_heatmaps,
    boxplot_model_eval,
)
from pycave.bayes import GaussianMixture
from utils.helpers import decode_hour_from_sincos, group_trips_by_zone, smape, set_global_seed
from utils.zone_id import assign_zone_names, filter_by_zone

# ───────────────────────── Constantes ───────────────────────
BASE_SEED = 372
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ────────────────────── Funções utilitárias ─────────────────
def gen_synth_data(gmm_model: GaussianMixture, n_samples: int) -> pd.DataFrame:
    """Gera amostras sintéticas (já no espaço original)."""
    synth_scaled = gmm_model.sample(n_samples).cpu().numpy()
    synth_df = pd.DataFrame(synth_scaled, columns=["sin_hr", "cos_hr"])

    # Recupera hora do dia
    synth_df["hora_do_dia"] = decode_hour_from_sincos(
        synth_df["sin_hr"], synth_df["cos_hr"]
    ).astype(int)
    return synth_df


def comparar_distribuicao_horaria(
    synth_df: pd.DataFrame, real_df: pd.DataFrame
) -> None:
    """Compara distribuições horárias (real × sintético)."""

    real_counts = (
        pd.to_datetime(real_df["tpep_pickup_datetime"])
        .dt.hour.value_counts()
        .sort_index()
    )
    synth_counts = synth_df["hora_do_dia"].value_counts().sort_index()

    real_probs = real_counts.div(real_counts.sum()).mul(100).round(2)
    synth_probs = synth_counts.div(synth_counts.sum()).mul(100).round(2)

    comp = (
        pd.DataFrame({"real_%": real_probs, "synth_%": synth_probs})
        .fillna(0)
        .round(2)
    )
    comp["Δ (pp)"] = (comp["synth_%"] - comp["real_%"]).round(2)

    print("\n# -------- Distribuição horária (% do total) --------")
    print(comp.to_string())
    print("Max |Δ| =", comp["Δ (pp)"].abs().max(), "pp")
    print("----------------------------------------------------\n")


# ─────────────────────────── main ───────────────────────────
def main() -> None:
    """Pipeline principal."""
    set_global_seed(BASE_SEED)

    # 1) Carrega dados reais
    (
        dados_reais_orig,
        _gmm_placeholder,
        dados_reais_linear_input,
        hour_counts_dict_real,
        GMM_FEATURES,
        DATASAMPLER_FEATURES,
    ) = load_real_data()

    print(f"# DEBUG: Dados reais carregados: {dados_reais_orig.shape} linhas")

    # 2) Split temporal 60/20/20 para GMM
    gmm_full = dados_reais_orig[["tpep_pickup_datetime"] + GMM_FEATURES].dropna()
    gmm_train, gmm_val, gmm_hold = split_dataset_weekly(
        gmm_full,
        train_frac=0.60,
        val_frac=0.20,
        datetime_col="tpep_pickup_datetime",
    )
    print(
        f"# DEBUG: Split GMM – train: {len(gmm_train)}, "
        f"val: {len(gmm_val)}, hold: {len(gmm_hold)}"
    )

    X_train = gmm_train[["sin_hr", "cos_hr"]].to_numpy(dtype=np.float32)

    # 3) Treina GMM com Optuna (múltiplos seeds)
    print("# DEBUG: Iniciando busca Optuna…")
    gmm_model, best_params, best_bic, best_aic = multiple_optuna_runs(
        X_train,
        feature_names = ["sin_hr", "cos_hr"],
        search_space={
            "n_components": (1, 10),
            "cov_reg": (1e-5, 1e-3),
            "cov_type": ["full", "diag"],
        },
        seeds=[35],
        n_trials=50,
    )
    print(f"# DEBUG: Melhor conjunto de hiperparâmetros: {best_params}")
    print(f"# DEBUG: BIC={best_bic:.2f}, AIC={best_aic:.2f}")

    # 4) Ajusta GMM final no set train completo
    gmm_model.fit(X_train)
    print("# DEBUG: GMM final ajustado")

    # 5) Gera dados sintéticos
    n_synth = int(len(X_train) * SYNTHETIC_MULTIPLIER)
    print(f"# DEBUG: Gerando {n_synth:,} amostras sintéticas")
    synth_raw_data = gen_synth_data(gmm_model, n_synth)

    # 6) Compara distribuição horária
    comparar_distribuicao_horaria(synth_df=synth_raw_data, real_df=gmm_train)

    # --- Continuação do pipeline (treino DLinear, plots etc.) ---
    # Adicione prints semelhantes em cada etapa conforme necessário


# ─────────────────────── Entrypoint ─────────────────────────
if __name__ == "__main__":
    main()
