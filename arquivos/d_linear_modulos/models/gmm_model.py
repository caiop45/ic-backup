# ─────────────────────────────────────────────────────────
#  util_gmm_optuna.py
# ─────────────────────────────────────────────────────────
from pathlib import Path
from datetime import datetime
import json, numpy as np, torch, optuna
from pycave.bayes import GaussianMixture
from scipy.stats import wasserstein_distance
import pandas as pd

# ------------------- funções auxiliares -------------------
def _count_params(gmm, d: int) -> float:
    k = gmm.num_components
    cov_type = gmm.covariance_type
    if cov_type == "full":
        cov_params = k * d * (d + 1) / 2
    elif cov_type == "diag":
        cov_params = k * d
    else:
        raise ValueError(f"Tipo de covariância desconhecido: {cov_type}")
    return (k - 1) + k * d + cov_params


def _log_like(gmm, X) -> float:
    try:
        return gmm.score_samples(X).sum()
    except AttributeError:
        return gmm.score(X) * X.shape[0]


def _calculate_bic(gmm, X):
    N, d = X.shape
    n_params = _count_params(gmm, d)
    ll = _log_like(gmm, X)
    return n_params * np.log(N) - 2 * ll


def _calculate_aic(gmm, X):
    # N não entra diretamente na fórmula do AIC; mantido p/ interface.
    N, d = X.shape
    n_params = _count_params(gmm, d)
    ll = _log_like(gmm, X)
    return 2 * n_params - 2 * ll

def fit_gmm_multi_objective(
    X_scaled: np.ndarray,
    feature_names: list, # NOVO: Adicione os nomes das features como argumento
    search_space: dict | None = None,
    n_trials: int = 30,
    seed: int | None = None,
):
    """
    Ajusta um GMM minimizando BIC e Distância de Wasserstein com Optuna.

    Retorna
    -------
    study: O objeto de estudo completo do Optuna.
    """
    if search_space is None:
        search_space = {
            "n_components": (20, 60),
            "cov_reg": (1e-6, 1e-3),
            "cov_type": ["full"],
        }

    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    # NOVO: Crie um DataFrame dos dados reais para facilitar o cálculo
    df_real = pd.DataFrame(X_scaled, columns=feature_names)

    # ---------- função objetivo ----------
    # ---------- função objetivo ----------
    def objective(trial):
        # --- ETAPA 1: Sugerir hiperparâmetros e treinar o GMM ---
        nc_low, nc_high = search_space["n_components"]
        nc = trial.suggest_int("n_components", nc_low, nc_high)

        # Usando o espaço de busca ajustado (e.g., com limite inferior de 1e-5)
        cov_low, cov_high = search_space["cov_reg"]
        cov_reg = trial.suggest_float("covariance_regularization", cov_low, cov_high, log=True)

        cov_type = trial.suggest_categorical("covariance_type", search_space["cov_type"])

        gmm = GaussianMixture(
            num_components=nc,
            covariance_type=cov_type,
            covariance_regularization=cov_reg,
            trainer_params={"max_epochs": 200, "accelerator": "auto", "devices": 1, "enable_progress_bar": False},
        )
        
        # Tenta treinar o modelo, se falhar, descarta o trial
        try:
            gmm.fit(X_scaled)
        except (torch._C._LinAlgError, RuntimeError) as e:
            print(f"Trial descartado durante o FIT: {e}")
            raise optuna.TrialPruned()

        # --- ETAPA 2: Calcular as métricas de avaliação ---

        # Objetivo 1: BIC (sempre calculado)
        bic_val = _calculate_bic(gmm, X_scaled)
        
        # Tenta gerar amostras e calcular a segunda métrica
        try:
            # Objetivo 2: Distância de Wasserstein
            # 1. Gerar dados sintéticos
            X_synth_torch = gmm.sample(X_scaled.shape[0])
            df_synth = pd.DataFrame(X_synth_torch.cpu().numpy(), columns=feature_names)

            # 2. Calcular a distância para cada feature e tirar a média
            wasserstein_distances = [
                wasserstein_distance(df_real[col], df_synth[col]) for col in feature_names
            ]
            wasserstein_mean = np.mean(wasserstein_distances)

            # Retorna ambos os objetivos se tudo deu certo
            return bic_val, wasserstein_mean

        except ValueError as e:
            # Se a amostragem falhar (ValueError de pvals), o modelo é instável.
            # Imprime um aviso e descarta ("prune") este trial.
            print(f"Trial descartado devido a modelo instável na amostragem: {e}")
            raise optuna.TrialPruned()

    # MUDANÇA: Criar um estudo MULTIOBJETIVO
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(
        directions=["minimize", "minimize"], # Queremos minimizar o BIC E a distância
        sampler=sampler
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    return study


def multiple_optuna_runs( # MUDANÇA: Adaptar a função que chama
    X_scaled,
    feature_names: list, # NOVO: Passe os nomes das features
    search_space,
    seeds: list[int],
    n_trials: int = 30,
    save_dir: str | Path | None = None,
):
    """
    Executa várias otimizações multiobjetivo e escolhe o melhor resultado final.
    """
    all_pareto_trials = []
    
    for i, seed in enumerate(seeds):
        print(f"🔁 Rodada Multiobjetivo {i+1}/{len(seeds)} — seed={seed}")

        # Chama a nova função de otimização
        study = fit_gmm_multi_objective(
            X_scaled,
            feature_names=feature_names,
            search_space=search_space,
            n_trials=n_trials,
            seed=seed,
        )
        all_pareto_trials.extend(study.best_trials)

    print(f"\nAnalisando {len(all_pareto_trials)} soluções de compromisso encontradas...")
    
    # CRITÉRIO DE ESCOLHA FINAL: Dos melhores modelos, qual tem a menor
    # distância de Wasserstein (ou seja, gera os dados mais fiéis)?
    best_trial_overall = min(all_pareto_trials, key=lambda t: t.values[1]) # values[1] é a Wasserstein
    
    best_params = best_trial_overall.params
    best_bic = best_trial_overall.values[0]
    best_wasserstein = best_trial_overall.values[1] # A nova métrica
    
    print(f"Melhor modelo escolhido: BIC={best_bic:.2f}, Wasserstein={best_wasserstein:.4f}")

    # ---------- refit final com os melhores parâmetros ----------
    best_gmm = GaussianMixture(
        num_components=best_params["n_components"],
        covariance_type=best_params["covariance_type"],
        covariance_regularization=best_params["covariance_regularization"],
        trainer_params={"max_epochs": 400, "accelerator": "auto", "devices": 1},
    )
    best_gmm.fit(X_scaled)
    
    # Retornamos aic como None ou o valor de wasserstein para manter a assinatura
    return best_gmm, best_params, best_bic, best_wasserstein