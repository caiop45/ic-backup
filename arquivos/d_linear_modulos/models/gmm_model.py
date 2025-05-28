# ─────────────────────────────────────────────────────────
#  util_gmm_optuna.py
# ─────────────────────────────────────────────────────────
from pathlib import Path
from datetime import datetime
import json, numpy as np, torch
import optuna
from pycave.bayes import GaussianMixture


import numpy as np


#Verificar se a implementação do cálculo de Bayesian Informação
def _calculate_bic(gmm, X):
    """
    Calcula manualmente o BIC para um GMM já ajustado.

    Parâmetros
    ----------
    gmm : GaussianMixture
        Modelo já ajustado (PyCave ou similar).
    X : array-like, shape (N, d)
        Dados usados no ajuste.

    Retorna
    -------
    bic : float
        Valor do BIC.
    """
    N, d = X.shape
    k = gmm.num_components
    cov_type = gmm.covariance_type  

    # 1) Conta parâmetros de covariância
    if cov_type == "full":
        cov_params = k * d * (d + 1) / 2
    elif cov_type == "diag":
        cov_params = k * d
    else:
        raise ValueError(f"Tipo de covariância desconhecido: {cov_type}")

    # 2) Outros parâmetros: pesos (k−1) + médias (k·d)
    n_params = (k - 1) + k * d + cov_params

    # 3) Log-verossimilhança total
    try:
        # se existir método score_samples, que retorna log-densidade por amostra
        log_likelihood = gmm.score_samples(X).sum()
    except AttributeError:
        # caso só exista score (média de log-likelihood por amostra)
        log_likelihood = gmm.score(X) * N

    # 4) BIC
    return n_params * np.log(N) - 2 * log_likelihood


def fit_gmm_bic_optuna(
    X_scaled,
    search_space: dict | None = None,
    n_trials: int = 30,
    save_dir: str | Path | None = None,
    seed: int | None = None,
):
    """
    Ajusta um GMM minimizando o BIC com Optuna.

    Retorna
    -------
    gmm          – modelo final treinado com os melhores hiperparâmetros
    best_params  – dicionário de hiperparâmetros ótimos
    best_bic     – valor mínimo de BIC
    """
    # ---------- espaço de busca padrão ----------
    if search_space is None:
        search_space = {
            "n_components": (20, 60),          # intervalo fechado [min, max]
            "cov_reg": (1e-6, 1e-3),           # escala log-uniforme
            "cov_type": ["full"],              # ou ["full", "diag"]
        }

    # ---------- controlando reprodutibilidade ----------
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    # ---------- função objetivo ----------
    def objective(trial):
        # nº de componentes
        nc_low, nc_high = search_space["n_components"]
        nc = trial.suggest_int("n_components", nc_low, nc_high)

        # regularização da covariância (log-scale)
        cov_low, cov_high = search_space["cov_reg"]
        cov_reg = trial.suggest_float(
            "covariance_regularization", cov_low, cov_high, log=True
        )

        # tipo de matriz de covariância
        cov_type = trial.suggest_categorical("covariance_type",
                                             search_space["cov_type"])

        # ajusta o modelo
        gmm = GaussianMixture(
            num_components=nc,
            covariance_type=cov_type,
            covariance_regularization=cov_reg,
            trainer_params={"max_epochs": 200,
                            "accelerator": "auto",
                            "devices": 1},
        )
        gmm.fit(X_scaled)
        bic_val = _calculate_bic(gmm, X_scaled)

        return bic_val

    # ---------- roda o Optuna ----------
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_trial.params
    best_bic    = study.best_value

    # ---------- refit final com os melhores hiperparâmetros ----------
    best_gmm = GaussianMixture(
        num_components=best_params["n_components"],
        covariance_type=best_params["covariance_type"],
        covariance_regularization=best_params["covariance_regularization"],
        
        trainer_params={"max_epochs": 200,
                        "accelerator": "auto",
                        "devices": 1},
    )
    best_gmm.fit(X_scaled)

    # ---------- opcional: salvar em disco ----------
    if save_dir is not None:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = Path(save_dir) / f"gmm_optuna_best_{ts}.json"
        with open(out_path, "w", encoding="utf-8") as fp:
            json.dump({"best_params": best_params, "best_bic": best_bic},
                      fp, indent=2)

    return best_gmm, best_params, best_bic


def multiple_optuna_runs(
    X_scaled,
    search_space,
    seeds: list[int],
    n_trials: int = 30,
    save_dir: str | Path | None = None,
):
    """
    Executa várias otimizações (uma por seed) e devolve o melhor resultado.
    """
    resultados   = []
    melhor_bic   = float("inf")
    melhor_gmm   = None
    melhor_params = None

    for i, seed in enumerate(seeds):
        print(f"🔁 Rodada {i+1}/{len(seeds)} — seed={seed}")

        gmm, params, bic = fit_gmm_bic_optuna(
            X_scaled,
            search_space=search_space,
            n_trials=n_trials,
            save_dir=None,        # salvar tudo só no fim, se quiser
            seed=seed,
        )
        resultados.append({"seed": seed, "params": params, "bic": bic})

        if bic < melhor_bic:
            melhor_bic   = bic
            melhor_gmm   = gmm
            melhor_params = params

    # opcional: dump geral dos resultados
    if save_dir is not None:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        out_path = Path(save_dir) / "gmm_multiple_optuna_results.json"
        with open(out_path, "w", encoding="utf-8") as fp:
            json.dump(
                {
                    "melhor_bic": melhor_bic,
                    "melhor_params": melhor_params,
                    "resultados_todas_as_runs": resultados,
                },
                fp,
                indent=2,
            )

    return melhor_gmm, melhor_params, melhor_bic
