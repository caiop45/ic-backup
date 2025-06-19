# ─────────────────────────────────────────────────────────
#  util_gmm_optuna.py
# ─────────────────────────────────────────────────────────
from pathlib import Path
from datetime import datetime
import json, numpy as np, torch, optuna
from pycave.bayes import GaussianMixture


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


# ------------------- otimização ---------------------------
def fit_gmm_bic_optuna(
    X_scaled,
    search_space: dict | None = None,
    n_trials: int = 30,
    save_dir: str | Path | None = None,
    seed: int | None = None,
):
    """
    Ajusta um GMM minimizando o BIC com Optuna (AIC como critério de desempate).

    Retorna
    -------
    best_gmm, best_params, best_bic, best_aic
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

    # ---------- função objetivo ----------
    def objective(trial):
        nc_low, nc_high = search_space["n_components"]
        nc = trial.suggest_int("n_components", nc_low, nc_high)

        cov_low, cov_high = search_space["cov_reg"]
        cov_reg = trial.suggest_float(
            "covariance_regularization", cov_low, cov_high, log=True
        )

        cov_type = trial.suggest_categorical(
            "covariance_type", search_space["cov_type"]
        )

        gmm = GaussianMixture(
            num_components=nc,
            covariance_type=cov_type,
            covariance_regularization=cov_reg,
            trainer_params={"max_epochs": 200, "accelerator": "auto", "devices": 1},
        )
        gmm.fit(X_scaled)

        bic_val = _calculate_bic(gmm, X_scaled)
        aic_val = _calculate_aic(gmm, X_scaled)

        # --- garante valores válidos/finitos ---
        if not np.isfinite(bic_val):
            bic_val = float("inf")
        if not np.isfinite(aic_val):
            aic_val = float("inf")

        trial.set_user_attr("aic", aic_val)
        return bic_val

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    # ---------- escolhe melhor trial entre COMPLETAS ----------
    complete_trials = [
        t
        for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
        and t.value is not None
        and np.isfinite(t.value)
    ]
    if not complete_trials:
        raise RuntimeError("Nenhuma trial terminou com BIC válido.")

    best_trial = min(
        complete_trials,
        key=lambda t: (t.value, t.user_attrs.get("aic", float("inf"))),
    )
    best_params = best_trial.params
    best_bic = best_trial.value
    best_aic = best_trial.user_attrs["aic"]

    # ---------- refit final ----------
    best_gmm = GaussianMixture(
        num_components=best_params["n_components"],
        covariance_type=best_params["covariance_type"],
        covariance_regularization=best_params["covariance_regularization"],
        trainer_params={"max_epochs": 400, "accelerator": "auto", "devices": 1},
    )
    best_gmm.fit(X_scaled)

 

    return best_gmm, best_params, best_bic, best_aic


def multiple_optuna_runs(
    X_scaled,
    search_space,
    seeds: list[int],
    n_trials: int = 30,
    save_dir: str | Path | None = None,
):
    """
    Executa várias otimizações (uma por seed) e devolve o melhor resultado.
    Critério: menor BIC; empate → menor AIC.
    """
    resultados = []
    melhor_bic = float("inf")
    melhor_aic = float("inf")
    melhor_gmm = None
    melhor_params = None

    for i, seed in enumerate(seeds):
        print(f"🔁 Rodada {i+1}/{len(seeds)} — seed={seed}")

        gmm, params, bic, aic = fit_gmm_bic_optuna(
            X_scaled,
            search_space=search_space,
            n_trials=n_trials,
            save_dir=None,
            seed=seed,
        )
        resultados.append({"seed": seed, "params": params, "bic": bic, "aic": aic})

        if (bic < melhor_bic) or (bic == melhor_bic and aic < melhor_aic):
            melhor_bic = bic
            melhor_aic = aic
            melhor_gmm = gmm
            melhor_params = params

    return melhor_gmm, melhor_params, melhor_bic, melhor_aic
