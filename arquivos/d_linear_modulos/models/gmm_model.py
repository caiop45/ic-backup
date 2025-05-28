from pycave.bayes import GaussianMixture

def fit_gmm(X_scaled, n_components):
    """
    Ajusta um GMM a X_scaled e devolve o modelo.
    """
    gmm = GaussianMixture(
        num_components=n_components,
        covariance_type="full",
        covariance_regularization=1e-4,
        trainer_params={"max_epochs": 200, "accelerator": "auto", "devices": 1},
    )
    gmm.fit(X_scaled)
    return gmm
