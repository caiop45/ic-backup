import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pycave.bayes import GaussianMixture
import config # Importa as configurações

def train_gmm_and_generate(data_gmm, n_components, multiplier=config.SYNTHETIC_MULTIPLIER):
    """Treina um GMM, gera dados sintéticos e faz o pós-processamento."""
    print(f"Treinando GMM com nc={n_components}...")
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(data_gmm).astype(np.float32)

    gmm = GaussianMixture(
        num_components=n_components,
        covariance_type='full',
        covariance_regularization=1e-5,
        trainer_params={'max_epochs': 100, 'accelerator': 'gpu', 'devices': 1, 'enable_progress_bar': False, 'logger': False} # Silencia logs do trainer
    )
    gmm.fit(train_scaled)

    print(f"Gerando {multiplier}x dados sintéticos...")
    n_samples = int(len(data_gmm) * multiplier)
    synthetic_scaled = gmm.sample(n_samples).cpu().numpy() # Move para CPU antes de converter para numpy
    synthetic_df = pd.DataFrame(
        scaler.inverse_transform(synthetic_scaled),
        columns=config.FEATURES_GMM
    )

    # Pós-processamento dos dados sintéticos
    # Converte timestamp de volta para data e arredonda hora
    synthetic_df['data_do_dia'] = pd.to_datetime(synthetic_df['data_do_dia'], unit='s', errors='coerce').dt.date
    synthetic_df['hora_do_dia'] = synthetic_df['hora_do_dia'].round().astype(int)
    # Remove linhas onde a conversão de data falhou (NaN)
    synthetic_df.dropna(subset=['data_do_dia'], inplace=True)
    # Garante que a hora do dia esteja dentro do intervalo esperado [0, 23]
    synthetic_df['hora_do_dia'] = synthetic_df['hora_do_dia'].clip(0, 23)

    print("Geração de dados sintéticos concluída.")
    return synthetic_df