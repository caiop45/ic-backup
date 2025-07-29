import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.preprocessing import StandardScaler
from models.vae import VAE
from data_processing.loader import load_real_data, split_dataset_weekly
from data_processing.gmm_preparer import scale_features
from utils.helpers import decode_hour_synth


# Carrega os dados reais
(
    dados_reais_orig,
    _dados_gmm,
    _dados_dlinear,
    _hour_counts,
    GMM_FEATURES,
) = load_real_data()

# Prepara dataset para o VAE (mesma abordagem do GMM)
gmm_full = dados_reais_orig[["tpep_pickup_datetime"] + GMM_FEATURES].dropna()
train_df, _val_df, _hold_df = split_dataset_weekly(
    gmm_full, train_frac=0.8, val_frac=0.1, datetime_col="tpep_pickup_datetime"
)
train_data = train_df[GMM_FEATURES].astype("float32")

# Escala os dados
scaler, X_scaled = scale_features(train_data)
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
dataset = TensorDataset(X_tensor)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

# Define o VAE
input_dim = X_tensor.shape[1]
model = VAE(input_dim=input_dim, latent_dim=4)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


def loss_fn(recon_x, x, mu, logvar):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction="sum")
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl

# Treinamento simples
model.train()
for epoch in range(30):
    total = 0.0
    for batch, in dataloader:
        recon, mu, logvar = model(batch)
        loss = loss_fn(recon, batch, mu, logvar)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item()
    print(f"Epoch {epoch+1:02d} | Loss: {total/len(dataset):.4f}")

# ----- Amostragem de novas viagens -----
model.eval()
with torch.no_grad():
    z = torch.randn(1000, 4)  # 1000 viagens sintéticas
    synth_scaled = model.decode(z).cpu().numpy()

synth_df = pd.DataFrame(scaler.inverse_transform(synth_scaled), columns=GMM_FEATURES)
# Converte sin/cos de volta para hora do dia
synth_df["hora_do_dia"] = decode_hour_synth(synth_df["sin_hr"], synth_df["cos_hr"])

print(synth_df.head())