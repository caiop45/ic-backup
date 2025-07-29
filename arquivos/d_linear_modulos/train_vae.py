# ──────────────────── main.py ────────────────────
import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from config import SYNTHETIC_MULTIPLIER
from data_processing.loader import load_real_data, split_dataset_weekly
from data_processing.gmm_preparer import scale_features           # scaler continua útil
from synthetic_data.date_sampler import make_date_sampler
from utils.helpers import decode_hour_from_sincos
from utils.zone_id import assign_zone_names
from models.vae import VAE                    # —— NOVO —— #

# -------------------------------------------------
# Pequeno *helper* para treinar o VAE
# -------------------------------------------------
def treinar_vae(
    X_scaled: np.ndarray,
    input_dim: int,
    hidden_dims=(32, 16, 8),
    latent_dim: int = 4,
    epochs: int = 5,
    batch_size: int = 512,
    lr: float = 1e-3,
    device: str | torch.device = "cpu",
):
    print(f"# DEBUG: Treinando VAE em {device} com {epochs} épocas…")
    model = VAE(input_dim, hidden_dims, latent_dim).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)

    ds  = TensorDataset(torch.from_numpy(X_scaled))
    dl  = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    mse = nn.MSELoss(reduction="mean")

    model.train()
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        for (x,) in dl:
            x = x.to(device)
            opt.zero_grad()

            recon, mu, logvar = model(x)
            recon_loss = mse(recon, x)
            kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + 1e-3 * kld   # β‑term bem leve
            loss.backward()
            opt.step()

            running_loss += loss.item()

        # DEBUG: imprime 1 linha a cada 25 épocas
        if epoch % 25 == 0 or epoch == 1 or epoch == epochs:
            avg = running_loss / len(dl)
            print(f"  Epoch {epoch:>3}/{epochs} – loss={avg:.4f}")

    print("# DEBUG: VAE treinado ✔️")
    return model


def amostrar_do_vae(
    vae: VAE,
    n_amostras: int,
    scaler,                    # StandardScaler retornado de scale_features
    feature_names: list[str],
    device: str | torch.device = "cpu",
) -> pd.DataFrame:
    print(f"# DEBUG: Amostrando {n_amostras:,} observações do VAE…")
    vae.eval()
    with torch.no_grad():
        z = torch.randn(n_amostras, vae.fc_mu.out_features, device=device)
        synth = vae.decode(z).cpu().numpy()

    synth_df = pd.DataFrame(
        scaler.inverse_transform(synth),
        columns=feature_names,
    )
    synth_df["hora_do_dia"] = decode_hour_from_sincos(
        synth_df["sin_hr"], synth_df["cos_hr"]
    )
    print("# DEBUG: Amostragem concluída, shape =", synth_df.shape)
    return synth_df


# -------------------------------------------------
# Pipeline principal
# -------------------------------------------------
def process_data():
    print("# DEBUG: Carregando dados reais…")
    (
        dados_reais_orig,
        _gmm_placeholder,
        dados_reais_temporal_model_input,
        hour_counts_dict_real,
        GMM_FEATURES,
        DATASAMPLER_FEATURES,
    ) = load_real_data()
    print("# DEBUG: Linhas carregadas:", len(dados_reais_orig))
    print("# DEBUG: Nº de features GMM:", len(GMM_FEATURES))

    # -------- separação --------
    gmm_full = dados_reais_orig[["tpep_pickup_datetime"] + GMM_FEATURES].dropna()
    #rint(gmm_full.head())
    print("# DEBUG: gmm_full shape:", gmm_full.shape)

    gmm_train, gmm_val, gmm_hold = split_dataset_weekly(
        gmm_full,
        train_frac=0.80,
        val_frac=0.20,
        datetime_col="tpep_pickup_datetime",
    )
    print("# DEBUG: split shapes – train:", gmm_train.shape,
          "val:", gmm_val.shape, "hold:", gmm_hold.shape)
   # dados_reais_gmm_train = gmm_train[GMM_FEATURES].astype(np.float32)
    print("--------")
    print(gmm_train.head())
    dados_reais_gmm_train = gmm_train[["sin_hr", "cos_hr"]].astype(np.float32)
    # -------- scaler + amostrador de data --------
    print("# DEBUG: Escalando features…")
    scaler, X_scaled = scale_features(dados_reais_gmm_train)
    print("# DEBUG: X_scaled shape:", X_scaled.shape)

   # sample_date = make_date_sampler(gmm_train, seed=42)

    # --------- treino do VAE ---------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    vae = treinar_vae(
        X_scaled=X_scaled,
        input_dim=X_scaled.shape[1],
        hidden_dims=(32, 16, 8),
        latent_dim=4,
        epochs=5,
        batch_size=512,
        lr=1e-3,
        device=device,
    )

    # --------- pré‑processamento real ---------
    dados_reais_gmm_train["hora_do_dia"] = decode_hour_from_sincos(
        dados_reais_gmm_train["sin_hr"], dados_reais_gmm_train["cos_hr"]
    )
    dados_reais_gmm_train["num_viagens"] = 1
    #dados_reais_gmm_train = assign_zone_names(dados_reais_gmm_train)

    # --------- multiplicador sintético ---------
    n_synth = int(len(dados_reais_gmm_train) * SYNTHETIC_MULTIPLIER)
    print("# DEBUG: n_synth calculado =", n_synth)
    if n_synth == 0:
        raise ValueError("SYNTHETIC_MULTIPLIER gerou n_synth=0!")

    # --------- seeds ---------
    seed = torch.initial_seed()
    torch.manual_seed(3)
    np.random.seed(3)
    rng = np.random.default_rng(3)
    print("# DEBUG: seed =", seed)

    # --------- execuções ---------
    all_runs = []
    for run in range(1, 2):
        print(f"\n===== RUN {run}/5 =====")
        synth_raw_data = amostrar_do_vae(
            vae=vae,
            n_amostras=n_synth,
            scaler=scaler,
            feature_names=["sin_hr", "cos_hr"],
            device=device,
        )
        #synth_raw_data = assign_zone_names(synth_raw_data)

        synth_data = synth_raw_data.copy()
        synth_data = synth_data[['hora_do_dia']]#, 'PULocationID', 'DOLocationID']]


         # 1) Contagem real (gmm_full) ───────────────────────
        gmm_hourly = (
            gmm_full['tpep_pickup_datetime']
            .dt.hour                      # extrai hora 0‑23
            .value_counts()
            .sort_index()
            .rename('viagens_reais')
        )

        # 2) Contagem sintética (synth_data) ────────────────
        synth_hourly = (
            synth_data['hora_do_dia']
            .astype(int)                  # garante inteiro 0‑23
            .value_counts()
            .sort_index()
            .rename('viagens_sinteticas')
        )

        # 3) Junta, calcula diferença percentual ────────────
        idx = pd.RangeIndex(24)           # garante todas as 24 horas
        cmp = pd.concat([gmm_hourly, synth_hourly], axis=1).reindex(idx, fill_value=0)
        cmp['dif_%'] = 100 * (cmp['viagens_sinteticas'] - cmp['viagens_reais']) \
                             / cmp['viagens_reais'].replace(0, np.nan)

        print("\n# DEBUG: comparação synth × real por hora (diferença %)")
        print(cmp.round(2).to_string())

        print("# DEBUG: synth_data head:")
        print(synth_data.head(3).to_string(index=False))
    return all_runs


def main():
    processed_data = process_data()
    print("\n# DEBUG: Loops concluídos – iterando em processed_data")
    for idx, data in enumerate(processed_data, start=1):
        X_val = data["validation"]["X_val"]
        print(f"Run {idx}: shape X_val={X_val.shape}")


if __name__ == "__main__":
    main()
