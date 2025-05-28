import torch, numpy as np, pandas as pd

from config import (
    WINDOW, SYNTHETIC_MULTIPLIER, NUM_EXECUCOES, DATA_SAMPLER_SEED, SAVE_DIR
)
from data_processing.loader            import load_real_data
from data_processing.gmm_preparer      import scale_features
from data_processing.dlinear_preparer  import _prep, build_pairs_df, apply_growth_weighting
from synthetic_data.date_sampler       import make_date_sampler
from synthetic_data.generator          import (
    synth_samples_cod1, equal_freq, perturb_counts  #  qmap/jitter ficam opcionais
)
from models.gmm_model                  import fit_gmm
from models.dlinear                    import DLinear, train_model
from evaluation.metrics                import compute_metrics
from evaluation.plotting               import generate_plots

def main():

    # Seeds globais
    torch.manual_seed(42)
    np.random.seed(42)

    # ----- 1) Carrega dados reais -----
    (
        dados_reais_orig,
        dados_reais_gmm,
        dados_reais_dlinear_input,
        hour_counts_dict_real,
        GMM_FEATURES,
    ) = load_real_data()

    if dados_reais_gmm.empty:
        raise ValueError("dados_reais_gmm vazio após pré-processamento.")

    # ----- 2) Escala features p/ GMM -----
    gmm_scaler, X_scaled = scale_features(dados_reais_gmm)

    # Sample-função de datas
    sample_date = make_date_sampler(dados_reais_dlinear_input, seed=DATA_SAMPLER_SEED)

    # DataFrame final de métricas
    df_res = pd.DataFrame(columns=["tipo_dado", "metrica", "valor", "nc", "epochs", "seed"])

    # ----- 3) Loop principal -----
    for nc in [40, 45, 50]:
        print(f"\n🔄  Treinando GMM com nc={nc}")
        gmm = fit_gmm(X_scaled, nc)

        n_synth = int(len(dados_reais_gmm) * SYNTHETIC_MULTIPLIER)
        if n_synth == 0:
            continue

        for run in range(NUM_EXECUCOES):
            seed = torch.initial_seed() + nc + run
            torch.manual_seed(seed); np.random.seed(seed)
            rng = np.random.default_rng(seed)
            print(f"  Execução {run+1}/{NUM_EXECUCOES} | seed={seed}")

            # ---------- geração sintética ----------
            s_raw  = synth_samples_cod1(gmm, n_synth, gmm_scaler, GMM_FEATURES)
            s_eq   = equal_freq(s_raw, hour_counts_dict_real, rng)
            s_pert = perturb_counts(s_eq, rng)              # jitter/qmap opcionais
            if s_pert.empty:
                continue

            s_pert["tpep_pickup_datetime"] = (
                s_pert["hora_do_dia"].apply(sample_date) +
                pd.to_timedelta(s_pert["hora_do_dia"], unit="h")
            )
            s_pert = s_pert.dropna(subset=["tpep_pickup_datetime"])
            s_dlin = s_pert[["tpep_pickup_datetime", "hora_do_dia", "num_viagens"]].copy()

            real_plus = pd.concat([dados_reais_dlinear_input, s_dlin], ignore_index=True)

            groups = {
                "real"          : dados_reais_dlinear_input,
                "synthetic"     : s_dlin,
                "real+synthetic": real_plus,
            }
            prepared = {k: _prep(v) for k, v in groups.items()}
            pairs    = {k: build_pairs_df(v) for k, v in prepared.items()}

            # ---------- treino DLinear ----------
            for epochs in [50, 80, 100, 150, 200, 250]:
                for typ in ["real", "synthetic", "real+synthetic"]:
                    tr = pairs.get(typ, pd.DataFrame())
                    va = pairs.get("real", pd.DataFrame())
                    if tr.empty or va.empty:
                        continue

                    Xtr = tr[[f"h{k}_train" for k in range(WINDOW)]].values.astype(np.float32)
                    ytr = tr["target_train"].values.astype(np.float32)
                    Xva = va[[f"h{k}_val"   for k in range(WINDOW)]].values.astype(np.float32)
                    yva = va["target_val"].values.astype(np.float32)
                    if not Xtr.size or not Xva.size:
                        continue

                    Xt  = torch.tensor(apply_growth_weighting(Xtr))
                    Xv  = torch.tensor(apply_growth_weighting(Xva))
                    yt  = torch.tensor(ytr); yv = torch.tensor(yva)

                    mdl = DLinear(input_len=WINDOW)
                    train_model(mdl, Xt, yt, Xv, yv, epochs=epochs, lr=0.01, patience=10)

                    with torch.no_grad():
                        pred  = mdl(Xv).squeeze().cpu().numpy()
                    true = yv.squeeze().cpu().numpy()
                    if pred.ndim == 0:  pred = pred.reshape(1)
                    if true.ndim == 0: true = true.reshape(1)

                    for met, val in compute_metrics(true, pred).items():
                        df_res.loc[len(df_res)] = [typ, met, val, nc, epochs, seed]
        print(f"🏁  nc={nc} concluído.")

    # ----- 4) Plots & CSV -----
    generate_plots(df_res)
    print(f"🏁  Processo finalizado. Resultados em: {SAVE_DIR}")

if __name__ == "__main__":
    main()
