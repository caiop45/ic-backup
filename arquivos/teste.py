import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score, \
    log_loss, precision_recall_fscore_support, mean_absolute_error, mean_absolute_percentage_error, r2_score, mean_squared_error
from scipy.stats import pearsonr, gaussian_kde
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import sys
sys.path.insert(0, '/home/caioloss/LightGBM/python-package')
from lightgbm import LGBMRegressor
from scipy.special import rel_entr
from unidecode import unidecode
import re
import warnings
import logging
logging.getLogger("lightgbm").setLevel(logging.ERROR)



logging.getLogger("lightgbm").setLevel(logging.CRITICAL)
warnings.simplefilter(action='ignore')
optuna.logging.set_verbosity(optuna.logging.ERROR)

# --------------------------------------------------
# Parâmetros de execução
# --------------------------------------------------
n_runs = 2  # <-- defina aqui quantas vezes quer rodar
errors_file = 'kfold_errors_regressão.csv'
features_file = 'top_features_regressão.csv'

# remove arquivos antigos para começar do zero
import os
for f in (errors_file, features_file):
    if os.path.exists(f):
        os.remove(f)

# --------------------------------------------------
# Loop de runs
# --------------------------------------------------
for run in range(1, n_runs + 1):
    print(f"\n=== Iniciando run {run}/{n_runs} ===")

    """# Regressão tempo total"""

    processos_fin = pd.read_excel(
        r"/home/caioloss/arquivos/Dados_do_Processo_Encerrados.xlsx"
    ).dropna(subset=['Resultado', 'Motivo de Encerramento '])
  #  procesos_fin = processos_fin[:500]
    print('total listado', len(processos_fin))

    processos_fin['Vara/Local'] = processos_fin['Vara/Local'] + ' ' + processos_fin['Comarca']
    processos_fin['duracao_dias'] = (
        processos_fin['Data de encerramento'] - processos_fin['Data de Citação']
    ).dt.days
    processos_fin = processos_fin[
        ~processos_fin['Autor'].isin(['CONSTRUTORA TENDA S.A', 'TENDA NEGÓCIOS IMOBILIÁRIOS S/A'])
    ]
    gp = processos_fin.groupby('Comarca')
    processos_fin = pd.concat([gp.get_group(g) for g in gp.groups if len(gp.get_group(g)) >= 2])
    gp = processos_fin.groupby('Advogado da Parte Contrária')
    processos_fin = pd.concat([gp.get_group(g) for g in gp.groups if len(gp.get_group(g)) >= 2])

    print("total após filtragem", len(processos_fin))
    acordo = processos_fin[
        np.logical_and(processos_fin['Esfera'] == 'Judicial', processos_fin['Resultado'] == 'ACORDO')
    ].rename(columns={'Processo - ID': 'id_processo'})
    perdas = processos_fin[
        np.logical_and(processos_fin['Esfera'] == 'Judicial', processos_fin['Resultado'].str.contains('Procedente'))
    ].rename(columns={'Processo - ID': 'id_processo'})
    improc = processos_fin[
        np.logical_and(processos_fin['Esfera'] == 'Judicial', processos_fin['Resultado'].str.contains('IMPROCEDENTE'))
    ].rename(columns={'Processo - ID': 'id_processo'})
    print('acordos', len(acordo))
    print('julgado procedente (perda)', len(perdas))
    print('julgado improcedente (ganho)', len(improc))

    classify = pd.concat([perdas, improc, acordo])[[
        'id_processo', 'Comarca', 'Nome do Empreendimento', 'Tipo do Empreendimento',
        'Advogado da Parte Contrária', 'Causa de Pedir', 'Valor da Causa', 'duracao_dias', 'Resultado'
    ]].dropna(how='any')

    col_class = {}
    for s in classify.columns:
        if s not in ['id_processo', 'Valor da Causa', 'duracao_dias', 'Resultado']:
            classify[s] = classify[s].apply(lambda x: unidecode(x).lower().strip())
            quantized, classes = classify[s].factorize()
            col_class[unidecode(s)] = {n: i for n, i in enumerate(classes)}
            classify[s] = quantized

    value_scaler = StandardScaler()
    value_scaler.fit(classify['Valor da Causa'].values.reshape(len(classify), 1))
    classify['valor_inicial'] = value_scaler.transform(
        classify['Valor da Causa'].values.reshape(len(classify), 1)
    )
    time_scaler = StandardScaler()
    classify = classify.drop(columns=['Valor da Causa']).set_index('id_processo')
    time_scaler.fit(classify['duracao_dias'].values.reshape(len(classify), 1))
    classify['duracao_dias'] = time_scaler.transform(
        classify['duracao_dias'].values.reshape(len(classify), 1)
    )

    class_featurelist = [c for c in classify.columns if c not in ['Resultado', 'duracao_dias']]

    def objective(trial, x_train, y_train):
        params = {
            "boosting_type": trial.suggest_categorical('boosting_type', ['gbdt', 'dart']),
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=10),
            "learning_rate": trial.suggest_loguniform("learning_rate", 1E-5, 0.5),
            "num_leaves": trial.suggest_int("num_leaves", 20, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 30),
            "min_child_weight": trial.suggest_loguniform("min_child_weight", 1e-5, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 1, 50),
            "subsample": trial.suggest_float("subsample", 0.01, 1.0),
            "subsample_freq": trial.suggest_int("subsample_freq", 0, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.01, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 10),
            "reg_lambda": trial.suggest_float("reg_lambda", 0, 10),
        }

        kf = KFold(n_splits=8, shuffle=True, random_state=42)
        mmetric = []

        for train_index, test_index in kf.split(x_train, y_train):
            aux_x_train, aux_x_test = x_train.iloc[train_index], x_train.iloc[test_index]
            aux_y_train, aux_y_test = y_train.iloc[train_index], y_train.iloc[test_index]

            model = LGBMRegressor(**params, verbose=-1, n_jobs = 40)
            model.fit(aux_x_train, aux_y_train)
            y_pred = model.predict(aux_x_test)
            mae = mean_absolute_error(aux_y_test, y_pred)
            mmetric.append(mae)

        return float(np.sqrt(np.mean(np.square(mmetric))))

    study = optuna.create_study(
        sampler=TPESampler(), pruner=MedianPruner(), direction="minimize"
    )
    x_train = classify[class_featurelist].rename(
        columns=lambda col: re.sub(r'[^a-zA-Z0-9_]', '_', col)
    )
    y_train = classify['duracao_dias']
    study.optimize(
        lambda trial: objective(trial, x_train, y_train),
        n_trials=200, timeout=None,
        show_progress_bar=True
    )
    best_params = study.best_params

    # Avaliação final
    kf = KFold(n_splits=10, shuffle=True)
    mean_mdae = []
    mean_mae = []
    imports = []

    for train_index, test_index in kf.split(x_train, y_train):
        aux_x_train, aux_x_test = x_train.iloc[train_index], x_train.iloc[test_index]
        aux_y_train, aux_y_test = y_train.iloc[train_index], y_train.iloc[test_index]

        model = LGBMRegressor(**best_params, verbose=-1)
        fitted = model.fit(aux_x_train, aux_y_train)
        y_pred = model.predict(aux_x_test)

        ref = time_scaler.inverse_transform(aux_y_test.values.reshape(-1, 1)).flatten()
        pred = time_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

        mean_mdae.append(np.median(np.abs(ref - pred)))
        mean_mae.append(mean_absolute_error(ref, pred))

        try:
            importances = pd.Series(fitted.feature_importances_, index=x_train.columns)
        except:
            importances = pd.Series(fitted.coef_, index=x_train.columns)
        imports.append(importances)

    # preparar DataFrames de saída
    errors_df = pd.DataFrame({
        'median_error': mean_mdae,
        'mae': mean_mae
    })
    import_frame = pd.DataFrame(imports).median()
    top_features = import_frame.sort_values(ascending=False).head(10)
    top_feat_df = pd.DataFrame({
        'Posicao': range(1, len(top_features) + 1),
        'Feature': top_features.index,
        'Importance': top_features.values
    })

    # grava em append, só escreve header na primeira run
    errors_df.to_csv(
        errors_file,
        index=False,
        mode='a',
        header=(run == 1)
    )
    top_feat_df.to_csv(
        features_file,
        index=False,
        mode='a',
        header=(run == 1)
    )

print(f"\nTodas as {n_runs} runs concluídas.")
