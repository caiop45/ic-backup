import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from pycave.bayes import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# --------------------------------------------------
# 1) LEITURA E PREPARAÇÃO INICIAL DOS DADOS REAIS
# --------------------------------------------------
dados = pd.read_parquet('/home-ext/caioloss/Dados/viagens_lat_long.parquet')

# Converter coluna de data
dados['tpep_pickup_datetime'] = pd.to_datetime(dados['tpep_pickup_datetime'])
# Extrair hora e dia da semana
dados['hora_do_dia'] = dados['tpep_pickup_datetime'].dt.hour
dados['dia_da_semana'] = dados['tpep_pickup_datetime'].dt.dayofweek

# Filtrar apenas segunda(0) a quarta(2), se desejar
dados = dados[dados['dia_da_semana'].between(0, 2)]

# Criar coluna "data_do_dia" (somente a data, sem hora)
dados['data_do_dia'] = dados['tpep_pickup_datetime'].dt.date

# Exemplo de features usadas no GMM
features = ['data_do_dia', 'hora_do_dia', 'PU_longitude', 'PU_latitude', 'DO_longitude', 'DO_latitude']
dados_gmm = dados[features].dropna().sample(frac=1.0, random_state=42)
dados_gmm['data_do_dia'] = pd.to_datetime(dados_gmm['data_do_dia']).apply(lambda x: x.toordinal())
print("Dados carregados e filtrados. Total de registros (para GMM):", len(dados_gmm))

# 