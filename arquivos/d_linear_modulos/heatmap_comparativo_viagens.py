#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para gerar heatmaps interativos comparativos de viagens de táxi de NYC.
Gera heatmaps para dados sintéticos e reais, incluindo outliers para análise completa.
Usa apenas 10% dos dados de forma aleatória para melhorar a performance.
"""

import pandas as pd
import folium
from folium.plugins import HeatMap
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def carregar_dados_sinteticos(caminho_arquivo: str, fracao: float = 0.1, seed: int = 42) -> pd.DataFrame:
    """
    Carrega os dados sintéticos de viagens de táxi.
    
    Args:
        caminho_arquivo: Caminho para o arquivo parquet com os dados sintéticos
        fracao: Fração dos dados a serem carregados (0.1 = 10%)
        seed: Semente para reprodutibilidade
        
    Returns:
        DataFrame com os dados de viagens sintéticas (amostra)
    """
    print(f"Carregando dados sintéticos de: {caminho_arquivo}")
    df = pd.read_parquet(caminho_arquivo)
    print(f"Dados sintéticos carregados: {df.shape[0]:,} viagens")
    print(f"Colunas disponíveis: {df.columns.tolist()}")
    
    # Amostrar 10% dos dados de forma aleatória
    df_amostra = df.sample(frac=fracao, random_state=seed, replace=False)
    print(f"Amostra sintética (10%): {len(df_amostra):,} viagens")
    
    return df_amostra

def carregar_dados_reais(caminho_arquivo: str, fracao: float = 0.1, seed: int = 42) -> pd.DataFrame:
    """
    Carrega os dados reais de viagens de táxi.
    
    Args:
        caminho_arquivo: Caminho para o arquivo parquet com os dados reais
        fracao: Fração dos dados a serem carregados (0.1 = 10%)
        seed: Semente para reprodutibilidade
        
    Returns:
        DataFrame com os dados de viagens reais (amostra)
    """
    print(f"Carregando dados reais de: {caminho_arquivo}")
    df = pd.read_parquet(caminho_arquivo)
    print(f"Dados reais carregados: {df.shape[0]:,} viagens")
    print(f"Colunas disponíveis: {df.columns.tolist()}")
    
    # Amostrar 10% dos dados de forma aleatória
    df_amostra = df.sample(frac=fracao, random_state=seed, replace=False)
    print(f"Amostra real (10%): {len(df_amostra):,} viagens")
    
    return df_amostra

def analisar_dados_comparativo(df_sintetico: pd.DataFrame, df_real: pd.DataFrame) -> None:
    """
    Realiza análise exploratória comparativa dos dados sintéticos e reais.
    
    Args:
        df_sintetico: DataFrame com os dados de viagens sintéticas (amostra)
        df_real: DataFrame com os dados de viagens reais (amostra)
    """
    print("\n" + "="*60)
    print("ANÁLISE EXPLORATÓRIA COMPARATIVA (AMOSTRAS DE 10%)")
    print("="*60)
    
    # Estatísticas básicas das coordenadas - Sintéticos
    print("\nESTATÍSTICAS DOS DADOS SINTÉTICOS (AMOSTRA):")
    coord_cols_synth = ['PU_latitude', 'PU_longitude', 'DO_latitude', 'DO_longitude']
    print(df_sintetico[coord_cols_synth].describe())
    
    # Estatísticas básicas das coordenadas - Reais
    print("\nESTATÍSTICAS DOS DADOS REAIS (AMOSTRA):")
    coord_cols_real = ['PU_latitude', 'PU_longitude', 'DO_latitude', 'DO_longitude']
    print(df_real[coord_cols_real].describe())
    
    # Verificar se há valores nulos
    print(f"\nValores nulos - Sintéticos:")
    print(df_sintetico.isnull().sum())
    print(f"\nValores nulos - Reais:")
    print(df_real.isnull().sum())
    
    # Verificar se as coordenadas estão dentro dos limites de NYC (apenas para informação)
    nyc_lat_min, nyc_lat_max = 40.5, 40.9
    nyc_lon_min, nyc_lon_max = -74.3, -73.7
    
    print(f"\nVerificação de coordenadas dentro de NYC (apenas informativo):")
    print(f"Limites NYC: Lat [{nyc_lat_min}, {nyc_lat_max}], Lon [{nyc_lon_min}, {nyc_lon_max}]")
    
    # Sintéticos
    pu_nyc_synth = ((df_sintetico['PU_latitude'] >= nyc_lat_min) & (df_sintetico['PU_latitude'] <= nyc_lat_max) &
                    (df_sintetico['PU_longitude'] >= nyc_lon_min) & (df_sintetico['PU_longitude'] <= nyc_lon_max))
    do_nyc_synth = ((df_sintetico['DO_latitude'] >= nyc_lat_min) & (df_sintetico['DO_latitude'] <= nyc_lat_max) &
                    (df_sintetico['DO_longitude'] >= nyc_lon_min) & (df_sintetico['DO_longitude'] <= nyc_lon_max))
    
    # Reais
    pu_nyc_real = ((df_real['PU_latitude'] >= nyc_lat_min) & (df_real['PU_latitude'] <= nyc_lat_max) &
                   (df_real['PU_longitude'] >= nyc_lon_min) & (df_real['PU_longitude'] <= nyc_lon_max))
    do_nyc_real = ((df_real['DO_latitude'] >= nyc_lat_min) & (df_real['DO_latitude'] <= nyc_lat_max) &
                   (df_real['DO_longitude'] >= nyc_lon_min) & (df_real['DO_longitude'] <= nyc_lon_max))
    
    print(f"\nSINTÉTICOS (AMOSTRA):")
    print(f"  Pontos de embarque em NYC: {pu_nyc_synth.sum():,} ({pu_nyc_synth.mean()*100:.1f}%)")
    print(f"  Pontos de desembarque em NYC: {do_nyc_synth.sum():,} ({do_nyc_synth.mean()*100:.1f}%)")
    
    print(f"\nREAIS (AMOSTRA):")
    print(f"  Pontos de embarque em NYC: {pu_nyc_real.sum():,} ({pu_nyc_real.mean()*100:.1f}%)")
    print(f"  Pontos de desembarque em NYC: {do_nyc_real.sum():,} ({do_nyc_real.mean()*100:.1f}%)")

def criar_heatmap_embarque(df: pd.DataFrame, output_path: str, titulo: str = "Pontos de Embarque") -> None:
    """
    Cria um heatmap interativo para pontos de embarque (PU).
    
    Args:
        df: DataFrame com os dados de viagens (amostra)
        output_path: Caminho para salvar o arquivo HTML
        titulo: Título do heatmap
    """
    print(f"\nCriando heatmap de pontos de EMBARQUE: {titulo}")
    
    # Preparar dados para o heatmap (sem filtrar outliers)
    pu_data = df[['PU_latitude', 'PU_longitude']].dropna()
    print(f"Pontos de embarque válidos: {len(pu_data):,}")
    
    # Converter para lista de [lat, lon] para o Folium
    heatmap_data = pu_data.values.tolist()
    
    # Calcular centro do mapa
    center_lat = pu_data['PU_latitude'].mean()
    center_lon = pu_data['PU_longitude'].mean()
    
    # Criar mapa base
    mapa = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=10,  # Zoom menor para ver outliers
        tiles='OpenStreetMap'
    )
    
    # Adicionar heatmap
    HeatMap(
        heatmap_data,
        radius=15,
        blur=10,
        max_zoom=13,
        gradient={0.2: 'blue', 0.4: 'lime', 0.6: 'orange', 1: 'red'}
    ).add_to(mapa)
    
    # Adicionar camadas de tiles
    folium.TileLayer('cartodbpositron', name='Mapa Claro').add_to(mapa)
    folium.TileLayer('cartodbdark_matter', name='Mapa Escuro').add_to(mapa)
    folium.LayerControl().add_to(mapa)
    
    # Adicionar informações no mapa
    folium.Marker(
        [center_lat, center_lon],
        popup=f'Centro do mapa<br>{titulo}<br>Total de viagens: {len(heatmap_data):,}<br>(Amostra de 10%)',
        icon=folium.Icon(color='red', icon='info-sign')
    ).add_to(mapa)
    
    # Salvar mapa
    mapa.save(output_path)
    print(f"Heatmap de embarque salvo em: {output_path}")

def criar_heatmap_desembarque(df: pd.DataFrame, output_path: str, titulo: str = "Pontos de Desembarque") -> None:
    """
    Cria um heatmap interativo para pontos de desembarque (DO).
    
    Args:
        df: DataFrame com os dados de viagens (amostra)
        output_path: Caminho para salvar o arquivo HTML
        titulo: Título do heatmap
    """
    print(f"\nCriando heatmap de pontos de DESEMBARQUE: {titulo}")
    
    # Preparar dados para o heatmap (sem filtrar outliers)
    do_data = df[['DO_latitude', 'DO_longitude']].dropna()
    print(f"Pontos de desembarque válidos: {len(do_data):,}")
    
    # Converter para lista de [lat, lon] para o Folium
    heatmap_data = do_data.values.tolist()
    
    # Calcular centro do mapa
    center_lat = do_data['DO_latitude'].mean()
    center_lon = do_data['DO_longitude'].mean()
    
    # Criar mapa base
    mapa = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=10,  # Zoom menor para ver outliers
        tiles='OpenStreetMap'
    )
    
    # Adicionar heatmap
    HeatMap(
        heatmap_data,
        radius=15,
        blur=10,
        max_zoom=13,
        gradient={0.2: 'blue', 0.4: 'lime', 0.6: 'orange', 1: 'red'}
    ).add_to(mapa)
    
    # Adicionar camadas de tiles
    folium.TileLayer('cartodbpositron', name='Mapa Claro').add_to(mapa)
    folium.TileLayer('cartodbdark_matter', name='Mapa Escuro').add_to(mapa)
    folium.LayerControl().add_to(mapa)
    
    # Adicionar informações no mapa
    folium.Marker(
        [center_lat, center_lon],
        popup=f'Centro do mapa<br>{titulo}<br>Total de viagens: {len(heatmap_data):,}<br>(Amostra de 10%)',
        icon=folium.Icon(color='red', icon='info-sign')
    ).add_to(mapa)
    
    # Salvar mapa
    mapa.save(output_path)
    print(f"Heatmap de desembarque salvo em: {output_path}")

def criar_heatmap_combinado(df: pd.DataFrame, output_path: str, titulo: str = "Pontos Combinados") -> None:
    """
    Cria um heatmap interativo combinado com opções para visualizar embarque/desembarque.
    
    Args:
        df: DataFrame com os dados de viagens (amostra)
        output_path: Caminho para salvar o arquivo HTML
        titulo: Título do heatmap
    """
    print(f"\nCriando heatmap COMBINADO: {titulo}")
    
    # Dados de embarque (sem filtrar outliers)
    pu_data = df[['PU_latitude', 'PU_longitude']].dropna()
    
    # Dados de desembarque (sem filtrar outliers)
    do_data = df[['DO_latitude', 'DO_longitude']].dropna()
    
    # Calcular centro do mapa
    center_lat = (pu_data['PU_latitude'].mean() + do_data['DO_latitude'].mean()) / 2
    center_lon = (pu_data['PU_longitude'].mean() + do_data['DO_longitude'].mean()) / 2
    
    # Criar mapa base
    mapa = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=10,  # Zoom menor para ver outliers
        tiles='OpenStreetMap'
    )
    
    # Adicionar heatmap de embarque
    pu_heatmap = HeatMap(
        pu_data.values.tolist(),
        radius=15,
        blur=10,
        max_zoom=13,
        gradient={0.2: 'blue', 0.4: 'lime', 0.6: 'orange', 1: 'red'},
        name='Pontos de Embarque (PU)'
    )
    pu_heatmap.add_to(mapa)
    
    # Adicionar heatmap de desembarque
    do_heatmap = HeatMap(
        do_data.values.tolist(),
        radius=15,
        blur=10,
        max_zoom=13,
        gradient={0.2: 'purple', 0.4: 'cyan', 0.6: 'yellow', 1: 'red'},
        name='Pontos de Desembarque (DO)'
    )
    do_heatmap.add_to(mapa)
    
    # Adicionar camadas de tiles
    folium.TileLayer('cartodbpositron', name='Mapa Claro').add_to(mapa)
    folium.TileLayer('cartodbdark_matter', name='Mapa Escuro').add_to(mapa)
    
    # Controle de camadas
    folium.LayerControl().add_to(mapa)
    
    # Adicionar informações no mapa
    folium.Marker(
        [center_lat, center_lon],
        popup=f'Centro do mapa<br>{titulo}<br>Embarques: {len(pu_data):,}<br>Desembarques: {len(do_data):,}<br>(Amostra de 10%)',
        icon=folium.Icon(color='red', icon='info-sign')
    ).add_to(mapa)
    
    # Salvar mapa
    mapa.save(output_path)
    print(f"Heatmap combinado salvo em: {output_path}")

def gerar_graficos_comparativos(df_sintetico: pd.DataFrame, df_real: pd.DataFrame, output_dir: str) -> None:
    """
    Gera gráficos comparativos entre dados sintéticos e reais.
    
    Args:
        df_sintetico: DataFrame com os dados sintéticos (amostra)
        df_real: DataFrame com os dados reais (amostra)
        output_dir: Diretório para salvar os gráficos
    """
    print(f"\nGerando gráficos comparativos...")
    
    # Configurar estilo
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Distribuição de coordenadas de embarque - Latitude
    axes[0, 0].hist(df_sintetico['PU_latitude'], bins=50, alpha=0.7, label='Sintético', color='blue', density=True)
    axes[0, 0].hist(df_real['PU_latitude'], bins=50, alpha=0.7, label='Real', color='red', density=True)
    axes[0, 0].set_title('Distribuição Latitude - Embarque (Amostra 10%)')
    axes[0, 0].set_xlabel('Latitude')
    axes[0, 0].set_ylabel('Densidade')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Distribuição de coordenadas de embarque - Longitude
    axes[0, 1].hist(df_sintetico['PU_longitude'], bins=50, alpha=0.7, label='Sintético', color='blue', density=True)
    axes[0, 1].hist(df_real['PU_longitude'], bins=50, alpha=0.7, label='Real', color='red', density=True)
    axes[0, 1].set_title('Distribuição Longitude - Embarque (Amostra 10%)')
    axes[0, 1].set_xlabel('Longitude')
    axes[0, 1].set_ylabel('Densidade')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Distribuição de coordenadas de desembarque - Latitude
    axes[0, 2].hist(df_sintetico['DO_latitude'], bins=50, alpha=0.7, label='Sintético', color='blue', density=True)
    axes[0, 2].hist(df_real['DO_latitude'], bins=50, alpha=0.7, label='Real', color='red', density=True)
    axes[0, 2].set_title('Distribuição Latitude - Desembarque (Amostra 10%)')
    axes[0, 2].set_xlabel('Latitude')
    axes[0, 2].set_ylabel('Densidade')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Distribuição de coordenadas de desembarque - Longitude
    axes[1, 0].hist(df_sintetico['DO_longitude'], bins=50, alpha=0.7, label='Sintético', color='blue', density=True)
    axes[1, 0].hist(df_real['DO_longitude'], bins=50, alpha=0.7, label='Real', color='red', density=True)
    axes[1, 0].set_title('Distribuição Longitude - Desembarque (Amostra 10%)')
    axes[1, 0].set_xlabel('Longitude')
    axes[1, 0].set_ylabel('Densidade')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Scatter plot comparativo - Embarque
    axes[1, 1].scatter(df_sintetico['PU_longitude'], df_sintetico['PU_latitude'], alpha=0.1, s=1, color='blue', label='Sintético')
    axes[1, 1].scatter(df_real['PU_longitude'], df_real['PU_latitude'], alpha=0.1, s=1, color='red', label='Real')
    axes[1, 1].set_title('Comparação Espacial - Embarque (Amostra 10%)')
    axes[1, 1].set_xlabel('Longitude')
    axes[1, 1].set_ylabel('Latitude')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Histograma de distâncias (aproximada)
    # Sintéticos
    lat_diff_synth = df_sintetico['DO_latitude'] - df_sintetico['PU_latitude']
    lon_diff_synth = df_sintetico['DO_longitude'] - df_sintetico['PU_longitude']
    distance_synth = np.sqrt(lat_diff_synth**2 + lon_diff_synth**2) * 111000  # km aproximado
    
    # Reais
    lat_diff_real = df_real['DO_latitude'] - df_real['PU_latitude']
    lon_diff_real = df_real['DO_longitude'] - df_real['PU_longitude']
    distance_real = np.sqrt(lat_diff_real**2 + lon_diff_real**2) * 111000  # km aproximado
    
    axes[1, 2].hist(distance_synth, bins=50, alpha=0.7, label='Sintético', color='blue', density=True)
    axes[1, 2].hist(distance_real, bins=50, alpha=0.7, label='Real', color='red', density=True)
    axes[1, 2].set_title('Distribuição de Distâncias de Viagem (Amostra 10%)')
    axes[1, 2].set_xlabel('Distância (km)')
    axes[1, 2].set_ylabel('Densidade')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Salvar gráfico
    output_path = Path(output_dir) / 'comparacao_sintetico_vs_real_amostra.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Gráficos comparativos salvos em: {output_path}")

def main():
    """
    Função principal que executa todo o pipeline de geração de heatmaps comparativos.
    """
    print("="*70)
    print("GERADOR DE HEATMAPS COMPARATIVOS - VIAGENS SINTÉTICAS vs REAIS")
    print("USANDO AMOSTRA DE 10% DOS DADOS PARA MELHOR PERFORMANCE")
    print("="*70)
    
    # Configurações
    caminho_dados_sinteticos = "/home-ext/caioloss/Dados/viagens_synth_vae_od.parquet"
    caminho_dados_reais = "/home-ext/caioloss/Dados/viagens_lat_long.parquet"
    output_dir = "/home/caioloss/arquivos/d_linear_modulos/save_data"
    fracao_amostra = 0.1  # 10% dos dados
    seed_aleatoria = 42   # Para reprodutibilidade
    
    # Criar diretório de saída se não existir
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        # 1. Carregar dados (apenas 10% de cada)
        df_sintetico = carregar_dados_sinteticos(caminho_dados_sinteticos, fracao_amostra, seed_aleatoria)
        df_real = carregar_dados_reais(caminho_dados_reais, fracao_amostra, seed_aleatoria)
        
        # 2. Analisar dados
        analisar_dados_comparativo(df_sintetico, df_real)
        
        # 3. Gerar heatmaps para dados sintéticos
        print("\n" + "="*50)
        print("GERANDO HEATMAPS - DADOS SINTÉTICOS (AMOSTRA 10%)")
        print("="*50)
        
        # Heatmap de embarque sintético
        output_embarque_synth = Path(output_dir) / "heatmap_embarque_sintetico_amostra.html"
        criar_heatmap_embarque(df_sintetico, str(output_embarque_synth), "Embarque - Dados Sintéticos")
        
        # Heatmap de desembarque sintético
        output_desembarque_synth = Path(output_dir) / "heatmap_desembarque_sintetico_amostra.html"
        criar_heatmap_desembarque(df_sintetico, str(output_desembarque_synth), "Desembarque - Dados Sintéticos")
        
        # Heatmap combinado sintético
        output_combinado_synth = Path(output_dir) / "heatmap_combinado_sintetico_amostra.html"
        criar_heatmap_combinado(df_sintetico, str(output_combinado_synth), "Combinado - Dados Sintéticos")
        
        # 4. Gerar heatmaps para dados reais
        print("\n" + "="*50)
        print("GERANDO HEATMAPS - DADOS REAIS (AMOSTRA 10%)")
        print("="*50)
        
        # Heatmap de embarque real
        output_embarque_real = Path(output_dir) / "heatmap_embarque_real_amostra.html"
        criar_heatmap_embarque(df_real, str(output_embarque_real), "Embarque - Dados Reais")
        
        # Heatmap de desembarque real
        output_desembarque_real = Path(output_dir) / "heatmap_desembarque_real_amostra.html"
        criar_heatmap_desembarque(df_real, str(output_desembarque_real), "Desembarque - Dados Reais")
        
        # Heatmap combinado real
        output_combinado_real = Path(output_dir) / "heatmap_combinado_real_amostra.html"
        criar_heatmap_combinado(df_real, str(output_combinado_real), "Combinado - Dados Reais")
        
        # 5. Gerar gráficos comparativos
        gerar_graficos_comparativos(df_sintetico, df_real, output_dir)
        
        print("\n" + "="*50)
        print("PROCESSO CONCLUÍDO COM SUCESSO!")
        print("="*50)
        print(f"Arquivos gerados em: {output_dir}")
        print(f"Usando amostra de {fracao_amostra*100}% dos dados (seed={seed_aleatoria})")
        print("\nDADOS SINTÉTICOS:")
        print("- heatmap_embarque_sintetico_amostra.html")
        print("- heatmap_desembarque_sintetico_amostra.html")
        print("- heatmap_combinado_sintetico_amostra.html")
        print("\nDADOS REAIS:")
        print("- heatmap_embarque_real_amostra.html")
        print("- heatmap_desembarque_real_amostra.html")
        print("- heatmap_combinado_real_amostra.html")
        print("\nANÁLISE:")
        print("- comparacao_sintetico_vs_real_amostra.png")
        
    except FileNotFoundError as e:
        print(f"ERRO: Arquivo de dados não encontrado: {e}")
        print("Verifique se os arquivos existem e os caminhos estão corretos.")
    except Exception as e:
        print(f"ERRO durante a execução: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
