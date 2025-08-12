#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script auxiliar para visualizar os heatmaps comparativos gerados.
Abre os arquivos HTML no navegador padrão do sistema.
"""

import webbrowser
import os
from pathlib import Path

def abrir_heatmaps_comparativos():
    """
    Abre os heatmaps comparativos gerados no navegador padrão.
    """
    # Diretório onde os heatmaps foram salvos
    save_dir = Path("/home/caioloss/arquivos/d_linear_modulos/save_data")
    
    # Lista de arquivos HTML para abrir (amostra de 10%)
    heatmap_files = [
        "heatmap_embarque_sintetico_amostra.html",
        "heatmap_desembarque_sintetico_amostra.html", 
        "heatmap_combinado_sintetico_amostra.html",
        "heatmap_embarque_real_amostra.html",
        "heatmap_desembarque_real_amostra.html",
        "heatmap_combinado_real_amostra.html"
    ]
    
    print("="*70)
    print("ABRINDO HEATMAPS COMPARATIVOS NO NAVEGADOR")
    print("USANDO AMOSTRA DE 10% DOS DADOS")
    print("="*70)
    
    for filename in heatmap_files:
        file_path = save_dir / filename
        if file_path.exists():
            # Converter para URL file://
            file_url = f"file://{file_path.absolute()}"
            print(f"Abrindo: {filename}")
            print(f"URL: {file_url}")
            webbrowser.open(file_url)
        else:
            print(f"ERRO: Arquivo não encontrado: {filename}")
    
    print("\n" + "="*70)
    print("INSTRUÇÕES DE USO - HEATMAPS COMPARATIVOS")
    print("="*70)
    print("DADOS SINTÉTICOS (gerados pelo VAE):")
    print("1. heatmap_embarque_sintetico_amostra.html - Apenas pontos de embarque")
    print("2. heatmap_desembarque_sintetico_amostra.html - Apenas pontos de desembarque")
    print("3. heatmap_combinado_sintetico_amostra.html - Ambos com controle de camadas")
    print()
    print("DADOS REAIS (originais):")
    print("4. heatmap_embarque_real_amostra.html - Apenas pontos de embarque")
    print("5. heatmap_desembarque_real_amostra.html - Apenas pontos de desembarque")
    print("6. heatmap_combinado_real_amostra.html - Ambos com controle de camadas")
    print()
    print("CARACTERÍSTICAS DOS HEATMAPS:")
    print("- Amostra: 10% dos dados originais (aleatória, seed=42)")
    print("- Incluem outliers para análise completa")
    print("- Zoom inicial menor para visualizar toda a área")
    print("- Controle de camadas para alternar entre mapas claro/escuro")
    print()
    print("CONTROLES DO MAPA:")
    print("- Zoom: scroll do mouse ou botões +/-")
    print("- Pan: arrastar o mapa")
    print("- Controle de camadas: ícone no canto superior direito")
    print()
    print("COMPARAÇÃO SINTÉTICO vs REAL:")
    print("- Sintéticos: ~109k viagens (10% de 1.1M)")
    print("- Reais: ~955k viagens (10% de 9.6M)")
    print("- Ambos incluem outliers para análise completa")
    print("="*70)

if __name__ == "__main__":
    abrir_heatmaps_comparativos()
