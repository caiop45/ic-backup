import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import config # Importa as configurações

def plot_results(df_resultados, save_dir=config.SAVE_DIR):
    """Gera boxplots dos resultados das métricas."""
    print(f"Gerando gráficos em {save_dir}...")
    sns.set_style("whitegrid")
    # Use uma fonte compatível ou a padrão
    plt.rcParams['font.family'] = 'sans-serif'
    # plt.style.use('seaborn-v0_8-white') # Removido se causar warning/erro

    metricas = ['R²', 'MAE', 'RMSE']
    ncs_unicos = sorted(df_resultados['nc'].unique())

    for metrica in metricas:
        df_metrica = df_resultados[df_resultados['metrica'] == metrica]

        for nc_atual in ncs_unicos:
            df_nc = df_metrica[df_metrica['nc'] == nc_atual]
            lista_epochs = sorted(df_nc['epochs'].unique())

            if not lista_epochs:
                print(f"Sem dados para métrica '{metrica}' e nc={nc_atual}. Pulando gráfico.")
                continue

            num_param = len(lista_epochs)
            cols = 3
            rows = (num_param + cols - 1) // cols

            fig, axs = plt.subplots(rows, cols, figsize=(cols * 6.5, rows * 5.5), squeeze=False) # Garante que axs seja sempre 2D
            axs = axs.flatten() # Transforma em 1D para fácil iteração

            for i, ep in enumerate(lista_epochs):
                ax = axs[i]
                df_sub = df_nc[df_nc['epochs'] == ep]

                if df_sub.empty:
                    ax.text(0.5, 0.5, 'Sem dados', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
                    ax.set_title(f'nc={nc_atual}, epochs={ep}', fontsize=12, fontweight='bold')
                    ax.set_xlabel('')
                    ax.set_ylabel(metrica, fontsize=11)
                    continue

                sns.boxplot(
                    x='tipo_dado',
                    y='valor',
                    data=df_sub,
                    ax=ax,
                    order=['real', 'synthetic', 'real+synthetic'],
                    palette='viridis' # Exemplo de paleta
                )

                ax.set_title(f'nc={nc_atual}, epochs={ep}', fontsize=12, fontweight='bold')
                ax.set_xlabel('Tipo de Dado', fontsize=11) # Adiciona label X
                ax.set_ylabel(metrica, fontsize=11)
                ax.tick_params(axis='x', rotation=15) # Rotaciona labels se necessário

                try:
                    ymin, ymax = ax.get_ylim()
                    yrange = ymax - ymin

                    # Adiciona estatísticas (média e mediana) se houver dados
                    for tipo_idx, tipo in enumerate(['real', 'synthetic', 'real+synthetic']):
                        dados_tipo = df_sub[df_sub['tipo_dado'] == tipo]['valor']
                        if not dados_tipo.empty:
                            mediana = dados_tipo.median()
                            media = dados_tipo.mean()

                            # Ajusta posição do texto para evitar sobreposição
                            ax.text(
                                tipo_idx, ymin - yrange * 0.05, # Posição ligeiramente abaixo
                                f'Med: {mediana:.2f}',
                                ha='center', va='top', # Ajusta alinhamento vertical
                                color='black',
                                fontsize=10, fontweight='bold', # Tamanho menor
                                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2')
                            )

                            ax.text(
                                tipo_idx, ymax + yrange * 0.05, # Posição ligeiramente acima
                                f'Média: {media:.2f}',
                                ha='center', va='bottom', # Ajusta alinhamento vertical
                                color='darkred',
                                fontsize=10, fontweight='bold', # Tamanho menor
                                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2')
                            )
                except Exception as e:
                    print(f"Erro ao adicionar texto de estatísticas para nc={nc_atual}, epochs={ep}, tipo={tipo}: {e}")


            # Desativa eixos extras
            for j in range(i + 1, len(axs)):
                 axs[j].axis('off')


            plt.suptitle(
                f'Variação da métrica {metrica} - nc={nc_atual}',
                fontsize=16,
                fontweight='bold'
                # y=1.02 # Ajuste a posição Y se necessário
            )
            plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Ajusta layout para caber o supertítulo

            filename = f"nc_{nc_atual}_boxplot_{metrica}.png"
            filepath = os.path.join(save_dir, filename)
            try:
                plt.savefig(filepath, bbox_inches='tight')
                print(f"Gráfico salvo: {filepath}")
            except Exception as e:
                print(f"Erro ao salvar gráfico {filepath}: {e}")
            plt.close(fig) # Fecha a figura para liberar memória
    print("Geração de gráficos concluída.")