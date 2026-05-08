import matplotlib
matplotlib.use("Agg")  # Backend não-interativo para Colab
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import passo6_strategy_config as config

class StrategyVisualizer:
    def __init__(self, analysis_df_datasets, analysis_df_strategies):
        self.analysis_df_datasets = analysis_df_datasets
        self.analysis_df_strategies = analysis_df_strategies
        self.output_dir = os.path.join(config.OUTPUT_DIR, 'visualizacoes_estrategias')
        os.makedirs(self.output_dir, exist_ok=True)
        
        sns.set_theme(style="whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 12

    def plot_gain_vs_non_aggregated(self):
        """Gera gráfico de barras mostrando o ganho de usar dados agregados vs não agregados."""
        if self.analysis_df_datasets is None:
            return
            
        print("Gerando gráfico de ganho: Agregados vs Não Agregados...")
        
        # Agrupar por Modelo e Dataset (média das estratégias)
        avg_gain = self.analysis_df_datasets.groupby(['Modelo', 'Dataset'])['Melhoria_RMSE_%'].mean().reset_index()
        
        plt.figure(figsize=(12, 7))
        
        palette = sns.color_palette("Set2", n_colors=len(avg_gain['Dataset'].unique()))
        
        ax = sns.barplot(x='Modelo', y='Melhoria_RMSE_%', hue='Dataset', data=avg_gain, palette=palette)
        
        plt.axhline(0, color='black', linewidth=1.5, linestyle='--')
        
        plt.title('Melhoria Percentual do RMSE ao usar Dados Agregados (vs Não Agregados)', fontsize=16)
        plt.ylabel('Melhoria no RMSE (%) - Valores Positivos são Melhores')
        plt.xlabel('Modelo Preditivo')
        
        for p in ax.patches:
            height = p.get_height()
            if not pd.isna(height):
                ax.annotate(f'{height:.1f}%', 
                            (p.get_x() + p.get_width() / 2., height), 
                            ha='center', va='bottom' if height > 0 else 'top', 
                            xytext=(0, 5 if height > 0 else -15), 
                            textcoords='offset points')
                            
        plt.tight_layout()
        filename = 'ganho_agregados_vs_nao_agregados.png'
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300)
        plt.close()

    def plot_gain_between_strategies(self):
        """Gera gráfico de barras mostrando o ganho de A2 e A3 em relação a A1."""
        if self.analysis_df_strategies is None:
            return
            
        print("Gerando gráfico de ganho entre estratégias (A2/A3 vs A1)...")
        
        # Filtrar apenas A2 e A3 (A1 tem ganho 0% por definição)
        df_plot = self.analysis_df_strategies[self.analysis_df_strategies['Estrategia'] != config.BASELINE_STRATEGY].copy()
        
        # Agrupar por Estratégia e Modelo (média dos datasets)
        avg_gain = df_plot.groupby(['Estrategia', 'Modelo'])['Melhoria_RMSE_vs_A1_%'].mean().reset_index()
        
        plt.figure(figsize=(12, 7))
        
        palette = sns.color_palette("coolwarm", n_colors=len(avg_gain['Estrategia'].unique()))
        
        ax = sns.barplot(x='Modelo', y='Melhoria_RMSE_vs_A1_%', hue='Estrategia', data=avg_gain, palette=palette)
        
        plt.axhline(0, color='black', linewidth=1.5, linestyle='--')
        
        plt.title('Melhoria Percentual do RMSE em Relação à Estratégia A1 (Inclusão Direta)', fontsize=16)
        plt.ylabel('Melhoria no RMSE (%) - Valores Positivos são Melhores')
        plt.xlabel('Modelo Preditivo')
        
        for p in ax.patches:
            height = p.get_height()
            if not pd.isna(height):
                ax.annotate(f'{height:.1f}%', 
                            (p.get_x() + p.get_width() / 2., height), 
                            ha='center', va='bottom' if height > 0 else 'top', 
                            xytext=(0, 5 if height > 0 else -15), 
                            textcoords='offset points')
                            
        plt.tight_layout()
        filename = 'ganho_percentual_estrategias_a2_a3.png'
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300)
        plt.close()
        
    def plot_strategy_radar(self):
        """Gera um gráfico de radar comparando as estratégias em múltiplas métricas."""
        if self.analysis_df_strategies is None:
            return
            
        print("Gerando gráfico de radar das estratégias...")
        
        # Carregar métricas completas do Passo 5
        eval_file = os.path.join(config.BASE_DIR, 'resultados_avaliacao', 'metricas_avaliacao_expandidas.csv')
        if not os.path.exists(eval_file):
            return
            
        df_eval = pd.read_csv(eval_file)
        
        # Agrupar por estratégia e normalizar métricas para o radar (0 a 1)
        # Para RMSE, MAE, MAPE: menor é melhor, então invertemos
        # Para R2: maior é melhor
        
        metrics = ['RMSE', 'MAE', 'MAPE', 'R2']
        avg_metrics = df_eval.groupby('Estrategia')[metrics].mean().reset_index()
        
        # Normalização Min-Max
        for m in metrics:
            min_val = avg_metrics[m].min()
            max_val = avg_metrics[m].max()
            if max_val > min_val:
                if m in ['RMSE', 'MAE', 'MAPE']:
                    # Inverter: 1 é o melhor (menor erro), 0 é o pior (maior erro)
                    avg_metrics[f'{m}_norm'] = 1 - ((avg_metrics[m] - min_val) / (max_val - min_val))
                else:
                    # R2: 1 é o melhor (maior R2), 0 é o pior (menor R2)
                    avg_metrics[f'{m}_norm'] = (avg_metrics[m] - min_val) / (max_val - min_val)
            else:
                avg_metrics[f'{m}_norm'] = 0.5
                
        # Preparar dados para o radar
        import numpy as np
        from math import pi
        
        categories = metrics
        N = len(categories)
        
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]
        
        plt.figure(figsize=(10, 10))
        ax = plt.subplot(111, polar=True)
        
        ax.set_theta_offset(pi / 2)
        ax.set_theta_direction(-1)
        
        plt.xticks(angles[:-1], categories)
        ax.set_rlabel_position(0)
        plt.yticks([0.25, 0.5, 0.75, 1.0], ["0.25", "0.50", "0.75", "1.00"], color="grey", size=10)
        plt.ylim(0, 1.1)
        
        colors = ['b', 'r', 'g', 'm']
        for i, row in avg_metrics.iterrows():
            values = row[[f'{m}_norm' for m in metrics]].values.flatten().tolist()
            values += values[:1]
            
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=row['Estrategia'], color=colors[i%len(colors)])
            ax.fill(angles, values, colors[i%len(colors)], alpha=0.1)
            
        plt.title('Comparação Multidimensional das Estratégias (Normalizado, Maior é Melhor)', size=16, y=1.1)
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'radar_estrategias.png'), dpi=300)
        plt.close()

    def generate_all_visualizations(self):
        """Gera todas as visualizações de análise de estratégias."""
        if self.analysis_df_datasets is not None and not self.analysis_df_datasets.empty:
            self.plot_gain_vs_non_aggregated()
            self.plot_gain_between_strategies()
            self.plot_strategy_radar()
            print(f"Todas as visualizações salvas em: {self.output_dir}")
        else:
            print("Nenhum dado de análise disponível para visualização.")

if __name__ == "__main__":
    # Teste simples
    import passo6_strategy_processor as processor
    analyzer = processor.StrategyAnalyzer()
    df_datasets, df_strategies = analyzer.run_analysis()
    viz = StrategyVisualizer(df_datasets, df_strategies)
    viz.generate_all_visualizations()
