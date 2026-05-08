import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Backend não-interativo para Colab
import matplotlib.pyplot as plt
import seaborn as sns
import os
import passo5_eval_config as config

class EvaluationVisualizer:
    def __init__(self, results_df):
        self.results_df = results_df
        self.output_dir = os.path.join(config.OUTPUT_DIR, 'visualizacoes_avaliacao')
        os.makedirs(self.output_dir, exist_ok=True)
        
        sns.set_theme(style="whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 12

    def plot_rmse_comparison(self):
        """Gera gráfico de barras comparando o RMSE dos modelos por estratégia."""
        print("Gerando gráfico de comparação de RMSE...")
        
        plt.figure(figsize=(14, 8))
        
        # Agrupar por Modelo e Estratégia (média dos datasets)
        avg_rmse = self.results_df.groupby(['Modelo', 'Estrategia'])['RMSE'].mean().reset_index()
        
        sns.barplot(x='Modelo', y='RMSE', hue='Estrategia', data=avg_rmse, palette='viridis')
        
        plt.title('Comparação de RMSE por Modelo e Estratégia de Agregação (Média dos Datasets)', fontsize=16)
        plt.ylabel('RMSE (Menor é Melhor)')
        plt.xlabel('Modelo Preditivo')
        plt.legend(title='Estratégia', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        filename = 'comparacao_rmse_modelos_estrategias.png'
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300)
        plt.close()

    def plot_performance_heatmap(self):
        """Gera um mapa de calor com as métricas de performance."""
        print("Gerando heatmap de performance...")
        
        # Pivotar a tabela para ter Modelos nas linhas e Estratégias nas colunas
        pivot_rmse = self.results_df.pivot_table(
            index='Modelo', 
            columns='Estrategia', 
            values='RMSE', 
            aggfunc='mean'
        )
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(pivot_rmse, annot=True, cmap='YlGnBu_r', fmt='.3f', linewidths=.5)
        
        plt.title('Heatmap de RMSE (Média dos Datasets)', fontsize=16)
        plt.ylabel('Modelo')
        plt.xlabel('Estratégia de Agregação')
        
        plt.tight_layout()
        filename = 'heatmap_rmse_performance.png'
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300)
        plt.close()

    def plot_dataset_comparison(self):
        """Compara a performance entre os diferentes datasets (Inner, Left, Outer, Nao Agregado)."""
        print("Gerando gráfico de comparação de datasets...")
        
        plt.figure(figsize=(12, 6))
        
        # Agrupar por Dataset e Modelo
        avg_rmse = self.results_df.groupby(['Dataset', 'Modelo'])['RMSE'].mean().reset_index()
        
        sns.barplot(x='Dataset', y='RMSE', hue='Modelo', data=avg_rmse, palette='Set2')
        
        plt.title('Comparação de RMSE por Dataset e Modelo (Média das Estratégias)', fontsize=16)
        plt.ylabel('RMSE (Menor é Melhor)')
        plt.xlabel('Dataset Agregado')
        
        plt.tight_layout()
        filename = 'comparacao_rmse_datasets.png'
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300)
        plt.close()
        
    def plot_mae_mape_comparison(self):
        """Gera gráficos para MAE e MAPE."""
        print("Gerando gráficos de MAE e MAPE...")
        
        # MAE
        plt.figure(figsize=(14, 6))
        avg_mae = self.results_df.groupby(['Modelo', 'Dataset'])['MAE'].mean().reset_index()
        sns.barplot(x='Modelo', y='MAE', hue='Dataset', data=avg_mae, palette='magma')
        plt.title('Comparação de MAE por Modelo e Dataset', fontsize=16)
        plt.ylabel('MAE (Menor é Melhor)')
        plt.legend(title='Dataset', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'comparacao_mae.png'), dpi=300)
        plt.close()
        
        # MAPE
        plt.figure(figsize=(14, 6))
        avg_mape = self.results_df.groupby(['Modelo', 'Dataset'])['MAPE'].mean().reset_index()
        sns.barplot(x='Modelo', y='MAPE', hue='Dataset', data=avg_mape, palette='plasma')
        plt.title('Comparação de MAPE (%) por Modelo e Dataset', fontsize=16)
        plt.ylabel('MAPE (%)')
        plt.legend(title='Dataset', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'comparacao_mape.png'), dpi=300)
        plt.close()
        
    def plot_r2_comparison(self):
        """Gera gráfico para R2."""
        print("Gerando gráfico de R2...")
        
        plt.figure(figsize=(14, 6))
        avg_r2 = self.results_df.groupby(['Modelo', 'Estrategia'])['R2'].mean().reset_index()
        sns.barplot(x='Modelo', y='R2', hue='Estrategia', data=avg_r2, palette='viridis')
        plt.title('Comparação de R² por Modelo e Estratégia', fontsize=16)
        plt.ylabel('R² (Maior é Melhor)')
        plt.ylim(0, 1.0)
        plt.legend(title='Estratégia', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'comparacao_r2.png'), dpi=300)
        plt.close()

    def generate_all_visualizations(self):
        """Gera todas as visualizações de avaliação."""
        if self.results_df is not None and not self.results_df.empty:
            self.plot_rmse_comparison()
            self.plot_performance_heatmap()
            self.plot_dataset_comparison()
            self.plot_mae_mape_comparison()
            self.plot_r2_comparison()
            print(f"Todas as 6 visualizações salvas em: {self.output_dir}")
        else:
            print("Nenhum dado de resultado disponível para visualização.")

if __name__ == "__main__":
    # Teste simples
    import passo5_eval_processor as processor
    evaluator = processor.ModelEvaluator()
    results_df = evaluator.run_evaluation()
    viz = EvaluationVisualizer(results_df)
    viz.generate_all_visualizations()
