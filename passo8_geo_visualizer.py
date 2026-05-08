import matplotlib
matplotlib.use("Agg")  # Backend não-interativo para Colab
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import passo8_geo_config as config

class GeoVisualizer:
    def __init__(self, predictions_dict):
        self.predictions_dict = predictions_dict
        self.output_dir = os.path.join(config.OUTPUT_DIR, 'visualizacoes_geograficas')
        os.makedirs(self.output_dir, exist_ok=True)
        
        sns.set_theme(style="whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 12

    def plot_error_by_class(self, key):
        """Gera um boxplot comparando o erro absoluto entre as classes económicas."""
        print(f"Gerando Boxplot de Erro por Classe para {key}...")
        
        if key not in self.predictions_dict:
            return
            
        df = self.predictions_dict[key]['raw']
        
        plt.figure(figsize=(10, 6))
        
        # Ordem das classes
        order = ['Pobre', 'Médio', 'Rico']
        
        sns.boxplot(x='Classe_Economica', y='Erro_Absoluto', data=df, order=order, palette='Set2')
        sns.stripplot(x='Classe_Economica', y='Erro_Absoluto', data=df, order=order, color='black', alpha=0.3, jitter=True)
        
        plt.title(f'Distribuição do Erro Absoluto por Classe Económica - {key}', fontsize=16)
        plt.ylabel('Erro Absoluto (Valor Agregado Industrial % PIB)')
        plt.xlabel('Classe Económica (Baseada no PIB per capita)')
        
        plt.tight_layout()
        filename = f'boxplot_erro_classe_{key}.png'
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300)
        plt.close()

    def plot_scatter_pib_error(self, key):
        """Gera um scatter plot relacionando PIB per capita e Erro Absoluto."""
        print(f"Gerando Scatter Plot PIB vs Erro para {key}...")
        
        if key not in self.predictions_dict:
            return
            
        df = self.predictions_dict[key]['aggregated']
        
        # Juntar com o PIB médio
        raw_df = self.predictions_dict[key]['raw']
        pib_mean = raw_df.groupby('country')['pib_per_capita'].mean().reset_index()
        
        df_plot = pd.merge(df, pib_mean, on='country')
        
        plt.figure(figsize=(10, 6))
        
        # Cores por classe
        palette = {'Pobre': 'red', 'Médio': 'orange', 'Rico': 'green'}
        
        sns.scatterplot(x='pib_per_capita', y='Erro_Absoluto', hue='Classe_Economica', data=df_plot, palette=palette, s=100, alpha=0.7)
        
        # Adicionar nomes de alguns países (top 5 maiores erros)
        top_errors = df_plot.nlargest(5, 'Erro_Absoluto')
        for i, row in top_errors.iterrows():
            plt.annotate(row['country'], (row['pib_per_capita'], row['Erro_Absoluto']), 
                         xytext=(5, 5), textcoords='offset points', fontsize=9)
                         
        plt.title(f'Relação entre PIB per capita e Erro de Previsão - {key}', fontsize=16)
        plt.ylabel('Erro Absoluto Médio')
        plt.xlabel('PIB per capita (Média)')
        plt.xscale('log') # Escala logarítmica para o PIB
        
        plt.tight_layout()
        filename = f'scatter_pib_erro_{key}.png'
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300)
        plt.close()

    def generate_all_visualizations(self):
        """Gera todas as visualizações geográficas."""
        if not self.predictions_dict:
            print("Nenhum dado geográfico disponível para visualização.")
            return
            
        for key in self.predictions_dict.keys():
            self.plot_error_by_class(key)
            self.plot_scatter_pib_error(key)
            
        print(f"Todas as visualizações geográficas salvas em: {self.output_dir}")

if __name__ == "__main__":
    # Teste simples
    import passo8_geo_processor as processor
    analyzer = processor.GeoAnalyzer()
    preds_dict = analyzer.run_analysis()
    viz = GeoVisualizer(preds_dict)
    viz.generate_all_visualizations()
