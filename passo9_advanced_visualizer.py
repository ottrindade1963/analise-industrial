import matplotlib
matplotlib.use("Agg")  # Backend não-interativo para Colab
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import passo9_advanced_config as config

class AdvancedVisualizer:
    def __init__(self, sensitivity_results, robustness_results):
        self.df_sens = pd.DataFrame(sensitivity_results) if sensitivity_results else pd.DataFrame()
        self.df_rob = pd.DataFrame(robustness_results) if robustness_results else pd.DataFrame()
        
        self.output_dir = os.path.join(config.OUTPUT_DIR, 'visualizacoes_avancadas')
        os.makedirs(self.output_dir, exist_ok=True)
        
        sns.set_theme(style="whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 12

    def plot_sensitivity(self):
        """Gera gráfico de análise de sensibilidade (What-If)."""
        if self.df_sens.empty:
            return
            
        print("Gerando gráfico de Análise de Sensibilidade...")
        
        plt.figure(figsize=(12, 7))
        
        # Converter string de variação para numérico para ordenar corretamente no eixo X
        self.df_sens['Variacao_Num'] = self.df_sens['Variacao_Aplicada'].str.replace('%', '').astype(float)
        self.df_sens = self.df_sens.sort_values('Variacao_Num')
        
        # Plotar linhas para cada variável e modelo
        sns.lineplot(
            data=self.df_sens, 
            x='Variacao_Aplicada', 
            y='Impacto_Previsao_%', 
            hue='Variavel', 
            style='Modelo',
            markers=True, 
            dashes=False, 
            linewidth=2.5,
            markersize=10
        )
        
        plt.axhline(0, color='black', linewidth=1.5, linestyle='--')
        plt.axvline(2, color='gray', linewidth=1, linestyle=':') # Posição do 0% (se existisse)
        
        plt.title('Análise de Sensibilidade (What-If)\nImpacto de Variações nas Features na Previsão Final', fontsize=16)
        plt.ylabel('Impacto na Previsão Média (%)')
        plt.xlabel('Variação Aplicada na Feature')
        
        plt.legend(title='Feature / Modelo', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        filename = 'analise_sensibilidade_what_if.png'
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()

    def plot_robustness(self):
        """Gera gráfico de degradação do modelo com ruído."""
        if self.df_rob.empty:
            return
            
        print("Gerando gráfico de Teste de Robustez...")
        
        plt.figure(figsize=(10, 6))
        
        # Converter string de ruído para numérico
        self.df_rob['Ruido_Num'] = self.df_rob['Nivel_Ruido'].str.replace('%', '').astype(float)
        self.df_rob = self.df_rob.sort_values('Ruido_Num')
        
        sns.barplot(
            data=self.df_rob, 
            x='Nivel_Ruido', 
            y='Degradacao_MAE', 
            hue='Modelo',
            palette='viridis'
        )
        
        plt.title('Teste de Robustez: Degradação da Previsão com Adição de Ruído', fontsize=16)
        plt.ylabel('Degradação (Erro Absoluto Médio vs Previsão Base)')
        plt.xlabel('Nível de Ruído Adicionado aos Dados')
        
        plt.tight_layout()
        
        filename = 'teste_robustez_ruido.png'
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300)
        plt.close()
        
    def plot_sensitivity_heatmap(self):
        """Gera um heatmap de sensibilidade por variável e variação."""
        if self.df_sens.empty:
            return
            
        print("Gerando heatmap de sensibilidade...")
        
        # Pivotar a tabela
        pivot_sens = self.df_sens.pivot_table(
            index='Variavel', 
            columns='Variacao_Aplicada', 
            values='Impacto_Previsao_%', 
            aggfunc='mean'
        )
        
        # Ordenar colunas numericamente
        cols = pivot_sens.columns.tolist()
        cols_sorted = sorted(cols, key=lambda x: float(x.replace('%', '')))
        pivot_sens = pivot_sens[cols_sorted]
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(pivot_sens, annot=True, cmap='coolwarm', center=0, fmt='.2f', linewidths=.5)
        
        plt.title('Heatmap de Sensibilidade: Impacto Médio na Previsão (%)', fontsize=16)
        plt.ylabel('Variável Alterada')
        plt.xlabel('Variação Aplicada')
        
        plt.tight_layout()
        filename = 'heatmap_sensibilidade.png'
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300)
        plt.close()

    def generate_all_visualizations(self):
        """Gera todas as visualizações avançadas."""
        if not self.df_sens.empty or not self.df_rob.empty:
            self.plot_sensitivity()
            self.plot_robustness()
            self.plot_sensitivity_heatmap()
            print(f"Todas as visualizações avançadas salvas em: {self.output_dir}")
        else:
            print("Nenhum dado avançado disponível para visualização.")

if __name__ == "__main__":
    import passo9_advanced_processor as processor
    analyzer = processor.AdvancedAnalyzer()
    sens_res, rob_res = analyzer.run_all_analyses()
    viz = AdvancedVisualizer(sens_res, rob_res)
    viz.generate_all_visualizations()
