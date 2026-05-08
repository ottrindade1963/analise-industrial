import pandas as pd
import numpy as np
import os
import passo6_strategy_config as config

class StrategyAnalyzer:
    def __init__(self):
        self.results_filepath = os.path.join(config.RESULTS_DIR, 'metricas_avaliacao_expandidas.csv')
        self.df = None
        self.analysis_results_strategies = None
        self.analysis_results_datasets = None

    def load_results(self):
        """Carrega os resultados da avaliação do Passo 5."""
        if os.path.exists(self.results_filepath):
            self.df = pd.read_csv(self.results_filepath)
            print(f"Resultados carregados: {len(self.df)} registos.")
            return True
        else:
            print(f"ERRO: Ficheiro de resultados não encontrado: {self.results_filepath}")
            return False

    def calculate_gains_vs_non_aggregated(self):
        """Calcula o ganho de usar dados agregados (qualitativos) vs não agregados."""
        if self.df is None:
            return None
            
        print("Calculando ganhos: Agregados vs Não Agregados...")
        
        # Separar o baseline (Não Agregado)
        baseline_df = self.df[self.df['Dataset'] == config.BASELINE_DATASET].copy()
        
        # Como o não agregado tem as mesmas métricas para todas as estratégias (são idênticas),
        # pegamos apenas uma (ex: A1_Direta) para servir de baseline único por modelo
        baseline_df = baseline_df[baseline_df['Estrategia'] == 'A1_Direta']
        
        baseline_df = baseline_df.rename(columns={
            'RMSE': 'RMSE_NaoAgregado', 
            'MAE': 'MAE_NaoAgregado', 
            'MAPE': 'MAPE_NaoAgregado',
            'R2': 'R2_NaoAgregado'
        })
        baseline_df = baseline_df[['Modelo', 'RMSE_NaoAgregado', 'MAE_NaoAgregado', 'MAPE_NaoAgregado', 'R2_NaoAgregado']]
        
        # Juntar com os outros resultados (apenas os agregados)
        agregados_df = self.df[self.df['Dataset'] != config.BASELINE_DATASET].copy()
        comparison_df = pd.merge(agregados_df, baseline_df, on=['Modelo'], how='left')
        
        # Calcular ganho percentual (negativo é melhor para RMSE/MAE/MAPE, positivo é melhor para R2)
        comparison_df['Melhoria_RMSE_%'] = -((comparison_df['RMSE'] - comparison_df['RMSE_NaoAgregado']) / comparison_df['RMSE_NaoAgregado']) * 100
        comparison_df['Melhoria_MAE_%'] = -((comparison_df['MAE'] - comparison_df['MAE_NaoAgregado']) / comparison_df['MAE_NaoAgregado']) * 100
        comparison_df['Melhoria_MAPE_%'] = -((comparison_df['MAPE'] - comparison_df['MAPE_NaoAgregado']) / comparison_df['MAPE_NaoAgregado']) * 100
        comparison_df['Melhoria_R2_%'] = ((comparison_df['R2'] - comparison_df['R2_NaoAgregado']) / np.abs(comparison_df['R2_NaoAgregado'])) * 100
        
        self.analysis_results_datasets = comparison_df
        
        out_filepath = os.path.join(config.OUTPUT_DIR, 'analise_ganhos_vs_nao_agregado.csv')
        comparison_df.to_csv(out_filepath, index=False)
        
        return comparison_df

    def calculate_gains_between_strategies(self):
        """Calcula o ganho de A2 e A3 em relação a A1 (apenas para datasets agregados)."""
        if self.df is None:
            return None
            
        print("Calculando ganhos entre estratégias (A2/A3 vs A1)...")
        
        # Filtrar apenas datasets agregados
        agregados_df = self.df[self.df['Dataset'] != config.BASELINE_DATASET].copy()
        
        # Separar o baseline (A1)
        baseline_df = agregados_df[agregados_df['Estrategia'] == config.BASELINE_STRATEGY].copy()
        baseline_df = baseline_df.rename(columns={
            'RMSE': 'RMSE_A1', 
            'MAE': 'MAE_A1',
            'MAPE': 'MAPE_A1',
            'R2': 'R2_A1'
        })
        baseline_df = baseline_df[['Modelo', 'Dataset', 'RMSE_A1', 'MAE_A1', 'MAPE_A1', 'R2_A1']]
        
        # Juntar com os outros resultados
        comparison_df = pd.merge(agregados_df, baseline_df, on=['Modelo', 'Dataset'], how='left')
        
        # Calcular ganho percentual
        comparison_df['Melhoria_RMSE_vs_A1_%'] = -((comparison_df['RMSE'] - comparison_df['RMSE_A1']) / comparison_df['RMSE_A1']) * 100
        comparison_df['Melhoria_MAE_vs_A1_%'] = -((comparison_df['MAE'] - comparison_df['MAE_A1']) / comparison_df['MAE_A1']) * 100
        comparison_df['Melhoria_MAPE_vs_A1_%'] = -((comparison_df['MAPE'] - comparison_df['MAPE_A1']) / comparison_df['MAPE_A1']) * 100
        comparison_df['Melhoria_R2_vs_A1_%'] = ((comparison_df['R2'] - comparison_df['R2_A1']) / np.abs(comparison_df['R2_A1'])) * 100
        
        self.analysis_results_strategies = comparison_df
        
        out_filepath = os.path.join(config.OUTPUT_DIR, 'analise_ganhos_entre_estrategias.csv')
        comparison_df.to_csv(out_filepath, index=False)
        
        return comparison_df

    def run_analysis(self):
        """Executa o pipeline de análise."""
        if self.load_results():
            self.calculate_gains_vs_non_aggregated()
            self.calculate_gains_between_strategies()
            return self.analysis_results_datasets, self.analysis_results_strategies
        return None, None

if __name__ == "__main__":
    analyzer = StrategyAnalyzer()
    analyzer.run_analysis()
