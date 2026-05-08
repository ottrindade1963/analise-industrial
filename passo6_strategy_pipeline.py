
import os
import sys

# ═══ Compatibilidade Colab: garantir que os módulos locais são encontrados ═══
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
os.chdir(_SCRIPT_DIR)
# ═══════════════════════════════════════════════════════════════════════════════

import passo6_strategy_config as config
from passo6_strategy_processor import StrategyAnalyzer
from passo6_strategy_visualizer import StrategyVisualizer

def run_strategy_analysis_pipeline():
    """
    Executa o pipeline completo de Análise de Estratégias de Agregação:
    1. Carrega os resultados da avaliação do Passo 5
    2. Calcula o ganho de usar dados agregados vs não agregados
    3. Calcula o ganho percentual de A2 e A3 em relação a A1 (Baseline)
    4. Salva os resultados da análise em CSV
    5. Gera visualizações comparativas (H2)
    """
    print("="*50)
    print("INICIANDO PIPELINE DE ANÁLISE DE ESTRATÉGIAS (PASSO 6)")
    print("="*50)
    
    # 1. Processamento: Analisar ganhos
    print("\n[1/2] Analisando Ganhos Percentuais das Estratégias...")
    analyzer = StrategyAnalyzer()
    df_datasets, df_strategies = analyzer.run_analysis()
    
    # 2. Visualização: Gerar gráficos de ganho
    print("\n[2/2] Gerando Visualizações de Análise...")
    visualizer = StrategyVisualizer(df_datasets, df_strategies)
    visualizer.generate_all_visualizations()
    
    print("\n" + "="*50)
    print("PIPELINE CONCLUÍDO COM SUCESSO!")
    print(f"Análises salvas em: {config.OUTPUT_DIR}")
    print(f"Visualizações salvas em: {os.path.join(config.OUTPUT_DIR, 'visualizacoes_estrategias')}")
    print("="*50)

if __name__ == "__main__":
    run_strategy_analysis_pipeline()
