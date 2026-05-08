
import os
import sys

# ═══ Compatibilidade Colab: garantir que os módulos locais são encontrados ═══
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
os.chdir(_SCRIPT_DIR)
# ═══════════════════════════════════════════════════════════════════════════════

import passo9_advanced_config as config
from passo9_advanced_processor import AdvancedAnalyzer
from passo9_advanced_visualizer import AdvancedVisualizer

def run_advanced_analysis_pipeline():
    """
    Executa o pipeline completo de Análises Avançadas:
    1. Carrega os melhores modelos treinados
    2. Executa Análise de Sensibilidade (What-If) nas features chave
    3. Executa Teste de Robustez adicionando ruído aos dados
    4. Salva os resultados em CSV
    5. Gera visualizações avançadas
    """
    print("="*50)
    print("INICIANDO PIPELINE DE ANÁLISES AVANÇADAS (PASSO 9)")
    print("="*50)
    
    # 1. Processamento: Analisar sensibilidade e robustez
    print("\n[1/2] Executando Análises de Sensibilidade e Robustez...")
    analyzer = AdvancedAnalyzer()
    sens_res, rob_res = analyzer.run_all_analyses()
    
    # 2. Visualização: Gerar gráficos avançados
    print("\n[2/2] Gerando Visualizações Avançadas...")
    visualizer = AdvancedVisualizer(sens_res, rob_res)
    visualizer.generate_all_visualizations()
    
    print("\n" + "="*50)
    print("PIPELINE CONCLUÍDO COM SUCESSO!")
    print(f"Análises salvas em: {config.OUTPUT_DIR}")
    print(f"Visualizações salvas em: {os.path.join(config.OUTPUT_DIR, 'visualizacoes_avancadas')}")
    print("="*50)

if __name__ == "__main__":
    run_advanced_analysis_pipeline()
