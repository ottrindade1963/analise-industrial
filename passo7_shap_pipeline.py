
import os
import sys

# ═══ Compatibilidade Colab: garantir que os módulos locais são encontrados ═══
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
os.chdir(_SCRIPT_DIR)
# ═══════════════════════════════════════════════════════════════════════════════

import passo7_shap_config as config
from passo7_shap_processor import ShapAnalyzer
from passo7_shap_visualizer import ShapVisualizer

def run_shap_analysis_pipeline():
    """
    Executa o pipeline completo de Interpretabilidade (Explainable AI):
    1. Carrega os modelos treinados (RF, XGBoost)
    2. Carrega os dados de teste
    3. Calcula os valores SHAP para cada modelo
    4. Salva a importância média absoluta em CSV
    5. Gera visualizações SHAP (Summary, Bar, Dependence)
    """
    print("="*50)
    print("INICIANDO PIPELINE DE INTERPRETABILIDADE SHAP (PASSO 7)")
    print("="*50)
    
    # 1. Processamento: Calcular SHAP
    print("\n[1/2] Calculando Valores SHAP...")
    analyzer = ShapAnalyzer()
    shap_vals, X_test, feat_names = analyzer.run_analysis()
    
    # 2. Visualização: Gerar gráficos SHAP
    print("\n[2/2] Gerando Visualizações SHAP...")
    visualizer = ShapVisualizer(shap_vals, X_test, feat_names)
    visualizer.generate_all_visualizations()
    
    print("\n" + "="*50)
    print("PIPELINE CONCLUÍDO COM SUCESSO!")
    print(f"Dados SHAP salvos em: {config.OUTPUT_DIR}")
    print(f"Visualizações salvas em: {os.path.join(config.OUTPUT_DIR, 'visualizacoes_shap')}")
    print("="*50)

if __name__ == "__main__":
    run_shap_analysis_pipeline()
