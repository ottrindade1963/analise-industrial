
import os
import sys

# ═══ Compatibilidade Colab: garantir que os módulos locais são encontrados ═══
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
os.chdir(_SCRIPT_DIR)
# ═══════════════════════════════════════════════════════════════════════════════

import passo5_eval_config as config
from passo5_eval_processor import ModelEvaluator
from passo5_eval_visualizer import EvaluationVisualizer

def run_evaluation_pipeline():
    """
    Executa o pipeline completo de Avaliação de Performance:
    1. Carrega os modelos treinados no Passo 4
    2. Carrega os dados de teste do Passo 3
    3. Avalia cada modelo (RMSE, MAE, MAPE)
    4. Salva as métricas em CSV
    5. Gera visualizações comparativas (H1, H3)
    """
    print("="*50)
    print("INICIANDO PIPELINE DE AVALIAÇÃO DE PERFORMANCE (PASSO 5)")
    print("="*50)
    
    # 1. Processamento: Avaliar modelos
    print("\n[1/2] Avaliando Modelos e Calculando Métricas...")
    evaluator = ModelEvaluator()
    results_df = evaluator.run_evaluation()
    
    # 2. Visualização: Gerar gráficos comparativos
    print("\n[2/2] Gerando Visualizações Comparativas...")
    visualizer = EvaluationVisualizer(results_df)
    visualizer.generate_all_visualizations()
    
    print("\n" + "="*50)
    print("PIPELINE CONCLUÍDO COM SUCESSO!")
    print(f"Métricas salvas em: {config.OUTPUT_DIR}")
    print(f"Visualizações salvas em: {os.path.join(config.OUTPUT_DIR, 'visualizacoes_avaliacao')}")
    print("="*50)

if __name__ == "__main__":
    run_evaluation_pipeline()
