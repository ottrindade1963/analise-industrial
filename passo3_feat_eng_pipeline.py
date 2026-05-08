"""Pipeline de Engenharia de Features com Geração Automática de Metadados."""
import os
import sys

# ═══ Compatibilidade Colab: garantir que os módulos locais são encontrados ═══
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
os.chdir(_SCRIPT_DIR)
# ═══════════════════════════════════════════════════════════════════════════════

import passo3_feat_eng_config as config
from passo3_feat_eng_processor import load_and_process_datasets
from passo3_feat_eng_visualizer import FeatureVisualizer


def run_feature_engineering_pipeline():
    """
    Executa o pipeline completo de Engenharia de Features:
    1. Carrega os datasets agregados (Inner, Left, Outer)
    2. Aplica as 3 estratégias (A1, A2, A3)
    3. Salva os novos datasets
    4. Gera visualizações (Heatmaps, PCA Variance)
    5. Gera metadados automáticos
    """
    print("="*50)
    print("INICIANDO PIPELINE DE ENGENHARIA DE FEATURES (PASSO 3)")
    print("="*50)
    
    # 1. Processamento: Carregar e aplicar estratégias
    print("\n[1/3] Processando Datasets e Aplicando Estratégias...")
    datasets_dict = load_and_process_datasets()
    
    # 2. Visualização: Gerar gráficos
    print("\n[2/3] Gerando Visualizações...")
    visualizer = FeatureVisualizer(datasets_dict)
    visualizer.generate_all_visualizations()
    
    # 3. Geração de Metadados
    print("\n[3/3] Gerando Metadados do Passo 3...")
    try:
        from metadata_generator import generate_metadata_passo3
        generate_metadata_passo3(datasets_dict, output_dir=config.OUTPUT_DIR)
        print("  ✅ Metadados do Passo 3 gerados com sucesso!")
    except Exception as e:
        print(f"  ⚠️ Não foi possível gerar metadados: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*50)
    print("PIPELINE CONCLUÍDO COM SUCESSO!")
    print(f"Datasets salvos em: {config.OUTPUT_DIR}")
    print(f"Visualizações salvas em: {os.path.join(config.OUTPUT_DIR, 'visualizacoes')}")
    print("="*50)

if __name__ == "__main__":
    run_feature_engineering_pipeline()
