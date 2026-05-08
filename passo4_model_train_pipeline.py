"""Pipeline de Treinamento de Modelos com Geração Automática de Metadados."""
import os
import sys

# ═══ Compatibilidade Colab: garantir que os módulos locais são encontrados ═══
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
os.chdir(_SCRIPT_DIR)
# ═══════════════════════════════════════════════════════════════════════════════

import time
import passo4_model_train_config as config
from passo4_model_train_processor import run_training_for_all
from passo4_model_train_visualizer import TrainingVisualizer


def run_model_training_pipeline():
    """
    Executa o pipeline completo de Treinamento de Modelos:
    1. Carrega os datasets processados no Passo 3
    2. Divide em Treino/Validação/Teste (Walk-forward temporal)
    3. Treina 7 modelos (RF, XGBoost, TFT, SARIMAX, LSTM, Bayes_PartialPooling, Bayes_CompletePooling)
    4. Salva os modelos em disco (.pkl)
    5. Gera visualizações de histórico de treino
    6. Gera metadados automáticos
    """
    print("="*50)
    print("INICIANDO PIPELINE DE TREINAMENTO DE MODELOS (PASSO 4)")
    print("="*50)
    
    # 1. Processamento: Treinar modelos
    print("\n[1/3] Treinando Modelos...")
    start_time = time.time()
    run_training_for_all()
    total_time = time.time() - start_time
    
    # 2. Visualizacao: Gerar graficos de treino
    print("\n[2/3] Gerando Visualizacoes de Treino...")
    try:
        visualizer = TrainingVisualizer()
        visualizer.plot_real_training_metrics()
        visualizer.plot_predictions_vs_actual()
        visualizer.plot_best_model_analysis()
        visualizer.plot_predictions_comparison()
        print("  -> Visualizacoes geradas com sucesso!")
    except Exception as e:
        print(f"  AVISO: Nao foi possivel gerar visualizacoes: {e}")
        print(f"  Os modelos foram treinados e salvos com sucesso.")
        print(f"  As visualizacoes podem ser geradas posteriormente no Passo 5.")
    
    # 3. Geração de Metadados
    print("\n[3/3] Gerando Metadados do Passo 4...")
    try:
        from metadata_generator import generate_metadata_passo4
        import pandas as pd
        
        # Carregar resumo do treinamento
        summary_path = os.path.join(config.OUTPUT_DIR, 'resumo_treinamento_completo.csv')
        all_summaries = []
        if os.path.exists(summary_path):
            df_summary = pd.read_csv(summary_path)
            all_summaries = df_summary.to_dict('records')
        
        generate_metadata_passo4(
            all_summaries=all_summaries,
            training_times={"total_segundos": round(total_time, 1), "total_minutos": round(total_time/60, 2)},
            output_dir=config.OUTPUT_DIR
        )
        print("  ✅ Metadados do Passo 4 gerados com sucesso!")
    except Exception as e:
        print(f"  ⚠️ Não foi possível gerar metadados: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*50)
    print("PIPELINE CONCLUÍDO COM SUCESSO!")
    print(f"Modelos salvos em: {config.OUTPUT_DIR}")
    print(f"Visualizações salvas em: {os.path.join(config.OUTPUT_DIR, 'visualizacoes_treino')}")
    print(f"Tempo total: {total_time/60:.1f} minutos")
    print("="*50)

if __name__ == "__main__":
    run_model_training_pipeline()
