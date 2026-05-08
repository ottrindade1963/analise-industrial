
import os
import sys

# ═══ Compatibilidade Colab: garantir que os módulos locais são encontrados ═══
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
os.chdir(_SCRIPT_DIR)
# ═══════════════════════════════════════════════════════════════════════════════

import passo8_geo_config as config
from passo8_geo_processor import GeoAnalyzer
from passo8_geo_visualizer import GeoVisualizer

def run_geo_analysis_pipeline():
    """
    Executa o pipeline completo de Análise Geográfica e Comparativa:
    1. Carrega os modelos treinados e dados de teste
    2. Faz previsões e calcula o erro absoluto por país
    3. Classifica os países em Pobres, Médios e Ricos com base no PIB per capita
    4. Agrega o erro médio por país e classe económica
    5. Gera visualizações comparativas (Boxplots, Scatter plots)
    """
    print("="*50)
    print("INICIANDO PIPELINE DE ANÁLISE GEOGRÁFICA (PASSO 8)")
    print("="*50)
    
    # 1. Processamento: Classificar países e calcular erros
    print("\n[1/2] Classificando Países e Calculando Erros...")
    analyzer = GeoAnalyzer()
    preds_dict = analyzer.run_analysis()
    
    # 2. Visualização: Gerar gráficos comparativos
    print("\n[2/2] Gerando Visualizações Geográficas...")
    visualizer = GeoVisualizer(preds_dict)
    visualizer.generate_all_visualizations()
    
    print("\n" + "="*50)
    print("PIPELINE CONCLUÍDO COM SUCESSO!")
    print(f"Dados geográficos salvos em: {config.OUTPUT_DIR}")
    print(f"Visualizações salvas em: {os.path.join(config.OUTPUT_DIR, 'visualizacoes_geograficas')}")
    print("="*50)

if __name__ == "__main__":
    run_geo_analysis_pipeline()
