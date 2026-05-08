"""Pipeline Mestre do Passo 2.1: Limpeza, Agregação e EDA dos Agregados com Metadados Automáticos."""
import os
import sys

# ═══ Compatibilidade Colab: garantir que os módulos locais são encontrados ═══
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
os.chdir(_SCRIPT_DIR)
# ═══════════════════════════════════════════════════════════════════════════════

import pandas as pd


def executar_passo2_1_completo():
    print("=" * 70)
    print("INICIANDO PASSO 2.1 COMPLETO: LIMPEZA, AGREGAÇÃO E EDA")
    print("=" * 70)
    
    # 1. Limpeza
    print("\n[1/5] EXECUTANDO LIMPEZA DE DADOS QUANTITATIVOS...")
    from passo2_1_limpeza_pipeline import executar_limpeza
    df_wdi_limpo, df_wgi_limpo, stats_wdi, stats_wgi = executar_limpeza()
    
    # 2. Agregação
    print("\n[2/5] EXECUTANDO AGREGAÇÃO (3 MÉTODOS)...")
    from passo2_1_agregacao_pipeline import executar_agregacao
    executar_agregacao()
    
    # 3. EDA Dados Não Agregados Limpos (WDI + WGI)
    print("\n[3/5] EXECUTANDO ANÁLISE EXPLORATÓRIA DOS DADOS NÃO AGREGADOS LIMPOS...")
    try:
        from passo2_1_eda_nao_agreg_visualizer import executar_eda_nao_agregado
        executar_eda_nao_agregado(df_wdi_limpo, df_wgi_limpo)
    except Exception as e:
        print(f"  AVISO: Não foi possível executar EDA dos não agregados: {e}")
        print("  Continuando com EDA dos agregados...")
    
    # 4. EDA Agregados
    print("\n[4/5] EXECUTANDO ANÁLISE EXPLORATÓRIA DOS AGREGADOS...")
    from passo2_1_eda_agreg_pipeline import executar_eda_agregados
    executar_eda_agregados()
    
    # 5. Geração de Metadados
    print("\n[5/5] GERANDO METADADOS DO PASSO 2.1...")
    try:
        from metadata_generator import generate_metadata_passo2_1
        from passo2_1_limpeza_config import (
            DATA_PATH, DATA_PATH_WGI, OUTPUT_DIR, METODOS_IMPUTACAO
        )
        
        # Carregar dados originais para comparação
        df_wdi_original = None
        df_wgi_original = None
        
        if os.path.exists(DATA_PATH):
            df_wdi_original = pd.read_csv(DATA_PATH)
        if os.path.exists(DATA_PATH_WGI):
            df_wgi_original = pd.read_csv(DATA_PATH_WGI)
        
        # Info dos métodos de agregação
        metodos_agregacao = {}
        
        # Inner
        inner_path = 'agregado_metodo1_inner/agregado_inner.csv'
        if os.path.exists(inner_path):
            df_inner = pd.read_csv(inner_path)
            metodos_agregacao['inner'] = {
                "linhas": int(df_inner.shape[0]),
                "colunas": int(df_inner.shape[1]),
                "paises": int(df_inner['country_code'].nunique()) if 'country_code' in df_inner.columns else 0
            }
        
        # Left
        left_path = 'agregado_metodo2_left_imputado/agregado_left_imputado.csv'
        if os.path.exists(left_path):
            df_left = pd.read_csv(left_path)
            metodos_agregacao['left'] = {
                "linhas": int(df_left.shape[0]),
                "colunas": int(df_left.shape[1]),
                "paises": int(df_left['country_code'].nunique()) if 'country_code' in df_left.columns else 0
            }
        
        # Outer
        outer_path = 'agregado_metodo3_outer_completo/agregado_outer_completo.csv'
        if os.path.exists(outer_path):
            df_outer = pd.read_csv(outer_path)
            metodos_agregacao['outer'] = {
                "linhas": int(df_outer.shape[0]),
                "colunas": int(df_outer.shape[1]),
                "paises": int(df_outer['country_code'].nunique()) if 'country_code' in df_outer.columns else 0,
                "fontes": df_outer['fonte_dados'].value_counts().to_dict() if 'fonte_dados' in df_outer.columns else {}
            }
        
        generate_metadata_passo2_1(
            df_wdi_original=df_wdi_original,
            df_wdi_limpo=df_wdi_limpo,
            df_wgi_original=df_wgi_original,
            df_wgi_limpo=df_wgi_limpo,
            stats_wdi=stats_wdi,
            stats_wgi=stats_wgi,
            metodos_imputacao=METODOS_IMPUTACAO,
            metodos_agregacao=metodos_agregacao,
            output_dir=OUTPUT_DIR
        )
        print("  ✅ Metadados do Passo 2.1 gerados com sucesso!")
    except Exception as e:
        print(f"  ⚠️ Não foi possível gerar metadados: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("PASSO 2.1 CONCLUÍDO COM SUCESSO!")
    print("=" * 70)

if __name__ == "__main__":
    executar_passo2_1_completo()
