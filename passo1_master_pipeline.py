"""Pipeline Mestre do Passo 1: Extração de Dados Reais com Metadados Automáticos."""
import os
import sys

# ═══ Compatibilidade Colab: garantir que os módulos locais são encontrados ═══
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
os.chdir(_SCRIPT_DIR)
# ═══════════════════════════════════════════════════════════════════════════════

import pandas as pd


def executar_passo1_completo():
    print("=" * 70)
    print("INICIANDO PASSO 1 COMPLETO: EXTRAÇÃO DE DADOS REAIS")
    print("=" * 70)
    
    # 1. Extração Quantitativa (WDI)
    print("\n[1/3] EXECUTANDO EXTRAÇÃO DE DADOS QUANTITATIVOS (WDI)...")
    from passo1_extracao_pipeline import executar
    result = executar()
    
    # Compatibilidade: executar() pode retornar tuple ou None
    if result and isinstance(result, tuple):
        df_wdi, codigos = result
    else:
        df_wdi = None
        codigos = []
    
    # 2. Extração Qualitativa (WGI)
    print("\n[2/3] EXECUTANDO EXTRAÇÃO DE DADOS QUALITATIVOS (WGI)...")
    from passo1_extracao_quali_processor import executar_extracao_wgi
    executar_extracao_wgi()
    
    # 3. Geração de Metadados
    print("\n[3/3] GERANDO METADADOS DO PASSO 1...")
    try:
        from metadata_generator import generate_metadata_passo1
        from passo1_extracao_config import INDICADORES, DATA_DIR, PAISES_EMERGENTES
        
        # Carregar WGI se disponível
        df_wgi = None
        wgi_path = 'dados_qualitativos.csv'
        if os.path.exists(wgi_path):
            df_wgi = pd.read_csv(wgi_path)
        
        # Carregar WDI se não foi retornado
        if df_wdi is None:
            wdi_path = os.path.join(DATA_DIR, 'wdi_emergentes_final.csv')
            if os.path.exists(wdi_path):
                df_wdi = pd.read_csv(wdi_path)
        
        # Indicadores WGI
        indicadores_wgi = {
            "CC.EST": "wgi_control_corruption",
            "GE.EST": "wgi_gov_effectiveness",
            "PV.EST": "wgi_political_stability",
            "RQ.EST": "wgi_regulatory_quality",
            "RL.EST": "wgi_rule_law",
            "VA.EST": "wgi_voice_accountability"
        }
        
        generate_metadata_passo1(
            df_wdi=df_wdi,
            df_wgi=df_wgi,
            indicadores_wdi=INDICADORES,
            indicadores_wgi=indicadores_wgi,
            paises_extraidos=PAISES_EMERGENTES if codigos == [] else codigos,
            output_dir=DATA_DIR
        )
        print("  ✅ Metadados do Passo 1 gerados com sucesso!")
    except Exception as e:
        print(f"  ⚠️ Não foi possível gerar metadados: {e}")
    
    print("\n" + "=" * 70)
    print("PASSO 1 CONCLUÍDO COM SUCESSO!")
    print("=" * 70)

if __name__ == "__main__":
    executar_passo1_completo()
