"""Pipeline Mestre do Passo 2: Análise Exploratória dos Dados Brutos."""
import os
import sys

# ═══ Compatibilidade Colab: garantir que os módulos locais são encontrados ═══
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
os.chdir(_SCRIPT_DIR)
# ═══════════════════════════════════════════════════════════════════════════════


def executar_passo2_completo():
    print("=" * 70)
    print("INICIANDO PASSO 2 COMPLETO: ANÁLISE EXPLORATÓRIA (EDA)")
    print("=" * 70)
    
    # 1. EDA Quantitativa
    print("\n[1/2] EXECUTANDO EDA DOS DADOS QUANTITATIVOS (WDI)...")
    from passo2_eda_quant_pipeline import executar_eda
    executar_eda()
    
    # 2. EDA Qualitativa
    print("\n[2/2] EXECUTANDO EDA DOS DADOS QUALITATIVOS (WGI)...")
    from passo2_eda_quali_pipeline import executar_eda_qualitativa
    executar_eda_qualitativa()
    
    print("\n" + "=" * 70)
    print("PASSO 2 CONCLUÍDO COM SUCESSO!")
    print("=" * 70)

if __name__ == "__main__":
    executar_passo2_completo()
