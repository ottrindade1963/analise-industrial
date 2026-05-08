"""Orquestração da análise exploratória dos 3 datasets agregados."""

import pandas as pd
from passo2_1_eda_agreg_config import DATASETS
from passo2_1_eda_agreg_processor import (
    resumo_geral,
    analise_missing,
    estatisticas_descritivas,
    intervalos_confianca,
    testes_normalidade,
    correlacao_quant_qual,
    estatisticas_por_decada
)
from passo2_1_eda_agreg_visualizer import gerar_todas_visualizacoes


def analisar_dataset(chave):
    """Executa análise exploratória completa de um dataset."""
    info = DATASETS[chave]
    nome = info["nome"]
    path = info["path"]
    output_dir = info["output"]
    
    # Carregar
    df = pd.read_csv(path)
    
    # Estatísticas
    resumo_geral(df, nome)
    analise_missing(df, nome)
    estatisticas_descritivas(df, nome)
    intervalos_confianca(df, nome)
    testes_normalidade(df, nome)
    correlacao_quant_qual(df, nome)
    estatisticas_por_decada(df, nome)
    
    # Visualizações
    gerar_todas_visualizacoes(df, output_dir, nome)
    
    return df


def executar_eda_agregados():
    """Executa a EDA completa dos 3 datasets, um por um."""
    
    print("╔" + "═" * 68 + "╗")
    print("║  ANÁLISE EXPLORATÓRIA — DATASETS AGREGADOS (QUANT. + QUAL.)       ║")
    print("╚" + "═" * 68 + "╝")
    
    resultados = {}
    
    for chave in DATASETS:
        resultados[chave] = analisar_dataset(chave)
        print(f"\n  ✅ {DATASETS[chave]['nome']} — CONCLUÍDO")
        print(f"     Gráficos em: {DATASETS[chave]['output']}/")
    
    # Resumo final
    print("\n" + "═" * 70)
    print("  ✅ ANÁLISE EXPLORATÓRIA COMPLETA — 3 DATASETS ANALISADOS")
    print("═" * 70)
    for chave, df in resultados.items():
        print(f"  • {DATASETS[chave]['nome']}: {df.shape[0]:,} linhas × {df.shape[1]} colunas → 16 gráficos")
    print(f"\n  📁 Total: 48 gráficos gerados (16 por dataset)")
    print("═" * 70)
    
    return resultados

if __name__ == "__main__":
    executar_eda_agregados()
