"""Orquestração da agregação de datasets quantitativo e qualitativo."""

from passo2_1_agregacao_processor import (
    carregar_datasets,
    metodo1_inner_join,
    metodo2_left_join_imputado,
    metodo3_outer_join_rastreavel
)
from passo2_1_agregacao_exporter import exportar_todos


def executar_agregacao():
    """Executa a agregação completa com os 3 métodos."""
    
    print("=" * 60)
    print("  AGREGAÇÃO DE DATASETS — QUANTITATIVO + QUALITATIVO")
    print("=" * 60)
    
    # Carregar dados
    print("\n  ── Carregando Datasets ──")
    df_quant, df_qual = carregar_datasets()
    
    # Aplicar os 3 métodos
    df_m1 = metodo1_inner_join(df_quant, df_qual)
    df_m2 = metodo2_left_join_imputado(df_quant, df_qual)
    df_m3 = metodo3_outer_join_rastreavel(df_quant, df_qual)
    
    # Exportar
    exportar_todos(df_m1, df_m2, df_m3)
    
    # Resumo final
    print("\n" + "=" * 60)
    print("  ✅ AGREGAÇÃO CONCLUÍDA — 3 MÉTODOS APLICADOS")
    print("=" * 60)
    print(f"\n  Método 1 (Inner):        {df_m1.shape[0]:,} linhas × {df_m1.shape[1]} colunas")
    print(f"  Método 2 (Left+Imput.):  {df_m2.shape[0]:,} linhas × {df_m2.shape[1]} colunas")
    print(f"  Método 3 (Outer+Fonte):  {df_m3.shape[0]:,} linhas × {df_m3.shape[1]} colunas")
    print(f"\n  📁 Ficheiros gerados: 6 (CSV + XLSX por método)")
    print("=" * 60)
    
    return df_m1, df_m2, df_m3

if __name__ == "__main__":
    executar_agregacao()
