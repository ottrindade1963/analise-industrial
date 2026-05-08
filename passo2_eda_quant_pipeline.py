"""Pipeline da análise exploratória completa."""

from passo2_eda_quant_processor import (
    carregar_dados, resumo_geral, tabela_missing,
    estatisticas_descritivas, intervalos_confianca,
    teste_normalidade, estatisticas_por_decada,
)
from passo2_eda_quant_visualizer import (
    plot_missing, plot_histogramas, plot_boxplots, plot_correlacao,
    plot_evolucao_temporal, plot_top_paises, plot_scatter_pib_industria,
    plot_violin_emprego, plot_pairplot, plot_heatmap_paises,
)


def executar_eda():
    print("=" * 60)
    print("  ANÁLISE EXPLORATÓRIA — PAÍSES EMERGENTES")
    print("=" * 60)

    # ── Carregar dados ──
    df = carregar_dados()

    # ── Estatísticas descritivas completas ──
    resumo_geral(df)
    tabela_missing(df)
    estatisticas_descritivas(df)
    intervalos_confianca(df)
    teste_normalidade(df)
    estatisticas_por_decada(df)

    # ── Visualizações ──
    print(f"\n{'='*60}")
    print(f"  📊 GERANDO VISUALIZAÇÕES")
    print(f"{'='*60}")
    plot_missing(df)
    plot_histogramas(df)
    plot_boxplots(df)
    plot_correlacao(df)
    plot_evolucao_temporal(df)
    plot_top_paises(df)
    plot_scatter_pib_industria(df)
    plot_violin_emprego(df)
    plot_pairplot(df)
    plot_heatmap_paises(df)

    print("\n" + "=" * 60)
    print("  ✅ ANÁLISE CONCLUÍDA — Estatísticas + 10 gráficos!")
    print("  📂 Gráficos em: resultados_eda/")
    print("=" * 60)

if __name__ == "__main__":
    executar_eda()
