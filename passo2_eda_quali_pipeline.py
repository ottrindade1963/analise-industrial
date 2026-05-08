"""Orquestração da análise exploratória do dataset qualitativo WGI + ICRG."""

from passo2_eda_quali_processor import (
    carregar_dados, tabela_missing, estatisticas_descritivas,
    intervalos_confianca, testes_normalidade,
    classificacao_governanca, estatisticas_por_periodo,
    correlacao_indicadores
)
from passo2_eda_quali_visualizer import (
    plot_heatmap_missing, plot_histogramas, plot_boxplots,
    plot_correlacao, plot_evolucao_temporal, plot_radar_top_bottom,
    plot_heatmap_paises, plot_violin_periodos, plot_scatter_wgi_icrg,
    plot_classificacao_governanca, plot_pairplot, plot_evolucao_emergentes
)


def executar_eda_qualitativa():
    """Executa a análise exploratória completa do dataset qualitativo."""
    
    print("=" * 60)
    print("  ANÁLISE EXPLORATÓRIA — INDICADORES DE GOVERNANÇA")
    print("  WGI (Banco Mundial) + ICRG (QoG)")
    print("=" * 60)
    
    # ── Carregamento e Estatísticas ──
    df = carregar_dados()
    tabela_missing(df)
    estatisticas_descritivas(df)
    intervalos_confianca(df)
    testes_normalidade(df)
    classificacao_governanca(df)
    estatisticas_por_periodo(df)
    correlacao_indicadores(df)
    
    # ── Visualizações ──
    print("\n" + "=" * 60)
    print("  GERANDO VISUALIZAÇÕES")
    print("=" * 60)
    
    plot_heatmap_missing(df)
    plot_histogramas(df)
    plot_boxplots(df)
    plot_correlacao(df)
    plot_evolucao_temporal(df)
    plot_radar_top_bottom(df)
    plot_heatmap_paises(df)
    plot_violin_periodos(df)
    plot_scatter_wgi_icrg(df)
    plot_classificacao_governanca(df)
    plot_pairplot(df)
    plot_evolucao_emergentes(df)
    
    print("\n" + "=" * 60)
    print("  ✅ ANÁLISE EXPLORATÓRIA QUALITATIVA CONCLUÍDA!")
    print("  📁 Gráficos salvos em: resultados_eda_quali/")
    print("=" * 60)
    
    return df

if __name__ == "__main__":
    executar_eda_qualitativa()
