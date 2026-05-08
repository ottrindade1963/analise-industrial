"""Visualizações para análise exploratória de indicadores de governança."""
import matplotlib
matplotlib.use('Agg')  # Backend não-interativo para Colab
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from passo2_eda_quali_config import (
    WGI_COLS, ICRG_COL, ALL_INDICATORS, LABELS, SHORT_LABELS,
    GOV_THRESHOLDS, DECADES, OUTPUT_DIR, DPI,
    FIGSIZE_LARGE, FIGSIZE_MEDIUM, FIGSIZE_SMALL,
    PALETTE, PALETTE_SEQ
)

sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10


def _salvar(fig, nome):
    path = f"{OUTPUT_DIR}/{nome}"
    fig.savefig(path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"   ✅ {path}")


# ── 1. HEATMAP DE MISSING ──────────────────────────────────
def plot_heatmap_missing(df):
    """Heatmap de valores ausentes por indicador e período."""
    print("\n📊 1/12 Heatmap de valores ausentes...")
    
    df_temp = df.copy()
    df_temp['periodo'] = pd.cut(
        df_temp['year'],
        bins=[1995, 2004, 2014, 2024],
        labels=['1996-2004', '2005-2014', '2015-2024']
    )
    
    missing_pct = df_temp.groupby('periodo')[ALL_INDICATORS].apply(
        lambda x: x.isnull().mean() * 100
    )
    missing_pct.columns = [SHORT_LABELS[c] for c in ALL_INDICATORS]
    
    fig, ax = plt.subplots(figsize=FIGSIZE_SMALL)
    sns.heatmap(missing_pct, annot=True, fmt='.1f', cmap='YlOrRd',
                ax=ax, vmin=0, vmax=100, linewidths=0.5)
    ax.set_title('Valores Ausentes por Indicador e Período (%)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Período')
    _salvar(fig, '01_heatmap_missing.png')


# ── 2. HISTOGRAMAS DE DISTRIBUIÇÃO ─────────────────────────
def plot_histogramas(df):
    """Histogramas de distribuição de cada indicador WGI."""
    print("📊 2/12 Histogramas de distribuição...")
    
    fig, axes = plt.subplots(2, 3, figsize=FIGSIZE_LARGE)
    axes = axes.flatten()
    
    for i, col in enumerate(WGI_COLS):
        s = df[col].dropna()
        axes[i].hist(s, bins=40, color='steelblue', edgecolor='white', alpha=0.8)
        axes[i].axvline(s.mean(), color='red', linestyle='--', linewidth=1.5, label=f'Média: {s.mean():.3f}')
        axes[i].axvline(s.median(), color='green', linestyle=':', linewidth=1.5, label=f'Mediana: {s.median():.3f}')
        axes[i].set_title(SHORT_LABELS[col], fontsize=11, fontweight='bold')
        axes[i].set_xlabel('Score (0-1)')
        axes[i].legend(fontsize=8)
    
    fig.suptitle('Distribuição dos Indicadores WGI', fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    _salvar(fig, '02_histogramas.png')


# ── 3. BOXPLOTS COMPARATIVOS ──────────────────────────────
def plot_boxplots(df):
    """Boxplots comparativos de todos os indicadores."""
    print("📊 3/12 Boxplots comparativos...")
    
    fig, ax = plt.subplots(figsize=FIGSIZE_MEDIUM)
    
    data_to_plot = [df[col].dropna() for col in WGI_COLS]
    labels_plot = [SHORT_LABELS[col] for col in WGI_COLS]
    
    bp = ax.boxplot(data_to_plot, labels=labels_plot, patch_artist=True,
                    medianprops=dict(color='red', linewidth=2))
    
    colors = sns.color_palette('Set2', len(WGI_COLS))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Score (0-1)', fontsize=11)
    ax.set_title('Comparação dos Indicadores WGI — Boxplots', fontsize=13, fontweight='bold')
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Limiar médio (0.5)')
    ax.legend()
    fig.tight_layout()
    _salvar(fig, '03_boxplots.png')


# ── 4. MATRIZ DE CORRELAÇÃO (PEARSON + SPEARMAN) ──────────
def plot_correlacao(df):
    """Matrizes de correlação de Pearson e Spearman lado a lado."""
    print("📊 4/12 Matrizes de correlação...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    short = [SHORT_LABELS[c] for c in ALL_INDICATORS]
    
    # Pearson
    corr_p = df[ALL_INDICATORS].corr(method='pearson')
    corr_p.index = short
    corr_p.columns = short
    sns.heatmap(corr_p, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                ax=ax1, vmin=-1, vmax=1, linewidths=0.5, square=True)
    ax1.set_title('Correlação de Pearson', fontsize=12, fontweight='bold')
    
    # Spearman
    corr_s = df[ALL_INDICATORS].corr(method='spearman')
    corr_s.index = short
    corr_s.columns = short
    sns.heatmap(corr_s, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                ax=ax2, vmin=-1, vmax=1, linewidths=0.5, square=True)
    ax2.set_title('Correlação de Spearman', fontsize=12, fontweight='bold')
    
    fig.suptitle('Correlações entre Indicadores de Governança', fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    _salvar(fig, '04_correlacao.png')


# ── 5. EVOLUÇÃO TEMPORAL GLOBAL ───────────────────────────
def plot_evolucao_temporal(df):
    """Evolução temporal das médias globais dos indicadores WGI."""
    print("📊 5/12 Evolução temporal...")
    
    fig, ax = plt.subplots(figsize=FIGSIZE_MEDIUM)
    
    medias_anuais = df.groupby('year')[WGI_COLS].mean()
    
    colors = sns.color_palette('tab10', len(WGI_COLS))
    for i, col in enumerate(WGI_COLS):
        ax.plot(medias_anuais.index, medias_anuais[col],
                marker='o', markersize=4, linewidth=2,
                color=colors[i], label=SHORT_LABELS[col])
    
    ax.set_xlabel('Ano', fontsize=11)
    ax.set_ylabel('Score Médio Global (0-1)', fontsize=11)
    ax.set_title('Evolução Temporal dos Indicadores WGI — Média Global', fontsize=13, fontweight='bold')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    _salvar(fig, '05_evolucao_temporal.png')


# ── 6. RADAR CHART — TOP 10 vs BOTTOM 10 ─────────────────
def plot_radar_top_bottom(df):
    """Radar chart comparando Top 10 vs Bottom 10 países."""
    print("📊 6/12 Radar chart Top vs Bottom...")
    
    media_pais = df.groupby('country_code')[WGI_COLS].mean()
    media_geral = media_pais.mean(axis=1).sort_values()
    
    top10 = media_pais.loc[media_geral.tail(10).index].mean()
    bottom10 = media_pais.loc[media_geral.head(10).index].mean()
    media_global = media_pais.mean()
    
    categories = [SHORT_LABELS[c] for c in WGI_COLS]
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    for data, label, color, alpha in [
        (top10, 'Top 10', '#2ecc71', 0.25),
        (media_global, 'Média Global', '#3498db', 0.15),
        (bottom10, 'Bottom 10', '#e74c3c', 0.25)
    ]:
        values = data.values.tolist() + data.values.tolist()[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=label, color=color)
        ax.fill(angles, values, alpha=alpha, color=color)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title('Perfil de Governança: Top 10 vs Bottom 10', fontsize=13, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    _salvar(fig, '06_radar_top_bottom.png')


# ── 7. HEATMAP POR PAÍS (TOP 30) ─────────────────────────
def plot_heatmap_paises(df):
    """Heatmap do perfil de governança dos Top 30 países."""
    print("📊 7/12 Heatmap perfil dos países...")
    
    media_pais = df.groupby('country_code')[WGI_COLS].mean()
    media_geral = media_pais.mean(axis=1).sort_values(ascending=False)
    top30 = media_pais.loc[media_geral.head(30).index]
    
    top30.columns = [SHORT_LABELS[c] for c in WGI_COLS]
    top30 = top30.sort_values(top30.columns[0], ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 12))
    sns.heatmap(top30, annot=True, fmt='.2f', cmap=PALETTE,
                ax=ax, linewidths=0.5, vmin=0, vmax=1)
    ax.set_title('Perfil de Governança — Top 30 Países', fontsize=13, fontweight='bold')
    ax.set_ylabel('')
    fig.tight_layout()
    _salvar(fig, '07_heatmap_paises.png')


# ── 8. VIOLIN PLOTS POR PERÍODO ──────────────────────────
def plot_violin_periodos(df):
    """Violin plots dos indicadores WGI por período temporal."""
    print("📊 8/12 Violin plots por período...")
    
    df_temp = df.copy()
    df_temp['periodo'] = pd.cut(
        df_temp['year'],
        bins=[1995, 2004, 2014, 2024],
        labels=['1996-2004', '2005-2014', '2015-2024']
    )
    
    fig, axes = plt.subplots(2, 3, figsize=FIGSIZE_LARGE)
    axes = axes.flatten()
    
    for i, col in enumerate(WGI_COLS):
        subset = df_temp[['periodo', col]].dropna()
        sns.violinplot(data=subset, x='periodo', y=col, hue='periodo',
                       ax=axes[i], palette='Set2', inner='quartile', legend=False)
        axes[i].set_title(SHORT_LABELS[col], fontsize=11, fontweight='bold')
        axes[i].set_xlabel('')
        axes[i].set_ylabel('Score')
        axes[i].set_ylim(0, 1)
    
    fig.suptitle('Distribuição dos Indicadores WGI por Período', fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    _salvar(fig, '08_violin_periodos.png')


# ── 9. SCATTER WGI vs ICRG ───────────────────────────────
def plot_scatter_wgi_icrg(df):
    """Scatter plot da média WGI vs ICRG por país."""
    print("📊 9/12 Scatter WGI vs ICRG...")
    
    if ICRG_COL is None or ICRG_COL not in df.columns:
        print("   ⚠️ Coluna ICRG não encontrada. Pulando gráfico.")
        return
        
    media_pais = df.groupby('country_code')[ALL_INDICATORS].mean()
    media_pais['wgi_media'] = media_pais[WGI_COLS].mean(axis=1)
    media_pais = media_pais.dropna(subset=['wgi_media', ICRG_COL])
    
    if len(media_pais) < 5:
        print("   ⚠️  Dados insuficientes para scatter WGI vs ICRG")
        return
    
    fig, ax = plt.subplots(figsize=FIGSIZE_SMALL)
    
    scatter = ax.scatter(
        media_pais['wgi_media'], media_pais[ICRG_COL],
        c=media_pais['wgi_media'], cmap=PALETTE,
        s=60, alpha=0.7, edgecolors='gray', linewidth=0.5
    )
    
    # Linha de tendência
    z = np.polyfit(media_pais['wgi_media'], media_pais[ICRG_COL], 1)
    p = np.poly1d(z)
    x_line = np.linspace(media_pais['wgi_media'].min(), media_pais['wgi_media'].max(), 100)
    ax.plot(x_line, p(x_line), 'r--', linewidth=2, alpha=0.7, label='Tendência linear')
    
    # Correlação
    from scipy.stats import pearsonr
    r, p_val = pearsonr(media_pais['wgi_media'], media_pais[ICRG_COL])
    ax.text(0.05, 0.95, f'r = {r:.3f} (p = {p_val:.2e})',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.colorbar(scatter, ax=ax, label='Média WGI')
    ax.set_xlabel('Média WGI (0-1)', fontsize=11)
    ax.set_ylabel('ICRG (0-1)', fontsize=11)
    ax.set_title('Relação entre Média WGI e ICRG por País', fontsize=13, fontweight='bold')
    ax.legend()
    fig.tight_layout()
    _salvar(fig, '09_scatter_wgi_icrg.png')


# ── 10. BARRAS — CLASSIFICAÇÃO DE GOVERNANÇA ─────────────
def plot_classificacao_governanca(df):
    """Gráfico de barras com a distribuição de países por nível de governança."""
    print("📊 10/12 Classificação de governança...")
    
    media_pais = df.groupby('country_code')[WGI_COLS].mean().mean(axis=1)
    
    contagem = {}
    cores = ['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71', '#27ae60']
    for nivel, (low, high) in GOV_THRESHOLDS.items():
        contagem[nivel] = ((media_pais >= low) & (media_pais < high)).sum()
    
    fig, ax = plt.subplots(figsize=FIGSIZE_SMALL)
    bars = ax.bar(contagem.keys(), contagem.values(), color=cores, edgecolor='white', linewidth=1.5)
    
    for bar, val in zip(bars, contagem.values()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                str(val), ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    ax.set_ylabel('Número de Países', fontsize=11)
    ax.set_xlabel('Nível de Governança', fontsize=11)
    ax.set_title('Distribuição de Países por Nível de Governança (Média WGI)', fontsize=13, fontweight='bold')
    fig.tight_layout()
    _salvar(fig, '10_classificacao_governanca.png')


# ── 11. PAIRPLOT (4 INDICADORES-CHAVE) ───────────────────
def plot_pairplot(df):
    """Pairplot dos 4 indicadores WGI mais relevantes."""
    print("📊 11/12 Pairplot (4 indicadores-chave)...")
    
    key_cols = [
        'wgi_control_corruption', 'wgi_gov_effectiveness',
        'wgi_rule_law', 'wgi_voice_accountability'
    ]
    
    subset = df[key_cols].dropna()
    subset.columns = [SHORT_LABELS[c] for c in key_cols]
    
    g = sns.pairplot(subset, diag_kind='kde', plot_kws={'alpha': 0.3, 's': 10},
                     diag_kws={'fill': True})
    g.figure.suptitle('Relações Cruzadas — 4 Indicadores-Chave', fontsize=14, fontweight='bold', y=1.02)
    _salvar(g.figure, '11_pairplot.png')


# ── 12. EVOLUÇÃO TEMPORAL — PAÍSES EMERGENTES ────────────
def plot_evolucao_emergentes(df):
    """Evolução temporal dos indicadores para os 10 países emergentes do estudo."""
    print("📊 12/12 Evolução temporal — Emergentes...")
    
    emergentes = ['BRA', 'IND', 'CHN', 'ZAF', 'MEX', 'IDN', 'NGA', 'EGY', 'VNM', 'PHL']
    df_emerg = df[df['country_code'].isin(emergentes)]
    
    if df_emerg.empty:
        print("   ⚠️  Nenhum país emergente encontrado no dataset")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    colors = sns.color_palette('tab10', len(emergentes))
    
    for i, col in enumerate(WGI_COLS):
        for j, pais in enumerate(emergentes):
            dados_pais = df_emerg[df_emerg['country_code'] == pais].sort_values('year')
            if not dados_pais.empty:
                axes[i].plot(dados_pais['year'], dados_pais[col],
                            linewidth=1.5, color=colors[j], alpha=0.8, label=pais)
        
        axes[i].set_title(SHORT_LABELS[col], fontsize=11, fontweight='bold')
        axes[i].set_xlabel('')
        axes[i].set_ylabel('Score')
        axes[i].set_ylim(0, 1)
        axes[i].axhline(0.5, color='gray', linestyle='--', alpha=0.3)
    
    # Legenda única
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    
    fig.suptitle('Evolução da Governança — Países Emergentes', fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    _salvar(fig, '12_evolucao_emergentes.png')
