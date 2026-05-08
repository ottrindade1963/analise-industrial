"""Visualizações para análise exploratória dos datasets agregados."""
import matplotlib
matplotlib.use('Agg')  # Backend não-interativo para Colab
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from passo2_1_eda_agreg_config import QUANT_VARS, QUAL_VARS, ALL_VARS, LABELS, EMERGENTES_DESTAQUE

plt.rcParams['figure.max_open_warning'] = 50


def _salvar(fig, output_dir, nome):
    """Salva e fecha figura."""
    fig.savefig(f"{output_dir}/{nome}.png", dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_heatmap_missing(df, output_dir, titulo):
    """1. Heatmap de valores ausentes."""
    cols = [c for c in ALL_VARS if c in df.columns]
    fig, ax = plt.subplots(figsize=(14, 6))
    missing = df[cols].isnull().astype(int)
    sns.heatmap(missing.T, cbar_kws={'label': 'Ausente'}, cmap='YlOrRd',
                yticklabels=[LABELS.get(c, c) for c in cols], ax=ax)
    ax.set_title(f'Valores Ausentes — {titulo}', fontsize=13)
    ax.set_xlabel('Observações')
    _salvar(fig, output_dir, '01_heatmap_missing')


def plot_histogramas_quant(df, output_dir, titulo):
    """2. Histogramas das variáveis quantitativas."""
    cols = [c for c in QUANT_VARS if c in df.columns]
    n = len(cols)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    for i, col in enumerate(cols):
        s = df[col].dropna()
        axes[i].hist(s, bins=30, color='steelblue', edgecolor='white', alpha=0.8)
        axes[i].set_title(LABELS.get(col, col), fontsize=10)
        axes[i].axvline(s.mean(), color='red', linestyle='--', label='Média')
        axes[i].axvline(s.median(), color='green', linestyle='-', label='Mediana')
        axes[i].legend(fontsize=7)
    for j in range(n, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle(f'Histogramas — Variáveis Quantitativas — {titulo}', fontsize=13)
    plt.tight_layout()
    _salvar(fig, output_dir, '02_histogramas_quant')


def plot_histogramas_qual(df, output_dir, titulo):
    """3. Histogramas das variáveis qualitativas (governança)."""
    cols = [c for c in QUAL_VARS if c in df.columns]
    n = len(cols)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    for i, col in enumerate(cols):
        s = df[col].dropna()
        axes[i].hist(s, bins=30, color='darkorange', edgecolor='white', alpha=0.8)
        axes[i].set_title(LABELS.get(col, col), fontsize=10)
        axes[i].axvline(s.mean(), color='red', linestyle='--', label='Média')
        axes[i].axvline(s.median(), color='green', linestyle='-', label='Mediana')
        axes[i].legend(fontsize=7)
    for j in range(n, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle(f'Histogramas — Variáveis Qualitativas — {titulo}', fontsize=13)
    plt.tight_layout()
    _salvar(fig, output_dir, '03_histogramas_qual')


def plot_boxplots_quant(df, output_dir, titulo):
    """4. Boxplots das variáveis quantitativas."""
    cols = [c for c in QUANT_VARS if c in df.columns]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    for i, col in enumerate(cols):
        s = df[col].dropna()
        axes[i].boxplot(s, vert=True)
        axes[i].set_title(LABELS.get(col, col), fontsize=10)
    for j in range(len(cols), len(axes)):
        axes[j].set_visible(False)
    fig.suptitle(f'Boxplots — Variáveis Quantitativas — {titulo}', fontsize=13)
    plt.tight_layout()
    _salvar(fig, output_dir, '04_boxplots_quant')


def plot_boxplots_qual(df, output_dir, titulo):
    """5. Boxplots das variáveis qualitativas."""
    cols = [c for c in QUAL_VARS if c in df.columns]
    fig, ax = plt.subplots(figsize=(12, 6))
    data_plot = df[cols].dropna()
    data_plot.columns = [LABELS.get(c, c) for c in cols]
    data_plot.boxplot(ax=ax, rot=30)
    ax.set_title(f'Boxplots — Variáveis Qualitativas — {titulo}', fontsize=13)
    ax.set_ylabel('Valor')
    plt.tight_layout()
    _salvar(fig, output_dir, '05_boxplots_qual')


def plot_correlacao_completa(df, output_dir, titulo):
    """6. Matriz de correlação completa (quant + qual)."""
    cols = [c for c in ALL_VARS if c in df.columns]
    corr = df[cols].corr(method='pearson')
    corr.index = [LABELS.get(c, c) for c in corr.index]
    corr.columns = [LABELS.get(c, c) for c in corr.columns]
    
    fig, ax = plt.subplots(figsize=(14, 11))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, vmin=-1, vmax=1, ax=ax, annot_kws={'size': 8})
    ax.set_title(f'Correlação Pearson — {titulo}', fontsize=13)
    plt.tight_layout()
    _salvar(fig, output_dir, '06_correlacao_completa')


def plot_correlacao_cruzada(df, output_dir, titulo):
    """7. Correlação cruzada Quant × Qual."""
    q_cols = [c for c in QUANT_VARS if c in df.columns]
    g_cols = [c for c in QUAL_VARS if c in df.columns]
    if not q_cols or not g_cols:
        return
    
    corr = df[q_cols + g_cols].corr().loc[q_cols, g_cols]
    corr.index = [LABELS.get(c, c) for c in corr.index]
    corr.columns = [LABELS.get(c, c) for c in corr.columns]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, vmin=-1, vmax=1, ax=ax)
    ax.set_title(f'Correlação Cruzada (Quant. × Qual.) — {titulo}', fontsize=13)
    plt.tight_layout()
    _salvar(fig, output_dir, '07_correlacao_cruzada')


def plot_evolucao_temporal_quant(df, output_dir, titulo):
    """8. Evolução temporal das variáveis quantitativas."""
    cols = [c for c in QUANT_VARS if c in df.columns]
    medias = df.groupby('year')[cols].mean()
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    for i, col in enumerate(cols):
        axes[i].plot(medias.index, medias[col], color='steelblue', linewidth=1.5)
        axes[i].set_title(LABELS.get(col, col), fontsize=10)
        axes[i].grid(True, alpha=0.3)
    for j in range(len(cols), len(axes)):
        axes[j].set_visible(False)
    fig.suptitle(f'Evolução Temporal (Média) — Quant. — {titulo}', fontsize=13)
    plt.tight_layout()
    _salvar(fig, output_dir, '08_evolucao_quant')


def plot_evolucao_temporal_qual(df, output_dir, titulo):
    """9. Evolução temporal das variáveis qualitativas."""
    cols = [c for c in QUAL_VARS if c in df.columns]
    medias = df.groupby('year')[cols].mean()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    for col in cols:
        ax.plot(medias.index, medias[col], linewidth=1.5, label=LABELS.get(col, col))
    ax.set_title(f'Evolução Temporal (Média) — Qual. — {titulo}', fontsize=13)
    ax.set_xlabel('Ano')
    ax.set_ylabel('Score')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _salvar(fig, output_dir, '09_evolucao_qual')


def plot_top10_pib(df, output_dir, titulo):
    """10. Top 10 países por PIB per capita."""
    if 'pib_per_capita_ppc' not in df.columns:
        return
    top = df.groupby('country_code')['pib_per_capita_ppc'].mean().nlargest(10)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    top.sort_values().plot(kind='barh', color='steelblue', ax=ax)
    ax.set_title(f'Top 10 Países — PIB per capita (média) — {titulo}', fontsize=13)
    ax.set_xlabel('PIB per capita (PPC)')
    plt.tight_layout()
    _salvar(fig, output_dir, '10_top10_pib')


def plot_scatter_pib_governanca(df, output_dir, titulo):
    """11. Scatter PIB per capita vs Governança (média WGI)."""
    if 'pib_per_capita_ppc' not in df.columns:
        return
    g_cols = [c for c in QUAL_VARS[:6] if c in df.columns]
    if not g_cols:
        return
    
    df_temp = df.copy()
    df_temp['wgi_media'] = df_temp[g_cols].mean(axis=1)
    
    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(df_temp['wgi_media'], df_temp['pib_per_capita_ppc'],
                         alpha=0.3, s=15, c='steelblue')
    ax.set_xlabel('Média WGI (Governança)', fontsize=11)
    ax.set_ylabel('PIB per capita (PPC)', fontsize=11)
    ax.set_title(f'PIB per capita vs Governança — {titulo}', fontsize=13)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _salvar(fig, output_dir, '11_scatter_pib_governanca')


def plot_violin_emprego(df, output_dir, titulo):
    """12. Violin plot emprego industrial por década."""
    if 'emprego_industria_percent_emprego_total' not in df.columns:
        return
    df_temp = df.copy()
    df_temp['decada'] = (df_temp['year'] // 10) * 10
    df_temp = df_temp.dropna(subset=['emprego_industria_percent_emprego_total'])
    
    fig, ax = plt.subplots(figsize=(12, 6))
    decadas = sorted(df_temp['decada'].unique())
    data = [df_temp[df_temp['decada'] == d]['emprego_industria_percent_emprego_total'].values for d in decadas]
    parts = ax.violinplot(data, positions=range(len(decadas)), showmeans=True, showmedians=True)
    ax.set_xticks(range(len(decadas)))
    ax.set_xticklabels([str(int(d)) + 's' for d in decadas])
    ax.set_title(f'Emprego Industrial por Década — {titulo}', fontsize=13)
    ax.set_ylabel('Emprego Industrial (%)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _salvar(fig, output_dir, '12_violin_emprego')


def plot_radar_emergentes(df, output_dir, titulo):
    """13. Radar dos países emergentes de destaque (governança)."""
    g_cols = [c for c in QUAL_VARS[:6] if c in df.columns]
    if not g_cols:
        return
    
    paises = [p for p in EMERGENTES_DESTAQUE if p in df['country_code'].values][:6]
    if not paises:
        return
    
    medias = df[df['country_code'].isin(paises)].groupby('country_code')[g_cols].mean()
    
    # Normalizar 0-1
    min_vals = medias.min()
    max_vals = medias.max()
    rng = max_vals - min_vals
    rng[rng == 0] = 1
    norm = (medias - min_vals) / rng
    
    angles = np.linspace(0, 2 * np.pi, len(g_cols), endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors = plt.cm.Set2(np.linspace(0, 1, len(paises)))
    
    for i, pais in enumerate(paises):
        if pais in norm.index:
            valores = norm.loc[pais].tolist() + [norm.loc[pais].iloc[0]]
            ax.plot(angles, valores, 'o-', linewidth=1.5, label=pais, color=colors[i])
            ax.fill(angles, valores, alpha=0.1, color=colors[i])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([LABELS.get(c, c) for c in g_cols], fontsize=8)
    ax.set_title(f'Perfil de Governança — Emergentes — {titulo}', fontsize=12, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
    plt.tight_layout()
    _salvar(fig, output_dir, '13_radar_emergentes')


def plot_heatmap_paises(df, output_dir, titulo):
    """14. Heatmap perfil dos top 20 países (todas variáveis)."""
    cols = [c for c in ALL_VARS if c in df.columns]
    medias = df.groupby('country_code')[cols].mean()
    
    # Normalizar
    norm = (medias - medias.min()) / (medias.max() - medias.min())
    top20 = norm.dropna().mean(axis=1).nlargest(20).index
    norm = norm.loc[top20]
    norm.columns = [LABELS.get(c, c) for c in norm.columns]
    
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(norm, annot=True, fmt='.2f', cmap='YlGnBu', ax=ax, annot_kws={'size': 7})
    ax.set_title(f'Perfil Normalizado — Top 20 Países — {titulo}', fontsize=13)
    plt.tight_layout()
    _salvar(fig, output_dir, '14_heatmap_paises')


def plot_pairplot(df, output_dir, titulo):
    """15. Pairplot de variáveis-chave (4 quant + 2 qual)."""
    cols_sel = []
    for c in ['pib_per_capita_ppc', 'comercio_percent_pib',
              'emprego_industria_percent_emprego_total', 'valor_agregado_industrial_percent_pib',
              'wgi_gov_effectiveness', 'wgi_control_corruption']:
        if c in df.columns:
            cols_sel.append(c)
    
    if len(cols_sel) < 4:
        return
    
    df_plot = df[cols_sel].dropna().sample(min(1000, len(df)), random_state=42)
    df_plot.columns = [LABELS.get(c, c) for c in df_plot.columns]
    
    g = sns.pairplot(df_plot, diag_kind='kde', plot_kws={'alpha': 0.4, 's': 10})
    g.figure.suptitle(f'Pairplot — Variáveis-Chave — {titulo}', y=1.02, fontsize=13)
    g.figure.savefig(f"{output_dir}/15_pairplot.png", dpi=150, bbox_inches='tight')
    plt.close(g.figure)


def plot_evolucao_emergentes(df, output_dir, titulo):
    """16. Evolução temporal dos emergentes de destaque (PIB + Governança)."""
    if 'pib_per_capita_ppc' not in df.columns:
        return
    g_cols = [c for c in QUAL_VARS[:6] if c in df.columns]
    
    paises = [p for p in EMERGENTES_DESTAQUE if p in df['country_code'].values][:6]
    if not paises:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # PIB
    for pais in paises:
        dados = df[df['country_code'] == pais].sort_values('year')
        axes[0].plot(dados['year'], dados['pib_per_capita_ppc'], label=pais, linewidth=1.5)
    axes[0].set_title('PIB per capita', fontsize=11)
    axes[0].set_xlabel('Ano')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    
    # Governança média
    if g_cols:
        for pais in paises:
            dados = df[df['country_code'] == pais].sort_values('year')
            media_wgi = dados[g_cols].mean(axis=1)
            axes[1].plot(dados['year'], media_wgi, label=pais, linewidth=1.5)
        axes[1].set_title('Média WGI (Governança)', fontsize=11)
        axes[1].set_xlabel('Ano')
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)
    
    fig.suptitle(f'Evolução — Emergentes de Destaque — {titulo}', fontsize=13)
    plt.tight_layout()
    _salvar(fig, output_dir, '16_evolucao_emergentes')


def gerar_todas_visualizacoes(df, output_dir, titulo):
    """Gera todas as 16 visualizações para um dataset."""
    print(f"\n  ── Gerando Visualizações ({titulo}) ──")
    
    plot_heatmap_missing(df, output_dir, titulo)
    print("    ✅ 01 — Heatmap missing")
    
    plot_histogramas_quant(df, output_dir, titulo)
    print("    ✅ 02 — Histogramas quantitativos")
    
    plot_histogramas_qual(df, output_dir, titulo)
    print("    ✅ 03 — Histogramas qualitativos")
    
    plot_boxplots_quant(df, output_dir, titulo)
    print("    ✅ 04 — Boxplots quantitativos")
    
    plot_boxplots_qual(df, output_dir, titulo)
    print("    ✅ 05 — Boxplots qualitativos")
    
    plot_correlacao_completa(df, output_dir, titulo)
    print("    ✅ 06 — Correlação completa")
    
    plot_correlacao_cruzada(df, output_dir, titulo)
    print("    ✅ 07 — Correlação cruzada Quant × Qual")
    
    plot_evolucao_temporal_quant(df, output_dir, titulo)
    print("    ✅ 08 — Evolução temporal quantitativa")
    
    plot_evolucao_temporal_qual(df, output_dir, titulo)
    print("    ✅ 09 — Evolução temporal qualitativa")
    
    plot_top10_pib(df, output_dir, titulo)
    print("    ✅ 10 — Top 10 PIB")
    
    plot_scatter_pib_governanca(df, output_dir, titulo)
    print("    ✅ 11 — Scatter PIB vs Governança")
    
    plot_violin_emprego(df, output_dir, titulo)
    print("    ✅ 12 — Violin emprego industrial")
    
    plot_radar_emergentes(df, output_dir, titulo)
    print("    ✅ 13 — Radar emergentes")
    
    plot_heatmap_paises(df, output_dir, titulo)
    print("    ✅ 14 — Heatmap perfil países")
    
    plot_pairplot(df, output_dir, titulo)
    print("    ✅ 15 — Pairplot")
    
    plot_evolucao_emergentes(df, output_dir, titulo)
    print("    ✅ 16 — Evolução emergentes")
