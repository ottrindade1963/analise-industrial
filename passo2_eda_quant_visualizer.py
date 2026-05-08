"""Funções de visualização para análise exploratória."""
import matplotlib
matplotlib.use('Agg')  # Backend não-interativo para Colab
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from passo2_eda_quant_config import COLUNAS_NUMERICAS, NOMES_CURTOS, OUTPUT_DIR

sns.set_theme(style="whitegrid", palette="Set2", font_scale=1.1)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _salvar(fig, nome):
    fig.savefig(f"{OUTPUT_DIR}/{nome}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  💾 Salvo: {OUTPUT_DIR}/{nome}.png")


# ── 1. Mapa de calor de valores ausentes ──
def plot_missing(df):
    print("\n🔍 1. Mapa de valores ausentes...")
    df2 = df.copy()
    df2["decada"] = (df2["ano"] // 10) * 10
    missing = df2.groupby("decada")[COLUNAS_NUMERICAS].apply(lambda x: x.isnull().mean() * 100)
    missing.columns = [NOMES_CURTOS[c] for c in missing.columns]
    fig, ax = plt.subplots(figsize=(12, 4))
    sns.heatmap(missing, annot=True, fmt=".0f", cmap="YlOrRd", ax=ax)
    ax.set_title("Valores Ausentes por Década (%)")
    _salvar(fig, "01_missing_heatmap")


# ── 2. Histogramas ──
def plot_histogramas(df):
    print("📊 2. Histogramas...")
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    for ax, col in zip(axes.flat, COLUNAS_NUMERICAS):
        df[col].dropna().hist(bins=40, ax=ax, edgecolor="white", alpha=0.7)
        ax.set_title(NOMES_CURTOS[col], fontsize=10)
    fig.suptitle("Distribuição das Variáveis", fontsize=14, y=1.02)
    fig.tight_layout()
    _salvar(fig, "02_histogramas")


# ── 3. Boxplots ──
def plot_boxplots(df):
    print("📦 3. Boxplots...")
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    for ax, col in zip(axes.flat, COLUNAS_NUMERICAS):
        sns.boxplot(y=df[col], ax=ax, color="skyblue")
        ax.set_title(NOMES_CURTOS[col], fontsize=10)
    fig.suptitle("Boxplots — Identificação de Outliers", fontsize=14, y=1.02)
    fig.tight_layout()
    _salvar(fig, "03_boxplots")


# ── 4. Matriz de correlação ──
def plot_correlacao(df):
    print("🔗 4. Matriz de correlação...")
    corr = df[COLUNAS_NUMERICAS].corr()
    labels = [NOMES_CURTOS[c] for c in COLUNAS_NUMERICAS]
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title("Matriz de Correlação")
    _salvar(fig, "04_correlacao")


# ── 5. Evolução temporal ──
def plot_evolucao_temporal(df):
    print("📈 5. Evolução temporal...")
    medias = df.groupby("ano")[COLUNAS_NUMERICAS].mean()
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    for ax, col in zip(axes.flat, COLUNAS_NUMERICAS):
        medias[col].plot(ax=ax, marker="o", markersize=3, linewidth=1.5)
        ax.set_title(NOMES_CURTOS[col], fontsize=10)
        ax.set_xlabel("")
    fig.suptitle("Evolução Temporal — Média Global por Ano", fontsize=14, y=1.02)
    fig.tight_layout()
    _salvar(fig, "05_evolucao_temporal")


# ── 6. Top 10 países por PIB ──
def plot_top_paises(df):
    print("🏆 6. Top 10 países...")
    ultimo_ano = df["ano"].max()
    top = (df[df["ano"] == ultimo_ano]
           .nlargest(10, "pib_per_capita_ppc")[["pais", "pib_per_capita_ppc"]])
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=top, y="pais", x="pib_per_capita_ppc", ax=ax, palette="viridis")
    ax.set_title(f"Top 10 — PIB per Capita PPC ({ultimo_ano})")
    ax.set_xlabel("USD (constantes)")
    ax.set_ylabel("")
    _salvar(fig, "06_top10_pib")


# ── 7. Scatter PIB vs Indústria ──
def plot_scatter_pib_industria(df):
    print("🔵 7. Scatter PIB vs Indústria...")
    ultimo_ano = df["ano"].max()
    d = df[df["ano"] == ultimo_ano].dropna(
        subset=["pib_per_capita_ppc", "valor_agregado_industrial_percent_pib"])
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.scatterplot(data=d, x="pib_per_capita_ppc",
                    y="valor_agregado_industrial_percent_pib",
                    size="populacao_total", sizes=(20, 500), alpha=0.6,
                    ax=ax, legend=False)
    ax.set_title(f"PIB per Capita vs Valor Agregado Industrial ({ultimo_ano})")
    ax.set_xlabel("PIB per capita PPC (USD)")
    ax.set_ylabel("Valor Agregado Industrial (% PIB)")
    _salvar(fig, "07_scatter_pib_industria")


# ── 8. Violin plot emprego industrial ──
def plot_violin_emprego(df):
    print("🎻 8. Violin plot emprego industrial...")
    df2 = df.dropna(subset=["emprego_industria_percent_emprego_total"]).copy()
    df2["decada"] = (df2["ano"] // 10 * 10).astype(str) + "s"
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.violinplot(data=df2, x="decada",
                   y="emprego_industria_percent_emprego_total",
                   palette="muted", ax=ax)
    ax.set_title("Emprego na Indústria por Década")
    ax.set_xlabel("")
    ax.set_ylabel("Emprego Indústria (% total)")
    _salvar(fig, "08_violin_emprego")


# ── 9. Pairplot ──
def plot_pairplot(df):
    print("🔷 9. Pairplot...")
    cols = ["pib_per_capita_ppc", "comercio_percent_pib",
            "valor_agregado_industrial_percent_pib",
            "emprego_industria_percent_emprego_total"]
    amostra = df[cols].dropna().sample(min(500, len(df)), random_state=42)
    amostra.columns = [NOMES_CURTOS[c] for c in cols]
    g = sns.pairplot(amostra, diag_kind="kde", plot_kws={"alpha": 0.4, "s": 15})
    g.fig.suptitle("Pairplot — Variáveis-Chave", y=1.02)
    _salvar(g.fig, "09_pairplot")


# ── 10. Heatmap por país ──
def plot_heatmap_paises(df):
    print("🗺️ 10. Heatmap por país...")
    medias = df.groupby("pais")[COLUNAS_NUMERICAS].mean()
    top20 = medias.nlargest(20, "pib_per_capita_ppc")
    norm = (top20 - top20.min()) / (top20.max() - top20.min())
    norm.columns = [NOMES_CURTOS[c] for c in norm.columns]
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(norm, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax)
    ax.set_title("Perfil Normalizado — Top 20 Países por PIB per Capita")
    _salvar(fig, "10_heatmap_paises")
