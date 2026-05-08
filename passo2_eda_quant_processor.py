"""Carregamento dos dados e estatísticas descritivas completas."""

import pandas as pd
import numpy as np
from scipy import stats
from passo2_eda_quant_config import DATA_PATH, COLUNAS_NUMERICAS, NOMES_CURTOS

try:
    from IPython.display import display
except ImportError:
    def display(obj):
        print(obj if isinstance(obj, str) else obj.to_string())


def carregar_dados():
    """Carrega o CSV e retorna o DataFrame."""
    df = pd.read_csv(DATA_PATH)
    print(f"✅ Dados carregados: {df.shape[0]} linhas x {df.shape[1]} colunas")
    return df


def resumo_geral(df):
    """Imprime resumo geral do dataset."""
    print(f"\n{'='*60}")
    print(f"  📋 RESUMO GERAL DO DATASET")
    print(f"{'='*60}")
    print(f"  Período: {df['ano'].min()} – {df['ano'].max()}")
    print(f"  Países: {df['pais'].nunique()}")
    print(f"  Registros: {len(df)}")


def tabela_missing(df):
    """Tabela detalhada de valores ausentes."""
    print(f"\n{'='*60}")
    print(f"  🔍 VALORES AUSENTES")
    print(f"{'='*60}")
    total = df[COLUNAS_NUMERICAS].isnull().sum()
    percent = (df[COLUNAS_NUMERICAS].isnull().mean() * 100).round(1)
    tab = pd.DataFrame({
        "Variável": [NOMES_CURTOS[c] for c in COLUNAS_NUMERICAS],
        "Ausentes": total.values,
        "% Ausentes": percent.values,
        "Válidos": (len(df) - total).values,
    })
    display(tab.to_string(index=False))
    return percent


def estatisticas_descritivas(df):
    """Tabela completa: média, mediana, desvio-padrão, variância,
    mínimo, máximo, quartis, IQR, assimetria, curtose, CV."""
    print(f"\n{'='*60}")
    print(f"  📊 ESTATÍSTICAS DESCRITIVAS COMPLETAS")
    print(f"{'='*60}")

    linhas = []
    for col in COLUNAS_NUMERICAS:
        s = df[col].dropna()
        n = len(s)
        media = s.mean()
        mediana = s.median()
        moda_val = s.mode().iloc[0] if not s.mode().empty else np.nan
        dp = s.std()
        var = s.var()
        minimo = s.min()
        maximo = s.max()
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        amplitude = maximo - minimo
        assimetria = s.skew()
        curtose = s.kurtosis()
        cv = (dp / media * 100) if media != 0 else np.nan

        linhas.append({
            "Variável": NOMES_CURTOS[col],
            "N": n,
            "Média": round(media, 2),
            "Mediana": round(mediana, 2),
            "Moda": round(moda_val, 2),
            "Desvio-Padrão": round(dp, 2),
            "Variância": round(var, 2),
            "Mínimo": round(minimo, 2),
            "Q1 (25%)": round(q1, 2),
            "Q3 (75%)": round(q3, 2),
            "Máximo": round(maximo, 2),
            "IQR": round(iqr, 2),
            "Amplitude": round(amplitude, 2),
            "Assimetria": round(assimetria, 2),
            "Curtose": round(curtose, 2),
            "CV (%)": round(cv, 1),
        })

    tab = pd.DataFrame(linhas)
    display(tab)
    return tab


def intervalos_confianca(df, confianca=0.95):
    """Intervalos de confiança para a média de cada variável."""
    print(f"\n{'='*60}")
    print(f"  📐 INTERVALOS DE CONFIANÇA ({int(confianca*100)}%) PARA A MÉDIA")
    print(f"{'='*60}")

    linhas = []
    for col in COLUNAS_NUMERICAS:
        s = df[col].dropna()
        n = len(s)
        media = s.mean()
        erro_padrao = stats.sem(s)
        margem = erro_padrao * stats.t.ppf((1 + confianca) / 2, n - 1)
        li = media - margem
        ls = media + margem

        linhas.append({
            "Variável": NOMES_CURTOS[col],
            "N": n,
            "Média": round(media, 2),
            "Erro Padrão": round(erro_padrao, 2),
            "IC Inferior": round(li, 2),
            "IC Superior": round(ls, 2),
            "Margem": round(margem, 2),
        })

    tab = pd.DataFrame(linhas)
    display(tab)
    return tab


def teste_normalidade(df):
    """Teste de Shapiro-Wilk (amostra) e D'Agostino-Pearson para normalidade."""
    print(f"\n{'='*60}")
    print(f"  🧪 TESTES DE NORMALIDADE")
    print(f"{'='*60}")

    linhas = []
    for col in COLUNAS_NUMERICAS:
        s = df[col].dropna()
        # D'Agostino-Pearson (funciona com n >= 20)
        if len(s) >= 20:
            stat_dag, p_dag = stats.normaltest(s)
        else:
            stat_dag, p_dag = np.nan, np.nan

        # Shapiro-Wilk (amostra de até 5000)
        amostra = s.sample(min(5000, len(s)), random_state=42)
        stat_shap, p_shap = stats.shapiro(amostra)

        linhas.append({
            "Variável": NOMES_CURTOS[col],
            "Shapiro-Wilk (stat)": round(stat_shap, 4),
            "Shapiro-Wilk (p)": f"{p_shap:.2e}",
            "Normal (Shapiro)?": "Sim" if p_shap > 0.05 else "Não",
            "D'Agostino (stat)": round(stat_dag, 4) if not np.isnan(stat_dag) else "N/A",
            "D'Agostino (p)": f"{p_dag:.2e}" if not np.isnan(p_dag) else "N/A",
            "Normal (D'Ag.)?": "Sim" if (not np.isnan(p_dag) and p_dag > 0.05) else "Não",
        })

    tab = pd.DataFrame(linhas)
    display(tab)
    return tab


def estatisticas_por_decada(df):
    """Média e desvio-padrão por década para cada variável."""
    print(f"\n{'='*60}")
    print(f"  📅 ESTATÍSTICAS POR DÉCADA")
    print(f"{'='*60}")

    df2 = df.copy()
    df2["decada"] = (df2["ano"] // 10 * 10).astype(str) + "s"

    for col in COLUNAS_NUMERICAS:
        resumo = df2.groupby("decada")[col].agg(
            ["count", "mean", "median", "std", "min", "max"]
        ).round(2)
        resumo.columns = ["N", "Média", "Mediana", "Desvio-Padrão", "Mín", "Máx"]
        print(f"\n  ▸ {NOMES_CURTOS[col]}:")
        display(resumo)
