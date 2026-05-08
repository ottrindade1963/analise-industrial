"""Estatísticas descritivas completas para cada dataset agregado."""

import pandas as pd
import numpy as np
from scipy import stats
from passo2_1_eda_agreg_config import QUANT_VARS, QUAL_VARS, ALL_VARS, LABELS


def resumo_geral(df, nome_dataset):
    """Imprime resumo geral do dataset."""
    print(f"\n{'═' * 70}")
    print(f"  {nome_dataset}")
    print(f"{'═' * 70}")
    print(f"  Registros: {len(df):,}")
    print(f"  Países: {df['country_code'].nunique()}")
    print(f"  Período: {df['year'].min()} - {df['year'].max()}")
    print(f"  Variáveis: {len(df.columns)} ({len(QUANT_VARS)} quant. + {len(QUAL_VARS)} qual.)")
    if 'fonte_dados' in df.columns:
        print(f"\n  Distribuição por fonte:")
        for fonte, n in df['fonte_dados'].value_counts().items():
            print(f"    • {fonte}: {n:,} ({n/len(df)*100:.1f}%)")


def analise_missing(df, nome_dataset):
    """Analisa valores ausentes."""
    print(f"\n  ── Valores Ausentes ({nome_dataset}) ──")
    cols = [c for c in ALL_VARS if c in df.columns]
    missing = df[cols].isnull().sum()
    total = len(df)
    tabela = pd.DataFrame({
        'Variável': [LABELS.get(c, c) for c in cols],
        'Ausentes': missing.values,
        '% Ausentes': [f"{v/total*100:.1f}%" for v in missing.values]
    })
    print(tabela.to_string(index=False))
    print(f"\n  Total missing: {df[cols].isnull().sum().sum():,} valores")


def estatisticas_descritivas(df, nome_dataset):
    """Estatísticas descritivas completas: N, Média, Mediana, Moda, DP, Var, Q1, Q3, IQR, etc."""
    print(f"\n  ── Estatísticas Descritivas ({nome_dataset}) ──")
    cols = [c for c in ALL_VARS if c in df.columns]
    
    resultados = []
    for col in cols:
        s = df[col].dropna()
        if len(s) == 0:
            continue
        moda = s.mode().iloc[0] if len(s.mode()) > 0 else np.nan
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        resultados.append({
            'Variável': LABELS.get(col, col),
            'N': int(len(s)),
            'Média': f"{s.mean():.4f}",
            'Mediana': f"{s.median():.4f}",
            'Moda': f"{moda:.4f}",
            'DP': f"{s.std():.4f}",
            'Variância': f"{s.var():.4f}",
            'Mín': f"{s.min():.4f}",
            'Q1': f"{q1:.4f}",
            'Q3': f"{q3:.4f}",
            'Máx': f"{s.max():.4f}",
            'IQR': f"{q3-q1:.4f}",
            'Amplitude': f"{s.max()-s.min():.4f}",
            'Assimetria': f"{s.skew():.4f}",
            'Curtose': f"{s.kurtosis():.4f}",
            'CV (%)': f"{(s.std()/s.mean()*100):.1f}" if s.mean() != 0 else "N/A"
        })
    
    tabela = pd.DataFrame(resultados)
    print(tabela.to_string(index=False))
    return tabela


def intervalos_confianca(df, nome_dataset, confianca=0.95):
    """Intervalos de confiança para cada variável."""
    print(f"\n  ── Intervalos de Confiança {int(confianca*100)}% ({nome_dataset}) ──")
    cols = [c for c in ALL_VARS if c in df.columns]
    
    resultados = []
    for col in cols:
        s = df[col].dropna()
        if len(s) < 2:
            continue
        media = s.mean()
        erro_padrao = stats.sem(s)
        margem = erro_padrao * stats.t.ppf((1 + confianca) / 2, len(s) - 1)
        resultados.append({
            'Variável': LABELS.get(col, col),
            'Média': f"{media:.4f}",
            'Erro Padrão': f"{erro_padrao:.4f}",
            'IC Inferior': f"{media - margem:.4f}",
            'IC Superior': f"{media + margem:.4f}",
            'Margem': f"{margem:.4f}"
        })
    
    tabela = pd.DataFrame(resultados)
    print(tabela.to_string(index=False))


def testes_normalidade(df, nome_dataset):
    """Testes de normalidade (Shapiro-Wilk e D'Agostino)."""
    print(f"\n  ── Testes de Normalidade ({nome_dataset}) ──")
    cols = [c for c in ALL_VARS if c in df.columns]
    
    resultados = []
    for col in cols:
        s = df[col].dropna()
        if len(s) < 20:
            continue
        # Shapiro (amostra máx 5000)
        amostra = s.sample(min(5000, len(s)), random_state=42)
        stat_shap, p_shap = stats.shapiro(amostra)
        # D'Agostino
        stat_dag, p_dag = stats.normaltest(s)
        resultados.append({
            'Variável': LABELS.get(col, col),
            'Shapiro W': f"{stat_shap:.4f}",
            'Shapiro p': f"{p_shap:.4e}",
            "D'Agostino K²": f"{stat_dag:.4f}",
            "D'Agostino p": f"{p_dag:.4e}",
            'Normal?': 'Sim' if (p_shap > 0.05 and p_dag > 0.05) else 'Não'
        })
    
    tabela = pd.DataFrame(resultados)
    print(tabela.to_string(index=False))


def correlacao_quant_qual(df, nome_dataset):
    """Correlação cruzada entre variáveis quantitativas e qualitativas."""
    print(f"\n  ── Correlação Quant. × Qual. ({nome_dataset}) ──")
    q_cols = [c for c in QUANT_VARS if c in df.columns]
    g_cols = [c for c in QUAL_VARS if c in df.columns]
    
    if not q_cols or not g_cols:
        print("  Dados insuficientes para correlação cruzada.")
        return None
    
    corr = df[q_cols + g_cols].corr(method='pearson')
    # Extrair apenas cruzamento quant × qual
    cross = corr.loc[q_cols, g_cols]
    cross.index = [LABELS.get(c, c) for c in cross.index]
    cross.columns = [LABELS.get(c, c) for c in cross.columns]
    print(cross.round(3).to_string())
    return cross


def estatisticas_por_decada(df, nome_dataset):
    """Estatísticas por década."""
    print(f"\n  ── Estatísticas por Década ({nome_dataset}) ──")
    df_temp = df.copy()
    df_temp['decada'] = (df_temp['year'] // 10) * 10
    
    cols = [c for c in ALL_VARS if c in df.columns]
    for col in cols[:5]:  # Top 5 variáveis para não sobrecarregar
        s = df_temp.groupby('decada')[col].agg(['count', 'mean', 'median', 'std', 'min', 'max'])
        s = s.round(3)
        print(f"\n  {LABELS.get(col, col)}:")
        print(f"  {s.to_string()}")
