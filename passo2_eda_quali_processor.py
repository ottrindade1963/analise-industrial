"""Carregamento de dados e estatísticas descritivas para indicadores de governança."""

import pandas as pd
import numpy as np
from scipy import stats
from passo2_eda_quali_config import (
    DATA_PATH, WGI_COLS, ICRG_COL, ALL_INDICATORS,
    LABELS, DECADES, GOV_THRESHOLDS
)


def carregar_dados():
    """Carrega o dataset qualitativo e mostra resumo geral."""
    df = pd.read_csv(DATA_PATH)
    
    print("=" * 60)
    print("  RESUMO GERAL DO DATASET")
    print("=" * 60)
    print(f"  Registros:  {len(df):,}")
    print(f"  Variáveis:  {len(df.columns)}")
    print(f"  Países:     {df['country_code'].nunique()}")
    print(f"  Período:    {df['year'].min()} - {df['year'].max()}")
    print(f"  Fonte WGI:  Banco Mundial (6 indicadores)")
    print(f"  Fonte ICRG: QoG Standard Dataset")
    print("=" * 60)
    
    return df


def tabela_missing(df):
    """Tabela detalhada de valores ausentes por indicador."""
    print("\n" + "=" * 60)
    print("  VALORES AUSENTES")
    print("=" * 60)
    
    missing_data = []
    for col in ALL_INDICATORS:
        n_miss = df[col].isnull().sum()
        pct = n_miss / len(df) * 100
        n_valid = df[col].notna().sum()
        missing_data.append({
            'Indicador': LABELS[col],
            'Válidos': n_valid,
            'Ausentes': n_miss,
            '% Ausentes': round(pct, 1)
        })
    
    missing_df = pd.DataFrame(missing_data)
    print(missing_df.to_string(index=False))
    return missing_df


def estatisticas_descritivas(df):
    """Estatísticas descritivas completas para cada indicador."""
    print("\n" + "=" * 60)
    print("  ESTATÍSTICAS DESCRITIVAS COMPLETAS")
    print("=" * 60)
    
    stats_data = []
    for col in ALL_INDICATORS:
        s = df[col].dropna()
        if len(s) == 0:
            continue
        
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        
        row = {
            'Indicador': LABELS[col],
            'N': len(s),
            'Média': round(s.mean(), 4),
            'Mediana': round(s.median(), 4),
            'Moda': round(s.mode().iloc[0], 4) if len(s.mode()) > 0 else None,
            'Desvio Padrão': round(s.std(), 4),
            'Variância': round(s.var(), 4),
            'Mínimo': round(s.min(), 4),
            'Q1 (25%)': round(q1, 4),
            'Q3 (75%)': round(q3, 4),
            'Máximo': round(s.max(), 4),
            'IQR': round(iqr, 4),
            'Amplitude': round(s.max() - s.min(), 4),
            'Assimetria': round(s.skew(), 4),
            'Curtose': round(s.kurtosis(), 4),
            'CV (%)': round(s.std() / s.mean() * 100, 2) if s.mean() != 0 else None
        }
        stats_data.append(row)
    
    stats_df = pd.DataFrame(stats_data)
    print(stats_df.to_string(index=False))
    return stats_df


def intervalos_confianca(df, confianca=0.95):
    """Intervalos de confiança para a média de cada indicador."""
    print("\n" + "=" * 60)
    print(f"  INTERVALOS DE CONFIANÇA ({int(confianca*100)}%)")
    print("=" * 60)
    
    ic_data = []
    for col in ALL_INDICATORS:
        s = df[col].dropna()
        if len(s) < 2:
            continue
        
        media = s.mean()
        erro_padrao = stats.sem(s)
        ic = stats.t.interval(confianca, df=len(s)-1, loc=media, scale=erro_padrao)
        margem = media - ic[0]
        
        ic_data.append({
            'Indicador': LABELS[col],
            'N': len(s),
            'Média': round(media, 4),
            'Erro Padrão': round(erro_padrao, 4),
            'IC Inferior': round(ic[0], 4),
            'IC Superior': round(ic[1], 4),
            'Margem Erro': round(margem, 4)
        })
    
    ic_df = pd.DataFrame(ic_data)
    print(ic_df.to_string(index=False))
    return ic_df


def testes_normalidade(df):
    """Testes de normalidade Shapiro-Wilk e D'Agostino-Pearson."""
    print("\n" + "=" * 60)
    print("  TESTES DE NORMALIDADE")
    print("=" * 60)
    
    norm_data = []
    for col in ALL_INDICATORS:
        s = df[col].dropna()
        if len(s) < 20:
            continue
        
        # Shapiro-Wilk (amostra de 5000 se necessário)
        sample = s.sample(min(5000, len(s)), random_state=42)
        sw_stat, sw_p = stats.shapiro(sample)
        
        # D'Agostino-Pearson
        da_stat, da_p = stats.normaltest(s)
        
        norm_data.append({
            'Indicador': LABELS[col],
            'Shapiro-Wilk': round(sw_stat, 4),
            'p-valor (SW)': f"{sw_p:.2e}",
            'D\'Agostino': round(da_stat, 4),
            'p-valor (DA)': f"{da_p:.2e}",
            'Normal?': 'Sim' if sw_p > 0.05 else 'Não'
        })
    
    norm_df = pd.DataFrame(norm_data)
    print(norm_df.to_string(index=False))
    return norm_df


def classificacao_governanca(df):
    """Classifica os países por nível de governança (média dos WGI)."""
    print("\n" + "=" * 60)
    print("  CLASSIFICAÇÃO DE GOVERNANÇA")
    print("=" * 60)
    
    # Média dos 6 indicadores WGI por país
    media_pais = df.groupby('country_code')[WGI_COLS].mean().mean(axis=1)
    
    classificacao = []
    for nivel, (low, high) in GOV_THRESHOLDS.items():
        paises = media_pais[(media_pais >= low) & (media_pais < high)]
        classificacao.append({
            'Nível': nivel,
            'Faixa': f"{low:.2f} - {high:.2f}",
            'N.º Países': len(paises),
            '% do Total': round(len(paises) / len(media_pais) * 100, 1)
        })
    
    class_df = pd.DataFrame(classificacao)
    print(class_df.to_string(index=False))
    return class_df, media_pais


def estatisticas_por_periodo(df):
    """Estatísticas descritivas agrupadas por período temporal."""
    print("\n" + "=" * 60)
    print("  ESTATÍSTICAS POR PERÍODO")
    print("=" * 60)
    
    for periodo, (ano_ini, ano_fim) in DECADES.items():
        subset = df[(df['year'] >= ano_ini) & (df['year'] <= ano_fim)]
        print(f"\n  --- {periodo} ({len(subset):,} registros) ---")
        
        periodo_stats = []
        for col in WGI_COLS:
            s = subset[col].dropna()
            if len(s) == 0:
                continue
            periodo_stats.append({
                'Indicador': LABELS[col],
                'N': len(s),
                'Média': round(s.mean(), 4),
                'Mediana': round(s.median(), 4),
                'DP': round(s.std(), 4),
                'Mín': round(s.min(), 4),
                'Máx': round(s.max(), 4)
            })
        
        print(pd.DataFrame(periodo_stats).to_string(index=False))


def correlacao_indicadores(df):
    """Matriz de correlação de Pearson e Spearman entre indicadores."""
    print("\n" + "=" * 60)
    print("  CORRELAÇÕES ENTRE INDICADORES")
    print("=" * 60)
    
    # Pearson
    corr_pearson = df[ALL_INDICATORS].corr(method='pearson')
    print("\n  Correlação de Pearson:")
    labels_short = {c: LABELS[c][:20] for c in ALL_INDICATORS}
    corr_display = corr_pearson.rename(index=labels_short, columns=labels_short)
    print(corr_display.round(3).to_string())
    
    # Spearman (mais adequada para dados ordinais/não-normais)
    corr_spearman = df[ALL_INDICATORS].corr(method='spearman')
    print("\n  Correlação de Spearman:")
    corr_display_sp = corr_spearman.rename(index=labels_short, columns=labels_short)
    print(corr_display_sp.round(3).to_string())
    
    return corr_pearson, corr_spearman
