"""Funções de agregação de datasets quantitativo e qualitativo."""

import pandas as pd
import numpy as np
from passo2_1_agregacao_config import (
    QUANT_PATH, QUAL_PATH,
    QUANT_KEY_PAIS, QUANT_KEY_ANO,
    QUAL_KEY_PAIS, QUAL_KEY_ANO,
    KEY_PAIS, KEY_ANO, QUAL_COLS
)


def carregar_datasets():
    """Carrega e padroniza os dois datasets."""
    # Quantitativo
    df_quant = pd.read_csv(QUANT_PATH)
    df_quant = df_quant.rename(columns={
        QUANT_KEY_PAIS: KEY_PAIS,
        QUANT_KEY_ANO: KEY_ANO
    })
    
    # Qualitativo
    df_qual = pd.read_csv(QUAL_PATH)
    df_qual = df_qual.rename(columns={
        QUAL_KEY_PAIS: KEY_PAIS,
        QUAL_KEY_ANO: KEY_ANO
    })
    
    print(f"  Quantitativo: {df_quant.shape[0]:,} registros | {df_quant[KEY_PAIS].nunique()} países")
    print(f"  Qualitativo:  {df_qual.shape[0]:,} registros | {df_qual[KEY_PAIS].nunique()} países")
    
    return df_quant, df_qual


def metodo1_inner_join(df_quant, df_qual):
    """
    MÉTODO 1 — Inner Join (Junção Interna Estrita)
    Retém apenas observações com correspondência em ambos os datasets.
    Fundamentação: Allison (2001), Wooldridge (2010).
    """
    print("\n  ── Método 1: Inner Join ──")
    
    df = pd.merge(df_quant, df_qual, on=[KEY_PAIS, KEY_ANO], how='inner')
    
    # Ordenar
    df = df.sort_values([KEY_PAIS, KEY_ANO]).reset_index(drop=True)
    
    print(f"  Resultado: {df.shape[0]:,} registros | {df[KEY_PAIS].nunique()} países")
    print(f"  Período: {df[KEY_ANO].min()} - {df[KEY_ANO].max()}")
    print(f"  Missing WGI: {df[QUAL_COLS[:6]].isnull().any(axis=1).sum()} linhas com algum NaN")
    
    return df


def metodo2_left_join_imputado(df_quant, df_qual):
    """
    MÉTODO 2 — Left Join com Imputação Temporal
    Preserva todo o dataset quantitativo e imputa lacunas qualitativas.
    Fundamentação: Schafer & Graham (2002), Van Buuren (2018).
    """
    print("\n  ── Método 2: Left Join + Imputação Temporal ──")
    
    # Left join: preserva todas as linhas do quantitativo
    df = pd.merge(df_quant, df_qual, on=[KEY_PAIS, KEY_ANO], how='left')
    
    missing_antes = df[QUAL_COLS].isnull().sum().sum()
    
    # Imputação por interpolação temporal dentro de cada país
    for col in QUAL_COLS:
        df[col] = df.groupby(KEY_PAIS)[col].transform(
            lambda s: s.interpolate(method='linear', limit_direction='both')
        )
    
    missing_depois = df[QUAL_COLS].isnull().sum().sum()
    
    # Ordenar
    df = df.sort_values([KEY_PAIS, KEY_ANO]).reset_index(drop=True)
    
    print(f"  Resultado: {df.shape[0]:,} registros | {df[KEY_PAIS].nunique()} países")
    print(f"  Período: {df[KEY_ANO].min()} - {df[KEY_ANO].max()}")
    print(f"  Missing antes imputação: {missing_antes:,} valores")
    print(f"  Missing após imputação:  {missing_depois:,} valores")
    print(f"  Valores imputados: {missing_antes - missing_depois:,}")
    
    return df


def metodo3_outer_join_rastreavel(df_quant, df_qual):
    """
    MÉTODO 3 — Outer Join com Indicador de Fonte
    Preserva todas as observações de ambos os datasets com rastreabilidade.
    Fundamentação: Rubin (1976), Graham (2009), Enders (2010).
    """
    print("\n  ── Método 3: Outer Join + Indicador de Fonte ──")
    
    # Outer join: preserva tudo
    df = pd.merge(df_quant, df_qual, on=[KEY_PAIS, KEY_ANO], how='outer', indicator=True)
    
    # Criar indicador de fonte legível
    fonte_map = {
        'both': 'ambos',
        'left_only': 'apenas_quantitativo',
        'right_only': 'apenas_qualitativo'
    }
    df['fonte_dados'] = df['_merge'].map(fonte_map)
    df = df.drop(columns=['_merge'])
    
    # Ordenar
    df = df.sort_values([KEY_PAIS, KEY_ANO]).reset_index(drop=True)
    
    # Estatísticas
    contagem_fonte = df['fonte_dados'].value_counts()
    print(f"  Resultado: {df.shape[0]:,} registros | {df[KEY_PAIS].nunique()} países")
    print(f"  Período: {df[KEY_ANO].min()} - {df[KEY_ANO].max()}")
    print(f"  Distribuição por fonte:")
    for fonte, n in contagem_fonte.items():
        print(f"    • {fonte}: {n:,} ({n/len(df)*100:.1f}%)")
    
    return df
