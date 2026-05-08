"""Processamento e limpeza de dados."""

import pandas as pd
import numpy as np
from passo2_1_limpeza_config import (
    COLUNAS_NUMERICAS, THRESHOLD_MISSING_PAIS, THRESHOLD_MISSING_LINHA,
    RANGES_VALIDOS, METODOS_IMPUTACAO, NOMES_CURTOS
)


def carregar_dados(path):
    """Carrega o CSV."""
    df = pd.read_csv(path)
    print(f"✅ Dados brutos carregados: {df.shape[0]} linhas x {df.shape[1]} colunas")
    return df


def remover_paises_incompletos(df):
    """Remove países com >50% missing em variáveis numéricas."""
    print(f"\n🔍 Fase 1: Remover países com >{THRESHOLD_MISSING_PAIS}% missing...")
    
    miss_por_pais = df.groupby("pais")[COLUNAS_NUMERICAS].apply(
        lambda x: x.isnull().mean().mean() * 100
    )
    paises_ruins = miss_por_pais[miss_por_pais > THRESHOLD_MISSING_PAIS].index.tolist()
    
    if paises_ruins:
        print(f"  Removendo {len(paises_ruins)} país(es):")
        for p in paises_ruins:
            print(f"    - {p}: {miss_por_pais[p]:.1f}% missing")
        df = df[~df["pais"].isin(paises_ruins)]
    
    print(f"  Resultado: {df.shape[0]} linhas")
    return df


def remover_linhas_incompletas(df):
    """Remove linhas com >40% missing em variáveis numéricas."""
    print(f"\n🔍 Fase 2: Remover linhas com >{THRESHOLD_MISSING_LINHA}% missing...")
    
    missing_por_linha = df[COLUNAS_NUMERICAS].isnull().mean(axis=1) * 100
    linhas_ruins = (missing_por_linha > THRESHOLD_MISSING_LINHA).sum()
    
    df = df[missing_por_linha <= THRESHOLD_MISSING_LINHA]
    
    print(f"  Removidas {linhas_ruins} linhas")
    print(f"  Resultado: {df.shape[0]} linhas")
    return df


def imputar_valores(df):
    """Imputa valores ausentes usando técnicas específicas por variável."""
    print(f"\n🔍 Fase 3: Imputar valores ausentes...")
    
    df = df.sort_values(["pais", "ano"]).reset_index(drop=True)
    
    for col in COLUNAS_NUMERICAS:
        metodo = METODOS_IMPUTACAO[col]
        missing_antes = df[col].isnull().sum()
        
        if missing_antes == 0:
            continue
        
        if metodo == "interpolate_linear":
            # Interpolação linear por país
            df[col] = df.groupby("pais")[col].transform(
                lambda x: x.interpolate(method="linear", limit_direction="both")
            )
        
        elif metodo == "forward_backward_fill":
            # Forward fill depois backward fill por país
            df[col] = df.groupby("pais")[col].transform(
                lambda x: x.ffill().bfill()
            )
        
        elif metodo == "media_movel_3anos":
            # Média móvel de 3 anos por país
            df[col] = df.groupby("pais")[col].transform(
                lambda x: x.fillna(x.rolling(window=3, center=True, min_periods=1).mean())
            )
        
        elif metodo == "media_por_decada":
            # Média por país-década
            df["decada"] = (df["ano"] // 10 * 10)
            media_decada = df.groupby(["pais", "decada"])[col].transform("mean")
            df[col] = df[col].fillna(media_decada)
            df = df.drop("decada", axis=1)
        
        missing_depois = df[col].isnull().sum()
        print(f"  {NOMES_CURTOS[col]}: {missing_antes} → {missing_depois} missing")
    
    return df


def validar_ranges(df):
    """Valida se os valores estão dentro de ranges esperados."""
    print(f"\n🔍 Fase 4: Validar ranges...")
    
    problemas = 0
    for col, (minv, maxv) in RANGES_VALIDOS.items():
        fora_range = ((df[col] < minv) | (df[col] > maxv)).sum()
        if fora_range > 0:
            print(f"  ⚠️  {NOMES_CURTOS[col]}: {fora_range} valores fora do range [{minv}, {maxv}]")
            problemas += fora_range
    
    if problemas == 0:
        print(f"  ✅ Todos os valores dentro dos ranges esperados")
    
    return df


def gerar_relatorio(df_original, df_limpo):
    """Gera relatório de limpeza."""
    print(f"\n{'='*60}")
    print(f"  📊 RELATÓRIO DE LIMPEZA")
    print(f"{'='*60}")
    
    print(f"\n  Dados Originais:  {df_original.shape[0]:,} linhas x {df_original.shape[1]} colunas")
    print(f"  Dados Limpos:     {df_limpo.shape[0]:,} linhas x {df_limpo.shape[1]} colunas")
    print(f"  Linhas removidas: {df_original.shape[0] - df_limpo.shape[0]:,} ({(1 - df_limpo.shape[0]/df_original.shape[0])*100:.1f}%)")
    
    print(f"\n  Países originais:  {df_original['pais'].nunique()}")
    print(f"  Países após limpeza: {df_limpo['pais'].nunique()}")
    
    print(f"\n  Missing values (original):")
    for col in COLUNAS_NUMERICAS:
        miss = df_original[col].isnull().sum()
        print(f"    {NOMES_CURTOS[col]}: {miss} ({miss/len(df_original)*100:.1f}%)")
    
    print(f"\n  Missing values (limpo):")
    for col in COLUNAS_NUMERICAS:
        miss = df_limpo[col].isnull().sum()
        print(f"    {NOMES_CURTOS[col]}: {miss} ({miss/len(df_limpo)*100:.1f}%)")
    
    return {
        "linhas_originais": df_original.shape[0],
        "linhas_limpas": df_limpo.shape[0],
        "linhas_removidas": df_original.shape[0] - df_limpo.shape[0],
        "paises_originais": df_original['pais'].nunique(),
        "paises_limpas": df_limpo['pais'].nunique(),
    }


def salvar_dados(df, output_dir):
    """Salva dados em CSV e XLSX."""
    print(f"\n💾 Salvando dados limpos...")
    
    csv_path = f"{output_dir}/wdi_emergentes_limpo.csv"
    xlsx_path = f"{output_dir}/wdi_emergentes_limpo.xlsx"
    
    df.to_csv(csv_path, index=False)
    print(f"  ✅ CSV: {csv_path}")
    
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Dados", index=False)
        
        # Adicionar metadados em segunda aba
        metadados = pd.DataFrame({
            "Metadado": ["Total de linhas", "Total de colunas", "Período", "Países", "Data de limpeza"],
            "Valor": [
                len(df),
                len(df.columns),
                f"{df['ano'].min()}-{df['ano'].max()}",
                df['pais'].nunique(),
                pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            ]
        })
        metadados.to_excel(writer, sheet_name="Metadados", index=False)
    
    print(f"  ✅ XLSX: {xlsx_path}")
    
    return csv_path, xlsx_path


# ========== FUNÇÕES PARA LIMPEZA DE WGI ==========

COLUNAS_WGI = [
    'wgi_control_corruption',
    'wgi_gov_effectiveness',
    'wgi_political_stability',
    'wgi_regulatory_quality',
    'wgi_rule_law',
    'wgi_voice_accountability'
]

NOMES_CURTOS_WGI = {
    'wgi_control_corruption': 'Controle da Corrupção',
    'wgi_gov_effectiveness': 'Efetividade Governamental',
    'wgi_political_stability': 'Estabilidade Política',
    'wgi_regulatory_quality': 'Qualidade Regulatória',
    'wgi_rule_law': 'Estado de Direito',
    'wgi_voice_accountability': 'Voz e Responsabilidade'
}

RANGES_VALIDOS_WGI = {
    col: (-2.5, 2.5) for col in COLUNAS_WGI
}


def carregar_dados_wgi(path):
    """Carrega dados WGI (qualitativos)."""
    df = pd.read_csv(path)
    print(f"✅ Dados WGI brutos carregados: {df.shape[0]} linhas x {df.shape[1]} colunas")
    return df


def remover_paises_incompletos_wgi(df):
    """Remove países WGI com >50% missing."""
    print(f"\n🔍 Fase 1 (WGI): Remover países com >{THRESHOLD_MISSING_PAIS}% missing...")
    
    miss_por_pais = df.groupby("country_code")[COLUNAS_WGI].apply(
        lambda x: x.isnull().mean().mean() * 100
    )
    paises_ruins = miss_por_pais[miss_por_pais > THRESHOLD_MISSING_PAIS].index.tolist()
    
    if paises_ruins:
        print(f"  Removendo {len(paises_ruins)} país(es):")
        for p in paises_ruins:
            print(f"    - {p}: {miss_por_pais[p]:.1f}% missing")
        df = df[~df["country_code"].isin(paises_ruins)]
    else:
        print(f"  Nenhum país removido")
    
    print(f"  Resultado: {df.shape[0]} linhas")
    return df


def remover_linhas_incompletas_wgi(df):
    """Remove linhas WGI com >40% missing."""
    print(f"\n🔍 Fase 2 (WGI): Remover linhas com >{THRESHOLD_MISSING_LINHA}% missing...")
    
    missing_por_linha = df[COLUNAS_WGI].isnull().mean(axis=1) * 100
    linhas_ruins = (missing_por_linha > THRESHOLD_MISSING_LINHA).sum()
    
    df = df[missing_por_linha <= THRESHOLD_MISSING_LINHA]
    
    print(f"  Removidas {linhas_ruins} linhas")
    print(f"  Resultado: {df.shape[0]} linhas")
    return df


def imputar_valores_wgi(df):
    """Imputa valores ausentes em WGI."""
    print(f"\n🔍 Fase 3 (WGI): Imputar valores ausentes...")
    
    df = df.sort_values(["country_code", "year"]).reset_index(drop=True)
    
    for col in COLUNAS_WGI:
        missing_antes = df[col].isnull().sum()
        
        if missing_antes == 0:
            continue
        
        # Interpolação linear por país
        df[col] = df.groupby("country_code")[col].transform(
            lambda x: x.interpolate(method="linear", limit_direction="both")
        )
        
        missing_depois = df[col].isnull().sum()
        print(f"  {NOMES_CURTOS_WGI[col]}: {missing_antes} → {missing_depois} missing")
    
    return df


def validar_ranges_wgi(df):
    """Valida ranges de WGI."""
    print(f"\n🔍 Fase 4 (WGI): Validar ranges...")
    
    problemas = 0
    for col, (minv, maxv) in RANGES_VALIDOS_WGI.items():
        fora_range = ((df[col] < minv) | (df[col] > maxv)).sum()
        if fora_range > 0:
            print(f"  ⚠️  {NOMES_CURTOS_WGI[col]}: {fora_range} valores fora do range [{minv}, {maxv}]")
            problemas += fora_range
    
    if problemas == 0:
        print(f"  ✅ Todos os valores dentro dos ranges esperados")
    
    return df


def gerar_relatorio_wgi(df_original, df_limpo):
    """Gera relatório de limpeza WGI."""
    print(f"\n{'='*60}")
    print(f"  📊 RELATÓRIO DE LIMPEZA WGI")
    print(f"{'='*60}")
    
    print(f"\n  Dados Originais:  {df_original.shape[0]:,} linhas x {df_original.shape[1]} colunas")
    print(f"  Dados Limpos:     {df_limpo.shape[0]:,} linhas x {df_limpo.shape[1]} colunas")
    print(f"  Linhas removidas: {df_original.shape[0] - df_limpo.shape[0]:,} ({(1 - df_limpo.shape[0]/df_original.shape[0])*100:.1f}%)")
    
    pais_col_original = 'country_code' if 'country_code' in df_original.columns else 'codigo_iso3'
    pais_col_limpo = 'country_code' if 'country_code' in df_limpo.columns else 'codigo_iso3'
    
    print(f"\n  Países originais:  {df_original[pais_col_original].nunique()}")
    print(f"  Países após limpeza: {df_limpo[pais_col_limpo].nunique()}")
    
    print(f"\n  Missing values (original):")
    for col in COLUNAS_WGI:
        miss = df_original[col].isnull().sum()
        print(f"    {NOMES_CURTOS_WGI[col]}: {miss} ({miss/len(df_original)*100:.1f}%)")
    
    print(f"\n  Missing values (limpo):")
    for col in COLUNAS_WGI:
        miss = df_limpo[col].isnull().sum()
        print(f"    {NOMES_CURTOS_WGI[col]}: {miss} ({miss/len(df_limpo)*100:.1f}%)")
    
    return {
        "linhas_originais": df_original.shape[0],
        "linhas_limpas": df_limpo.shape[0],
        "linhas_removidas": df_original.shape[0] - df_limpo.shape[0],
        "paises_originais": df_original['country_code'].nunique(),
        "paises_limpas": df_limpo['country_code'].nunique(),
    }
