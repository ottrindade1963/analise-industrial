"""Pipeline completo de limpeza e alinhamento de WDI e WGI."""

import os
import pandas as pd
from passo2_1_limpeza_config import DATA_PATH, OUTPUT_DIR, DATA_PATH_WGI
from passo2_1_limpeza_processor import (
    carregar_dados, remover_paises_incompletos, remover_linhas_incompletas,
    imputar_valores, validar_ranges, gerar_relatorio, salvar_dados,
    carregar_dados_wgi, remover_paises_incompletos_wgi, remover_linhas_incompletas_wgi,
    imputar_valores_wgi, validar_ranges_wgi, gerar_relatorio_wgi
)


def alinhar_paises_e_anos(df_wdi, df_wgi):
    """Alinha WDI e WGI para ter os mesmos países e anos."""
    print("\n" + "="*60)
    print("  ALINHANDO PAÍSES E ANOS ENTRE WDI E WGI")
    print("="*60)
    
    # Renomear colunas de chave para padrão
    df_wdi_temp = df_wdi.copy()
    df_wdi_temp = df_wdi_temp.rename(columns={'codigo_iso3': 'country_code', 'ano': 'year'})
    
    # Encontrar países em comum
    paises_wdi = set(df_wdi_temp['country_code'].unique())
    paises_wgi = set(df_wgi['country_code'].unique())
    paises_comuns = paises_wdi & paises_wgi
    
    # Encontrar anos em comum
    anos_wdi = set(df_wdi_temp['year'].unique())
    anos_wgi = set(df_wgi['year'].unique())
    anos_comuns = anos_wdi & anos_wgi
    
    print(f"  Países WDI: {len(paises_wdi)}")
    print(f"  Países WGI: {len(paises_wgi)}")
    print(f"  Países em comum: {len(paises_comuns)}")
    
    print(f"  Anos WDI: {min(anos_wdi)}-{max(anos_wdi)}")
    print(f"  Anos WGI: {min(anos_wgi)}-{max(anos_wgi)}")
    print(f"  Anos em comum: {len(anos_comuns)}")
    
    # Filtrar para países e anos em comum
    df_wdi_alinhado = df_wdi_temp[
        (df_wdi_temp['country_code'].isin(paises_comuns)) &
        (df_wdi_temp['year'].isin(anos_comuns))
    ].copy()
    
    df_wgi_alinhado = df_wgi[
        (df_wgi['country_code'].isin(paises_comuns)) &
        (df_wgi['year'].isin(anos_comuns))
    ].copy()
    
    print(f"\n  ✅ WDI após alinhamento: {df_wdi_alinhado.shape[0]} linhas x {len(paises_comuns)} países")
    print(f"  ✅ WGI após alinhamento: {df_wgi_alinhado.shape[0]} linhas x {len(paises_comuns)} países")
    
    # Restaurar nomes de colunas originais para WDI
    df_wdi_alinhado = df_wdi_alinhado.rename(columns={'country_code': 'codigo_iso3', 'year': 'ano'})
    
    return df_wdi_alinhado, df_wgi_alinhado


def executar_limpeza():
    """Executa o pipeline completo de limpeza para WDI e WGI."""
    
    print("="*60)
    print("  LIMPEZA E TRATAMENTO DE DADOS")
    print("  WDI (Quantitativo) + WGI (Qualitativo)")
    print("="*60)
    
    # ========== PASSO 1: CARREGAR DADOS BRUTOS ==========
    print("\n[PASSO 1] Carregando dados brutos...")
    df_wdi = carregar_dados(DATA_PATH)
    df_wgi = carregar_dados_wgi(DATA_PATH_WGI)
    
    # ========== PASSO 2: ALINHAR PAÍSES E ANOS ==========
    df_wdi, df_wgi = alinhar_paises_e_anos(df_wdi, df_wgi)
    
    # Salvar referência para relatório
    df_wdi_alinhado = df_wdi.copy()
    
    # ========== PASSO 3: LIMPAR WDI ==========
    print("\n[PASSO 3] Limpando WDI...")
    df_wdi = remover_paises_incompletos(df_wdi)
    df_wdi = remover_linhas_incompletas(df_wdi)
    df_wdi = imputar_valores(df_wdi)
    df_wdi = validar_ranges(df_wdi)
    stats_wdi = gerar_relatorio(df_wdi_alinhado, df_wdi)
    
    # ========== PASSO 4: LIMPAR WGI ==========
    print("\n[PASSO 4] Limpando WGI...")
    df_wgi_original = df_wgi.copy()  # Guardar original para relatório
    df_wgi = remover_paises_incompletos_wgi(df_wgi)
    df_wgi = remover_linhas_incompletas_wgi(df_wgi)
    df_wgi = imputar_valores_wgi(df_wgi)
    df_wgi = validar_ranges_wgi(df_wgi)
    stats_wgi = gerar_relatorio_wgi(df_wgi_original, df_wgi)
    
    # ========== PASSO 5: ALINHAMENTO FINAL ==========
    print("\n[PASSO 5] Alinhamento final após limpeza...")
    
    # Garantir que WDI e WGI têm exatamente os mesmos países e anos
    paises_wdi_final = set(df_wdi['codigo_iso3'].unique())
    paises_wgi_final = set(df_wgi['country_code'].unique())
    paises_comuns_final = paises_wdi_final & paises_wgi_final
    
    # Manter apenas países em comum
    df_wdi = df_wdi[df_wdi['codigo_iso3'].isin(paises_comuns_final)]
    df_wgi = df_wgi[df_wgi['country_code'].isin(paises_comuns_final)]
    
    # Garantir que apenas combinações país-ano em comum são mantidas
    wdi_pais_ano = set(zip(df_wdi['codigo_iso3'], df_wdi['ano']))
    wgi_pais_ano = set(zip(df_wgi['country_code'], df_wgi['year']))
    pais_ano_comuns = wdi_pais_ano & wgi_pais_ano
    
    # Filtrar para manter apenas combinações em comum
    df_wdi['_pais_ano'] = list(zip(df_wdi['codigo_iso3'], df_wdi['ano']))
    df_wdi = df_wdi[df_wdi['_pais_ano'].isin(pais_ano_comuns)].drop(columns=['_pais_ano'])
    
    df_wgi['_pais_ano'] = list(zip(df_wgi['country_code'], df_wgi['year']))
    df_wgi = df_wgi[df_wgi['_pais_ano'].isin(pais_ano_comuns)].drop(columns=['_pais_ano'])
    
    print(f"  WDI após alinhamento final: {df_wdi.shape[0]} linhas x {len(paises_comuns_final)} países")
    print(f"  WGI após alinhamento final: {df_wgi.shape[0]} linhas x {len(paises_comuns_final)} países")
    print(f"  Combinações país-ano em comum: {len(pais_ano_comuns)}")
    print(f"  ✅ WDI e WGI agora têm EXATAMENTE os mesmos dados!" if df_wdi.shape[0] == df_wgi.shape[0] else f"  ⚠️  Diferença: {abs(df_wdi.shape[0] - df_wgi.shape[0])} linhas")
    
    # ========== PASSO 6: SALVAR DADOS LIMPOS ==========
    print("\n[PASSO 6] Salvando dados limpos...")
    
    # Criar diretório se não existir
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Salvar WDI limpo
    print("💾 Salvando dados limpos...")
    csv_wdi = os.path.join(OUTPUT_DIR, "wdi_emergentes_limpo.csv")
    xlsx_wdi = os.path.join(OUTPUT_DIR, "wdi_emergentes_limpo.xlsx")
    df_wdi.to_csv(csv_wdi, index=False)
    df_wdi.to_excel(xlsx_wdi, index=False)
    print(f"  ✅ CSV: {csv_wdi}")
    print(f"  ✅ XLSX: {xlsx_wdi}")
    
    # Salvar WGI limpo
    print("💾 Salvando dados WGI limpos...")
    csv_wgi = os.path.join(OUTPUT_DIR, "wgi_emergentes_limpo.csv")
    xlsx_wgi = os.path.join(OUTPUT_DIR, "wgi_emergentes_limpo.xlsx")
    df_wgi.to_csv(csv_wgi, index=False)
    df_wgi.to_excel(xlsx_wgi, index=False)
    print(f"  ✅ CSV: {csv_wgi}")
    print(f"  ✅ XLSX: {xlsx_wgi}")
    
    print("\n" + "="*60)
    print("  ✅ LIMPEZA CONCLUÍDA COM SUCESSO!")
    print(f"  📁 Ficheiros gerados em: {OUTPUT_DIR}/")
    print("="*60)
    
    return df_wdi, df_wgi, stats_wdi, stats_wgi


if __name__ == "__main__":
    executar_limpeza()
