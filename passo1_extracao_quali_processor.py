"""Processador para extração de dados qualitativos (WGI) via API do Banco Mundial."""

import os
import pandas as pd
import requests
import time
from passo1_extracao_config import BASE_URL, DATA_INICIO, DATA_FIM
from passo1_extracao_processor import obter_paises, filtrar_emergentes

# Mapeamento dos indicadores WGI na API do Banco Mundial
WGI_INDICATORS = {
    "GOV_WGI_CC.EST": "wgi_control_corruption",
    "GOV_WGI_GE.EST": "wgi_gov_effectiveness",
    "GOV_WGI_PV.EST": "wgi_political_stability",
    "GOV_WGI_RQ.EST": "wgi_regulatory_quality",
    "GOV_WGI_RL.EST": "wgi_rule_law",
    "GOV_WGI_VA.EST": "wgi_voice_accountability"
}

def normalizar_wgi(valor):
    """Normaliza o valor WGI de [-2.5, 2.5] para [0, 1]."""
    if pd.isna(valor):
        return None
    # A escala teórica é -2.5 a 2.5, mas na prática pode passar um pouco
    # Limitamos aos extremos teóricos antes de normalizar
    valor = max(-2.5, min(2.5, valor))
    return (valor + 2.5) / 5.0

def baixar_indicador_wgi(indicator_id, indicator_name, codigos_paises):
    """Baixa dados de um indicador WGI para os países informados."""
    all_data = []
    for i in range(0, len(codigos_paises), 40):
        chunk = ";".join(codigos_paises[i:i+40])
        url = f"{BASE_URL}/country/{chunk}/indicator/{indicator_id}"
        params = {"format": "json", "per_page": 2000, "date": f"{DATA_INICIO}:{DATA_FIM}", "source": 3}
        try:
            resp = requests.get(url, params=params, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            if len(data) > 1 and data[1]:
                for e in data[1]:
                    if e["value"] is not None:
                        all_data.append({
                            "country_code": e["countryiso3code"],
                            "year": int(e["date"]),
                            indicator_name: normalizar_wgi(e["value"]),
                        })
            time.sleep(0.2)
        except Exception as err:
            print(f"   ! Erro bloco {i//40+1}: {err}")
    return pd.DataFrame(all_data)

def executar_extracao_wgi():
    """Executa a extração completa dos dados WGI via API."""
    print("=" * 70)
    print("PROCESSANDO WORLDWIDE GOVERNANCE INDICATORS (WGI) VIA API")
    print("=" * 70)
    
    # 1. Obter países emergentes
    paises = filtrar_emergentes(obter_paises())
    codigos = paises["codigo_pais"].tolist()
    print(f"✅ {len(codigos)} países em desenvolvimento identificados.")
    
    # 2. Baixar indicadores
    df_final = pd.DataFrame()
    for cod, nome in WGI_INDICATORS.items():
        print(f"-> Baixando: {nome} ({cod})...")
        df_ind = baixar_indicador_wgi(cod, nome, codigos)
        if df_ind.empty:
            continue
        if df_final.empty:
            df_final = df_ind
        else:
            df_final = pd.merge(df_final, df_ind, on=["country_code", "year"], how="outer")
            
    # 3. Exportar
    if not df_final.empty:
        df_final.sort_values(["country_code", "year"], inplace=True)
        
        # Salvar na raiz do projeto
        df_final.to_csv("dados_qualitativos.csv", index=False)
        df_final.to_excel("dados_qualitativos.xlsx", index=False)
        
        print("\n" + "=" * 70)
        print("RESUMO FINAL")
        print("=" * 70)
        print(f"Painel final: {len(df_final)} linhas (país-ano), {len(df_final.columns)} colunas")
        print(f"Colunas: {list(df_final.columns)}")
        print(f"Países únicos: {df_final['country_code'].nunique()}")
        print(f"Anos cobertos: {df_final['year'].min()} a {df_final['year'].max()}")
        
        print("\nCobertura de dados:")
        for col in WGI_INDICATORS.values():
            if col in df_final.columns:
                validos = df_final[col].notna().sum()
                pct = (validos / len(df_final)) * 100
                print(f"  {col:<35}: {validos:>6} ({pct:>5.1f}%)")
                
        print("\n" + "=" * 70)
        print("✓ PROCESSAMENTO CONCLUÍDO COM SUCESSO!")
        print("=" * 70)
    else:
        print("\n❌ Nenhum dado coletado.")

if __name__ == "__main__":
    executar_extracao_wgi()
