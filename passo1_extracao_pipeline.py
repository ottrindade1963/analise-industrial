"""Pipeline principal de extração com geração automática de metadados."""

import os
import pandas as pd
from passo1_extracao_config import DATA_DIR, INDICADORES
from passo1_extracao_processor import obter_paises, filtrar_emergentes, baixar_indicador


def executar():
    """Executa o pipeline completo de extração WDI."""
    os.makedirs(DATA_DIR, exist_ok=True)
    print("=== INICIANDO DOWNLOAD DE DADOS WDI ===")

    # 1. Países
    paises = filtrar_emergentes(obter_paises())
    codigos = paises["codigo_pais"].tolist()
    print(f"✅ {len(codigos)} países em desenvolvimento identificados.")
    paises.to_csv(f"{DATA_DIR}/paises_emergentes.csv", index=False)

    # 2. Indicadores
    df_final = pd.DataFrame()
    for cod, nome in INDICADORES.items():
        print(f"-> Baixando: {nome}...")
        df_ind = baixar_indicador(cod, nome, codigos)
        if df_ind.empty:
            continue
        if df_final.empty:
            df_final = df_ind
        else:
            df_final = pd.merge(df_final, df_ind, on=["pais", "codigo_iso3", "ano"], how="outer")

    # 3. Exportar
    if not df_final.empty:
        df_final.sort_values(["pais", "ano"], inplace=True)
        df_final.to_csv(f"{DATA_DIR}/wdi_emergentes_final.csv", index=False)
        df_final.to_excel(f"{DATA_DIR}/wdi_emergentes_final.xlsx", index=False)
        print(f"\n✅ Concluído! {len(df_final)} registros salvos em '{DATA_DIR}/'")
    else:
        print("\n❌ Nenhum dado coletado.")

    return df_final, codigos

if __name__ == "__main__":
    executar()
