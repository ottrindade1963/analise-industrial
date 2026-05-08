"""Funções de extração de dados da API do Banco Mundial."""

import pandas as pd
import requests
import time
from passo1_extracao_config import BASE_URL, DATA_INICIO, DATA_FIM


def obter_paises():
    """Retorna lista de países da API do Banco Mundial."""
    url = f"{BASE_URL}/country"
    resp = requests.get(url, params={"format": "json", "per_page": 300}, timeout=30)
    resp.raise_for_status()
    return resp.json()[1]


def filtrar_emergentes(paises_raw):
    """Filtra apenas países em desenvolvimento (não HIC, não agregados)."""
    df = pd.DataFrame(paises_raw)
    df["region_name"] = df["region"].apply(lambda x: x.get("value") if isinstance(x, dict) else None)
    df["income_code"] = df["incomeLevel"].apply(lambda x: x.get("id") if isinstance(x, dict) else None)
    df.rename(columns={"id": "codigo_pais", "name": "nome_pais"}, inplace=True)

    # Regiões do Banco Mundial para África e Médio Oriente
    regioes_alvo = [
        "Sub-Saharan Africa ", 
        "Middle East, North Africa, Afghanistan & Pakistan"
    ]
    
    return df[
        (df["region_name"].notna()) &
        (df["region_name"].isin(regioes_alvo)) &
        (df["region_name"] != "Aggregates") &
        (df["income_code"].notna()) &
        (df["income_code"] != "HIC") &
        (df["codigo_pais"].str.match(r"^[A-Z]{3}$"))
    ].copy()


def baixar_indicador(indicator_id, indicator_name, codigos_paises):
    """Baixa dados de um indicador para os países informados."""
    all_data = []
    for i in range(0, len(codigos_paises), 40):
        chunk = ";".join(codigos_paises[i:i+40])
        url = f"{BASE_URL}/country/{chunk}/indicator/{indicator_id}"
        params = {"format": "json", "per_page": 2000, "date": f"{DATA_INICIO}:{DATA_FIM}"}

        try:
            resp = requests.get(url, params=params, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            if len(data) > 1 and data[1]:
                for e in data[1]:
                    if e["value"] is not None:
                        all_data.append({
                            "pais": e["country"]["value"],
                            "codigo_iso3": e["countryiso3code"],
                            "ano": int(e["date"]),
                            indicator_name: e["value"],
                        })
            time.sleep(0.2)
        except Exception as err:
            print(f"   ! Erro bloco {i//40+1}: {err}")

    return pd.DataFrame(all_data)
