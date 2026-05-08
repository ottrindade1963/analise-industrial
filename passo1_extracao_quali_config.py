"""Configurações para extração de indicadores qualitativos (WGI) via API."""
import os

# API do Banco Mundial
BASE_URL = "http://api.worldbank.org/v2"
SOURCE_WGI = 3  # Source 3 = Worldwide Governance Indicators

# Indicadores WGI na API (usando Estimate: -2.5 a +2.5)
WGI_INDICATORS = {
    "GOV_WGI_CC.EST": "wgi_control_corruption",
    "GOV_WGI_GE.EST": "wgi_gov_effectiveness",
    "GOV_WGI_PV.EST": "wgi_political_stability",
    "GOV_WGI_RQ.EST": "wgi_regulatory_quality",
    "GOV_WGI_RL.EST": "wgi_rule_law",
    "GOV_WGI_VA.EST": "wgi_voice_accountability"
}

# Período de cobertura
ANO_MINIMO = 1996
ANO_MAXIMO = 2024

# Caminhos de saída
OUTPUT_CSV = "dados_qualitativos.csv"
OUTPUT_XLSX = "dados_qualitativos.xlsx"

# Configurações de requisição
TIMEOUT = 60
DELAY_ENTRE_REQUESTS = 0.2
