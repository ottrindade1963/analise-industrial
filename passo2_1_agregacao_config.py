"""Configurações para agregação de datasets quantitativo e qualitativo."""
import os

# Caminhos dos datasets de entrada (raiz do repositório)
BASE_DIR = os.path.dirname(__file__)
QUANT_PATH = os.path.join(BASE_DIR, "dados_limpos", "wdi_emergentes_limpo.csv")
QUAL_PATH = os.path.join(BASE_DIR, "dados_limpos", "wgi_emergentes_limpo.csv")

# Chaves de junção
QUANT_KEY_PAIS = "codigo_iso3"
QUANT_KEY_ANO = "ano"
QUAL_KEY_PAIS = "country_code"
QUAL_KEY_ANO = "year"

# Nomes padronizados após junção
KEY_PAIS = "country_code"
KEY_ANO = "year"

# Colunas quantitativas (excluindo chaves)
QUANT_COLS = [
    'pais',
    'pib_per_capita_ppc',
    'formacao_bruta_capital_fixo_percent_pib',
    'matricula_ensino_secundario_percent',
    'comercio_percent_pib',
    'investimento_estrangeiro_direto_percent_pib',
    'populacao_total',
    'emprego_industria_percent_emprego_total',
    'valor_agregado_industrial_percent_pib'
]

# Colunas qualitativas (excluindo chaves)
QUAL_COLS = [
    'wgi_control_corruption',
    'wgi_gov_effectiveness',
    'wgi_political_stability',
    'wgi_regulatory_quality',
    'wgi_rule_law',
    'wgi_voice_accountability'
]

# Pastas de saída
OUTPUT_DIR_M1 = os.path.join(BASE_DIR, "agregado_metodo1_inner")
OUTPUT_DIR_M2 = os.path.join(BASE_DIR, "agregado_metodo2_left_imputado")
OUTPUT_DIR_M3 = os.path.join(BASE_DIR, "agregado_metodo3_outer_completo")

for d in [OUTPUT_DIR_M1, OUTPUT_DIR_M2, OUTPUT_DIR_M3]:
    os.makedirs(d, exist_ok=True)
