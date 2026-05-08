import os
import sys

# Detectar se está no Colab
IN_COLAB = 'google.colab' in sys.modules

# Detectar o diretório raiz dinamicamente
if IN_COLAB:
    import glob
    # Encontrar o diretório do repositório clonado
    dirs = [d for d in os.listdir('/content') if os.path.isdir(f'/content/{d}') and d not in ['.config', 'sample_data']]
    REPO_DIR = dirs[0] if dirs else os.getcwd()
    os.chdir(REPO_DIR)
    BASE_DIR = REPO_DIR
else:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

"""Configurações do projeto."""
BASE_URL = "https://api.worldbank.org/v2"
DATA_DIR = "data/raw"

INDICADORES = {
    "NY.GDP.PCAP.PP.KD": "pib_per_capita_ppc",
    "NE.GDI.FTOT.ZS": "formacao_bruta_capital_fixo_percent_pib",
    "SE.SEC.ENRR": "matricula_ensino_secundario_percent",
    "NE.TRD.GNFS.ZS": "comercio_percent_pib",
    "BX.KLT.DINV.WD.GD.ZS": "investimento_estrangeiro_direto_percent_pib",
    "SP.POP.TOTL": "populacao_total",
    "SL.IND.EMPL.ZS": "emprego_industria_percent_emprego_total",
    "NV.IND.TOTL.ZS": "valor_agregado_industrial_percent_pib",
}

DATA_INICIO = 1996
DATA_FIM = 2023

# Países emergentes de África e Médio Oriente
PAISES_EMERGENTES = [
    'DZA', 'AGO', 'BEN', 'BWA', 'BFA', 'BDI', 'CMR', 'CPV', 'CAF', 'TCD', 'COM', 'COG',
    'COD', 'CIV', 'DJI', 'EGY', 'GNQ', 'ERI', 'ETH', 'GAB', 'GMB', 'GHA', 'GIN', 'GNB',
    'KEN', 'LSO', 'LBR', 'LBY', 'MDG', 'MWI', 'MLI', 'MRT', 'MUS', 'MAR', 'MOZ', 'NAM',
    'NER', 'NGA', 'RWA', 'STP', 'SEN', 'SYC', 'SLE', 'SOM', 'ZAF', 'SSD', 'SDN', 'SWZ',
    'TZA', 'TGO', 'TUN', 'UGA', 'ZMB', 'ZWE',
    'BHR', 'IRN', 'IRQ', 'ISR', 'JOR', 'KWT', 'LBN', 'OMN', 'QAT', 'SAU', 'ARE', 'YEM'
]

# Configurações de Saída
os.makedirs(DATA_DIR, exist_ok=True)
