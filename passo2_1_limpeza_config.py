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

"""Configurações para limpeza e tratamento de dados."""
import os

# Caminhos
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "raw", "wdi_emergentes_final.csv")
DATA_PATH_WGI = os.path.join(os.path.dirname(__file__), "dados_qualitativos.csv")
OUTPUT_DIR = "dados_limpos"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Colunas
COLUNAS_NUMERICAS = [
    "pib_per_capita_ppc",
    "formacao_bruta_capital_fixo_percent_pib",
    "matricula_ensino_secundario_percent",
    "comercio_percent_pib",
    "investimento_estrangeiro_direto_percent_pib",
    "populacao_total",
    "emprego_industria_percent_emprego_total",
    "valor_agregado_industrial_percent_pib",
]

# Limiares de qualidade
THRESHOLD_MISSING_PAIS = 50  # Remover países com >50% missing
THRESHOLD_MISSING_LINHA = 40  # Remover linhas com >40% missing
THRESHOLD_MISSING_VARIAVEL = 5  # Mínimo de valores válidos (%)

# Ranges válidos (para validação)
RANGES_VALIDOS = {
    "pib_per_capita_ppc": (0, 100000),
    "formacao_bruta_capital_fixo_percent_pib": (-10, 100),
    "matricula_ensino_secundario_percent": (0, 200),  # Permite >100%
    "comercio_percent_pib": (0, 400),  # Permite >100%
    "investimento_estrangeiro_direto_percent_pib": (-100, 200),  # Permite negativos
    "populacao_total": (0, 2e9),
    "emprego_industria_percent_emprego_total": (0, 100),
    "valor_agregado_industrial_percent_pib": (0, 100),
}

# Métodos de imputação por variável
METODOS_IMPUTACAO = {
    "pib_per_capita_ppc": "interpolate_linear",
    "formacao_bruta_capital_fixo_percent_pib": "media_movel_3anos",
    "matricula_ensino_secundario_percent": "forward_backward_fill",
    "comercio_percent_pib": "interpolate_linear",
    "investimento_estrangeiro_direto_percent_pib": "media_por_decada",
    "populacao_total": "interpolate_linear",
    "emprego_industria_percent_emprego_total": "forward_backward_fill",
    "valor_agregado_industrial_percent_pib": "media_movel_3anos",
}

# Nomes descritivos
NOMES_CURTOS = {
    "pib_per_capita_ppc": "PIB per capita (PPC)",
    "formacao_bruta_capital_fixo_percent_pib": "Form. Bruta Capital (%PIB)",
    "matricula_ensino_secundario_percent": "Matrícula Secundário (%)",
    "comercio_percent_pib": "Comércio (%PIB)",
    "investimento_estrangeiro_direto_percent_pib": "IDE (%PIB)",
    "populacao_total": "População Total",
    "emprego_industria_percent_emprego_total": "Emprego Indústria (%)",
    "valor_agregado_industrial_percent_pib": "Valor Agregado Ind. (%PIB)",
}
