"""Configurações da análise exploratória."""
import os

# Caminho do CSV — mesmo diretório dos módulos (raiz do repositório)
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "raw", "wdi_emergentes_final.csv")
OUTPUT_DIR = "resultados_eda"

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
