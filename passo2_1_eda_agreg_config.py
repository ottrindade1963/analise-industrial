"""Configurações para análise exploratória dos datasets agregados."""
import os

BASE_DIR = os.path.dirname(__file__)

# Caminhos dos 3 datasets agregados
DATASETS = {
    "metodo1_inner": {
        "nome": "Método 1 — Inner Join",
        "path": os.path.join(BASE_DIR, "agregado_metodo1_inner", "agregado_inner.csv"),
        "output": os.path.join(BASE_DIR, "eda_metodo1_inner"),
    },
    "metodo2_left": {
        "nome": "Método 2 — Left Join + Imputação",
        "path": os.path.join(BASE_DIR, "agregado_metodo2_left_imputado", "agregado_left_imputado.csv"),
        "output": os.path.join(BASE_DIR, "eda_metodo2_left"),
    },
    "metodo3_outer": {
        "nome": "Método 3 — Outer Join + Fonte",
        "path": os.path.join(BASE_DIR, "agregado_metodo3_outer_completo", "agregado_outer_completo.csv"),
        "output": os.path.join(BASE_DIR, "eda_metodo3_outer"),
    },
}

# Criar pastas de saída
for d in DATASETS.values():
    os.makedirs(d["output"], exist_ok=True)

# Colunas quantitativas
QUANT_VARS = [
    'pib_per_capita_ppc',
    'formacao_bruta_capital_fixo_percent_pib',
    'matricula_ensino_secundario_percent',
    'comercio_percent_pib',
    'investimento_estrangeiro_direto_percent_pib',
    'populacao_total',
    'emprego_industria_percent_emprego_total',
    'valor_agregado_industrial_percent_pib'
]

# Colunas qualitativas (governança)
QUAL_VARS = [
    'wgi_control_corruption',
    'wgi_gov_effectiveness',
    'wgi_political_stability',
    'wgi_regulatory_quality',
    'wgi_rule_law',
    'wgi_voice_accountability'
]

# Todas as variáveis numéricas
ALL_VARS = QUANT_VARS + QUAL_VARS

# Labels curtos para gráficos
LABELS = {
    'pib_per_capita_ppc': 'PIB per capita',
    'formacao_bruta_capital_fixo_percent_pib': 'Form. Capital (%PIB)',
    'matricula_ensino_secundario_percent': 'Matrícula Sec. (%)',
    'comercio_percent_pib': 'Comércio (%PIB)',
    'investimento_estrangeiro_direto_percent_pib': 'IDE (%PIB)',
    'populacao_total': 'População',
    'emprego_industria_percent_emprego_total': 'Emprego Ind. (%)',
    'valor_agregado_industrial_percent_pib': 'Valor Agr. Ind. (%PIB)',
    'wgi_control_corruption': 'Controle Corrupção',
    'wgi_gov_effectiveness': 'Efetividade Gov.',
    'wgi_political_stability': 'Estab. Política',
    'wgi_regulatory_quality': 'Qualidade Reg.',
    'wgi_rule_law': 'Estado de Direito',
    'wgi_voice_accountability': 'Voz e Respons.',
    'icrg_qog': 'ICRG (QoG)'
}

# Países emergentes de destaque
EMERGENTES_DESTAQUE = ['BRA', 'IND', 'CHN', 'ZAF', 'MEX', 'IDN', 'TUR', 'EGY', 'NGA', 'VNM']
