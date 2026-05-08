"""Configurações para análise exploratória do dataset qualitativo WGI + ICRG."""
import os

# Caminho do dataset (raiz do repositório)
DATA_PATH = os.path.join(os.path.dirname(__file__), "dados_qualitativos.csv")

# Indicadores WGI (escala 0-1, normalizado de -2.5 a +2.5)
WGI_COLS = [
    'wgi_control_corruption',
    'wgi_gov_effectiveness',
    'wgi_political_stability',
    'wgi_regulatory_quality',
    'wgi_rule_law',
    'wgi_voice_accountability'
]

# Indicador ICRG (escala 0-1)
ICRG_COL = None

# Todos os indicadores
ALL_INDICATORS = WGI_COLS

# Nomes legíveis para gráficos
LABELS = {
    'wgi_control_corruption': 'Controle de Corrupção',
    'wgi_gov_effectiveness': 'Efetividade Governamental',
    'wgi_political_stability': 'Estabilidade Política',
    'wgi_regulatory_quality': 'Qualidade Regulatória',
    'wgi_rule_law': 'Estado de Direito',
    'wgi_voice_accountability': 'Voz e Responsabilidade'
}

# Abreviações para gráficos compactos
SHORT_LABELS = {
    'wgi_control_corruption': 'Corrupção',
    'wgi_gov_effectiveness': 'Efetividade',
    'wgi_political_stability': 'Estabilidade',
    'wgi_regulatory_quality': 'Regulação',
    'wgi_rule_law': 'Estado Dir.',
    'wgi_voice_accountability': 'Voz e Resp.'
}

# Classificação de governança (limiares para WGI normalizado 0-1)
GOV_THRESHOLDS = {
    'Fraca': (0.0, 0.30),
    'Baixa': (0.30, 0.45),
    'Média': (0.45, 0.60),
    'Boa': (0.60, 0.75),
    'Forte': (0.75, 1.0)
}

# Décadas para agrupamento
DECADES = {
    '1996-2004': (1996, 2004),
    '2005-2014': (2005, 2014),
    '2015-2024': (2015, 2024)
}

# Pasta de saída
OUTPUT_DIR = "resultados_eda_quali"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Estilo dos gráficos
FIGSIZE_LARGE = (14, 8)
FIGSIZE_MEDIUM = (12, 7)
FIGSIZE_SMALL = (10, 6)
DPI = 150
PALETTE = 'RdYlGn'
PALETTE_SEQ = 'YlOrRd_r'
