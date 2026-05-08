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

# Configurações de Caminhos
# Nomes dos Ficheiros de Entrada (Caminhos RELATIVOS para compatibilidade)
DATASETS = {
    'nao_agregado': 'dados_limpos/wdi_emergentes_limpo.csv',
    'inner': 'agregado_metodo1_inner/agregado_inner.csv',
    'left': 'agregado_metodo2_left_imputado/agregado_left_imputado.csv',
    'outer': 'agregado_metodo3_outer_completo/agregado_outer_completo.csv'
}

# Diretório de saída
OUTPUT_DIR = 'dados_engenharia'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Variável Alvo
TARGET_VAR = 'valor_agregado_industrial_percent_pib'

# Variáveis Qualitativas (Institucionais e Política)
QUALITATIVE_VARS = [
    'wgi_control_corruption', 
    'wgi_gov_effectiveness', 
    'wgi_political_stability', 
    'wgi_regulatory_quality', 
    'wgi_rule_law', 
    'wgi_voice_accountability'
]

# Variáveis Quantitativas (para interações)
QUANTITATIVE_VARS_FOR_INTERACTION = [
    'formacao_bruta_capital_fixo_percent_pib', # Formação Bruta de Capital Fixo
    'investimento_estrangeiro_direto_percent_pib', # Investimento Estrangeiro Direto
    'comercio_percent_pib' # Abertura Comercial
]
