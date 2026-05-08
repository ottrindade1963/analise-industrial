import os

# Configurações de Caminhos
# Detecção automática de ambiente (Colab vs Local)
    # Se não estiver no Colab, usa o diretório atual

DATA_DIR = os.path.join(BASE_DIR, 'dados_engenharia')
MODEL_DIR = os.path.join(BASE_DIR, 'modelos_treinados')
RESULTS_DIR = os.path.join(BASE_DIR, 'resultados_avaliacao')
OUTPUT_DIR = os.path.join(BASE_DIR, 'analises_avancadas')

# Garantir que o diretório de saída existe
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Variável Alvo
TARGET_VAR = 'valor_agregado_industrial_percent_pib'

# Configurações de Análise de Sensibilidade
# Expandido para incluir mais variáveis relevantes para o desenvolvimento industrial
SENSITIVITY_VARS = [
    'wgi_control_corruption', 
    'wgi_rule_law', 
    'wgi_government_effectiveness',
    'wgi_political_stability',
    'fbc_percent_pib',
    'investimento_estrangeiro_direto_percent_pib',
    'comercio_percent_pib'
]
SENSITIVITY_STEPS = [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3] # Variações percentuais (ex: -30% a +30%)

# Datasets e Modelos a analisar
DATASETS = ['nao_agregado', 'inner', 'left', 'outer']
STRATEGIES = ['A1_Direta', 'A2_PCA', 'A3_Interacao']
MODELS = ['RandomForest', 'XGBoost', 'SARIMAX', 'LSTM', 'TFT']
