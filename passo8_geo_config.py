import os

# Configurações de Caminhos
# Detecção automática de ambiente (Colab vs Local)
    # Se não estiver no Colab, usa o diretório atual

DATA_DIR = os.path.join(BASE_DIR, 'dados_engenharia')
MODEL_DIR = os.path.join(BASE_DIR, 'modelos_treinados')
OUTPUT_DIR = os.path.join(BASE_DIR, 'analise_geografica')

# Garantir que o diretório de saída existe
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Variável Alvo
TARGET_VAR = 'valor_agregado_industrial_percent_pib'

# Configurações de Classificação de Países
# Percentis para dividir em Pobres (0-33%), Médios (33-66%), Ricos (66-100%)
PERCENTILE_LOW = 0.33
PERCENTILE_HIGH = 0.66

# Datasets e Estratégias a analisar
DATASETS = ['nao_agregado', 'inner', 'left', 'outer']
STRATEGIES = ['A1_Direta', 'A2_PCA', 'A3_Interacao']
MODELS = ['RandomForest', 'XGBoost', 'SARIMAX', 'LSTM', 'TFT']
