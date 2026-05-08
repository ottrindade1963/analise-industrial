import os

# Configurações de Caminhos
# Detecção automática de ambiente (Colab vs Local)

DATA_DIR = os.path.join(BASE_DIR, 'dados_engenharia')
MODEL_DIR = os.path.join(BASE_DIR, 'modelos_treinados')
OUTPUT_DIR = os.path.join(BASE_DIR, 'resultados_avaliacao')

# Garantir que o diretório de saída existe
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Variável Alvo
TARGET_VAR = 'valor_agregado_industrial_percent_pib'

# A divisão temporal é feita por ano (importada de passo4_model_train_config)
# TRAIN_END_YEAR = 2014 (1995-2014)
# VAL_END_YEAR = 2017 (2015-2017)
# TEST: 2018-2023

# Modelos, Datasets e Estratégias
MODELS = ['RandomForest', 'XGBoost', 'SARIMAX', 'LSTM', 'TFT']
DATASETS = ['nao_agregado', 'inner', 'left', 'outer']
STRATEGIES = ['A1_Direta', 'A2_PCA', 'A3_Interacao']
