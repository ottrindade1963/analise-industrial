import os

# Configurações de Caminhos
# Detecção automática de ambiente (Colab vs Local)
    # Se não estiver no Colab, usa o diretório atual

DATA_DIR = os.path.join(BASE_DIR, 'dados_engenharia')
MODEL_DIR = os.path.join(BASE_DIR, 'modelos_treinados')
OUTPUT_DIR = os.path.join(BASE_DIR, 'interpretabilidade_shap')

# Garantir que o diretório de saída existe
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Variável Alvo
TARGET_VAR = 'valor_agregado_industrial_percent_pib'

# Modelos Suportados pelo SHAP (TreeExplainer)
MODELS_FOR_SHAP = ['RandomForest', 'XGBoost']

# Datasets e Estratégias
DATASETS = ['nao_agregado', 'inner', 'left', 'outer']
STRATEGIES = ['A1_Direta', 'A2_PCA', 'A3_Interacao']

# Número de features a mostrar nos gráficos
TOP_N_FEATURES = 15
