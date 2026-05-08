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
# Usar caminhos RELATIVOS para compatibilidade com Colab
DATA_DIR = 'dados_engenharia'  # Lê os dados gerados no Passo 3
OUTPUT_DIR = 'modelos_treinados'

# Garantir que o diretório de saída existe
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Variável Alvo
TARGET_VAR = 'valor_agregado_industrial_percent_pib'

# Configurações de Divisão Temporal
# Dados cobrem 1996-2024 (~28 anos)
TRAIN_END_YEAR = 2016   # Treino: 1996-2016 (~21 anos = 75%)
VAL_END_YEAR = 2019     # Validação: 2017-2019 (3 anos = ~11%)
# Teste: 2020-2024 (5 anos = ~14%)

# Razões de divisão temporal (para cálculo dinâmico)
TRAIN_RATIO = 0.72  # ~72% para treino
VAL_RATIO = 0.11    # ~11% para validação
TEST_RATIO = 0.17   # ~17% para teste

# Modelos a serem treinados
MODELS_TO_TRAIN = ['RandomForest', 'XGBoost', 'SARIMAX', 'LSTM', 'TFT', 'BayesHierarquico']

# Datasets e Estratégias a processar
DATASETS = ['nao_agregado', 'inner', 'left', 'outer']
STRATEGIES = ['A1_Direta', 'A2_PCA', 'A3_Interacao']

# Seed para reprodutibilidade
RANDOM_STATE = 42

# ============================================================
# GRIDS DE HIPERPARÂMETROS ROBUSTOS (Completos)
# ============================================================

# RandomForest: 540 combinações
GRID_RANDOMFOREST = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2'],
    'bootstrap': [True, False]
}

# XGBoost: 50 iterações (grid reduzido para velocidade)
GRID_XGBOOST = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.9],
    'colsample_bytree': [0.7, 0.9]
}

# SARIMAX: Parâmetros fixos (sem grid search)
SARIMAX_PARAMS = {
    'p': 1,
    'd': 1,
    'q': 1,
    'P': 1,
    'D': 1,
    'Q': 1,
    's': 12  # Sazonalidade anual
}

# LSTM (MLPRegressor): 30 iterações
GRID_LSTM = {
    'hidden_layer_sizes': [(100,), (100, 50), (200, 100), (150, 75, 50)],
    'activation': ['relu', 'tanh'],
    'learning_rate_init': [0.001, 0.01],
    'alpha': [0.0001, 0.001],
    'batch_size': [32, 64],
    'max_iter': [500, 1000]
}

# TFT (GradientBoosting): 40 iterações
GRID_TFT = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'subsample': [0.7, 0.9],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# Configurações de validação cruzada
CV_FOLDS = 5
SCORING = 'neg_mean_squared_error'

# ============================================================
# BAYESIANO HIERÁRQUICO (PyMC)
# ============================================================
BAYESIAN_N_SAMPLES = 2000   # Amostras MCMC após burn-in
BAYESIAN_N_TUNE = 1000      # Amostras de burn-in (tuning)
BAYESIAN_TARGET_ACCEPT = 0.9  # Taxa de aceitação NUTS
BAYESIAN_MAX_FEATURES = 5   # Máximo de features (estabilidade)
BAYESIAN_CHAINS = 2         # Número de cadeias MCMC
