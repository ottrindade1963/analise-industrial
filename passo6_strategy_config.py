import os

# Configurações de Caminhos
# Detecção automática de ambiente (Colab vs Local)

RESULTS_DIR = os.path.join(BASE_DIR, 'resultados_avaliacao') # Lê os resultados do Passo 5
OUTPUT_DIR = os.path.join(BASE_DIR, 'analise_estrategias')

# Garantir que o diretório de saída existe
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Nomes das Estratégias
STRATEGIES = {
    'A1_Direta': 'A1: Inclusão Direta',
    'A2_PCA': 'A2: Fator Latente (PCA)',
    'A3_Interacao': 'A3: Termos de Interação'
}

# Baselines para comparação
BASELINE_DATASET = 'nao_agregado' # Baseline para comparar o ganho de usar dados qualitativos
BASELINE_STRATEGY = 'A1_Direta' # Baseline para comparar o ganho de técnicas avançadas (PCA/Interação)
