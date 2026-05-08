"""
Passo 4 - Visualizador de Treinamento de Modelos
==================================================
Gera 3 categorias de visualizações:
  1. Comparativas de Métricas GLOBAIS (R², RMSE, MAE entre modelos/datasets)
  2. Comparativas de 10 PAÍSES selecionados (Angola, Gana, Quênia, etc.)
  3. MAPAS GEOGRÁFICOS com R² por país para cada modelo
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import passo4_model_train_config as config

# 10 países selecionados para análise detalhada
PAISES_SELECIONADOS = [
    'Angola', 'Ghana', 'Kenya', 'Nigeria', 'Mali',
    'Tanzania', 'Egypt', 'Pakistan', 'Iran', 'Afghanistan'
]

# Nomes em português para labels nos gráficos
PAISES_LABELS = {
    'Angola': 'Angola', 'Ghana': 'Gana', 'Kenya': 'Quênia',
    'Nigeria': 'Nigéria', 'Mali': 'Mali', 'Tanzania': 'Tanzânia',
    'Egypt': 'Egito', 'Pakistan': 'Paquistão', 'Iran': 'Irã',
    'Afghanistan': 'Afeganistão'
}

# Coordenadas aproximadas dos países emergentes
COORDS_PAISES = {
    'Afghanistan': (33.9, 67.7), 'Angola': (-11.2, 17.9), 'Bangladesh': (23.7, 90.4),
    'Benin': (9.3, 2.3), 'Burkina Faso': (12.4, -1.6), 'Cambodia': (12.6, 104.9),
    'Cameroon': (7.4, 12.4), 'Chad': (15.5, 18.7), 'Congo, Dem. Rep.': (-4.3, 15.3),
    'Congo, Rep.': (-4.3, 15.3), 'Egypt, Arab Rep.': (26.8, 30.8), 'Egypt': (26.8, 30.8),
    'Ethiopia': (9.1, 40.5), 'Ghana': (7.9, -1.0), 'Guinea': (9.9, -9.7),
    'Haiti': (19.0, -72.1), 'India': (20.6, 78.9), 'Indonesia': (-0.8, 113.9),
    'Iran, Islamic Rep.': (32.4, 53.7), 'Iran': (32.4, 53.7), 'Iraq': (33.2, 43.7),
    "Cote d'Ivoire": (7.5, -5.5), 'Kenya': (-0.02, 37.9), 'Lao PDR': (19.9, 102.5),
    'Madagascar': (-18.8, 46.9), 'Malawi': (-13.3, 34.3), 'Mali': (17.6, -4.0),
    'Mauritania': (21.0, -10.9), 'Mozambique': (-18.7, 35.5), 'Myanmar': (21.9, 95.9),
    'Nepal': (28.4, 84.1), 'Niger': (17.6, 8.1), 'Nigeria': (9.1, 8.7),
    'Pakistan': (30.4, 69.3), 'Rwanda': (-1.9, 29.9), 'Senegal': (14.5, -14.5),
    'Sierra Leone': (8.5, -11.8), 'Somalia': (5.2, 46.2), 'South Sudan': (6.9, 31.3),
    'Sudan': (12.9, 30.2), 'Syrian Arab Republic': (34.8, 38.9), 'Tanzania': (-6.4, 34.9),
    'Togo': (8.6, 1.2), 'Uganda': (1.4, 32.3), 'Vietnam': (14.1, 108.3),
    'Yemen, Rep.': (15.6, 48.5), 'Zambia': (-13.1, 27.8), 'Zimbabwe': (-19.0, 29.2),
    'Central African Republic': (6.6, 20.9), 'Liberia': (6.4, -9.4),
    'Gambia, The': (13.4, -16.6), 'Guinea-Bissau': (12.0, -15.2),
    'Comoros': (-11.9, 43.9), 'Eritrea': (15.2, 39.8), 'Djibouti': (11.8, 42.6),
    'Burundi': (-3.4, 29.9), 'Lesotho': (-29.6, 28.2)
}


class TrainingVisualizer:
    """Gera visualizações do treinamento de modelos."""

    def __init__(self):
        self.output_dir = config.OUTPUT_DIR
        self.viz_dir = os.path.join(self.output_dir, 'visualizacoes_treino')
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.viz_dir, exist_ok=True)

        sns.set_theme(style="whitegrid")
        plt.rcParams['figure.figsize'] = (12, 7)

        # Carregar dados
        self.global_metrics_df = self._load_global_metrics()
        self.per_country_data = self._load_per_country_metrics()

    def _load_global_metrics(self):
        """Carrega todas as métricas globais dos CSVs gerados."""
        all_dfs = []
        for f in os.listdir(self.output_dir):
            if f.endswith('_metricas_globais.csv'):
                df = pd.read_csv(os.path.join(self.output_dir, f))
                all_dfs.append(df)

        if all_dfs:
            return pd.concat(all_dfs, ignore_index=True)

        # Fallback: tentar carregar do resumo
        resumo_path = os.path.join(self.output_dir, 'resumo_treinamento_completo.csv')
        if os.path.exists(resumo_path):
            df = pd.read_csv(resumo_path)
            rename_map = {'Estratégia': 'Estrategia', 'Global_R2': 'Val_R2',
                         'Global_RMSE': 'Val_RMSE', 'Global_MAE': 'Val_MAE'}
            df = df.rename(columns=rename_map)
            return df

        return pd.DataFrame()

    def _load_per_country_metrics(self):
        """Carrega todas as métricas por país dos CSVs gerados."""
        all_dfs = []
        for f in os.listdir(self.output_dir):
            if f.endswith('_metricas_por_pais.csv'):
                df = pd.read_csv(os.path.join(self.output_dir, f))
                all_dfs.append(df)

        if all_dfs:
            return pd.concat(all_dfs, ignore_index=True)
        return pd.DataFrame()

    def _get_country_col(self, df):
        """Retorna o nome da coluna de país."""
        for col in ['Pais', 'pais', 'País', 'country', 'Country']:
            if col in df.columns:
                return col
        return None

    def _get_model_col(self, df):
        """Retorna o nome da coluna de modelo."""
        for col in ['Modelo', 'modelo', 'Model', 'model']:
            if col in df.columns:
                return col
        return None

    def _get_r2_col(self, df):
        """Retorna o nome da coluna de R²."""
        for col in ['R2', 'r2', 'Val_R2', 'val_r2']:
            if col in df.columns:
                return col
        return None

    def _get_rmse_col(self, df):
        """Retorna o nome da coluna de RMSE."""
        for col in ['RMSE', 'rmse', 'Val_RMSE', 'val_rmse']:
            if col in df.columns:
                return col
        return None

    def _match_country(self, name):
        """Verifica se um nome de país corresponde aos 10 selecionados."""
        name_lower = str(name).lower()
        mappings = {
            'angola': 'Angola', 'ghana': 'Ghana', 'kenya': 'Kenya',
            'nigeria': 'Nigeria', 'mali': 'Mali', 'tanzania': 'Tanzania',
            'egypt': 'Egypt', 'pakistan': 'Pakistan', 'iran': 'Iran',
            'afghanistan': 'Afghanistan'
        }
        for key, val in mappings.items():
            if key in name_lower:
                return val
        return None

    # ════════════════════════════════════════════════════════════════
    # 1. VISUALIZAÇÕES COMPARATIVAS DE MÉTRICAS GLOBAIS
    # ════════════════════════════════════════════════════════════════

    def plot_global_r2_comparison(self):
        """Gráfico de barras comparando R² global entre modelos para cada dataset/estratégia."""
        if self.global_metrics_df.empty:
            print("  -> Sem dados de métricas globais para visualizar.")
            return

        df = self.global_metrics_df.copy()
        df['Dataset_Estrategia'] = df['Dataset'] + '\n' + df['Estrategia']

        fig, ax = plt.subplots(figsize=(16, 8))
        datasets_strats = sorted(df['Dataset_Estrategia'].unique())
        models = sorted(df['Modelo'].unique())
        n_models = len(models)
        x = np.arange(len(datasets_strats))
        width = 0.8 / n_models

        colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63', '#9C27B0']

        for i, model in enumerate(models):
            model_data = df[df['Modelo'] == model].copy()
            model_data['Dataset_Estrategia'] = model_data['Dataset'] + '\n' + model_data['Estrategia']
            vals = []
            for ds in datasets_strats:
                row = model_data[model_data['Dataset_Estrategia'] == ds]
                vals.append(row['Val_R2'].values[0] if len(row) > 0 else 0)
            ax.bar(x + i * width, vals, width, label=model, color=colors[i % len(colors)])

        ax.set_xlabel('Dataset × Estratégia', fontsize=12)
        ax.set_ylabel('R² Global (Validação)', fontsize=12)
        ax.set_title('Comparação de R² Global entre Modelos\n(Todos os Datasets e Estratégias)',
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * (n_models - 1) / 2)
        ax.set_xticklabels(datasets_strats, rotation=45, ha='right', fontsize=9)
        ax.legend(loc='upper left', fontsize=10)
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, 'global_r2_comparacao_modelos.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  ✓ global_r2_comparacao_modelos.png")

    def plot_global_metrics_heatmap(self):
        """Heatmap de R² global: Modelos × Datasets/Estratégias."""
        if self.global_metrics_df.empty:
            return

        df = self.global_metrics_df.copy()
        df['Dataset_Estrategia'] = df['Dataset'] + ' × ' + df['Estrategia']

        pivot = df.pivot_table(values='Val_R2', index='Modelo', columns='Dataset_Estrategia', aggfunc='first')

        fig, ax = plt.subplots(figsize=(14, 6))
        sns.heatmap(pivot, annot=True, fmt='.4f', cmap='RdYlGn', center=0.5,
                    vmin=0, vmax=1, ax=ax, linewidths=0.5,
                    cbar_kws={'label': 'R² Global'})
        ax.set_title('Heatmap de R² Global\n(Modelos × Datasets/Estratégias)', fontsize=14, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('')
        plt.xticks(rotation=45, ha='right', fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, 'global_r2_heatmap.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  ✓ global_r2_heatmap.png")

    def plot_global_rmse_comparison(self):
        """Gráfico de barras comparando RMSE global entre modelos."""
        if self.global_metrics_df.empty:
            return

        df = self.global_metrics_df.copy()
        df['Dataset_Estrategia'] = df['Dataset'] + '\n' + df['Estrategia']

        fig, ax = plt.subplots(figsize=(16, 8))
        datasets_strats = sorted(df['Dataset_Estrategia'].unique())
        models = sorted(df['Modelo'].unique())
        n_models = len(models)
        x = np.arange(len(datasets_strats))
        width = 0.8 / n_models

        colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63', '#9C27B0']

        for i, model in enumerate(models):
            model_data = df[df['Modelo'] == model].copy()
            model_data['Dataset_Estrategia'] = model_data['Dataset'] + '\n' + model_data['Estrategia']
            vals = []
            for ds in datasets_strats:
                row = model_data[model_data['Dataset_Estrategia'] == ds]
                vals.append(row['Val_RMSE'].values[0] if len(row) > 0 else 0)
            ax.bar(x + i * width, vals, width, label=model, color=colors[i % len(colors)])

        ax.set_xlabel('Dataset × Estratégia', fontsize=12)
        ax.set_ylabel('RMSE Global (Validação)', fontsize=12)
        ax.set_title('Comparação de RMSE Global entre Modelos\n(Menor é Melhor)', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * (n_models - 1) / 2)
        ax.set_xticklabels(datasets_strats, rotation=45, ha='right', fontsize=9)
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, 'global_rmse_comparacao_modelos.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  ✓ global_rmse_comparacao_modelos.png")

    def plot_global_best_model_ranking(self):
        """Ranking dos melhores modelos por R² global."""
        if self.global_metrics_df.empty:
            return

        df = self.global_metrics_df.copy()
        df['Combinacao'] = df['Modelo'] + '\n(' + df['Dataset'] + ' × ' + df['Estrategia'] + ')'
        df_sorted = df.nlargest(15, 'Val_R2')

        fig, ax = plt.subplots(figsize=(12, 8))
        colors = ['gold' if i == 0 else 'silver' if i == 1 else '#CD7F32' if i == 2 else '#2196F3'
                  for i in range(len(df_sorted))]
        bars = ax.barh(range(len(df_sorted)), df_sorted['Val_R2'].values, color=colors)
        ax.set_yticks(range(len(df_sorted)))
        ax.set_yticklabels(df_sorted['Combinacao'].values, fontsize=9)
        ax.set_xlabel('R² Global (Validação)', fontsize=12)
        ax.set_title('Top 15 Melhores Combinações\n(Modelo × Dataset × Estratégia)', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

        for bar, val in zip(bars, df_sorted['Val_R2'].values):
            ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                    f'{val:.4f}', va='center', fontsize=9)

        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, 'global_ranking_top15.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  ✓ global_ranking_top15.png")

    def plot_ganho_preditivo(self):
        """Gráfico de ganho preditivo: agregados vs não agregado."""
        if self.global_metrics_df.empty:
            return

        df = self.global_metrics_df.copy()
        baseline = df[df['Dataset'] == 'nao_agregado']
        aggregated = df[df['Dataset'] != 'nao_agregado']

        if baseline.empty or aggregated.empty:
            return

        # Melhor R² por modelo no baseline
        best_baseline = baseline.groupby('Modelo')['Val_R2'].max().reset_index()
        best_baseline.columns = ['Modelo', 'Baseline_R2']

        # Melhor R² por modelo nos agregados
        best_agg = aggregated.groupby('Modelo')['Val_R2'].max().reset_index()
        best_agg.columns = ['Modelo', 'Agregado_R2']

        merged = pd.merge(best_baseline, best_agg, on='Modelo')
        merged['Ganho'] = merged['Agregado_R2'] - merged['Baseline_R2']
        merged['Ganho_Pct'] = (merged['Ganho'] / merged['Baseline_R2'].abs().clip(lower=0.01)) * 100

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Gráfico 1: Comparação Baseline vs Agregado
        x = np.arange(len(merged))
        width = 0.35
        ax1.bar(x - width/2, merged['Baseline_R2'], width, label='Não Agregado (Baseline)', color='#FF6B6B')
        ax1.bar(x + width/2, merged['Agregado_R2'], width, label='Melhor Agregado', color='#4ECDC4')
        ax1.set_xticks(x)
        ax1.set_xticklabels(merged['Modelo'], rotation=45, ha='right')
        ax1.set_ylabel('R² Global')
        ax1.set_title('Baseline vs Melhor Agregado', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)

        # Gráfico 2: Ganho percentual
        colors = ['#4CAF50' if g > 0 else '#F44336' for g in merged['Ganho_Pct']]
        ax2.bar(merged['Modelo'], merged['Ganho_Pct'], color=colors)
        ax2.set_ylabel('Ganho Preditivo (%)')
        ax2.set_title('Ganho com Agregação WDI+WGI', fontsize=12, fontweight='bold')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.grid(axis='y', alpha=0.3)
        for i, (v, m) in enumerate(zip(merged['Ganho_Pct'], merged['Modelo'])):
            ax2.text(i, v + 0.5 if v >= 0 else v - 1.5, f'{v:+.1f}%', ha='center', fontsize=10, fontweight='bold')

        plt.suptitle('Ganho Preditivo: Agregação WDI+WGI vs Apenas WDI', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, 'ganho_preditivo_agregacao.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  ✓ ganho_preditivo_agregacao.png")

    # ════════════════════════════════════════════════════════════════
    # 2. VISUALIZAÇÕES COMPARATIVAS DOS 10 PAÍSES SELECIONADOS
    # ════════════════════════════════════════════════════════════════

    def plot_10_paises_r2_comparison(self):
        """Comparação de R² dos 10 países selecionados por modelo."""
        if self.per_country_data.empty:
            print("  -> Sem dados por país para visualizar.")
            return

        df = self.per_country_data.copy()
        country_col = self._get_country_col(df)
        model_col = self._get_model_col(df)
        r2_col = self._get_r2_col(df)

        if not all([country_col, model_col, r2_col]):
            print("  -> Colunas necessárias não encontradas.")
            return

        # Mapear países para os 10 selecionados
        df['Pais_Match'] = df[country_col].apply(self._match_country)
        df_10 = df[df['Pais_Match'].notna()].copy()

        if df_10.empty:
            print(f"  -> Nenhum dos 10 países encontrado nos dados.")
            return

        # Melhor R² por país e modelo
        pivot = df_10.groupby(['Pais_Match', model_col])[r2_col].max().reset_index()
        pivot_table = pivot.pivot_table(values=r2_col, index='Pais_Match', columns=model_col, aggfunc='first')

        ordered_countries = [c for c in PAISES_SELECIONADOS if c in pivot_table.index]
        pivot_table = pivot_table.reindex(ordered_countries)

        # Usar labels em português
        pivot_table.index = [PAISES_LABELS.get(c, c) for c in pivot_table.index]

        fig, ax = plt.subplots(figsize=(14, 8))
        pivot_table.plot(kind='bar', ax=ax, width=0.75,
                        color=['#2196F3', '#4CAF50', '#FF9800', '#E91E63', '#9C27B0'])
        ax.set_xlabel('País', fontsize=12)
        ax.set_ylabel('R² (Validação)', fontsize=12)
        ax.set_title('Comparação de R² por Modelo\n(10 Países Selecionados - Melhor Resultado por Dataset/Estratégia)',
                    fontsize=14, fontweight='bold')
        ax.legend(title='Modelo', loc='upper right')
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, '10paises_r2_por_modelo.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  ✓ 10paises_r2_por_modelo.png")

    def plot_10_paises_rmse_comparison(self):
        """Comparação de RMSE dos 10 países selecionados por modelo."""
        if self.per_country_data.empty:
            return

        df = self.per_country_data.copy()
        country_col = self._get_country_col(df)
        model_col = self._get_model_col(df)
        rmse_col = self._get_rmse_col(df)

        if not all([country_col, model_col, rmse_col]):
            return

        df['Pais_Match'] = df[country_col].apply(self._match_country)
        df_10 = df[df['Pais_Match'].notna()].copy()

        if df_10.empty:
            return

        pivot = df_10.groupby(['Pais_Match', model_col])[rmse_col].min().reset_index()
        pivot_table = pivot.pivot_table(values=rmse_col, index='Pais_Match', columns=model_col, aggfunc='first')

        ordered_countries = [c for c in PAISES_SELECIONADOS if c in pivot_table.index]
        pivot_table = pivot_table.reindex(ordered_countries)
        pivot_table.index = [PAISES_LABELS.get(c, c) for c in pivot_table.index]

        fig, ax = plt.subplots(figsize=(14, 8))
        pivot_table.plot(kind='bar', ax=ax, width=0.75,
                        color=['#2196F3', '#4CAF50', '#FF9800', '#E91E63', '#9C27B0'])
        ax.set_xlabel('País', fontsize=12)
        ax.set_ylabel('RMSE (Validação)', fontsize=12)
        ax.set_title('Comparação de RMSE por Modelo\n(10 Países Selecionados - Menor é Melhor)',
                    fontsize=14, fontweight='bold')
        ax.legend(title='Modelo', loc='upper right')
        ax.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, '10paises_rmse_por_modelo.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  ✓ 10paises_rmse_por_modelo.png")

    def plot_10_paises_radar(self):
        """Gráfico radar dos 10 países para o melhor modelo."""
        if self.per_country_data.empty:
            return

        df = self.per_country_data.copy()
        country_col = self._get_country_col(df)
        r2_col = self._get_r2_col(df)

        if not all([country_col, r2_col]):
            return

        df['Pais_Match'] = df[country_col].apply(self._match_country)
        df_10 = df[df['Pais_Match'].notna()].copy()

        if df_10.empty:
            return

        # Melhor R² por país (qualquer modelo)
        best_per_country = df_10.groupby('Pais_Match')[r2_col].max().reset_index()
        ordered = [c for c in PAISES_SELECIONADOS if c in best_per_country['Pais_Match'].values]
        best_per_country = best_per_country[best_per_country['Pais_Match'].isin(ordered)]
        best_per_country = best_per_country.set_index('Pais_Match').reindex(ordered)

        # Normalizar R² para [0, 1] (clip negatives to 0)
        values = best_per_country[r2_col].clip(lower=0).values
        categories = [PAISES_LABELS.get(c, c) for c in best_per_country.index.tolist()]

        N = len(categories)
        if N < 3:
            return

        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]
        values_plot = np.append(values, values[0])

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        ax.plot(angles, values_plot, 'o-', linewidth=2, color='#2196F3')
        ax.fill(angles, values_plot, alpha=0.25, color='#2196F3')
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=11)
        ax.set_ylim(0, 1)
        ax.set_title('Melhor R² por País (10 Países Selecionados)\n(Valores negativos clipped a 0)',
                     fontsize=13, fontweight='bold', pad=20)
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, '10paises_radar_r2.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  ✓ 10paises_radar_r2.png")

    def plot_10_paises_evolucao_temporal(self):
        """Gráfico mostrando previsões vs real para os 10 países."""
        pred_files = [f for f in os.listdir(self.output_dir) if f.endswith('_predictions.pkl')]

        if not pred_files:
            print("  -> Sem ficheiros de previsões para gráfico temporal.")
            return

        # Usar o primeiro ficheiro de previsões disponível
        pred_path = os.path.join(self.output_dir, pred_files[0])
        try:
            with open(pred_path, 'rb') as f:
                predictions = pickle.load(f)
        except:
            print("  -> Erro ao carregar previsões.")
            return

        # Encontrar modelo com dados por país
        best_model = None
        for model_name in ['SARIMAX', 'XGBoost', 'TFT', 'RandomForest', 'LSTM', 'BayesHierarquico']:
            if model_name in predictions and predictions[model_name].get('per_country'):
                best_model = model_name
                break

        if best_model is None:
            print("  -> Nenhum modelo com previsões por país encontrado.")
            return

        per_country_preds = predictions[best_model]['per_country']

        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        axes_flat = axes.flatten()

        for i, country in enumerate(PAISES_SELECIONADOS):
            if i >= 10:
                break
            ax = axes_flat[i]

            # Procurar o país nos dados
            found = False
            for key in per_country_preds.keys():
                if self._match_country(key) == country:
                    data = per_country_preds[key]
                    y_true = data['y_true']
                    y_pred = data['y_pred']
                    ax.plot(range(len(y_true)), y_true, 'b-o', markersize=4, label='Real')
                    ax.plot(range(len(y_pred)), y_pred, 'r--s', markersize=4, label='Previsto')
                    found = True
                    break

            label = PAISES_LABELS.get(country, country)
            ax.set_title(label, fontsize=10, fontweight='bold')
            if i == 0:
                ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
            if not found:
                ax.text(0.5, 0.5, 'Sem dados', ha='center', va='center', transform=ax.transAxes)

        plt.suptitle(f'Previsões vs Valores Reais - {best_model}\n(10 Países Selecionados, Período de Validação)',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, '10paises_previsoes_vs_real.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  ✓ 10paises_previsoes_vs_real.png")

    # ════════════════════════════════════════════════════════════════
    # 3. MAPAS GEOGRÁFICOS
    # ════════════════════════════════════════════════════════════════

    def plot_mapa_r2_por_modelo(self):
        """Mapas geográficos com R² por país para cada modelo."""
        if self.per_country_data.empty:
            print("  -> Sem dados por país para mapas geográficos.")
            return

        try:
            import geopandas as gpd
            self._plot_mapa_geopandas()
        except ImportError:
            self._plot_mapa_scatter()

    def _plot_mapa_geopandas(self):
        """Gera mapas geográficos usando geopandas."""
        import geopandas as gpd

        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

        df = self.per_country_data.copy()
        country_col = self._get_country_col(df)
        model_col = self._get_model_col(df)
        r2_col = self._get_r2_col(df)

        models = df[model_col].unique()

        for model_name in models:
            df_model = df[df[model_col] == model_name].copy()
            best_r2 = df_model.groupby(country_col)[r2_col].max().reset_index()
            best_r2.columns = ['name', 'R2']

            world_merged = world.merge(best_r2, on='name', how='left')

            fig, ax = plt.subplots(figsize=(16, 10))
            world_merged.plot(column='R2', ax=ax, legend=True, cmap='RdYlGn',
                            missing_kwds={'color': 'lightgray', 'label': 'Sem dados'},
                            legend_kwds={'label': 'R² (Validação)', 'orientation': 'horizontal',
                                        'shrink': 0.6, 'pad': 0.05},
                            vmin=-1, vmax=1)

            for country in PAISES_SELECIONADOS:
                country_geo = world_merged[world_merged['name'] == country]
                if not country_geo.empty:
                    country_geo.boundary.plot(ax=ax, color='black', linewidth=2)

            ax.set_title(f'Mapa de R² por País - {model_name}\n(Melhor Dataset/Estratégia)',
                        fontsize=14, fontweight='bold')
            ax.set_xlim(-20, 80)
            ax.set_ylim(-40, 45)
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            plt.tight_layout()
            plt.savefig(os.path.join(self.viz_dir, f'mapa_r2_{model_name.lower()}.png'),
                       dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  ✓ mapa_r2_{model_name.lower()}.png")

    def _plot_mapa_scatter(self):
        """Gera mapas com scatter plot geográfico (fallback sem geopandas)."""
        df = self.per_country_data.copy()
        country_col = self._get_country_col(df)
        model_col = self._get_model_col(df)
        r2_col = self._get_r2_col(df)

        models = df[model_col].unique()

        for model_name in models:
            df_model = df[df[model_col] == model_name].copy()
            best_r2 = df_model.groupby(country_col)[r2_col].max().reset_index()

            fig, ax = plt.subplots(figsize=(16, 10))

            for _, row in best_r2.iterrows():
                country = row[country_col]
                r2_val = row[r2_col]

                # Procurar coordenadas
                coord = None
                for key, val in COORDS_PAISES.items():
                    if key.lower() in str(country).lower() or str(country).lower() in key.lower():
                        coord = val
                        break

                if coord is None:
                    continue

                lat, lon = coord

                # Cor baseada no R²
                if r2_val is None or np.isnan(r2_val):
                    color = 'gray'
                elif r2_val > 0.5:
                    color = '#4CAF50'
                elif r2_val > 0:
                    color = '#FFC107'
                elif r2_val > -10:
                    color = '#FF9800'
                else:
                    color = '#F44336'

                is_selected = self._match_country(country) is not None
                size = 150 if is_selected else 60
                edge = 'black' if is_selected else 'none'
                lw = 2 if is_selected else 0

                ax.scatter(lon, lat, c=color, s=size, edgecolors=edge, linewidth=lw, zorder=5)

                if is_selected:
                    label = PAISES_LABELS.get(self._match_country(country), country)
                    ax.annotate(label, (lon, lat), fontsize=8, fontweight='bold',
                               xytext=(5, 5), textcoords='offset points')

            # Legenda
            legend_elements = [
                mpatches.Patch(color='#4CAF50', label='R² > 0.5 (Bom)'),
                mpatches.Patch(color='#FFC107', label='0 < R² ≤ 0.5 (Moderado)'),
                mpatches.Patch(color='#FF9800', label='-10 < R² ≤ 0 (Fraco)'),
                mpatches.Patch(color='#F44336', label='R² ≤ -10 (Muito Fraco)'),
                mpatches.Patch(color='gray', label='Sem dados'),
            ]
            ax.legend(handles=legend_elements, loc='lower left', fontsize=10)

            ax.set_title(f'Mapa Geográfico de R² por País - {model_name}\n(Melhor Dataset/Estratégia)',
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('Longitude', fontsize=11)
            ax.set_ylabel('Latitude', fontsize=11)
            ax.set_xlim(-25, 120)
            ax.set_ylim(-35, 45)
            ax.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self.viz_dir, f'mapa_r2_{model_name.lower()}.png'),
                       dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  ✓ mapa_r2_{model_name.lower()}.png")

    def plot_mapa_melhor_modelo(self):
        """Mapa geográfico mostrando qual modelo é o melhor para cada país."""
        if self.per_country_data.empty:
            return

        df = self.per_country_data.copy()
        country_col = self._get_country_col(df)
        model_col = self._get_model_col(df)
        r2_col = self._get_r2_col(df)

        # Para cada país, encontrar o melhor modelo
        best_model_per_country = df.loc[df.groupby(country_col)[r2_col].idxmax()]

        model_colors = {
            'RandomForest': '#2196F3',
            'XGBoost': '#4CAF50',
            'TFT': '#FF9800',
            'SARIMAX': '#E91E63',
            'LSTM': '#9C27B0',
            'BayesHierarquico': '#795548'
        }

        fig, ax = plt.subplots(figsize=(16, 10))

        for _, row in best_model_per_country.iterrows():
            country = row[country_col]
            model = row[model_col]

            # Procurar coordenadas
            coord = None
            for key, val in COORDS_PAISES.items():
                if key.lower() in str(country).lower() or str(country).lower() in key.lower():
                    coord = val
                    break

            if coord is None:
                continue

            lat, lon = coord
            color = model_colors.get(model, 'gray')
            is_selected = self._match_country(country) is not None
            size = 150 if is_selected else 60
            edge = 'black' if is_selected else 'none'
            lw = 2 if is_selected else 0

            ax.scatter(lon, lat, c=color, s=size, edgecolors=edge, linewidth=lw, zorder=5)

            if is_selected:
                label = PAISES_LABELS.get(self._match_country(country), country)
                ax.annotate(f"{label}\n({model})", (lon, lat), fontsize=7,
                           fontweight='bold', xytext=(5, 5), textcoords='offset points')

        # Legenda
        legend_elements = [mpatches.Patch(color=c, label=m) for m, c in model_colors.items()]
        ax.legend(handles=legend_elements, loc='lower left', fontsize=11, title='Melhor Modelo')

        ax.set_title('Mapa: Melhor Modelo por País\n(Baseado no Maior R² de Validação)',
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Longitude', fontsize=11)
        ax.set_ylabel('Latitude', fontsize=11)
        ax.set_xlim(-25, 120)
        ax.set_ylim(-35, 45)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, 'mapa_melhor_modelo_por_pais.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  ✓ mapa_melhor_modelo_por_pais.png")

    # ════════════════════════════════════════════════════════════════
    # MÉTODOS PRINCIPAIS (chamados pelo pipeline)
    # ════════════════════════════════════════════════════════════════

    def generate_all_visualizations(self):
        """Gera todas as visualizações."""
        print("\n" + "=" * 60)
        print("GERANDO VISUALIZAÇÕES DO TREINAMENTO")
        print("=" * 60)

        print("\n[1/3] Visualizações Comparativas de Métricas GLOBAIS...")
        self.plot_global_r2_comparison()
        self.plot_global_metrics_heatmap()
        self.plot_global_rmse_comparison()
        self.plot_global_best_model_ranking()
        self.plot_ganho_preditivo()

        print("\n[2/3] Visualizações dos 10 PAÍSES Selecionados...")
        self.plot_10_paises_r2_comparison()
        self.plot_10_paises_rmse_comparison()
        self.plot_10_paises_radar()
        self.plot_10_paises_evolucao_temporal()

        print("\n[3/3] MAPAS GEOGRÁFICOS...")
        self.plot_mapa_r2_por_modelo()
        self.plot_mapa_melhor_modelo()

        print(f"\n✓ Todas as visualizações salvas em: {self.viz_dir}/")
        print("=" * 60)

    # Métodos legados para compatibilidade com o pipeline existente
    def plot_real_training_metrics(self):
        """Método legado - gera visualizações globais."""
        self.plot_global_r2_comparison()
        self.plot_global_metrics_heatmap()
        self.plot_global_rmse_comparison()

    def plot_predictions_vs_actual(self):
        """Método legado - gera visualizações dos 10 países."""
        self.plot_10_paises_evolucao_temporal()

    def plot_best_model_analysis(self):
        """Método legado - gera ranking e ganho preditivo."""
        self.plot_global_best_model_ranking()
        self.plot_ganho_preditivo()

    def plot_predictions_comparison(self):
        """Método legado - gera comparações e mapas."""
        self.plot_10_paises_r2_comparison()
        self.plot_10_paises_rmse_comparison()
        self.plot_10_paises_radar()
        self.plot_mapa_r2_por_modelo()
        self.plot_mapa_melhor_modelo()
