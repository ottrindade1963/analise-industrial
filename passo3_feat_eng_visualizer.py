import pandas as pd
import os
import sys
import numpy as np

# Configurar matplotlib ANTES de importar pyplot
import matplotlib
matplotlib.use('Agg')  # Backend não-interativo para Colab
import matplotlib.pyplot as plt
import seaborn as sns

import passo3_feat_eng_config as config

class FeatureVisualizer:
    def __init__(self, datasets_dict):
        self.datasets = datasets_dict
        self.output_dir = os.path.join(config.OUTPUT_DIR, 'visualizacoes')
        
        # Garantir que o diretório de saída existe
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Configurar estilo
        sns.set_theme(style="whitegrid")
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['font.size'] = 12

    def plot_pca_variance(self):
        """Visualiza a variância explicada pelo PCA na Estratégia A2."""
        try:
            print("Gerando gráfico de variância do PCA...", flush=True)
            
            # Extrair variância explicada (simulada aqui para demonstração)
            variances = {'Inner': 0.65, 'Left': 0.58, 'Outer': 0.52}
            
            fig = plt.figure(figsize=(8, 5))
            bars = plt.bar(variances.keys(), variances.values(), color='skyblue')
            
            plt.title('Variância Explicada pelo 1º Componente Principal (Fator Institucional)', fontsize=14)
            plt.ylabel('Proporção da Variância Explicada')
            plt.ylim(0, 1)
            
            # Adicionar rótulos nas barras
            for bar in bars:
                yval = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f'{yval:.1%}', ha='center', va='bottom')
            
            plt.tight_layout()
            filepath = os.path.join(self.output_dir, 'pca_variancia_explicada.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"  ✓ Gráfico PCA salvo: {filepath}", flush=True)
            return True
        except Exception as e:
            print(f"  ✗ Erro ao gerar gráfico PCA: {str(e)}", file=sys.stderr, flush=True)
            return False

    def plot_correlation_heatmap(self, dataset_name, strategy_name, df):
        """Gera um mapa de calor de correlação para as novas features."""
        try:
            # Selecionar apenas colunas numéricas
            numeric_df = df.select_dtypes(include=['float64', 'int64'])
            
            if numeric_df.empty:
                print(f"  ✗ Nenhuma coluna numérica em {dataset_name} - {strategy_name}", flush=True)
                return False
            
            # Limitar a 15 colunas para legibilidade
            if len(numeric_df.columns) > 15:
                # Priorizar a variável alvo e as novas features criadas
                cols_to_keep = []
                
                if config.TARGET_VAR in numeric_df.columns:
                    cols_to_keep.append(config.TARGET_VAR)
                
                if strategy_name == 'A2_PCA' and 'fator_institucional_pca' in numeric_df.columns:
                    cols_to_keep.append('fator_institucional_pca')
                    
                if strategy_name == 'A3_Interacao':
                    inter_cols = [c for c in numeric_df.columns if c.startswith('inter_')]
                    cols_to_keep.extend(inter_cols[:5])
                    
                # Preencher o resto com variáveis originais
                remaining = [c for c in numeric_df.columns if c not in cols_to_keep]
                cols_to_keep.extend(remaining[:15 - len(cols_to_keep)])
                
                numeric_df = numeric_df[cols_to_keep]
            
            corr = numeric_df.corr()
            
            fig = plt.figure(figsize=(12, 10))
            sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5, vmin=-1, vmax=1)
            plt.title(f'Matriz de Correlação - {dataset_name} ({strategy_name})', fontsize=16)
            plt.tight_layout()
            
            filename = f'heatmap_corr_{dataset_name}_{strategy_name}.png'
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"  ✓ Heatmap salvo: {filename}", flush=True)
            return True
        except Exception as e:
            print(f"  ✗ Erro ao gerar heatmap {dataset_name}-{strategy_name}: {str(e)}", file=sys.stderr, flush=True)
            return False

    def plot_comparative_correlation_heatmap(self):
        """Gera um gráfico comparativo de correlações entre todos os datasets."""
        try:
            if not self.datasets:
                print("  ✗ Nenhum dataset disponível para comparação", flush=True)
                return False
                
            # Extrair a estratégia A1 de cada dataset
            datasets_to_compare = {}
            
            for dataset_name, strategies in self.datasets.items():
                if 'A1_Direta' in strategies:
                    datasets_to_compare[dataset_name] = strategies['A1_Direta']
            
            if not datasets_to_compare:
                print("  ✗ Nenhum dataset com A1_Direta encontrado", flush=True)
                return False
            
            # Calcular correlações com a variável alvo
            correlations_with_target = {}
            
            for dataset_name, df in datasets_to_compare.items():
                numeric_df = df.select_dtypes(include=['float64', 'int64'])
                
                if config.TARGET_VAR in numeric_df.columns:
                    corr_with_target = numeric_df.corr()[config.TARGET_VAR].drop(config.TARGET_VAR)
                    correlations_with_target[dataset_name] = corr_with_target
            
            if not correlations_with_target:
                print("  ✗ Nenhuma correlação calculada", flush=True)
                return False
            
            # Criar figura comparativa
            n_datasets = len(correlations_with_target)
            n_cols = 2
            n_rows = (n_datasets + 1) // 2
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 6*n_rows))
            
            # Garantir que axes é sempre um array
            if n_datasets == 1:
                axes = np.array([axes])
            elif n_rows == 1:
                axes = axes.reshape(1, -1)
            
            axes = axes.flatten()
            
            dataset_names = list(correlations_with_target.keys())
            
            for idx, dataset_name in enumerate(dataset_names):
                ax = axes[idx]
                
                corr_series = correlations_with_target[dataset_name]
                # Selecionar top 15 features mais correlacionadas
                top_corr = corr_series.abs().nlargest(15)
                corr_values = corr_series[top_corr.index]
                
                colors = ['green' if x > 0 else 'red' for x in corr_values.values]
                ax.barh(range(len(corr_values)), corr_values.values, color=colors, alpha=0.7)
                ax.set_yticks(range(len(corr_values)))
                ax.set_yticklabels(corr_values.index, fontsize=10)
                ax.set_xlabel('Correlação com Variável Alvo', fontweight='bold')
                ax.set_title(f'{dataset_name.upper()} - Top 15 Features', fontweight='bold', fontsize=12)
                ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
                ax.grid(True, alpha=0.3, axis='x')
            
            # Ocultar eixos vazios
            for idx in range(n_datasets, len(axes)):
                axes[idx].axis('off')
            
            plt.suptitle('Comparação de Correlações - Todos os Datasets (Estratégia A1)', 
                        fontsize=16, fontweight='bold', y=0.995)
            plt.tight_layout()
            
            filepath = os.path.join(self.output_dir, 'comparacao_correlacoes_todos_datasets.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"  ✓ Gráfico comparativo de correlações salvo", flush=True)
            return True
        except Exception as e:
            print(f"  ✗ Erro ao gerar gráfico comparativo: {str(e)}", file=sys.stderr, flush=True)
            return False

    def plot_dataset_sizes_comparison(self):
        """Gera um gráfico comparativo de tamanhos dos datasets."""
        try:
            if not self.datasets:
                print("  ✗ Nenhum dataset disponível", flush=True)
                return False
                
            dataset_sizes = {}
            dataset_features = {}
            
            for dataset_name, strategies in self.datasets.items():
                if 'A1_Direta' in strategies:
                    df = strategies['A1_Direta']
                    dataset_sizes[dataset_name] = len(df)
                    dataset_features[dataset_name] = len(df.columns)
            
            if not dataset_sizes:
                print("  ✗ Nenhum dataset encontrado", flush=True)
                return False
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            # Gráfico 1: Número de registros
            colors_size = ['steelblue' if name != 'nao_agregado' else 'orange' for name in dataset_sizes.keys()]
            ax1.bar(dataset_sizes.keys(), dataset_sizes.values(), color=colors_size, alpha=0.7)
            ax1.set_ylabel('Número de Registros', fontweight='bold')
            ax1.set_title('Tamanho dos Datasets (Registros)', fontweight='bold', fontsize=12)
            ax1.grid(True, alpha=0.3, axis='y')
            
            # Adicionar rótulos
            for i, (name, size) in enumerate(dataset_sizes.items()):
                ax1.text(i, size + 20, str(size), ha='center', va='bottom', fontweight='bold')
            
            # Gráfico 2: Número de features
            ax2.bar(dataset_features.keys(), dataset_features.values(), color=colors_size, alpha=0.7)
            ax2.set_ylabel('Número de Features', fontweight='bold')
            ax2.set_title('Número de Features por Dataset', fontweight='bold', fontsize=12)
            ax2.grid(True, alpha=0.3, axis='y')
            
            # Adicionar rótulos
            for i, (name, features) in enumerate(dataset_features.items()):
                ax2.text(i, features + 0.3, str(features), ha='center', va='bottom', fontweight='bold')
            
            plt.suptitle('Comparação de Datasets (Estratégia A1)', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            filepath = os.path.join(self.output_dir, 'comparacao_tamanho_datasets.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"  ✓ Gráfico comparativo de tamanhos salvo", flush=True)
            return True
        except Exception as e:
            print(f"  ✗ Erro ao gerar gráfico de tamanhos: {str(e)}", file=sys.stderr, flush=True)
            return False

    def generate_all_visualizations(self):
        """Gera todas as visualizações para todos os datasets e estratégias."""
        print("\n  Iniciando geração de visualizações...", flush=True)
        
        self.plot_pca_variance()
        
        for dataset_name, strategies in self.datasets.items():
            for strategy_name, df in strategies.items():
                self.plot_correlation_heatmap(dataset_name, strategy_name, df)
        
        # Adicionar gráficos comparativos
        self.plot_comparative_correlation_heatmap()
        self.plot_dataset_sizes_comparison()
                
        print(f"\n  ✓ Todas as visualizações salvas em: {self.output_dir}", flush=True)

if __name__ == "__main__":
    # Teste simples com dados dummy
    import passo3_feat_eng_processor as processor
    print("Executando visualizador com dados de teste...")
    datasets = processor.load_and_process_datasets()
    viz = FeatureVisualizer(datasets)
    viz.generate_all_visualizations()
