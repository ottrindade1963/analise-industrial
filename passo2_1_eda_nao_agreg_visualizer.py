"""Visualizador de EDA para dados não agregados limpos (WDI + WGI) - Passo 2.1."""
import matplotlib
matplotlib.use('Agg')  # Backend não-interativo para Colab
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from passo2_1_limpeza_config import NOMES_CURTOS

class EDANaoAgregadoVisualizer:
    def __init__(self, df_wdi_limpo, df_wgi_limpo):
        """Inicializa o visualizador com dados limpos."""
        self.df_wdi = df_wdi_limpo
        self.df_wgi = df_wgi_limpo
        self.output_dir = 'resultados_eda_passo2_1'
        os.makedirs(self.output_dir, exist_ok=True)
        
    def plot_wdi_statistics(self):
        """Gera estatísticas e visualizações do WDI limpo."""
        print("  Gerando visualizações WDI limpo...")
        
        # 1. Distribuição de valores por variável (WDI)
        numeric_cols = self.df_wdi.select_dtypes(include=[np.number]).columns
        numeric_cols = [c for c in numeric_cols if c not in ['ano', 'year']]
        
        fig, axes = plt.subplots(4, 2, figsize=(14, 12))
        axes = axes.flatten()
        
        for idx, col in enumerate(numeric_cols[:8]):
            ax = axes[idx]
            self.df_wdi[col].dropna().hist(bins=30, ax=ax, color='steelblue', edgecolor='black')
            ax.set_title(f'Distribuição: {NOMES_CURTOS.get(col, col)}', fontweight='bold')
            ax.set_xlabel('Valor')
            ax.set_ylabel('Frequência')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '01_wdi_distribuicoes.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Correlação entre variáveis WDI
        corr_matrix = self.df_wdi[numeric_cols].corr()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                   square=True, linewidths=1, cbar_kws={'label': 'Correlação'})
        plt.title('Matriz de Correlação - WDI Limpo', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '02_wdi_correlacao.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Série temporal por país (variável alvo)
        if 'valor_agregado_industrial_percent_pib' in self.df_wdi.columns:
            fig, ax = plt.subplots(figsize=(14, 6))
            
            for pais in self.df_wdi['pais'].unique()[:10]:  # Top 10 países
                df_pais = self.df_wdi[self.df_wdi['pais'] == pais].sort_values('ano')
                ax.plot(df_pais['ano'], df_pais['valor_agregado_industrial_percent_pib'], 
                       marker='o', label=pais, alpha=0.7)
            
            ax.set_xlabel('Ano', fontweight='bold')
            ax.set_ylabel('Valor Agregado Industrial (% PIB)', fontweight='bold')
            ax.set_title('Série Temporal - Valor Agregado Industrial (Top 10 Países)', 
                        fontsize=12, fontweight='bold')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, '03_wdi_serie_temporal.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # 4. Boxplot por década (WDI)
        self.df_wdi['decada'] = (self.df_wdi['ano'] // 10 * 10).astype(str)
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        for idx, col in enumerate(numeric_cols[:8]):
            ax = axes[idx]
            self.df_wdi.boxplot(column=col, by='decada', ax=ax)
            ax.set_title(f'{NOMES_CURTOS.get(col, col)}', fontweight='bold')
            ax.set_xlabel('Década')
            ax.set_ylabel('Valor')
            plt.sca(ax)
            plt.xticks(rotation=45)
        
        plt.suptitle('Distribuição por Década - WDI Limpo', fontsize=14, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '04_wdi_boxplot_decada.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ 4 visualizações WDI geradas")
        
    def plot_wgi_statistics(self):
        """Gera estatísticas e visualizações do WGI limpo."""
        print("  Gerando visualizações WGI limpo...")
        
        wgi_cols = [c for c in self.df_wgi.columns if c.startswith('wgi_')]
        
        # 1. Distribuição de indicadores WGI
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        axes = axes.flatten()
        
        for idx, col in enumerate(wgi_cols):
            ax = axes[idx]
            self.df_wgi[col].dropna().hist(bins=20, ax=ax, color='seagreen', edgecolor='black')
            ax.set_title(f'Distribuição: {col.replace("wgi_", "").upper()}', fontweight='bold', fontsize=10)
            ax.set_xlabel('Índice')
            ax.set_ylabel('Frequência')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '05_wgi_distribuicoes.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Correlação entre indicadores WGI
        corr_wgi = self.df_wgi[wgi_cols].corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_wgi, annot=True, fmt='.2f', cmap='viridis', 
                   square=True, linewidths=1, cbar_kws={'label': 'Correlação'})
        plt.title('Matriz de Correlação - Indicadores WGI Limpos', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '06_wgi_correlacao.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Série temporal de indicadores WGI
        fig, ax = plt.subplots(figsize=(14, 6))
        
        for col in wgi_cols:
            df_media = self.df_wgi.groupby('ano')[col].mean()
            ax.plot(df_media.index, df_media.values, marker='o', label=col.replace('wgi_', '').upper(), linewidth=2)
        
        ax.set_xlabel('Ano', fontweight='bold')
        ax.set_ylabel('Índice Médio', fontweight='bold')
        ax.set_title('Série Temporal - Indicadores WGI (Média Global)', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '07_wgi_serie_temporal.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Ranking de países por indicador WGI (último ano)
        ultimo_ano = self.df_wgi['ano'].max()
        df_ultimo = self.df_wgi[self.df_wgi['ano'] == ultimo_ano].sort_values('wgi_rule_law', ascending=False).head(15)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        x_pos = np.arange(len(df_ultimo))
        ax.barh(x_pos, df_ultimo['wgi_rule_law'], color='steelblue', alpha=0.7)
        ax.set_yticks(x_pos)
        ax.set_yticklabels(df_ultimo['pais'])
        ax.set_xlabel('Índice de Estado de Direito (WGI)', fontweight='bold')
        ax.set_title(f'Top 15 Países - Estado de Direito ({ultimo_ano})', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '08_wgi_ranking_paises.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ 4 visualizações WGI geradas")
        
    def plot_summary_statistics(self):
        """Gera resumo estatístico dos dados não agregados."""
        print("  Gerando resumo estatístico...")
        
        # Criar figura com resumo
        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.3)
        
        # 1. Cobertura de dados WDI
        ax1 = fig.add_subplot(gs[0, 0])
        numeric_cols_wdi = self.df_wdi.select_dtypes(include=[np.number]).columns
        numeric_cols_wdi = [c for c in numeric_cols_wdi if c not in ['ano', 'year']]
        missing_pct = (self.df_wdi[numeric_cols_wdi].isnull().sum() / len(self.df_wdi) * 100).sort_values(ascending=False)
        missing_pct[:8].plot(kind='barh', ax=ax1, color='coral')
        ax1.set_xlabel('% Missing')
        ax1.set_title('Cobertura de Dados WDI', fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='x')
        
        # 2. Cobertura de dados WGI
        ax2 = fig.add_subplot(gs[0, 1])
        wgi_cols = [c for c in self.df_wgi.columns if c.startswith('wgi_')]
        missing_pct_wgi = (self.df_wgi[wgi_cols].isnull().sum() / len(self.df_wgi) * 100).sort_values(ascending=False)
        missing_pct_wgi.plot(kind='barh', ax=ax2, color='lightgreen')
        ax2.set_xlabel('% Missing')
        ax2.set_title('Cobertura de Dados WGI', fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')
        
        # 3. Contagem de países e anos
        ax3 = fig.add_subplot(gs[1, 0])
        info_text = f"""
        DADOS NÃO AGREGADOS (WDI + WGI)
        
        WDI (Quantitativo):
        • Países: {self.df_wdi['pais'].nunique()}
        • Período: {self.df_wdi['ano'].min()}-{self.df_wdi['ano'].max()}
        • Registros: {len(self.df_wdi):,}
        • Variáveis: {len(numeric_cols_wdi)}
        
        WGI (Qualitativo):
        • Países: {self.df_wgi['pais'].nunique()}
        • Período: {self.df_wgi['ano'].min()}-{self.df_wgi['ano'].max()}
        • Registros: {len(self.df_wgi):,}
        • Indicadores: {len(wgi_cols)}
        """
        ax3.text(0.1, 0.5, info_text, fontsize=11, verticalalignment='center', 
                family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax3.axis('off')
        
        # 4. Estatísticas descritivas WDI
        ax4 = fig.add_subplot(gs[1, 1])
        stats_wdi = self.df_wdi[numeric_cols_wdi].describe().T[['mean', 'std', 'min', 'max']]
        ax4.axis('tight')
        ax4.axis('off')
        table = ax4.table(cellText=stats_wdi.round(2).values, 
                         colLabels=['Média', 'Desvio', 'Mín', 'Máx'],
                         rowLabels=[NOMES_CURTOS.get(c, c) for c in stats_wdi.index],
                         cellLoc='center', loc='center', fontsize=9)
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        ax4.set_title('Estatísticas WDI', fontweight='bold', pad=20)
        
        # 5. Evolução temporal
        ax5 = fig.add_subplot(gs[2, :])
        registros_por_ano_wdi = self.df_wdi.groupby('ano').size()
        registros_por_ano_wgi = self.df_wgi.groupby('ano').size()
        
        ax5.plot(registros_por_ano_wdi.index, registros_por_ano_wdi.values, marker='o', 
                label='WDI', linewidth=2, markersize=6)
        ax5.plot(registros_por_ano_wgi.index, registros_por_ano_wgi.values, marker='s', 
                label='WGI', linewidth=2, markersize=6)
        ax5.set_xlabel('Ano', fontweight='bold')
        ax5.set_ylabel('Número de Registros', fontweight='bold')
        ax5.set_title('Evolução Temporal - Cobertura de Dados', fontweight='bold')
        ax5.legend(fontsize=11)
        ax5.grid(True, alpha=0.3)
        
        plt.savefig(os.path.join(self.output_dir, '09_resumo_estatistico.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Resumo estatístico gerado")

def executar_eda_nao_agregado(df_wdi_limpo, df_wgi_limpo):
    """Função principal para executar EDA dos dados não agregados."""
    print("\n  INICIANDO EDA DOS DADOS NÃO AGREGADOS LIMPOS...")
    
    viz = EDANaoAgregadoVisualizer(df_wdi_limpo, df_wgi_limpo)
    viz.plot_wdi_statistics()
    viz.plot_wgi_statistics()
    viz.plot_summary_statistics()
    
    print(f"  ✓ EDA dos dados não agregados concluído!")
    print(f"  ✓ Visualizações salvas em: {viz.output_dir}/")
