import matplotlib
matplotlib.use("Agg")  # Backend não-interativo para Colab
import matplotlib.pyplot as plt
import shap
import os
import passo7_shap_config as config

class ShapVisualizer:
    def __init__(self, shap_values_dict, X_test_dict, feature_names_dict):
        self.shap_values_dict = shap_values_dict
        self.X_test_dict = X_test_dict
        self.feature_names_dict = feature_names_dict
        self.output_dir = os.path.join(config.OUTPUT_DIR, 'visualizacoes_shap')
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_summary(self, key):
        """Gera o gráfico de resumo SHAP (Summary Plot)."""
        print(f"Gerando SHAP Summary Plot para {key}...")
        
        shap_values = self.shap_values_dict[key]
        X_test = self.X_test_dict[key]
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test, max_display=config.TOP_N_FEATURES, show=False)
        
        plt.title(f'SHAP Summary Plot - {key}', fontsize=16)
        plt.tight_layout()
        
        filename = f'shap_summary_{key}.png'
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()

    def plot_bar(self, key):
        """Gera o gráfico de barras SHAP (Importância Média Absoluta)."""
        print(f"Gerando SHAP Bar Plot para {key}...")
        
        shap_values = self.shap_values_dict[key]
        X_test = self.X_test_dict[key]
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test, plot_type="bar", max_display=config.TOP_N_FEATURES, show=False)
        
        plt.title(f'SHAP Feature Importance (Média Absoluta) - {key}', fontsize=16)
        plt.tight_layout()
        
        filename = f'shap_bar_{key}.png'
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()

    def plot_real_dependence(self, key):
        """Gera um gráfico de dependência SHAP real para a feature mais importante."""
        print(f"Gerando SHAP Dependence Plot real para {key}...")
        
        shap_values = self.shap_values_dict[key]
        X_test = self.X_test_dict[key]
        
        # Encontrar a feature mais importante (maior média absoluta SHAP)
        import numpy as np
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        top_feature_idx = np.argmax(mean_abs_shap)
        top_feature_name = X_test.columns[top_feature_idx]
        
        plt.figure(figsize=(10, 6))
        
        try:
            # Tentar gerar o gráfico de dependência real
            shap.dependence_plot(
                top_feature_name, 
                shap_values, 
                X_test, 
                show=False,
                interaction_index="auto" # SHAP escolhe automaticamente a melhor feature para interação
            )
            
            plt.title(f'SHAP Dependence Plot: {top_feature_name} - {key}', fontsize=14)
            plt.tight_layout()
            
            filename = f'shap_dependence_{key}.png'
            plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"  -> Erro ao gerar Dependence Plot real: {e}")
            plt.close()

    def generate_all_visualizations(self):
        """Gera todas as visualizações SHAP para os modelos analisados."""
        if not self.shap_values_dict:
            print("Nenhum valor SHAP disponível para visualização.")
            return
            
        for key in self.shap_values_dict.keys():
            self.plot_summary(key)
            self.plot_bar(key)
            self.plot_real_dependence(key)
            
        print(f"Todas as visualizações SHAP salvas em: {self.output_dir}")

if __name__ == "__main__":
    # Teste simples
    import passo7_shap_processor as processor
    analyzer = processor.ShapAnalyzer()
    shap_vals, X_test, feat_names = analyzer.run_analysis()
    viz = ShapVisualizer(shap_vals, X_test, feat_names)
    viz.generate_all_visualizations()
