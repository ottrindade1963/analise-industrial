import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression
import warnings
warnings.filterwarnings('ignore')

import passo3_feat_eng_config as config

class FeatureEngineer:
    def __init__(self, df, dataset_name):
        self.df = df.copy()
        self.dataset_name = dataset_name
        self.quant_vars = [v for v in config.QUANTITATIVE_VARS_FOR_INTERACTION if v in self.df.columns]
        self.qual_vars = [v for v in config.QUALITATIVE_VARS if v in self.df.columns]
        
    def apply_strategy_a1(self):
        """
        Estratégia A1: Direta (Apenas Limpeza de NaNs)
        Nota: StandardScaler será aplicado no Passo 4 por cada modelo conforme necessário
        """
        df_a1 = self.df.copy()
        
        # Preencher NaNs com a média para evitar problemas nos modelos
        for col in self.quant_vars + self.qual_vars:
            if col in df_a1.columns:
                df_a1[col] = df_a1[col].fillna(df_a1[col].mean())
            
        return df_a1
        
    def apply_strategy_a2(self):
        """
        Estratégia A2: Redução de Dimensionalidade (PCA)
        Aplica PCA nas variáveis qualitativas para criar um "Fator Institucional".
        """
        df_a2 = self.apply_strategy_a1()
        
        if len(self.qual_vars) > 1:
            # Extrair dados qualitativos
            qual_data = df_a2[self.qual_vars]
            
            # Aplicar PCA
            pca = PCA(n_components=1)
            fator_institucional = pca.fit_transform(qual_data)
            
            # Adicionar o fator latente
            df_a2['fator_institucional_pca'] = fator_institucional
            
            # Remover as variáveis originais para reduzir dimensionalidade
            df_a2 = df_a2.drop(columns=self.qual_vars)
            
            print(f"[{self.dataset_name}] PCA Variância Explicada: {pca.explained_variance_ratio_[0]:.2%}")
            
        return df_a2
        
    def apply_strategy_a3(self):
        """
        Estratégia A3: Termos de Interação e Polinomiais
        Cria interações complexas e seleciona as melhores features.
        """
        df_a3 = self.apply_strategy_a1()
        
        if self.qual_vars and self.quant_vars:
            # Criar interações entre a melhor variável qualitativa e as quantitativas
            # Usamos Rule of Law ou Control of Corruption como proxy principal se disponíveis
            main_qual = None
            for proxy in ['wgi_rule_law', 'wgi_control_corruption', self.qual_vars[0]]:
                if proxy in self.qual_vars:
                    main_qual = proxy
                    break
                    
            if main_qual:
                for quant in self.quant_vars:
                    interaction_name = f"inter_{main_qual}_X_{quant}"
                    df_a3[interaction_name] = df_a3[main_qual] * df_a3[quant]
                    
                print(f"[{self.dataset_name}] Criadas {len(self.quant_vars)} variáveis de interação com {main_qual}.")
                
            # Criar termos polinomiais (quadráticos) para as variáveis quantitativas
            for quant in self.quant_vars:
                poly_name = f"{quant}_squared"
                df_a3[poly_name] = df_a3[quant] ** 2
                
            # Seleção de features (Feature Selection) para não explodir a dimensionalidade
            # Selecionamos as top 15 features mais correlacionadas com o target
            features_to_select = [c for c in df_a3.columns if c not in ['ano', 'year', 'country_code', 'codigo_iso3', 'country', 'fonte_dados', 'pais', config.TARGET_VAR]]
            
            if len(features_to_select) > 15:
                X = df_a3[features_to_select].fillna(0)
                y = df_a3[config.TARGET_VAR].fillna(df_a3[config.TARGET_VAR].mean())
                
                selector = SelectKBest(score_func=f_regression, k=15)
                selector.fit(X, y)
                
                selected_mask = selector.get_support()
                selected_features = [features_to_select[i] for i in range(len(features_to_select)) if selected_mask[i]]
                
                # Manter apenas as colunas de identificação, target e features selecionadas
                cols_to_keep = [c for c in df_a3.columns if c not in features_to_select] + selected_features
                # Ensure 'ano' is kept
                if 'ano' not in cols_to_keep and 'ano' in df_a3.columns:
                    cols_to_keep.append('ano')
                df_a3 = df_a3[cols_to_keep]
                
                print(f"[{self.dataset_name}] Selecionadas as top 15 features de {len(features_to_select)} disponíveis.")
            
        return df_a3
        
    def process_all_strategies(self):
        """Aplica as estratégias e retorna um dicionário com os DataFrames."""
        if self.dataset_name == 'nao_agregado':
            # Para dados não agregados (apenas WDI), usar apenas A1 para economizar tempo
            print(f"[{self.dataset_name}] Dataset não agregado (WDI): Usando apenas estratégia A1 (sem variáveis qualitativas).")
            return {
                'A1_Direta': self.apply_strategy_a1()
            }
            
        return {
            'A1_Direta': self.apply_strategy_a1(),
            'A2_PCA': self.apply_strategy_a2(),
            'A3_Interacao': self.apply_strategy_a3()
        }

def load_and_process_datasets():
    results = {}
    
    for name, filepath in config.DATASETS.items():
        if os.path.exists(filepath):
            print(f"\nProcessando dataset: {name} ({filepath})")
            df = pd.read_csv(filepath)
            
            engineer = FeatureEngineer(df, name)
            strategies_dfs = engineer.process_all_strategies()
            
            for strategy, df_strat in strategies_dfs.items():
                # Manter colunas de identificação (país, código) para uso no Passo 4
                # Apenas remover colunas de metadados desnecessárias
                cols_to_drop = ['fonte_dados']
                df_strat = df_strat.drop(columns=[c for c in cols_to_drop if c in df_strat.columns])
                
                out_filename = f"{name}_{strategy}.csv"
                out_filepath = os.path.join(config.OUTPUT_DIR, out_filename)
                df_strat.to_csv(out_filepath, index=False)
                print(f"  -> Salvo: {out_filename} (Shape: {df_strat.shape})")
                
            results[name] = strategies_dfs
        else:
            print(f"\nAVISO: Ficheiro não encontrado: {filepath}")
            
    return results

if __name__ == "__main__":
    load_and_process_datasets()
