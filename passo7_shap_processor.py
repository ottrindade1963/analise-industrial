import pandas as pd
import numpy as np
import joblib
import os
import shap
import passo7_shap_config as config


class ShapAnalyzer:
    def __init__(self):
        self.shap_values_dict = {}
        self.feature_names_dict = {}
        self.X_test_dict = {}

    def load_data_and_model(self, model_name, dataset_name, strategy_name):
        """Carrega os dados de teste e o modelo treinado (compatível com novo formato)."""
        data_filename = f"{dataset_name}_{strategy_name}.csv"
        data_filepath = os.path.join(config.DATA_DIR, data_filename)
        
        if not os.path.exists(data_filepath):
            return None, None
            
        df = pd.read_csv(data_filepath)
        
        # Carregar modelo (novo formato: dict com metadados)
        model_filename = f"{model_name}_{dataset_name}_{strategy_name}.pkl"
        model_filepath = os.path.join(config.MODEL_DIR, model_filename)
        
        if not os.path.exists(model_filepath):
            return None, None
            
        model_data = joblib.load(model_filepath)
        
        # Extrair modelo e metadados
        if isinstance(model_data, dict) and 'model' in model_data:
            model = model_data['model']
            feature_names = model_data.get('feature_names', [])
            scaler = model_data.get('scaler', None)
            year_col = model_data.get('year_col', None)
        else:
            # Compatibilidade com formato antigo
            model = model_data
            feature_names = None
            scaler = None
            year_col = 'ano' if 'ano' in df.columns else ('year' if 'year' in df.columns else None)
        
        # Detectar coluna de ano
        if year_col is None:
            year_col = 'ano' if 'ano' in df.columns else ('year' if 'year' in df.columns else None)
        
        # Padronizar coluna de ano
        if 'year' in df.columns and 'ano' not in df.columns:
            df.rename(columns={'year': 'ano'}, inplace=True)
            year_col = 'ano'
        
        # Importar config do passo 4 para ter os anos de corte
        import passo4_model_train_config as train_config
        
        # Máscara para o conjunto de teste
        if year_col and year_col in df.columns:
            test_mask = df[year_col] > train_config.VAL_END_YEAR
        else:
            n = len(df)
            test_start = int(n * 0.85)
            test_mask = pd.Series([False]*test_start + [True]*(n-test_start), index=df.index)
        
        # Preparar features
        if feature_names:
            X = df[feature_names].copy()
        else:
            id_cols = ['pais', 'country', 'country_code', 'codigo_iso3', 'fonte_dados']
            cols_to_drop = [c for c in id_cols if c in df.columns] + [config.TARGET_VAR]
            X = df.drop(columns=cols_to_drop, errors='ignore')
            X = X.select_dtypes(include=[np.number])
        
        X_test = X.loc[test_mask].copy()
        X_test = X_test.fillna(X_test.median())
        X_test = X_test.fillna(0)
        
        # Escalar features
        if scaler is not None:
            X_test_scaled = pd.DataFrame(
                scaler.transform(X_test),
                columns=X_test.columns,
                index=X_test.index
            )
        else:
            X_test_scaled = X_test
        
        return model, X_test_scaled

    def calculate_shap_values(self, model, X_test, model_name, dataset_name, strategy_name):
        """Calcula os valores SHAP para o modelo."""
        key = f"{model_name}_{dataset_name}_{strategy_name}"
        print(f"  -> Calculando valores SHAP para {key}...")
        
        try:
            explainer = shap.TreeExplainer(model)
            
            if len(X_test) > 1000:
                X_sample = shap.sample(X_test, 1000)
                shap_values = explainer.shap_values(X_sample)
                self.X_test_dict[key] = X_sample
            else:
                shap_values = explainer.shap_values(X_test)
                self.X_test_dict[key] = X_test
                
            self.shap_values_dict[key] = shap_values
            self.feature_names_dict[key] = X_test.columns.tolist()
            
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            importance_df = pd.DataFrame({
                'Feature': X_test.columns,
                'Mean_Abs_SHAP': mean_abs_shap
            }).sort_values(by='Mean_Abs_SHAP', ascending=False)
            
            out_filepath = os.path.join(config.OUTPUT_DIR, f'shap_importance_{key}.csv')
            importance_df.to_csv(out_filepath, index=False)
            
            return True
        except Exception as e:
            print(f"  -> Erro ao calcular SHAP para {key}: {e}")
            return False

    def run_analysis(self):
        """Executa a análise SHAP para todas as combinações suportadas."""
        for dataset in config.DATASETS:
            for strategy in config.STRATEGIES:
                # Para o dataset não agregado, processamos apenas uma vez (A1) pois as estratégias são iguais
                if dataset == 'nao_agregado' and strategy != 'A1_Direta':
                    continue
                    
                print(f"\nAnalisando SHAP: {dataset} - {strategy}")
                for model_name in config.MODELS_FOR_SHAP:
                    model, X_test = self.load_data_and_model(model_name, dataset, strategy)
                    
                    if model is not None and X_test is not None:
                        self.calculate_shap_values(model, X_test, model_name, dataset, strategy)
                    else:
                        print(f"  -> Modelo ou dados não encontrados para {model_name}")
                        
        return self.shap_values_dict, self.X_test_dict, self.feature_names_dict


if __name__ == "__main__":
    analyzer = ShapAnalyzer()
    analyzer.run_analysis()
