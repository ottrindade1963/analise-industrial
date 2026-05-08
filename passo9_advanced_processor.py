import pandas as pd
import numpy as np
import os
import joblib
import passo9_advanced_config as config


class AdvancedAnalyzer:
    def __init__(self):
        self.sensitivity_results = []
        self.robustness_results = []

    def load_data_and_model(self, model_name, dataset_name, strategy_name):
        """Carrega os dados de teste e o modelo treinado (compatível com novo formato)."""
        # Carregar dados
        data_path = os.path.join(config.DATA_DIR, f"{dataset_name}_{strategy_name}.csv")
        if not os.path.exists(data_path):
            return None, None
            
        df = pd.read_csv(data_path)
        
        # Carregar modelo (novo formato: dict com metadados)
        model_filename = f"{model_name}_{dataset_name}_{strategy_name}.pkl"
        model_path = os.path.join(config.MODEL_DIR, model_filename)
        
        if not os.path.exists(model_path):
            return None, None
            
        model_data = joblib.load(model_path)
        
        # Extrair modelo e metadados
        if isinstance(model_data, dict) and 'model' in model_data:
            model = model_data['model']
            feature_names = model_data.get('feature_names', [])
            scaler = model_data.get('scaler', None)
            year_col = model_data.get('year_col', None)
        else:
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

    def run_sensitivity_analysis(self, model, X_test, model_name):
        """Analisa o impacto de variações nas variáveis-chave na previsão final."""
        print(f"  -> Executando Análise de Sensibilidade para {model_name}...")
        
        # Previsão base (sem alterações)
        base_preds = model.predict(X_test)
        base_mean = np.mean(base_preds)
        
        if base_mean == 0:
            base_mean = 1e-6  # Evitar divisão por zero
        
        for var in config.SENSITIVITY_VARS:
            if var in X_test.columns:
                for step in config.SENSITIVITY_STEPS:
                    if step == 0.0:
                        continue
                        
                    # Criar cópia e aplicar variação
                    X_modified = X_test.copy()
                    
                    # Variação baseada no desvio padrão
                    std_dev = X_modified[var].std()
                    if std_dev == 0:
                        std_dev = 1.0
                    X_modified[var] = X_modified[var] + (std_dev * step * 3)
                    
                    # Nova previsão
                    new_preds = model.predict(X_modified)
                    new_mean = np.mean(new_preds)
                    
                    # Calcular impacto percentual na previsão média
                    impact_pct = ((new_mean - base_mean) / abs(base_mean)) * 100
                    
                    self.sensitivity_results.append({
                        'Modelo': model_name,
                        'Variavel': var,
                        'Variacao_Aplicada': f"{step*100:+.0f}%",
                        'Impacto_Previsao_%': impact_pct
                    })

    def run_robustness_check(self, model, X_test, model_name):
        """Verifica a robustez do modelo adicionando ruído aos dados."""
        print(f"  -> Executando Teste de Robustez (Ruído) para {model_name}...")
        
        base_preds = model.predict(X_test)
        
        noise_levels = [0.01, 0.05, 0.10, 0.20, 0.30]
        
        for noise in noise_levels:
            X_noisy = X_test.copy()
            
            # Adicionar ruído gaussiano a todas as features numéricas
            for col in X_noisy.columns:
                std_dev = X_noisy[col].std()
                if std_dev > 0:
                    noise_array = np.random.normal(0, std_dev * noise, size=len(X_noisy))
                    X_noisy[col] = X_noisy[col] + noise_array
                    
            noisy_preds = model.predict(X_noisy)
            
            # Calcular degradação (MAE entre previsão base e previsão com ruído)
            from sklearn.metrics import mean_absolute_error
            degradation = mean_absolute_error(base_preds, noisy_preds)
            
            self.robustness_results.append({
                'Modelo': model_name,
                'Nivel_Ruido': f"{noise*100:.0f}%",
                'Degradacao_MAE': degradation
            })

    def run_all_analyses(self):
        """Executa todas as análises avançadas."""
        print(f"\nExecutando análises avançadas para todos os cenários...")
        
        for dataset in config.DATASETS:
            for strategy in config.STRATEGIES:
                print(f"\nAnalisando cenário: {dataset} - {strategy}")
                for model_name in config.MODELS:
                    # Focar em modelos baseados em árvores (RF e XGBoost)
                    if model_name not in ['RandomForest', 'XGBoost']:
                        continue
                        
                    model, X_test = self.load_data_and_model(model_name, dataset, strategy)
                    
                    if model is not None and X_test is not None and len(X_test) > 0:
                        model_scenario_name = f"{model_name} ({dataset}-{strategy})"
                        self.run_sensitivity_analysis(model, X_test, model_scenario_name)
                        self.run_robustness_check(model, X_test, model_scenario_name)
                    else:
                        print(f"  -> Modelo ou dados não encontrados para {model_name}")
                
        # Salvar resultados
        if self.sensitivity_results:
            df_sens = pd.DataFrame(self.sensitivity_results)
            out_sens = os.path.join(config.OUTPUT_DIR, 'analise_sensibilidade.csv')
            df_sens.to_csv(out_sens, index=False)
            
        if self.robustness_results:
            df_rob = pd.DataFrame(self.robustness_results)
            out_rob = os.path.join(config.OUTPUT_DIR, 'teste_robustez.csv')
            df_rob.to_csv(out_rob, index=False)
            
        return self.sensitivity_results, self.robustness_results


if __name__ == "__main__":
    analyzer = AdvancedAnalyzer()
    analyzer.run_all_analyses()
