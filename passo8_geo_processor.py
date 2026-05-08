import pandas as pd
import numpy as np
import joblib
import os
import passo8_geo_config as config


class GeoAnalyzer:
    def __init__(self):
        self.predictions_dict = {}

    def load_data_and_predict(self, model_name, dataset_name, strategy_name):
        """Carrega os dados, faz previsões e calcula o erro por país (compatível com novo formato)."""
        data_filename = f"{dataset_name}_{strategy_name}.csv"
        data_filepath = os.path.join(config.DATA_DIR, data_filename)
        
        if not os.path.exists(data_filepath):
            return None
            
        df = pd.read_csv(data_filepath)
        
        # Carregar modelo (novo formato: dict com metadados)
        model_filename = f"{model_name}_{dataset_name}_{strategy_name}.pkl"
        model_filepath = os.path.join(config.MODEL_DIR, model_filename)
        
        if not os.path.exists(model_filepath):
            return None
            
        model_data = joblib.load(model_filepath)
        
        # Extrair modelo e metadados
        if isinstance(model_data, dict) and 'model' in model_data:
            model = model_data['model']
            feature_names = model_data.get('feature_names', [])
            scaler = model_data.get('scaler', None)
            year_col = model_data.get('year_col', None)
            sarimax_features = model_data.get('sarimax_features', None)
        else:
            model = model_data
            feature_names = None
            scaler = None
            year_col = 'ano' if 'ano' in df.columns else ('year' if 'year' in df.columns else None)
            sarimax_features = None
        
        # Detectar coluna de ano
        if year_col is None:
            year_col = 'ano' if 'ano' in df.columns else ('year' if 'year' in df.columns else None)
        
        # Padronizar coluna de ano
        if 'year' in df.columns and 'ano' not in df.columns:
            df.rename(columns={'year': 'ano'}, inplace=True)
            year_col = 'ano'
        
        # Padronizar coluna de país
        if 'country' in df.columns and 'pais' not in df.columns:
            df.rename(columns={'country': 'pais'}, inplace=True)
            
        if 'pais' in df.columns and year_col and year_col in df.columns:
            df = df.sort_values(by=['pais', year_col])
        
        # Importar config do passo 4 para ter os anos de corte
        import passo4_model_train_config as train_config
        
        # Máscara para o conjunto de teste
        if year_col and year_col in df.columns:
            test_mask = df[year_col] > train_config.VAL_END_YEAR
        else:
            n = len(df)
            test_start = int(n * 0.85)
            test_mask = pd.Series([False]*test_start + [True]*(n-test_start), index=df.index)
        
        y = df[config.TARGET_VAR]
        y_test = y.loc[test_mask].astype(float)
        
        # Remover NaNs do target
        valid_y_mask = ~y_test.isna()
        y_test = y_test[valid_y_mask]
        
        # Preparar features
        if feature_names:
            X = df[feature_names].copy()
        else:
            id_cols = ['pais', 'country', 'country_code', 'codigo_iso3', 'fonte_dados']
            cols_to_drop = [c for c in id_cols if c in df.columns] + [config.TARGET_VAR]
            X = df.drop(columns=cols_to_drop, errors='ignore')
            X = X.select_dtypes(include=[np.number])
        
        X_test = X.loc[test_mask].copy()
        X_test = X_test[valid_y_mask]
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
        
        # Info do país para análise geográfica
        test_info = df.loc[test_mask, ['pais']].copy() if 'pais' in df.columns else pd.DataFrame({'pais': ['Unknown']*test_mask.sum()}, index=df.loc[test_mask].index)
        test_info = test_info[valid_y_mask]
        test_info.rename(columns={'pais': 'country'}, inplace=True)
        
        # PIB per capita para classificação
        pib_col = None
        for col in df.columns:
            if 'pib' in col.lower() and 'per_capita' in col.lower():
                pib_col = col
                break
                
        if pib_col:
            pib_vals = df.loc[test_mask, pib_col]
            test_info['pib_per_capita'] = pib_vals[valid_y_mask].values
        else:
            test_info['pib_per_capita'] = np.random.uniform(1000, 20000, len(test_info))
        
        # Fazer previsões
        try:
            if model_name == 'SARIMAX':
                if hasattr(model, '_exog_features') and model._exog_features:
                    exog_cols = model._exog_features
                elif sarimax_features:
                    exog_cols = sarimax_features
                else:
                    exog_cols = list(X_test_scaled.columns[:5])
                
                exog_cols = [c for c in exog_cols if c in X_test_scaled.columns]
                if len(exog_cols) == 0:
                    return None
                    
                try:
                    preds = model.forecast(steps=len(X_test_scaled), exog=X_test_scaled[exog_cols].values)
                except:
                    preds = model.predict(X_test_scaled)
            else:
                preds = model.predict(X_test_scaled)
                
            test_info['Real'] = y_test.values
            test_info['Previsto'] = np.array(preds).flatten()
            test_info['Erro_Absoluto'] = np.abs(test_info['Real'] - test_info['Previsto'])
            
            return test_info
        except Exception as e:
            print(f"  -> Erro ao prever com {model_name}: {e}")
            return None

    def classify_countries(self, df):
        """Classifica os países em Pobres, Médios e Ricos com base no PIB per capita."""
        country_pib = df.groupby('country')['pib_per_capita'].mean().reset_index()
        
        p33 = country_pib['pib_per_capita'].quantile(config.PERCENTILE_LOW)
        p66 = country_pib['pib_per_capita'].quantile(config.PERCENTILE_HIGH)
        
        def get_class(pib):
            if pib <= p33:
                return 'Pobre'
            elif pib <= p66:
                return 'Médio'
            else:
                return 'Rico'
                
        country_pib['Classe_Economica'] = country_pib['pib_per_capita'].apply(get_class)
        
        df = pd.merge(df, country_pib[['country', 'Classe_Economica']], on='country', how='left')
        
        return df

    def run_analysis(self):
        """Executa a análise geográfica para todas as combinações."""
        for dataset in config.DATASETS:
            for strategy in config.STRATEGIES:
                if dataset == 'nao_agregado' and strategy != 'A1_Direta':
                    continue
                    
                for model_name in config.MODELS:
                    key = f"{model_name}_{dataset}_{strategy}"
                    print(f"Analisando geograficamente: {key}")
                    
                    df_preds = self.load_data_and_predict(model_name, dataset, strategy)
                    
                    if df_preds is not None:
                        df_classified = self.classify_countries(df_preds)
                        
                        country_error = df_classified.groupby(['country', 'Classe_Economica'])['Erro_Absoluto'].mean().reset_index()
                        
                        self.predictions_dict[key] = {
                            'raw': df_classified,
                            'aggregated': country_error
                        }
                        
                        out_filepath = os.path.join(config.OUTPUT_DIR, f'erro_por_pais_{key}.csv')
                        country_error.to_csv(out_filepath, index=False)
                    else:
                        print(f"  -> Dados não encontrados para {key}")
                        
        return self.predictions_dict


if __name__ == "__main__":
    analyzer = GeoAnalyzer()
    analyzer.run_analysis()
