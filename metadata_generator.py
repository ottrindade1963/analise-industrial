"""
Módulo de Geração Automática de Metadados
==========================================
Gera metadados completos em formato JSON para todos os passos do pipeline:
  - Passo 1: fontes de dados, datas de download, dimensões, indicadores extraídos
  - Passo 2.1: valores removidos/imputados, estatísticas antes/depois, métodos de agregação
  - Passo 3: features criadas, transformações aplicadas, correlações, PCA variância explicada
  - Passo 4: hiperparâmetros, tempo de treino, convergência, métricas detalhadas
"""

import os
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime


def _safe_serialize(obj):
    """Converte objetos não serializáveis para JSON."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif isinstance(obj, (set, frozenset)):
        return list(obj)
    elif pd.isna(obj):
        return None
    return str(obj)


def save_metadata(metadata, filepath):
    """Salva metadados em JSON com formatação legível."""
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2, default=_safe_serialize)
    print(f"  📋 Metadados salvos: {filepath}")


# ═══════════════════════════════════════════════════════════════════════════════
# PASSO 1: Extração de Dados
# ═══════════════════════════════════════════════════════════════════════════════

def generate_metadata_passo1(df_wdi, df_wgi, indicadores_wdi, indicadores_wgi,
                              paises_extraidos, output_dir='data/raw'):
    """
    Gera metadados completos do Passo 1 (Extração).
    
    Args:
        df_wdi: DataFrame com dados WDI extraídos
        df_wgi: DataFrame com dados WGI extraídos
        indicadores_wdi: dict {codigo_api: nome_coluna} dos indicadores WDI
        indicadores_wgi: dict {codigo_api: nome_coluna} dos indicadores WGI
        paises_extraidos: lista de códigos ISO3 dos países
        output_dir: diretório de saída
    """
    metadata = {
        "passo": "1 - Extração de Dados",
        "descricao": "Download de dados WDI e WGI da API do Banco Mundial",
        "timestamp": datetime.now().isoformat(),
        "fontes": {
            "wdi": {
                "nome": "World Development Indicators (WDI)",
                "api": "https://api.worldbank.org/v2",
                "organizacao": "Banco Mundial",
                "data_download": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "periodo_solicitado": {"inicio": 1996, "fim": 2023}
            },
            "wgi": {
                "nome": "Worldwide Governance Indicators (WGI)",
                "api": "https://api.worldbank.org/v2 (source=3)",
                "organizacao": "Banco Mundial",
                "data_download": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "periodo_solicitado": {"inicio": 1996, "fim": 2024}
            }
        },
        "paises": {
            "total_extraidos": len(paises_extraidos),
            "codigos_iso3": sorted(paises_extraidos),
            "criterio_selecao": "Países em desenvolvimento da África Subsaariana e Médio Oriente/Norte de África (excluindo HIC e agregados)"
        },
        "indicadores_wdi": {
            "total": len(indicadores_wdi),
            "lista": [{"codigo_api": k, "nome_coluna": v} for k, v in indicadores_wdi.items()]
        },
        "indicadores_wgi": {
            "total": len(indicadores_wgi),
            "lista": [{"codigo_api": k, "nome_coluna": v} for k, v in indicadores_wgi.items()]
        },
        "dimensoes": {
            "wdi": {
                "linhas": int(df_wdi.shape[0]) if df_wdi is not None else 0,
                "colunas": int(df_wdi.shape[1]) if df_wdi is not None else 0,
                "paises_unicos": int(df_wdi['codigo_iso3'].nunique()) if df_wdi is not None and 'codigo_iso3' in df_wdi.columns else 0,
                "periodo_efetivo": {
                    "inicio": int(df_wdi['ano'].min()) if df_wdi is not None and 'ano' in df_wdi.columns else None,
                    "fim": int(df_wdi['ano'].max()) if df_wdi is not None and 'ano' in df_wdi.columns else None
                }
            },
            "wgi": {
                "linhas": int(df_wgi.shape[0]) if df_wgi is not None else 0,
                "colunas": int(df_wgi.shape[1]) if df_wgi is not None else 0,
                "paises_unicos": int(df_wgi['country_code'].nunique()) if df_wgi is not None and 'country_code' in df_wgi.columns else 0,
                "periodo_efetivo": {
                    "inicio": int(df_wgi['year'].min()) if df_wgi is not None and 'year' in df_wgi.columns else None,
                    "fim": int(df_wgi['year'].max()) if df_wgi is not None and 'year' in df_wgi.columns else None
                }
            }
        },
        "cobertura_wdi": {},
        "cobertura_wgi": {},
        "ficheiros_gerados": [
            f"{output_dir}/wdi_emergentes_final.csv",
            f"{output_dir}/wdi_emergentes_final.xlsx",
            "dados_qualitativos.csv",
            "dados_qualitativos.xlsx"
        ]
    }

    # Cobertura por indicador WDI
    if df_wdi is not None:
        for col in indicadores_wdi.values():
            if col in df_wdi.columns:
                total = len(df_wdi)
                validos = df_wdi[col].notna().sum()
                metadata["cobertura_wdi"][col] = {
                    "valores_validos": int(validos),
                    "valores_missing": int(total - validos),
                    "cobertura_percentual": round(float(validos / total * 100), 2) if total > 0 else 0
                }

    # Cobertura por indicador WGI
    if df_wgi is not None:
        for col in indicadores_wgi.values():
            if col in df_wgi.columns:
                total = len(df_wgi)
                validos = df_wgi[col].notna().sum()
                metadata["cobertura_wgi"][col] = {
                    "valores_validos": int(validos),
                    "valores_missing": int(total - validos),
                    "cobertura_percentual": round(float(validos / total * 100), 2) if total > 0 else 0
                }

    save_metadata(metadata, os.path.join(output_dir, 'metadata_passo1.json'))
    return metadata


# ═══════════════════════════════════════════════════════════════════════════════
# PASSO 2.1: Limpeza e Agregação
# ═══════════════════════════════════════════════════════════════════════════════

def generate_metadata_passo2_1(df_wdi_original, df_wdi_limpo, df_wgi_original, df_wgi_limpo,
                                stats_wdi, stats_wgi, metodos_imputacao,
                                metodos_agregacao=None, output_dir='dados_limpos'):
    """
    Gera metadados completos do Passo 2.1 (Limpeza e Agregação).
    
    Args:
        df_wdi_original: DataFrame WDI antes da limpeza
        df_wdi_limpo: DataFrame WDI após limpeza
        df_wgi_original: DataFrame WGI antes da limpeza
        df_wgi_limpo: DataFrame WGI após limpeza
        stats_wdi: dict com estatísticas de limpeza WDI
        stats_wgi: dict com estatísticas de limpeza WGI
        metodos_imputacao: dict {variavel: metodo} de imputação
        metodos_agregacao: dict com info dos 3 métodos de agregação
        output_dir: diretório de saída
    """
    metadata = {
        "passo": "2.1 - Limpeza, Imputação e Agregação",
        "descricao": "Limpeza de dados WDI e WGI, imputação de valores em falta, e agregação por 3 métodos",
        "timestamp": datetime.now().isoformat(),
        "limpeza_wdi": {
            "dimensoes_originais": {
                "linhas": int(df_wdi_original.shape[0]) if df_wdi_original is not None else 0,
                "colunas": int(df_wdi_original.shape[1]) if df_wdi_original is not None else 0,
                "paises": int(df_wdi_original['codigo_iso3'].nunique()) if df_wdi_original is not None and 'codigo_iso3' in df_wdi_original.columns else 0
            },
            "dimensoes_apos_limpeza": {
                "linhas": int(df_wdi_limpo.shape[0]) if df_wdi_limpo is not None else 0,
                "colunas": int(df_wdi_limpo.shape[1]) if df_wdi_limpo is not None else 0,
                "paises": int(df_wdi_limpo['codigo_iso3'].nunique()) if df_wdi_limpo is not None and 'codigo_iso3' in df_wdi_limpo.columns else 0
            },
            "linhas_removidas": int(df_wdi_original.shape[0] - df_wdi_limpo.shape[0]) if df_wdi_original is not None and df_wdi_limpo is not None else 0,
            "metodos_imputacao": metodos_imputacao,
            "missing_antes_por_variavel": {},
            "missing_depois_por_variavel": {}
        },
        "limpeza_wgi": {
            "dimensoes_originais": {
                "linhas": int(df_wgi_original.shape[0]) if df_wgi_original is not None else 0,
                "colunas": int(df_wgi_original.shape[1]) if df_wgi_original is not None else 0,
                "paises": int(df_wgi_original['country_code'].nunique()) if df_wgi_original is not None and 'country_code' in df_wgi_original.columns else 0
            },
            "dimensoes_apos_limpeza": {
                "linhas": int(df_wgi_limpo.shape[0]) if df_wgi_limpo is not None else 0,
                "colunas": int(df_wgi_limpo.shape[1]) if df_wgi_limpo is not None else 0,
                "paises": int(df_wgi_limpo['country_code'].nunique()) if df_wgi_limpo is not None and 'country_code' in df_wgi_limpo.columns else 0
            },
            "linhas_removidas": int(df_wgi_original.shape[0] - df_wgi_limpo.shape[0]) if df_wgi_original is not None and df_wgi_limpo is not None else 0,
            "metodo_imputacao_wgi": "interpolacao_linear_por_pais",
            "missing_antes_por_variavel": {},
            "missing_depois_por_variavel": {}
        },
        "agregacao": {
            "metodo1_inner": {
                "descricao": "Inner Join - Apenas registos com dados em ambas as fontes",
                "resultado": metodos_agregacao.get('inner', {}) if metodos_agregacao else {}
            },
            "metodo2_left": {
                "descricao": "Left Join com Imputação - Mantém todos os registos WDI, imputa WGI em falta",
                "resultado": metodos_agregacao.get('left', {}) if metodos_agregacao else {}
            },
            "metodo3_outer": {
                "descricao": "Outer Join Rastreável - Todos os registos com coluna de rastreabilidade",
                "resultado": metodos_agregacao.get('outer', {}) if metodos_agregacao else {}
            }
        },
        "estatisticas_resumo": {
            "wdi": stats_wdi if stats_wdi else {},
            "wgi": stats_wgi if stats_wgi else {}
        },
        "ficheiros_gerados": [
            f"{output_dir}/wdi_emergentes_limpo.csv",
            f"{output_dir}/wdi_emergentes_limpo.xlsx",
            f"{output_dir}/wgi_emergentes_limpo.csv",
            f"{output_dir}/wgi_emergentes_limpo.xlsx",
            "agregado_metodo1_inner/agregado_inner.csv",
            "agregado_metodo2_left_imputado/agregado_left_imputado.csv",
            "agregado_metodo3_outer_completo/agregado_outer_completo.csv"
        ]
    }

    # Missing antes/depois WDI
    if df_wdi_original is not None and df_wdi_limpo is not None:
        num_cols = df_wdi_original.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            if col in df_wdi_original.columns:
                metadata["limpeza_wdi"]["missing_antes_por_variavel"][col] = int(df_wdi_original[col].isna().sum())
            if col in df_wdi_limpo.columns:
                metadata["limpeza_wdi"]["missing_depois_por_variavel"][col] = int(df_wdi_limpo[col].isna().sum())

    # Missing antes/depois WGI
    if df_wgi_original is not None and df_wgi_limpo is not None:
        num_cols_wgi = df_wgi_original.select_dtypes(include=[np.number]).columns
        for col in num_cols_wgi:
            if col in df_wgi_original.columns:
                metadata["limpeza_wgi"]["missing_antes_por_variavel"][col] = int(df_wgi_original[col].isna().sum())
            if col in df_wgi_limpo.columns:
                metadata["limpeza_wgi"]["missing_depois_por_variavel"][col] = int(df_wgi_limpo[col].isna().sum())

    save_metadata(metadata, os.path.join(output_dir, 'metadata_passo2_1.json'))
    return metadata


# ═══════════════════════════════════════════════════════════════════════════════
# PASSO 3: Engenharia de Features
# ═══════════════════════════════════════════════════════════════════════════════

def generate_metadata_passo3(datasets_dict, output_dir='dados_engenharia'):
    """
    Gera metadados completos do Passo 3 (Engenharia de Features).
    
    Args:
        datasets_dict: dict retornado por load_and_process_datasets()
                       {dataset_name: {strategy_name: DataFrame}}
        output_dir: diretório de saída
    """
    metadata = {
        "passo": "3 - Engenharia de Features",
        "descricao": "Aplicação de 3 estratégias de engenharia de features (A1: Direta, A2: PCA, A3: Interação)",
        "timestamp": datetime.now().isoformat(),
        "estrategias": {
            "A1_Direta": {
                "descricao": "Limpeza de NaNs (preenchimento pela média). Sem transformações adicionais.",
                "transformacoes": ["fillna(mean)"]
            },
            "A2_PCA": {
                "descricao": "Redução de dimensionalidade via PCA nas variáveis qualitativas WGI, criando um fator institucional latente.",
                "transformacoes": ["PCA(n_components=1) nas variáveis WGI", "Remoção das variáveis WGI originais", "Criação de fator_institucional_pca"]
            },
            "A3_Interacao": {
                "descricao": "Criação de termos de interação (qualitativo × quantitativo) e termos polinomiais quadráticos, com seleção das top 15 features.",
                "transformacoes": ["Interações wgi_rule_law × variáveis quantitativas", "Termos quadráticos", "SelectKBest(k=15, f_regression)"]
            }
        },
        "datasets_gerados": {},
        "pca_info": {},
        "correlacoes_com_target": {},
        "ficheiros_gerados": []
    }

    target_var = 'valor_agregado_industrial_percent_pib'

    for dataset_name, strategies in datasets_dict.items():
        metadata["datasets_gerados"][dataset_name] = {}
        
        for strategy_name, df in strategies.items():
            filename = f"{dataset_name}_{strategy_name}.csv"
            metadata["ficheiros_gerados"].append(f"{output_dir}/{filename}")
            
            # Info do dataset
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = [c for c in num_cols if c != target_var and c not in ['ano', 'year']]
            
            ds_info = {
                "shape": list(df.shape),
                "n_features": len(feature_cols),
                "features": feature_cols,
                "missing_total": int(df[feature_cols].isna().sum().sum()) if feature_cols else 0
            }
            
            # Correlações com target
            if target_var in df.columns and feature_cols:
                corr = df[feature_cols + [target_var]].corr()[target_var].drop(target_var)
                top_5 = corr.abs().nlargest(5)
                ds_info["top_5_correlacoes"] = {k: round(float(corr[k]), 4) for k in top_5.index}
            
            # PCA info (para A2)
            if strategy_name == 'A2_PCA' and 'fator_institucional_pca' in df.columns:
                # Recalcular variância explicada
                from sklearn.decomposition import PCA
                from sklearn.preprocessing import StandardScaler
                import passo3_feat_eng_config as cfg3
                
                qual_vars = [v for v in cfg3.QUALITATIVE_VARS if v in df.columns]
                if not qual_vars:
                    # Ler o dataset original para obter variáveis WGI
                    try:
                        ds_path = cfg3.DATASETS.get(dataset_name)
                        if ds_path and os.path.exists(ds_path):
                            df_orig = pd.read_csv(ds_path)
                            qual_vars = [v for v in cfg3.QUALITATIVE_VARS if v in df_orig.columns]
                            if qual_vars:
                                qual_data = df_orig[qual_vars].dropna()
                                if len(qual_data) > 0:
                                    pca = PCA(n_components=1)
                                    pca.fit(qual_data)
                                    ds_info["pca_variancia_explicada"] = round(float(pca.explained_variance_ratio_[0]), 4)
                                    metadata["pca_info"][dataset_name] = {
                                        "variancia_explicada": round(float(pca.explained_variance_ratio_[0]), 4),
                                        "variaveis_originais": qual_vars,
                                        "n_componentes": 1
                                    }
                    except:
                        pass
            
            metadata["datasets_gerados"][dataset_name][strategy_name] = ds_info

    save_metadata(metadata, os.path.join(output_dir, 'metadata_passo3.json'))
    return metadata


# ═══════════════════════════════════════════════════════════════════════════════
# PASSO 4: Treinamento de Modelos
# ═══════════════════════════════════════════════════════════════════════════════

def generate_metadata_passo4(all_summaries, training_times=None, 
                              bayesian_diagnostics=None, output_dir='modelos_treinados'):
    """
    Gera metadados completos do Passo 4 (Treinamento de Modelos).
    
    Args:
        all_summaries: lista de dicts com resumo de cada modelo treinado
        training_times: dict {dataset_strategy: tempo_segundos}
        bayesian_diagnostics: dict com info de convergência MCMC
        output_dir: diretório de saída
    """
    import passo4_model_train_config as config
    
    metadata = {
        "passo": "4 - Treinamento de Modelos",
        "descricao": "Treinamento de 7 modelos (5 clássicos + 2 Bayesianos) com previsão global e por país",
        "timestamp": datetime.now().isoformat(),
        "configuracao": {
            "divisao_temporal": {
                "treino": f"<= {config.TRAIN_END_YEAR}",
                "validacao": f"{config.TRAIN_END_YEAR + 1} - {config.VAL_END_YEAR}",
                "teste": f">= {config.VAL_END_YEAR + 1}"
            },
            "variavel_alvo": config.TARGET_VAR,
            "random_state": config.RANDOM_STATE,
            "datasets": config.DATASETS,
            "estrategias": config.STRATEGIES
        },
        "modelos": {
            "RandomForest": {
                "tipo": "Ensemble (painel)",
                "biblioteca": "scikit-learn",
                "hiperparametros_grid": config.GRID_RANDOMFOREST,
                "busca": "RandomizedSearchCV(n_iter=30, cv=3)"
            },
            "XGBoost": {
                "tipo": "Gradient Boosting (painel)",
                "biblioteca": "xgboost",
                "hiperparametros_grid": config.GRID_XGBOOST,
                "busca": "RandomizedSearchCV(n_iter=30, cv=3)"
            },
            "TFT": {
                "tipo": "GradientBoosting proxy (painel)",
                "biblioteca": "scikit-learn",
                "hiperparametros_grid": config.GRID_TFT,
                "busca": "RandomizedSearchCV(n_iter=30, cv=3)"
            },
            "SARIMAX": {
                "tipo": "Série temporal (por país)",
                "biblioteca": "statsmodels",
                "parametros": {
                    "order": "(1, d, 1) onde d é determinado por ADF test",
                    "seasonal_order": "(0, 0, 0, 0)",
                    "exog": "Top 3 features por correlação com target"
                }
            },
            "LSTM": {
                "tipo": "MLP Regressor (por país)",
                "biblioteca": "scikit-learn",
                "parametros": {
                    "hidden_layer_sizes": "(128, 64, 32)",
                    "activation": "relu",
                    "solver": "adam",
                    "max_iter": 5000,
                    "early_stopping": True
                }
            },
            "Bayes_PartialPooling": {
                "tipo": "Bayesiano Hierárquico (partial pooling)",
                "biblioteca": "PyMC",
                "parametros": {
                    "n_samples": config.BAYESIAN_N_SAMPLES,
                    "n_tune": config.BAYESIAN_N_TUNE,
                    "target_accept": config.BAYESIAN_TARGET_ACCEPT,
                    "max_features": config.BAYESIAN_MAX_FEATURES,
                    "chains": config.BAYESIAN_CHAINS,
                    "sampler": "NUTS"
                },
                "formulacao": "y_{i,t} ~ Normal(alpha_i + X*beta_i, sigma); alpha_i ~ Normal(mu_alpha, sigma_alpha); beta_i ~ Normal(mu_beta, sigma_beta)"
            },
            "Bayes_CompletePooling": {
                "tipo": "Bayesiano Global (complete pooling)",
                "biblioteca": "PyMC",
                "parametros": {
                    "n_samples": min(config.BAYESIAN_N_SAMPLES, 800),
                    "n_tune": min(config.BAYESIAN_N_TUNE, 400),
                    "target_accept": config.BAYESIAN_TARGET_ACCEPT,
                    "max_features": config.BAYESIAN_MAX_FEATURES,
                    "chains": config.BAYESIAN_CHAINS,
                    "sampler": "NUTS",
                    "init": "advi"
                },
                "formulacao": "y_{i,t} ~ Normal(alpha + X*beta, sigma); alpha ~ Normal(y_mean, 10); beta ~ Normal(0, 5)"
            }
        },
        "resultados": {},
        "tempos_treino": training_times if training_times else {},
        "convergencia_bayesiana": bayesian_diagnostics if bayesian_diagnostics else {},
        "metricas_globais": {
            "descricao": "R², RMSE e MAE calculados no conjunto de validação (2017-2019)"
        }
    }

    # Adicionar resultados por dataset/estratégia
    if all_summaries:
        df_summary = pd.DataFrame(all_summaries)
        
        # Melhor modelo global
        if 'Global_R2' in df_summary.columns:
            valid = df_summary[df_summary['Global_R2'].notna()]
            if not valid.empty:
                best = valid.loc[valid['Global_R2'].idxmax()]
                metadata["melhor_modelo_global"] = {
                    "dataset": str(best.get('Dataset', '')),
                    "estrategia": str(best.get('Estrategia', '')),
                    "modelo": str(best.get('Modelo', '')),
                    "r2": float(best['Global_R2']),
                    "rmse": float(best.get('Global_RMSE', 0)),
                    "mae": float(best.get('Global_MAE', 0))
                }
                
                # Ranking por modelo
                ranking = valid.groupby('Modelo')['Global_R2'].agg(['mean', 'max', 'min', 'std']).round(4)
                metadata["ranking_modelos"] = ranking.to_dict('index')

        # Resultados detalhados
        for _, row in df_summary.iterrows():
            key = f"{row.get('Dataset', '')}_{row.get('Estrategia', '')}"
            if key not in metadata["resultados"]:
                metadata["resultados"][key] = {}
            metadata["resultados"][key][row.get('Modelo', '')] = {
                "global_r2": float(row['Global_R2']) if pd.notna(row.get('Global_R2')) else None,
                "global_rmse": float(row['Global_RMSE']) if pd.notna(row.get('Global_RMSE')) else None,
                "global_mae": float(row['Global_MAE']) if pd.notna(row.get('Global_MAE')) else None,
                "per_country_mean_r2": float(row['PerCountry_Mean_R2']) if pd.notna(row.get('PerCountry_Mean_R2')) else None,
                "per_country_median_r2": float(row['PerCountry_Median_R2']) if pd.notna(row.get('PerCountry_Median_R2')) else None,
                "n_countries": int(row['N_Countries']) if pd.notna(row.get('N_Countries')) else 0
            }

    save_metadata(metadata, os.path.join(output_dir, 'metadata_passo4.json'))
    return metadata
