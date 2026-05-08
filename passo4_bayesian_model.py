"""
Passo 4 - Modelo Bayesiano Hierárquico (PyMC)
==============================================================================
Implementa 2 estratégias Bayesianas viáveis computacionalmente:

1. PARTIAL POOLING (Hierárquico) - Parâmetros por país "puxados" para média global
   - Melhor para previsão POR PAÍS (partial pooling = empréstimo de força)
   - Gera previsão global E por país

2. COMPLETE POOLING (Global) - Parâmetros globais partilhados
   - Baseline Bayesiano para comparação
   - Rápido de treinar (poucos parâmetros)
   - Gera previsão global E por país (usando parâmetros globais)

NOTA: No Pooling foi removido por ser computacionalmente inviável
(61 países × 5 features = 366 parâmetros independentes com NUTS).
==============================================================================
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

try:
    import pymc as pm
    import arviz as az
    PYMC_AVAILABLE = True
except ImportError:
    PYMC_AVAILABLE = False
    print("⚠️ PyMC não instalado. Instale com: pip install pymc arviz")

from sklearn.preprocessing import StandardScaler
import passo4_model_train_config as config


# ═══════════════════════════════════════════════════════════════════════════════
# CLASSE BASE
# ═══════════════════════════════════════════════════════════════════════════════

class BaseBayesianModel:
    """Classe base para modelos Bayesianos."""
    
    def __init__(self, n_samples=1000, n_tune=500, target_accept=0.9, max_features=5, chains=2):
        self.n_samples = n_samples
        self.n_tune = n_tune
        self.target_accept = target_accept
        self.max_features = max_features
        self.chains = chains
        self.model = None
        self.trace = None
        self.scaler = StandardScaler()
        self.countries = None
        self.country_idx_map = None
        self.feature_cols = None
        self.strategy_type = None
    
    def _prepare_train_data(self, df, feature_cols, country_col, year_col):
        """Prepara dados para treino: escala features e cria índices de país."""
        # Limitar features para viabilidade computacional
        if len(feature_cols) > self.max_features:
            # Selecionar features com maior correlação com o target
            correlations = df[feature_cols].corrwith(df[config.TARGET_VAR]).abs()
            feature_cols = correlations.nlargest(self.max_features).index.tolist()
        
        self.feature_cols = feature_cols
        self.countries = sorted(df[country_col].unique())
        self.country_idx_map = {c: i for i, c in enumerate(self.countries)}
        
        # Divisão temporal
        train_mask = df[year_col] <= config.TRAIN_END_YEAR
        df_train = df[train_mask].copy()
        
        # Remover NaN
        cols_needed = feature_cols + [config.TARGET_VAR, country_col]
        df_train = df_train.dropna(subset=cols_needed)
        
        X = df_train[feature_cols].values
        y = df_train[config.TARGET_VAR].values
        country_idx = df_train[country_col].map(self.country_idx_map).values
        
        # Escalar
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y, country_idx, df_train
    
    def predict(self, df, feature_cols=None, country_col='pais'):
        """Faz previsão para novos dados."""
        if feature_cols is None:
            feature_cols = self.feature_cols
        
        # Usar apenas features que foram usadas no treino
        feat_to_use = [f for f in feature_cols if f in self.feature_cols]
        if not feat_to_use:
            feat_to_use = self.feature_cols
        
        df_pred = df.dropna(subset=feat_to_use + [country_col]).copy()
        if len(df_pred) == 0:
            return np.array([]), []
        
        X = df_pred[feat_to_use].values
        X_scaled = self.scaler.transform(X)
        countries = df_pred[country_col].values
        
        y_pred = self._predict_internal(X_scaled, countries)
        return y_pred, df_pred.index.tolist()


# ═══════════════════════════════════════════════════════════════════════════════
# 1. PARTIAL POOLING - Parâmetros hierárquicos (o melhor dos dois mundos)
# ═══════════════════════════════════════════════════════════════════════════════

class BayesianPartialPooling(BaseBayesianModel):
    """
    Modelo Bayesiano Hierárquico com PARTIAL POOLING.
    
    Parâmetros por país são "puxados" para a média global (shrinkage).
    Países com poucos dados → parâmetros mais próximos da média global.
    Países com muitos dados → parâmetros mais individualizados.
    
    Formulação:
        y_{i,t} ~ Normal(μ_{i,t}, σ)
        μ_{i,t} = α_i + X_{i,t} · β_i
        
        α_i ~ Normal(μ_α, σ_α)     ← Intercepto hierárquico
        β_i ~ Normal(μ_β, σ_β)     ← Coeficientes hierárquicos
        
        μ_α ~ Normal(0, 10)        ← Hiperprior global
        σ_α ~ HalfNormal(5)        ← Variabilidade entre países
        μ_β ~ Normal(0, 5)         ← Hiperprior coeficientes
        σ_β ~ HalfNormal(2)        ← Variabilidade coeficientes
        σ ~ HalfNormal(5)          ← Erro residual
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.strategy_type = "partial_pooling"
    
    def fit(self, df, feature_cols, country_col, year_col):
        X_scaled, y, country_idx, df_train = self._prepare_train_data(
            df, feature_cols, country_col, year_col)
        
        n_countries = len(self.countries)
        n_features = X_scaled.shape[1]
        
        print(f"     [Partial Pooling] Países: {n_countries} | Features: {n_features} | Amostras: {len(X_scaled)}")
        
        with pm.Model() as self.model:
            X_data = pm.Data('X', X_scaled)
            country_data = pm.Data('country_idx', country_idx.astype(int))
            
            # Hiperpriors (nível global)
            mu_alpha = pm.Normal('mu_alpha', mu=y.mean(), sigma=10)
            sigma_alpha = pm.HalfNormal('sigma_alpha', sigma=5)
            mu_beta = pm.Normal('mu_beta', mu=0, sigma=5, shape=n_features)
            sigma_beta = pm.HalfNormal('sigma_beta', sigma=2, shape=n_features)
            
            # Priors por país (hierárquicos)
            alpha = pm.Normal('alpha', mu=mu_alpha, sigma=sigma_alpha, shape=n_countries)
            beta = pm.Normal('beta', mu=mu_beta, sigma=sigma_beta, shape=(n_countries, n_features))
            sigma_y = pm.HalfNormal('sigma_y', sigma=5)
            
            # Likelihood
            mu = alpha[country_data] + pm.math.sum(beta[country_data] * X_data, axis=1)
            y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma_y, observed=y)
        
        print(f"     Executando MCMC (samples={self.n_samples}, tune={self.n_tune})...")
        with self.model:
            self.trace = pm.sample(
                draws=self.n_samples, tune=self.n_tune,
                target_accept=self.target_accept,
                cores=1, chains=self.chains, return_inferencedata=True,
                progressbar=False, random_seed=config.RANDOM_STATE
            )
        
        self._print_diagnostics()
        return self
    
    def _predict_internal(self, X_scaled, countries):
        alpha_post = self.trace.posterior['alpha'].mean(dim=['chain', 'draw']).values
        beta_post = self.trace.posterior['beta'].mean(dim=['chain', 'draw']).values
        
        y_pred = np.zeros(len(X_scaled))
        for i in range(len(X_scaled)):
            country = countries[i]
            if country in self.country_idx_map:
                c_idx = self.country_idx_map[country]
                y_pred[i] = alpha_post[c_idx] + np.dot(beta_post[c_idx], X_scaled[i])
            else:
                # País desconhecido: usar média global
                y_pred[i] = alpha_post.mean() + np.dot(beta_post.mean(axis=0), X_scaled[i])
        return y_pred
    
    def _print_diagnostics(self):
        try:
            rhat = az.rhat(self.trace)
            max_rhat = max(rhat['alpha'].max().values, rhat['beta'].max().values)
            print(f"     ✓ MCMC concluído! Max R-hat: {max_rhat:.4f}")
            if max_rhat > 1.1:
                print(f"     ⚠️ R-hat > 1.1 indica convergência incompleta")
        except:
            print(f"     ✓ MCMC concluído!")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. COMPLETE POOLING - Parâmetros globais partilhados (baseline)
# ═══════════════════════════════════════════════════════════════════════════════

class BayesianCompletePooling(BaseBayesianModel):
    """
    Modelo Bayesiano com COMPLETE POOLING.
    
    Todos os países partilham os mesmos parâmetros.
    Serve como baseline Bayesiano para comparação.
    
    Formulação:
        y_{i,t} ~ Normal(μ_{i,t}, σ)
        μ_{i,t} = α + X_{i,t} · β
        
        α ~ Normal(ȳ, 10)          ← Intercepto global
        β ~ Normal(0, 5)           ← Coeficientes globais
        σ ~ HalfNormal(5)          ← Erro residual
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.strategy_type = "complete_pooling"
    
    def fit(self, df, feature_cols, country_col, year_col):
        X_scaled, y, country_idx, df_train = self._prepare_train_data(
            df, feature_cols, country_col, year_col)
        
        n_features = X_scaled.shape[1]
        # Complete Pooling é simples - usar menos amostras (converge rápido)
        n_samples_cp = min(self.n_samples, 800)
        n_tune_cp = min(self.n_tune, 400)
        
        print(f"     [Complete Pooling] Features: {n_features} | Amostras: {len(X_scaled)}")
        
        with pm.Model() as self.model:
            X_data = pm.Data('X', X_scaled)
            
            # Parâmetros GLOBAIS (sem distinção por país)
            alpha = pm.Normal('alpha', mu=y.mean(), sigma=10)
            beta = pm.Normal('beta', mu=0, sigma=5, shape=n_features)
            sigma_y = pm.HalfNormal('sigma_y', sigma=5)
            
            # Likelihood
            mu = alpha + pm.math.dot(X_data, beta)
            y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma_y, observed=y)
        
        print(f"     Executando MCMC (samples={n_samples_cp}, tune={n_tune_cp})...")
        with self.model:
            self.trace = pm.sample(
                draws=n_samples_cp, tune=n_tune_cp,
                target_accept=self.target_accept,
                cores=1, chains=self.chains, return_inferencedata=True,
                progressbar=False, random_seed=config.RANDOM_STATE,
                init='advi'
            )
        
        self._print_diagnostics()
        return self
    
    def _predict_internal(self, X_scaled, countries):
        alpha_post = self.trace.posterior['alpha'].mean(dim=['chain', 'draw']).values
        beta_post = self.trace.posterior['beta'].mean(dim=['chain', 'draw']).values
        
        # Complete pooling: mesmos parâmetros para todos os países
        y_pred = alpha_post + X_scaled @ beta_post
        return y_pred
    
    def _print_diagnostics(self):
        try:
            rhat = az.rhat(self.trace)
            max_rhat = max(float(rhat['alpha'].values), float(rhat['beta'].max().values))
            print(f"     ✓ MCMC concluído! Max R-hat: {max_rhat:.4f}")
        except:
            print(f"     ✓ MCMC concluído!")


# ═══════════════════════════════════════════════════════════════════════════════
# FUNÇÃO PRINCIPAL - Treina ambas estratégias
# ═══════════════════════════════════════════════════════════════════════════════

def train_all_bayesian_models(df, feature_cols, country_col, year_col, 
                               n_samples=None, n_tune=None, target_accept=None,
                               max_features=None, chains=None):
    """
    Treina os 2 modelos Bayesianos (Partial Pooling + Complete Pooling).
    
    Retorna dict com resultados de cada estratégia.
    """
    if not PYMC_AVAILABLE:
        print("  ⚠️ PyMC não disponível. Pulando modelos Bayesianos.")
        return {}
    
    # Usar configurações do config ou parâmetros passados
    params = {
        'n_samples': n_samples or getattr(config, 'BAYESIAN_N_SAMPLES', 1000),
        'n_tune': n_tune or getattr(config, 'BAYESIAN_N_TUNE', 500),
        'target_accept': target_accept or getattr(config, 'BAYESIAN_TARGET_ACCEPT', 0.9),
        'max_features': max_features or getattr(config, 'BAYESIAN_MAX_FEATURES', 5),
        'chains': chains or getattr(config, 'BAYESIAN_CHAINS', 2),
    }
    
    results = {}
    
    # 1. PARTIAL POOLING (Hierárquico)
    print(f"\n  -> Treinando Bayesiano Partial Pooling (Hierárquico)...")
    try:
        model_pp = BayesianPartialPooling(**params)
        model_pp.fit(df, feature_cols, country_col, year_col)
        results['Bayes_PartialPooling'] = model_pp
    except Exception as e:
        print(f"     ❌ Erro no Partial Pooling: {e}")
    
    # 2. COMPLETE POOLING (Global)
    print(f"\n  -> Treinando Bayesiano Complete Pooling (Global)...")
    try:
        model_cp = BayesianCompletePooling(**params)
        model_cp.fit(df, feature_cols, country_col, year_col)
        results['Bayes_CompletePooling'] = model_cp
    except Exception as e:
        print(f"     ❌ Erro no Complete Pooling: {e}")
    
    return results
