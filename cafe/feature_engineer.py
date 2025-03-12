import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Optional, Union
from sklearn.decomposition import PCA
from sklearn.feature_selection import (
    VarianceThreshold, 
    SelectKBest, 
    SelectFromModel,
    SelectPercentile, 
    SelectFwe, 
    SelectFpr, 
    SelectFdr,
    mutual_info_classif, 
    mutual_info_regression,
    f_classif,
    f_regression,
    chi2,
    SelectorMixin
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import joblib
import os
from sklearn.base import BaseEstimator


class AdvancedFeatureSelector:
    """
    Classe auxiliar que implementa métodos avançados de seleção de features.
    Centraliza diferentes estratégias de seleção e fornece uma interface unificada.
    """
    def __init__(self, method: str = 'kbest', params: Optional[Dict] = None, task: str = 'classification'):
        """
        Inicializa o seletor de features com o método especificado.
        
        Args:
            method: Método de seleção ('kbest', 'percentile', 'model', 'fwe', 'fpr', 'fdr', 'variance')
            params: Parâmetros específicos do método
            task: Tipo de tarefa ('classification' ou 'regression')
        """
        self.method = method
        self.params = params or {}
        self.task = task
        self.selector = None
        self._configure_selector()
        
    def _configure_selector(self) -> None:
        """Configura o seletor de features com base no método e parâmetros."""
        # Determinar a função de pontuação baseada no tipo de tarefa
        if self.task == 'classification':
            score_func = self.params.get('score_func', mutual_info_classif)
            if isinstance(score_func, str):
                score_funcs = {
                    'mutual_info': mutual_info_classif,
                    'f_classif': f_classif,
                    'chi2': chi2
                }
                score_func = score_funcs.get(score_func, mutual_info_classif)
        else:  # regression
            score_func = self.params.get('score_func', mutual_info_regression)
            if isinstance(score_func, str):
                score_funcs = {
                    'mutual_info': mutual_info_regression,
                    'f_regression': f_regression
                }
                score_func = score_funcs.get(score_func, mutual_info_regression)
        
        # Configurar o seletor com base no método escolhido
        if self.method == 'kbest':
            k = self.params.get('k', 10)
            self.selector = SelectKBest(score_func=score_func, k=k)
            
        elif self.method == 'percentile':
            percentile = self.params.get('percentile', 20)
            self.selector = SelectPercentile(score_func=score_func, percentile=percentile)
            
        elif self.method == 'model':
            threshold = self.params.get('threshold', 'mean')
            # Escolher o modelo apropriado baseado na tarefa
            if self.task == 'classification':
                estimator = self.params.get('estimator', RandomForestClassifier(n_estimators=100, random_state=42))
            else:
                estimator = self.params.get('estimator', 
                                            RandomForestRegressor(n_estimators=100, random_state=42))
            self.selector = SelectFromModel(estimator=estimator, threshold=threshold)
            
        elif self.method == 'fwe':
            alpha = self.params.get('alpha', 0.05)
            self.selector = SelectFwe(score_func=score_func, alpha=alpha)
            
        elif self.method == 'fpr':
            alpha = self.params.get('alpha', 0.05)
            self.selector = SelectFpr(score_func=score_func, alpha=alpha)
            
        elif self.method == 'fdr':
            alpha = self.params.get('alpha', 0.05)
            self.selector = SelectFdr(score_func=score_func, alpha=alpha)
            
        elif self.method == 'variance':
            threshold = self.params.get('threshold', 0.1)
            self.selector = VarianceThreshold(threshold=threshold)
            
        else:
            raise ValueError(f"Método de seleção desconhecido: {self.method}")
    
    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Optional[Union[pd.Series, np.ndarray]] = None) -> 'AdvancedFeatureSelector':
        """
        Ajusta o seletor aos dados fornecidos.
        
        Args:
            X: Features para ajuste
            y: Target (opcional, mas necessário para alguns métodos)
            
        Returns:
            O próprio objeto AdvancedFeatureSelector (para encadeamento)
        """
        # Converter para numpy se for pandas
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        y_array = y.values if isinstance(y, pd.Series) else y
        
        self.selector.fit(X_array, y_array)
        return self
    
    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """
        Transforma os dados usando o seletor ajustado.
        
        Args:
            X: Features para transformação
            
        Returns:
            Dados transformados (mesmo tipo que a entrada)
        """
        is_pandas = isinstance(X, pd.DataFrame)
        X_array = X.values if is_pandas else X
        
        X_transformed = self.selector.transform(X_array)
        
        if is_pandas:
            # Recuperar os índices das features selecionadas
            if hasattr(self.selector, 'get_support'):
                selected_indices = self.selector.get_support(indices=True)
                selected_columns = X.columns[selected_indices]
                return pd.DataFrame(X_transformed, index=X.index, columns=selected_columns)
            else:
                # Fallback se o seletor não tiver get_support
                return pd.DataFrame(X_transformed, index=X.index)
                
        return X_transformed
    
    def fit_transform(self, X: Union[pd.DataFrame, np.ndarray], y: Optional[Union[pd.Series, np.ndarray]] = None) -> Union[pd.DataFrame, np.ndarray]:
        """
        Ajusta o seletor e transforma os dados em uma operação.
        
        Args:
            X: Features para ajuste e transformação
            y: Target (opcional, mas necessário para alguns métodos)
            
        Returns:
            Dados transformados (mesmo tipo que a entrada)
        """
        return self.fit(X, y).transform(X)
    
    def get_support(self, indices: bool = False) -> Union[np.ndarray, List[int]]:
        """
        Retorna a máscara de features selecionadas.
        
        Args:
            indices: Se True, retorna os índices em vez de uma máscara booleana
            
        Returns:
            Máscara ou índices das features selecionadas
        """
        if not hasattr(self.selector, 'get_support'):
            raise AttributeError(f"O seletor {self.method} não suporta o método get_support.")
        return self.selector.get_support(indices=indices)
    
    def get_feature_names_out(self, feature_names: Optional[List[str]] = None) -> List[str]:
        """
        Retorna os nomes das features após a seleção.
        
        Args:
            feature_names: Nomes originais das features
            
        Returns:
            Lista com os nomes das features selecionadas
        """
        if not hasattr(self.selector, 'get_support'):
            raise AttributeError(f"O seletor {self.method} não suporta obtenção de nomes de features.")
            
        if feature_names is None:
            return [f"feature_{i}" for i in self.get_support(indices=True)]
            
        support = self.get_support()
        return [name for i, name in enumerate(feature_names) if support[i]]


class FeatureEngineer:
    def __init__(self, config: Optional[Dict] = None):
        self.config = {
            'dimensionality_reduction': None,
            'feature_selection': None,  # Agora suporta métodos avançados
            'feature_selection_params': {},  # Parâmetros para seleção
            'generate_features': False,  # Alterado para False por padrão
            'correlation_threshold': 0.8,  # Alterado para limiar mais restritivo
            'min_pca_components': 10,
            'task': 'classification',  # Necessário para seleção apropriada
            'verbosity': 1
        }
        if config:
            self.config.update(config)
        
        self.feature_pipeline = None
        self.fitted = False
        self.input_feature_names = []
        self.output_feature_names = []
        self.target_col = None
        self.feature_selector = None
        
        self._setup_logging()
        self.logger.info(f"FeatureEngineer inicializado com configuração: {self.config}")

    def _setup_logging(self):
        self.logger = logging.getLogger("AutoFE.FeatureEngineer")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.logger.setLevel({0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}.get(self.config['verbosity'], logging.INFO))

    def _remove_high_correlation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove features com alta correlação utilizando o limiar definido na configuração.
        Retorna um DataFrame sem as features altamente correlacionadas.
        """
        self.logger.info(f"Aplicando remoção de alta correlação. Threshold: {self.config['correlation_threshold']}")
        
        numeric_df = df.select_dtypes(include=['number'])
        if numeric_df.empty:
            self.logger.info("Nenhuma feature numérica encontrada para análise de correlação")
            return df
            
        try:
            corr_matrix = numeric_df.corr().abs()
            
            # Criar matriz triangular superior
            upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            
            # Identificar colunas com correlação acima do limiar
            to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > self.config['correlation_threshold'])]
            
            if to_drop:
                self.logger.info(f"Removendo {len(to_drop)} colunas altamente correlacionadas: {to_drop[:5]}..." + 
                                 (f" e {len(to_drop) - 5} mais..." if len(to_drop) > 5 else ""))
                return df.drop(columns=to_drop, errors='ignore')
            else:
                self.logger.info("Nenhuma coluna altamente correlacionada encontrada")
            return df
        except Exception as e:
            self.logger.warning(f"Erro ao calcular correlações: {e}. Retornando DataFrame original.")
            return df
    
    def _generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Gera features polinomiais e de interação para melhorar o poder preditivo do modelo.
        Retorna o DataFrame original com novas features adicionadas.
        """
        if not self.config['generate_features']:
            self.logger.info("Geração de features desativada na configuração")
            return df

        self.logger.info("Iniciando geração de features polinomiais e de interação")
        num_data = df.select_dtypes(include=['number'])
        if num_data.empty:
            self.logger.warning("Nenhuma feature numérica encontrada. Pulando geração de features polinomiais.")
            return df

        try:
            # Limitar o número de features para evitar explosão combinatória
            if num_data.shape[1] > 10:
                self.logger.info(f"Muitas features numéricas ({num_data.shape[1]}). Limitando a 10 para geração polinomial.")
                # Selecionar as 10 features com maior variância
                variances = num_data.var()
                top_features = variances.nlargest(10).index.tolist()
                num_data = num_data[top_features]
            
            # Usar grau 2 e apenas interações (sem termos quadráticos) para limitar expansão
            poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
            new_features = poly.fit_transform(num_data)
            
            # Gera nomes de features mais descritivos
            feature_names = poly.get_feature_names_out(num_data.columns)
            
            # Remove colunas originais dos nomes das features transformadas
            poly_feature_names = [name for name in feature_names if ' ' in name]  # Features interativas contêm espaço
            
            # Cria DataFrame apenas com as novas features
            df_poly = pd.DataFrame(
                new_features[:, len(num_data.columns):],
                columns=poly_feature_names,
                index=df.index
            )
            
            self.logger.info(f"Geradas {len(poly_feature_names)} novas features polinomiais")
            result_df = pd.concat([df, df_poly], axis=1)
            
            # Aplicar remoção de correlação imediatamente para controlar dimensionalidade
            result_df = self._remove_high_correlation(result_df)
            
            return result_df
        except Exception as e:
            self.logger.error(f"Erro ao gerar features polinomiais: {e}")
            return df

    def _setup_feature_pipeline(self, df: pd.DataFrame, target_data=None) -> None:
        """
        Configura o pipeline de feature engineering baseado nas configurações.
        Adiciona etapas como PCA ou seleção de features conforme necessário.
        
        Args:
            df: DataFrame a ser transformado
            target_data: Dados alvo para métodos supervisionados
        """
        pipeline_steps = []
        
        # Adicionar PCA se configurado
        if self.config['dimensionality_reduction'] == 'pca':
            # Limitar o número de componentes para controlar dimensionalidade
            n_components = min(self.config['min_pca_components'], df.shape[1], int(df.shape[1] * 0.5))
            if n_components > 1:
                pipeline_steps.append(('pca', PCA(n_components=n_components)))
                self.logger.info(f"PCA configurado com {n_components} componentes")
            else:
                self.logger.warning("Número de features insuficiente para aplicar PCA. PCA será ignorado.")

        # Adicionar seleção de features se configurado
        feature_selection_method = self.config['feature_selection']
        if feature_selection_method:
            # Configurar o seletor de features avançado
            try:
                feature_selection_params = self.config.get('feature_selection_params', {})
                self.feature_selector = AdvancedFeatureSelector(
                    method=feature_selection_method,
                    params=feature_selection_params,
                    task=self.config['task']
                )
                pipeline_steps.append(('feature_selection', self.feature_selector))
                self.logger.info(f"Seleção de features configurada: {feature_selection_method}")
            except Exception as e:
                self.logger.error(f"Erro ao configurar seletor de features: {e}")
                
        # Construir pipeline final
        self.feature_pipeline = Pipeline(pipeline_steps) if pipeline_steps else None
        if self.feature_pipeline:
            self.logger.info(f"Pipeline de features configurado com {len(pipeline_steps)} etapas")
        else:
            self.logger.info("Nenhum pipeline de features configurado")

    def fit(self, df: pd.DataFrame, target_col: Optional[str] = None) -> 'FeatureEngineer':
        """
        Ajusta o FeatureEngineer aos dados, criando o pipeline de transformação
        e aprendendo os parâmetros necessários.
        
        Args:
            df: DataFrame com os dados de treino
            target_col: Nome da coluna alvo, se existir
            
        Returns:
            O próprio objeto FeatureEngineer, permitindo encadeamento de métodos
        """
        if df.empty:
            raise ValueError("Não é possível ajustar com um DataFrame vazio")
            
        self.target_col = target_col
        df_proc = df.copy()
        target_data = None
        
        # Remover coluna alvo se presente
        if target_col and target_col in df_proc.columns:
            target_data = df_proc[target_col].copy()
            df_proc = df_proc.drop(columns=[target_col])
            self.logger.info(f"Coluna alvo '{target_col}' removida para processamento")
        
        self.logger.info(f"Iniciando ajuste com DataFrame de formato {df_proc.shape}")
        
        # Aplicar transformações de engenharia de features
        # 1. Primeiro, gerar features se configurado
        df_proc = self._generate_features(df_proc)
        
        # 2. Depois, remover correlações altas para reduzir dimensionalidade
        df_proc = self._remove_high_correlation(df_proc)

        if df_proc.empty:
            self.logger.error("Erro: O DataFrame está vazio após engenharia de features. Ajuste as configurações.")
            raise ValueError("O DataFrame está vazio após as transformações.")

        # Salvar nomes das features de entrada
        self.input_feature_names = df_proc.columns.tolist()
        self.logger.info(f"Features de entrada salvas: {len(self.input_feature_names)}")
        
        # Configurar pipeline de features
        self._setup_feature_pipeline(df_proc, target_data)
        
        # Aplicar pipeline de features se existir
        if self.feature_pipeline:
            try:
                # Para métodos de seleção supervisionados, precisamos do target
                self.feature_pipeline.fit(df_proc, target_data)
                
                # Tentar obter os nomes das features de saída
                if hasattr(self.feature_pipeline, 'get_feature_names_out'):
                    self.output_feature_names = self.feature_pipeline.get_feature_names_out()
                elif isinstance(self.feature_pipeline.steps[-1][1], AdvancedFeatureSelector):
                    # Se o último passo for nosso seletor customizado
                    self.output_feature_names = self.feature_pipeline.steps[-1][1].get_feature_names_out(self.input_feature_names)
                else:
                    # Último recurso: usar nomes genéricos
                    transformed_shape = self.feature_pipeline.transform(df_proc).shape[1]
                    self.output_feature_names = [f"feature_{i}" for i in range(transformed_shape)]
                
                self.logger.info(f"Pipeline de features ajustado. Features de saída: {len(self.output_feature_names)}")
            except Exception as e:
                self.logger.error(f"Erro ao ajustar o pipeline de features: {e}")
                raise
        else:
            # Se não há pipeline, as features de saída são as mesmas de entrada
            self.output_feature_names = self.input_feature_names
            self.logger.info("Sem pipeline de features. Features de saída = features de entrada.")
        
        self.fitted = True
        self.logger.info(f"FeatureEngineer ajustado com sucesso. Features de entrada: {len(self.input_feature_names)}, Features de saída: {len(self.output_feature_names)}")
        
        return self

    def transform(self, df: pd.DataFrame, target_col: Optional[str] = None) -> pd.DataFrame:
        """
        Aplica as transformações de engenharia de features aos dados.
        
        Args:
            df: DataFrame a ser transformado
            target_col: Nome da coluna alvo, se existir
            
        Returns:
            DataFrame transformado
        """
        if not self.fitted:
            raise ValueError("O FeatureEngineer precisa ser ajustado antes de transformar dados. Use .fit() primeiro.")

        self.logger.info(f"Transformando DataFrame de formato {df.shape}")
        df_proc = df.copy()
        target_data = None
        
        # Separar target se presente
        if target_col and target_col in df_proc.columns:
            target_data = df_proc[target_col].copy()
            df_proc = df_proc.drop(columns=[target_col])
            self.logger.info(f"Coluna alvo '{target_col}' separada para preservação")
        
        # Aplicar transformações de engenharia de features
        # 1. Primeiro, gerar features se configurado
        df_proc = self._generate_features(df_proc)
        
        # 2. Depois, remover correlações altas para reduzir dimensionalidade
        df_proc = self._remove_high_correlation(df_proc)

        # Verificar compatibilidade das colunas
        self._check_columns_compatibility(df_proc)
        
        # Aplicar transformação do pipeline se existir
        if self.feature_pipeline:
            try:
                self.logger.info("Aplicando pipeline de features")
                transformed_data = self.feature_pipeline.transform(df_proc)
                # Criar DataFrame com os dados transformados
                result_df = pd.DataFrame(
                    transformed_data, 
                    index=df_proc.index, 
                    columns=self.output_feature_names[:transformed_data.shape[1]]
                )
                self.logger.info(f"Transformação completa. DataFrame resultante: {result_df.shape}")
            except Exception as e:
                self.logger.error(f"Erro na transformação dos dados: {e}")
                raise
        else:
            # Se não há pipeline, o resultado é o próprio DataFrame processado
            self.logger.info("Sem pipeline de features. Usando DataFrame processado diretamente.")
            result_df = df_proc
        
        # Adicionar coluna target se existir
        if target_data is not None:
            result_df[target_col] = target_data.loc[result_df.index]
            self.logger.info(f"Coluna target '{target_col}' reincorporada ao DataFrame")
        
        return result_df

    def _check_columns_compatibility(self, df: pd.DataFrame) -> None:
        """Verifica e ajusta as colunas para compatibilidade com o modelo ajustado"""
        # Verificar colunas ausentes
        missing_cols = set(self.input_feature_names) - set(df.columns)
        if missing_cols:
            self.logger.warning(f"Colunas ausentes na transformação: {missing_cols}. Adicionando com zeros.")
            for col in missing_cols:
                df[col] = 0
                
        # Manter apenas colunas conhecidas pelo modelo
        extra_cols = set(df.columns) - set(self.input_feature_names)
        if extra_cols:
            self.logger.warning(f"Colunas extras ignoradas: {extra_cols}")
            df_reduced = df[self.input_feature_names]
            # Substituir o dataframe original com o reduzido (in-place)
            df.drop(columns=df.columns, inplace=True)
            for col in df_reduced.columns:
                df[col] = df_reduced[col]
            
    def get_feature_importances(self, df: pd.DataFrame, target_col: Optional[str] = None) -> pd.DataFrame:
        """
        Calcula importâncias de features usando diferentes métodos.
        
        Args:
            df: DataFrame com os dados
            target_col: Nome da coluna alvo
            
        Returns:
            DataFrame com features e suas importâncias
        """
        if not target_col and self.target_col:
            target_col = self.target_col
            
        if not target_col:
            self.logger.error("Coluna alvo não especificada para calcular importâncias")
            raise ValueError("Coluna alvo necessária para calcular importâncias de features")
            
        if target_col not in df.columns:
            self.logger.error(f"Coluna alvo '{target_col}' não encontrada no DataFrame")
            raise ValueError(f"Coluna alvo '{target_col}' não encontrada")
            
        # Separar features e target
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Usar RandomForest para calcular importâncias
        if self.config['task'] == 'classification':
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            
        model.fit(X, y)
        
        # Criar DataFrame com importâncias
        importances = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importances
            
    def save(self, filepath: str) -> None:
        """Salva o objeto FeatureEngineer em um arquivo."""
        if not self.fitted:
            raise ValueError("Não é possível salvar um FeatureEngineer não ajustado.")
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        joblib.dump(self, filepath)
        self.logger.info(f"FeatureEngineer salvo em {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'FeatureEngineer':
        """Carrega um objeto FeatureEngineer de um arquivo."""
        feature_engineer = joblib.load(filepath)
        feature_engineer.logger.info(f"FeatureEngineer carregado de {filepath}")
        return feature_engineer
    
    def fit_transform(self, df: pd.DataFrame, target_col: Optional[str] = None) -> pd.DataFrame:
        """
        Ajusta o FeatureEngineer e transforma os dados em uma única operação.
        
        Args:
            df: DataFrame com os dados
            target_col: Nome da coluna alvo (opcional)
            
        Returns:
            DataFrame transformado
        """
        return self.fit(df, target_col).transform(df, target_col)

def create_feature_engineer(config: Optional[Dict] = None) -> FeatureEngineer:
    """Função auxiliar para criar uma instância de FeatureEngineer."""
    return FeatureEngineer(config)