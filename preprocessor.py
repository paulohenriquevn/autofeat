import pandas as pd
import numpy as np
import logging
import networkx as nx
from typing import List, Dict, Callable, Optional
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from scipy import stats
import joblib
import os

class DateTimeTransformer:
    """
    Transforma colunas de data/hora em características numéricas úteis para modelos.
    """
    def __init__(self, drop_original=True, extract_features=None):
        """
        Inicializa o transformador de data/hora.
        
        Args:
            drop_original: Se True, remove a coluna original após extrair os recursos
            extract_features: Lista de features de data/hora para extrair. Se None, 
                              extrai todas as features disponíveis.
        """
        self.drop_original = drop_original
        self.datetime_columns = []
        
        # Features de data/hora que podem ser extraídas
        self.available_features = [
            'year', 'month', 'day', 'weekday', 'quarter', 'is_weekend',
            'hour', 'minute', 'second', 'is_month_start', 'is_month_end',
            'is_year_start', 'is_year_end', 'days_in_month'
        ]
        
        # Se extract_features não for especificado, usar as principais features
        self.extract_features = extract_features or [
            'year', 'month', 'day', 'weekday', 'quarter', 'is_weekend'
        ]
        
        # Validar as features solicitadas
        invalid_features = [f for f in self.extract_features if f not in self.available_features]
        if invalid_features:
            raise ValueError(f"Features inválidas: {invalid_features}. "
                            f"Features disponíveis: {self.available_features}")

    def fit(self, X, y=None):
        """Identifica colunas de data/hora no DataFrame."""
        self.datetime_columns = X.select_dtypes(include=['datetime64']).columns.tolist()
        return self

    def transform(self, X):
        """
        Transforma colunas de data/hora em features numéricas.
        
        Args:
            X: DataFrame com colunas de data/hora
            
        Returns:
            DataFrame com as colunas de data/hora transformadas em features numéricas
        """
        X_transformed = X.copy()
        
        for col in self.datetime_columns:
            # Extrair recursos de data/hora
            for feature in self.extract_features:
                if feature == 'year':
                    X_transformed[f'{col}_year'] = X_transformed[col].dt.year
                elif feature == 'month':
                    X_transformed[f'{col}_month'] = X_transformed[col].dt.month
                elif feature == 'day':
                    X_transformed[f'{col}_day'] = X_transformed[col].dt.day
                elif feature == 'weekday':
                    X_transformed[f'{col}_weekday'] = X_transformed[col].dt.weekday
                elif feature == 'quarter':
                    X_transformed[f'{col}_quarter'] = X_transformed[col].dt.quarter
                elif feature == 'is_weekend':
                    X_transformed[f'{col}_is_weekend'] = (X_transformed[col].dt.weekday >= 5).astype(int)
                elif feature == 'hour':
                    X_transformed[f'{col}_hour'] = X_transformed[col].dt.hour
                elif feature == 'minute':
                    X_transformed[f'{col}_minute'] = X_transformed[col].dt.minute
                elif feature == 'second':
                    X_transformed[f'{col}_second'] = X_transformed[col].dt.second
                elif feature == 'is_month_start':
                    X_transformed[f'{col}_is_month_start'] = X_transformed[col].dt.is_month_start.astype(int)
                elif feature == 'is_month_end':
                    X_transformed[f'{col}_is_month_end'] = X_transformed[col].dt.is_month_end.astype(int)
                elif feature == 'is_year_start':
                    X_transformed[f'{col}_is_year_start'] = X_transformed[col].dt.is_year_start.astype(int)
                elif feature == 'is_year_end':
                    X_transformed[f'{col}_is_year_end'] = X_transformed[col].dt.is_year_end.astype(int)
                elif feature == 'days_in_month':
                    X_transformed[f'{col}_days_in_month'] = X_transformed[col].dt.days_in_month
            
            # Remover coluna original se configurado para isso
            if self.drop_original:
                X_transformed = X_transformed.drop(columns=[col])
        
        return X_transformed

    def fit_transform(self, X, y=None):
        """Ajusta e transforma em uma única operação."""
        return self.fit(X, y).transform(X)

class PreProcessor:
    def __init__(self, config: Optional[Dict] = None):
        self.config = {
            'missing_values_strategy': 'median',
            'outlier_method': 'iqr',  # Opções: 'zscore', 'iqr', 'isolation_forest'
            'categorical_strategy': 'onehot',
            'datetime_features': ['year', 'month', 'day', 'weekday', 'is_weekend'],
            'scaling': 'standard',
            'verbosity': 1,
        }
        if config:
            self.config.update(config)
        
        self.preprocessor = None
        self.datetime_transformer = None
        self.column_types = {}
        self.fitted = False
        self.feature_names = []
        self.target_col = None
        
        self._setup_logging()
        self.logger.info("PreProcessor inicializado com sucesso.")

    def _setup_logging(self):
        self.logger = logging.getLogger("AutoFE.PreProcessor")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.logger.setLevel({0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}.get(self.config['verbosity'], logging.INFO))

    def _identify_column_types(self, df: pd.DataFrame) -> Dict:
        """Identifica o tipo de cada coluna do DataFrame."""
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        self.logger.info(f"Colunas identificadas: {len(numeric_cols)} numéricas, "
                         f"{len(categorical_cols)} categóricas, {len(datetime_cols)} de data/hora")
        
        return {
            'numeric': numeric_cols, 
            'categorical': categorical_cols,
            'datetime': datetime_cols
        }
    
    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers do DataFrame usando o método especificado."""
        if df.empty:
            self.logger.warning("DataFrame vazio antes da remoção de outliers. Pulando esta etapa.")
            return df
            
        # Seleciona apenas colunas numéricas para tratamento de outliers
        numeric_df = df.select_dtypes(include=['number'])
        if numeric_df.empty:
            return df
            
        method = self.config.get('outlier_method', 'none')
        
        # Para esse exemplo, vamos desativar a remoção de outliers para evitar perda de amostras
        if method.lower() == 'none':
            return df
            
        # Se insistir em usar algum método, aplicaremos mas limitando a remoção
        if method == 'zscore':
            z_scores = np.abs(stats.zscore(numeric_df, nan_policy='omit'))
            # Usar um threshold mais permissivo
            mask = (z_scores < 5).all(axis=1)  # Alterado de 3 para 5
            filtered_df = df[mask]
        elif method == 'iqr':
            Q1 = numeric_df.quantile(0.25)
            Q3 = numeric_df.quantile(0.75)
            IQR = Q3 - Q1
            # Usar um threshold mais permissivo
            mask = ~((numeric_df < (Q1 - 3 * IQR)) | (numeric_df > (Q3 + 3 * IQR))).any(axis=1)  # Alterado de 1.5 para 3
            filtered_df = df[mask]
        elif method == 'isolation_forest':
            # Reduzir a taxa de contaminação
            clf = IsolationForest(contamination=0.01, random_state=42)  # Alterado de 0.05 para 0.01
            outliers = clf.fit_predict(numeric_df)
            filtered_df = df[outliers == 1]
        else:
            return df  # Caso o método não seja reconhecido, retorna o DataFrame original

        if filtered_df.empty or len(filtered_df) < len(df) * 0.8:  # Se remover mais de 20% das amostras
            self.logger.warning("Muitas amostras seriam removidas na remoção de outliers! Retornando DataFrame original.")
            return df

        return filtered_df
    
    def _process_datetime_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Processa colunas de data/hora, extraindo características numéricas úteis.
        
        Args:
            df: DataFrame com colunas de data/hora
            
        Returns:
            DataFrame com colunas de data/hora transformadas em características numéricas
        """
        if not self.column_types.get('datetime'):
            return df
            
        datetime_cols = self.column_types['datetime']
        if not datetime_cols:
            return df
            
        self.logger.info(f"Processando {len(datetime_cols)} colunas de data/hora: {datetime_cols}")
        
        # Inicializar o transformador se não existir
        if self.datetime_transformer is None:
            self.datetime_transformer = DateTimeTransformer(
                extract_features=self.config.get('datetime_features')
            )
            self.datetime_transformer.fit(df)
        
        # Aplicar a transformação
        return self.datetime_transformer.transform(df)
    
    def _build_transformers(self) -> List:
        """Constrói os transformadores para colunas numéricas e categóricas"""
        # Configurar imputer
        if self.config['missing_values_strategy'] == 'knn':
            num_imputer = KNNImputer()
        else:
            num_imputer = SimpleImputer(strategy=self.config['missing_values_strategy'])
        
        # Configurar scaler
        scalers = {
            'standard': StandardScaler(), 
            'minmax': MinMaxScaler(), 
            'robust': RobustScaler()
        }
        scaler = scalers.get(self.config['scaling'], 'passthrough')

        # Pipeline para features numéricas
        numeric_transformer = Pipeline([
            ('imputer', num_imputer),
            ('scaler', scaler)
        ])

        # Pipeline para features categóricas
        categorical_encoder = (
            OneHotEncoder(handle_unknown='ignore', sparse_output=False) 
            if self.config['categorical_strategy'] == 'onehot' 
            else OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        )
        
        categorical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', categorical_encoder)
        ])

        # Montar transformers
        transformers = []
        if self.column_types['numeric']:
            transformers.append(('num', numeric_transformer, self.column_types['numeric']))
        if self.column_types['categorical']:
            transformers.append(('cat', categorical_transformer, self.column_types['categorical']))
            
        return transformers

    def fit(self, df: pd.DataFrame, target_col: Optional[str] = None) -> 'PreProcessor':
        """
        Ajusta o preprocessador aos dados, aprendendo os parâmetros necessários para as transformações.
        
        Args:
            df: DataFrame com os dados de treinamento
            target_col: Nome da coluna alvo (opcional)
            
        Returns:
            A própria instância do PreProcessor, permitindo encadear métodos
        """
        if df.empty:
            raise ValueError("Não é possível ajustar com um DataFrame vazio")
            
        self.target_col = target_col
        df_proc = df.copy()
        
        # Remover coluna alvo se presente
        if target_col and target_col in df_proc.columns:
            df_proc = df_proc.drop(columns=[target_col])
            self.logger.info(f"Coluna alvo '{target_col}' removida para processamento")
        
        # Aplicar remoção de outliers
        df_proc = self._remove_outliers(df_proc)

        if df_proc.empty:
            self.logger.error("Erro: O DataFrame está vazio após pré-processamento. Ajuste as configurações.")
            raise ValueError("O DataFrame está vazio após as transformações.")

        # Identificar tipos de colunas
        self.column_types = self._identify_column_types(df_proc)
        
        # Processar colunas de data/hora
        if self.column_types.get('datetime'):
            df_proc = self._process_datetime_columns(df_proc)
            
            # Atualizar tipos de colunas após o processamento de datas
            self.column_types = self._identify_column_types(df_proc)

        # Configurar pipeline de transformação
        transformers = self._build_transformers()
        self.preprocessor = ColumnTransformer(transformers, remainder='passthrough')
        
        try:
            self.preprocessor.fit(df_proc)
            self.feature_names = df_proc.columns.tolist()
            self.fitted = True
            self.logger.info(f"Preprocessador ajustado com sucesso com {len(self.feature_names)} features")
            return self
        except Exception as e:
            self.logger.error(f"Erro ao ajustar o preprocessador: {e}")
            raise

    def transform(self, df: pd.DataFrame, target_col: Optional[str] = None) -> pd.DataFrame:
        """
        Aplica as transformações aprendidas a um conjunto de dados.
        
        Args:
            df: DataFrame a ser transformado
            target_col: Nome da coluna alvo (opcional)
            
        Returns:
            DataFrame transformado
        """
        if not self.fitted:
            raise ValueError("O preprocessador precisa ser ajustado antes de transformar dados. Use .fit() primeiro.")

        df_proc = df.copy()
        target_data = None
        
        # Separar target se presente
        if target_col and target_col in df_proc.columns:
            target_data = df_proc[target_col].copy()
            df_proc = df_proc.drop(columns=[target_col])
        
        # Aplicar remoção de outliers
        df_proc = self._remove_outliers(df_proc)

        # Processar colunas de data/hora
        if self.column_types.get('datetime'):
            df_proc = self._process_datetime_columns(df_proc)

        # Verificar e ajustar colunas para compatibilidade com o modelo de preprocessamento
        self._check_columns_compatibility(df_proc)
        
        # Aplicar transformação
        try:
            df_transformed = self.preprocessor.transform(df_proc)
            
            # Determinar nomes das colunas
            if hasattr(self.preprocessor, 'get_feature_names_out'):
                feature_names = self.preprocessor.get_feature_names_out()
            else:
                feature_names = [f"feature_{i}" for i in range(df_transformed.shape[1])]

            # Criar DataFrame com os dados transformados
            result_df = pd.DataFrame(
                df_transformed, 
                index=df_proc.index, 
                columns=feature_names
            )
            
            # Adicionar coluna target se existir
            if target_data is not None:
                result_df[target_col] = target_data.loc[result_df.index]
                
            return result_df
            
        except Exception as e:
            self.logger.error(f"Erro na transformação dos dados: {e}")
            raise

    def _check_columns_compatibility(self, df: pd.DataFrame) -> None:
        """Verifica e ajusta as colunas para compatibilidade com o modelo ajustado"""
        # Verificar colunas ausentes
        missing_cols = set(self.feature_names) - set(df.columns)
        if missing_cols:
            self.logger.warning(f"Colunas ausentes na transformação: {missing_cols}. Adicionando com zeros.")
            for col in missing_cols:
                df[col] = 0
                
        # Manter apenas colunas conhecidas pelo modelo
        extra_cols = set(df.columns) - set(self.feature_names)
        if extra_cols:
            self.logger.warning(f"Colunas extras ignoradas: {extra_cols}")
    
    def fit_transform(self, df: pd.DataFrame, target_col: Optional[str] = None) -> pd.DataFrame:
        """
        Ajusta o preprocessador e transforma os dados em uma única operação.
        
        Args:
            df: DataFrame com os dados
            target_col: Nome da coluna alvo (opcional)
            
        Returns:
            DataFrame transformado
        """
        return self.fit(df, target_col).transform(df, target_col)
            
    def save(self, filepath: str) -> None:
        """
        Salva o preprocessador em um arquivo para uso futuro.
        
        Args:
            filepath: Caminho do arquivo onde o preprocessador será salvo
        """
        if not self.fitted:
            raise ValueError("Não é possível salvar um preprocessador não ajustado.")
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        joblib.dump(self, filepath)
        self.logger.info(f"Preprocessador salvo em {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'PreProcessor':
        """
        Carrega um preprocessador previamente salvo.
        
        Args:
            filepath: Caminho do arquivo onde o preprocessador foi salvo
            
        Returns:
            Instância de PreProcessor carregada
        """
        preprocessor = joblib.load(filepath)
        preprocessor.logger.info(f"Preprocessador carregado de {filepath}")
        return preprocessor


def create_preprocessor(config: Optional[Dict] = None) -> PreProcessor:
    """
    Função auxiliar para criar uma instância de PreProcessor com configurações opcionais.
    
    Args:
        config: Dicionário com configurações personalizadas
        
    Returns:
        Instância configurada do PreProcessor
    """
    return PreProcessor(config)