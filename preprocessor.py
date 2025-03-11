### preprocessor.py ###
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

class PreProcessor:
    def __init__(self, config: Optional[Dict] = None):
        self.config = {
            'missing_values_strategy': 'median',
            'outlier_method': 'iqr',  # Opções: 'zscore', 'iqr', 'isolation_forest'
            'categorical_strategy': 'onehot',
            'scaling': 'standard',
            'verbosity': 1,
        }
        if config:
            self.config.update(config)
        
        self.preprocessor = None
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
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        return {'numeric': numeric_cols, 'categorical': categorical_cols}
    
    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            self.logger.warning("DataFrame vazio antes da remoção de outliers. Pulando esta etapa.")
            return df
            
        # Seleciona apenas colunas numéricas para tratamento de outliers
        numeric_df = df.select_dtypes(include=['number'])
        if numeric_df.empty:
            return df
            
        method = self.config['outlier_method']
        if method == 'zscore':
            z_scores = np.abs(stats.zscore(numeric_df, nan_policy='omit'))
            mask = (z_scores < 3).all(axis=1)
            filtered_df = df[mask]
        elif method == 'iqr':
            Q1 = numeric_df.quantile(0.25)
            Q3 = numeric_df.quantile(0.75)
            IQR = Q3 - Q1
            mask = ~((numeric_df < (Q1 - 1.5 * IQR)) | (numeric_df > (Q3 + 1.5 * IQR))).any(axis=1)
            filtered_df = df[mask]
        elif method == 'isolation_forest':
            clf = IsolationForest(contamination=0.05, random_state=42)
            outliers = clf.fit_predict(numeric_df)
            filtered_df = df[outliers == 1]
        else:
            return df  # Caso o método não seja reconhecido, retorna o DataFrame original

        if filtered_df.empty:
            self.logger.warning("Todas as amostras foram removidas na remoção de outliers! Retornando DataFrame original.")
            return df

        return filtered_df

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

        self.column_types = self._identify_column_types(df_proc)
        self.logger.info(f"Colunas identificadas: {len(self.column_types['numeric'])} numéricas, {len(self.column_types['categorical'])} categóricas")

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
            
    def save(self, filepath: str) -> None:
        if not self.fitted:
            raise ValueError("Não é possível salvar um preprocessador não ajustado.")
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        joblib.dump(self, filepath)
        self.logger.info(f"Preprocessador salvo em {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'PreProcessor':
        preprocessor = joblib.load(filepath)
        preprocessor.logger.info(f"Preprocessador carregado de {filepath}")
        return preprocessor


def create_preprocessor(config: Optional[Dict] = None) -> PreProcessor:
    return PreProcessor(config)