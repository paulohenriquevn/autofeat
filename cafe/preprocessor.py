import pandas as pd
import numpy as np
import logging
import networkx as nx
from typing import List, Dict, Callable, Optional, Union, Any
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, OrdinalEncoder,
    Binarizer, KernelCenterer, MaxAbsScaler, Normalizer, PowerTransformer,
    QuantileTransformer, FunctionTransformer, LabelBinarizer, LabelEncoder,
    MultiLabelBinarizer, KBinsDiscretizer
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from scipy import stats
import joblib
import os


# Implementação da classe TargetEncoder, já que ela não faz parte do scikit-learn padrão
class TargetEncoder:
    """
    Target Encoder que substitui uma variável categórica pela média do target para cada categoria.
    Útil para variáveis categóricas de alta cardinalidade.
    """
    def __init__(self, smoothing: float = 1.0, min_samples_leaf: int = 1, noise_level: float = 0.0):
        """
        Inicializa o Target Encoder.
        
        Args:
            smoothing: Fator de suavização (regularização)
            min_samples_leaf: Número mínimo de amostras por categoria
            noise_level: Nível de ruído para adicionar (reduz overfitting)
        """
        self.smoothing = smoothing
        self.min_samples_leaf = min_samples_leaf
        self.noise_level = noise_level
        self.mapping = {}
        self.prior = None
        
    def fit(self, X: Union[pd.Series, np.ndarray], y: Union[pd.Series, np.ndarray]) -> 'TargetEncoder':
        """
        Ajusta o encoder aos dados.
        
        Args:
            X: Features categóricas
            y: Target
            
        Returns:
            O próprio objeto TargetEncoder (para encadeamento)
        """
        # Converter para Series se necessário
        if isinstance(X, np.ndarray):
            X = pd.Series(X)
        if isinstance(y, np.ndarray):
            y = pd.Series(y)
            
        # Calcular a média global (prior)
        self.prior = y.mean()
        
        # Calcular a média do target por categoria
        stats = pd.DataFrame({'X': X, 'y': y}).groupby('X')['y']
        count = stats.count()
        mean = stats.mean()
        
        # Aplicar suavização bayesiana
        smoove = 1 / (1 + np.exp(-(count - self.min_samples_leaf) / self.smoothing))
        self.mapping = dict(mean * smoove + self.prior * (1 - smoove))
        
        return self
        
    def transform(self, X: Union[pd.Series, np.ndarray]) -> np.ndarray:
        """
        Transforma os dados categóricos em valores numéricos.
        
        Args:
            X: Features categóricas
            
        Returns:
            Array com valores codificados
        """
        if isinstance(X, np.ndarray):
            X = pd.Series(X)
            
        # Aplicar a transformação baseada no mapping
        encoded = X.map(self.mapping).fillna(self.prior).values.reshape(-1, 1)
        
        # Adicionar ruído se configurado
        if self.noise_level > 0:
            noise = np.random.normal(0, self.noise_level, encoded.shape)
            encoded = encoded + noise
            
        return encoded
        
    def fit_transform(self, X: Union[pd.Series, np.ndarray], y: Union[pd.Series, np.ndarray]) -> np.ndarray:
        """
        Ajusta o encoder e transforma os dados em uma única operação.
        
        Args:
            X: Features categóricas
            y: Target
            
        Returns:
            Array com valores codificados
        """
        return self.fit(X, y).transform(X)


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
            'categorical_strategy': 'onehot',  # Opções ampliadas: 'onehot', 'ordinal', 'target', 'label', 'binary'
            'datetime_features': ['year', 'month', 'day', 'weekday', 'is_weekend'],
            'scaling': 'standard',  # Opções ampliadas: 'standard', 'minmax', 'robust', 'maxabs', 'power', 'quantile', 'normalize'
            'numeric_transformers': [],  # Lista de transformadores adicionais para features numéricas
            'verbosity': 1,
            # Novos parâmetros para configurações específicas
            'scaling_params': {},  # Parâmetros para o scaler escolhido
            'binarizer_threshold': 0.5,  # Threshold para Binarizer
            'kbins_n_bins': 5,  # Número de bins para KBinsDiscretizer
            'kbins_encode': 'ordinal',  # Estratégia de encoding para KBinsDiscretizer: 'onehot', 'ordinal', 'onehot-dense'
            'power_transformer_method': 'yeo-johnson',  # Método para PowerTransformer: 'yeo-johnson' ou 'box-cox'
            'quantile_n_quantiles': 1000,  # Número de quantis para QuantileTransformer
            'target_encoding_smoothing': 10.0,  # Fator de suavização para TargetEncoder
        }
        if config:
            self.config.update(config)
        
        self.preprocessor = None
        self.datetime_transformer = None
        self.column_types = {}
        self.fitted = False
        self.feature_names = []
        self.target_col = None
        self.target_data = None  # Armazenar os dados target para TargetEncoder
        
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
    
    def _get_scaler(self, scaling_type: str) -> Any:
        """
        Retorna o scaler apropriado baseado na configuração.
        
        Args:
            scaling_type: Tipo de scaling a ser usado
            
        Returns:
            Instância do scaler configurado
        """
        scaling_params = self.config.get('scaling_params', {})
        
        if scaling_type == 'standard':
            return StandardScaler(**scaling_params)
        elif scaling_type == 'minmax':
            return MinMaxScaler(**scaling_params)
        elif scaling_type == 'robust':
            return RobustScaler(**scaling_params)
        elif scaling_type == 'maxabs':
            return MaxAbsScaler(**scaling_params)
        elif scaling_type == 'power':
            method = self.config.get('power_transformer_method', 'yeo-johnson')
            return PowerTransformer(method=method, **scaling_params)
        elif scaling_type == 'quantile':
            n_quantiles = self.config.get('quantile_n_quantiles', 1000)
            return QuantileTransformer(n_quantiles=n_quantiles, **scaling_params)
        elif scaling_type == 'normalize':
            return Normalizer(**scaling_params)
        elif scaling_type == 'kernel':
            return KernelCenterer()
        else:
            self.logger.warning(f"Tipo de scaling '{scaling_type}' não reconhecido. Usando StandardScaler.")
            return StandardScaler()
    
    def _get_additional_numeric_transformers(self) -> List:
        """
        Retorna transformadores adicionais para features numéricas.
        
        Returns:
            Lista de transformadores adicionais configurados
        """
        transformers = []
        
        # Adicionar transformadores solicitados na configuração
        for transformer_name in self.config.get('numeric_transformers', []):
            if transformer_name == 'binarizer':
                threshold = self.config.get('binarizer_threshold', 0.5)
                transformers.append(('binarizer', Binarizer(threshold=threshold)))
            elif transformer_name == 'kbins':
                n_bins = self.config.get('kbins_n_bins', 5)
                encode = self.config.get('kbins_encode', 'ordinal')
                transformers.append(('kbins', KBinsDiscretizer(n_bins=n_bins, encode=encode)))
            elif transformer_name == 'function':
                # Exemplo de transformador personalizado - usuário precisaria fornecer a função através de scaling_params
                if 'function_transformer' in self.config.get('scaling_params', {}):
                    func = self.config['scaling_params']['function_transformer']
                    transformers.append(('function', FunctionTransformer(func=func)))
                else:
                    self.logger.warning("FunctionTransformer solicitado, mas função não fornecida em scaling_params.")
        
        return transformers
    
    def _get_categorical_transformer(self) -> Pipeline:
        """
        Retorna o transformador apropriado para features categóricas baseado na configuração.
        
        Returns:
            Pipeline com o transformador categórico configurado
        """
        strategy = self.config.get('categorical_strategy', 'onehot')
        
        # Pipeline base com imputação
        steps = [('imputer', SimpleImputer(strategy='most_frequent'))]
        
        # Adicionar o encoder apropriado
        if strategy == 'onehot':
            steps.append(('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)))
        elif strategy == 'ordinal':
            steps.append(('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)))
        elif strategy == 'target':
            # Target Encoder requer dados de target para treinamento
            if self.target_data is not None:
                smoothing = self.config.get('target_encoding_smoothing', 10.0)
                steps.append(('encoder', TargetEncoder(smoothing=smoothing)))
            else:
                self.logger.warning("Target encoding solicitado, mas dados de target não disponíveis. Usando OneHotEncoder.")
                steps.append(('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)))
        elif strategy == 'label':
            steps.append(('encoder', LabelEncoder()))
        elif strategy == 'binary':
            steps.append(('encoder', LabelBinarizer()))
        elif strategy == 'multilabel':
            steps.append(('encoder', MultiLabelBinarizer()))
        else:
            self.logger.warning(f"Estratégia categórica '{strategy}' não reconhecida. Usando OneHotEncoder.")
            steps.append(('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)))
        
        return Pipeline(steps)

    def _build_transformers(self) -> List:
        """Constrói os transformadores para colunas numéricas e categóricas"""
        # Configurar imputer
        if self.config['missing_values_strategy'] == 'knn':
            num_imputer = KNNImputer()
        else:
            num_imputer = SimpleImputer(strategy=self.config['missing_values_strategy'])
        
        # Obter scaler configurado
        scaler = self._get_scaler(self.config['scaling'])

        # Pipeline base para features numéricas
        numeric_steps = [('imputer', num_imputer), ('scaler', scaler)]
        
        # Adicionar transformadores adicionais
        numeric_steps.extend(self._get_additional_numeric_transformers())
        
        # Pipeline para features numéricas
        numeric_transformer = Pipeline(numeric_steps)

        # Pipeline para features categóricas
        categorical_transformer = self._get_categorical_transformer()

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
        
        # Remover coluna alvo se presente e preservar para target encoding
        if target_col and target_col in df_proc.columns:
            self.target_data = df_proc[target_col].copy()
            df_proc = df_proc.drop(columns=[target_col])
            self.logger.info(f"Coluna alvo '{target_col}' removida para processamento e preservada para target encoding")
        
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
            # Se estivermos usando target encoding, configurar manualmente
            if self.config.get('categorical_strategy') == 'target' and self.target_data is not None:
                # Ajustar normalmente o preprocessor para colunas numéricas
                # Para categóricas, precisamos fazer o fit manualmente com os dados do target
                self.preprocessor.fit(df_proc)
                
                # Para cada coluna categórica, tentar ajustar o target encoder separadamente
                for col in self.column_types.get('categorical', []):
                    try:
                        encoder = TargetEncoder(smoothing=self.config.get('target_encoding_smoothing', 10.0))
                        encoder.fit(df_proc[col], self.target_data)
                        # Armazenar o encoder ajustado
                        if not hasattr(self, 'target_encoders'):
                            self.target_encoders = {}
                        self.target_encoders[col] = encoder
                    except Exception as e:
                        self.logger.warning(f"Erro ao ajustar TargetEncoder para coluna {col}: {e}")
            else:
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
            # Caso especial: se estamos usando target encoding
            if hasattr(self, 'target_encoders') and self.config.get('categorical_strategy') == 'target':
                # Aplicar transformação para colunas numéricas normalmente
                df_transformed = df_proc.copy()
                
                # Para categóricas, aplicar cada target encoder separadamente
                for col, encoder in self.target_encoders.items():
                    if col in df_transformed.columns:
                        encoded_values = encoder.transform(df_transformed[col])
                        df_transformed[f"{col}_encoded"] = encoded_values
                        df_transformed = df_transformed.drop(columns=[col])
                
                # Aplicar outras transformações para colunas numéricas
                numeric_cols = df_transformed.select_dtypes(include=['number']).columns.tolist()
                if numeric_cols:
                    numeric_transformer = self.preprocessor.transformers_[0][1]  # Acessar transformador numérico
                    numeric_transformed = numeric_transformer.transform(df_transformed[numeric_cols])
                    
                    for i, col in enumerate(numeric_cols):
                        df_transformed[col] = numeric_transformed[:, i]
            else:
                # Caminho normal: aplicar todo o preprocessor
                df_transformed_array = self.preprocessor.transform(df_proc)
                
                # Determinar nomes das colunas
                if hasattr(self.preprocessor, 'get_feature_names_out'):
                    feature_names = self.preprocessor.get_feature_names_out()
                else:
                    feature_names = [f"feature_{i}" for i in range(df_transformed_array.shape[1])]
                
                # Criar DataFrame com os dados transformados
                df_transformed = pd.DataFrame(df_transformed_array, index=df_proc.index, columns=feature_names)
            
            # Adicionar coluna target se existir
            if target_data is not None:
                df_transformed[target_col] = target_data.loc[df_transformed.index]
                
            return df_transformed
            
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
        
    def get_transformer_description(self) -> Dict[str, Any]:
        """
        Retorna uma descrição dos transformadores configurados e ativos.
        
        Returns:
            Dicionário com descrição dos transformadores ativos
        """
        if not self.fitted:
            return {"status": "não ajustado"}
            
        description = {
            "numeric_columns": self.column_types.get('numeric', []),
            "categorical_columns": self.column_types.get('categorical', []),
            "datetime_columns": self.column_types.get('datetime', []),
            "transformers": {
                "scaling": self.config['scaling'],
                "missing_values": self.config['missing_values_strategy'],
                "categorical_strategy": self.config['categorical_strategy'],
                "additional_transformers": self.config.get('numeric_transformers', [])
            },
            "output_features": len(self.feature_names),
        }
        
        return description


def create_preprocessor(config: Optional[Dict] = None) -> PreProcessor:
    """
    Função auxiliar para criar uma instância de PreProcessor com configurações opcionais.
    
    Args:
        config: Dicionário com configurações personalizadas
        
    Returns:
        Instância configurada do PreProcessor
    """
    return PreProcessor(config)