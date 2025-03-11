import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Callable, Optional, Union, Tuple
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
import joblib
import os

class FeatureEngineer:
    def __init__(self, config: Optional[Dict] = None):
        self.config = {
            'dimensionality_reduction': None,
            'feature_selection': 'variance',
            'generate_features': True,
            'correlation_threshold': 0.95,
            'min_pca_components': 10,
            'verbosity': 1
        }
        if config:
            self.config.update(config)
        
        self.feature_pipeline = None
        self.fitted = False
        self.input_feature_names = []
        self.output_feature_names = []
        self.target_col = None
        
        self._setup_logging()
        self.logger.info("FeatureEngineer inicializado com sucesso.")

    def _setup_logging(self):
        self.logger = logging.getLogger("AutoFE.FeatureEngineer")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.logger.setLevel({0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}.get(self.config['verbosity'], logging.INFO))

    def _remove_high_correlation(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric_df = df.select_dtypes(include=['number'])
        if numeric_df.empty:
            return df
            
        try:
            corr_matrix = numeric_df.corr().abs()
            upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > self.config['correlation_threshold'])]
            
            if to_drop:
                self.logger.info(f"Removendo {len(to_drop)} colunas altamente correlacionadas: {to_drop[:5]}...")
                return df.drop(columns=to_drop, errors='ignore')
            return df
        except Exception as e:
            self.logger.warning(f"Erro ao calcular correlações: {e}. Retornando DataFrame original.")
            return df
    
    def _generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.config['generate_features']:
            return df

        num_data = df.select_dtypes(include=['number'])
        if num_data.empty:
            self.logger.warning("Nenhuma feature numérica encontrada. Pulando geração de features polinomiais.")
            return df

        try:
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
            return pd.concat([df, df_poly], axis=1)
        except Exception as e:
            self.logger.error(f"Erro ao gerar features polinomiais: {e}")
            return df

    def _setup_feature_pipeline(self, df: pd.DataFrame) -> None:
        pipeline_steps = []
        
        # Adicionar PCA se configurado
        if self.config['dimensionality_reduction'] == 'pca':
            n_components = min(self.config['min_pca_components'], df.shape[1])
            if n_components > 1:
                pipeline_steps.append(('pca', PCA(n_components=n_components)))
                self.logger.info(f"PCA configurado com {n_components} componentes")
            else:
                self.logger.warning("Número de features insuficiente para aplicar PCA. PCA será ignorado.")

        # Adicionar seleção de features se configurado
        if self.config['feature_selection'] == 'variance':
            pipeline_steps.append(('feature_selection', VarianceThreshold(threshold=0.01)))
            
        # Construir pipeline final
        self.feature_pipeline = Pipeline(pipeline_steps) if pipeline_steps else None

    def fit(self, df: pd.DataFrame, target_col: Optional[str] = None) -> 'FeatureEngineer':
        if df.empty:
            raise ValueError("Não é possível ajustar com um DataFrame vazio")
            
        self.target_col = target_col
        df_proc = df.copy()
        
        # Remover coluna alvo se presente
        if target_col and target_col in df_proc.columns:
            df_proc = df_proc.drop(columns=[target_col])
            self.logger.info(f"Coluna alvo '{target_col}' removida para processamento")
        
        # Aplicar transformações de engenharia de features
        df_proc = self._generate_features(df_proc)
        df_proc = self._remove_high_correlation(df_proc)

        if df_proc.empty:
            self.logger.error("Erro: O DataFrame está vazio após engenharia de features. Ajuste as configurações.")
            raise ValueError("O DataFrame está vazio após as transformações.")

        # Salvar nomes das features de entrada
        self.input_feature_names = df_proc.columns.tolist()
        
        # Configurar pipeline de features
        self._setup_feature_pipeline(df_proc)
        
        # Aplicar pipeline de features se existir
        if self.feature_pipeline:
            try:
                self.feature_pipeline.fit(df_proc)
                # Tentar obter os nomes das features de saída
                if hasattr(self.feature_pipeline, 'get_feature_names_out'):
                    self.output_feature_names = self.feature_pipeline.get_feature_names_out()
                else:
                    self.output_feature_names = [f"feature_{i}" for i in range(self.feature_pipeline.transform(df_proc).shape[1])]
            except Exception as e:
                self.logger.error(f"Erro ao ajustar o pipeline de features: {e}")
                raise
        else:
            # Se não há pipeline, as features de saída são as mesmas de entrada
            self.output_feature_names = self.input_feature_names
        
        self.fitted = True
        self.logger.info(f"FeatureEngineer ajustado com sucesso. Features de entrada: {len(self.input_feature_names)}, Features de saída: {len(self.output_feature_names)}")
        
        return self

    def transform(self, df: pd.DataFrame, target_col: Optional[str] = None) -> pd.DataFrame:
        if not self.fitted:
            raise ValueError("O FeatureEngineer precisa ser ajustado antes de transformar dados. Use .fit() primeiro.")

        df_proc = df.copy()
        target_data = None
        
        # Separar target se presente
        if target_col and target_col in df_proc.columns:
            target_data = df_proc[target_col].copy()
            df_proc = df_proc.drop(columns=[target_col])
        
        # Aplicar transformações de engenharia de features
        df_proc = self._generate_features(df_proc)
        df_proc = self._remove_high_correlation(df_proc)

        # Verificar compatibilidade das colunas
        self._check_columns_compatibility(df_proc)
        
        # Aplicar transformação do pipeline se existir
        if self.feature_pipeline:
            try:
                transformed_data = self.feature_pipeline.transform(df_proc)
                # Criar DataFrame com os dados transformados
                result_df = pd.DataFrame(
                    transformed_data, 
                    index=df_proc.index, 
                    columns=self.output_feature_names
                )
            except Exception as e:
                self.logger.error(f"Erro na transformação dos dados: {e}")
                raise
        else:
            # Se não há pipeline, o resultado é o próprio DataFrame processado
            result_df = df_proc
        
        # Adicionar coluna target se existir
        if target_data is not None:
            result_df[target_col] = target_data.loc[result_df.index]
        
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
            
    def save(self, filepath: str) -> None:
        if not self.fitted:
            raise ValueError("Não é possível salvar um FeatureEngineer não ajustado.")
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        joblib.dump(self, filepath)
        self.logger.info(f"FeatureEngineer salvo em {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'FeatureEngineer':
        feature_engineer = joblib.load(filepath)
        feature_engineer.logger.info(f"FeatureEngineer carregado de {filepath}")
        return feature_engineer

def create_feature_engineer(config: Optional[Dict] = None) -> FeatureEngineer:
    return FeatureEngineer(config)