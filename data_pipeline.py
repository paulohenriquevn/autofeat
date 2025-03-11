import pandas as pd
import logging
from typing import Dict, Optional, Union, Tuple, List
from preprocessor import PreProcessor
from feature_engineer import FeatureEngineer

class DataPipeline:
    """
    Classe que combina PreProcessor e FeatureEngineer para criar um pipeline completo
    de processamento e engenharia de features.
    """
    def __init__(self, preprocessor_config: Optional[Dict] = None, feature_engineer_config: Optional[Dict] = None):
        self.preprocessor = PreProcessor(preprocessor_config)
        self.feature_engineer = FeatureEngineer(feature_engineer_config)
        self.fitted = False
        self.target_col = None
        
        self.logger = logging.getLogger("AutoFE.DataPipeline")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        
        self.logger.info("DataPipeline inicializado com sucesso.")
        
    def fit(self, df: pd.DataFrame, target_col: Optional[str] = None) -> 'DataPipeline':
        """
        Ajusta o pipeline completo de processamento de dados e engenharia de features.
        
        Args:
            df: DataFrame com os dados de treinamento
            target_col: Nome da coluna alvo (opcional)
            
        Returns:
            A própria instância do DataPipeline
        """
        if df.empty:
            raise ValueError("Não é possível ajustar com um DataFrame vazio")
        
        self.target_col = target_col
        self.logger.info(f"Iniciando ajuste do pipeline com {df.shape[0]} amostras e {df.shape[1]} features")
        
        # Ajustar o preprocessador
        self.preprocessor.fit(df, target_col=target_col)
        
        # Transformar os dados com o preprocessador
        df_preprocessed = self.preprocessor.transform(df, target_col=target_col)
        
        # Ajustar o feature engineer com os dados preprocessados
        self.feature_engineer.fit(df_preprocessed, target_col=target_col)
        
        self.fitted = True
        self.logger.info("Pipeline completo ajustado com sucesso")
        
        return self
    
    def transform(self, df: pd.DataFrame, target_col: Optional[str] = None) -> pd.DataFrame:
        """
        Transforma dados usando o pipeline completo.
        
        Args:
            df: DataFrame com os dados a serem transformados
            target_col: Nome da coluna alvo (opcional)
            
        Returns:
            DataFrame com os dados transformados
        """
        if not self.fitted:
            raise ValueError("O pipeline precisa ser ajustado antes de transformar dados. Use .fit() primeiro.")
        
        target = target_col or self.target_col
        self.logger.info(f"Transformando dados com {df.shape[0]} amostras")
        
        # Aplicar preprocessamento
        df_preprocessed = self.preprocessor.transform(df, target_col=target)
        
        # Aplicar engenharia de features
        df_transformed = self.feature_engineer.transform(df_preprocessed, target_col=target)
        
        self.logger.info(f"Transformação concluída. Resultado: {df_transformed.shape[0]} amostras, {df_transformed.shape[1]} features")
        
        return df_transformed
    
    def fit_transform(self, df: pd.DataFrame, target_col: Optional[str] = None) -> pd.DataFrame:
        """
        Ajusta o pipeline e transforma os dados em uma única operação.
        
        Args:
            df: DataFrame com os dados
            target_col: Nome da coluna alvo (opcional)
            
        Returns:
            DataFrame com os dados transformados
        """
        self.fit(df, target_col=target_col)
        return self.transform(df, target_col=target_col)
    
    def save(self, base_path: str) -> None:
        """
        Salva o pipeline completo em arquivos separados.
        
        Args:
            base_path: Caminho base para salvar os arquivos
        """
        if not self.fitted:
            raise ValueError("Não é possível salvar um pipeline não ajustado.")
        
        preprocessor_path = f"{base_path}_preprocessor.pkl"
        feature_engineer_path = f"{base_path}_feature_engineer.pkl"
        
        self.preprocessor.save(preprocessor_path)
        self.feature_engineer.save(feature_engineer_path)
        
        self.logger.info(f"Pipeline completo salvo em {base_path}_*.pkl")
    
    @classmethod
    def load(cls, base_path: str) -> 'DataPipeline':
        """
        Carrega um pipeline completo a partir de arquivos.
        
        Args:
            base_path: Caminho base onde os arquivos foram salvos
            
        Returns:
            Nova instância de DataPipeline com os componentes carregados
        """
        pipeline = cls()
        
        preprocessor_path = f"{base_path}_preprocessor.pkl"
        feature_engineer_path = f"{base_path}_feature_engineer.pkl"
        
        pipeline.preprocessor = PreProcessor.load(preprocessor_path)
        pipeline.feature_engineer = FeatureEngineer.load(feature_engineer_path)
        pipeline.fitted = True
        
        pipeline.logger.info(f"Pipeline completo carregado de {base_path}_*.pkl")
        
        return pipeline
    
def create_data_pipeline(preprocessor_config: Optional[Dict] = None, 
                        feature_engineer_config: Optional[Dict] = None) -> DataPipeline:
    return DataPipeline(preprocessor_config, feature_engineer_config)
