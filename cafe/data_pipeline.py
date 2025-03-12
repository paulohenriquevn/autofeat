import pandas as pd
import logging
from typing import Dict, Optional

from .preprocessor import PreProcessor
from .feature_engineer import FeatureEngineer
from .performance_validator import PerformanceValidator

class DataPipeline:
    """
    Classe que combina PreProcessor e FeatureEngineer para criar um pipeline completo
    de processamento e engenharia de features, com validação de performance.
    """
    def __init__(self, preprocessor_config: Optional[Dict] = None, 
                feature_engineer_config: Optional[Dict] = None,
                validator_config: Optional[Dict] = None,
                auto_validate: bool = True):
        """
        Inicializa o pipeline de dados com configurações opcionais.
        
        Args:
            preprocessor_config: Configuração para o preprocessador
            feature_engineer_config: Configuração para o engenheiro de features
            validator_config: Configuração para o validador de performance
            auto_validate: Se True, aplica validação automática para decidir qual dataset usar
        """
        self.preprocessor = PreProcessor(preprocessor_config)
        self.feature_engineer = FeatureEngineer(feature_engineer_config)
        self.validator = PerformanceValidator(validator_config)
        self.auto_validate = auto_validate
        self.fitted = False
        self.target_col = None
        self.validation_results = None
        self.using_original_data = False
        
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
        Se auto_validate=True, compara a performance antes e depois das transformações
        e decide qual versão usar.
        
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
        
        # Se não houver coluna alvo ou não estiver em modo de validação automática, 
        # simplesmente ajustar o pipeline normalmente
        if target_col is None or not self.auto_validate:
            # Ajustar o preprocessador
            self.preprocessor.fit(df, target_col=target_col)
            
            # Transformar os dados com o preprocessador
            df_preprocessed = self.preprocessor.transform(df, target_col=target_col)
            
            # Ajustar o feature engineer com os dados preprocessados
            self.feature_engineer.fit(df_preprocessed, target_col=target_col)
            
            self.fitted = True
            self.using_original_data = False
            self.logger.info("Pipeline completo ajustado com sucesso")
            
            return self
        
        # Se temos coluna alvo e auto_validate está ativado, vamos comparar performance
        self.logger.info("Validação automática ativada. Comparando performance antes e depois das transformações.")
        
        # Separar features e target
        y = df[target_col]
        X_original = df.drop(columns=[target_col])
        
        # Primeiro, ajustar o pipeline sem validação
        self.fitted = True  # Temporariamente marcar como ajustado para poder transformar os dados
        
        # Ajustar o preprocessador e o feature engineer
        self.preprocessor.fit(df, target_col=target_col)
        df_preprocessed = self.preprocessor.transform(df, target_col=target_col)
        self.feature_engineer.fit(df_preprocessed, target_col=target_col)
        
        # Transformar os dados completos pelo pipeline
        df_transformed = self.transform(df, target_col=target_col)
        X_transformed = df_transformed.drop(columns=[target_col])
        
        # Usar o validador para comparar performance
        X_best, results = self.validator.get_best_dataset(
            X_original, y, X_transformed, include_validation_results=True
        )
        
        self.validation_results = results
        self.using_original_data = (results['best_choice'] == 'original')
        
        # Log dos resultados da validação
        self.logger.info(f"Resultado da validação: {results['best_choice']} é melhor")
        self.logger.info(f"Performance original: {results['performance_original']:.4f}")
        self.logger.info(f"Performance transformada: {results['performance_transformed']:.4f}")
        self.logger.info(f"Diferença: {results['performance_diff_pct']:.2f}%")
        
        self.fitted = True
        return self
    
    def transform(self, df: pd.DataFrame, target_col: Optional[str] = None) -> pd.DataFrame:
        """
        Transforma dados usando o pipeline completo.
        Se o pipeline foi ajustado com validação e os dados originais tiveram 
        melhor performance, retorna os dados originais.
        
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
        
        # Se estamos usando dados originais devido à validação, apenas retornar o DataFrame original
        if self.using_original_data:
            self.logger.info("Usando dados originais com base na validação de performance.")
            return df
        
        # Caso contrário, aplicar o pipeline normal
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
        
        # Salvar também informações sobre a validação
        import json
        validation_info = {
            'using_original_data': self.using_original_data,
            'validation_results': self.validation_results
        }
        
        with open(f"{base_path}_validation_info.json", 'w') as f:
            json.dump(validation_info, f)
        
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
        pipeline = cls(auto_validate=False)  # Desativar auto validação ao carregar
        
        preprocessor_path = f"{base_path}_preprocessor.pkl"
        feature_engineer_path = f"{base_path}_feature_engineer.pkl"
        
        pipeline.preprocessor = PreProcessor.load(preprocessor_path)
        pipeline.feature_engineer = FeatureEngineer.load(feature_engineer_path)
        
        # Carregar informações de validação se existirem
        import json
        import os
        
        validation_info_path = f"{base_path}_validation_info.json"
        if os.path.exists(validation_info_path):
            with open(validation_info_path, 'r') as f:
                validation_info = json.load(f)
                
            pipeline.using_original_data = validation_info.get('using_original_data', False)
            pipeline.validation_results = validation_info.get('validation_results')
        
        pipeline.fitted = True
        
        pipeline.logger.info(f"Pipeline completo carregado de {base_path}_*.pkl")
        
        return pipeline
    
    def get_validation_results(self) -> Dict:
        """
        Retorna os resultados da validação de performance.
        
        Returns:
            Dicionário com os resultados da validação ou None se a validação não foi realizada
        """
        return self.validation_results
    
    def get_feature_importance(self, df: pd.DataFrame, target_col: Optional[str] = None) -> pd.DataFrame:
        """
        Retorna a importância das features nos dados originais.
        Útil para entender quais features são mais importantes antes da transformação.
        
        Args:
            df: DataFrame com os dados
            target_col: Nome da coluna alvo
            
        Returns:
            DataFrame com as features ordenadas por importância
        """
        target = target_col or self.target_col
        
        if target is None:
            raise ValueError("É necessário fornecer a coluna alvo para calcular a importância das features")
            
        y = df[target]
        X = df.drop(columns=[target])
        
        return self.validator.get_feature_importance(X, y)
    
def create_data_pipeline(preprocessor_config: Optional[Dict] = None, 
                        feature_engineer_config: Optional[Dict] = None,
                        validator_config: Optional[Dict] = None,
                        auto_validate: bool = True) -> DataPipeline:
    """
    Função auxiliar para criar uma instância do DataPipeline.
    
    Args:
        preprocessor_config: Configuração para o preprocessador
        feature_engineer_config: Configuração para o engenheiro de features
        validator_config: Configuração para o validador de performance
        auto_validate: Se True, aplica validação automática
        
    Returns:
        Instância configurada do DataPipeline
    """
    return DataPipeline(
        preprocessor_config=preprocessor_config,
        feature_engineer_config=feature_engineer_config,
        validator_config=validator_config,
        auto_validate=auto_validate
    )