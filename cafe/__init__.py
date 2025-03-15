"""
CAFE (Component Automated Feature Engineer) - Sistema de Engenharia Automática de Features.

Um sistema abrangente para automatizar o processamento de dados e a engenharia 
de features em projetos de machine learning.
"""

__version__ = "0.1.1"

from .preprocessor import PreProcessor, create_preprocessor
from .feature_engineer import FeatureEngineer, create_feature_engineer
from .performance_validator import PerformanceValidator
from .data_pipeline import DataPipeline, create_data_pipeline
from .explorer import Explorer, TransformationTree, HeuristicSearch
from .enhanced_datapipeline import EnhancedDataPipeline

__all__ = [
    "PreProcessor",
    "create_preprocessor",
    "FeatureEngineer",
    "create_feature_engineer",
    "PerformanceValidator",
    "DataPipeline",
    "create_data_pipeline",
    "Explorer",
    "TransformationTree",
    "HeuristicSearch",
    "EnhancedDataPipeline"
]