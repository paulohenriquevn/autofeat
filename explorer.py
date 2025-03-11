import logging
import networkx as nx
import pandas as pd
import numpy as np
from typing import Dict, Callable, Optional, List

from preprocessor import PreProcessor
from feature_engineer import FeatureEngineer
from data_pipeline import DataPipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AutoFE.Explorer")

class TransformationTree:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.graph.add_node("root", data=None)
        logger.info("TransformationTree inicializada.")
    
    def add_transformation(self, parent: str, name: str, data, score: float = 0.0):
        """Adiciona uma transformação à árvore."""
        self.graph.add_node(name, data=data, score=score)
        self.graph.add_edge(parent, name)
        feature_diff = data.shape[1] - self.graph.nodes[parent]['data'].shape[1] if self.graph.nodes[parent]['data'] is not None else 0
        logger.info(f"Transformação '{name}' adicionada com score {score}. Dimensão do conjunto: {data.shape}. Alteração nas features: {feature_diff}")
    
    def get_best_transformations(self, heuristic: Callable[[Dict], float]) -> List[str]:
        """Retorna as melhores transformações baseadas em uma heurística."""
        scored_nodes = {node: heuristic(self.graph.nodes[node]['data']) for node in self.graph.nodes if node != "root"}
        best_transformations = sorted(scored_nodes, key=scored_nodes.get, reverse=True)
        logger.info(f"Melhores transformações ordenadas: {best_transformations}")
        return best_transformations

class HeuristicSearch:
    def __init__(self, heuristic: Callable[[pd.DataFrame], float]):
        self.heuristic = heuristic
    
    def search(self, tree: TransformationTree) -> str:
        """Executa uma busca heurística na árvore de transformações."""
        best_nodes = tree.get_best_transformations(self.heuristic)
        best_node = best_nodes[0] if best_nodes else None
        logger.info(f"Melhor transformação encontrada: {best_node}")
        return best_node
    
    @staticmethod
    def custom_heuristic(df: pd.DataFrame) -> float:
        """Heurística baseada na matriz de correlação e diversidade categórica."""
        correlation_penalty = 0
        categorical_diversity_score = 0
        
        # Penaliza alta correlação entre features
        if df.shape[1] > 1:
            correlation_matrix = df.corr().abs()
            high_corr = (correlation_matrix > 0.95).sum().sum() - df.shape[1]  # Remove a diagonal
            correlation_penalty = high_corr / (df.shape[1] ** 2)  # Normaliza penalização
        
        # Avalia diversidade de variáveis categóricas
        categorical_features = df.select_dtypes(include=['object', 'category'])
        if not categorical_features.empty:
            unique_counts = categorical_features.nunique()
            categorical_diversity_score = unique_counts.mean() / max(1, unique_counts.max())  # Normaliza entre 0 e 1
        
        # Score final: penaliza alta correlação e recompensa diversidade categórica
        final_score = -correlation_penalty + categorical_diversity_score
        return final_score

class Explorer:
    def __init__(self, heuristic: Callable[[pd.DataFrame], float] = None, target_col: Optional[str] = None):
        self.tree = TransformationTree()
        self.search = HeuristicSearch(heuristic or HeuristicSearch.custom_heuristic)
        self.target_col = target_col
    
    def add_transformation(self, parent: str, name: str, data, score: float = 0.0):
        """Adiciona uma transformação com uma pontuação atribuída."""
        self.tree.add_transformation(parent, name, data, score)
    
    def find_best_transformation(self) -> str:
        """Retorna a melhor transformação com base na busca heurística."""
        return self.search.search(self.tree)
    
    def analyze_transformations(self, df):
        """Testa diferentes transformações e escolhe a melhor combinação de processamento e features."""
        logger.info("Iniciando análise de transformações.")
        if isinstance(df, np.ndarray):
            df = pd.DataFrame(df, columns=[f"feature_{i}" for i in range(df.shape[1])])
        
        base_node = "root"
        self.tree.graph.nodes[base_node]['data'] = df
        
        # Combinações de configurações para testar
        preprocessor_configs = [
            {"missing_values_strategy": "mean"},
            {"missing_values_strategy": "median"},
            {"missing_values_strategy": "most_frequent"},
            {"outlier_method": "iqr"},
            {"outlier_method": "zscore"},
            {"outlier_method": "isolation_forest"},
            {"categorical_strategy": "onehot"},
            {"categorical_strategy": "ordinal"},
            {"scaling": "standard"},
            {"scaling": "minmax"},
            {"scaling": "robust"}
        ]
        
        feature_configs = [
            {"dimensionality_reduction": "pca"},
            {"dimensionality_reduction": None},
            {"feature_selection": "variance"},
            {"feature_selection": None},
            {"generate_features": True},
            {"generate_features": False},
            {"correlation_threshold": 0.95}
        ]
        
        # Testar cada combinação de preprocessador
        for preproc_config in preprocessor_configs:
            preproc_name = "_".join([f"{key}-{value}" for key, value in preproc_config.items()])
            logger.info(f"Testando preprocessamento: {preproc_name}")
            
            try:
                # Ajustar e transformar com o preprocessador
                preprocessed_df = PreProcessor(preproc_config).fit(df, target_col=self.target_col).transform(df, target_col=self.target_col)
                
                if preprocessed_df.empty:
                    logger.warning(f"O preprocessamento {preproc_name} resultou em DataFrame vazio. Pulando.")
                    continue
                
                # Adicionar o resultado do preprocessamento à árvore
                preproc_node = f"preproc_{preproc_name}"
                score = self.search.heuristic(preprocessed_df)
                self.add_transformation(base_node, preproc_node, preprocessed_df, score)
                
                # Para cada preprocessamento, testar engenharia de features
                for feat_config in feature_configs:
                    feat_name = "_".join([f"{key}-{value}" for key, value in feat_config.items()])
                    logger.info(f"Testando feature engineering: {feat_name} sobre {preproc_name}")
                    
                    try:
                        # Ajustar e transformar com o feature engineer
                        transformed_df = FeatureEngineer(feat_config).fit(preprocessed_df, target_col=self.target_col).transform(preprocessed_df, target_col=self.target_col)
                        
                        if transformed_df.empty:
                            logger.warning(f"A engenharia de features {feat_name} resultou em DataFrame vazio. Pulando.")
                            continue
                        
                        # Adicionar o resultado da engenharia de features à árvore
                        full_node = f"{preproc_node}_feat_{feat_name}"
                        score = self.search.heuristic(transformed_df)
                        self.add_transformation(preproc_node, full_node, transformed_df, score)
                        
                    except Exception as e:
                        logger.warning(f"Erro ao aplicar feature engineering {feat_name}: {e}")
                
            except Exception as e:
                logger.warning(f"Erro ao aplicar preprocessamento {preproc_name}: {e}")
        
        # Encontrar a melhor transformação
        best_transformation = self.find_best_transformation()
        logger.info(f"Melhor pipeline encontrado: {best_transformation}")
        
        # Retornar o melhor resultado
        if best_transformation:
            return self.tree.graph.nodes[best_transformation]['data']
        else:
            logger.warning("Nenhuma transformação válida encontrada. Retornando DataFrame original.")
            return df
    
    def get_best_pipeline_config(self) -> Dict:
        """Retorna a configuração do melhor pipeline encontrado."""
        best_transformation = self.find_best_transformation()
        if not best_transformation:
            logger.warning("Nenhuma transformação válida encontrada.")
            return {}
            
        # Analisar o nome da transformação para extrair as configurações
        config_parts = best_transformation.split('_')
        
        # Inicializar configurações
        preprocessor_config = {}
        feature_config = {}
        
        # Extrair configurações do preprocessador
        preproc_parts = []
        feat_parts = []
        
        # Dividir entre preprocessador e feature engineer
        in_feat_section = False
        for part in config_parts:
            if part == 'feat':
                in_feat_section = True
                continue
            
            if in_feat_section:
                feat_parts.append(part)
            else:
                preproc_parts.append(part)
        
        # Processar partes do preprocessador
        for i in range(1, len(preproc_parts), 2):  # Começar de 1 para pular "preproc_"
            if i+1 < len(preproc_parts):
                key = preproc_parts[i]
                value = preproc_parts[i+1]
                # Converter strings para tipos apropriados
                if value == 'True':
                    value = True
                elif value == 'False':
                    value = False
                elif value.isdigit():
                    value = int(value)
                elif value.replace('.', '', 1).isdigit():
                    value = float(value)
                preprocessor_config[key] = value
        
        # Processar partes do feature engineer
        for i in range(0, len(feat_parts), 2):
            if i+1 < len(feat_parts):
                key = feat_parts[i]
                value = feat_parts[i+1]
                # Converter strings para tipos apropriados
                if value == 'True':
                    value = True
                elif value == 'False':
                    value = False
                elif value.isdigit():
                    value = int(value)
                elif value.replace('.', '', 1).isdigit():
                    value = float(value)
                elif value == 'None':
                    value = None
                feature_config[key] = value
        
        return {
            'preprocessor_config': preprocessor_config,
            'feature_engineer_config': feature_config
        }
    
    def create_optimal_pipeline(self) -> DataPipeline:
        """Cria um pipeline otimizado com base na melhor configuração encontrada."""
        config = self.get_best_pipeline_config()
        if not config:
            logger.warning("Usando configuração padrão para o pipeline.")
            return DataPipeline()
            
        return DataPipeline(
            preprocessor_config=config.get('preprocessor_config', {}),
            feature_engineer_config=config.get('feature_engineer_config', {})
        )
