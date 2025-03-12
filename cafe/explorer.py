import logging
import networkx as nx
import pandas as pd
import numpy as np
from typing import Dict, Callable, Optional, List, Any, Union

from .preprocessor import PreProcessor
from .feature_engineer import FeatureEngineer
from .data_pipeline import DataPipeline

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
    
    def get_best_transformations(self, heuristic: Callable[[pd.DataFrame], float]) -> List[str]:
        """Retorna as melhores transformações baseadas em uma heurística."""
        # Corrigido: passar o DataFrame, não o dicionário do nó inteiro
        scored_nodes = {node: heuristic(self.graph.nodes[node]['data']) for node in self.graph.nodes if node != "root" and self.graph.nodes[node]['data'] is not None}
        best_transformations = sorted(scored_nodes, key=scored_nodes.get, reverse=True)
        if best_transformations:
            logger.info(f"Melhores transformações ordenadas: {best_transformations[:3]} (de {len(best_transformations)} transformações)")
        else:
            logger.warning("Nenhuma transformação válida encontrada para ordenar.")
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
    def strict_dimension_heuristic(df: pd.DataFrame, original_feature_count: int = None, max_expansion_factor: float = 2.0) -> float:
        """
        Heurística que impõe um limite estrito no número de features,
        rejeitando completamente transformações que ultrapassem um limite máximo.
        
        Args:
            df (pd.DataFrame): DataFrame transformado a ser avaliado
            original_feature_count (int): Número original de features no dataset
            max_expansion_factor (float): Fator máximo de expansão permitido (ex: 2.0 = 2x mais features)
            
        Returns:
            float: Pontuação da heurística (maior é melhor, -inf para transformações rejeitadas)
        """
        # Verificar se df é None ou não é um DataFrame
        if df is None or not isinstance(df, pd.DataFrame):
            logger.warning(f"Objeto inválido passado para a heurística: {type(df)}")
            return float('-inf')
            
        # Determinar o número original de features se não fornecido
        if original_feature_count is None:
            # Tentar detectar automaticamente o número de features originais
            if hasattr(df, 'original_feature_count'):
                original_feature_count = df.original_feature_count
            else:
                # Assumir um valor razoável baseado em datasets comuns
                original_feature_count = 13  # Wine dataset
        
        # Número máximo de features permitido
        max_features = int(original_feature_count * max_expansion_factor)
        
        # Verificar se o DataFrame excede o limite máximo de features
        # Desconsiderando possíveis colunas target
        target_cols = ['target', 'classe', 'class', 'y', 'label']
        df_cols = [col for col in df.columns if col.lower() not in target_cols]
        
        if len(df_cols) > max_features:
            # Rejeitar completamente, retornando uma pontuação extremamente baixa
            return float('-inf')
        
        # Para transformações dentro do limite, calcular pontuação normal
        
        # ---- Componente 1: Proximidade ao número original (quanto mais próximo, melhor) ----
        dimension_score = 1.0 - (abs(len(df_cols) - original_feature_count) / original_feature_count)
        
        # ---- Componente 2: Penalidade por correlação alta ----
        correlation_penalty = 0
        numeric_df = df.select_dtypes(include=['number'])
        if not numeric_df.empty and numeric_df.shape[1] > 1:
            try:
                correlation_matrix = numeric_df.corr().abs()
                # Remove a diagonal
                np.fill_diagonal(correlation_matrix.values, 0)
                high_corr = (correlation_matrix > 0.8).sum().sum() / 2  # Divide por 2 pois a matriz é simétrica
                correlation_penalty = high_corr / (numeric_df.shape[1] * (numeric_df.shape[1] - 1) / 2)
            except Exception as e:
                logger.warning(f"Erro ao calcular correlação: {e}")
                pass
        
        # ---- Componente 3: Variância explicada ----
        # Premia features com maior variância (mais informativas)
        variance_score = 0
        if not numeric_df.empty:
            try:
                normalized_variances = numeric_df.var() / numeric_df.var().max()
                variance_score = normalized_variances.mean()
            except Exception as e:
                logger.warning(f"Erro ao calcular variância: {e}")
                pass
        
        # Calcular pontuação final com pesos
        final_score = (
            dimension_score * 5.0 +       # Peso muito alto para proximidade dimensional
            variance_score * 2.0 -        # Peso médio para variância explicada
            correlation_penalty * 3.0     # Peso alto para penalidade por correlação
        )
        
        return final_score

class Explorer:
    def __init__(self, heuristic: Callable[[pd.DataFrame], float] = None, target_col: Optional[str] = None):
        self.tree = TransformationTree()
        # Utilizar a heurística de dimensão estrita por padrão, limitando a expansão para 2x
        self.search = HeuristicSearch(heuristic or (lambda df: HeuristicSearch.strict_dimension_heuristic(df, max_expansion_factor=2.0)))
        self.target_col = target_col
        self.original_feature_count = None
    
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
        
        # Armazenar número de features original para referência
        target_cols = ['target', 'classe', 'class', 'y', 'label']
        feature_cols = [col for col in df.columns if col.lower() not in target_cols]
        self.original_feature_count = len(feature_cols)
        logger.info(f"Número de features original: {self.original_feature_count}")
        
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
            {"scaling": "robust"},
        ]

        # Lista expandida de configurações para FeatureEngineer
        feature_configs = [
            # Configurações originais
            {"dimensionality_reduction": "pca", "generate_features": False, "correlation_threshold": 0.8},
            {"dimensionality_reduction": None, "generate_features": False, "correlation_threshold": 0.8},
            {"dimensionality_reduction": None, "generate_features": True, "correlation_threshold": 0.8},
            {"feature_selection": "variance", "generate_features": False, "correlation_threshold": 0.8},
            {"feature_selection": "mutual_info", "generate_features": False, "correlation_threshold": 0.8},
            {"feature_selection": None, "generate_features": False, "correlation_threshold": 0.8},
            {"correlation_threshold": 0.95},
            {"correlation_threshold": 0.90},
            {"correlation_threshold": 0.85},
            {"correlation_threshold": 0.80},
            
            # Novas configurações de seleção de features
            # SelectKBest com diferentes configurações
            {
                "feature_selection": "kbest", 
                "feature_selection_params": {"k": 10}, 
                "correlation_threshold": 0.8
            },
            {
                "feature_selection": "kbest", 
                "feature_selection_params": {"k": 15}, 
                "correlation_threshold": 0.8
            },
            
            # SelectPercentile com diferentes percentis
            {
                "feature_selection": "percentile", 
                "feature_selection_params": {"percentile": 20}, 
                "correlation_threshold": 0.8
            },
            {
                "feature_selection": "percentile", 
                "feature_selection_params": {"percentile": 30}, 
                "correlation_threshold": 0.8
            },
            
            # SelectFromModel
            {
                "feature_selection": "model", 
                "feature_selection_params": {"threshold": "mean"}, 
                "correlation_threshold": 0.8
            },
            {
                "feature_selection": "model", 
                "feature_selection_params": {"threshold": "median"}, 
                "correlation_threshold": 0.8
            },
            
            # Métodos estatísticos
            {
                "feature_selection": "fwe", 
                "feature_selection_params": {"alpha": 0.05}, 
                "correlation_threshold": 0.8
            },
            {
                "feature_selection": "fpr", 
                "feature_selection_params": {"alpha": 0.05}, 
                "correlation_threshold": 0.8
            },
            {
                "feature_selection": "fdr", 
                "feature_selection_params": {"alpha": 0.05}, 
                "correlation_threshold": 0.8
            },
            
            # Combinações com geração de features
            {
                "feature_selection": "kbest", 
                "feature_selection_params": {"k": 10}, 
                "generate_features": True, 
                "correlation_threshold": 0.85
            },
            {
                "feature_selection": "percentile", 
                "feature_selection_params": {"percentile": 20}, 
                "generate_features": True, 
                "correlation_threshold": 0.85
            },
            {
                "feature_selection": "model", 
                "feature_selection_params": {"threshold": "mean"}, 
                "generate_features": True, 
                "correlation_threshold": 0.85
            },
        ]

        # Testar cada combinação de preprocessador
        for preproc_config in preprocessor_configs:
            # Criar um nome simplificado para o preprocessador
            preproc_name = '_'.join([f"{key}-{value}" for key, value in preproc_config.items()])
            logger.info(f"Testando preprocessamento: {preproc_name}")
            
            try:
                # Ajustar e transformar com o preprocessador
                preprocessed_df = PreProcessor(preproc_config).fit(df, target_col=self.target_col).transform(df, target_col=self.target_col)
                
                if preprocessed_df.empty:
                    logger.warning(f"O preprocessamento {preproc_name} resultou em DataFrame vazio. Pulando.")
                    continue
                
                # Adicionar o resultado do preprocessamento à árvore
                preproc_node = f"preproc_{preproc_name}"
                # Usar a heurística que restringe dimensionalidade
                score = self.search.heuristic(preprocessed_df)
                self.add_transformation(base_node, preproc_node, preprocessed_df, score)
                
                # Para cada preprocessamento, testar engenharia de features
                for feat_config in feature_configs:
                    # Criar um nome para a configuração de features
                    # Precisamos de um tratamento especial para feature_selection_params que é um dict
                    feat_name_parts = []
                    for key, value in feat_config.items():
                        if key == 'feature_selection_params' and isinstance(value, dict):
                            # Criar uma string para o dicionário de parâmetros
                            params_str = '-'.join([f"{k}_{v}" for k, v in value.items()])
                            feat_name_parts.append(f"{key}-{params_str}")
                        else:
                            feat_name_parts.append(f"{key}-{value}")
                    
                    feat_name = '_'.join(feat_name_parts)
                    logger.info(f"Testando feature engineering: {feat_name} sobre {preproc_name}")
                    
                    try:
                        # Ajustar e transformar com o feature engineer
                        # Adicionamos o task do target_col à configuração
                        feat_config_with_task = feat_config.copy()
                        # Adicionar informação de 'task' para métodos supervisionados se tivermos target_col
                        if self.target_col:
                            feat_config_with_task['task'] = 'classification'  # Podemos inferir isso do target se necessário
                        
                        transformed_df = FeatureEngineer(feat_config_with_task).fit(preprocessed_df, target_col=self.target_col).transform(preprocessed_df, target_col=self.target_col)
                        
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
        """
        Retorna a configuração do melhor pipeline encontrado.
        Versão atualizada para lidar com parâmetros aninhados, como feature_selection_params.
        """
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
        
        # Processar partes do preprocessador (lógica original)
        i = 1  # Começar de 1 para pular "preproc_"
        while i < len(preproc_parts):
            # Verificar se temos um par key-value
            if i + 1 < len(preproc_parts):
                key = preproc_parts[i]
                value = preproc_parts[i+1]
                # Converter strings para tipos apropriados
                value = self._convert_string_to_type(value)
                preprocessor_config[key] = value
                i += 2
            else:
                i += 1
        
        # Processar partes do feature engineer (lógica atualizada)
        # Precisamos lidar com parâmetros aninhados como feature_selection_params
        i = 0
        feature_selection_params = {}  # Para armazenar parâmetros aninhados
        inside_nested_params = False
        current_nested_key = None
        
        while i < len(feat_parts):
            if i + 1 < len(feat_parts):
                key = feat_parts[i]
                value = feat_parts[i+1]
                
                # Verificar se estamos entrando em parâmetros aninhados
                if key == 'feature_selection_params':
                    inside_nested_params = True
                    current_nested_key = key
                    i += 2
                    continue
                
                # Se estamos dentro de parâmetros aninhados, processá-los
                if inside_nested_params:
                    # Verificar se o formato indica que ainda estamos dentro de parâmetros aninhados
                    if '-' in key and '_' in key:
                        # É um par chave-valor de parâmetros aninhados (ex: "k_10")
                        nested_key, nested_value = key.split('_', 1)
                        feature_selection_params[nested_key] = self._convert_string_to_type(nested_value)
                        i += 1
                    else:
                        # Saímos dos parâmetros aninhados
                        inside_nested_params = False
                        feature_config[current_nested_key] = feature_selection_params
                        # Não incrementar i, processar o par atual normalmente
                    
                # Processar normalmente se não estamos em parâmetros aninhados
                if not inside_nested_params:
                    value = self._convert_string_to_type(value)
                    feature_config[key] = value
                    i += 2
            else:
                i += 1
        
        # Se terminarmos ainda dentro de parâmetros aninhados, finalizá-los
        if inside_nested_params and current_nested_key:
            feature_config[current_nested_key] = feature_selection_params
        
        # Se feature_config estiver vazio, adicionar configuração padrão
        # para garantir controle de correlação e evitar expansão excessiva
        if not feature_config:
            feature_config = {
                "correlation_threshold": 0.8,
                "generate_features": False
            }
        
        return {
            'preprocessor_config': preprocessor_config,
            'feature_engineer_config': feature_config
        }
    
    def _convert_string_to_type(self, value_str: str) -> Any:
        """Converte uma string para o tipo apropriado."""
        if value_str == 'True':
            return True
        elif value_str == 'False':
            return False
        elif value_str == 'None':
            return None
        elif value_str.isdigit():
            return int(value_str)
        elif value_str.replace('.', '', 1).isdigit():
            return float(value_str)
        return value_str
    
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