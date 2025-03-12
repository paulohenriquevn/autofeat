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
    """
    Estrutura de dados em árvore para armazenar e organizar as diferentes transformações
    testadas pelo Explorer.
    """
    def __init__(self):
        self.graph = nx.DiGraph()
        self.graph.add_node("root", data=None)
        logger.info("TransformationTree inicializada.")
    
    def add_transformation(self, parent: str, name: str, data, score: float = 0.0):
        """
        Adiciona uma transformação à árvore.
        
        Args:
            parent: Nó pai ao qual a transformação está conectada
            name: Nome da transformação
            data: DataFrame resultante da transformação
            score: Pontuação da transformação
        """
        self.graph.add_node(name, data=data, score=score)
        self.graph.add_edge(parent, name)
        feature_diff = data.shape[1] - self.graph.nodes[parent]['data'].shape[1] if self.graph.nodes[parent]['data'] is not None else 0
        logger.info(f"Transformação '{name}' adicionada com score {score}. Dimensão do conjunto: {data.shape}. Alteração nas features: {feature_diff}")
    
    def get_best_transformations(self, heuristic: Callable[[pd.DataFrame], float]) -> List[str]:
        """
        Retorna as melhores transformações baseadas em uma heurística.
        
        Args:
            heuristic: Função que avalia a qualidade de um DataFrame transformado
            
        Returns:
            Lista dos nomes das transformações, ordenadas pela qualidade
        """
        # Avaliar cada nó com a heurística
        scored_nodes = {node: heuristic(self.graph.nodes[node]['data']) 
                      for node in self.graph.nodes 
                      if node != "root" and self.graph.nodes[node]['data'] is not None}
        
        # Ordenar os nós por pontuação (maior para menor)
        best_transformations = sorted(scored_nodes, key=scored_nodes.get, reverse=True)
        
        if best_transformations:
            logger.info(f"Melhores transformações ordenadas: {best_transformations[:3]} (de {len(best_transformations)} transformações)")
        else:
            logger.warning("Nenhuma transformação válida encontrada para ordenar.")
            
        return best_transformations

class HeuristicSearch:
    """
    Classe para realizar a busca heurística pelas melhores transformações
    na árvore de transformações.
    """
    def __init__(self, heuristic: Callable[[pd.DataFrame], float]):
        """
        Inicializa o mecanismo de busca com uma heurística.
        
        Args:
            heuristic: Função que avalia a qualidade de um DataFrame transformado
        """
        self.heuristic = heuristic
    
    def search(self, tree: TransformationTree) -> str:
        """
        Executa uma busca heurística na árvore de transformações.
        
        Args:
            tree: Árvore de transformações a ser pesquisada
            
        Returns:
            Nome da melhor transformação encontrada
        """
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
            df: DataFrame transformado a ser avaliado
            original_feature_count: Número original de features no dataset
            max_expansion_factor: Fator máximo de expansão permitido (ex: 2.0 = 2x mais features)
            
        Returns:
            Pontuação da heurística (maior é melhor, -inf para transformações rejeitadas)
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
    """
    Classe principal que explora diferentes combinações de transformações
    para encontrar a configuração ótima para um dataset.
    """
    def __init__(self, heuristic: Callable[[pd.DataFrame], float] = None, target_col: Optional[str] = None):
        """
        Inicializa o Explorer.
        
        Args:
            heuristic: Função opcional para avaliar transformações (se None, usa a heurística padrão)
            target_col: Nome da coluna alvo (necessário para métodos supervisionados)
        """
        self.tree = TransformationTree()
        # Utilizar a heurística de dimensão estrita por padrão, limitando a expansão para 2x
        self.search = HeuristicSearch(heuristic or (lambda df: HeuristicSearch.strict_dimension_heuristic(df, max_expansion_factor=2.0)))
        self.target_col = target_col
        self.original_feature_count = None
    
    def add_transformation(self, parent: str, name: str, data, score: float = 0.0):
        """
        Adiciona uma transformação com uma pontuação atribuída.
        
        Args:
            parent: Nó pai ao qual a transformação está conectada
            name: Nome da transformação
            data: DataFrame resultante da transformação
            score: Pontuação da transformação
        """
        self.tree.add_transformation(parent, name, data, score)
    
    def find_best_transformation(self) -> str:
        """
        Retorna a melhor transformação com base na busca heurística.
        
        Returns:
            Nome da melhor transformação encontrada
        """
        return self.search.search(self.tree)
    
    def analyze_transformations(self, df):
        """
        Testa diferentes transformações e escolhe a melhor combinação de processamento e features.
        
        Args:
            df: DataFrame com os dados a serem analisados
            
        Returns:
            DataFrame com a melhor transformação aplicada
        """
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
        
        # Combinações EXPANDIDAS de configurações para testar
        preprocessor_configs = [
            # Estratégias para valores ausentes
            {"missing_values_strategy": "mean"},
            {"missing_values_strategy": "median"},
            {"missing_values_strategy": "most_frequent"},
            {"missing_values_strategy": "knn"},
            
            # Métodos para outliers
            {"outlier_method": "iqr"},
            {"outlier_method": "zscore"},
            {"outlier_method": "isolation_forest"},
            {"outlier_method": "none"},
            
            # Estratégias para variáveis categóricas
            {"categorical_strategy": "onehot"},
            {"categorical_strategy": "ordinal"},
            {"categorical_strategy": "target"},
            {"categorical_strategy": "label"},
            {"categorical_strategy": "binary"},
            {"categorical_strategy": "multilabel"},
            
            # Métodos de scaling
            {"scaling": "standard"},
            {"scaling": "minmax"},
            {"scaling": "robust"},
            {"scaling": "maxabs"},
            {"scaling": "power", "power_transformer_method": "yeo-johnson"},
            {"scaling": "power", "power_transformer_method": "box-cox"},
            {"scaling": "quantile", "quantile_n_quantiles": 1000},
            {"scaling": "normalize"},
            
            # Transformadores adicionais para features numéricas
            {"numeric_transformers": ["binarizer"], "binarizer_threshold": 0.5},
            {"numeric_transformers": ["binarizer"], "binarizer_threshold": 0.75},
            {"numeric_transformers": ["kbins"], "kbins_n_bins": 5, "kbins_encode": "ordinal"},
            {"numeric_transformers": ["kbins"], "kbins_n_bins": 10, "kbins_encode": "onehot"},
            
            # Combinações de múltiplos transformadores
            {"scaling": "robust", "missing_values_strategy": "knn", "outlier_method": "none"},
            {"scaling": "power", "power_transformer_method": "yeo-johnson", "missing_values_strategy": "median"},
            {"scaling": "quantile", "missing_values_strategy": "most_frequent", "categorical_strategy": "target"},
            {"scaling": "robust", "numeric_transformers": ["binarizer"], "binarizer_threshold": 0.5},
            {"scaling": "standard", "numeric_transformers": ["kbins"], "kbins_n_bins": 5},
            
            # Configurações específicas para tipos de dados com alta cardinalidade
            {"categorical_strategy": "target", "target_encoding_smoothing": 5.0},
            {"categorical_strategy": "target", "target_encoding_smoothing": 20.0},
            
            # Configurações para dados temporais
            {"datetime_features": ["year", "month", "day", "weekday", "is_weekend"]},
            {"datetime_features": ["year", "month", "quarter", "is_month_start", "is_month_end"]},
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
            {
                "feature_selection": "percentile", 
                "feature_selection_params": {"percentile": 50}, 
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
            
            # Novas combinações com diferentes funções de score
            {
                "feature_selection": "kbest", 
                "feature_selection_params": {
                    "k": 10, 
                    "score_func": "f_classif"
                }, 
                "correlation_threshold": 0.8
            },
            {
                "feature_selection": "kbest", 
                "feature_selection_params": {
                    "k": 10, 
                    "score_func": "chi2"
                }, 
                "correlation_threshold": 0.8
            },
            {
                "feature_selection": "percentile", 
                "feature_selection_params": {
                    "percentile": 30, 
                    "score_func": "mutual_info"
                }, 
                "correlation_threshold": 0.8
            },
            
            # Combinações de redução de dimensionalidade com seleção
            {
                "dimensionality_reduction": "pca", 
                "feature_selection": "model", 
                "correlation_threshold": 0.85
            },
            {
                "dimensionality_reduction": "pca", 
                "feature_selection": "kbest", 
                "feature_selection_params": {"k": 5}, 
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
        
        Returns:
            Dicionário com as configurações ótimas do preprocessador e feature engineer
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
        """
        Converte uma string para o tipo apropriado.
        
        Args:
            value_str: String a ser convertida
            
        Returns:
            Valor convertido para o tipo apropriado
        """
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
        """
        Cria um pipeline otimizado com base na melhor configuração encontrada.
        
        Returns:
            Instância configurada do DataPipeline
        """
        config = self.get_best_pipeline_config()
        if not config:
            logger.warning("Usando configuração padrão para o pipeline.")
            return DataPipeline()
            
        return DataPipeline(
            preprocessor_config=config.get('preprocessor_config', {}),
            feature_engineer_config=config.get('feature_engineer_config', {})
        )
        
    def get_transformation_statistics(self) -> Dict[str, Any]:
        """
        Retorna estatísticas sobre as transformações testadas.
        
        Returns:
            Dict com estatísticas sobre as transformações testadas
        """
        if not hasattr(self.tree, 'graph') or not self.tree.graph.nodes:
            return {"error": "Nenhuma transformação foi testada ainda."}
            
        # Contar transformações por tipo
        preproc_count = 0
        feat_count = 0
        total_transforms = 0
        
        scores = []
        
        for node in self.tree.graph.nodes:
            if node == "root":
                continue
                
            total_transforms += 1
            
            if "feat_" in node:
                feat_count += 1
            elif "preproc_" in node:
                preproc_count += 1
                
            # Coletar scores se disponíveis
            if 'score' in self.tree.graph.nodes[node]:
                score = self.tree.graph.nodes[node]['score']
                if score != float('-inf'):
                    scores.append(score)
        
        # Preparar estatísticas
        stats = {
            "total_transformations_tested": total_transforms,
            "preprocessor_transformations": preproc_count,
            "feature_engineering_transformations": feat_count,
            "average_score": np.mean(scores) if scores else None,
            "max_score": np.max(scores) if scores else None,
            "min_score": np.min(scores) if scores else None,
            "original_feature_count": self.original_feature_count
        }
        
        # Adicionar informação da melhor transformação
        best_transform = self.find_best_transformation()
        if best_transform:
            stats["best_transformation"] = best_transform
            
            # Tamanho do dataset na melhor transformação
            best_data = self.tree.graph.nodes[best_transform]['data']
            if best_data is not None:
                stats["best_transformation_shape"] = best_data.shape
                
                # Calcular a redução/expansão de features
                target_cols = ['target', 'classe', 'class', 'y', 'label']
                feature_cols = [col for col in best_data.columns if col.lower() not in target_cols]
                stats["transformed_feature_count"] = len(feature_cols)
                stats["feature_change_pct"] = (len(feature_cols) - self.original_feature_count) / self.original_feature_count * 100
                
                # Verificar o tipo da transformação
                if "power" in best_transform:
                    stats["transformation_type"] = "power_transform"
                elif "quantile" in best_transform:
                    stats["transformation_type"] = "quantile_transform"
                elif "robust" in best_transform:
                    stats["transformation_type"] = "robust_scaling"
                elif "kbins" in best_transform:
                    stats["transformation_type"] = "discretization"
                elif "target" in best_transform:
                    stats["transformation_type"] = "target_encoding"
                else:
                    stats["transformation_type"] = "standard"
                
                # Adicionar informações sobre redução/expansão
                if stats["feature_change_pct"] < 0:
                    stats["dimension_change"] = "reduction"
                    stats["dimension_change_amount"] = abs(stats["feature_change_pct"])
                else:
                    stats["dimension_change"] = "expansion"
                    stats["dimension_change_amount"] = stats["feature_change_pct"]
        
        # Adicionar informações sobre as transformações mais populares
        # (aquelas que foram bem-sucedidas, independente do score)
        transformation_types = {}
        for node in self.tree.graph.nodes:
            if node == "root":
                continue
                
            for transform_type in ["power", "robust", "standard", "minmax", "quantile", 
                                  "target", "kbins", "binarizer", "onehot", "ordinal"]:
                if transform_type in node:
                    transformation_types[transform_type] = transformation_types.get(transform_type, 0) + 1
                    
        # Adicionar os tipos de transformação mais comuns
        if transformation_types:
            # Ordenar por frequência (mais comum primeiro)
            sorted_types = sorted(transformation_types.items(), key=lambda x: x[1], reverse=True)
            stats["most_common_transformations"] = sorted_types[:5]  # Top 5
        
        # Adicionar informações sobre configurações do preprocessador mais bem-sucedidas
        if scores and len(scores) > 0:
            # Encontrar os nós com os melhores scores que são apenas preprocessadores (não feature engineering)
            best_preproc_nodes = []
            for node in self.tree.graph.nodes:
                if node != "root" and "preproc_" in node and "feat_" not in node:
                    if 'score' in self.tree.graph.nodes[node]:
                        score = self.tree.graph.nodes[node]['score']
                        if score != float('-inf'):
                            best_preproc_nodes.append((node, score))
            
            # Ordenar por score e pegar os top 3
            if best_preproc_nodes:
                best_preproc_nodes.sort(key=lambda x: x[1], reverse=True)
                stats["best_preprocessor_configurations"] = [node[0] for node in best_preproc_nodes[:3]]
        
        return stats
    
    def visualize_transformations(self, output_path: Optional[str] = None) -> None:
        """
        Visualiza a árvore de transformações testadas e gera um gráfico.
        
        Args:
            output_path: Caminho opcional para salvar a visualização. Se None, apenas exibe.
        """
        import matplotlib.pyplot as plt
        import networkx as nx
        
        if not hasattr(self.tree, 'graph') or not self.tree.graph.nodes:
            logger.warning("Nenhuma transformação foi testada ainda. Não há nada para visualizar.")
            return
            
        # Copiar o grafo para não modificar o original
        graph = self.tree.graph.copy()
        
        # Encontrar o melhor nó
        best_node = self.find_best_transformation()
        
        # Colorir os nós
        node_colors = []
        for node in graph.nodes():
            if node == "root":
                node_colors.append("green")
            elif node == best_node:
                node_colors.append("red")  # Melhor nó
            elif "preproc_" in node and "feat_" not in node:
                node_colors.append("skyblue")  # Apenas preprocessamento
            else:
                node_colors.append("orange")  # Nós de feature engineering
                
        # Definir os rótulos dos nós
        node_labels = {}
        for node in graph.nodes():
            if node == "root":
                node_labels[node] = "Dados Originais"
            else:
                # Abreviar o nome para melhor visualização
                parts = node.split('_')
                if "preproc" in parts and len(parts) > 2:
                    short_name = parts[1][:7] + "..."
                    node_labels[node] = short_name
                elif "feat" in parts and len(parts) > 2:
                    short_name = parts[-1][:7] + "..."
                    node_labels[node] = short_name
                else:
                    node_labels[node] = node
        
        # Criar a figura
        plt.figure(figsize=(15, 10))
        
        # Desenhar o grafo
        pos = nx.spring_layout(graph, seed=42)  # Posicionamento consistente
        nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=500, alpha=0.8)
        nx.draw_networkx_edges(graph, pos, width=1.0, alpha=0.5)
        nx.draw_networkx_labels(graph, pos, labels=node_labels, font_size=10)
        
        plt.title("Árvore de Transformações")
        plt.axis("off")
        
        # Adicionar legenda
        plt.plot([0], [0], 'o', color='green', label='Dados Originais')
        plt.plot([0], [0], 'o', color='skyblue', label='Preprocessamento')
        plt.plot([0], [0], 'o', color='orange', label='Feature Engineering')
        plt.plot([0], [0], 'o', color='red', label='Melhor Transformação')
        plt.legend()
        
        # Salvar ou mostrar
        if output_path:
            plt.savefig(output_path, bbox_inches='tight')
            logger.info(f"Visualização salva em {output_path}")
        else:
            plt.show()
            
def export_transformation_graph(self, output_path: str) -> None:
        """
        Exporta a árvore de transformações em formato GraphML para visualização em ferramentas externas.
        
        Args:
            output_path: Caminho para salvar o arquivo GraphML
        """
        import networkx as nx
        
        if not hasattr(self.tree, 'graph') or not self.tree.graph.nodes:
            logger.warning("Nenhuma transformação foi testada ainda. Não há nada para exportar.")
            return
            
        # Copiar o grafo para não modificar o original
        graph = self.tree.graph.copy()
        
        # Remover os atributos 'data' que contêm os DataFrames, pois não são serializáveis
        for node in graph.nodes():
            if 'data' in graph.nodes[node]:
                graph.nodes[node]['data'] = None
                
            # Adicionar atributos para melhor visualização
            if node == "root":
                graph.nodes[node]['type'] = "original"
                graph.nodes[node]['label'] = "Original Data"
            elif "preproc_" in node and "feat_" not in node:
                graph.nodes[node]['type'] = "preprocessor"
                graph.nodes[node]['label'] = node.replace("preproc_", "")
            else:
                graph.nodes[node]['type'] = "feature_engineer"
                graph.nodes[node]['label'] = node.split("feat_")[-1] if "feat_" in node else node
                
            # Adicionar score como atributo
            if 'score' in graph.nodes[node]:
                graph.nodes[node]['score'] = float(graph.nodes[node]['score'])
        
        # Exportar para GraphML
        nx.write_graphml(graph, output_path)
        logger.info(f"Grafo de transformações exportado para {output_path}")
        
def get_feature_importance_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analisa a importância das features nos dados originais e transformados.
        
        Args:
            df: DataFrame original com a coluna target
            
        Returns:
            Dicionário com análise de importância de features
        """
        if self.target_col is None or self.target_col not in df.columns:
            logger.warning("Coluna target não especificada ou não encontrada. Impossível calcular importância de features.")
            return {"error": "Coluna target não disponível"}
            
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.inspection import permutation_importance
        
        # Encontrar o melhor dataset transformado
        best_transform = self.find_best_transformation()
        if not best_transform or best_transform not in self.tree.graph.nodes:
            logger.warning("Nenhuma transformação válida encontrada para análise de importância.")
            return {"error": "Nenhuma transformação válida"}
            
        best_data = self.tree.graph.nodes[best_transform]['data']
        
        # Separar features e target nos dados originais
        X_orig = df.drop(columns=[self.target_col])
        y_orig = df[self.target_col]
        
        # Separar features e target nos dados transformados
        X_trans = best_data.drop(columns=[self.target_col])
        y_trans = best_data[self.target_col]
        
        # Determinar o tipo de tarefa (classificação ou regressão)
        task = 'classification'  # Padrão
        if y_orig.dtype.kind in 'fc':  # float ou complex
            task = 'regression'
            
        # Criar e treinar modelos
        if task == 'classification':
            model_orig = RandomForestClassifier(n_estimators=50, random_state=42)
            model_trans = RandomForestClassifier(n_estimators=50, random_state=42)
        else:
            model_orig = RandomForestRegressor(n_estimators=50, random_state=42)
            model_trans = RandomForestRegressor(n_estimators=50, random_state=42)
            
        # Treinar modelos
        model_orig.fit(X_orig, y_orig)
        model_trans.fit(X_trans, y_trans)
        
        # Calcular importâncias das features originais
        orig_importances = pd.DataFrame({
            'feature': X_orig.columns,
            'importance': model_orig.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Calcular importâncias das features transformadas
        trans_importances = pd.DataFrame({
            'feature': X_trans.columns,
            'importance': model_trans.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Calcular importância por permutação (mais robusta) para o dataset transformado
        perm_importance = permutation_importance(
            model_trans, X_trans, y_trans, n_repeats=10, random_state=42
        )
        
        perm_importances = pd.DataFrame({
            'feature': X_trans.columns,
            'importance': perm_importance.importances_mean
        }).sort_values('importance', ascending=False)
        
        # Preparar resultados
        result = {
            'original_importances': orig_importances.to_dict('records'),
            'transformed_importances': trans_importances.to_dict('records'),
            'permutation_importances': perm_importances.to_dict('records'),
            'top_original_features': orig_importances.head(10)['feature'].tolist(),
            'top_transformed_features': trans_importances.head(10)['feature'].tolist(),
            'transformation': best_transform
        }
        
        return resultimport
