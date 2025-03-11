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
    def dimension_preserving_heuristic(df: pd.DataFrame, original_feature_count: int = None) -> float:
        """
        Heurística que prioriza manter o número original de features enquanto
        maximiza a qualidade das features (baixa correlação, boa distribuição).
        
        Args:
            df (pd.DataFrame): DataFrame transformado a ser avaliado
            original_feature_count (int, opcional): Número original de features para referência.
                Se não for fornecido, usa o número atual de features.
            
        Returns:
            float: Pontuação da qualidade do DataFrame (maior é melhor)
        """
        # Se não for fornecido, assume o número atual como referência
        if original_feature_count is None:
            # Tenta identificar o número esperado de features no contexto atual
            target_cols = ['target', 'classe', 'class', 'y']
            possible_target_col = [col for col in df.columns if col.lower() in target_cols]
            current_features = df.shape[1] - len(possible_target_col)
            original_feature_count = current_features
        
        # Penalidade por desvio do número original de features
        # Quanto maior a diferença, maior a penalidade
        feature_count_deviation = abs(df.shape[1] - original_feature_count)
        feature_penalty = feature_count_deviation / max(1, original_feature_count)
        
        # Penaliza alta correlação entre features
        correlation_penalty = 0
        if df.shape[1] > 1:
            # Selecionar apenas colunas numéricas para correlação
            numeric_df = df.select_dtypes(include=['number'])
            if not numeric_df.empty and numeric_df.shape[1] > 1:
                try:
                    correlation_matrix = numeric_df.corr().abs()
                    # Remove a diagonal (correlação de cada feature consigo mesma)
                    high_corr = (correlation_matrix > 0.90).sum().sum() - numeric_df.shape[1]  
                    correlation_penalty = high_corr / (numeric_df.shape[1] ** 2)  # Normaliza penalização
                except Exception as e:
                    # Em caso de erro no cálculo da correlação, define penalidade zero
                    correlation_penalty = 0
        
        # Recompensa distribuição mais próxima da normal (menor skewness)
        normality_score = 0
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            skew_values = []
            for col in numeric_cols:
                # Usando skewness como medida (0 = distribuição perfeitamente simétrica)
                try:
                    if df[col].nunique() > 1:  # Evita erro com colunas constantes
                        skew = abs(df[col].skew())
                        skew_values.append(skew)
                except:
                    continue
            
            if skew_values:
                avg_skew = sum(skew_values) / len(skew_values)
                normality_score = 1 / (1 + avg_skew)  # Normaliza entre 0 e 1
        
        # Avalia diversidade de variáveis categóricas (recompensa)
        categorical_diversity_score = 0
        categorical_features = df.select_dtypes(include=['object', 'category'])
        if not categorical_features.empty:
            unique_counts = categorical_features.nunique()
            if unique_counts.max() > 0:
                categorical_diversity_score = unique_counts.mean() / unique_counts.max()  # Normaliza entre 0 e 1
        
        # Penalidade forte para desvios grandes do número de features
        dimensional_penalty = feature_penalty * 5  # Peso alto para enfatizar a preservação dimensional
        
        # Score final: balanceia os diferentes componentes
        final_score = (
            normality_score * 0.3 +           # 30% para normalidade
            categorical_diversity_score * 0.2  # 20% para diversidade categórica
            - correlation_penalty * 0.5       # 50% de penalidade por correlação alta
            - dimensional_penalty             # Penalidade dimensão com peso 5x
        )
        
        return final_score

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
    
    @staticmethod
    def combined_heuristic(df: pd.DataFrame, original_feature_count: int = None) -> float:
        """
        Heurística combinada que integra tanto a preservação da dimensionalidade original
        quanto a análise de qualidade das features baseada na correlação e diversidade.
        
        Args:
            df (pd.DataFrame): DataFrame transformado a ser avaliado
            original_feature_count (int, opcional): Número original de features para referência.
                Se não for fornecido, tenta detectar automaticamente.
            
        Returns:
            float: Pontuação da qualidade do DataFrame (maior é melhor)
        """
        # Determinar o número original de features
        if original_feature_count is None:
            # Tenta identificar o número esperado de features no contexto atual
            target_cols = ['target', 'classe', 'class', 'y', 'label']
            possible_target_col = [col for col in df.columns if col.lower() in target_cols]
            original_feature_count = df.shape[1] - len(possible_target_col)
        
        # ---- COMPONENTE 1: PRESERVAÇÃO DIMENSIONAL ----
        # Penalidade por desvio do número original de features
        feature_count_deviation = abs(df.shape[1] - original_feature_count)
        feature_penalty = feature_count_deviation / max(1, original_feature_count)
        
        # ---- COMPONENTE 2: CORRELAÇÃO ENTRE FEATURES ----
        correlation_penalty = 0
        if df.shape[1] > 1:
            numeric_df = df.select_dtypes(include=['number'])
            if not numeric_df.empty and numeric_df.shape[1] > 1:
                try:
                    correlation_matrix = numeric_df.corr().abs()
                    # Remove a diagonal principal
                    high_corr = (correlation_matrix > 0.95).sum().sum() - numeric_df.shape[1]
                    correlation_penalty = high_corr / (numeric_df.shape[1] ** 2)
                except Exception as e:
                    correlation_penalty = 0
        
        # ---- COMPONENTE 3: DIVERSIDADE CATEGÓRICA ----
        categorical_diversity_score = 0
        categorical_features = df.select_dtypes(include=['object', 'category'])
        if not categorical_features.empty:
            try:
                unique_counts = categorical_features.nunique()
                if unique_counts.max() > 0:
                    categorical_diversity_score = unique_counts.mean() / unique_counts.max()
            except Exception:
                categorical_diversity_score = 0
        
        # ---- COMPONENTE 4: NORMALIDADE DAS DISTRIBUIÇÕES ----
        normality_score = 0
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            skew_values = []
            for col in numeric_cols:
                try:
                    if df[col].nunique() > 1:
                        skew = abs(df[col].skew())
                        skew_values.append(skew)
                except:
                    continue
            
            if skew_values:
                avg_skew = sum(skew_values) / len(skew_values)
                normality_score = 1 / (1 + avg_skew)
        
        # ---- PONDERAÇÃO FINAL ----
        # Definir pesos para cada componente
        dimension_weight = 4.0       # Peso alto para preservação dimensional
        correlation_weight = 1.0     # Peso médio para penalidade por correlação
        categorical_weight = 0.5     # Peso baixo para diversidade categórica
        normality_weight = 0.5       # Peso baixo para normalidade
        
        # Calcular pontuação final
        final_score = (
            -dimension_weight * feature_penalty +              # Penalidade pela diferença dimensional
            -correlation_weight * correlation_penalty +        # Penalidade por alta correlação
            categorical_weight * categorical_diversity_score + # Recompensa por diversidade categórica
            normality_weight * normality_score                 # Recompensa por normalidade
        )
        
        return final_score
    
    @staticmethod
    def strict_dimension_heuristic(df: pd.DataFrame, original_feature_count: int = None, max_expansion_factor: float = 1.1) -> float:
        """
        Heurística que impõe um limite estrito no número de features,
        rejeitando completamente transformações que ultrapassem um limite máximo.
        
        Args:
            df (pd.DataFrame): DataFrame transformado a ser avaliado
            original_feature_count (int): Número original de features no dataset
            max_expansion_factor (float): Fator máximo de expansão permitido (ex: 1.5 = 50% mais features)
            
        Returns:
            float: Pontuação da heurística (maior é melhor, -inf para transformações rejeitadas)
        """
        # Determinar o número original de features se não fornecido
        if original_feature_count is None:
            # Assumimos que é o wine dataset com 13 features originais
            original_feature_count = 13
        
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
            except Exception:
                pass
        
        # ---- Componente 3: Variância explicada ----
        # Premia features com maior variância (mais informativas)
        variance_score = 0
        if not numeric_df.empty:
            try:
                normalized_variances = numeric_df.var() / numeric_df.var().max()
                variance_score = normalized_variances.mean()
            except Exception:
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
        self.search = HeuristicSearch(heuristic or HeuristicSearch.strict_dimension_heuristic)
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
            {"missing_values_strategy": "weighted_mean"},
            {"missing_values_strategy": "interpolation"},
            {"outlier_method": "iqr"},
            {"outlier_method": "zscore"},
            {"outlier_method": "isolation_forest"},
            {"outlier_method": "iqr_zscore"},  # Combinação de métodos
            {"outlier_method": "iqr_isolation_forest"},
            {"categorical_strategy": "onehot"},
            {"categorical_strategy": "ordinal"},
            {"categorical_strategy": "target_encoding"},
            {"categorical_strategy": "binary"},
            {"scaling": "standard"},
            {"scaling": "minmax"},
            {"scaling": "robust"},
            {"scaling": "normalization"},
            {"scaling": "power_transform"},
        ]

        feature_configs = [
            {"dimensionality_reduction": "pca"},
            {"dimensionality_reduction": "ica"},
            {"dimensionality_reduction": "umap"},
            {"dimensionality_reduction": None},
            {"feature_selection": "variance"},
            {"feature_selection": "mutual_info"},
            {"feature_selection": "lasso"},
            {"feature_selection": None},
            {"generate_features": True},
            {"generate_features": False},
            {"correlation_threshold": 0.95},
            {"correlation_threshold": 0.90},
            {"correlation_threshold": 0.85},
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
