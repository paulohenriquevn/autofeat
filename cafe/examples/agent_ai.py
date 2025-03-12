import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Any, Union, Tuple
import logging
import re
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Importações do CAFE
from cafe import (
    PreProcessor, 
    FeatureEngineer, 
    PerformanceValidator, 
    DataPipeline, 
    Explorer,
    TransformationTree,
    HeuristicSearch
)

class CAFEAgent:
    """
    Agente de IA Generativa para interagir com o sistema CAFE.
    Fornece uma interface em linguagem natural para explorar, analisar e transformar dados
    utilizando os componentes do CAFE.
    """
    
    def __init__(self, verbose: bool = True):
        """
        Inicializa o agente CAFE.
        
        Args:
            verbose: Se True, exibe mensagens detalhadas durante a execução
        """
        self.verbose = verbose
        self.datasets = {}  # Dicionário para armazenar datasets carregados
        self.current_dataset = None  # Nome do dataset atualmente em uso
        self.transformed_datasets = {}  # Dicionário para armazenar datasets transformados
        self.explorers = {}  # Dicionário para armazenar explorers por dataset
        self.pipelines = {}  # Dicionário para armazenar pipelines por dataset
        self.results = {}  # Dicionário para armazenar resultados de análises
        
        # Configurar logging
        self.logger = logging.getLogger("CAFEAgent")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO if verbose else logging.WARNING)
        
        self.logger.info("CAFEAgent inicializado com sucesso")
        
        # Comandos suportados pelo agente
        self.commands = {
            'carregar': self._load_data,
            'explorar': self._explore_data,
            'transformar': self._transform_data,
            'validar': self._validate_transformations,
            'recomendar': self._recommend_transformations,
            'explicar': self._explain_transformations,
            'salvar': self._save_data,
            'pipeline': self._create_pipeline,
            'ajuda': self._help,
        }
        
        # Análises suportadas
        self.analysis_types = {
            'missing': self._analyze_missing_values,
            'outliers': self._analyze_outliers,
            'correlacao': self._analyze_correlation,
            'distribuicao': self._analyze_distribution,
            'importancia': self._analyze_feature_importance,
            'categoricas': self._analyze_categorical,
            'temporais': self._analyze_datetime,
        }
        
    def process_command(self, command: str) -> str:
        """
        Processa um comando em linguagem natural e retorna a resposta.
        
        Args:
            command: Comando em linguagem natural
            
        Returns:
            Resposta ao comando
        """
        try:
            # Normalizar o comando
            command = command.lower().strip()
            
            # Verificar comandos de ajuda
            if 'ajuda' in command or 'help' in command:
                return self._help(command)
                
            # Identificar qual comando está sendo solicitado
            for cmd_key, cmd_func in self.commands.items():
                if cmd_key in command:
                    # Extrair argumentos do comando
                    args = self._extract_args(command, cmd_key)
                    return cmd_func(args)
            
            # Se não identificar nenhum comando específico, tentar interpretar como uma análise
            for analysis_key, analysis_func in self.analysis_types.items():
                if analysis_key in command:
                    args = self._extract_args(command, analysis_key)
                    return analysis_func(args)
            
            # Se não encontrar nenhum comando ou análise
            return "Não entendi o comando. Por favor, tente novamente ou digite 'ajuda' para ver os comandos disponíveis."
        
        except Exception as e:
            self.logger.error(f"Erro ao processar comando: {e}", exc_info=True)
            return f"Ocorreu um erro ao processar seu comando: {str(e)}"
    
    def _extract_args(self, command: str, cmd_key: str) -> Dict[str, Any]:
        """Extrai argumentos de um comando em linguagem natural"""
        args = {}
        
        # Extrair nome do dataset
        dataset_pattern = r'(?:dataset|dados|arquivo|tabela|df)\s+([a-zA-Z0-9_]+)'
        dataset_match = re.search(dataset_pattern, command)
        if dataset_match:
            args['dataset'] = dataset_match.group(1)
        elif self.current_dataset:
            args['dataset'] = self.current_dataset
            
        # Extrair caminho do arquivo
        file_pattern = r'(?:arquivo|caminho|file|path)\s+[\'\"]?([a-zA-Z0-9_\./\\-]+\.(?:csv|xlsx|parquet|json))[\'\"]?'
        file_match = re.search(file_pattern, command)
        if file_match:
            args['file_path'] = file_match.group(1)
            
        # Extrair coluna alvo
        target_pattern = r'(?:alvo|target|classe|class|y|variável dependente)\s+[\'\"]?([a-zA-Z0-9_]+)[\'\"]?'
        target_match = re.search(target_pattern, command)
        if target_match:
            args['target_col'] = target_match.group(1)
            
        # Extrair colunas específicas
        columns_pattern = r'(?:colunas|features|variaveis|campos)\s+[\'\"]?([a-zA-Z0-9_, ]+)[\'\"]?'
        columns_match = re.search(columns_pattern, command)
        if columns_match:
            columns_str = columns_match.group(1)
            args['columns'] = [col.strip() for col in columns_str.split(',')]
            
        # Extrair parâmetros de configuração
        config_pattern = r'(?:config|configuracao|parametros)\s+(\{[^\}]+\})'
        config_match = re.search(config_pattern, command)
        if config_match:
            try:
                config_str = config_match.group(1)
                args['config'] = json.loads(config_str)
            except json.JSONDecodeError:
                self.logger.warning(f"Não foi possível interpretar a configuração JSON: {config_str}")
                
        # Extrair limiar (threshold)
        threshold_pattern = r'(?:limiar|threshold|corte)\s+(0\.\d+|\d+\.?\d*)'
        threshold_match = re.search(threshold_pattern, command)
        if threshold_match:
            args['threshold'] = float(threshold_match.group(1))
            
        # Extrair nome de arquivo para salvar
        save_pattern = r'(?:salvar|exportar|save|gravar)\s+(?:como|as|para)?\s+[\'\"]?([a-zA-Z0-9_\./\\-]+)[\'\"]?'
        save_match = re.search(save_pattern, command)
        if save_match:
            args['output_path'] = save_match.group(1)
            
        return args
        
    def _load_data(self, args: Dict[str, Any]) -> str:
        """Carrega um conjunto de dados"""
        if 'file_path' not in args:
            return "Por favor, especifique um caminho de arquivo para carregar os dados."
            
        file_path = args['file_path']
        dataset_name = args.get('dataset', Path(file_path).stem)
        
        try:
            # Determinar o formato do arquivo pela extensão
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith('.xlsx'):
                df = pd.read_excel(file_path)
            elif file_path.endswith('.parquet'):
                df = pd.read_parquet(file_path)
            elif file_path.endswith('.json'):
                df = pd.read_json(file_path)
            else:
                return f"Formato de arquivo não suportado: {file_path}"
                
            # Armazenar o dataset
            self.datasets[dataset_name] = df
            self.current_dataset = dataset_name
            
            # Gerar informações básicas sobre o dataset
            info = self._generate_dataset_info(df)
            
            return f"Dataset '{dataset_name}' carregado com sucesso!\n\n{info}"
            
        except Exception as e:
            self.logger.error(f"Erro ao carregar dataset: {e}", exc_info=True)
            return f"Erro ao carregar o dataset: {str(e)}"
    
    def _generate_dataset_info(self, df: pd.DataFrame) -> str:
        """Gera informações básicas sobre um DataFrame"""
        info = f"Dimensões: {df.shape[0]} linhas x {df.shape[1]} colunas\n\n"
        
        # Tipos de dados
        dtype_counts = df.dtypes.value_counts().to_dict()
        info += "Tipos de dados:\n"
        for dtype, count in dtype_counts.items():
            info += f"- {dtype}: {count} colunas\n"
        
        # Valores ausentes
        missing = df.isnull().sum()
        if missing.sum() > 0:
            info += "\nColunas com valores ausentes:\n"
            for col, count in missing[missing > 0].items():
                percentage = count / len(df) * 100
                info += f"- {col}: {count} ({percentage:.2f}%)\n"
        else:
            info += "\nNão há valores ausentes no dataset.\n"
            
        # Primeiras linhas
        info += "\nPrimeiras 5 linhas:\n"
        info += df.head().to_string()
        
        return info
        
    def _explore_data(self, args: Dict[str, Any]) -> str:
        """Explora um conjunto de dados"""
        dataset_name = args.get('dataset', self.current_dataset)
        
        if not dataset_name or dataset_name not in self.datasets:
            return "Dataset não encontrado. Por favor, carregue um dataset primeiro."
            
        df = self.datasets[dataset_name]
        
        # Verificar se é solicitada uma análise específica
        if 'columns' in args:
            columns = args['columns']
            df_subset = df[columns]
            return self._generate_dataset_info(df_subset)
        
        # Análise exploratória completa
        info = self._generate_dataset_info(df)
        
        # Estatísticas descritivas para colunas numéricas
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            stats = df[numeric_cols].describe().transpose()
            info += "\n\nEstatísticas para colunas numéricas:\n"
            info += stats.to_string()
            
        # Distribuição de categorias para colunas categóricas
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(cat_cols) > 0:
            info += "\n\nDistribuição de colunas categóricas:\n"
            for col in cat_cols[:5]:  # Limitar a 5 colunas para não sobrecarregar
                value_counts = df[col].value_counts().head(5)
                info += f"\n{col}:\n{value_counts.to_string()}\n"
                if df[col].nunique() > 5:
                    info += f"... e {df[col].nunique() - 5} outros valores únicos\n"
        
        return info
        
    def _transform_data(self, args: Dict[str, Any]) -> str:
        """Transforma um conjunto de dados usando o CAFE"""
        dataset_name = args.get('dataset', self.current_dataset)
        
        if not dataset_name or dataset_name not in self.datasets:
            return "Dataset não encontrado. Por favor, carregue um dataset primeiro."
            
        df = self.datasets[dataset_name]
        target_col = args.get('target_col')
        
        try:
            # Configurar o pipeline do CAFE
            preprocessor_config = args.get('preprocessor_config', {})
            feature_engineer_config = args.get('feature_engineer_config', {})
            
            # Criar DataPipeline
            pipeline = DataPipeline(
                preprocessor_config=preprocessor_config,
                feature_engineer_config=feature_engineer_config,
                auto_validate=True
            )
            
            # Transformar os dados
            df_transformed = pipeline.fit_transform(df, target_col=target_col)
            
            # Armazenar o resultado
            transformed_name = f"{dataset_name}_transformed"
            self.transformed_datasets[transformed_name] = df_transformed
            self.pipelines[transformed_name] = pipeline
            
            # Gerar relatório de transformação
            report = self._generate_transformation_report(df, df_transformed, pipeline)
            
            return f"Dataset transformado com sucesso! Novo dataset: '{transformed_name}'\n\n{report}"
            
        except Exception as e:
            self.logger.error(f"Erro ao transformar dataset: {e}", exc_info=True)
            return f"Erro ao transformar o dataset: {str(e)}"
    
    def _generate_transformation_report(self, df_original: pd.DataFrame, 
                                       df_transformed: pd.DataFrame,
                                       pipeline: DataPipeline) -> str:
        """Gera um relatório das transformações aplicadas"""
        report = "Relatório de Transformação:\n"
        report += f"Dimensões originais: {df_original.shape[0]} linhas x {df_original.shape[1]} colunas\n"
        report += f"Dimensões após transformação: {df_transformed.shape[0]} linhas x {df_transformed.shape[1]} colunas\n"
        
        # Mudança nas features
        feature_diff = df_transformed.shape[1] - df_original.shape[1]
        if feature_diff > 0:
            report += f"Adicionadas {feature_diff} novas features\n"
        elif feature_diff < 0:
            report += f"Removidas {abs(feature_diff)} features\n"
        else:
            report += "Número de features mantido igual\n"
            
        # Resultados da validação
        validation_results = pipeline.get_validation_results()
        if validation_results:
            report += "\nResultados da Validação de Performance:\n"
            report += f"Performance dataset original: {validation_results['performance_original']:.4f}\n"
            report += f"Performance dataset transformado: {validation_results['performance_transformed']:.4f}\n"
            report += f"Diferença: {validation_results['performance_diff_pct']:.2f}%\n"
            report += f"Decisão do sistema: Usar dataset {validation_results['best_choice']}\n"
        
        # Listar novas features
        if feature_diff > 0:
            new_features = set(df_transformed.columns) - set(df_original.columns)
            if len(new_features) > 0:
                report += "\nNovas features criadas:\n"
                for feature in new_features:
                    if feature != 'target' and (target_col := pipeline.target_col) and feature != target_col:
                        report += f"- {feature}\n"
        
        return report
    
    def _validate_transformations(self, args: Dict[str, Any]) -> str:
        """Valida as transformações de um conjunto de dados"""
        dataset_name = args.get('dataset', self.current_dataset)
        transformed_name = f"{dataset_name}_transformed"
        
        if not dataset_name or dataset_name not in self.datasets:
            return "Dataset original não encontrado."
            
        if transformed_name not in self.transformed_datasets:
            return "Dataset transformado não encontrado. Execute uma transformação primeiro."
            
        if transformed_name not in self.pipelines:
            return "Pipeline não encontrado para este dataset transformado."
            
        pipeline = self.pipelines[transformed_name]
        validation_results = pipeline.get_validation_results()
        
        if not validation_results:
            return "Não há resultados de validação disponíveis para este pipeline."
            
        # Gerar relatório de validação mais detalhado
        report = "Relatório de Validação de Performance:\n\n"
        report += f"Performance no dataset original: {validation_results['performance_original']:.4f}\n"
        report += f"Performance no dataset transformado: {validation_results['performance_transformed']:.4f}\n"
        report += f"Diferença absoluta: {validation_results['performance_diff']:.4f}\n"
        report += f"Diferença percentual: {validation_results['performance_diff_pct']:.2f}%\n\n"
        
        report += f"Decisão do sistema: Usar dataset {validation_results['best_choice'].upper()}\n\n"
        
        report += "Performance por fold de validação cruzada:\n"
        report += "Original: " + ", ".join([f"{score:.4f}" for score in validation_results['scores_original']]) + "\n"
        report += "Transformado: " + ", ".join([f"{score:.4f}" for score in validation_results['scores_transformed']]) + "\n\n"
        
        report += f"Redução de features: {validation_results['feature_reduction']*100:.1f}%\n"
        
        # Obter e incluir importância de features
        try:
            df = self.datasets[dataset_name]
            target_col = pipeline.target_col
            if target_col and target_col in df.columns:
                feature_importance = pipeline.get_feature_importance(df, target_col=target_col)
                report += "\nImportância das Features (Top 10):\n"
                for _, row in feature_importance.head(10).iterrows():
                    report += f"- {row['feature']}: {row['importance']:.4f}\n"
        except Exception as e:
            report += f"\nNão foi possível calcular a importância das features: {str(e)}\n"
        
        return report
    
    def _recommend_transformations(self, args: Dict[str, Any]) -> str:
        """Recomenda transformações para um conjunto de dados"""
        dataset_name = args.get('dataset', self.current_dataset)
        
        if not dataset_name or dataset_name not in self.datasets:
            return "Dataset não encontrado. Por favor, carregue um dataset primeiro."
            
        df = self.datasets[dataset_name]
        target_col = args.get('target_col')
        
        try:
            # Criar o Explorer para buscar a melhor configuração
            explorer = Explorer(target_col=target_col)
            
            # Executar análise de transformações
            self.logger.info(f"Iniciando análise de transformações para o dataset '{dataset_name}'")
            best_data = explorer.analyze_transformations(df)
            best_config = explorer.get_best_pipeline_config()
            
            # Armazenar o Explorer para uso futuro
            self.explorers[dataset_name] = explorer
            
            # Gerar relatório de recomendações
            report = "Recomendações de Transformações:\n\n"
            
            # Detalhes da configuração recomendada
            report += "Configuração de Preprocessamento Recomendada:\n"
            for key, value in best_config.get('preprocessor_config', {}).items():
                report += f"- {key}: {value}\n"
                
            report += "\nConfiguração de Feature Engineering Recomendada:\n"
            for key, value in best_config.get('feature_engineer_config', {}).items():
                report += f"- {key}: {value}\n"
                
            # Obter estatísticas das transformações
            transformation_stats = explorer.get_transformation_statistics()
            report += "\nEstatísticas das Transformações:\n"
            report += f"- Total de transformações testadas: {transformation_stats.get('total_transformations_tested', 0)}\n"
            report += f"- Melhor transformação: {transformation_stats.get('best_transformation', 'N/A')}\n"
            
            if 'transformed_feature_count' in transformation_stats and 'original_feature_count' in transformation_stats:
                feature_change = transformation_stats['transformed_feature_count'] - transformation_stats['original_feature_count']
                if feature_change > 0:
                    report += f"- Adição de features: +{feature_change} features\n"
                elif feature_change < 0:
                    report += f"- Redução de features: {feature_change} features\n"
                else:
                    report += "- Número de features mantido igual\n"
                    
            # Adicionar informações sobre transformações mais comuns
            if 'most_common_transformations' in transformation_stats:
                report += "\nTransformações Mais Comuns Testadas:\n"
                for transform_type, count in transformation_stats['most_common_transformations']:
                    report += f"- {transform_type}: {count} vezes\n"
            
            # Instruções para aplicar as recomendações
            report += "\nPara aplicar estas recomendações, execute o comando:\n"
            report += f"transformar dataset {dataset_name}"
            if target_col:
                report += f" target {target_col}"
            
            return report
            
        except Exception as e:
            self.logger.error(f"Erro ao recomendar transformações: {e}", exc_info=True)
            return f"Erro ao gerar recomendações: {str(e)}"
    
    def _explain_transformations(self, args: Dict[str, Any]) -> str:
        """Explica as transformações aplicadas a um conjunto de dados"""
        dataset_name = args.get('dataset', self.current_dataset)
        transformed_name = f"{dataset_name}_transformed"
        
        if transformed_name not in self.pipelines:
            return "Não encontrei um pipeline para este dataset. Execute uma transformação primeiro."
            
        pipeline = self.pipelines[transformed_name]
        
        # Gerar explicações sobre as transformações
        explanation = "Explicação das Transformações Aplicadas:\n\n"
        
        # Explicar preprocessamento
        preprocessor = pipeline.preprocessor
        if hasattr(preprocessor, 'get_transformer_description'):
            transformer_desc = preprocessor.get_transformer_description()
            
            explanation += "1. Etapa de Pré-processamento:\n"
            
            # Explicar tratamento de valores ausentes
            missing_strategy = transformer_desc['transformers'].get('missing_values', 'none')
            explanation += f"   - Tratamento de valores ausentes: {missing_strategy}\n"
            
            # Explicar transformação de escala
            scaling = transformer_desc['transformers'].get('scaling', 'none')
            explanation += f"   - Normalização/padronização: {scaling}\n"
            
            # Explicar estratégia categórica
            cat_strategy = transformer_desc['transformers'].get('categorical_strategy', 'none')
            explanation += f"   - Codificação de variáveis categóricas: {cat_strategy}\n"
            
            # Explicar transformadores adicionais
            additional = transformer_desc['transformers'].get('additional_transformers', [])
            if additional:
                explanation += f"   - Transformadores adicionais: {', '.join(additional)}\n"
        
        # Explicar feature engineering
        feature_engineer = pipeline.feature_engineer
        if hasattr(feature_engineer, 'config'):
            explanation += "\n2. Etapa de Engenharia de Features:\n"
            
            # Explicar redução de dimensionalidade
            dim_reduction = feature_engineer.config.get('dimensionality_reduction')
            if dim_reduction:
                explanation += f"   - Redução de dimensionalidade: {dim_reduction}\n"
            
            # Explicar seleção de features
            feature_selection = feature_engineer.config.get('feature_selection')
            if feature_selection:
                explanation += f"   - Seleção de features: {feature_selection}\n"
                
                # Parâmetros da seleção
                selection_params = feature_engineer.config.get('feature_selection_params', {})
                if selection_params:
                    explanation += "     Parâmetros:\n"
                    for key, value in selection_params.items():
                        explanation += f"     - {key}: {value}\n"
            
            # Explicar geração de features
            if feature_engineer.config.get('generate_features'):
                explanation += "   - Geração de features polinomiais ativada\n"
            
            # Explicar remoção de alta correlação
            correlation = feature_engineer.config.get('correlation_threshold')
            explanation += f"   - Limiar de correlação para remoção: {correlation}\n"
        
        # Adicionar impacto das transformações
        validation_results = pipeline.get_validation_results()
        if validation_results:
            explanation += "\n3. Impacto das Transformações:\n"
            
            perf_diff = validation_results['performance_diff_pct']
            if perf_diff > 0:
                explanation += f"   - Melhoria de performance: +{perf_diff:.2f}%\n"
            elif perf_diff < 0:
                explanation += f"   - Redução de performance: {perf_diff:.2f}%\n"
            else:
                explanation += "   - Performance mantida igual\n"
                
            feature_reduction = validation_results.get('feature_reduction', 0) * 100
            if feature_reduction > 0:
                explanation += f"   - Redução de features: {feature_reduction:.1f}%\n"
            elif feature_reduction < 0:
                explanation += f"   - Aumento de features: {abs(feature_reduction):.1f}%\n"
            else:
                explanation += "   - Número de features mantido igual\n"
                
            explanation += f"   - Decisão do sistema: Usar dataset {validation_results['best_choice'].upper()}\n"
        
        return explanation
        
    def _save_data(self, args: Dict[str, Any]) -> str:
        """Salva um conjunto de dados em um arquivo"""
        dataset_name = args.get('dataset', self.current_dataset)
        
        # Verificar se temos um dataset correspondente (pode ser original ou transformado)
        if dataset_name in self.datasets:
            df = self.datasets[dataset_name]
        elif dataset_name in self.transformed_datasets:
            df = self.transformed_datasets[dataset_name]
        else:
            return "Dataset não encontrado."
            
        if 'output_path' not in args:
            return "Por favor, especifique um caminho de arquivo para salvar os dados."
            
        output_path = args['output_path']
        
        try:
            # Determinar o formato do arquivo pela extensão
            if output_path.endswith('.csv'):
                df.to_csv(output_path, index=False)
            elif output_path.endswith('.xlsx'):
                df.to_excel(output_path, index=False)
            elif output_path.endswith('.parquet'):
                df.to_parquet(output_path, index=False)
            elif output_path.endswith('.json'):
                df.to_json(output_path, orient='records')
            else:
                return f"Formato de arquivo não suportado: {output_path}"
                
            return f"Dataset '{dataset_name}' salvo com sucesso em: {output_path}"
            
        except Exception as e:
            self.logger.error(f"Erro ao salvar dataset: {e}", exc_info=True)
            return f"Erro ao salvar o dataset: {str(e)}"
    
    def _create_pipeline(self, args: Dict[str, Any]) -> str:
        """Cria e salva um pipeline CAFE"""
        dataset_name = args.get('dataset', self.current_dataset)
        
        if not dataset_name or dataset_name not in self.datasets:
            return "Dataset não encontrado. Por favor, carregue um dataset primeiro."
            
        df = self.datasets[dataset_name]
        target_col = args.get('target_col')
        
        if 'output_path' not in args:
            return "Por favor, especifique um caminho para salvar o pipeline."
            
        output_path = args['output_path']
        
        try:
            # Verificar se já temos um pipeline para um dataset transformado
            transformed_name = f"{dataset_name}_transformed"
            if transformed_name in self.pipelines:
                pipeline = self.pipelines[transformed_name]
                pipeline.save(output_path)
                return f"Pipeline existente salvo com sucesso em: {output_path}"
            
            # Verificar se temos um explorer com recomendações
            if dataset_name in self.explorers:
                explorer = self.explorers[dataset_name]
                pipeline = explorer.create_optimal_pipeline()
                
                # Ajustar o pipeline aos dados
                pipeline.fit(df, target_col=target_col)
                
                # Salvar o pipeline
                pipeline.save(output_path)
                
                return f"Pipeline otimizado criado e salvo com sucesso em: {output_path}"
            
            # Caso contrário, criar um pipeline padrão
            pipeline = DataPipeline()
            pipeline.fit(df, target_col=target_col)
            pipeline.save(output_path)
            
            return f"Pipeline padrão criado e salvo com sucesso em: {output_path}"
            
        except Exception as e:
            self.logger.error(f"Erro ao criar pipeline: {e}", exc_info=True)
            return f"Erro ao criar o pipeline: {str(e)}"
    
    def _help(self, args: Dict[str, Any]) -> str:
        """Retorna informações de ajuda sobre os comandos disponíveis"""
        help_text = "CAFEAgent - Assistente para Engenharia Automática de Features\n\n"
        
        # Verificar se é solicitada ajuda para um comando específico
        command_pattern = r'(?:ajuda|help)\s+(?:sobre|para|com)?\s+([a-zA-Z]+)'
        command_match = re.search(command_pattern, str(args))
        
        if command_match:
            command = command_match.group(1).lower()
            
            if command in self.commands:
                help_text += f"Ajuda para o comando '{command}':\n\n"
                
                if command == 'carregar':
                    help_text += "Carrega um dataset de um arquivo.\n"
                    help_text += "Uso: carregar arquivo [caminho] dataset [nome]\n"
                    help_text += "Exemplo: carregar arquivo dados.csv dataset vendas\n"
                    help_text += "Formatos suportados: CSV, Excel, Parquet, JSON\n"
                
                elif command == 'explorar':
                    help_text += "Analisa e explora um dataset carregado.\n"
                    help_text += "Uso: explorar dataset [nome] colunas [col1,col2,...]\n"
                    help_text += "Exemplo: explorar dataset vendas\n"
                    help_text += "Você também pode especificar colunas específicas para analisar.\n"
                
                elif command == 'transformar':
                    help_text += "Transforma um dataset usando o CAFE.\n"
                    help_text += "Uso: transformar dataset [nome] target [coluna_alvo]\n"
                    help_text += "Exemplo: transformar dataset vendas target lucro\n"
                    help_text += "O dataset transformado será armazenado como '[nome]_transformed'.\n"
                
                elif command == 'validar':
                    help_text += "Valida as transformações aplicadas a um dataset.\n"
                    help_text += "Uso: validar dataset [nome]\n"
                    help_text += "Exemplo: validar dataset vendas\n"
                    help_text += "Mostra métricas de performance antes e depois das transformações.\n"
                
                elif command == 'recomendar':
                    help_text += "Recomenda transformações para um dataset.\n"
                    help_text += "Uso: recomendar dataset [nome] target [coluna_alvo]\n"
                    help_text += "Exemplo: recomendar dataset vendas target lucro\n"
                    help_text += "Analisa múltiplas transformações e recomenda a melhor configuração.\n"
                
                elif command == 'explicar':
                    help_text += "Explica as transformações aplicadas a um dataset.\n"
                    help_text += "Uso: explicar dataset [nome]\n"
                    help_text += "Exemplo: explicar dataset vendas\n"
                    help_text += "Fornece uma explicação detalhada das transformações aplicadas.\n"
                
                elif command == 'salvar':
                    help_text += "Salva um dataset em um arquivo.\n"
                    help_text += "Uso: salvar dataset [nome] como [caminho]\n"
                    help_text += "Exemplo: salvar dataset vendas_transformed como vendas_processados.csv\n"
                    help_text += "Formatos suportados: CSV, Excel, Parquet, JSON\n"
                
                elif command == 'pipeline':
                    help_text += "Cria e salva um pipeline CAFE.\n"
                    help_text += "Uso: pipeline dataset [nome] target [coluna_alvo] salvar [caminho]\n"
                    help_text += "Exemplo: pipeline dataset vendas target lucro salvar modelo_vendas\n"
                    help_text += "O pipeline pode ser carregado posteriormente para transformar novos dados.\n"
                
                elif command == 'ajuda':
                    help_text += "Mostra informações de ajuda sobre os comandos disponíveis.\n"
                    help_text += "Uso: ajuda ou ajuda sobre [comando]\n"
                    help_text += "Exemplo: ajuda sobre transformar\n"
                
                return help_text
            
            elif command in self.analysis_types:
                help_text += f"Ajuda para a análise '{command}':\n\n"
                
                if command == 'missing':
                    help_text += "Analisa valores ausentes no dataset.\n"
                    help_text += "Uso: analisar missing dataset [nome]\n"
                    help_text += "Exemplo: analisar missing dataset vendas\n"
                
                elif command == 'outliers':
                    help_text += "Detecta e analisa outliers no dataset.\n"
                    help_text += "Uso: analisar outliers dataset [nome] colunas [col1,col2,...]\n"
                    help_text += "Exemplo: analisar outliers dataset vendas colunas preco,quantidade\n"
                
                elif command == 'correlacao':
                    help_text += "Analisa correlações entre as variáveis numéricas.\n"
                    help_text += "Uso: analisar correlacao dataset [nome] limiar [valor]\n"
                    help_text += "Exemplo: analisar correlacao dataset vendas limiar 0.7\n"
                
                elif command == 'distribuicao':
                    help_text += "Analisa a distribuição das variáveis numéricas.\n"
                    help_text += "Uso: analisar distribuicao dataset [nome] colunas [col1,col2,...]\n"
                    help_text += "Exemplo: analisar distribuicao dataset vendas colunas preco\n"
                
                elif command == 'importancia':
                    help_text += "Analisa a importância das features para a variável alvo.\n"
                    help_text += "Uso: analisar importancia dataset [nome] target [coluna_alvo]\n"
                    help_text += "Exemplo: analisar importancia dataset vendas target lucro\n"
                
                elif command == 'categoricas':
                    help_text += "Analisa variáveis categóricas no dataset.\n"
                    help_text += "Uso: analisar categoricas dataset [nome]\n"
                    help_text += "Exemplo: analisar categoricas dataset vendas\n"
                
                elif command == 'temporais':
                    help_text += "Analisa colunas de data/hora no dataset.\n"
                    help_text += "Uso: analisar temporais dataset [nome]\n"
                    help_text += "Exemplo: analisar temporais dataset vendas\n"
                
                return help_text
            
            else:
                return f"Comando ou análise '{command}' não reconhecido."
        
        # Ajuda geral
        help_text += "Comandos disponíveis:\n\n"
        
        help_text += "1. Manipulação de Dados:\n"
        help_text += "   - carregar: Carrega um dataset de um arquivo\n"
        help_text += "   - explorar: Analisa e explora um dataset carregado\n"
        help_text += "   - salvar: Salva um dataset em um arquivo\n\n"
        
        help_text += "2. Transformação de Dados:\n"
        help_text += "   - transformar: Transforma um dataset usando o CAFE\n"
        help_text += "   - recomendar: Recomenda transformações para um dataset\n"
        help_text += "   - validar: Valida as transformações aplicadas\n"
        help_text += "   - explicar: Explica as transformações aplicadas\n"
        help_text += "   - pipeline: Cria e salva um pipeline CAFE\n\n"
        
        help_text += "3. Análises Específicas:\n"
        help_text += "   - analisar missing: Analisa valores ausentes\n"
        help_text += "   - analisar outliers: Detecta e analisa outliers\n"
        help_text += "   - analisar correlacao: Analisa correlações entre variáveis\n"
        help_text += "   - analisar distribuicao: Analisa distribuição de variáveis\n"
        help_text += "   - analisar importancia: Analisa importância das features\n"
        help_text += "   - analisar categoricas: Analisa variáveis categóricas\n"
        help_text += "   - analisar temporais: Analisa colunas de data/hora\n\n"
        
        help_text += "Para obter ajuda sobre um comando específico, digite 'ajuda sobre [comando]'.\n"
        
        return help_text
    
    #
    # Métodos para análises específicas
    #
    
    def _analyze_missing_values(self, args: Dict[str, Any]) -> str:
        """Analisa valores ausentes em um dataset"""
        dataset_name = args.get('dataset', self.current_dataset)
        
        if not dataset_name or dataset_name not in self.datasets:
            return "Dataset não encontrado. Por favor, carregue um dataset primeiro."
            
        df = self.datasets[dataset_name]
        
        # Calcular estatísticas de valores ausentes
        missing_counts = df.isnull().sum()
        missing_pct = (missing_counts / len(df)) * 100
        missing_df = pd.DataFrame({
            'Contagem': missing_counts,
            'Percentual': missing_pct
        })
        missing_df = missing_df[missing_df['Contagem'] > 0].sort_values('Percentual', ascending=False)
        
        if len(missing_df) == 0:
            return "Não foram encontrados valores ausentes no dataset."
            
        # Gerar relatório
        report = "Análise de Valores Ausentes:\n\n"
        report += f"Total de valores ausentes: {missing_counts.sum()} ({missing_pct.mean():.2f}% do dataset)\n\n"
        report += "Colunas com valores ausentes:\n"
        report += missing_df.to_string()
        
        # Recomendações
        report += "\n\nRecomendações para tratamento:\n"
        
        for col, pct in zip(missing_df.index, missing_df['Percentual']):
            if pct > 50:
                report += f"- Coluna '{col}': {pct:.2f}% de valores ausentes. Considere remover esta coluna.\n"
            elif pct > 20:
                report += f"- Coluna '{col}': {pct:.2f}% de valores ausentes. Recomendado usar técnicas avançadas de imputação como KNN.\n"
            else:
                dtype = df[col].dtype
                if pd.api.types.is_numeric_dtype(dtype):
                    report += f"- Coluna '{col}': {pct:.2f}% de valores ausentes. Recomendado substituir por média ou mediana.\n"
                else:
                    report += f"- Coluna '{col}': {pct:.2f}% de valores ausentes. Recomendado substituir pelo valor mais frequente.\n"
        
        return report
        
    def _analyze_outliers(self, args: Dict[str, Any]) -> str:
        """Detecta e analisa outliers em um dataset"""
        dataset_name = args.get('dataset', self.current_dataset)
        
        if not dataset_name or dataset_name not in self.datasets:
            return "Dataset não encontrado. Por favor, carregue um dataset primeiro."
            
        df = self.datasets[dataset_name]
        
        # Selecionar colunas para análise
        if 'columns' in args:
            columns = args['columns']
            numeric_cols = [col for col in columns if col in df.columns and pd.api.types.is_numeric_dtype(df[col].dtype)]
        else:
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        if not numeric_cols:
            return "Não foram encontradas colunas numéricas para análise de outliers."
            
        # Limitar o número de colunas para análise detalhada
        if len(numeric_cols) > 5:
            numeric_cols = numeric_cols[:5]
            note = "Nota: Limitando a análise às 5 primeiras colunas numéricas."
        else:
            note = ""
            
        # Gerar relatório
        report = f"Análise de Outliers:\n\n{note}\n\n"
        
        outliers_summary = {}
        
        for col in numeric_cols:
            # Calcular estatísticas
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            # Detectar outliers
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
            outliers_count = len(outliers)
            outliers_pct = (outliers_count / len(df)) * 100
            
            # Adicionar ao resumo
            outliers_summary[col] = {
                'count': outliers_count,
                'percentage': outliers_pct,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'min': df[col].min(),
                'max': df[col].max()
            }
            
            # Adicionar ao relatório
            report += f"Coluna '{col}':\n"
            report += f"- Total de outliers: {outliers_count} ({outliers_pct:.2f}% dos dados)\n"
            report += f"- Limite inferior: {lower_bound:.4f}, Limite superior: {upper_bound:.4f}\n"
            report += f"- Valor mínimo: {df[col].min():.4f}, Valor máximo: {df[col].max():.4f}\n"
            
            if outliers_count > 0:
                report += f"- Exemplos de outliers: {outliers.head(3).tolist()}\n"
                
            report += "\n"
            
        # Adicionar recomendações
        report += "Recomendações para tratamento de outliers:\n"
        
        for col, stats in outliers_summary.items():
            if stats['percentage'] > 10:
                report += f"- Coluna '{col}': Alto percentual de outliers ({stats['percentage']:.2f}%). "
                report += "Considere verificar se estes são erros de medição ou valores válidos.\n"
            elif stats['percentage'] > 0:
                report += f"- Coluna '{col}': {stats['percentage']:.2f}% de outliers. "
                report += "Recomendado usar métodos robustos ou transformações como log ou Box-Cox.\n"
                
        return report
        
    def _analyze_correlation(self, args: Dict[str, Any]) -> str:
        """Analisa correlações entre variáveis numéricas"""
        dataset_name = args.get('dataset', self.current_dataset)
        
        if not dataset_name or dataset_name not in self.datasets:
            return "Dataset não encontrado. Por favor, carregue um dataset primeiro."
            
        df = self.datasets[dataset_name]
        
        # Selecionar apenas colunas numéricas
        numeric_df = df.select_dtypes(include=['number'])
        
        if numeric_df.empty:
            return "Não foram encontradas colunas numéricas para análise de correlação."
            
        threshold = args.get('threshold', 0.7)
        
        # Calcular matriz de correlação
        corr_matrix = numeric_df.corr()
        
        # Identificar pares com alta correlação
        high_corr_pairs = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                col1 = corr_matrix.columns[i]
                col2 = corr_matrix.columns[j]
                corr_value = corr_matrix.iloc[i, j]
                
                if abs(corr_value) >= threshold:
                    high_corr_pairs.append((col1, col2, corr_value))
        
        # Gerar relatório
        report = f"Análise de Correlação (limiar = {threshold}):\n\n"
        
        if not high_corr_pairs:
            report += f"Não foram encontrados pares de features com correlação acima de {threshold}.\n"
        else:
            report += f"Foram encontrados {len(high_corr_pairs)} pares de features com alta correlação:\n\n"
            
            # Ordenar por correlação absoluta (maior para menor)
            high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
            
            for col1, col2, corr in high_corr_pairs:
                report += f"- '{col1}' e '{col2}': {corr:.4f}\n"
                
            # Recomendações
            report += "\nRecomendações:\n"
            report += "- Features altamente correlacionadas podem ser redundantes e aumentar o risco de multicolinearidade.\n"
            report += "- Considere remover uma das features de cada par com alta correlação.\n"
            report += f"- O CAFE pode fazer isso automaticamente com o parâmetro 'correlation_threshold' ajustado para {threshold}.\n"
        
        return report
        
    def _analyze_distribution(self, args: Dict[str, Any]) -> str:
        """Analisa a distribuição das variáveis numéricas"""
        dataset_name = args.get('dataset', self.current_dataset)
        
        if not dataset_name or dataset_name not in self.datasets:
            return "Dataset não encontrado. Por favor, carregue um dataset primeiro."
            
        df = self.datasets[dataset_name]
        
        # Selecionar colunas para análise
        if 'columns' in args:
            columns = args['columns']
            numeric_cols = [col for col in columns if col in df.columns and pd.api.types.is_numeric_dtype(df[col].dtype)]
        else:
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()[:5]  # Limitar a 5 colunas
        
        if not numeric_cols:
            return "Não foram encontradas colunas numéricas para análise de distribuição."
            
        # Gerar relatório
        report = "Análise de Distribuição de Variáveis Numéricas:\n\n"
        
        for col in numeric_cols:
            # Calcular estatísticas básicas
            stats = df[col].describe()
            
            # Testar normalidade (se tiver mais de 20 amostras)
            from scipy import stats as scipy_stats
            
            if len(df) >= 20:
                _, shapiro_p = scipy_stats.shapiro(df[col].dropna())
                normal = shapiro_p > 0.05
            else:
                normal = "Não avaliado (amostra pequena)"
                
            # Calcular assimetria (skewness) e curtose
            skewness = df[col].skew()
            kurtosis = df[col].kurtosis()
            
            # Adicionar ao relatório
            report += f"Coluna '{col}':\n"
            report += f"- Média: {stats['mean']:.4f}\n"
            report += f"- Mediana: {stats['50%']:.4f}\n"
            report += f"- Desvio padrão: {stats['std']:.4f}\n"
            report += f"- Mínimo: {stats['min']:.4f}, Máximo: {stats['max']:.4f}\n"
            report += f"- Assimetria: {skewness:.4f} "
            
            if abs(skewness) < 0.5:
                report += "(aproximadamente simétrica)\n"
            elif skewness < 0:
                report += "(assimetria negativa - cauda à esquerda)\n"
            else:
                report += "(assimetria positiva - cauda à direita)\n"
                
            report += f"- Curtose: {kurtosis:.4f} "
            
            if kurtosis < -0.5:
                report += "(platicúrtica - mais plana que a normal)\n"
            elif kurtosis > 0.5:
                report += "(leptocúrtica - mais pontiaguda que a normal)\n"
            else:
                report += "(mesocúrtica - semelhante à normal)\n"
                
            report += f"- Distribuição normal: {'Sim' if normal == True else 'Não' if normal == False else normal}\n\n"
            
            # Adicionar recomendações específicas
            if abs(skewness) > 1 or not isinstance(normal, str) and not normal:
                report += "  Recomendações para normalização:\n"
                
                if skewness > 1:
                    report += "  - Considere transformação logarítmica (log)\n"
                elif skewness < -1:
                    report += "  - Considere transformação exponencial\n"
                    
                report += "  - O CAFE pode aplicar transformações 'power' como Box-Cox ou Yeo-Johnson para aproximar à distribuição normal\n"
                
            report += "\n"
        
        return report
        
    def _analyze_feature_importance(self, args: Dict[str, Any]) -> str:
        """Analisa a importância das features para a variável alvo"""
        dataset_name = args.get('dataset', self.current_dataset)
        
        if not dataset_name or dataset_name not in self.datasets:
            return "Dataset não encontrado. Por favor, carregue um dataset primeiro."
            
        df = self.datasets[dataset_name]
        
        # Obter coluna alvo
        target_col = args.get('target_col')
        
        if not target_col:
            return "Por favor, especifique a coluna alvo (target) para análise de importância."
            
        if target_col not in df.columns:
            return f"Coluna '{target_col}' não encontrada no dataset."
            
        try:
            # Separar features e target
            X = df.drop(columns=[target_col])
            y = df[target_col]
            
            # Verificar tipo de problema (classificação ou regressão)
            task = 'classification' if pd.api.types.is_categorical_dtype(y) or y.nunique() < 10 else 'regression'
            
            # Calcular importância com Random Forest
            if task == 'classification':
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                from sklearn.ensemble import RandomForestRegressor
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                
            # Preparar os dados
            from sklearn.preprocessing import OrdinalEncoder
            
            # Tratar colunas categóricas
            cat_cols = X.select_dtypes(include=['object', 'category']).columns
            if not cat_cols.empty:
                X_encoded = X.copy()
                encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                X_encoded[cat_cols] = encoder.fit_transform(X[cat_cols])
            else:
                X_encoded = X
                
            # Ajustar modelo
            model.fit(X_encoded, y)
            
            # Calcular importância
            importances = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Gerar relatório
            report = f"Análise de Importância de Features para '{target_col}':\n\n"
            report += f"Tipo de problema: {'Classificação' if task == 'classification' else 'Regressão'}\n\n"
            report += "Features ordenadas por importância:\n"
            
            for _, row in importances.iterrows():
                feature = row['feature']
                importance = row['importance']
                percentage = importance * 100
                report += f"- {feature}: {percentage:.2f}%\n"
                
            # Adicionar recomendações
            report += "\nRecomendações:\n"
            report += "- Features com baixa importância podem ser candidatas à remoção para simplificar o modelo.\n"
            report += "- O CAFE pode selecionar automaticamente as melhores features usando métodos como:\n"
            report += "  * SelectKBest: Seleciona as k melhores features\n"
            report += "  * SelectFromModel: Utiliza um modelo para selecionar features relevantes\n"
            report += "  * SelectPercentile: Seleciona um percentual das melhores features\n"
            
            return report
            
        except Exception as e:
            self.logger.error(f"Erro ao analisar importância de features: {e}", exc_info=True)
            return f"Erro ao analisar importância de features: {str(e)}"
        
    def _analyze_categorical(self, args: Dict[str, Any]) -> str:
        """Analisa variáveis categóricas no dataset"""
        dataset_name = args.get('dataset', self.current_dataset)
        
        if not dataset_name or dataset_name not in self.datasets:
            return "Dataset não encontrado. Por favor, carregue um dataset primeiro."
            
        df = self.datasets[dataset_name]
        
        # Selecionar colunas categóricas
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        
        if cat_cols.empty:
            return "Não foram encontradas colunas categóricas no dataset."
            
        # Gerar relatório
        report = "Análise de Variáveis Categóricas:\n\n"
        report += f"Total de colunas categóricas: {len(cat_cols)}\n\n"
        
        for col in cat_cols:
            # Calcular estatísticas
            value_counts = df[col].value_counts()
            unique_count = len(value_counts)
            missing_count = df[col].isnull().sum()
            missing_pct = missing_count / len(df) * 100
            
            # Adicionar ao relatório
            report += f"Coluna '{col}':\n"
            report += f"- Valores únicos: {unique_count}\n"
            report += f"- Valores ausentes: {missing_count} ({missing_pct:.2f}%)\n"
            
            # Mostrar distribuição para colunas com cardinalidade baixa ou média
            if unique_count <= 20:
                report += "- Distribuição:\n"
                for value, count in value_counts.head(10).items():
                    percentage = count / len(df) * 100
                    report += f"  * {value}: {count} ({percentage:.2f}%)\n"
                    
                if unique_count > 10:
                    report += f"  * ... e {unique_count - 10} outros valores\n"
            else:
                report += f"- Alta cardinalidade ({unique_count} valores únicos)\n"
                report += "- Valores mais frequentes:\n"
                for value, count in value_counts.head(5).items():
                    percentage = count / len(df) * 100
                    report += f"  * {value}: {count} ({percentage:.2f}%)\n"
            
            report += "\n"
            
        # Adicionar recomendações
        report += "Recomendações para codificação de variáveis categóricas:\n"
        
        for col in cat_cols:
            unique_count = df[col].nunique()
            
            if unique_count <= 2:
                report += f"- Coluna '{col}': Use Label Encoding (categorical_strategy='label' ou 'binary')\n"
            elif unique_count <= 10:
                report += f"- Coluna '{col}': Use One-Hot Encoding (categorical_strategy='onehot')\n"
            else:
                report += f"- Coluna '{col}': Alta cardinalidade. Considere Target Encoding (categorical_strategy='target')\n"
                report += f"  ou redução de cardinalidade agrupando categorias menos frequentes\n"
        
        return report
        
    def _analyze_datetime(self, args: Dict[str, Any]) -> str:
        """Analisa colunas de data/hora no dataset"""
        dataset_name = args.get('dataset', self.current_dataset)
        
        if not dataset_name or dataset_name not in self.datasets:
            return "Dataset não encontrado. Por favor, carregue um dataset primeiro."
            
        df = self.datasets[dataset_name]
        
        # Identificar colunas de data/hora
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        # Verificar colunas que podem ser datas em formato string
        for col in df.select_dtypes(include=['object']).columns:
            try:
                # Tentar converter para datetime
                pd.to_datetime(df[col], errors='raise')
                datetime_cols.append(col)
            except:
                pass
        
        if not datetime_cols:
            return "Não foram encontradas colunas de data/hora no dataset."
            
        # Gerar relatório
        report = "Análise de Colunas de Data/Hora:\n\n"
        report += f"Total de colunas de data/hora: {len(datetime_cols)}\n\n"
        
        for col in datetime_cols:
            # Converter para datetime se não for
            if df[col].dtype != 'datetime64[ns]':
                try:
                    datetime_series = pd.to_datetime(df[col])
                except:
                    report += f"Coluna '{col}' foi identificada como possível data/hora, mas não foi possível converter.\n\n"
                    continue
            else:
                datetime_series = df[col]
                
            # Calcular estatísticas
            min_date = datetime_series.min()
            max_date = datetime_series.max()
            range_days = (max_date - min_date).days
            missing_count = datetime_series.isnull().sum()
            missing_pct = missing_count / len(df) * 100
            
            # Adicionar ao relatório
            report += f"Coluna '{col}':\n"
            report += f"- Tipo de dados: {df[col].dtype}\n"
            report += f"- Data mínima: {min_date}\n"
            report += f"- Data máxima: {max_date}\n"
            report += f"- Intervalo: {range_days} dias\n"
            report += f"- Valores ausentes: {missing_count} ({missing_pct:.2f}%)\n\n"
            
            # Informações adicionais se for coluna datetime
            if df[col].dtype == 'datetime64[ns]':
                # Verificar distribuição por ano (se o intervalo for grande)
                if range_days > 365:
                    year_counts = df[col].dt.year.value_counts().sort_index()
                    report += "- Distribuição por ano:\n"
                    for year, count in year_counts.items():
                        percentage = count / len(df) * 100
                        report += f"  * {year}: {count} ({percentage:.2f}%)\n"
                    report += "\n"
                
                # Verificar distribuição por dia da semana
                weekday_counts = df[col].dt.weekday.value_counts().sort_index()
                weekday_names = ['Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta', 'Sábado', 'Domingo']
                report += "- Distribuição por dia da semana:\n"
                for weekday, count in weekday_counts.items():
                    percentage = count / len(df) * 100
                    report += f"  * {weekday_names[weekday]}: {count} ({percentage:.2f}%)\n"
                report += "\n"
        
        # Adicionar recomendações
        report += "Recomendações para processamento de datas:\n"
        report += "- O CAFE pode extrair características úteis de colunas de data/hora usando o parâmetro 'datetime_features'\n"
        report += "- Você pode escolher extrair, por exemplo: ano, mês, dia, dia da semana, trimestre, etc.\n"
        report += "- Recomendação de configuração:\n"
        report += "  datetime_features=['year', 'month', 'day', 'weekday', 'is_weekend', 'quarter']\n"
        
        return report


# Interface para uso do CAFEAgent
class CAFEAssistant:
    """
    Assistente de IA Generativa para o CAFE.
    Fornece uma interface amigável para interagir com o CAFEAgent.
    """
    
    def __init__(self, verbose: bool = True):
        """
        Inicializa o Assistente CAFE.
        
        Args:
            verbose: Se True, exibe mensagens detalhadas durante a execução
        """
        self.agent = CAFEAgent(verbose=verbose)
        self.history = []
        
    def process_message(self, message: str) -> str:
        """
        Processa uma mensagem em linguagem natural e retorna a resposta.
        
        Args:
            message: Mensagem em linguagem natural
            
        Returns:
            Resposta à mensagem
        """
        # Registrar a mensagem no histórico
        self.history.append({"role": "user", "content": message})
        
        # Processar a mensagem com o CAFEAgent
        response = self.agent.process_command(message)
        
        # Registrar a resposta no histórico
        self.history.append({"role": "assistant", "content": response})
        
        return response
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """
        Retorna o histórico da conversa.
        
        Returns:
            Lista de mensagens no formato {"role": "user"|"assistant", "content": "mensagem"}
        """
        return self.history
    
    def clear_history(self) -> None:
        """Limpa o histórico da conversa."""
        self.history = []

# Exemplos de uso
def cafe_assistant_demo():
    """Demonstração do assistente CAFE"""
    assistant = CAFEAssistant()
    
    print("Bem-vindo ao CAFE Assistant!")
    print("Digite 'sair' ou 'exit' para encerrar.\n")
    
    while True:
        message = input("Você: ")
        
        if message.lower() in ['sair', 'exit', 'quit']:
            print("\nObrigado por usar o CAFE Assistant!")
            break
            
        response = assistant.process_message(message)
        print(f"\nCAFE Assistant: {response}\n")

# Se executado diretamente
if __name__ == "__main__":
    cafe_assistant_demo()