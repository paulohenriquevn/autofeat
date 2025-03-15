from typing import Dict, List, Optional, Union, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

from cafe.data_pipeline import DataPipeline

class EnhancedDataPipeline(DataPipeline):
    def __init__(self, preprocessor_config: Optional[Dict] = None, 
                feature_engineer_config: Optional[Dict] = None,
                validator_config: Optional[Dict] = None,
                auto_validate: bool = True):
        """
        Inicializa o EnhancedDataPipeline com todas as funcionalidades do DataPipeline padrão.
        """
        super().__init__(
            preprocessor_config=preprocessor_config,
            feature_engineer_config=feature_engineer_config,
            validator_config=validator_config,
            auto_validate=auto_validate
        )
    
    def get_missing_values_report(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Gera um relatório detalhado sobre valores ausentes no dataset.
        
        Args:
            df: DataFrame a ser analisado
            
        Returns:
            DataFrame com informações sobre valores ausentes
        """
        if df.empty:
            return pd.DataFrame(columns=['coluna', 'valores_ausentes', 'porcentagem', 'recomendacao'])
        
        # Calcular estatísticas de valores ausentes
        missing_count = df.isnull().sum()
        missing_percent = (missing_count / len(df)) * 100
        
        # Criar DataFrame com estatísticas
        report = pd.DataFrame({
            'coluna': missing_count.index,
            'valores_ausentes': missing_count.values,
            'porcentagem': missing_percent.values.round(2)
        })
        
        # Ordenar pelo número de valores ausentes (decrescente)
        report = report.sort_values('valores_ausentes', ascending=False)
        
        # Adicionar recomendações baseadas na estratégia configurada
        strategy = self.preprocessor.config.get('missing_values_strategy', 'median')
        
        # Gerar recomendações personalizadas baseadas na porcentagem de valores ausentes
        def get_recommendation(row):
            col = row['coluna']
            pct = row['porcentagem']
            
            # Se não há valores ausentes
            if pct == 0:
                return "Sem valores ausentes"
            
            # Verificar o tipo de dados da coluna
            if col in df.columns:
                dtype = df[col].dtype
                is_numeric = pd.api.types.is_numeric_dtype(dtype)
                is_categorical = pd.api.types.is_categorical_dtype(dtype) or pd.api.types.is_object_dtype(dtype)
                
                # Recomendações baseadas na porcentagem e tipo de dados
                if pct > 50:
                    return f"Alta porcentagem de ausência. Considere remover esta coluna ou uma imputação avançada."
                elif pct > 20:
                    if is_numeric:
                        return f"Imputação usando {strategy} (configurado). Considere KNN para melhor precisão."
                    elif is_categorical:
                        return "Imputação usando o valor mais frequente ou criando uma categoria 'desconhecido'."
                else:
                    if is_numeric:
                        return f"Imputação usando {strategy} (configurado)."
                    elif is_categorical:
                        return "Imputação usando o valor mais frequente."
            
            return f"Imputação usando {strategy} (configurado)."
        
        report['recomendacao'] = report.apply(get_recommendation, axis=1)
        
        # Filtrar apenas colunas com valores ausentes
        report_with_missing = report[report['valores_ausentes'] > 0]
        
        if report_with_missing.empty:
            self.logger.info("Não foram encontrados valores ausentes no dataset.")
            
        return report_with_missing
    
    def visualize_missing_values(self, df: pd.DataFrame, figsize=(12, 8), top_n=20):
        """
        Visualiza os valores ausentes no dataset.
        
        Args:
            df: DataFrame a ser analisado
            figsize: Tamanho da figura
            top_n: Número máximo de colunas a serem mostradas
            
        Returns:
            Figura matplotlib com visualização dos valores ausentes
        """
        # Obter relatório de valores ausentes
        missing_report = self.get_missing_values_report(df)
        
        if missing_report.empty:
            print("Não há valores ausentes no dataset.")
            return None
        
        # Limitar ao top_n colunas com mais valores ausentes
        missing_report = missing_report.head(top_n)
        
        # Criar figura
        fig, ax = plt.subplots(figsize=figsize)
        
        # Gráfico de barras para valores ausentes
        bars = ax.barh(missing_report['coluna'], missing_report['porcentagem'], color='skyblue')
        
        # Adicionar valores nas barras
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 1, bar.get_y() + bar.get_height()/2, f'{width:.1f}%', 
                   ha='left', va='center')
        
        ax.set_xlabel('Porcentagem de Valores Ausentes (%)')
        ax.set_ylabel('Coluna')
        ax.set_title('Valores Ausentes por Coluna')
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        return fig
    
    def get_outliers_report(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Gera um relatório detalhado sobre outliers no dataset.
        
        Args:
            df: DataFrame a ser analisado
            
        Returns:
            DataFrame com informações sobre outliers
        """
        if df.empty:
            return pd.DataFrame(columns=['coluna', 'num_outliers', 'porcentagem', 'limite_inferior', 'limite_superior', 'recomendacao'])
        
        # Selecionar apenas colunas numéricas
        numeric_df = df.select_dtypes(include=['number'])
        if numeric_df.empty:
            self.logger.warning("Não foram encontradas colunas numéricas para análise de outliers.")
            return pd.DataFrame(columns=['coluna', 'num_outliers', 'porcentagem', 'limite_inferior', 'limite_superior', 'recomendacao'])
        
        # Método configurado para detecção de outliers
        outlier_method = self.preprocessor.config.get('outlier_method', 'iqr')
        
        # Preparar resultados
        results = []
        
        for col in numeric_df.columns:
            series = df[col].dropna()
            
            # Ignorar colunas binárias (0/1) ou com poucos valores únicos
            if series.nunique() <= 2 or (series.nunique() / len(series) < 0.01 and series.nunique() <= 10):
                continue
                
            # Detectar outliers com base no método configurado
            if outlier_method == 'zscore':
                from scipy import stats
                z_scores = np.abs(stats.zscore(series, nan_policy='omit'))
                outliers = series[z_scores > 3]
                lower_bound = series.mean() - 3 * series.std()
                upper_bound = series.mean() + 3 * series.std()
            elif outlier_method == 'iqr':
                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                outliers = series[(series < lower_bound) | (series > upper_bound)]
            else:  # isolation_forest ou outro método
                # Usar IQR como fallback
                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                outliers = series[(series < lower_bound) | (series > upper_bound)]
            
            # Calcular estatísticas
            outlier_count = len(outliers)
            outlier_percent = (outlier_count / len(series)) * 100
            
            # Gerar recomendação
            if outlier_percent > 10:
                recommendation = "Alta presença de outliers. Considere transformação logarítmica ou remoção seletiva."
            elif outlier_percent > 5:
                recommendation = "Presença moderada de outliers. Considere usar RobustScaler ou Winsorization."
            elif outlier_percent > 0:
                recommendation = "Baixa presença de outliers. O método padrão de tratamento deve ser suficiente."
            else:
                recommendation = "Sem outliers detectados."
            
            # Adicionar à lista de resultados
            results.append({
                'coluna': col,
                'num_outliers': outlier_count,
                'porcentagem': round(outlier_percent, 2),
                'limite_inferior': round(float(lower_bound), 4),
                'limite_superior': round(float(upper_bound), 4),
                'min': round(float(series.min()), 4),
                'max': round(float(series.max()), 4),
                'recomendacao': recommendation
            })
        
        # Criar DataFrame com resultados
        outliers_report = pd.DataFrame(results)
        
        # Ordenar por número de outliers (decrescente)
        if not outliers_report.empty:
            outliers_report = outliers_report.sort_values('num_outliers', ascending=False)
        
        return outliers_report
    
    def visualize_outliers(self, df: pd.DataFrame, columns=None, figsize=(15, 10), max_cols=5):
        """
        Visualiza outliers nas colunas numéricas do dataset.
        
        Args:
            df: DataFrame a ser analisado
            columns: Lista específica de colunas para visualizar (opcional)
            figsize: Tamanho da figura
            max_cols: Número máximo de colunas a serem visualizadas
            
        Returns:
            Figura matplotlib com visualização de outliers
        """
        # Obter relatório de outliers
        outliers_report = self.get_outliers_report(df)
        
        if outliers_report.empty:
            print("Não foram detectados outliers significativos no dataset.")
            return None
        
        # Selecionar colunas específicas ou as top N colunas com mais outliers
        if columns is not None:
            # Filtrar para incluir apenas colunas existentes no relatório
            cols_to_plot = [col for col in columns if col in outliers_report['coluna'].values]
            if not cols_to_plot:
                print("Nenhuma das colunas especificadas tem outliers significativos.")
                return None
        else:
            # Selecionar as top N colunas com mais outliers
            cols_to_plot = outliers_report.head(max_cols)['coluna'].tolist()
        
        # Número de colunas para visualizar
        n_cols = len(cols_to_plot)
        
        if n_cols == 0:
            print("Não há colunas com outliers para visualizar.")
            return None
        
        # Criar figura
        fig, axes = plt.subplots(n_cols, 1, figsize=figsize)
        if n_cols == 1:
            axes = [axes]
        
        # Para cada coluna, criar boxplot e histograma
        for i, col in enumerate(cols_to_plot):
            col_data = df[col].dropna()
            
            # Obter limites de outliers
            row = outliers_report[outliers_report['coluna'] == col].iloc[0]
            lower_bound = row['limite_inferior']
            upper_bound = row['limite_superior']
            
            # Criar boxplot na mesma figura
            ax = axes[i]
            sns.boxplot(x=col_data, ax=ax, color='lightblue')
            
            # Adicionar marcação dos limites
            ax.axvline(x=lower_bound, color='red', linestyle='--', alpha=0.7)
            ax.axvline(x=upper_bound, color='red', linestyle='--', alpha=0.7)
            
            # Adicionar título e informações
            ax.set_title(f"{col}: {row['num_outliers']} outliers ({row['porcentagem']}%)")
            ax.text(0.02, 0.85, f"Limite inferior: {lower_bound:.2f}", transform=ax.transAxes)
            ax.text(0.02, 0.75, f"Limite superior: {upper_bound:.2f}", transform=ax.transAxes)
        
        plt.tight_layout()
        
        return fig
    
    def get_feature_importance_report(self, df: pd.DataFrame, target_col: Optional[str] = None) -> pd.DataFrame:
        """
        Gera um relatório detalhado sobre a importância das features para a variável alvo.
        
        Args:
            df: DataFrame com os dados
            target_col: Nome da coluna alvo (opcional)
            
        Returns:
            DataFrame com informações sobre importância das features
        """
        target = target_col or self.target_col
        
        if target is None:
            self.logger.error("É necessário fornecer a coluna alvo para calcular a importância das features")
            return pd.DataFrame(columns=['feature', 'importance', 'normalized_importance', 'categoria'])
                
        if target not in df.columns:
            self.logger.error(f"Coluna alvo '{target}' não encontrada no DataFrame")
            return pd.DataFrame(columns=['feature', 'importance', 'normalized_importance', 'categoria'])
        
        # Obter importância usando o validador
        try:
            # Separar features e target
            X = df.drop(columns=[target])
            y = df[target]
            
            # Agora passamos X e y para o método get_feature_importance
            importance = self.validator.get_feature_importance(X, y)
            
            # Adicionar importância normalizada (porcentagem)
            importance['normalized_importance'] = (importance['importance'] / importance['importance'].sum() * 100).round(2)
            
            # Categorizar features por importância
            def categorize_importance(importance):
                if importance >= 75:
                    return "Muito Alta"
                elif importance >= 50:
                    return "Alta"
                elif importance >= 25:
                    return "Média"
                elif importance >= 10:
                    return "Baixa"
                else:
                    return "Muito Baixa"
            
            importance['categoria'] = importance['normalized_importance'].apply(categorize_importance)
            
            return importance
        except Exception as e:
            self.logger.error(f"Erro ao calcular importância de features: {e}")
            return pd.DataFrame(columns=['feature', 'importance', 'normalized_importance', 'categoria'])
    
    def visualize_feature_importance(self, df: pd.DataFrame, target_col: Optional[str] = None, figsize=(10, 8), top_n=15):
        """
        Visualiza a importância das features para a variável alvo.
        
        Args:
            df: DataFrame com os dados
            target_col: Nome da coluna alvo (opcional)
            figsize: Tamanho da figura
            top_n: Número máximo de features a serem mostradas
            
        Returns:
            Figura matplotlib com visualização da importância das features
        """
        # Obter relatório de importância de features
        importance_report = self.get_feature_importance_report(df, target_col)
        
        if importance_report.empty:
            print("Não foi possível calcular a importância das features.")
            return None
        
        # Limitar ao top_n features mais importantes
        importance_report = importance_report.head(top_n)
        
        # Criar figura
        fig, ax = plt.subplots(figsize=figsize)
        
        # Gráfico de barras para importância de features
        # Criar um mapa de cores baseado na categoria
        color_map = {
            'Muito Alta': '#1a53ff',
            'Alta': '#4d79ff',
            'Média': '#80a0ff',
            'Baixa': '#b3c6ff',
            'Muito Baixa': '#e6ecff'
        }
        
        colors = [color_map[cat] for cat in importance_report['categoria']]
        
        # Criar gráfico de barras horizontais
        bars = ax.barh(importance_report['feature'], importance_report['normalized_importance'], color=colors)
        
        # Adicionar valores nas barras
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.5, bar.get_y() + bar.get_height()/2, f'{width:.1f}%', 
                   ha='left', va='center')
        
        ax.set_xlabel('Importância Normalizada (%)')
        ax.set_ylabel('Feature')
        ax.set_title('Importância das Features para a Variável Alvo')
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        
        # Adicionar legenda para categorias
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color, label=cat) for cat, color in color_map.items()]
        ax.legend(handles=legend_elements, title='Categoria de Importância', loc='lower right')
        
        plt.tight_layout()
        
        return fig
    
    def get_transformations_report(self) -> Dict:
        """
        Gera um relatório das transformações aplicadas pelo pipeline.
        
        Returns:
            Dicionário com informações sobre as transformações aplicadas
        """
        if not self.fitted:
            return {"status": "Pipeline não ajustado. Execute fit() ou fit_transform() primeiro."}
        
        # Obter informações sobre transformações do preprocessador
        try:
            preprocessor_transformations = self.preprocessor.get_transformer_description()
        except Exception as e:
            preprocessor_transformations = {"error": f"Erro ao obter descrição do preprocessador: {e}"}
        
        # Informações sobre feature engineering
        feature_engineering_info = {}
        if hasattr(self.feature_engineer, 'config'):
            feature_engineering_info = self.feature_engineer.config.copy()
        
        # Resultados da validação
        validation_results = self.get_validation_results() or {}
        
        # Estatísticas de transformação
        transformation_stats = {
            "dimensoes_originais": None,
            "dimensoes_transformadas": None,
            "reducao_features": None,
            "reducao_features_pct": None,
            "ganho_performance": None,
            "ganho_performance_pct": None,
            "decisao_final": None
        }
        
        if validation_results:
            # Usar informações do relatório de validação, ou valores padrão se não disponíveis
            if 'feature_reduction' in validation_results:
                # Tentar recuperar número de features original do atributo do feature_engineer
                num_original_features = 0
                if hasattr(self.feature_engineer, 'input_feature_names'):
                    num_original_features = len(self.feature_engineer.input_feature_names)
                
                # Se isso não funcionar, use um valor padrão de 30 (comum em datasets)
                if num_original_features == 0:
                    num_original_features = validation_results.get('original_n_features', 30)
                    
                # Calcular o número de features transformadas
                transformed_features = int(num_original_features * (1 - validation_results['feature_reduction']))
                
                transformation_stats.update({
                    "dimensoes_originais": (0, num_original_features),  # (amostras, features)
                    "dimensoes_transformadas": (0, transformed_features),  # (amostras, features)
                    "reducao_features": num_original_features - transformed_features,
                    "reducao_features_pct": round(validation_results['feature_reduction'] * 100, 2),
                    "ganho_performance": round(validation_results['performance_diff'], 4),
                    "ganho_performance_pct": round(validation_results['performance_diff_pct'], 2),
                    "decisao_final": validation_results['best_choice'].upper()
                })
        
        # Juntar todas as informações no relatório
        report = {
            "preprocessamento": preprocessor_transformations,
            "engenharia_features": feature_engineering_info,
            "estatisticas": transformation_stats,
            "validacao": validation_results
        }
        
        return report
    
    def visualize_transformations(self, figsize=(14, 10)):
        """
        Visualiza as estatísticas das transformações aplicadas pelo pipeline.
        
        Args:
            figsize: Tamanho da figura
            
        Returns:
            Figura matplotlib com visualização das transformações
        """
        if not self.fitted:
            print("Pipeline não ajustado. Execute fit() ou fit_transform() primeiro.")
            return None
        
        # Obter resultados da validação
        validation_results = self.get_validation_results()
        
        if not validation_results:
            print("Não há resultados de validação disponíveis.")
            return None
        
        # Criar figura
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # 1. Gráfico de comparação de performance
        ax1 = axes[0, 0]
        performance = [
            validation_results['performance_original'],
            validation_results['performance_transformed']
        ]
        colors = ['blue', 'green'] if validation_results['performance_diff'] >= 0 else ['blue', 'red']
        
        bars = ax1.bar(['Original', 'Transformado'], performance, color=colors)
        
        # Adicionar valores nas barras
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                     f'{height:.4f}', ha='center', va='bottom')
        
        ax1.set_title('Comparação de Performance')
        ax1.set_ylabel('Performance (Métrica)')
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 2. Comparação de número de features
        ax2 = axes[0, 1]
        feature_reduction = validation_results.get('feature_reduction', 0) * 100
        original_features = validation_results.get('original_n_features', 0)
        transformed_features = int(original_features * (1 - validation_results['feature_reduction']))
        
        # Criar gráfico de barras para número de features antes/depois
        if feature_reduction >= 0:
            feature_bars = ax2.bar(['Original', 'Transformado'], [original_features, transformed_features], color=['blue', 'green'])
            title_suffix = f"Redução de {feature_reduction:.1f}%"
        else:
            feature_reduction = abs(feature_reduction)
            feature_bars = ax2.bar(['Original', 'Transformado'], [original_features, transformed_features], color=['blue', 'orange'])
            title_suffix = f"Aumento de {feature_reduction:.1f}%"
            
        ax2.set_title(f'Comparação de Features - {title_suffix}')
        ax2.set_ylabel('Número de Features')
        ax2.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Adicionar valores nas barras
        for bar in feature_bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2, height + 0.5,
                     f"{int(height)}", ha='center', va='bottom')
        
        # 3. Performance por fold
        ax3 = axes[1, 0]
        folds = list(range(1, len(validation_results['scores_original'])+1))
        
        ax3.plot(folds, validation_results['scores_original'], 'o-', label='Original', color='blue')
        ax3.plot(folds, validation_results['scores_transformed'], 'o-', label='Transformado', 
                 color='green' if validation_results['performance_diff'] >= 0 else 'red')
        
        ax3.set_title('Performance por Fold de Validação Cruzada')
        ax3.set_xlabel('Fold')
        ax3.set_ylabel('Performance')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Texto com resumo e decisão
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        performance_diff = validation_results['performance_diff']
        performance_diff_pct = validation_results['performance_diff_pct']
        best_choice = validation_results['best_choice']
        
        text = f"""
        RESUMO DA VALIDAÇÃO

        Performance:
        - Original:     {validation_results['performance_original']:.4f}
        - Transformado: {validation_results['performance_transformed']:.4f}
        - Diferença:    {performance_diff:.4f} ({performance_diff_pct:.2f}%)

        Features:
        - Original:     {original_features}
        - Transformado: {transformed_features}
        - Redução:      {abs(feature_reduction):.1f}%

        DECISÃO: Usar dados {best_choice.upper()}

        Validação:
        - Máxima queda permitida: {validation_results.get('max_performance_drop', 0.05)*100:.1f}%
        - Folds de validação: {len(validation_results['scores_original'])}
        """
        
        ax4.text(0.1, 0.9, text, fontsize=12, va='top', family='monospace')
        
        plt.tight_layout()
        
        return fig
    
    def generate_report(self, df: pd.DataFrame, target_col: Optional[str] = None, 
                        include_visualizations: bool = False, output_file: Optional[str] = None) -> str:
        """
        Gera um relatório completo em formato de texto com todas as análises.
        
        Args:
            df: DataFrame com os dados
            target_col: Nome da coluna alvo (opcional)
            include_visualizations: Se True, inclui caminhos para visualizações salvas
            output_file: Caminho para salvar o relatório em formato de texto
            
        Returns:
            String com o relatório completo
        """
        target = target_col or self.target_col
        
        # Inicializar buffer para o relatório
        report_buffer = StringIO()
        report_buffer.write("=" * 80 + "\n")
        report_buffer.write(" RELATÓRIO DE ANÁLISE CAFE (Component Automated Feature Engineer) ".center(80) + "\n")
        report_buffer.write("=" * 80 + "\n\n")
        
        # Informações gerais
        report_buffer.write("INFORMAÇÕES GERAIS\n")
        report_buffer.write("-" * 80 + "\n")
        report_buffer.write(f"Dimensões do dataset: {df.shape[0]} amostras x {df.shape[1]} colunas\n")
        report_buffer.write(f"Variável alvo: {target if target else 'Não especificada'}\n")
        report_buffer.write(f"Status do pipeline: {'Ajustado' if self.fitted else 'Não ajustado'}\n\n")
        
        # Informações sobre valores ausentes
        report_buffer.write("ANÁLISE DE VALORES AUSENTES\n")
        report_buffer.write("-" * 80 + "\n")
        missing_report = self.get_missing_values_report(df)
        if missing_report.empty:
            report_buffer.write("Não foram encontrados valores ausentes no dataset.\n\n")
        else:
            report_buffer.write(f"Total de colunas com valores ausentes: {len(missing_report)}\n\n")
            report_buffer.write(missing_report.to_string(index=False) + "\n\n")
            
            if include_visualizations:
                fig = self.visualize_missing_values(df)
                if fig:
                    missing_viz_path = "missing_values_vizualization.png" if not output_file else f"{output_file.rsplit('.', 1)[0]}_missing.png"
                    fig.savefig(missing_viz_path)
                    report_buffer.write(f"Visualização salva em: {missing_viz_path}\n\n")
        
        # Informações sobre outliers
        report_buffer.write("ANÁLISE DE OUTLIERS\n")
        report_buffer.write("-" * 80 + "\n")
        outliers_report = self.get_outliers_report(df)
        if outliers_report.empty:
            report_buffer.write("Não foram detectados outliers significativos no dataset.\n\n")
        else:
            report_buffer.write(f"Total de colunas com outliers: {len(outliers_report)}\n\n")
            # Reduzir colunas para melhor formatação no relatório
            compact_report = outliers_report[['coluna', 'num_outliers', 'porcentagem', 'limite_inferior', 'limite_superior', 'recomendacao']]
            report_buffer.write(compact_report.to_string(index=False) + "\n\n")
            
            if include_visualizations:
                fig = self.visualize_outliers(df)
                if fig:
                    outliers_viz_path = "outliers_vizualization.png" if not output_file else f"{output_file.rsplit('.', 1)[0]}_outliers.png"
                    fig.savefig(outliers_viz_path)
                    report_buffer.write(f"Visualização salva em: {outliers_viz_path}\n\n")
        
        # Informações sobre importância de features
        if target:
            report_buffer.write("ANÁLISE DE IMPORTÂNCIA DE FEATURES\n")
            report_buffer.write("-" * 80 + "\n")
            importance_report = self.get_feature_importance_report(df, target)
            if importance_report.empty:
                report_buffer.write("Não foi possível calcular a importância das features.\n\n")
            else:
                report_buffer.write(f"Total de features analisadas: {len(importance_report)}\n\n")
                # Exibir top 15 features mais importantes
                top_features = importance_report.head(15)
                report_buffer.write(top_features.to_string(index=False) + "\n\n")
                
                if include_visualizations:
                    fig = self.visualize_feature_importance(df, target)
                    if fig:
                        importance_viz_path = "feature_importance_vizualization.png" if not output_file else f"{output_file.rsplit('.', 1)[0]}_importance.png"
                        fig.savefig(importance_viz_path)
                        report_buffer.write(f"Visualização salva em: {importance_viz_path}\n\n")
        
        # Informações sobre transformações (se pipeline ajustado)
        if self.fitted:
            report_buffer.write("ANÁLISE DE TRANSFORMAÇÕES APLICADAS\n")
            report_buffer.write("-" * 80 + "\n")
            transformations_report = self.get_transformations_report()
            
            # Preprocessamento
            preproc_info = transformations_report.get('preprocessamento', {})
            report_buffer.write("1. Preprocessamento:\n")
            
            if 'transformers' in preproc_info:
                transformers = preproc_info['transformers']
                report_buffer.write(f"   - Scaling: {transformers.get('scaling', 'N/A')}\n")
                report_buffer.write(f"   - Tratamento de valores ausentes: {transformers.get('missing_values', 'N/A')}\n")
                report_buffer.write(f"   - Codificação de variáveis categóricas: {transformers.get('categorical_strategy', 'N/A')}\n")
                
                additional = transformers.get('additional_transformers', [])
                if additional:
                    report_buffer.write(f"   - Transformadores adicionais: {', '.join(additional)}\n")
            
            # Feature Engineering
            fe_info = transformations_report.get('engenharia_features', {})
            report_buffer.write("\n2. Engenharia de Features:\n")
            
            if fe_info:
                for key, value in fe_info.items():
                    if key == 'feature_selection_params' and isinstance(value, dict):
                        report_buffer.write(f"   - {key}:\n")
                        for param_key, param_value in value.items():
                            report_buffer.write(f"      * {param_key}: {param_value}\n")
                    else:
                        report_buffer.write(f"   - {key}: {value}\n")
            
            # Estatísticas de transformação
            stats = transformations_report.get('estatisticas', {})
            report_buffer.write("\n3. Estatísticas de Transformação:\n")
            
            if stats and stats['dimensoes_originais'] is not None:
                report_buffer.write(f"   - Features originais: {stats['dimensoes_originais'][1]}\n")
                report_buffer.write(f"   - Features após transformação: {stats['dimensoes_transformadas'][1]}\n")
                
                if stats['reducao_features_pct'] >= 0:
                    report_buffer.write(f"   - Redução de features: {stats['reducao_features_pct']}%\n")
                else:
                    report_buffer.write(f"   - Aumento de features: {abs(stats['reducao_features_pct'])}%\n")
                
                if stats['ganho_performance_pct'] >= 0:
                    report_buffer.write(f"   - Ganho de performance: +{stats['ganho_performance_pct']}%\n")
                else:
                    report_buffer.write(f"   - Perda de performance: {stats['ganho_performance_pct']}%\n")
                
                report_buffer.write(f"   - Decisão final: {stats['decisao_final']}\n")
                
            if include_visualizations:
                fig = self.visualize_transformations()
                if fig:
                    trans_viz_path = "transformations_vizualization.png" if not output_file else f"{output_file.rsplit('.', 1)[0]}_transformations.png"
                    fig.savefig(trans_viz_path)
                    report_buffer.write(f"\nVisualização salva em: {trans_viz_path}\n")
        
        # Finalização do relatório
        report_buffer.write("\n" + "=" * 80 + "\n")
        report_buffer.write("CONCLUSÃO\n")
        
        if self.fitted and self.validation_results:
            decision = self.validation_results.get('best_choice', '').upper()
            diff_pct = self.validation_results.get('performance_diff_pct', 0)
            
            if decision == 'TRANSFORMED':
                if diff_pct > 0:
                    report_buffer.write(f"O pipeline CAFE melhorou a performance em {diff_pct:.2f}%.\n")
                    report_buffer.write("Recomendamos utilizar o dataset transformado para seus modelos de machine learning.\n")
                else:
                    report_buffer.write(f"O pipeline CAFE manteve a performance ({diff_pct:.2f}%) com menos features.\n")
                    report_buffer.write("Recomendamos utilizar o dataset transformado para melhor eficiência e generalização.\n")
            else:
                report_buffer.write(f"O pipeline CAFE não conseguiu melhorar a performance ({diff_pct:.2f}%).\n")
                report_buffer.write("Recomendamos utilizar o dataset original para seus modelos de machine learning.\n")
        else:
            report_buffer.write("Nenhuma transformação foi aplicada ainda. Execute pipeline.fit_transform() para obter recomendações.\n")
        
        report_buffer.write("=" * 80 + "\n")
        
        report = report_buffer.getvalue()
        
        # Salvar o relatório em arquivo, se especificado
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
        
        return report