import pandas as pd
import logging
from typing import Dict


class ReportDataPipeline:
    def __init__(self, df: pd.DataFrame, target_col: str="", preprocessor=None, 
                feature_engineer=None, validator=None, 
                max_sample_size: int=10000, sampling_threshold: int=50000):
        """
        Versão aprimorada do ReportDataPipeline com melhor performance para datasets grandes.
        
        Args:
            df: DataFrame com os dados
            target_col: Nome da coluna alvo
            preprocessor: Instância do preprocessador (opcional)
            feature_engineer: Instância do feature engineer (opcional)
            validator: Instância do validator (opcional)
            max_sample_size: Tamanho máximo da amostra para análise
            sampling_threshold: Limiar para usar amostragem automática
        """
        self.df = df
        self.target_col = target_col
        self.preprocessor = preprocessor
        self.feature_engineer = feature_engineer
        self.validator = validator
        
        # Parâmetros para controle de performance
        self.max_sample_size = max_sample_size
        self.sampling_threshold = sampling_threshold
        self.use_sampling = len(df) > sampling_threshold
        
        # Inicializa o logger
        self.logger = logging.getLogger("CAFE.ImprovedReportDataPipeline")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        
        if self.use_sampling:
            self.logger.info(f"Dataset grande detectado ({len(df)} amostras). Usando amostragem automática.")
    
    def _get_sample_df(self) -> pd.DataFrame:
        """
        Retorna uma amostra do DataFrame se necessário.
        
        Returns:
            DataFrame original ou amostra se o DataFrame for grande
        """
        if not self.use_sampling:
            return self.df
            
        # Usar amostra estratificada se tiver target categórico
        if self.target_col and self.target_col in self.df.columns and pd.api.types.is_categorical_dtype(self.df[self.target_col].dtype):
            try:
                return self.df.groupby(self.target_col, group_keys=False).apply(
                    lambda x: x.sample(min(len(x), self.max_sample_size // len(self.df[self.target_col].unique())))
                )
            except Exception as e:
                self.logger.warning(f"Erro ao usar amostragem estratificada: {e}. Usando amostragem aleatória.")
                
        # Amostragem aleatória
        sample_size = min(self.max_sample_size, len(self.df))
        return self.df.sample(n=sample_size, random_state=42)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Calcula importância das features usando métodos eficientes para datasets grandes.
        
        Returns:
            DataFrame com importância das features
        """
        if self.target_col is None or not self.target_col:
            self.logger.error("É necessário fornecer a coluna alvo para calcular a importância das features")
            return pd.DataFrame(columns=['feature', 'importance', 'normalized_importance', 'categoria'])
                
        if self.target_col not in self.df.columns:
            self.logger.error(f"Coluna alvo '{self.target_col}' não encontrada no DataFrame")
            return pd.DataFrame(columns=['feature', 'importance', 'normalized_importance', 'categoria'])
        
        # Usar amostra para datasets grandes
        df_to_use = self._get_sample_df()
        
        try:
            # Separar features e target
            X = df_to_use.drop(columns=[self.target_col])
            y = df_to_use[self.target_col]
            
            # Se temos um validator disponível, usar seu método
            if self.validator and hasattr(self.validator, 'get_feature_importance'):
                importance = self.validator.get_feature_importance(X, y)
            else:
                # Detectar tipo de problema
                task = 'classification'
                if pd.api.types.is_numeric_dtype(y.dtype) and y.nunique() > 10:
                    task = 'regression'
                
                # Tratar dados categóricos
                X_processed = X.copy()
                cat_cols = X_processed.select_dtypes(include=['object', 'category']).columns
                if len(cat_cols) > 0:
                    self.logger.info(f"Detectadas {len(cat_cols)} colunas categóricas.")
                    X_processed = pd.get_dummies(X_processed, columns=cat_cols, drop_first=True)
                
                # Para datasets grandes, usar métodos mais leves
                if self.use_sampling or len(X) > 10000:
                    # Usar métodos baseados em correlação/informação mútua
                    if task == 'classification':
                        from sklearn.feature_selection import mutual_info_classif
                        importances = mutual_info_classif(X_processed, y)
                    else:
                        from sklearn.feature_selection import mutual_info_regression
                        importances = mutual_info_regression(X_processed, y)
                        
                    # Criar DataFrame com importâncias
                    importance = pd.DataFrame({
                        'feature': X_processed.columns,
                        'importance': importances
                    }).sort_values('importance', ascending=False)
                else:
                    # Para datasets menores, usar Random Forest (mais preciso)
                    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
                    
                    # Criar modelo apropriado
                    if task == 'classification':
                        model = RandomForestClassifier(n_estimators=100, random_state=42)
                    else:
                        model = RandomForestRegressor(n_estimators=100, random_state=42)
                    
                    # Treinar modelo
                    model.fit(X_processed, y)
                    
                    # Criar DataFrame com importâncias
                    importance = pd.DataFrame({
                        'feature': X_processed.columns,
                        'importance': model.feature_importances_
                    }).sort_values('importance', ascending=False)
            
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
    
    def get_missing_values(self) -> pd.DataFrame:
        """
        Gera relatório sobre valores ausentes usando métodos eficientes.
        
        Returns:
            DataFrame com estatísticas e recomendações sobre valores ausentes
        """
        if self.df.empty:
            return pd.DataFrame(columns=['coluna', 'valores_ausentes', 'porcentagem', 'recomendacao'])
        
        # Para datasets grandes, calcular valores ausentes em lotes
        if self.use_sampling and len(self.df) > 1000000:  # Para datasets realmente grandes
            # Calcular valores ausentes em lotes para economizar memória
            batch_size = 100000
            missing_count = pd.Series(0, index=self.df.columns)
            total_rows = len(self.df)
            
            for start_idx in range(0, total_rows, batch_size):
                end_idx = min(start_idx + batch_size, total_rows)
                batch = self.df.iloc[start_idx:end_idx]
                missing_count = missing_count.add(batch.isnull().sum(), fill_value=0)
                
            missing_percent = (missing_count / total_rows) * 100
        else:
            # Cálculo padrão para datasets menores
            missing_count = self.df.isnull().sum()
            missing_percent = (missing_count / len(self.df)) * 100
        
        report = pd.DataFrame({
            'coluna': missing_count.index,
            'valores_ausentes': missing_count.values,
            'porcentagem': missing_percent.values.round(2)
        })
        
        report = report.sort_values('valores_ausentes', ascending=False)
        
        # Determinar estratégia de tratamento
        strategy = "median"  # Valor padrão
        if self.preprocessor and hasattr(self.preprocessor, 'config'):
            strategy = self.preprocessor.config.get('missing_values_strategy', 'median')
        
        def get_recommendation(row):
            col = row['coluna']
            pct = row['porcentagem']
            
            if pct == 0:
                return "Sem valores ausentes"
            
            if col in self.df.columns:
                # Detectar tipo da coluna mais eficientemente
                sample_col = self.df[col].head(1000)  # Usar apenas uma amostra
                dtype = sample_col.dtype
                is_numeric = pd.api.types.is_numeric_dtype(dtype)
                is_categorical = pd.api.types.is_categorical_dtype(dtype) or pd.api.types.is_object_dtype(dtype)
                
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
        
        # Aplicar a função get_recommendation para cada linha (mais eficiente com apply)
        report['recomendacao'] = report.apply(get_recommendation, axis=1)
        
        report_with_missing = report[report['valores_ausentes'] > 0]
        
        if report_with_missing.empty:
            self.logger.info("Não foram encontrados valores ausentes no dataset.")
            
        return report_with_missing
    
    def get_outliers(self) -> pd.DataFrame:
        """
        Gera relatório sobre outliers usando métodos otimizados para datasets grandes.
        
        Returns:
            DataFrame com estatísticas e recomendações sobre outliers
        """
        if self.df.empty:
            return pd.DataFrame(columns=['coluna', 'num_outliers', 'porcentagem', 'limite_inferior', 'limite_superior', 'recomendacao'])
        
        # Selecionar apenas colunas numéricas
        numeric_cols = self.df.select_dtypes(include=['number']).columns
        if len(numeric_cols) == 0:
            self.logger.warning("Não foram encontradas colunas numéricas para análise de outliers.")
            return pd.DataFrame(columns=['coluna', 'num_outliers', 'porcentagem', 'limite_inferior', 'limite_superior', 'recomendacao'])
        
        # Determinar método para detecção de outliers
        outlier_method = "iqr"  # Valor padrão
        if self.preprocessor and hasattr(self.preprocessor, 'config'):
            outlier_method = self.preprocessor.config.get('outlier_method', 'iqr')
        
        # Usar amostra para datasets grandes
        df_to_use = self._get_sample_df()
        
        # Preparar resultados
        results = []
        
        for col in numeric_cols:
            # Processar uma coluna por vez para evitar sobrecarga de memória
            series = df_to_use[col].dropna()
            
            # Ignorar colunas binárias (0/1) ou com poucos valores únicos
            if series.nunique() <= 2 or (series.nunique() / len(series) < 0.01 and series.nunique() <= 10):
                continue
                
            # Detectar outliers com método escolhido, otimizado para datasets grandes
            if outlier_method == 'zscore':
                # Calcular Z-score sem armazenar toda a série em memória
                mean_val = series.mean()
                std_val = series.std()
                
                # Calcular limites
                lower_bound = mean_val - 3 * std_val
                upper_bound = mean_val + 3 * std_val
                
                # Contar outliers sem criar array intermediário grande
                outlier_count = ((series < lower_bound) | (series > upper_bound)).sum()
            
            elif outlier_method == 'iqr':
                # Cálculo de quartis (mais eficiente)
                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                iqr = q3 - q1
                
                # Definir limites
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                # Contar outliers sem criar array intermediário grande
                outlier_count = ((series < lower_bound) | (series > upper_bound)).sum()
            
            else:  # Método fallback
                # Usar IQR como método padrão
                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                iqr = q3 - q1
                
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                outlier_count = ((series < lower_bound) | (series > upper_bound)).sum()
            
            # Calcular percentual
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
    
    def get_transformations(self) -> Dict:
        results = {
            'performance_original': self.validator.performance_original,
            'performance_transformed': self.validator.performance_transformed,
            'performance_diff': self.validator.performance_transformed - self.validator.performance_original,
            'performance_diff_pct': ((self.validator.performance_transformed - self.validator.performance_original) / 
                                    max(abs(self.validator.performance_original), 1e-10)) * 100,
            'best_choice': self.validator.best_choice or 'original',
            'feature_reduction': 0.0  # Placeholder, será atualizado se possível
        }
        
        # Adicionar scores por fold se disponíveis
        if hasattr(self.validator, 'scores_original') and hasattr(self.validator, 'scores_transformed'):
            results['scores_original'] = self.validator.scores_original
            results['scores_transformed'] = self.validator.scores_transformed
        
        # Estimar redução de features se possível
        if self.feature_engineer and hasattr(self.feature_engineer, 'input_feature_names') and hasattr(self.feature_engineer, 'output_feature_names'):
            input_count = len(self.feature_engineer.input_feature_names)
            output_count = len(self.feature_engineer.output_feature_names)
            results['feature_reduction'] = 1 - (output_count / input_count) if input_count > 0 else 0.0
            results['original_n_features'] = input_count
        
        return results
    
    def get_report_summary(self) -> Dict:
        """
        Gera um resumo conciso das principais métricas e recomendações.
        
        Returns:
            Dicionário com resumo das análises e recomendações
        """
        summary = {
            'dados': {
                'amostras': len(self.df) if self.df is not None else 0,
                'features': len(self.df.columns) - (1 if self.target_col in self.df.columns else 0) if self.df is not None else 0,
                'tipos_colunas': {
                    'numericas': len(self.df.select_dtypes(include=['number']).columns) if self.df is not None else 0,
                    'categoricas': len(self.df.select_dtypes(include=['object', 'category']).columns) if self.df is not None else 0,
                    'temporais': len(self.df.select_dtypes(include=['datetime']).columns) if self.df is not None else 0
                }
            },
            'valores_ausentes': {
                'colunas_com_ausentes': 0,
                'porcentagem_media': 0.0,
                'recomendacao': "Sem valores ausentes"
            },
            'outliers': {
                'colunas_com_outliers': 0,
                'porcentagem_media': 0.0,
                'recomendacao': "Sem outliers significativos"
            },
            'transformacoes': {
                'features_originais': 0,
                'features_transformadas': 0,
                'reducao_features_pct': 0.0,
                'ganho_performance_pct': 0.0,
                'recomendacao': "Nenhuma transformação aplicada"
            }
        }
        
        # Obter informações de valores ausentes
        try:
            missing_report = self.get_missing_values()
            if not missing_report.empty:
                summary['valores_ausentes']['colunas_com_ausentes'] = len(missing_report)
                summary['valores_ausentes']['porcentagem_media'] = missing_report['porcentagem'].mean()
                
                # Gerar recomendação geral para valores ausentes
                if summary['valores_ausentes']['porcentagem_media'] > 20:
                    summary['valores_ausentes']['recomendacao'] = "Alta presença de valores ausentes. Considere técnicas avançadas de imputação ou remoção de colunas."
                elif summary['valores_ausentes']['porcentagem_media'] > 5:
                    summary['valores_ausentes']['recomendacao'] = "Presença moderada de valores ausentes. Recomendado usar métodos como median/mean/KNN."
                else:
                    summary['valores_ausentes']['recomendacao'] = "Baixa presença de valores ausentes. Métodos básicos de imputação são suficientes."
        except Exception as e:
            self.logger.warning(f"Erro ao obter resumo de valores ausentes: {e}")
        
        # Obter informações de outliers
        try:
            outliers_report = self.get_outliers()
            if not outliers_report.empty:
                summary['outliers']['colunas_com_outliers'] = len(outliers_report)
                summary['outliers']['porcentagem_media'] = outliers_report['porcentagem'].mean()
                
                # Gerar recomendação geral para outliers
                if summary['outliers']['porcentagem_media'] > 10:
                    summary['outliers']['recomendacao'] = "Alta presença de outliers. Considere transformações robustas ou remoção seletiva."
                elif summary['outliers']['porcentagem_media'] > 5:
                    summary['outliers']['recomendacao'] = "Presença moderada de outliers. Recomendado usar RobustScaler ou métodos de winsorização."
                else:
                    summary['outliers']['recomendacao'] = "Baixa presença de outliers. Considere usar métodos padrão de tratamento."
        except Exception as e:
            self.logger.warning(f"Erro ao obter resumo de outliers: {e}")
        
        # Obter informações de transformações
        try:
            transformations = self.get_transformations()
            stats = transformations.get('estatisticas', {})
            
            if stats and stats['dimensoes_originais'] is not None:
                summary['transformacoes']['features_originais'] = stats['dimensoes_originais'][1]
                summary['transformacoes']['features_transformadas'] = stats['dimensoes_transformadas'][1]
                summary['transformacoes']['reducao_features_pct'] = stats['reducao_features_pct']
                summary['transformacoes']['ganho_performance_pct'] = stats['ganho_performance_pct']
                
                # Gerar recomendação geral para transformações
                if stats['decisao_final'] == 'TRANSFORMED':
                    if stats['ganho_performance_pct'] > 5:
                        summary['transformacoes']['recomendacao'] = "Transformações melhoraram significativamente a performance. Recomendado usar o dataset transformado."
                    else:
                        summary['transformacoes']['recomendacao'] = "Transformações trouxeram ganhos moderados. Recomendado usar o dataset transformado para maior eficiência."
                else:
                    summary['transformacoes']['recomendacao'] = "Transformações não melhoraram a performance. Recomendado manter o dataset original."
        except Exception as e:
            self.logger.warning(f"Erro ao obter resumo de transformações: {e}")
        
        # Obter importância de features (top 5)
        try:
            importance_report = self.get_feature_importance()
            if not importance_report.empty:
                top_features = importance_report.head(5)['feature'].tolist()
                summary['features_importantes'] = {
                    'top_5': top_features,
                    'categorias': importance_report.head(5)['categoria'].tolist()
                }
        except Exception as e:
            self.logger.warning(f"Erro ao obter resumo de importância de features: {e}")
        
        return summary