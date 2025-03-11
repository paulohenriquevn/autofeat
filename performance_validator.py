"""
Módulo para validação de performance antes e depois da aplicação de transformações do AutoFE.
Este componente compara a performance de um modelo treinado nos dados originais
versus dados transformados pelo AutoFE e decide qual conjunto de dados usar.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Optional, Tuple, Union, List
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Configurar logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AutoFE.PerformanceValidator")

class PerformanceValidator:
    """
    Classe que valida a performance de transformações do AutoFE comparando
    a performance antes e depois das transformações.
    """
    def __init__(self, config: Optional[Dict] = None):
        """
        Inicializa o validador com configurações opcionais.
        
        Args:
            config: Dicionário com configurações como:
                - max_performance_drop: Máxima queda de performance permitida (0.05 = 5%)
                - cv_folds: Número de folds para validação cruzada
                - metric: Métrica para avaliar performance ('accuracy', 'f1', 'rmse', 'r2')
                - task: Tipo de tarefa ('classification' ou 'regression')
                - base_model: Modelo base para avaliação ('rf', 'lr', 'knn')
        """
        self.config = {
            'max_performance_drop': 0.05,  # 5% de queda máxima permitida
            'cv_folds': 5,
            'metric': 'accuracy',  # 'accuracy', 'f1', 'rmse', 'r2'
            'task': 'classification',  # 'classification' ou 'regression'
            'base_model': 'rf',  # 'rf', 'lr', 'knn'
            'verbose': True
        }
        
        if config:
            self.config.update(config)
            
        self.logger = logger
        self.performance_original = None
        self.performance_transformed = None
        self.best_choice = None
        
    def _get_base_model(self):
        """Retorna o modelo base de acordo com a configuração."""
        task = self.config['task']
        model_type = self.config['base_model']
        
        if task == 'classification':
            if model_type == 'rf':
                return RandomForestClassifier(n_estimators=100, random_state=42)
            elif model_type == 'lr':
                return LogisticRegression(random_state=42, max_iter=1000)
            elif model_type == 'knn':
                return KNeighborsClassifier(n_neighbors=5)
        else:  # regression
            if model_type == 'rf':
                from sklearn.ensemble import RandomForestRegressor
                return RandomForestRegressor(n_estimators=100, random_state=42)
            elif model_type == 'lr':
                from sklearn.linear_model import LinearRegression
                return LinearRegression()
            elif model_type == 'knn':
                from sklearn.neighbors import KNeighborsRegressor
                return KNeighborsRegressor(n_neighbors=5)
                
        # Default para classificação com RandomForest
        return RandomForestClassifier(n_estimators=100, random_state=42)
    
    def _compute_metric(self, y_true, y_pred):
        """Calcula a métrica especificada na configuração."""
        metric = self.config['metric']
        task = self.config['task']
        
        if task == 'classification':
            if metric == 'accuracy':
                return accuracy_score(y_true, y_pred)
            elif metric == 'f1':
                return f1_score(y_true, y_pred, average='weighted')
        else:  # regression
            if metric == 'rmse':
                return -np.sqrt(mean_squared_error(y_true, y_pred))  # Negativo para maximizar
            elif metric == 'r2':
                return r2_score(y_true, y_pred)
                
        # Default para classificação com acurácia
        return accuracy_score(y_true, y_pred)
    
    def evaluate(self, X_original: pd.DataFrame, y: pd.Series, X_transformed: pd.DataFrame) -> Dict:
        """
        Avalia a performance nos dados originais vs. transformados via validação cruzada.
        
        Args:
            X_original: DataFrame com features originais
            y: Series com os valores alvo
            X_transformed: DataFrame com features transformadas pelo AutoFE
            
        Returns:
            Dicionário com resultados da avaliação
        """
        cv_folds = self.config['cv_folds']
        metric = self.config['metric']
        task = self.config['task']
        
        if self.config['verbose']:
            self.logger.info(f"Iniciando avaliação de performance com {cv_folds} folds")
            self.logger.info(f"Dataset original: {X_original.shape}, Dataset transformado: {X_transformed.shape}")
        
        # Verificar compatibilidade dos índices e alinhar os datasets
        if not X_transformed.index.equals(X_original.index):
            self.logger.warning("Os índices dos datasets original e transformado não coincidem.")
            self.logger.info("Alinhando os datasets pelo índice comum...")
            
            # Encontrar os índices comuns
            common_indices = X_original.index.intersection(X_transformed.index)
            if len(common_indices) == 0:
                self.logger.error("Não há índices em comum entre os datasets original e transformado.")
                return {
                    'performance_original': 0.0,
                    'performance_transformed': 0.0,
                    'performance_diff': 0.0,
                    'performance_diff_pct': 0.0,
                    'best_choice': 'original',  # Default para original em caso de erro
                    'scores_original': [0.0],
                    'scores_transformed': [0.0],
                    'feature_reduction': 0.0,
                    'error': "Sem índices em comum entre os datasets"
                }
            
            # Filtrar para usar apenas os índices comuns
            X_original = X_original.loc[common_indices]
            X_transformed = X_transformed.loc[common_indices]
            y = y.loc[common_indices]
            
            self.logger.info(f"Datasets alinhados com {len(common_indices)} amostras em comum")
        
        # Preparar validação cruzada
        if task == 'classification':
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            scoring = 'accuracy' if metric == 'accuracy' else 'f1_weighted'
        else:  # regression
            from sklearn.model_selection import KFold
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
            scoring = 'neg_root_mean_squared_error' if metric == 'rmse' else 'r2'
        
        # Obter modelo base
        model = self._get_base_model()
        
        # Avaliar dados originais
        scores_original = cross_val_score(model, X_original, y, cv=cv, scoring=scoring)
        self.performance_original = np.mean(scores_original)
        
        # Avaliar dados transformados
        scores_transformed = cross_val_score(model, X_transformed, y, cv=cv, scoring=scoring)
        self.performance_transformed = np.mean(scores_transformed)
        
        # Calcular diferença de performance
        performance_diff = self.performance_transformed - self.performance_original
        performance_diff_pct = performance_diff / abs(self.performance_original) * 100
        
        # Decidir qual dataset usar
        max_drop = self.config['max_performance_drop']
        
        if performance_diff >= -max_drop:
            self.best_choice = 'transformed'
        else:
            self.best_choice = 'original'
            
        # Preparar resultados
        results = {
            'performance_original': self.performance_original,
            'performance_transformed': self.performance_transformed,
            'performance_diff': performance_diff,
            'performance_diff_pct': performance_diff_pct,
            'best_choice': self.best_choice,
            'scores_original': scores_original.tolist(),
            'scores_transformed': scores_transformed.tolist(),
            'feature_reduction': 1 - (X_transformed.shape[1] / X_original.shape[1])
        }
        
        if self.config['verbose']:
            self.logger.info(f"Performance original: {self.performance_original:.4f}")
            self.logger.info(f"Performance transformada: {self.performance_transformed:.4f}")
            self.logger.info(f"Diferença de performance: {performance_diff:.4f} ({performance_diff_pct:.2f}%)")
            self.logger.info(f"Redução de features: {results['feature_reduction']*100:.1f}%")
            self.logger.info(f"Melhor escolha: {self.best_choice}")
        
        return results
    
    def get_best_dataset(self, X_original: pd.DataFrame, y: pd.Series, X_transformed: pd.DataFrame, 
                         include_validation_results: bool = False) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict]]:
        """
        Retorna o melhor dataset (original ou transformado) com base na avaliação de performance.
        
        Args:
            X_original: DataFrame com features originais
            y: Series com os valores alvo
            X_transformed: DataFrame com features transformadas pelo AutoFE
            include_validation_results: Se True, retorna também os resultados da validação
            
        Returns:
            O melhor dataset ou tuple (melhor_dataset, resultados) se include_validation_results=True
        """
        results = self.evaluate(X_original, y, X_transformed)
        
        best_dataset = X_transformed if results['best_choice'] == 'transformed' else X_original
        
        if include_validation_results:
            return best_dataset, results
        else:
            return best_dataset
    
    def get_feature_importance(self, X_original: pd.DataFrame, y: pd.Series, feature_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Retorna a importância das features usando um modelo de árvore como RandomForest.
        Útil para entender quais features são mais importantes.
        
        Args:
            X_original: DataFrame com os dados para avaliação
            y: Series com os valores alvo
            feature_names: Lista opcional com nomes das features (se None, usa os nomes do DataFrame)
            
        Returns:
            DataFrame com as features ordenadas por importância
        """
        if self.config['task'] == 'classification':
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            
        # Ajustar modelo
        model.fit(X_original, y)
        
        # Obter nomes das features
        if feature_names is None:
            feature_names = X_original.columns.tolist()
            
        # Calcular importância
        importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance