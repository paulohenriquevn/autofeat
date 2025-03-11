"""
Teste para verificar o controle de dimensionalidade no pipeline de dados.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from explorer import Explorer
from data_pipeline import DataPipeline
from feature_engineer import FeatureEngineer
from preprocessor import PreProcessor

def test_dimensionality_control():
    """
    Testa se as alterações realizadas controlam efetivamente a dimensionalidade.
    """
    # Carregar o dataset Wine para teste
    wine = load_wine()
    df = pd.DataFrame(wine.data, columns=wine.feature_names)
    df['target'] = wine.target
    
    print(f"Dataset original: {df.shape} (features: {df.shape[1]-1})")
    
    # Configuração anterior que gerava alta dimensionalidade
    problematic_config = {
        'preprocessor_config': {'outlier_method': 'isolation_forest'},
        'feature_engineer_config': {'generate_features': True, 'correlation_threshold': 0.95}
    }
    
    # Criar e executar pipeline problemático
    problem_pipeline = DataPipeline(
        problematic_config.get('preprocessor_config', {}),
        problematic_config.get('feature_engineer_config', {})
    )
    
    problem_df = problem_pipeline.fit_transform(df, target_col='target')
    print(f"Pipeline com configuração problemática: {problem_df.shape} (features: {problem_df.shape[1]-1})")
    print(f"Expansão: {problem_df.shape[1] / df.shape[1]:.1f}x")
    
    # Testar nova implementação com Explorer
    explorer = Explorer(target_col='target')
    best_df = explorer.analyze_transformations(df)
    best_config = explorer.get_best_pipeline_config()
    
    print("\nMelhor configuração encontrada:")
    print(f"Preprocessor: {best_config.get('preprocessor_config', {})}")
    print(f"FeatureEngineer: {best_config.get('feature_engineer_config', {})}")
    
    improved_pipeline = DataPipeline(
        best_config.get('preprocessor_config', {}),
        best_config.get('feature_engineer_config', {})
    )
    
    improved_df = improved_pipeline.fit_transform(df, target_col='target')
    print(f"\nPipeline melhorado: {improved_df.shape} (features: {improved_df.shape[1]-1})")
    print(f"Expansão: {improved_df.shape[1] / df.shape[1]:.1f}x")
    
    # Verificar correlações
    def count_high_correlations(df, threshold=0.8):
        numeric_df = df.select_dtypes(include=['number']).drop(columns=['target'], errors='ignore')
        if numeric_df.shape[1] <= 1:
            return 0
        
        corr_matrix = numeric_df.corr().abs()
        # Obter triângulo superior (sem a diagonal)
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        return (upper > threshold).sum().sum()
    
    original_high_corr = count_high_correlations(df)
    problem_high_corr = count_high_correlations(problem_df)
    improved_high_corr = count_high_correlations(improved_df)
    
    print(f"\nCorrelações altas (>0.8):")
    print(f"Original: {original_high_corr}")
    print(f"Configuração problemática: {problem_high_corr}")
    print(f"Configuração melhorada: {improved_high_corr}")
    
    # Verificar se o pipeline melhorado resolveu o problema
    success = improved_df.shape[1] < problem_df.shape[1] and improved_high_corr < problem_high_corr
    print(f"\nControle de dimensionalidade bem-sucedido: {success}")

if __name__ == "__main__":
    print("Testando controle de dimensionalidade...")
    test_dimensionality_control()