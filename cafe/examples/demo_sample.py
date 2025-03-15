"""
Exemplo integrado de uso das classes ReportDataPipeline e ReportVisualizer
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# Importações do CAFE
from cafe.data_pipeline import DataPipeline
from cafe.report_datapipeline import ReportDataPipeline  # Classe para relatórios
from cafe.report_visualizer import ReportVisualizer  # Classe para visualizações

def integrated_demo():
    """
    Demonstração de como usar ReportDataPipeline e ReportVisualizer juntos
    para analisar dados e visualizar resultados.
    """
    print("\n" + "=" * 80)
    print("DEMONSTRAÇÃO INTEGRADA: REPORT DATA PIPELINE E REPORT VISUALIZER".center(80))
    print("=" * 80 + "\n")
    
    # 1. Carregar um dataset de exemplo
    print("1. Carregando dataset de exemplo...")
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    
    # Adicionar valores ausentes e outliers artificiais para demonstração
    print("   Adicionando valores ausentes e outliers artificiais...")
    rows, cols = df.shape
    
    # Valores ausentes (5%)
    mask = np.random.random(size=(rows, cols-1)) < 0.05
    for i in range(cols-1):  # excluir a coluna target
        df.iloc[mask[:, i], i] = np.nan
    
    # Outliers (em 5 colunas)
    for col in df.columns[:5]:
        outlier_indices = np.random.choice(range(len(df)), size=10, replace=False)
        df.loc[outlier_indices, col] = df[col].mean() + 10 * df[col].std()
    
    print(f"   Dataset carregado: {df.shape[0]} amostras, {df.shape[1]} colunas")
    
    # 2. Criar e aplicar o pipeline CAFE
    print("\n2. Configurando e aplicando o pipeline CAFE...")
    
    pipeline = DataPipeline(
        preprocessor_config={
            'missing_values_strategy': 'median',
            'outlier_method': 'iqr',
            'categorical_strategy': 'onehot',
            'scaling': 'standard'
        },
        feature_engineer_config={
            'correlation_threshold': 0.8,
            'generate_features': False,
            'feature_selection': 'kbest',
            'feature_selection_params': {'k': 15, 'score_func': 'mutual_info'},
            'task': 'classification'
        },
        auto_validate=True
    )
    
    # Aplicar pipeline ao dataset
    transformed_df = pipeline.fit_transform(df, target_col='target')
    print(f"   Dataset transformado: {transformed_df.shape[0]} amostras, {transformed_df.shape[1]} colunas")
    
    # 3. Criar instâncias de ReportDataPipeline e ReportVisualizer
    print("\n3. Criando instâncias de relatórios e visualizações...")
    
    # O ReportDataPipeline foca na geração de relatórios
    reporter = ReportDataPipeline(
        df=df,  # Dataset original
        target_col='target',
        preprocessor=pipeline.preprocessor,
        feature_engineer=pipeline.feature_engineer,
        validator=pipeline.validator
    )
    
    # O ReportVisualizer foca na visualização gráfica
    visualizer = ReportVisualizer()
    
    # 4. Gerar relatórios de análise
    print("\n4. Gerando relatórios de análise...")
    
    # 4.1 Análise de valores ausentes
    print("\n   4.1 Análise de valores ausentes")
    missing_report = reporter.get_missing_values()
    print(f"   Total de colunas com valores ausentes: {len(missing_report)}")
    if not missing_report.empty:
        print("   Primeiras 3 colunas com mais valores ausentes:")
        print(missing_report.head(3))
        
        # Visualização dos valores ausentes
        print("\n   Visualizando valores ausentes...")
        fig_missing = visualizer.visualize_missing_values(missing_report)
        if fig_missing:
            plt.figure(fig_missing.number)
            plt.savefig("missing_values.png")
            print("   Visualização salva em: missing_values.png")
    
    # 4.2 Análise de outliers
    print("\n   4.2 Análise de outliers")
    outliers_report = reporter.get_outliers()
    print(f"   Total de colunas com outliers: {len(outliers_report)}")
    if not outliers_report.empty:
        print("   Primeiras 3 colunas com mais outliers:")
        print(outliers_report.head(3)[['coluna', 'num_outliers', 'porcentagem', 'recomendacao']])
        
        # Visualização dos outliers
        print("\n   Visualizando outliers...")
        fig_outliers = visualizer.visualize_outliers(outliers_report, df)
        if fig_outliers:
            plt.figure(fig_outliers.number)
            plt.savefig("outliers.png")
            print("   Visualização salva em: outliers.png")
    
    # 4.3 Análise de importância de features
    print("\n   4.3 Análise de importância de features")
    importance_report = reporter.get_feature_importance()
    print("   Top 5 features mais importantes:")
    print(importance_report.head(5))
    
    # Visualização da importância das features
    print("\n   Visualizando importância de features...")
    fig_importance = visualizer.visualize_feature_importance(importance_report)
    if fig_importance:
        plt.figure(fig_importance.number)
        plt.savefig("feature_importance.png")
        print("   Visualização salva em: feature_importance.png")
    
    # 4.4 Análise de transformações
    print("\n   4.4 Análise de transformações")
    transformations_report = reporter.get_transformations()
    
    # Mostrar estatísticas
    stats = transformations_report.get('estatisticas', {})
    if stats and stats['dimensoes_originais'] is not None:
        print("   Estatísticas das transformações:")
        print(f"   - Features originais: {stats['dimensoes_originais'][1]}")
        print(f"   - Features após transformação: {stats['dimensoes_transformadas'][1]}")
        print(f"   - Redução de features: {abs(stats['reducao_features_pct'])}%")
        
        if stats['ganho_performance_pct'] >= 0:
            print(f"   - Ganho de performance: +{stats['ganho_performance_pct']}%")
        else:
            print(f"   - Perda de performance: {abs(stats['ganho_performance_pct'])}%")
            
        print(f"   - Decisão final: {stats['decisao_final']}")
        
        # Visualização das transformações
        print("\n   Visualizando transformações...")
        validation_results = transformations_report.get('validacao', {})
        fig_transformations = visualizer.visualize_transformations(validation_results, stats)
        if fig_transformations:
            plt.figure(fig_transformations.number)
            plt.savefig("transformations.png")
            print("   Visualização salva em: transformations.png")
    
    # 4.5 Gerar resumo conciso
    print("\n   4.5 Resumo conciso")
    summary = reporter.get_report_summary()
    
    print("\n   Resumo dos dados:")
    print(f"   - Amostras: {summary['dados']['amostras']}")
    print(f"   - Features: {summary['dados']['features']}")
    
    print("\n   Resumo dos valores ausentes:")
    print(f"   - Colunas com ausentes: {summary['valores_ausentes']['colunas_com_ausentes']}")
    print(f"   - Porcentagem média: {summary['valores_ausentes']['porcentagem_media']:.2f}%")
    print(f"   - Recomendação: {summary['valores_ausentes']['recomendacao']}")
    
    print("\n   Resumo dos outliers:")
    print(f"   - Colunas com outliers: {summary['outliers']['colunas_com_outliers']}")
    print(f"   - Porcentagem média: {summary['outliers']['porcentagem_media']:.2f}%")
    print(f"   - Recomendação: {summary['outliers']['recomendacao']}")
    
    print("\n   Resumo das transformações:")
    print(f"   - Features originais: {summary['transformacoes']['features_originais']}")
    print(f"   - Features transformadas: {summary['transformacoes']['features_transformadas']}")
    print(f"   - Redução: {summary['transformacoes']['reducao_features_pct']:.2f}%")
    print(f"   - Ganho performance: {summary['transformacoes']['ganho_performance_pct']:.2f}%")
    print(f"   - Recomendação: {summary['transformacoes']['recomendacao']}")
    
    # 5. Visualizações adicionais
    print("\n5. Gerando visualizações adicionais...")
    
    # 5.1 Visualização de distribuição de dados
    print("\n   5.1 Visualização de distribuição de dados")
    top_features = importance_report.head(6)['feature'].tolist() if not importance_report.empty else None
    fig_distribution = visualizer.visualize_data_distribution(df, columns=top_features)
    if fig_distribution:
        plt.figure(fig_distribution.number)
        plt.savefig("feature_distributions.png")
        print("   Visualização salva em: feature_distributions.png")
    
    # 5.2 Visualização de matriz de correlação
    print("\n   5.2 Visualização de matriz de correlação")
    correlation_plots = visualizer.visualize_correlation_matrix(df, target_col='target')
    if correlation_plots:
        if isinstance(correlation_plots, tuple):
            fig_corr, fig_target_corr = correlation_plots
            plt.figure(fig_corr.number)
            plt.savefig("correlation_matrix.png")
            plt.figure(fig_target_corr.number)
            plt.savefig("target_correlations.png")
            print("   Visualizações salvas em: correlation_matrix.png e target_correlations.png")
        else:
            plt.figure(correlation_plots.number)
            plt.savefig("correlation_matrix.png")
            print("   Visualização salva em: correlation_matrix.png")
    
    # 6. Gerar e salvar relatório completo
    print("\n6. Gerando relatório completo...")
    report = reporter.generate_report(output_file="cafe_report.txt")
    print(f"   Relatório completo salvo em: cafe_report.txt")
    
    # 7. Demonstração de fluxo de trabalho independente
    print("\n7. Demonstração de fluxo independente (apenas análise sem pipeline)")
    print("\n   7.1 Análise sem pipeline CAFE...")
    
    # Criar um novo dataset de exemplo
    from sklearn.datasets import load_wine
    wine_data = load_wine()
    wine_df = pd.DataFrame(wine_data.data, columns=wine_data.feature_names)
    wine_df['target'] = wine_data.target
    
    # Criar apenas o reporter (sem pipeline)
    standalone_reporter = ReportDataPipeline(
        df=wine_df,
        target_col='target'
    )
    
    # Obter relatórios básicos
    missing_count = standalone_reporter.get_missing_values()
    outliers_count = standalone_reporter.get_outliers()
    importance = standalone_reporter.get_feature_importance()
    
    print(f"   Dataset Wine: {wine_df.shape[0]} amostras, {wine_df.shape[1]} colunas")
    print(f"   - Valores ausentes encontrados: {not missing_count.empty}")
    print(f"   - Outliers encontrados: {len(outliers_count)} colunas")
    print(f"   - Top 3 features mais importantes: {', '.join(importance.head(3)['feature'].tolist()) if not importance.empty else 'N/A'}")
    
    # Usar o visualizador separadamente
    standalone_visualizer = ReportVisualizer()
    
    # Criar visualização
    fig_standalone = standalone_visualizer.visualize_feature_importance(importance)
    if fig_standalone:
        plt.figure(fig_standalone.number)
        plt.savefig("standalone_feature_importance.png")
        print("   Visualização independente salva em: standalone_feature_importance.png")
    
    print("\n" + "=" * 80)
    print("DEMONSTRAÇÃO CONCLUÍDA".center(80))
    print("=" * 80 + "\n")

if __name__ == "__main__":
    integrated_demo()