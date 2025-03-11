"""
Teste complexo para o sistema AutoFE com um dataset desafiador.

Este script:
1. Gera um dataset complexo com muitas features, correlações e valores atípicos
2. Aplica o sistema AutoFE para processamento
3. Avalia a eficácia do AutoFE em termos de qualidade do modelo e eficiência computacional
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import scipy.stats as stats

# Importar componentes do AutoFE
from preprocessor import PreProcessor
from feature_engineer import FeatureEngineer
from data_pipeline import DataPipeline
from explorer import Explorer, HeuristicSearch

# Configuração
np.random.seed(42)
OUTPUT_DIR = "complex_test_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_complex_dataset(n_samples=1000, n_features=50, n_informative=15, 
                            n_classes=3, correlation_level=0.8, noise_level=0.1,
                            missing_rate=0.05, outlier_rate=0.03):
    """
    Gera um dataset complexo com características desafiadoras.
    
    Args:
        n_samples: Número de amostras
        n_features: Número total de features
        n_informative: Número de features informativas
        n_classes: Número de classes
        correlation_level: Nível de correlação entre algumas features
        noise_level: Nível de ruído
        missing_rate: Taxa de valores ausentes
        outlier_rate: Taxa de outliers
        
    Returns:
        DataFrame com os dados gerados
    """
    print(f"Gerando dataset complexo com {n_samples} amostras e {n_features} features...")
    
    # Gerar features informativas correlacionadas
    informative_features = np.random.randn(n_samples, n_informative)
    
    # Adicionar correlação entre algumas features
    correlated_idx = int(n_informative * 0.6)  # 60% das features informativas serão correlacionadas
    for i in range(correlated_idx):
        # Criar correlação com a próxima feature
        informative_features[:, i+1] = (correlation_level * informative_features[:, i] + 
                                      (1 - correlation_level) * informative_features[:, i+1])
    
    # Gerar rótulos baseados nas features informativas
    beta = np.random.randn(n_informative)
    y_score = np.dot(informative_features, beta)
    y = np.floor(stats.rankdata(y_score) / (n_samples / n_classes)).astype(int)
    y = np.clip(y, 0, n_classes-1)  # Garantir que temos exatamente n_classes
    
    # Gerar features não-informativas (ruído)
    n_noise = n_features - n_informative
    noise_features = np.random.randn(n_samples, n_noise) * noise_level
    
    # Combinar todas as features
    X = np.hstack((informative_features, noise_features))
    
    # Adicionar mais features altamente correlacionadas
    for i in range(10):
        idx = np.random.randint(0, n_features)
        new_col = X[:, idx] * 0.95 + np.random.randn(n_samples) * 0.05
        X = np.hstack((X, new_col.reshape(-1, 1)))
    
    # Criar DataFrame
    feature_names = [f'feature_{i+1}' for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    # Adicionar valores ausentes
    mask = np.random.random(df.shape) < missing_rate
    for col in df.columns:
        if col != 'target':  # Não adicionar ausentes na coluna alvo
            df.loc[mask[:, df.columns.get_loc(col)], col] = np.nan
    
    # Adicionar outliers (usando Z-score)
    for col in feature_names:
        if np.random.random() < 0.3:  # Apenas 30% das colunas terão outliers
            outlier_mask = np.random.random(n_samples) < outlier_rate
            if outlier_mask.sum() > 0:
                outlier_values = df[col].mean() + (df[col].std() * np.random.choice([-10, 10], size=outlier_mask.sum()))
                df.loc[outlier_mask, col] = outlier_values
    
    # Adicionar features categóricas
    df['categoria_1'] = np.random.choice(['A', 'B', 'C', 'D'], size=n_samples)
    df['categoria_2'] = np.random.choice(['Alto', 'Médio', 'Baixo'], size=n_samples)
    
    # Adicionar features de data/hora transformadas em recursos numéricos
    # Em vez de usar datas diretamente, extraímos características numéricas 
    df['mes'] = np.random.randint(1, 13, size=n_samples)
    df['dia'] = np.random.randint(1, 29, size=n_samples)
    df['dia_semana'] = np.random.randint(0, 7, size=n_samples)
    
    # Adicionar algumas features computadas
    df['feature_comp_1'] = df['feature_1'] + df['feature_2']
    df['feature_comp_2'] = df['feature_3'] * df['feature_4']
    
    # Garantir que o target seja equilibrado
    class_counts = pd.Series(y).value_counts()
    min_class = class_counts.min()
    
    balanced_indices = []
    for class_label in range(n_classes):
        class_indices = np.where(y == class_label)[0]
        balanced_indices.extend(np.random.choice(class_indices, size=min_class, replace=False))
    
    df_balanced = df.iloc[balanced_indices].reset_index(drop=True)
    
    # Imprimir estatísticas do dataset
    print(f"Dataset final: {df_balanced.shape}")
    print(f"Distribuição de classes:\n{df_balanced['target'].value_counts()}")
    print(f"Valores ausentes: {df_balanced.isnull().sum().sum()} ({df_balanced.isnull().sum().sum() / (df_balanced.shape[0] * df_balanced.shape[1]) * 100:.2f}%)")
    
    # Calcular e mostrar número de pares correlacionados
    corr_matrix = df_balanced.select_dtypes(include=['number']).corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr = (upper > 0.8).sum().sum()
    print(f"Pares de features com alta correlação (>0.8): {high_corr}")
    
    return df_balanced

def analyze_dataset(df, title, output_path):
    """
    Analisa e visualiza características do dataset.
    """
    print(f"\n=== Analisando dataset: {title} ===")
    
    # Informações básicas
    print(f"Formato: {df.shape}")
    print(f"Tipos de dados:\n{df.dtypes.value_counts()}")
    print(f"Valores ausentes: {df.isnull().sum().sum()} ({df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100:.2f}%)")
    
    # Distribuição de classes
    if 'target' in df.columns:
        print(f"Distribuição de classes:\n{df['target'].value_counts()}")
    
    # Análise de correlação
    numeric_df = df.select_dtypes(include=['number'])
    if 'target' in numeric_df.columns:
        numeric_df = numeric_df.drop(columns=['target'])
    
    if not numeric_df.empty and numeric_df.shape[1] > 1:
        try:
            corr_matrix = numeric_df.corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            high_corr = (upper > 0.8).sum().sum()
            print(f"Pares de features com alta correlação (>0.8): {high_corr}")
            
            # Visualizar matriz de correlação
            plt.figure(figsize=(12, 10))
            sns.heatmap(corr_matrix, cmap='viridis', vmin=0, vmax=1, 
                        xticklabels=False, yticklabels=False)
            plt.title(f'Matriz de Correlação - {title}')
            plt.tight_layout()
            plt.savefig(f"{output_path}/{title.lower().replace(' ', '_')}_correlation.png")
            
            # Visualização PCA para ver estrutura geral dos dados
            if numeric_df.shape[1] > 2:
                pca = PCA(n_components=2)
                # Lidar com possíveis valores ausentes para PCA
                numeric_df_filled = numeric_df.fillna(numeric_df.mean())
                pca_data = pca.fit_transform(StandardScaler().fit_transform(numeric_df_filled))
                
                plt.figure(figsize=(10, 8))
                if 'target' in df.columns:
                    scatter = plt.scatter(pca_data[:, 0], pca_data[:, 1], 
                                         c=df['target'] if 'target' in df.columns else 'blue',
                                         alpha=0.6, cmap='viridis')
                    plt.colorbar(scatter, label='Classe')
                else:
                    plt.scatter(pca_data[:, 0], pca_data[:, 1], alpha=0.6)
                
                plt.title(f'Visualização PCA - {title}')
                plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} var)')
                plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} var)')
                plt.tight_layout()
                plt.savefig(f"{output_path}/{title.lower().replace(' ', '_')}_pca.png")
        
        except Exception as e:
            print(f"Erro na análise de correlação: {e}")

def preprocess_for_model(df):
    """
    Prepara o dataset para uso em modelos, lidando com diferentes tipos de dados.
    """
    # Cria uma cópia para não modificar o original
    df_proc = df.copy()
    
    # Separa features categóricas para codificação one-hot
    cat_cols = df_proc.select_dtypes(include=['object', 'category']).columns
    if not cat_cols.empty:
        df_dummies = pd.get_dummies(df_proc[cat_cols], drop_first=True)
        df_proc = df_proc.drop(columns=cat_cols)
        df_proc = pd.concat([df_proc, df_dummies], axis=1)
    
    # Seleciona apenas colunas numéricas
    return df_proc.select_dtypes(include=['number'])

def evaluate_performance(X_train, y_train, X_test, y_test, title, model_params=None):
    """
    Avalia a performance de um RandomForest com os dados fornecidos.
    """
    start_time = time.time()
    
    params = {'n_estimators': 100, 'random_state': 42}
    if model_params:
        params.update(model_params)
    
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    
    train_time = time.time() - start_time
    
    # Avaliar no conjunto de treino
    y_train_pred = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    
    # Avaliar no conjunto de teste
    y_test_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    # Calcular tempo de predição
    pred_start = time.time()
    model.predict(X_test)
    pred_time = time.time() - pred_start
    
    # Relatório de performance
    print(f"\n=== Performance do Modelo ({title}) ===")
    print(f"Tempo de treino: {train_time:.2f} segundos")
    print(f"Tempo de predição: {pred_time:.4f} segundos")
    print(f"Acurácia no treino: {train_accuracy:.4f}")
    print(f"Acurácia no teste: {test_accuracy:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    print(f"Matriz de confusão:\n{cm}")
    
    # Relatório de classificação
    report = classification_report(y_test, y_test_pred)
    print(f"Relatório de classificação:\n{report}")
    
    # Extrair importância das features
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 features mais importantes:")
        for i, (_, row) in enumerate(feature_importance.head(10).iterrows()):
            print(f"  {i+1}. {row['feature']}: {row['importance']:.4f}")
    
    return {
        'model': model,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'train_time': train_time,
        'pred_time': pred_time,
        'feature_importance': feature_importance if hasattr(model, 'feature_importances_') else None
    }

def visualize_results(results, output_path):
    """
    Visualiza e compara os resultados das diferentes abordagens.
    """
    # Preparar dados para visualização
    methods = list(results.keys())
    train_accuracies = [results[method]['train_accuracy'] for method in methods]
    test_accuracies = [results[method]['test_accuracy'] for method in methods]
    train_times = [results[method]['train_time'] for method in methods]
    pred_times = [results[method]['pred_time'] for method in methods]
    n_features = [results[method]['n_features'] for method in methods]
    
    # Acurácia
    plt.figure(figsize=(12, 8))
    bar_width = 0.35
    index = np.arange(len(methods))
    
    plt.bar(index, train_accuracies, bar_width, label='Treino', color='skyblue')
    plt.bar(index + bar_width, test_accuracies, bar_width, label='Teste', color='orange')
    
    plt.xlabel('Método')
    plt.ylabel('Acurácia')
    plt.title('Comparação de Acurácia entre Métodos')
    plt.xticks(index + bar_width/2, methods)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_path}/accuracy_comparison.png")
    
    # Tempo de processamento
    plt.figure(figsize=(12, 8))
    plt.bar(index, train_times, bar_width, label='Tempo de treino', color='skyblue')
    plt.bar(index + bar_width, pred_times, bar_width, label='Tempo de predição', color='orange')
    
    plt.xlabel('Método')
    plt.ylabel('Tempo (segundos)')
    plt.title('Comparação de Tempo de Processamento')
    plt.xticks(index + bar_width/2, methods)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_path}/time_comparison.png")
    
    # Número de features
    plt.figure(figsize=(12, 8))
    plt.bar(index, n_features, color='green')
    
    plt.xlabel('Método')
    plt.ylabel('Número de Features')
    plt.title('Comparação do Número de Features')
    plt.xticks(index, methods)
    plt.tight_layout()
    plt.savefig(f"{output_path}/features_comparison.png")
    
    # Feature Importance para o método AutoFE
    if 'AutoFE' in results and results['AutoFE']['feature_importance'] is not None:
        top_features = results['AutoFE']['feature_importance'].head(15)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x='importance', y='feature', data=top_features, palette='viridis')
        plt.title('Top 15 Features Mais Importantes - AutoFE')
        plt.tight_layout()
        plt.savefig(f"{output_path}/autofe_feature_importance.png")
    
    # Resumo em texto
    with open(f"{output_path}/results_summary.txt", 'w') as f:
        f.write("=== RESUMO DOS RESULTADOS ===\n\n")
        
        f.write("Acurácia:\n")
        for i, method in enumerate(methods):
            f.write(f"  {method}: Treino = {train_accuracies[i]:.4f}, Teste = {test_accuracies[i]:.4f}\n")
        
        f.write("\nTempo de Processamento:\n")
        for i, method in enumerate(methods):
            f.write(f"  {method}: Treino = {train_times[i]:.2f}s, Predição = {pred_times[i]:.4f}s\n")
        
        f.write("\nNúmero de Features:\n")
        for i, method in enumerate(methods):
            f.write(f"  {method}: {n_features[i]}\n")
        
        # Ganho/perda de performance
        baseline_accuracy = results["Baseline"]["test_accuracy"]
        for method in methods:
            if method != "Baseline":
                accuracy_diff = (results[method]["test_accuracy"] - baseline_accuracy) * 100
                f.write(f"\nImpacto do {method} na performance: {accuracy_diff:.2f}% {'(ganho)' if accuracy_diff >= 0 else '(perda)'}\n")
        
        # Ganho/perda de eficiência computacional
        baseline_time = results["Baseline"]["train_time"] + results["Baseline"]["pred_time"]
        for method in methods:
            if method != "Baseline":
                method_time = results[method]["train_time"] + results[method]["pred_time"]
                time_diff = ((baseline_time - method_time) / baseline_time) * 100
                f.write(f"Impacto do {method} na eficiência: {time_diff:.2f}% {'(ganho)' if time_diff >= 0 else '(perda)'}\n")

def main():
    print("\n" + "="*80)
    print(" TESTE COMPLEXO DO SISTEMA AUTOFE ".center(80, "="))
    print("="*80 + "\n")
    
    # 1. Gerar dataset complexo
    df = generate_complex_dataset(
        n_samples=2000,
        n_features=80,
        n_informative=20,
        n_classes=5,
        correlation_level=0.85,
        noise_level=0.15,
        missing_rate=0.08,
        outlier_rate=0.05
    )
    
    # Salvar o dataset
    df.to_csv(f"{OUTPUT_DIR}/complex_dataset.csv", index=False)
    
    # 2. Analisar dataset original
    analyze_dataset(df, "Dataset Original", OUTPUT_DIR)
    
    # 3. Dividir em conjuntos de treinamento e teste
    # Convertemos para numérico antes da divisão
    df_numeric = preprocess_for_model(df)
    X = df_numeric.drop(columns=['target'])
    y = df_numeric['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # 4. Treinar e avaliar modelo base (sem nenhum processamento)
    print("\n=== AVALIAÇÃO DO MODELO BASELINE ===")
    results = {}
    
    # Lidar com valores ausentes para o baseline
    X_train_baseline = X_train.fillna(X_train.mean())
    X_test_baseline = X_test.fillna(X_train.mean())
    
    baseline_results = evaluate_performance(
        X_train_baseline, y_train, X_test_baseline, y_test, "Baseline"
    )
    baseline_results['n_features'] = X_train_baseline.shape[1]
    results['Baseline'] = baseline_results
    
    # 5. Processar com AutoFE
    try:
        print("\n=== AVALIAÇÃO DO MODELO COM AUTOFE ===")
        
        # Criar o Explorer
        explorer = Explorer(target_col="target")
        
        # Analisar as transformações e encontrar a melhor
        print("Iniciando análise de transformações...")
        start_time = time.time()
        
        # Usar o dataset preparado adequadamente (numérico)
        best_data = explorer.analyze_transformations(df_numeric)
        exploration_time = time.time() - start_time
        print(f"Tempo de exploração: {exploration_time:.2f} segundos")
        
        # Obter a configuração ótima
        best_config = explorer.get_best_pipeline_config()
        
        print("\n[RESULTADO] Melhor configuração encontrada:")
        print("  Parâmetros do Preprocessor:")
        for key, value in best_config.get('preprocessor_config', {}).items():
            print(f"    - {key}: {value}")
        
        print("\n  Parâmetros do FeatureEngineer:")
        for key, value in best_config.get('feature_engineer_config', {}).items():
            print(f"    - {key}: {value}")
        
        # Criar e ajustar um pipeline com a melhor configuração
        pipeline = DataPipeline(
            preprocessor_config=best_config.get('preprocessor_config', {}),
            feature_engineer_config=best_config.get('feature_engineer_config', {})
        )
        
        # Processar os dados de treino e teste
        X_train_df = X_train.copy()
        X_train_df['target'] = y_train.values
        pipeline.fit(X_train_df, target_col="target")
        
        X_train_processed = pipeline.transform(X_train_df, target_col="target")
        
        X_test_df = X_test.copy()
        X_test_df['target'] = y_test.values
        X_test_processed = pipeline.transform(X_test_df, target_col="target")
        
        # Remover target das features
        X_train_autofe = X_train_processed.drop(columns=["target"])
        X_test_autofe = X_test_processed.drop(columns=["target"])
        
        # Analisar dataset processado
        analyze_dataset(X_train_processed, "Dataset Processado", OUTPUT_DIR)
        
        # Avaliar modelo com dados processados
        autofe_results = evaluate_performance(
            X_train_autofe, y_train, X_test_autofe, y_test, "AutoFE"
        )
        autofe_results['n_features'] = X_train_autofe.shape[1]
        results['AutoFE'] = autofe_results
    
    except Exception as e:
        print(f"Erro ao processar com AutoFE: {e}")
        import traceback
        traceback.print_exc()
    
    # 6. Processar com um método simples apenas para comparação
    print("\n=== AVALIAÇÃO DO MODELO COM PREPROCESSAMENTO SIMPLES ===")
    
    # Preencher valores ausentes com a média
    X_train_simple = X_train.fillna(X_train.mean())
    X_test_simple = X_test.fillna(X_train.mean())
    
    # Normalizar os dados
    scaler = StandardScaler()
    X_train_simple_scaled = pd.DataFrame(
        scaler.fit_transform(X_train_simple),
        columns=X_train_simple.columns
    )
    X_test_simple_scaled = pd.DataFrame(
        scaler.transform(X_test_simple),
        columns=X_test_simple.columns
    )
    
    simple_results = evaluate_performance(
        X_train_simple_scaled, y_train, X_test_simple_scaled, y_test, "Preprocessamento Simples"
    )
    simple_results['n_features'] = X_train_simple_scaled.shape[1]
    results['Preprocessamento Simples'] = simple_results
    
    # 7. Visualizar e comparar resultados
    visualize_results(results, OUTPUT_DIR)
    
    # 8. Resumo final
    print("\n" + "="*80)
    print(" RESUMO FINAL ".center(80, "="))
    print("="*80)
    
    print("\nComparação de Acurácia:")
    for method, result in results.items():
        print(f"  {method}: {result['test_accuracy']:.4f}")
    
    print("\nComparação de Número de Features:")
    for method, result in results.items():
        print(f"  {method}: {result['n_features']}")
    
    print("\nComparação de Tempo de Treino:")
    for method, result in results.items():
        print(f"  {method}: {result['train_time']:.2f} segundos")
    
    if 'AutoFE' in results:
        baseline_score = results['Baseline']['test_accuracy']
        autofe_score = results['AutoFE']['test_accuracy']
        
        autofe_improvement = (autofe_score - baseline_score) * 100
        print(f"\nGanho de performance com AutoFE: {autofe_improvement:.2f}% " +
            f"({'melhoria' if autofe_improvement >= 0 else 'perda'})")
    
    print(f"\nArquivos de análise salvos em: {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()