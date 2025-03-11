import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from scipy import stats
from sklearn.ensemble import IsolationForest

# Importar os componentes do AutoFE
from preprocessor import PreProcessor, create_preprocessor
from feature_engineer import FeatureEngineer, create_feature_engineer
from data_pipeline import DataPipeline, create_data_pipeline
from explorer import Explorer


# Configurar diretórios para resultados
OUTPUT_DIR = "autofe_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_sample_datasets():
    """
    Carrega um único dataset e divide em conjuntos de treinamento e inferência
    """
    # Carregar o dataset Wine
    wine = load_wine()
    wine_df = pd.DataFrame(wine.data, columns=wine.feature_names)
    wine_df['target'] = wine.target
    
    # Dividir o dataset em dois conjuntos (70% para treino, 30% para inferência)
    from sklearn.model_selection import train_test_split
    train_df, inference_df = train_test_split(
        wine_df, test_size=0.3, random_state=42, stratify=wine_df['target']
    )
    
    # Redefinir os índices para evitar problemas com índices duplicados
    train_df = train_df.reset_index(drop=True)
    inference_df = inference_df.reset_index(drop=True)
    
    print(f"Dataset de treinamento: {train_df.shape} amostras, {train_df.shape[1]-1} features, {train_df['target'].nunique()} classes")
    print(f"Dataset de inferência: {inference_df.shape} amostras, {inference_df.shape[1]-1} features, {inference_df['target'].nunique()} classes")
    
    return train_df, inference_df
    wine_df['target'] = wine.target
    
    # Dividir o dataset em dois conjuntos (70% para treino, 30% para inferência)
    from sklearn.model_selection import train_test_split
    train_df, inference_df = train_test_split(
        wine_df, test_size=0.3, random_state=42, stratify=wine_df['target']
    )
    
    # Redefinir os índices para evitar problemas com índices duplicados
    train_df = train_df.reset_index(drop=True)
    inference_df = inference_df.reset_index(drop=True)
    
    print(f"Dataset de treinamento (Wine - parte 1): {train_df.shape}")
    print(f"Dataset de inferência (Wine - parte 2): {inference_df.shape}")
    
    return train_df, inference_df

def explore_dataset(df, name):
    """
    Realiza uma exploração básica no dataset
    """
    print(f"\n=== Explorando dataset {name} ===")
    print(f"Formato: {df.shape}")
    print(f"Colunas: {df.columns.tolist()[:5]}... (total: {len(df.columns)})")
    print(f"Tipos de dados:\n{df.dtypes.value_counts()}")
    print(f"Valores nulos: {df.isnull().sum().sum()}")
    
    # Verificar distribuição de classes
    print(f"Distribuição de classes:")
    target_counts = df['target'].value_counts()
    for class_label, count in target_counts.items():
        print(f"  - Classe {class_label}: {count} amostras ({count/len(df)*100:.1f}%)")
    
    # Verificar estatísticas das features
    numeric_features = df.select_dtypes(include=['number']).drop(columns=['target'], errors='ignore')
    print(f"\nEstatísticas básicas das features numéricas:")
    stats_df = numeric_features.describe().T[['mean', 'std', 'min', 'max']]
    stats_df['variancia'] = numeric_features.var()
    stats_df['missing'] = numeric_features.isnull().sum()
    print(stats_df.head())
    
    # Calcular e mostrar correlações importantes
    corr_matrix = numeric_features.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr_pairs = [(corr_matrix.index[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]) 
                      for i, j in zip(*np.where(upper > 0.8))]
    
    if high_corr_pairs:
        print("\nPares de features altamente correlacionadas (>0.8):")
        for feat1, feat2, corr in sorted(high_corr_pairs, key=lambda x: x[2], reverse=True)[:5]:
            print(f"  - {feat1} e {feat2}: {corr:.4f}")
    
    # Salvar alguns gráficos exploratórios
    plt.figure(figsize=(12, 10))
    
    # Distribuição da variável alvo
    plt.subplot(2, 2, 1)
    ax = sns.countplot(x='target', data=df)
    for p in ax.patches:
        ax.annotate(f'{p.get_height()} ({p.get_height()/len(df)*100:.1f}%)', 
                   (p.get_x() + p.get_width()/2., p.get_height()), 
                   ha='center', va='bottom')
    plt.title(f'Distribuição de classes ({name})')
    
    # Histograma de algumas features numéricas
    plt.subplot(2, 2, 2)
    numeric_cols = df.select_dtypes(include=['number']).columns[:3]
    for col in numeric_cols:
        if col != 'target':
            sns.kdeplot(df[col], label=col)
    plt.title('Distribuição de algumas features')
    plt.legend()
    
    # Correlação com a variável alvo para algumas features
    plt.subplot(2, 2, 3)
    correlations = df.corr()['target'].sort_values(ascending=False)[1:6]
    correlations.plot(kind='bar')
    plt.title('Top 5 correlações com target')
    plt.xticks(rotation=45)
    
    # Análise de componentes principais para visualização 2D
    from sklearn.decomposition import PCA
    if len(numeric_features.columns) >= 2:
        plt.subplot(2, 2, 4)
        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(numeric_features)
        plt.scatter(pca_data[:, 0], pca_data[:, 1], c=df['target'], cmap='viridis', alpha=0.7)
        plt.colorbar(label='Classe')
        plt.title('Visualização PCA 2D')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} var)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} var)')
    else:
        plt.subplot(2, 2, 4)
        sns.boxplot(data=df[numeric_cols])
        plt.title('Boxplot para detecção de outliers')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/exploracao_{name}.png")
    
    return df

def find_best_configuration(train_df):
    """
    Usa o Explorer para encontrar a melhor configuração do pipeline
    """
    print("Buscando a melhor configuração para o pipeline AutoFE...")
    
    # Criar um Explorer com a coluna alvo especificada
    explorer = Explorer(target_col="target")
    
    # Analisar as transformações e encontrar a melhor
    print("Iniciando análise de transformações (pode levar alguns minutos)...")
    best_data = explorer.analyze_transformations(train_df)
    
    # Obter a configuração ótima
    best_config = explorer.get_best_pipeline_config()
    
    print("\n[RESULTADO] Melhor configuração encontrada:")
    print("  Parâmetros do Preprocessor:")
    for key, value in best_config.get('preprocessor_config', {}).items():
        print(f"    - {key}: {value}")
    
    print("\n  Parâmetros do FeatureEngineer:")
    for key, value in best_config.get('feature_engineer_config', {}).items():
        print(f"    - {key}: {value}")
    
    # Salvar a configuração como JSON
    import json
    with open(f"{OUTPUT_DIR}/best_config.json", "w") as f:
        json.dump(best_config, f, indent=4)
    print(f"\nConfiguração salva em {OUTPUT_DIR}/best_config.json")
    
    return best_config, best_data

def create_and_train_model(X_train, y_train, X_test, y_test):
    """
    Cria e treina um modelo de classificação usando os dados processados
    """
    print("Treinando modelo com dados transformados...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Avaliar no conjunto de teste
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n[RESULTADO] Modelo com dados transformados pelo AutoFE:")
    print(f"  - Acurácia: {accuracy:.4f}")
    
    # Extrair e mostrar as features mais importantes
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 features mais importantes:")
        for i, (_, row) in enumerate(feature_importance.head(10).iterrows()):
            print(f"  {i+1}. {row['feature']}: {row['importance']:.4f}")
    
    print("\nRelatório de classificação:")
    print(classification_report(y_test, y_pred))
    
    return model, accuracy

def compare_datasets(original_df, transformed_df, name):
    """
    Compara os datasets original e transformado
    """
    print(f"\n[ANÁLISE] Comparação dos datasets ({name.upper()}):")
    print(f"  - Formato original: {original_df.shape}")
    print(f"  - Formato transformado: {transformed_df.shape}")
    print(f"  - Expansão de dimensionalidade: {transformed_df.shape[1] / original_df.shape[1]:.1f}x")
    
    # Comparar estatísticas descritivas das features numéricas
    if 'target' in original_df.columns:
        orig_numeric = original_df.drop(columns=['target']).select_dtypes(include=['number'])
    else:
        orig_numeric = original_df.select_dtypes(include=['number'])
    
    if 'target' in transformed_df.columns:
        trans_numeric = transformed_df.drop(columns=['target']).select_dtypes(include=['number'])
    else:
        trans_numeric = transformed_df.select_dtypes(include=['number'])
    
    print(f"\n  Estatísticas dos datasets:")
    print(f"    - Variância média (original): {orig_numeric.var().mean():.4f}")
    print(f"    - Variância média (transformado): {trans_numeric.var().mean():.4f}")
    
    # Calcular e exibir a distribuição de correlações
    orig_corr = orig_numeric.corr().abs()
    orig_corr_upper = orig_corr.where(np.triu(np.ones(orig_corr.shape), k=1).astype(bool))
    high_corr_orig = (orig_corr_upper > 0.8).sum().sum()
    
    if trans_numeric.shape[1] > 1:
        trans_corr = trans_numeric.corr().abs()
        trans_corr_upper = trans_corr.where(np.triu(np.ones(trans_corr.shape), k=1).astype(bool))
        high_corr_trans = (trans_corr_upper > 0.8).sum().sum()
        print(f"    - Pares correlacionados >0.8 (original): {high_corr_orig}")
        print(f"    - Pares correlacionados >0.8 (transformado): {high_corr_trans}")
    
    # Visualizar as diferenças na distribuição
    plt.figure(figsize=(15, 10))
    
    # Correlação entre features originais
    plt.subplot(2, 2, 1)
    sns.heatmap(orig_corr, cmap='Blues', vmin=0, vmax=1, annot=False)
    plt.title(f'Correlação original ({name})')
    
    # Correlação entre features transformadas
    plt.subplot(2, 2, 2)
    if trans_numeric.shape[1] > 1:  # Verifica se há mais de uma coluna
        sns.heatmap(trans_corr, cmap='Blues', vmin=0, vmax=1, annot=False)
        plt.title(f'Correlação transformada ({name})')
    else:
        plt.text(0.5, 0.5, "Insuficiente para matriz de correlação", 
                 horizontalalignment='center', verticalalignment='center')
    
    # Histograma da distribuição das features originais e transformadas
    plt.subplot(2, 2, 3)
    orig_numeric_sample = orig_numeric.sample(min(5, orig_numeric.shape[1]), axis=1)
    for col in orig_numeric_sample.columns:
        sns.kdeplot(orig_numeric_sample[col], label=col)
    plt.title(f"Distribuição de features originais (amostra)")
    plt.legend()
    
    plt.subplot(2, 2, 4)
    if trans_numeric.shape[1] > 0:
        trans_numeric_sample = trans_numeric.sample(min(5, trans_numeric.shape[1]), axis=1)
        for col in trans_numeric_sample.columns:
            sns.kdeplot(trans_numeric_sample[col], label=col)
        plt.title(f"Distribuição de features transformadas (amostra)")
        plt.legend()
    else:
        plt.text(0.5, 0.5, "Sem features numéricas para exibir", 
                 horizontalalignment='center', verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/comparacao_{name}.png")
    
    return orig_numeric, trans_numeric

def main():
    print("=" * 80)
    print("               AVALIAÇÃO DE PERFORMANCE DO SISTEMA AutoFE")
    print("=" * 80)
    
    # 1. Carregar os datasets
    print("\n[FASE 1] PREPARAÇÃO DOS DADOS")
    print("-" * 70)
    train_dataset, inference_dataset = load_sample_datasets()
    
    # 2. Explorar os datasets
    explore_dataset(train_dataset, "treino")
    explore_dataset(inference_dataset, "inferencia")
    
    # 3. Encontrar a melhor configuração usando o dataset de treino
    print("\n[FASE 2] OTIMIZAÇÃO DE CONFIGURAÇÃO COM AUTOFE")
    print("-" * 70)
    best_config, best_data = find_best_configuration(train_dataset)
    
    # 4. Dividir o dataset de treino para avaliar o desempenho
    X = train_dataset.drop(columns=['target'])
    y = train_dataset['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 5. Criar e ajustar um pipeline com a melhor configuração
    pipeline = create_data_pipeline(
        preprocessor_config=best_config.get('preprocessor_config', {}),
        feature_engineer_config=best_config.get('feature_engineer_config', {})
    )
    
    # 6. Ajustar o pipeline e transformar os dados de treino
    print("\n[FASE 3] APLICAÇÃO DO PIPELINE DE TRANSFORMAÇÃO")
    print("-" * 70)
    pipeline.fit(X_train, target_col=None)  # Não temos target aqui pois já separamos X e y
    X_train_transformed = pipeline.transform(X_train)
    X_test_transformed = pipeline.transform(X_test)
    
    # 7. Salvar o pipeline
    pipeline_path = f"{OUTPUT_DIR}/best_pipeline"
    pipeline.save(pipeline_path)
    print(f"Pipeline salvo em {pipeline_path}_*.pkl")
    
    # 8. Treinar um modelo com os dados transformados
    print("\n[FASE 4] AVALIAÇÃO DE PERFORMANCE COM MODELO TRANSFORMADO")
    print("-" * 70)
    # Ajustar y_train e y_test para corresponder aos índices de X_train_transformed e X_test_transformed
    y_train_aligned = y_train.loc[X_train_transformed.index]
    y_test_aligned = y_test.loc[X_test_transformed.index]
    
    model, train_accuracy = create_and_train_model(
        X_train_transformed, y_train_aligned, X_test_transformed, y_test_aligned
    )
    
    # 9. Carregar o pipeline salvo
    print("\n[FASE 5] TESTE DE PERSISTÊNCIA E CARREGAMENTO")
    print("-" * 70)
    print("Carregando o pipeline salvo...")
    loaded_pipeline = DataPipeline.load(pipeline_path)
    
    # 10. Aplicar o pipeline carregado ao dataset de inferência
    print("\n=== Aplicando o pipeline carregado ao dataset de inferência ===")
    X_inference = inference_dataset.drop(columns=['target'])
    y_inference = inference_dataset['target']
    
    X_inference_transformed = loaded_pipeline.transform(X_inference)
    
    # 11. Avaliar o modelo no dataset de inferência
    print("\n[FASE 6] VALIDAÇÃO EM DADOS DE INFERÊNCIA")
    print("-" * 70)
    # Ajustar y_inference para corresponder aos índices de X_inference_transformed
    y_inference_aligned = y_inference.loc[X_inference_transformed.index]
    
    y_inference_pred = model.predict(X_inference_transformed)
    inference_accuracy = accuracy_score(y_inference_aligned, y_inference_pred)
    
    print(f"\n[RESULTADO] Performance no conjunto de inferência:")
    print(f"  - Acurácia: {inference_accuracy:.4f}")
    
    print("\nMatriz de confusão:")
    from sklearn.metrics import confusion_matrix
    confusion = confusion_matrix(y_inference_aligned, y_inference_pred)
    print(confusion)
    
    print("\nRelatório de classificação (inferência):")
    print(classification_report(y_inference_aligned, y_inference_pred))
    
    # 12. Comparar os datasets originais e transformados
    compare_datasets(X_train, X_train_transformed, "treino")
    compare_datasets(X_inference, X_inference_transformed, "inferencia")
    
    # 13. Resumo final
    print("\n" + "=" * 80)
    print("                       RESUMO DA ANÁLISE DO AUTOFE")
    print("=" * 80)
    
    print(f"\n[1] TRANSFORMAÇÃO DE DADOS:")
    print(f"  - Dataset de treino: {X_train.shape} → {X_train_transformed.shape}")
    print(f"  - Dataset de inferência: {X_inference.shape} → {X_inference_transformed.shape}")
    
    print(f"\n[2] PERFORMANCE DO MODELO COM AUTOFE:")
    print(f"  - Acurácia no conjunto de teste: {train_accuracy:.4f}")
    print(f"  - Acurácia na inferência: {inference_accuracy:.4f}")
    print(f"  - Diferença de performance: {(inference_accuracy - train_accuracy)*100:.2f}%")
    
    # 14. Comparação com um modelo baseline treinado sem transformações
    print("\n[3] COMPARAÇÃO COM MODELO BASELINE (SEM AUTOFE):")
    baseline_model = RandomForestClassifier(n_estimators=100, random_state=42)
    baseline_model.fit(X_train, y_train)
    baseline_score = baseline_model.score(X_test, y_test)
    baseline_inference = baseline_model.score(X_inference, y_inference)
    
    print(f"  - Baseline - Acurácia no teste: {baseline_score:.4f}")
    print(f"  - Baseline - Acurácia na inferência: {baseline_inference:.4f}")
    
    # Calcular e exibir o ganho/perda de performance
    test_diff = (train_accuracy - baseline_score)*100
    inference_diff = (inference_accuracy - baseline_inference)*100
    
    print("\n[4] IMPACTO DO AUTOFE NA PERFORMANCE:")
    print(f"  - No conjunto de teste: {test_diff:.2f}% {'(ganho)' if test_diff >= 0 else '(perda)'}")
    print(f"  - Na inferência: {inference_diff:.2f}% {'(ganho)' if inference_diff >= 0 else '(perda)'}")
    
    # Avaliar o impacto computacional
    print("\n[5] IMPACTO COMPUTACIONAL:")
    print(f"  - Aumento de dimensionalidade: {(X_train_transformed.shape[1] / X_train.shape[1]):.1f}x")
    print(f"  - Número de features original: {X_train.shape[1]}")
    print(f"  - Número de features após AutoFE: {X_train_transformed.shape[1]}")
    
    # Conclusão
    print("\n[6] CONCLUSÃO:")
    if inference_diff > 1.0:
        conclusion = "O AutoFE melhorou significativamente a performance do modelo"
    elif inference_diff > 0:
        conclusion = "O AutoFE trouxe uma pequena melhoria na performance"
    elif inference_diff >= -1.0:
        conclusion = "O AutoFE não alterou significativamente a performance"
    else:
        conclusion = "O AutoFE prejudicou a performance do modelo"
        
    print(f"  - {conclusion}")
    print(f"  - Recomendação: {'Utilizar o AutoFE' if inference_diff >= 0 else 'Utilizar o modelo baseline (mais simples)'}")
    
    print(f"\nArquivos de análise salvos em: {OUTPUT_DIR}/")
    
    # Restaurar o método original de remoção de outliers
    from preprocessor import PreProcessor
    PreProcessor._remove_outliers = original_remove_outliers_fn

if __name__ == "__main__":
    main()