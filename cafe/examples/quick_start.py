"""
Exemplo rápido de uso do CAFE (Component Automated Feature Engineer)
"""

import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Importar componentes do CAFE
from cafe import DataPipeline, Explorer

# Carregar dataset de exemplo
print("Carregando dataset de exemplo (Wine)...")
wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df['target'] = wine.target

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    df.drop('target', axis=1), 
    df['target'], 
    test_size=0.3, 
    random_state=42, 
    stratify=df['target']
)

# Reconstruir dataframes
train_df = X_train.copy()
train_df['target'] = y_train
test_df = X_test.copy()
test_df['target'] = y_test

print(f"Dataset carregado: {df.shape[0]} amostras, {df.shape[1]-1} features")
print(f"Conjunto de treino: {train_df.shape[0]} amostras")
print(f"Conjunto de teste: {test_df.shape[0]} amostras")

# Abordagem 1: Usar pipeline com configurações padrão
print("\n--- Abordagem 1: Pipeline com configurações padrão ---")
pipeline = DataPipeline()
train_transformed = pipeline.fit_transform(train_df, target_col='target')
test_transformed = pipeline.transform(test_df, target_col='target')

# Treinar modelo com dados transformados
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(train_transformed.drop('target', axis=1), train_transformed['target'])
preds = model.predict(test_transformed.drop('target', axis=1))
accuracy = accuracy_score(test_transformed['target'], preds)

print(f"Acurácia com pipeline padrão: {accuracy:.4f}")

# Abordagem 2: Usar Explorer para encontrar a melhor configuração
print("\n--- Abordagem 2: Usar Explorer para encontrar a melhor configuração ---")
explorer = Explorer(target_col='target')
best_data = explorer.analyze_transformations(train_df)
best_config = explorer.get_best_pipeline_config()

print("Melhor configuração encontrada:")
print(f"Preprocessor: {best_config.get('preprocessor_config', {})}")
print(f"FeatureEngineer: {best_config.get('feature_engineer_config', {})}")

# Criar pipeline otimizado
optimized_pipeline = DataPipeline(
    preprocessor_config=best_config.get('preprocessor_config', {}),
    feature_engineer_config=best_config.get('feature_engineer_config', {})
)

# Transformar dados
train_optimized = optimized_pipeline.fit_transform(train_df, target_col='target')
test_optimized = optimized_pipeline.transform(test_df, target_col='target')

# Treinar modelo com dados otimizados
model_opt = RandomForestClassifier(n_estimators=100, random_state=42)
model_opt.fit(train_optimized.drop('target', axis=1), train_optimized['target'])
preds_opt = model_opt.predict(test_optimized.drop('target', axis=1))
accuracy_opt = accuracy_score(test_optimized['target'], preds_opt)

print(f"Acurácia com pipeline otimizado: {accuracy_opt:.4f}")
print(f"Diferença de performance: {(accuracy_opt - accuracy) * 100:.2f}%")

# Comparação de número de features
print(f"\nFeatures originais: {df.shape[1]-1}")
print(f"Features após pipeline padrão: {train_transformed.shape[1]-1}")
print(f"Features após pipeline otimizado: {train_optimized.shape[1]-1}")

print("\nClassification Report (Pipeline Otimizado):")
print(classification_report(test_optimized['target'], preds_opt, target_names=wine.target_names))

# Salvar o pipeline otimizado
optimized_pipeline.save('wine_pipeline')
print("\nPipeline otimizado salvo como 'wine_pipeline'")