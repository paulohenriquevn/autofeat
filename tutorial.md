# Tutorial CAFE (Component Automated Feature Engineer)

Este notebook demonstra como usar o CAFE para automatizar o processo de engenharia de features e preparação de dados para machine learning. Seguiremos um fluxo de trabalho completo desde a aquisição dos dados até a construção e avaliação do modelo.

## 0. Instalação e Importações

Primeiro, vamos instalar o CAFE e importar as bibliotecas necessárias.


```python
# Instalar CAFE (descomente para instalar)
# !pip install cafe-autofe

# Importações
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, r2_score, mean_squared_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Importações do CAFE
from cafe import (
    PreProcessor, 
    FeatureEngineer, 
    PerformanceValidator, 
    DataPipeline, 
    Explorer
)

# Configuração de visualização
plt.style.use('ggplot')
sns.set(style="whitegrid")
%matplotlib inline
```

## 1. Aquisição de Dados

Vamos carregar um conjunto de dados para nosso exemplo. Você pode substituir este código para carregar seus próprios dados.


```python
# Opção 1: Carregar um dataset de exemplo do scikit-learn
from sklearn.datasets import load_wine
wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df['target'] = wine.target
task_type = 'classification'
target_col = 'target'

# Opção 2: Carregar dados de um arquivo CSV (descomente e ajuste conforme necessário)
# df = pd.read_csv('seu_arquivo.csv')
# task_type = 'classification'  # ou 'regression' dependendo do seu problema
# target_col = 'nome_da_coluna_alvo'

print(f"Dataset carregado: {df.shape[0]} amostras, {df.shape[1]} colunas")
df.head()
```

## 2. Exploração e Análise de Dados

### 2.1 Exame dos Dados


```python
# Informações gerais sobre o DataFrame
print("Informações do DataFrame:")
print(f"Linhas: {df.shape[0]}, Colunas: {df.shape[1]}")
print("\nTipos de dados:")
print(df.dtypes)

# Estatísticas descritivas
print("\nEstatísticas descritivas:")
df.describe().round(2)
```

### 2.2 Análise da Variável Alvo


```python
# Para classificação
if task_type == 'classification':
    plt.figure(figsize=(8, 5))
    sns.countplot(x=target_col, data=df)
    plt.title('Distribuição das Classes')
    plt.ylabel('Contagem')
    plt.show()
    
    print(f"Contagem de classes:\n{df[target_col].value_counts()}")
    
# Para regressão
elif task_type == 'regression':
    plt.figure(figsize=(10, 6))
    sns.histplot(df[target_col], kde=True)
    plt.title('Distribuição da Variável Alvo')
    plt.xlabel(target_col)
    plt.ylabel('Frequência')
    plt.show()
    
    print(f"Estatísticas da variável alvo:\n{df[target_col].describe()}")
```

### 2.3 Verificação de Valores Ausentes


```python
# Verificar valores ausentes
missing_values = df.isnull().sum()
missing_percent = (missing_values / len(df)) * 100

# Criar DataFrame com informações sobre valores ausentes
missing_info = pd.DataFrame({
    'Valores Ausentes': missing_values,
    'Percentual (%)': missing_percent.round(2)
})

# Exibir apenas colunas com valores ausentes
if missing_values.sum() > 0:
    print("Colunas com valores ausentes:")
    display(missing_info[missing_info['Valores Ausentes'] > 0])
else:
    print("Não há valores ausentes no dataset!")
```

### 2.4 Exploração de Correlações


```python
# Matriz de correlação
plt.figure(figsize=(12, 10))
corr_matrix = df.corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', annot=False, 
            center=0, square=True, linewidths=.5)
plt.title('Matriz de Correlação')
plt.tight_layout()
plt.show()

# Correlações com a variável alvo
if target_col in df.columns:
    target_corr = corr_matrix[target_col].sort_values(ascending=False)
    
    # Criar gráfico de barras para correlações com o target
    plt.figure(figsize=(10, 6))
    target_corr.drop(target_col).plot(kind='bar')
    plt.title(f'Correlação com {target_col}')
    plt.ylabel('Coeficiente de Correlação')
    plt.tight_layout()
    plt.show()
```

### 2.5 Visualização de Recursos Principais


```python
# Selecionar as features mais correlacionadas com o target
if target_col in df.columns:
    top_features = target_corr.drop(target_col).abs().sort_values(ascending=False).head(5).index.tolist()
    
    # Visualizar distribuição das principais features
    plt.figure(figsize=(15, 10))
    for i, feature in enumerate(top_features, 1):
        plt.subplot(2, 3, i)
        
        # Para classificação
        if task_type == 'classification':
            for target_value in df[target_col].unique():
                sns.kdeplot(df[df[target_col] == target_value][feature], 
                            label=f'Classe {target_value}')
            plt.title(f'Distribuição de {feature} por Classe')
            plt.legend()
            
        # Para regressão
        else:
            plt.scatter(df[feature], df[target_col], alpha=0.5)
            plt.title(f'{feature} vs {target_col}')
            plt.xlabel(feature)
            plt.ylabel(target_col)
            
    plt.tight_layout()
    plt.show()
```

## 3. Pré-processamento com CAFE

Agora vamos usar o CAFE para automatizar o pré-processamento de dados. O CAFE fornece uma abordagem modular para todas as etapas necessárias.

### 3.1 Configuração Básica do PreProcessor


```python
# Configuração para o preprocessador
preprocessor_config = {
    'missing_values_strategy': 'median',  # Estratégia para valores ausentes
    'outlier_method': 'iqr',              # Método para tratamento de outliers
    'categorical_strategy': 'onehot',     # Estratégia para codificação de variáveis categóricas
    'scaling': 'standard',                # Método de normalização/padronização
    'verbosity': 1                        # Nível de detalhamento dos logs
}

# Criar e aplicar o preprocessador
preprocessor = PreProcessor(preprocessor_config)
df_preprocessed = preprocessor.fit_transform(df, target_col=target_col)

print(f"DataFrame original: {df.shape}")
print(f"DataFrame pré-processado: {df_preprocessed.shape}")
df_preprocessed.head()
```

### 3.2 Configuração da Engenharia de Features


```python
# Configuração para o engenheiro de features
feature_engineer_config = {
    'correlation_threshold': 0.8,     # Limiar para remoção de features altamente correlacionadas
    'generate_features': True,        # Gerar features polinomiais
    'feature_selection': 'kbest',     # Método de seleção de features
    'feature_selection_params': {     # Parâmetros específicos para a seleção de features
        'k': 10                       # Número de features a selecionar
    },
    'task': task_type,                # Tipo de tarefa (classificação ou regressão)
    'verbosity': 1                    # Nível de detalhamento dos logs
}

# Criar e aplicar o engenheiro de features
feature_engineer = FeatureEngineer(feature_engineer_config)
df_engineered = feature_engineer.fit_transform(df_preprocessed, target_col=target_col)

print(f"DataFrame pré-processado: {df_preprocessed.shape}")
print(f"DataFrame após engenharia de features: {df_engineered.shape}")
df_engineered.head()
```

### 3.3 Validação de Performance


```python
# Configuração do validador de performance
validator_config = {
    'max_performance_drop': 0.05,  # Máxima queda de performance permitida (5%)
    'cv_folds': 5,                 # Número de folds para validação cruzada
    'metric': 'accuracy' if task_type == 'classification' else 'r2',
    'task': task_type,             # Tipo de tarefa
    'verbose': True                # Mostrar logs detalhados
}

# Separar features e target
X_original = df.drop(columns=[target_col])
X_engineered = df_engineered.drop(columns=[target_col])
y = df[target_col]

# Criar e aplicar o validador
validator = PerformanceValidator(validator_config)
validation_results = validator.evaluate(X_original, y, X_engineered)

# Mostrar resultados da validação
print("\nResultados da Validação:")
print(f"Performance dataset original: {validation_results['performance_original']:.4f}")
print(f"Performance dataset transformado: {validation_results['performance_transformed']:.4f}")
print(f"Diferença: {validation_results['performance_diff_pct']:.2f}%")
print(f"Melhor dataset: {validation_results['best_choice'].upper()}")
print(f"Redução de features: {validation_results['feature_reduction']*100:.1f}%")
```

### 3.4 Usando o Pipeline Completo


```python
# Criar pipeline completo com os componentes configurados
pipeline = DataPipeline(
    preprocessor_config=preprocessor_config,
    feature_engineer_config=feature_engineer_config,
    validator_config=validator_config,
    auto_validate=True  # Ativar validação automática
)

# Aplicar pipeline completo
df_transformed = pipeline.fit_transform(df, target_col=target_col)

print(f"Dataset original: {df.shape}")
print(f"Dataset transformado: {df_transformed.shape}")
df_transformed.head()
```

### 3.5 Usando o Explorer para Otimização Automática


```python
# Criar e aplicar o Explorer para encontrar a melhor configuração
explorer = Explorer(target_col=target_col)
best_data = explorer.analyze_transformations(df)

# Obter a configuração ótima
best_config = explorer.get_best_pipeline_config()

print("Melhor configuração encontrada pelo Explorer:")
print("\nConfiguração do Preprocessador:")
for key, value in best_config.get('preprocessor_config', {}).items():
    print(f"- {key}: {value}")
    
print("\nConfiguração do Engenheiro de Features:")
for key, value in best_config.get('feature_engineer_config', {}).items():
    print(f"- {key}: {value}")

# Visualizar árvore de transformações
explorer.visualize_transformations()

# Estatísticas das transformações
transformation_stats = explorer.get_transformation_statistics()
print("\nEstatísticas das Transformações:")
for key, value in transformation_stats.items():
    if not isinstance(value, (list, dict)):
        print(f"- {key}: {value}")

# Criar pipeline com a configuração ótima
optimal_pipeline = explorer.create_optimal_pipeline()
df_optimal = optimal_pipeline.fit_transform(df, target_col=target_col)

print(f"\nDataset original: {df.shape}")
print(f"Dataset com transformação ótima: {df_optimal.shape}")
```

## 4. Construção de Modelo

Agora que temos nossos dados pré-processados e com features otimizadas, vamos construir e treinar modelos.

### 4.1 Divisão em Treino e Teste


```python
# Usar o dataset transformado pelo pipeline ótimo
X = df_optimal.drop(columns=[target_col])
y = df_optimal[target_col]

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y if task_type == 'classification' else None
)

print(f"Conjunto de treino: {X_train.shape}")
print(f"Conjunto de teste: {X_test.shape}")
```

### 4.2 Treinamento e Avaliação do Modelo


```python
# Escolher o modelo adequado com base no tipo de tarefa
if task_type == 'classification':
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Treinar o modelo
    model.fit(X_train, y_train)
    
    # Avaliar no conjunto de teste
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Acurácia no conjunto de teste: {accuracy:.4f}")
    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred))
    
else:  # Regressão
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Treinar o modelo
    model.fit(X_train, y_train)
    
    # Avaliar no conjunto de teste
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"R² no conjunto de teste: {r2:.4f}")
    print(f"RMSE no conjunto de teste: {rmse:.4f}")
```

### 4.3 Importância das Features


```python
# Obter importância das features
feature_importances = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

# Visualizar importância das features
plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importances.head(15))
plt.title('Importância das Features')
plt.tight_layout()
plt.show()

# Exibir tabela com importância das features
feature_importances.head(15)
```

## 5. Salvando e Carregando o Pipeline

Demonstração de como salvar e carregar o pipeline para uso futuro.


```python
# Salvar o pipeline otimizado
optimal_pipeline.save('optimal_pipeline')
print("Pipeline salvo com sucesso!")

# Carregar o pipeline (em um novo projeto ou sessão)
loaded_pipeline = DataPipeline.load('optimal_pipeline')
print("Pipeline carregado com sucesso!")

# Verificar se o pipeline carregado funciona corretamente
df_new = df.copy()  # Simular novos dados
df_new_transformed = loaded_pipeline.transform(df_new, target_col=target_col)
print(f"Novos dados transformados: {df_new_transformed.shape}")
```

## 6. Fluxo de Trabalho Completo com CAFE

Aqui está um exemplo de fluxo de trabalho completo, resumindo todas as etapas anteriores em um processo simplificado.


```python
def complete_ml_workflow(df, target_col, task_type='classification'):
    """
    Fluxo de trabalho completo usando CAFE para automatizar
    pré-processamento, engenharia de features e modelagem.
    
    Args:
        df: DataFrame com os dados
        target_col: Nome da coluna alvo
        task_type: 'classification' ou 'regression'
        
    Returns:
        model: Modelo treinado
        pipeline: Pipeline CAFE otimizado
    """
    # 1. Explorar e entender os dados
    print(f"Dataset original: {df.shape[0]} linhas x {df.shape[1]} colunas")
    
    # 2. Usar Explorer para encontrar a melhor configuração
    explorer = Explorer(target_col=target_col)
    _ = explorer.analyze_transformations(df)
    best_config = explorer.get_best_pipeline_config()
    
    print("\nMelhor configuração encontrada pelo Explorer.")
    
    # 3. Criar pipeline otimizado
    pipeline = DataPipeline(
        preprocessor_config=best_config.get('preprocessor_config', {}),
        feature_engineer_config=best_config.get('feature_engineer_config', {}),
        auto_validate=True
    )
    
    # 4. Transformar os dados
    df_transformed = pipeline.fit_transform(df, target_col=target_col)
    print(f"Dataset transformado: {df_transformed.shape[0]} linhas x {df_transformed.shape[1]} colunas")
    
    # 5. Dividir em treino e teste
    X = df_transformed.drop(columns=[target_col])
    y = df_transformed[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, 
        stratify=y if task_type == 'classification' else None
    )
    
    # 6. Treinar modelo
    if task_type == 'classification':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Avaliar performance
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nAcurácia no conjunto de teste: {accuracy:.4f}")
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Avaliar performance
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        print(f"\nR² no conjunto de teste: {r2:.4f}")
    
    # 7. Salvar pipeline para uso futuro
    pipeline.save('cafe_pipeline')
    print("\nPipeline salvo como 'cafe_pipeline'")
    
    return model, pipeline

# Executar o fluxo de trabalho completo
model, pipeline = complete_ml_workflow(df, target_col, task_type)
```

## 7. Usando o Pipeline para Novas Previsões

Demonstração de como usar o pipeline e modelo salvo para fazer previsões em novos dados.


```python
def predict_with_cafe_pipeline(new_data, target_col=None, pipeline_path='cafe_pipeline'):
    """
    Usa um pipeline CAFE salvo para transformar novos dados e fazer previsões.
    
    Args:
        new_data: DataFrame com novos dados
        target_col: Nome da coluna alvo (opcional, para preservação)
        pipeline_path: Caminho do pipeline salvo
        
    Returns:
        DataFrame com os dados transformados
    """
    # Carregar o pipeline
    pipeline = DataPipeline.load(pipeline_path)
    print("Pipeline carregado com sucesso!")
    
    # Transformar os novos dados
    transformed_data = pipeline.transform(new_data, target_col=target_col)
    print(f"Dados transformados: {transformed_data.shape}")
    
    return transformed_data

# Simular novos dados (usando o mesmo DataFrame para demonstração)
new_data = df.copy()
transformed_data = predict_with_cafe_pipeline(new_data, target_col=target_col)

# Fazer previsões com o modelo treinado
if target_col in transformed_data.columns:
    X_new = transformed_data.drop(columns=[target_col])
else:
    X_new = transformed_data
    
predictions = model.predict(X_new)
print(f"Previsões feitas para {len(predictions)} amostras.")

# Ver algumas previsões
pd.DataFrame({
    'Previsão': predictions[:10],
    'Real (se disponível)': new_data[target_col].values[:10] if target_col in new_data.columns else ['N/A'] * 10
})
```

## 8. Conclusão

Neste tutorial, você aprendeu como usar o CAFE (Component Automated Feature Engineer) para automatizar o processo de preparação de dados e engenharia de features em um projeto de machine learning. Vimos como:

1. Explorar e analisar os dados
2. Usar o PreProcessor para pré-processamento automático
3. Aplicar o FeatureEngineer para melhorar as features
4. Validar as transformações com o PerformanceValidator
5. Integrar tudo com o DataPipeline
6. Usar o Explorer para encontrar a configuração ideal
7. Construir, treinar e avaliar modelos
8. Salvar e carregar o pipeline para uso futuro

O CAFE fornece uma abordagem estruturada e automatizada para um dos aspectos mais demorados e críticos do machine learning: a engenharia de features. Ao automatizar essas etapas, você pode se concentrar mais na compreensão do problema e na interpretação dos resultados.