"""
Script de demonstração avançada do AutoFE com validação de performance.
Inclui datasets mais complexos para simular casos reais de uso.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
from datetime import datetime, timedelta
from sklearn.datasets import (
    load_iris, load_wine, load_breast_cancer, load_diabetes, 
    fetch_california_housing, make_classification, make_regression
)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

# Importar o pipeline com validação
from cafe.data_pipeline import create_data_pipeline

# Criar diretório para resultados
OUTPUT_DIR = "autofe_demo_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def create_complex_timeseries_dataset():
    """
    Cria um dataset financeiro complexo com características temporais, 
    categóricas e numéricas, garantindo que as datas sejam adequadamente processadas.
    
    Returns:
        pandas.DataFrame: Dataset complexo com diferentes tipos de dados
        str: Tipo de tarefa ('classification' ou 'regression')
        list: Nomes das classes alvo
    """
    import pandas as pd
    import numpy as np
    
    # Número de amostras
    n_samples = 1000
    
    # Criar DataFrame base, mas sem incluir a coluna de data diretamente
    df = pd.DataFrame()
    
    # 1. Criar datas sequenciais e extrair características de tempo manualmente
    dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')
    
    # 2. Extrair características temporais da data em formato numérico
    df['day'] = [d.day for d in dates]
    df['month'] = [d.month for d in dates]
    df['year'] = [d.year for d in dates]
    df['day_of_week'] = [d.dayofweek for d in dates]
    df['quarter'] = [d.quarter for d in dates]
    df['is_weekend'] = [(1 if d.dayofweek >= 5 else 0) for d in dates]
    
    # Não incluímos a coluna 'date' para evitar problemas com Timestamps
    # Apenas salvar para referência
    reference_dates = dates
    
    # 3. Gerar valores base para preços (tendência + ruído)
    base_value = np.linspace(100, 200, n_samples) + np.random.normal(0, 10, n_samples)
    
    # 4. Criar preços com base no valor base (correlacionados)
    df['price_open'] = base_value + np.random.normal(0, 5, n_samples)
    df['price_high'] = df['price_open'] + np.abs(np.random.normal(0, 8, n_samples))
    df['price_low'] = df['price_open'] - np.abs(np.random.normal(0, 6, n_samples))
    df['price_close'] = df['price_open'] + np.random.normal(0, 7, n_samples)
    df['volume'] = df['price_close'] * (1 + np.random.normal(0, 0.2, n_samples)) * 1000
    
    # 5. Criar variáveis categóricas baseadas nos preços
    df['market_condition'] = np.select(
        [df['price_close'] > df['price_open'], 
         df['price_close'] < df['price_open'], 
         df['price_close'] == df['price_open']],
        ['bull', 'bear', 'neutral'],
        default='unknown'
    )
    
    df['volatility'] = np.select(
        [(df['price_high'] - df['price_low']) < 5, 
         (df['price_high'] - df['price_low']) < 10, 
         (df['price_high'] - df['price_low']) >= 10],
        ['low', 'medium', 'high'],
        default='unknown'
    )
    
    # 6. Calcular indicadores técnicos
    # Médias móveis - fazer de forma iterativa para garantir alinhamento
    df['sma_5'] = np.nan
    df['sma_10'] = np.nan
    
    # Preencher valores de SMA de forma explícita para evitar problemas de tamanho
    for i in range(4, n_samples):
        df.loc[i, 'sma_5'] = df['price_close'][i-4:i+1].mean()
        
    for i in range(9, n_samples):
        df.loc[i, 'sma_10'] = df['price_close'][i-9:i+1].mean()
        
    # RSI - indicador de força relativa simplificado
    df['rsi'] = np.random.uniform(0, 100, n_samples)
    
    # 7. Criar target (predição de movimento em 5 dias)
    # Inicializar com NaN
    df['future_5d_pct_change'] = np.nan
    
    # Calcular a variação percentual em 5 dias, deixando os últimos 5 como NaN
    for i in range(0, n_samples - 5):
        df.loc[i, 'future_5d_pct_change'] = (
            (df['price_close'][i + 5] - df['price_close'][i]) / df['price_close'][i] * 100
        )
    
    # 8. Classificar o target em 3 categorias
    df['target'] = np.select(
        [df['future_5d_pct_change'] < -2, df['future_5d_pct_change'] > 2],
        [0, 2],
        default=1  # 0: queda forte, 1: estável, 2: alta forte
    )
    
    # 9. Remover linhas com valores ausentes
    # Remover os últimos 5 dias (onde não temos target) e quaisquer NaN
    df_clean = df.iloc[:-5].dropna()
    
    # Verificar a consistência do dataset
    # Garantir que temos o mesmo número de elementos em cada coluna
    assert all(len(df_clean[col]) == len(df_clean) for col in df_clean.columns), "Colunas têm tamanhos diferentes"
    
    return df_clean, 'classification', ['Queda', 'Estável', 'Alta']

def create_noisy_data_dataset():
    """
    Cria um dataset com valores ausentes e outliers de forma controlada.
    
    Returns:
        pandas.DataFrame: Dataset complexo com valores ausentes e outliers
        str: Tipo de tarefa ('classification' ou 'regression')
        list: Nomes das classes alvo
    """
    import pandas as pd
    import numpy as np
    
    # Parâmetros do dataset
    n_samples = 1000
    n_features = 40
    
    # Criar features informativas e ruidosas
    np.random.seed(42)  # Para reprodutibilidade
    X_informative = np.random.normal(0, 1, (n_samples, 15))
    X_noise = np.random.normal(0, 1, (n_samples, n_features - 15))
    
    # Combinar features
    X = np.hstack((X_informative, X_noise))
    
    # Criar target baseado apenas nas features informativas
    y = 1.5 * X_informative[:, 0] + 0.8 * X_informative[:, 1] - 2 * X_informative[:, 2]
    y += 0.5 * X_informative[:, 3] ** 2 - 0.7 * X_informative[:, 4]
    y += np.random.normal(0, 1, n_samples)  # Adicionar ruído
    
    # Gerar nomes de features
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    
    # Criar DataFrame base
    df = pd.DataFrame(X, columns=feature_names)
    
    # Adicionar valores ausentes (10% dos dados)
    mask = np.random.random(X.shape) < 0.1
    X_with_missing = X.copy()
    X_with_missing[mask] = np.nan
    
    # Atualizar DataFrame com valores ausentes
    for i, col in enumerate(feature_names):
        df[col] = X_with_missing[:, i]
    
    # Adicionar outliers (2% dos dados)
    outlier_mask = np.random.random(X.shape) < 0.02
    for i, col in enumerate(feature_names):
        # Pegue apenas células que são outliers para esta coluna
        col_outliers = outlier_mask[:, i]
        # Defina valores de outlier
        df.loc[col_outliers, col] = np.random.normal(0, 10, size=np.sum(col_outliers))
    
    # Converter para classificação (3 classes)
    # Usando qcut para divisão em quantis
    df['target'] = pd.qcut(y, 3, labels=[0, 1, 2])
    
    # Garantir que o tipo do target é inteiro
    df['target'] = df['target'].astype(int)
    
    # Verificar a consistência do dataset
    assert all(len(df[col]) == len(df) for col in df.columns), "Colunas têm tamanhos diferentes"
    
    return df, 'classification', ['Low', 'Medium', 'High']


def create_high_dim_classification():
    """
    Cria um dataset de classificação de alta dimensionalidade.
    
    Returns:
        pandas.DataFrame: Dataset de classificação com 100 features
        str: Tipo de tarefa ('classification')
        list: Nomes das classes alvo
    """
    import pandas as pd
    import numpy as np
    from sklearn.datasets import make_classification
    
    # Gerar dataset sintético
    X, y = make_classification(
        n_samples=1000, 
        n_features=100,  # 100 features
        n_informative=20,  # Apenas 20 são realmente informativas
        n_redundant=30,   # 30 são redundantes (derivadas das informativas)
        n_repeated=10,    # 10 são repetições de outras features
        n_classes=4,
        random_state=42,
        class_sep=1.5
    )
    
    # Gerar nomes de features
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    
    # Criar DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    # Verificar consistência
    assert all(len(df[col]) == len(df) for col in df.columns), "Colunas têm tamanhos diferentes"
    
    return df, 'classification', [f'Class {i}' for i in range(4)]


def create_high_dim_regression():
    """
    Cria um dataset de regressão de alta dimensionalidade.
    
    Returns:
        pandas.DataFrame: Dataset de regressão com 100 features
        str: Tipo de tarefa ('regression')
        None: Não há nomes de classes para regressão
    """
    import pandas as pd
    import numpy as np
    from sklearn.datasets import make_regression
    
    # Gerar dataset sintético
    X, y = make_regression(
        n_samples=1000, 
        n_features=100,   # 100 features
        n_informative=20, # Apenas 20 são realmente informativas
        n_targets=1, 
        random_state=42,
        noise=0.1
    )
    
    # Gerar nomes de features
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    
    # Criar DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    # Verificar consistência
    assert all(len(df[col]) == len(df) for col in df.columns), "Colunas têm tamanhos diferentes"
    
    return df, 'regression', None


def load_dataset(dataset_name='iris'):
    """
    Carrega um dos datasets de exemplo.
    
    Args:
        dataset_name: Nome do dataset a ser carregado
        
    Returns:
        pandas.DataFrame: Dataset carregado
        str: Tipo de tarefa ('classification' ou 'regression')
        list/None: Nomes das classes alvo (ou None para regressão)
    """
    import pandas as pd
    import numpy as np
    from sklearn.datasets import (
        load_iris, load_wine, load_breast_cancer, load_diabetes, 
        fetch_california_housing
    )
    
    # Datasets simples de classificação
    if dataset_name == 'iris':
        data = load_iris()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        task = 'classification'
        target_names = data.target_names
        
    elif dataset_name == 'wine':
        data = load_wine()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        task = 'classification'
        target_names = data.target_names
        
    elif dataset_name == 'breast_cancer':
        data = load_breast_cancer()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        task = 'classification'
        target_names = np.array(['malignant', 'benign'])
        
    # Dataset de regressão
    elif dataset_name == 'diabetes':
        data = load_diabetes()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        task = 'regression'
        target_names = None
        
    elif dataset_name == 'california_housing':
        data = fetch_california_housing()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        task = 'regression'
        target_names = None
        
    # Datasets complexos
    elif dataset_name == 'high_dim_classification':
        return create_high_dim_classification()
    
    elif dataset_name == 'high_dim_regression':
        return create_high_dim_regression()
    
    elif dataset_name == 'complex_timeseries':
        return create_complex_timeseries_dataset()
    
    elif dataset_name == 'noisy_data':
        return create_noisy_data_dataset()
    
    elif dataset_name == 'local_file':
        return load_local_dataset('data/dados.csv', dataset_type='classification')
    
    else:
        raise ValueError(f"Dataset {dataset_name} não suportado.")
    
    return df, task, target_names


def visualize_results(validation_results, dataset_name):
    """Visualiza os resultados da validação de performance."""
    # Criar figura para visualização
    plt.figure(figsize=(15, 10))
    
    # 1. Gráfico de comparação de performance
    plt.subplot(2, 2, 1)
    performance = [
        validation_results['performance_original'],
        validation_results['performance_transformed']
    ]
    colors = ['blue', 'green'] if validation_results['performance_diff'] >= 0 else ['blue', 'red']
    
    bars = plt.bar(['Original', 'Transformado'], performance, color=colors)
    
    # Adicionar valores nas barras
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.4f}', ha='center', va='bottom')
    
    plt.title(f'Comparação de Performance - {dataset_name}')
    plt.ylabel('Performance (Métrica: Acurácia/R²)')
    plt.ylim(min(0, min(performance) - 0.1), max(1.1, max(performance) + 0.1))
    
    # 2. Comparação de número de features
    plt.subplot(2, 2, 2)
    feature_reduction = validation_results.get('feature_reduction', 0) * 100
    
    # Criar gráfico de barras para número de features antes/depois
    if feature_reduction >= 0:
        feature_bars = plt.bar(['Original', 'Transformado'], [1, 1-feature_reduction/100], color=['blue', 'green'])
        plt.ylabel('Proporção de Features (Original = 1)')
        title_suffix = f"Redução de {feature_reduction:.1f}%"
    else:
        feature_reduction = abs(feature_reduction)
        feature_bars = plt.bar(['Original', 'Transformado'], [1, 1+feature_reduction/100], color=['blue', 'orange'])
        plt.ylabel('Proporção de Features (Original = 1)')
        title_suffix = f"Aumento de {feature_reduction:.1f}%"
        
    plt.title(f'Comparação de Número de Features - {title_suffix}')
    
    # Adicionar valores nas barras
    for i, bar in enumerate(feature_bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f"{int(height * validation_results.get('original_n_features', 10))}", 
                ha='center', va='bottom')
    
    # 3. Performance por fold
    plt.subplot(2, 2, 3)
    folds = list(range(1, len(validation_results['scores_original'])+1))
    
    plt.plot(folds, validation_results['scores_original'], 'o-', label='Original', color='blue')
    plt.plot(folds, validation_results['scores_transformed'], 'o-', label='Transformado', 
             color='green' if validation_results['performance_diff'] >= 0 else 'red')
    
    plt.title('Performance por Fold de Validação Cruzada')
    plt.xlabel('Fold')
    plt.ylabel('Performance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Texto com resumo e decisão
    plt.subplot(2, 2, 4)
    plt.axis('off')
    
    performance_diff = validation_results['performance_diff']
    performance_diff_pct = validation_results['performance_diff_pct']
    best_choice = validation_results['best_choice']
    
    text = f"""
    RESUMO DA VALIDAÇÃO DE PERFORMANCE
    
    Dataset: {dataset_name}
    
    Performance:
    - Original:    {validation_results['performance_original']:.4f}
    - Transformado: {validation_results['performance_transformed']:.4f}
    - Diferença:    {performance_diff:.4f} ({performance_diff_pct:.2f}%)
    
    Features:
    - Redução:      {abs(feature_reduction):.1f}%
    
    DECISÃO: Usar dados {best_choice.upper()}
    
    Configuração do validador:
    - Máxima queda permitida: {validation_results.get('max_performance_drop', 0.05)*100:.1f}%
    - Folds de validação: {len(validation_results['scores_original'])}
    - Métrica: {validation_results.get('metric', 'accuracy')}
    """
    
    plt.text(0.1, 0.9, text, fontsize=12, va='top', family='monospace')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{dataset_name}_validation_results.png")

def run_demo(dataset_name, show_plots=True):
    """Executa a demonstração completa para um dataset específico."""
    print(f"\n{'='*80}")
    print(f" DEMONSTRAÇÃO DE VALIDAÇÃO DE PERFORMANCE - DATASET {dataset_name.upper()} ".center(80, "="))
    print(f"{'='*80}\n")
    
    # 1. Carregar dataset
    start_time = time.time()
    df, task, target_names = load_dataset(dataset_name)
    
    # Informações básicas do dataset
    n_features = df.shape[1] - 1  # excluir coluna target
    n_samples = df.shape[0]
    n_classes = len(np.unique(df['target'])) if task == 'classification' else None
    
    print(f"Dataset carregado: {n_samples} amostras, {n_features} features")
    if task == 'classification':
        print(f"Tipo: Classificação ({n_classes} classes)")
        print(f"Classes: {', '.join(str(c) for c in target_names)}")
    else:
        print(f"Tipo: Regressão")
        print(f"Range de Target: {df['target'].min():.2f} a {df['target'].max():.2f}")
    
    # Verificar tipos de dados
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if 'target' in numeric_cols:
        numeric_cols.remove('target')
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    print(f"\nTipos de features:")
    print(f"- Numéricas: {len(numeric_cols)}")
    print(f"- Categóricas: {len(categorical_cols)}")
    print(f"- Data/Hora: {len(datetime_cols)}")
    
    # Verificar valores ausentes
    missing_values = df.isnull().sum().sum()
    if missing_values > 0:
        print(f"\nValores ausentes: {missing_values} ({missing_values/(df.shape[0]*df.shape[1])*100:.2f}%)")
    
    # 2. Configurações do pipeline
    # Configurações específicas por dataset
    if task == 'classification':
        validator_config = {
            'max_performance_drop': 0.15 if dataset_name == 'iris' else 0.05,
            'cv_folds': 5,
            'metric': 'accuracy',
            'task': 'classification',
            'base_model': 'rf',
            'verbose': True
        }
        
        feature_config = {
            'correlation_threshold': 0.8,
            'generate_features': dataset_name in ['high_dim_classification', 'complex_timeseries', 'noisy_data']
        }
    else:  # regressão
        validator_config = {
            'max_performance_drop': 0.05,
            'cv_folds': 5,
            'metric': 'r2',
            'task': 'regression',
            'base_model': 'rf',
            'verbose': True
        }
        
        feature_config = {
            'correlation_threshold': 0.8,
            'generate_features': dataset_name in ['high_dim_regression', 'diabetes']
        }
    
    if len(datetime_cols) > 0:
        # Adicionar configuração para processamento de datas
        preprocessor_config = {
            'datetime_features': ['year', 'month', 'day', 'weekday', 'is_weekend', 'quarter'],
        }
    else:
        preprocessor_config = {}
    
    # 3. Criar e ajustar o pipeline
    pipeline = create_data_pipeline(
        preprocessor_config=preprocessor_config,
        feature_engineer_config=feature_config,
        validator_config=validator_config,
        auto_validate=True  # Ativar validação automática
    )
    
    # 4. Ajustar o pipeline ao dataset
    print("\nAjustando o pipeline com validação automática...")
    pipeline_start = time.time()
    transformed_df = pipeline.fit_transform(df, target_col='target')
    pipeline_time = time.time() - pipeline_start
    
    # 5. Obter e mostrar resultados da validação
    validation_results = pipeline.get_validation_results()
    
    if validation_results:
        # Adicionar o número original de features
        validation_results['original_n_features'] = n_features
        
        print("\nResultados da validação:")
        print(f"- Performance original: {validation_results['performance_original']:.4f}")
        print(f"- Performance transformada: {validation_results['performance_transformed']:.4f}")
        print(f"- Diferença: {validation_results['performance_diff']:.4f} ({validation_results['performance_diff_pct']:.2f}%)")
        print(f"- Melhor escolha: {validation_results['best_choice'].upper()}")
        
        # Informações sobre o dataset transformado
        print(f"\nDataset original: {df.shape[0]} amostras, {n_features} features")
        print(f"Dataset após transformação: {transformed_df.shape[0]} amostras, {transformed_df.shape[1]-1} features")
        
        # Feature importance
        print("\nFeature importance (dataset original):")
        importance = pipeline.get_feature_importance(df, target_col='target')
        for i, (_, row) in enumerate(importance.head(5).iterrows()):
            print(f"  {i+1}. {row['feature']}: {row['importance']:.4f}")
        
        # Visualizar resultados
        if show_plots:
            visualize_results(validation_results, dataset_name)
    else:
        print("\nValidação não foi realizada. Verifique as configurações.")
    
    # 6. Avaliar model com dataset final
    print("\nAvaliando modelo com dataset final...")
    X = transformed_df.drop(columns=['target'])
    y = transformed_df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if task == 'classification':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\nAcurácia final (conjunto de teste): {accuracy:.4f}")
        print("\nRelatório de classificação:")
        target_names_list = list(target_names) if target_names is not None else None
        print(classification_report(y_test, y_pred, target_names=target_names_list))
    else:  # regressão
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        print(f"\nPerformance final (conjunto de teste):")
        print(f"- R²: {r2:.4f}")
        print(f"- RMSE: {rmse:.4f}")
    
    # 7. Tempo total
    total_time = time.time() - start_time
    print(f"\nTempo de execução:")
    print(f"- Pipeline: {pipeline_time:.2f} segundos")
    print(f"- Total: {total_time:.2f} segundos")
    
    # Retornar o pipeline e os resultados para uso adicional
    return pipeline, validation_results

def compare_datasets(datasets):
    """
    Executa a demonstração para múltiplos datasets e compara os resultados.
    """
    results = {}
    
    for dataset in datasets:
        print(f"\nTestando dataset: {dataset}")
        _, validation = run_demo(dataset, show_plots=False)
        results[dataset] = validation
    
    # Visualizar comparação entre os datasets
    plt.figure(figsize=(14, 10))
    
    # 1. Comparação de diferença de performance
    plt.subplot(2, 2, 1)
    performance_diff = [results[ds]['performance_diff_pct'] for ds in datasets]
    bars = plt.bar(datasets, performance_diff)
    
    # Colorir barras
    for i, bar in enumerate(bars):
        bar.set_color('green' if performance_diff[i] >= 0 else 'red')
        
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.title('Diferença de Performance (%)')
    plt.ylabel('Diferença (%)')
    plt.xticks(rotation=45, ha='right')
    
    # 2. Comparação de redução de features
    plt.subplot(2, 2, 2)
    feature_reduction = [results[ds]['feature_reduction'] * 100 for ds in datasets]
    bars = plt.bar(datasets, feature_reduction)
    
    # Colorir barras
    for i, bar in enumerate(bars):
        bar.set_color('green' if feature_reduction[i] >= 0 else 'orange')
    
    plt.title('Redução de Features (%)')
    plt.ylabel('Redução (%)')
    plt.xticks(rotation=45, ha='right')
    
    # 3. Decisões tomadas
    plt.subplot(2, 1, 2)
    
    # Criar tabela para mostrar decisões
    cell_text = []
    for ds in datasets:
        cell_text.append([
            ds,
            f"{results[ds]['performance_original']:.4f}",
            f"{results[ds]['performance_transformed']:.4f}",
            f"{results[ds]['performance_diff_pct']:.2f}%",
            f"{results[ds]['feature_reduction'] * 100:.1f}%",
            results[ds]['best_choice'].upper()
        ])
    
    columns = ['Dataset', 'Perf. Original', 'Perf. Transformado', 'Diferença', 'Redução Features', 'Decisão']
    
    plt.axis('off')
    table = plt.table(cellText=cell_text, colLabels=columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    plt.title('Resumo das Decisões', y=0.8)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/comparison_datasets.png")
    plt.show()
    
    # Criar gráfico adicional para dimensionalidade vs. performance
    plt.figure(figsize=(12, 6))
    
    # Preparar dados para o gráfico
    dataset_names = []
    original_dims = []
    transformed_dims = []
    performance_gains = []
    decision_colors = []
    
    for ds in datasets:
        dataset_names.append(ds)
        orig_features = results[ds]['original_n_features']
        transformed_features = int(orig_features * (1 - results[ds]['feature_reduction']))
        
        original_dims.append(orig_features)
        transformed_dims.append(transformed_features)
        performance_gains.append(results[ds]['performance_diff_pct'])
        decision_colors.append('green' if results[ds]['best_choice'] == 'transformed' else 'red')
    
    # Plotar dimensionalidade original vs. transformada
    plt.subplot(1, 2, 1)
    
    # Linha de referência (sem mudança)
    max_dim = max(max(original_dims), max(transformed_dims))
    plt.plot([0, max_dim], [0, max_dim], 'k--', alpha=0.3)
    
    # Pontos para cada dataset
    for i, ds in enumerate(dataset_names):
        plt.scatter(original_dims[i], transformed_dims[i], s=100, 
                   color=decision_colors[i], alpha=0.7)
        plt.text(original_dims[i], transformed_dims[i], ds, fontsize=9)
    
    plt.xlabel('Dimensionalidade Original')
    plt.ylabel('Dimensionalidade após AutoFE')
    plt.title('Mudança na Dimensionalidade')
    
    # Plotar dimensionalidade vs. ganho de performance
    plt.subplot(1, 2, 2)
    
    # Linha de referência (sem ganho)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # Calcular a redução percentual de dimensionalidade
    dim_reduction_pct = [(o - t) / o * 100 for o, t in zip(original_dims, transformed_dims)]
    
    # Pontos para cada dataset
    for i, ds in enumerate(dataset_names):
        plt.scatter(dim_reduction_pct[i], performance_gains[i], s=100, 
                   color=decision_colors[i], alpha=0.7)
        plt.text(dim_reduction_pct[i], performance_gains[i], ds, fontsize=9)
    
    plt.xlabel('Redução de Dimensionalidade (%)')
    plt.ylabel('Ganho de Performance (%)')
    plt.title('Redução de Dimensionalidade vs. Ganho de Performance')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/dimensionality_vs_performance.png")
    plt.show()
    
    return results

if __name__ == "__main__":
    print("=" * 80)
    print(" DEMONSTRAÇÃO DA VALIDAÇÃO DE PERFORMANCE DO AUTOFE ".center(80, "="))
    print("=" * 80)
    
    # Datasets para avaliar
    datasets = [
        'iris',                    # Classificação simples
        'wine',                    # Classificação com mais features
        'breast_cancer',           # Classificação com muitas features
        'diabetes',                # Regressão simples
        'high_dim_classification', # Classificação com alta dimensionalidade (100 features)
        'high_dim_regression',     # Regressão com alta dimensionalidade (100 features)
        'complex_timeseries',      # Dataset com diferentes tipos de dados (datas, categóricos, numéricos)
        'noisy_data',              # Dataset com valores ausentes e outliers
        'local_file'               # Dataset carregado de um arquivo local
    ]
    
    # Executar demonstração comparando múltiplos datasets
    results = compare_datasets(datasets)
    
    print("\n" + "=" * 80)
    print(" CONCLUSÃO ".center(80, "="))
    print("=" * 80)
    
    # Contar quantos datasets foram beneficiados pelo AutoFE
    improved = sum(1 for ds in datasets if results[ds]['best_choice'] == 'transformed')
    not_improved = sum(1 for ds in datasets if results[ds]['best_choice'] == 'original')
    
    print(f"\nResultados gerais da validação em {len(datasets)} datasets:")
    print(f"- Datasets onde o AutoFE melhorou a performance: {improved}/{len(datasets)} ({improved/len(datasets)*100:.1f}%)")
    print(f"- Datasets onde foi melhor manter os dados originais: {not_improved}/{len(datasets)} ({not_improved/len(datasets)*100:.1f}%)")
    
    # Calcular estatísticas da redução de dimensionalidade
    dim_reductions = [results[ds]['feature_reduction'] * 100 for ds in datasets]
    avg_reduction = sum(dim_reductions) / len(dim_reductions)
    max_reduction = max(dim_reductions)
    min_reduction = min(dim_reductions)
    
    print(f"\nEstatísticas de redução de dimensionalidade:")
    print(f"- Redução média: {avg_reduction:.1f}%")
    print(f"- Redução máxima: {max_reduction:.1f}%")
    print(f"- Redução mínima: {min_reduction:.1f}%")
    
    # Calcular estatísticas da mudança de performance
    perf_changes = [results[ds]['performance_diff_pct'] for ds in datasets]
    avg_perf_change = sum(perf_changes) / len(perf_changes)
    max_perf_gain = max(perf_changes)
    max_perf_loss = min(perf_changes)
    
    print(f"\nEstatísticas de mudança de performance:")
    print(f"- Mudança média: {avg_perf_change:.2f}%")
    print(f"- Maior ganho: {max_perf_gain:.2f}%")
    print(f"- Maior perda: {max_perf_loss:.2f}%")
    
    # Datasets com maior ganho e maior perda
    best_dataset = max(datasets, key=lambda ds: results[ds]['performance_diff_pct'])
    worst_dataset = min(datasets, key=lambda ds: results[ds]['performance_diff_pct'])
    
    print(f"\nDataset com maior ganho de performance: {best_dataset} (+{results[best_dataset]['performance_diff_pct']:.2f}%)")
    print(f"Dataset com maior perda de performance: {worst_dataset} ({results[worst_dataset]['performance_diff_pct']:.2f}%)")
    
    # Resultados por tipo de dataset
    classification_datasets = ['iris', 'wine', 'breast_cancer', 'high_dim_classification', 'complex_timeseries', 'noisy_data']
    regression_datasets = ['diabetes', 'high_dim_regression']
    
    simple_datasets = ['iris', 'wine', 'diabetes']
    complex_datasets = ['breast_cancer', 'high_dim_classification', 'high_dim_regression', 'complex_timeseries', 'noisy_data']
    
    # Calcular médias por tipo
    class_improvement = sum(results[ds]['performance_diff_pct'] for ds in classification_datasets) / len(classification_datasets)
    reg_improvement = sum(results[ds]['performance_diff_pct'] for ds in regression_datasets) / len(regression_datasets)
    
    simple_improvement = sum(results[ds]['performance_diff_pct'] for ds in simple_datasets) / len(simple_datasets)
    complex_improvement = sum(results[ds]['performance_diff_pct'] for ds in complex_datasets) / len(complex_datasets)
    
    print(f"\nGanho médio por tipo de tarefa:")
    print(f"- Classificação: {class_improvement:.2f}%")
    print(f"- Regressão: {reg_improvement:.2f}%")
    
    print(f"\nGanho médio por complexidade:")
    print(f"- Datasets simples: {simple_improvement:.2f}%")
    print(f"- Datasets complexos: {complex_improvement:.2f}%")
    
    # Impacto da validação de performance
    all_transformed = sum(1 for ds in datasets if results[ds]['performance_diff'] >= 0)
    prevented_losses = sum(1 for ds in datasets if results[ds]['performance_diff'] < 0)
    
    print(f"\nImpacto do sistema de validação de performance:")
    print(f"- Transformações benéficas aplicadas: {all_transformed}")
    print(f"- Perdas de performance evitadas: {prevented_losses}")
    
    print(f"\nConclusão: O AutoFE com sistema de validação de performance demonstrou ser eficaz,")
    print(f"principalmente em datasets complexos onde a redução de dimensionalidade")
    
    if complex_improvement > simple_improvement:
        print(f"trouxe maiores ganhos de performance (+{complex_improvement:.2f}% em média) e eficiência.")
    else:
        print(f"trouxe ganhos de eficiência mantendo a performance em níveis aceitáveis.")
        
    print(f"\nO sistema evitou perdas de performance em {prevented_losses} datasets ao decidir")
    print(f"manter os dados originais quando as transformações seriam prejudiciais.")
    
    print(f"\nArquivos de análise salvos em: {OUTPUT_DIR}/")
    
    # Opção para executar um teste específico com visualização detalhada
    run_specific = input("\nDeseja executar um teste detalhado em um dataset específico? (s/n): ")
    if run_specific.lower() == 's':
        # Mostrar opções disponíveis
        print("\nDatasets disponíveis:")
        for i, ds in enumerate(datasets, 1):
            print(f"{i}. {ds}")
            
        try:
            choice = int(input("\nEscolha o número do dataset: "))
            if 1 <= choice <= len(datasets):
                selected_dataset = datasets[choice-1]
                print(f"\nExecutando análise detalhada do dataset '{selected_dataset}'...")
                run_demo(selected_dataset, show_plots=True)
            else:
                print("Opção inválida!")
        except ValueError:
            print("Por favor, digite um número válido.")
            
    print("\nDemo concluída!")