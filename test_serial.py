"""
Teste do AutoFE para lidar com colunas de data/hora.

Este script:
1. Gera um dataset com colunas de data/hora
2. Aplica o AutoFE, que deve processar corretamente essas colunas
3. Avalia o impacto das características extraídas de data/hora na performance do modelo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# Importar componentes do AutoFE
from preprocessor import PreProcessor
from feature_engineer import FeatureEngineer
from data_pipeline import DataPipeline
from explorer import Explorer

# Configuração
np.random.seed(42)
OUTPUT_DIR = "datetime_test_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_datetime_dataset(n_samples=1000, noise_level=0.1, seasonality=True):
    """
    Cria um dataset com padrões temporais para classificação.
    
    Args:
        n_samples: Número de amostras no dataset
        noise_level: Nível de ruído introduzido
        seasonality: Se deve incluir padrões sazonais
    
    Returns:
        DataFrame com colunas de data/hora e um alvo de classificação
    """
    print("Gerando dataset com padrões temporais...")
    
    # Criar sequência de datas
    start_date = datetime(2020, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_samples)]
    
    # Criar DataFrame
    df = pd.DataFrame({'data': dates})
    
    # Adicionar features numéricas
    df['valor_1'] = np.random.randn(n_samples)
    df['valor_2'] = np.random.randn(n_samples)
    
    # Adicionar ruído
    df['ruido'] = np.random.randn(n_samples) * noise_level
    
    # Adicionar padrões sazonais
    if seasonality:
        # Padrão mensal: valores mais altos no meio do mês
        df['padrao_mensal'] = np.sin(df['data'].dt.day * (2 * np.pi / 30))
        
        # Padrão semanal: valores mais altos nos fins de semana
        df['padrao_semanal'] = (df['data'].dt.weekday >= 5).astype(float) * 2 - 1
        
        # Padrão anual: valores mais altos no verão
        df['padrao_anual'] = np.sin((df['data'].dt.month - 1) * (2 * np.pi / 12))
    
    # Valores combinados
    df['valor_combinado'] = (
        df['valor_1'] + 
        df['valor_2'] + 
        df['ruido'] + 
        (3 * df['padrao_mensal'] if seasonality else 0) +
        (2 * df['padrao_semanal'] if seasonality else 0) +
        (4 * df['padrao_anual'] if seasonality else 0)
    )
    
    # Criar alvo baseado em características temporais e valores
    # Classe 0: Dias de semana com valor baixo
    # Classe 1: Dias de semana com valor alto
    # Classe 2: Fim de semana com valor baixo
    # Classe 3: Fim de semana com valor alto
    is_weekend = df['data'].dt.weekday >= 5
    is_high_value = df['valor_combinado'] > df['valor_combinado'].median()
    
    df['target'] = 0
    df.loc[~is_weekend & is_high_value, 'target'] = 1
    df.loc[is_weekend & ~is_high_value, 'target'] = 2
    df.loc[is_weekend & is_high_value, 'target'] = 3
    
    # Adicionar outra coluna de data (intervalo mais recente)
    df['data_recente'] = df['data'] + timedelta(days=365*2)
    
    # Adicionar timestamp
    df['timestamp'] = pd.to_datetime(df['data']) + pd.to_timedelta(
        np.random.randint(0, 24*60*60, size=n_samples), unit='s'
    )
    
    # Adicionar algumas colunas categóricas
    df['categoria'] = np.random.choice(['A', 'B', 'C'], size=n_samples)
    df['status'] = np.where(df['valor_combinado'] > 0, 'Positivo', 'Negativo')
    
    print(f"Dataset gerado com formato: {df.shape}")
    print(f"Tipos de dados:\n{df.dtypes}")
    print(f"Distribuição de classes:\n{df['target'].value_counts()}")
    
    return df

def visualize_temporal_patterns(df, output_path):
    """
    Visualiza padrões temporais nos dados.
    """
    plt.figure(figsize=(15, 10))
    
    # Agrupar por mês e dia da semana
    df['month'] = df['data'].dt.month
    df['weekday'] = df['data'].dt.weekday
    
    # Plotar distribuição de classes por mês
    plt.subplot(2, 2, 1)
    monthly_counts = df.groupby(['month', 'target']).size().unstack()
    monthly_counts.plot(kind='bar', stacked=True, ax=plt.gca())
    plt.title('Distribuição de Classes por Mês')
    plt.xlabel('Mês')
    plt.ylabel('Contagem')
    
    # Plotar distribuição de classes por dia da semana
    plt.subplot(2, 2, 2)
    weekday_counts = df.groupby(['weekday', 'target']).size().unstack()
    weekday_counts.plot(kind='bar', stacked=True, ax=plt.gca())
    plt.title('Distribuição de Classes por Dia da Semana')
    plt.xlabel('Dia da Semana (0=Segunda, 6=Domingo)')
    plt.ylabel('Contagem')
    
    # Plotar valores combinados ao longo do tempo
    plt.subplot(2, 1, 2)
    sample_df = df.sample(min(500, len(df)))
    scatter = plt.scatter(sample_df['data'], sample_df['valor_combinado'], 
                         c=sample_df['target'], cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label='Classe')
    plt.title('Valores ao Longo do Tempo')
    plt.xlabel('Data')
    plt.ylabel('Valor Combinado')
    
    plt.tight_layout()
    plt.savefig(f"{output_path}/temporal_patterns.png")
    
    # Relação entre dia da semana, valor e classe
    plt.figure(figsize=(12, 8))
    pivot_df = df.pivot_table(
        index='weekday', 
        columns='target', 
        values='valor_combinado', 
        aggfunc='mean'
    )
    sns.heatmap(pivot_df, annot=True, cmap='viridis', fmt='.2f')
    plt.title('Valor Médio por Dia da Semana e Classe')
    plt.xlabel('Classe')
    plt.ylabel('Dia da Semana (0=Segunda, 6=Domingo)')
    plt.tight_layout()
    plt.savefig(f"{output_path}/weekday_class_value.png")

def evaluate_model(X_train, y_train, X_test, y_test, feature_names=None, title="Modelo"):
    """
    Treina e avalia um modelo RandomForest.
    
    Returns:
        Dicionário com métricas e informações do modelo.
    """
    print(f"\n=== Avaliando {title} ===")
    
    start_time = time.time()
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    # Avaliação
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Tempo de treino: {train_time:.2f} segundos")
    print(f"Acurácia: {accuracy:.4f}")
    
    # Relatório de classificação
    print("\nRelatório de classificação:")
    print(classification_report(y_test, y_pred))
    
    # Importance das features
    importance = None
    if feature_names is not None:
        importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 features mais importantes:")
        for i, (_, row) in enumerate(importance.head(10).iterrows()):
            print(f"  {i+1}. {row['feature']}: {row['importance']:.4f}")
    
    result = {
        'model': model,
        'accuracy': accuracy,
        'train_time': train_time,
        'importance': importance
    }
    
    return result

def visualize_feature_importance(results, output_path):
    """
    Visualiza a importância das features para cada modelo.
    """
    plt.figure(figsize=(14, 10))
    
    for i, (title, result) in enumerate(results.items(), 1):
        if result.get('importance') is not None:
            plt.subplot(len(results), 1, i)
            importance = result['importance'].head(15)
            sns.barplot(x='importance', y='feature', data=importance)
            plt.title(f'Top 15 Features - {title}')
            
    plt.tight_layout()
    plt.savefig(f"{output_path}/feature_importance.png")
    
    # Destaque para as features de data
    plt.figure(figsize=(12, 8))
    date_features = []
    
    for title, result in results.items():
        if result.get('importance') is not None:
            # Filtrar features que contêm 'data', 'date', 'dia', 'mes', 'ano', etc.
            date_cols = [col for col in result['importance']['feature'] 
                          if any(term in col.lower() for term in ['data', 'date', 'dia', 'mes', 'year', 'month', 'week'])]
            if date_cols:
                date_importance = result['importance'][result['importance']['feature'].isin(date_cols)].copy()
                date_importance.loc[:, 'model'] = title
                date_features.append(date_importance)
    
    if date_features:
        date_df = pd.concat(date_features)
        sns.barplot(x='importance', y='feature', hue='model', data=date_df)
        plt.title('Importância das Features de Data/Hora')
        plt.tight_layout()
        plt.savefig(f"{output_path}/date_feature_importance.png")

def compare_results(results, output_path):
    """
    Compara resultados entre diferentes abordagens.
    """
    # Preparar dados para comparação
    models = list(results.keys())
    accuracies = [result['accuracy'] for result in results.values()]
    train_times = [result['train_time'] for result in results.values()]
    
    # Criar figura
    plt.figure(figsize=(12, 6))
    
    # Acurácia
    plt.subplot(1, 2, 1)
    bars = plt.bar(models, accuracies, color='skyblue')
    
    # Adicionar valores nas barras
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom')
    
    plt.title('Comparação de Acurácia')
    plt.ylim(0, 1)
    plt.ylabel('Acurácia')
    
    # Tempo de treino
    plt.subplot(1, 2, 2)
    bars = plt.bar(models, train_times, color='salmon')
    
    # Adicionar valores nas barras
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}s', ha='center', va='bottom')
    
    plt.title('Comparação de Tempo de Treino')
    plt.ylabel('Tempo (segundos)')
    
    plt.tight_layout()
    plt.savefig(f"{output_path}/results_comparison.png")
    
    # Salvar resumo em texto
    with open(f"{output_path}/summary.txt", "w") as f:
        f.write("=== RESUMO DOS RESULTADOS ===\n\n")
        f.write("Acurácia:\n")
        for model, acc in zip(models, accuracies):
            f.write(f"  {model}: {acc:.4f}\n")
        
        f.write("\nTempo de treino:\n")
        for model, time_val in zip(models, train_times):
            f.write(f"  {model}: {time_val:.2f} segundos\n")
        
        # Calcular melhorias
        baseline_acc = results['Modelo Base']['accuracy']
        for model in models:
            if model != 'Modelo Base':
                acc_diff = (results[model]['accuracy'] - baseline_acc) * 100
                f.write(f"\nMelhoria com {model}: {acc_diff:.2f}% "
                        f"({'ganho' if acc_diff >= 0 else 'perda'})")

def main():
    print("\n" + "="*80)
    print(" TESTE DO AUTOFE COM DADOS TEMPORAIS ".center(80, "="))
    print("="*80 + "\n")
    
    # 1. Criar dataset com padrões temporais
    df = create_datetime_dataset(n_samples=1500, noise_level=0.1, seasonality=True)
    
    # Salvar o dataset
    df.to_csv(f"{OUTPUT_DIR}/temporal_dataset.csv", index=False)
    
    # 2. Visualizar padrões temporais
    visualize_temporal_patterns(df, OUTPUT_DIR)
    
    # 3. Dividir em treino/teste
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['target'])
    
    # 4. Testar três abordagens diferentes:
    
    results = {}
    
    # 4.1 Modelo Base: Ignora colunas de data
    print("\n=== TESTE DO MODELO BASE (SEM FEATURES DE DATA) ===")
    # Selecionar apenas features numéricas
    X_train_base = train_df.select_dtypes(include=['number']).drop(columns=['target'])
    X_test_base = test_df.select_dtypes(include=['number']).drop(columns=['target'])
    
    # Avaliar modelo base
    base_results = evaluate_model(
        X_train_base, train_df['target'], 
        X_test_base, test_df['target'],
        feature_names=X_train_base.columns,
        title="Modelo Base"
    )
    results['Modelo Base'] = base_results
    
    # 4.2 Modelo com Features Manuais de Data
    print("\n=== TESTE DO MODELO COM FEATURES MANUAIS DE DATA ===")
    # Extrair features manuais de data
    train_manual = train_df.copy()
    test_manual = test_df.copy()
    
    for df in [train_manual, test_manual]:
        # Data principal
        df['dia'] = df['data'].dt.day
        df['mes'] = df['data'].dt.month
        df['ano'] = df['data'].dt.year
        df['dia_semana'] = df['data'].dt.weekday
        df['fim_de_semana'] = (df['data'].dt.weekday >= 5).astype(int)
        df['dia_do_ano'] = df['data'].dt.dayofyear
        
        # Data recente
        df['dia_recente'] = df['data_recente'].dt.day
        df['mes_recente'] = df['data_recente'].dt.month
        df['ano_recente'] = df['data_recente'].dt.year
        
        # Timestamp
        df['hora'] = df['timestamp'].dt.hour
        df['minuto'] = df['timestamp'].dt.minute
        
        # Remover colunas originais de data
        df.drop(columns=['data', 'data_recente', 'timestamp'], inplace=True)
    
    # Selecionar features numéricas
    X_train_manual = train_manual.select_dtypes(include=['number']).drop(columns=['target'])
    X_test_manual = test_manual.select_dtypes(include=['number']).drop(columns=['target'])
    
    # Avaliar modelo com features manuais
    manual_results = evaluate_model(
        X_train_manual, train_manual['target'], 
        X_test_manual, test_manual['target'],
        feature_names=X_train_manual.columns,
        title="Modelo com Features Manuais"
    )
    results['Modelo com Features Manuais'] = manual_results
    
    # 4.3 Modelo com AutoFE
    print("\n=== TESTE DO MODELO COM AUTOFE ===")
    
    # Definir configuração do preprocessor com processamento de data
    preprocessor_config = {
        'datetime_features': ['year', 'month', 'day', 'weekday', 'hour', 'minute', 'is_weekend'],
        'scaling': 'standard',
        'missing_values_strategy': 'median'
    }
    
    # Configuração do feature engineer
    feature_engineer_config = {
        'correlation_threshold': 0.8,
        'generate_features': True
    }
    
    # Criar pipeline
    pipeline = DataPipeline(
        preprocessor_config=preprocessor_config,
        feature_engineer_config=feature_engineer_config
    )
    
    # Ajustar o pipeline aos dados de treino
    pipeline.fit(train_df, target_col='target')
    
    # Transformar dados de treino e teste
    train_transformed = pipeline.transform(train_df, target_col='target')
    test_transformed = pipeline.transform(test_df, target_col='target')
    
    # Selecionar features para o modelo (excluir target)
    X_train_autofe = train_transformed.drop(columns=['target'])
    X_test_autofe = test_transformed.drop(columns=['target'])
    
    # Avaliar modelo com AutoFE
    autofe_results = evaluate_model(
        X_train_autofe, train_transformed['target'], 
        X_test_autofe, test_transformed['target'],
        feature_names=X_train_autofe.columns,
        title="Modelo com AutoFE"
    )
    results['Modelo com AutoFE'] = autofe_results
    
    # 5. Comparar e visualizar resultados
    visualize_feature_importance(results, OUTPUT_DIR)
    compare_results(results, OUTPUT_DIR)
    
    # 6. Resumo final
    print("\n" + "="*80)
    print(" CONCLUSÃO ".center(80, "="))
    print("="*80)
    
    # Calcular melhoria com AutoFE
    base_acc = results['Modelo Base']['accuracy']
    autofe_acc = results['Modelo com AutoFE']['accuracy']
    improvement = (autofe_acc - base_acc) * 100
    
    print(f"\nMelhoria de acurácia com AutoFE: {improvement:.2f}%")
    print(f"  - Modelo Base: {base_acc:.4f}")
    print(f"  - Modelo com AutoFE: {autofe_acc:.4f}")
    
    print("\nCaracterísticas importantes extraídas de dados temporais pelo AutoFE:")
    date_features = [col for col in results['Modelo com AutoFE']['importance']['feature'] 
                      if any(term in col.lower() for term in ['data', 'date', 'dia', 'mes', 'year', 'month', 'week', 'hour'])]
    
    for feat in date_features[:5]:  # Mostrar as 5 mais importantes
        importance = results['Modelo com AutoFE']['importance']
        feat_importance = importance[importance['feature'] == feat]['importance'].values[0]
        print(f"  - {feat}: {feat_importance:.4f}")
    
    print(f"\nResultados salvos em: {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()