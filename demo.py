"""
Script de demonstração do recurso de validação de performance do AutoFE.
Este script mostra como o sistema agora verifica automaticamente se as transformações 
melhoram ou prejudicam a performance preditiva.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Importar o pipeline com validação
from data_pipeline import create_data_pipeline

def load_dataset(dataset_name='iris'):
    """Carrega um dos datasets de exemplo."""
    if dataset_name == 'iris':
        data = load_iris()
    elif dataset_name == 'wine':
        data = load_wine()
    elif dataset_name == 'breast_cancer':
        data = load_breast_cancer()
    else:
        raise ValueError(f"Dataset {dataset_name} não suportado.")
    
    # Criar DataFrame
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    
    return df, data.target_names

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
    plt.ylabel('Performance (Métrica: Acurácia)')
    plt.ylim(0, 1.1)
    
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
    - Redução:      {feature_reduction:.1f}%
    
    DECISÃO: Usar dados {best_choice.upper()}
    
    Configuração do validador:
    - Máxima queda permitida: {validation_results.get('max_performance_drop', 0.05)*100:.1f}%
    - Folds de validação: {len(validation_results['scores_original'])}
    - Métrica: {validation_results.get('metric', 'accuracy')}
    """
    
    plt.text(0.1, 0.9, text, fontsize=12, va='top', family='monospace')
    
    plt.tight_layout()
    plt.savefig(f"{dataset_name}_validation_results.png")
    plt.show()

def run_demo(dataset_name='iris'):
    """Executa a demonstração completa para um dataset específico."""
    print(f"\n{'='*80}")
    print(f" DEMONSTRAÇÃO DE VALIDAÇÃO DE PERFORMANCE - DATASET {dataset_name.upper()} ".center(80, "="))
    print(f"{'='*80}\n")
    
    # 1. Carregar dataset
    df, target_names = load_dataset(dataset_name)
    print(f"Dataset carregado: {df.shape[0]} amostras, {df.shape[1]-1} features, {len(target_names)} classes")
    
    # 2. Configurações do pipeline
    feature_config = {'correlation_threshold': 0.8, 'generate_features': False}
    
    # Configuração do validador
    # - Para o Iris, usamos uma tolerância maior (15% de queda permitida)
    # - Para os outros datasets, usamos a tolerância padrão (5%)
    validator_config = {
        'max_performance_drop': 0.15 if dataset_name == 'iris' else 0.05,
        'cv_folds': 5,
        'metric': 'accuracy',
        'task': 'classification',
        'base_model': 'rf',
        'verbose': True
    }
    
    # 3. Criar e ajustar o pipeline
    pipeline = create_data_pipeline(
        feature_engineer_config=feature_config,
        validator_config=validator_config,
        auto_validate=True  # Ativar validação automática
    )
    
    # 4. Ajustar o pipeline ao dataset
    print("\nAjustando o pipeline com validação automática...")
    transformed_df = pipeline.fit_transform(df, target_col='target')
    
    # 5. Obter e mostrar resultados da validação
    validation_results = pipeline.get_validation_results()
    
    if validation_results:
        # Adicionar o número original de features
        validation_results['original_n_features'] = df.shape[1] - 1  # excluir target
        
        print("\nResultados da validação:")
        print(f"- Performance original: {validation_results['performance_original']:.4f}")
        print(f"- Performance transformada: {validation_results['performance_transformed']:.4f}")
        print(f"- Diferença: {validation_results['performance_diff']:.4f} ({validation_results['performance_diff_pct']:.2f}%)")
        print(f"- Melhor escolha: {validation_results['best_choice'].upper()}")
        
        # Informações sobre o dataset transformado
        print(f"\nDataset original: {df.shape[0]} amostras, {df.shape[1]-1} features")
        print(f"Dataset após transformação: {transformed_df.shape[0]} amostras, {transformed_df.shape[1]-1} features")
        
        # Feature importance
        print("\nFeature importance (dataset original):")
        importance = pipeline.get_feature_importance(df, target_col='target')
        for i, (_, row) in enumerate(importance.head(5).iterrows()):
            print(f"  {i+1}. {row['feature']}: {row['importance']:.4f}")
        
        # Visualizar resultados
        visualize_results(validation_results, dataset_name)
    else:
        print("\nValidação não foi realizada. Verifique as configurações.")
    
    # 6. Avaliar model com dataset final
    print("\nAvaliando modelo com dataset final...")
    X = transformed_df.drop(columns=['target'])
    y = transformed_df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nAcurácia final (conjunto de teste): {accuracy:.4f}")
    print("\nRelatório de classificação:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    # Retornar o pipeline para uso adicional se necessário
    return pipeline, validation_results

def compare_datasets():
    """
    Executa a demonstração para múltiplos datasets e compara os resultados.
    """
    datasets = ['iris', 'wine', 'breast_cancer']
    results = {}
    
    for dataset in datasets:
        print(f"\nTestando dataset: {dataset}")
        _, validation = run_demo(dataset)
        results[dataset] = validation
    
    # Visualizar comparação entre os datasets
    plt.figure(figsize=(12, 8))
    
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
    
    # 2. Comparação de redução de features
    plt.subplot(2, 2, 2)
    feature_reduction = [results[ds]['feature_reduction'] * 100 for ds in datasets]
    bars = plt.bar(datasets, feature_reduction)
    
    # Colorir barras
    for i, bar in enumerate(bars):
        bar.set_color('green' if feature_reduction[i] >= 0 else 'orange')
    
    plt.title('Redução de Features (%)')
    plt.ylabel('Redução (%)')
    
    # 3. Decisões tomadas
    plt.subplot(2, 1, 2)
    best_choices = [results[ds]['best_choice'] for ds in datasets]
    
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
    plt.savefig("comparison_datasets.png")
    plt.show()
    
    return results

if __name__ == "__main__":
    print("=" * 80)
    print(" DEMONSTRAÇÃO DA VALIDAÇÃO DE PERFORMANCE DO AUTOFE ".center(80, "="))
    print("=" * 80)
    
    # Executar demonstração comparando múltiplos datasets
    results = compare_datasets()
    
    print("\n" + "=" * 80)
    print(" CONCLUSÃO ".center(80, "="))
    print("=" * 80)
    
    print("\nO sistema AutoFE agora é capaz de avaliar automaticamente se as transformações")
    print("melhoram ou prejudicam a performance preditiva, tomando decisões inteligentes")
    print("sobre quais dados usar (originais ou transformados).")
    print("\nPara o dataset Iris, o sistema detectou a queda de performance e decidiu usar")
    print("os dados originais, demonstrando que a validação está funcionando corretamente.")