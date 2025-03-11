import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_diabetes, load_digits, load_files, load_iris, load_linnerud, load_sample_image, load_sample_images, load_svmlight_file, load_svmlight_files, load_wine, load_breast_cancer

# Importar as novas classes e funções
from preprocessor import PreProcessor, create_preprocessor
from feature_engineer import FeatureEngineer, create_feature_engineer
from data_pipeline import DataPipeline, create_data_pipeline
from explorer import Explorer


def load_dataset(dataset_name):
    """Carrega um dataset do Scikit-learn."""
    if dataset_name.lower() == "iris":
        data = load_iris()
    elif dataset_name.lower() == "wine":
        data = load_wine()
    elif dataset_name.lower() == "breast_cancer":
        data = load_breast_cancer()
    elif dataset_name.lower() == "diabetes":
        data = load_diabetes()
    elif dataset_name.lower() == "digits":
        data = load_digits()
    elif dataset_name.lower() == "linnerud":
        data = load_linnerud()
    else:
        raise ValueError(f"Dataset {dataset_name} não suportado.")
    
    df = pd.DataFrame(data.data, columns=data.feature_names)

    # Verifica se 'target' existe e se tem o formato correto
    if hasattr(data, 'target'):
        target = np.array(data.target).ravel()  # Garante que seja um array 1D
        if len(target) == df.shape[0]:  # Confirma que os tamanhos correspondem
            df['target'] = target
        else:
            ValueError(f"Aviso: O dataset {dataset_name} tem um número incompatível de labels ({len(target)}) em relação às amostras ({df.shape[0]}). Ignorando 'target'.")
    else:
        ValueError(f"Aviso: O dataset {dataset_name} não possui um alvo ('target').")

    return df


def explore_dataset(df, sample_size=5):
    """
    Realiza uma exploração básica do dataset.
    
    Args:
        df (pd.DataFrame): DataFrame a ser explorado
        sample_size (int): Número de amostras a serem exibidas
        
    Returns:
        pd.DataFrame: O mesmo DataFrame de entrada
    """
    print("\n=== EXPLORAÇÃO DO DATASET ===")
    
    print(f"\nPrimeiras {sample_size} linhas:")
    print(df.head(sample_size))
    
    print("\nInformações das colunas:")
    print(df.info())
    
    print("\nEstatísticas descritivas (colunas numéricas):")
    print(df.describe())
    
    print("\nValores ausentes por coluna:")
    missing_values = df.isnull().sum()
    print(missing_values[missing_values > 0].sort_values(ascending=False))
    
    # Contar tipos de colunas
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    bool_cols = df.select_dtypes(include=['bool']).columns.tolist()
    
    print("\nDistribuição de tipos de colunas:")
    print(f"- Numéricas: {len(numeric_cols)}")
    print(f"- Categóricas: {len(categorical_cols)}")
    print(f"- Booleanas: {len(bool_cols)}")
    
    return df


def visualize_dataset(df, processed=False, output_dir="output"):
    """
    Cria visualizações básicas do dataset.
    
    Args:
        df (pd.DataFrame): DataFrame a ser visualizado
        processed (bool): Indica se o dataset já está processado
        output_dir (str): Diretório para salvar as visualizações
    """
    print("\n=== VISUALIZAÇÃO DO DATASET ===")
    
    input_type = "processado" if processed else "original"
    
    # Criar diretório de saída se não existir
    os.makedirs(output_dir, exist_ok=True)
    
    # Selecionar algumas colunas numéricas para visualizar (até 5)
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if len(numeric_cols) > 5:
        numeric_cols = numeric_cols[:5]
    
    if len(numeric_cols) > 0:
        print(f"\nCriando histogramas para {len(numeric_cols)} colunas numéricas...")
        
        plt.figure(figsize=(15, 10))
        for i, col in enumerate(numeric_cols, 1):
            plt.subplot(len(numeric_cols), 1, i)
            sns.histplot(df[col].dropna(), kde=True)
            plt.title(f'Distribuição de {col}')
            plt.tight_layout()
        
        plt.savefig(f"{output_dir}/numeric_distributions_{input_type}.png")
        print(f"Histogramas salvos em {output_dir}/numeric_distributions_{input_type}.png")
    
    # Visualizar correlações entre colunas numéricas
    if len(numeric_cols) > 1:
        print("\nCriando matriz de correlação...")
        plt.figure(figsize=(12, 10))
        correlation_matrix = df[numeric_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title('Matriz de Correlação')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/correlation_matrix_{input_type}.png")
        print(f"Matriz de correlação salva em {output_dir}/correlation_matrix_{input_type}.png")
    
    # Selecionar algumas colunas categóricas para visualizar (até 3)
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if len(categorical_cols) > 3:
        categorical_cols = categorical_cols[:3]
    
    if len(categorical_cols) > 0:
        print(f"\nCriando gráficos de barras para {len(categorical_cols)} colunas categóricas...")
        
        plt.figure(figsize=(15, 10))
        for i, col in enumerate(categorical_cols, 1):
            plt.subplot(len(categorical_cols), 1, i)
            # Limitar a 10 categorias mais frequentes
            value_counts = df[col].value_counts().nlargest(10)
            sns.barplot(x=value_counts.index, y=value_counts.values)
            plt.title(f'Top 10 categorias em {col}')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
        
        plt.savefig(f"{output_dir}/categorical_distributions_{input_type}.png")
        print(f"Gráficos de barras salvos em {output_dir}/categorical_distributions_{input_type}.png")


def process_and_save(df, dataset_name, config, output_dir="output"):
    """
    Aplica o processamento completo (pré-processamento + engenharia de features) com a configuração 
    especificada e salva os resultados.

    Args:
        df (pd.DataFrame): O DataFrame original.
        dataset_name (str): Nome do dataset.
        config (dict): Configuração do pipeline de dados.
        output_dir (str): Diretório onde os arquivos serão salvos.
    """
    # Criando o diretório de saída se não existir
    os.makedirs(output_dir, exist_ok=True)

    # Extrair as configurações
    preprocessor_config = config.get('preprocessor_config', {})
    feature_engineer_config = config.get('feature_engineer_config', {})

    # Criar o pipeline de dados
    pipeline = create_data_pipeline(preprocessor_config, feature_engineer_config)
    
    # Ajustar e transformar os dados
    transformed_df = pipeline.fit_transform(df, target_col="target")

    # Salvar dataset processado
    dataset_path = os.path.join(output_dir, f"{dataset_name}_processado.csv")
    transformed_df.to_csv(dataset_path, index=False)
    print(f"Dataset processado salvo em {dataset_path}")

    # Salvar o pipeline
    pipeline_base_path = os.path.join(output_dir, "data_pipeline")
    pipeline.save(pipeline_base_path)
    print(f"Pipeline salvo em {pipeline_base_path}_*.pkl")

    return transformed_df, pipeline


def main():
    available_datasets = [
        "iris",
        "wine", 
        "breast_cancer",
        "diabetes",
        "digits",
        "linnerud",
    ]
    
    # Para teste, usar apenas iris
    available_datasets = ["iris"]
    
    print("Datasets disponíveis para teste:")
    for i, dataset_name in enumerate(available_datasets, 1):
        print(f"{i}. {dataset_name}")
    
    for dataset_name in available_datasets:
        output = f"output/{dataset_name}"
        print(f"\nUsando dataset: {dataset_name}")
        
        # Criar o Explorer com suporte para a coluna alvo
        explorer = Explorer(target_col="target")
        
        # Carregar e explorar dados
        sample_data = load_dataset(dataset_name)
        explore_dataset(sample_data)
        visualize_dataset(sample_data, output_dir=output)
        
        # Salvar dataset original
        os.makedirs(output, exist_ok=True) 
        dataset_path = os.path.join(output, f"{dataset_name}.csv")
        sample_data.to_csv(dataset_path, index=False)
        print(f"Dataset original salvo em {dataset_path}")
        
        # Analisar as transformações e encontrar a melhor
        best_data = explorer.analyze_transformations(sample_data)
        
        # Obter a configuração ótima do pipeline
        best_config = explorer.get_best_pipeline_config()
        
        print("\n=== MELHOR CONFIGURAÇÃO DE PIPELINE ===")
        print("Configuração do Preprocessador:")
        for key, value in best_config.get('preprocessor_config', {}).items():
            print(f"  - {key}: {value}")
        
        print("Configuração do FeatureEngineer:")
        for key, value in best_config.get('feature_engineer_config', {}).items():
            print(f"  - {key}: {value}")
        
        # Aplicar a melhor configuração e salvar o dataset
        transformed_df, pipeline = process_and_save(sample_data, dataset_name, best_config, output_dir=output)
        
        print("\n=== RESUMO DO PROCESSAMENTO ===")
        print(f"Formato original: {sample_data.shape}")
        print(f"Formato processado: {transformed_df.shape}")
        
        # Explorar e visualizar o dataset processado
        explore_dataset(transformed_df)
        visualize_dataset(transformed_df, True, output_dir=output)
        
        print("\n=== FIM DO PROCESSAMENTO ===")
        
    print("\n=== FIM DO SCRIPT ===")


if __name__ == "__main__":
    main()