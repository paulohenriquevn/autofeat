# CAFE (Component Automated Feature Engineer)

Um sistema abrangente para automatizar o processamento de dados e a engenharia de features em projetos de machine learning.

## Instalação

```bash
pip install cafe-autofe
```

## Visão Geral

O CAFE (Component Automated Feature Engineer) é composto por quatro componentes principais:

1. **PreProcessor**: Responsável pela limpeza e transformação inicial dos dados brutos.
2. **FeatureEngineer**: Gera e seleciona features de alta qualidade.
3. **PerformanceValidator**: Avalia e compara a performance de modelos treinados com diferentes conjuntos de dados.
4. **DataPipeline**: Integra os componentes em um pipeline unificado.
5. **Explorer**: Busca automaticamente a melhor configuração para um determinado conjunto de dados.

## Exemplo Rápido

```python
import pandas as pd
from cafe import DataPipeline

# Carregar dados
df = pd.read_csv('dataset.csv')

# Criar pipeline com configurações padrão
pipeline = DataPipeline()

# Ajustar e transformar os dados
df_transformed = pipeline.fit_transform(df, target_col='target')

# Salvar o pipeline
pipeline.save('pipeline_model')
```

## Usando o Explorer para Encontrar a Melhor Configuração

```python
import pandas as pd
from cafe import Explorer, DataPipeline

# Carregar dados
df = pd.read_csv('dataset.csv')

# Criar Explorer
explorer = Explorer(target_col='target')

# Encontrar a melhor configuração
best_data = explorer.analyze_transformations(df)
best_config = explorer.get_best_pipeline_config()

# Criar e ajustar um pipeline com a melhor configuração
pipeline = DataPipeline(
    preprocessor_config=best_config.get('preprocessor_config', {}),
    feature_engineer_config=best_config.get('feature_engineer_config', {})
)

# Transformar os dados
df_transformed = pipeline.fit_transform(df, target_col='target')
```

## Componentes

### PreProcessor

O componente `PreProcessor` é responsável pelo pré-processamento dos dados brutos, realizando operações como:

- Tratamento de valores ausentes
- Detecção e tratamento de outliers
- Codificação de variáveis categóricas
- Normalização/padronização de dados
- Processamento de colunas de data/hora

### FeatureEngineer

O componente `FeatureEngineer` é responsável pela geração e seleção de features, implementando operações como:

- Geração de features polinomiais
- Remoção de features altamente correlacionadas
- Redução de dimensionalidade
- Seleção de features baseada em importância

### PerformanceValidator

O componente `PerformanceValidator` é responsável por avaliar e comparar a performance de modelos treinados com diferentes configurações:

- Compara performance entre dados originais e transformados
- Utiliza validação cruzada para estimativas robustas
- Decide automaticamente qual conjunto de dados usar

### DataPipeline

O componente `DataPipeline` integra o `PreProcessor`, `FeatureEngineer` e `PerformanceValidator` em um único pipeline unificado:

- Combina todas as etapas de processamento de forma sequencial
- Gerencia o fluxo de dados entre os componentes
- Fornece uma API simplificada para o usuário final

### Explorer

O componente `Explorer` automatiza a busca pela melhor configuração para um conjunto de dados específico:

- Testa diferentes combinações de configurações
- Avalia cada configuração usando heurísticas
- Retorna a melhor configuração encontrada

## Documentação Completa

Para documentação completa, visite [GitHub](https://github.com/yourusername/cafe-autofe) ou a [documentação online](https://cafe-autofe.readthedocs.io/).

## Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](https://github.com/yourusername/cafe-autofe/blob/main/LICENSE) para detalhes.