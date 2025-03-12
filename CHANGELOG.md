# Changelog

Todas as mudanças notáveis neste projeto serão documentadas neste arquivo.

O formato é baseado em [Keep a Changelog](https://keepachangelog.com/pt-BR/1.0.0/),
e este projeto adere ao [Versionamento Semântico](https://semver.org/lang/pt-BR/spec/v2.0.0.html).

## [0.3.0] - 2025-03-15

### Adicionado
- Transformadores avançados no módulo `PreProcessor`:
  - Binarizer para transformação de features numéricas em valores binários
  - KernelCenterer para centralização no espaço de kernel
  - MaxAbsScaler para escalonamento pelo valor máximo absoluto
  - Normalizer para normalização de amostras para norma unitária
  - PowerTransformer com métodos 'yeo-johnson' e 'box-cox' para aproximar distribuição normal
  - QuantileTransformer para transformação baseada em quantis
  - KBinsDiscretizer para discretização de features contínuas em intervalos
  - FunctionTransformer para aplicação de funções customizadas
  - LabelBinarizer, LabelEncoder e MultiLabelBinarizer para codificação de rótulos
  - TargetEncoder (implementação própria) para substituição de variáveis categóricas pela média do target
- Nova interface unificada de configuração para todos os transformadores
- Parâmetros específicos para cada tipo de transformador (ex: quantile_n_quantiles, binarizer_threshold)
- Suporte para aplicação de múltiplos transformadores em sequence
- Explorer significativamente aprimorado:
  - Teste automático de dezenas de combinações diferentes de transformadores
  - Suporte a todos os novos transformadores adicionados
  - Método `get_transformation_statistics()` para análise detalhada das transformações
  - Método `visualize_transformations()` para visualização da árvore de transformações
  - Método `export_transformation_graph()` para exportação em formato GraphML
  - Método `get_feature_importance_analysis()` para comparação de features antes e depois das transformações
- Script de demonstração avançado para testar os novos transformadores
- Capacidade de leitura de datasets locais em vários formatos (CSV, Excel, JSON, Parquet, etc.)

### Melhorado
- Preservação de nomes de features e metadados em todas as transformações
- Documentação expandida para todos os novos transformadores
- Melhor detecção e tratamento automático de tipos de dados (numéricos, categóricos, datetime)
- Aprimoramento da heurística de avaliação no Explorer para melhor seleção de transformações
- Relatórios detalhados sobre as transformações e suas estatísticas
- Interface de linha de comando para execução de demonstrações
- Integração aprimorada entre o Explorer e o resto do framework

### Corrigido
- Tratamento mais robusto de valores ausentes em diversos tipos de transformações
- Melhor manipulação de valores extremos em transformadores sensíveis a outliers
- Preservação consistente da coluna target durante transformações complexas
- Manipulação adequada de tipos de dados mistos (numéricos, categóricos, datetime)
- Resolução de conflitos em transformações combinadas

## [0.2.0] - 2025-03-11

### Adicionado
- Implementação de métodos avançados de seleção de features no componente `FeatureEngineer`
  - Adicionado suporte para `SelectKBest`
  - Adicionado suporte para `SelectFromModel`
  - Adicionado suporte para `SelectPercentile`
  - Adicionado suporte para métodos estatísticos: `SelectFwe`, `SelectFpr`, `SelectFdr`
  - Adicionado suporte para múltiplas funções de pontuação: `mutual_info_classif`, `mutual_info_regression`, `f_classif`, `f_regression`, `chi2`
- Nova classe auxiliar `AdvancedFeatureSelector` que encapsula diferentes métodos de seleção sob uma interface unificada
- Novo parâmetro de configuração `feature_selection_params` para controle fino dos métodos de seleção
- Método `get_feature_importances()` para análise de importância de features

### Melhorado
- Preservação de nomes de features após transformações
- Melhor controle de dimensionalidade para evitar explosão de features
- Interface mais consistente para diferentes tipos de seleção
- Melhor compatibilidade com o ecossistema scikit-learn através da implementação de `SelectorMixin`
- Documentação expandida com exemplos detalhados de uso

### Corrigido
- Tratamento adequado de índices em DataFrames durante as transformações
- Preservação consistente das colunas alvo durante o ajuste e transformação

## [0.1.0] - 2025-01-15

### Adicionado
- Lançamento inicial do CAFE (Component Automated Feature Engineer)
- Componentes principais: `PreProcessor`, `FeatureEngineer`, `PerformanceValidator`, `DataPipeline`, `Explorer`
- Funcionalidades de pré-processamento: tratamento de valores ausentes, outliers, codificação de variáveis categóricas
- Funcionalidades básicas de engenharia de features: remoção de alta correlação, PCA, seleção simples de features
- Sistema de validação de performance que compara dados originais vs. transformados
- Explorador para busca automática de melhores configurações de pipeline
- Scripts de exemplo e documentação inicial