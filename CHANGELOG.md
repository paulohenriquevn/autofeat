# Changelog

Todas as mudanças notáveis neste projeto serão documentadas neste arquivo.

O formato é baseado em [Keep a Changelog](https://keepachangelog.com/pt-BR/1.0.0/),
e este projeto adere ao [Versionamento Semântico](https://semver.org/lang/pt-BR/spec/v2.0.0.html).

## [0.1.3] - 2025-03-115

### Adicionado
- Classes para relatórios avançados e visualizações interativas:
  - `ReportDataPipeline`: Nova classe dedicada à geração de relatórios detalhados sobre dados e transformações
  - `ReportVisualizer`: Nova classe especializada em visualizações gráficas para análise de dados

### Relatórios Avançados (`ReportDataPipeline`)
- Método `get_missing_values()` para análise detalhada de valores ausentes com recomendações personalizadas
- Método `get_outliers()` para identificação e análise estatística de outliers
- Método `get_feature_importance()` para avaliação da importância de features com categorização
- Método `get_transformations()` para análise detalhada das transformações aplicadas pelo pipeline
- Método `get_report_summary()` para resumo conciso das principais métricas e recomendações
- Método `generate_report()` para criação de relatórios completos em formato texto

### Visualizações Interativas (`ReportVisualizer`)
- Método `visualize_missing_values()` para gráficos de barras de valores ausentes
- Método `visualize_outliers()` para boxplots com identificação de limites e valores atípicos
- Método `visualize_feature_importance()` para gráficos de barras com código de cores por categoria de importância
- Método `visualize_transformations()` para painéis comparativos de performance e dimensionalidade
- Método `visualize_data_distribution()` para histogramas com KDE para análise de distribuição
- Método `visualize_correlation_matrix()` para heatmaps de correlação e gráficos de correlação com o target

### Melhorado
- Separação clara de responsabilidades entre geração de relatórios e visualização
- Flexibilidade para uso independente de componentes (só relatórios ou só visualizações)
- Geração automática de recomendações baseadas em análise estatística
- Compatibilidade com diferentes tipos de dados e cenários de transformação
- Integração transparente com componentes existentes do sistema CAFE

### Atualizado
- Scripts de exemplo atualizados para demonstrar o uso das novas funcionalidades
- Fluxo de trabalho mais transparente e documentado no processo de engenharia de features


## [0.1.1] - 2025-03-15

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

## [0.1.2] - 2025-03-11

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