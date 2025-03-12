# Changelog

Todas as mudanças notáveis neste projeto serão documentadas neste arquivo.

O formato é baseado em [Keep a Changelog](https://keepachangelog.com/pt-BR/1.0.0/),
e este projeto adere ao [Versionamento Semântico](https://semver.org/lang/pt-BR/spec/v2.0.0.html).

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