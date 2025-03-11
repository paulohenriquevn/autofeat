# Referência Técnica do CAFE

## PreProcessor
O `PreProcessor` é o componente  pela limpeza de dados é um **passo fundamental** antes de aplicar técnicas de engenharia de features. Dados inconsistentes, incompletos ou duplicados podem levar a modelos menos precisos e enviesados. O processo de limpeza de dados envolve as seguintes etapas:

### **2.1 Identificação e Correção de Erros**
Antes de qualquer transformação, devemos verificar se há **valores inconsistentes, erros de digitação ou formatos incompatíveis**. Algumas estratégias incluem:
- **Correção de tipos de dados:** Converter datas para o formato correto, padronizar strings e garantir que variáveis numéricas estejam no formato adequado.
- **Detecção de valores anômalos:** Identificar valores muito altos ou muito baixos que possam indicar erros de entrada.

### **2.2 Tratamento de Valores Ausentes**
Valores ausentes podem impactar significativamente a performance do modelo. Existem várias abordagens para lidar com esse problema:

#### **Quando Usar?**
- **Imputação pela média ou mediana:** Quando os dados têm **distribuição normal** e os valores ausentes não representam um viés no conjunto de dados.
- **Imputação por valor arbitrário:** Quando a ausência de dados pode ter um **significado semântico**.

### **2.3 Remoção de Registros ou Features Duplicadas**
Dados duplicados podem distorcer análises e aumentar o peso de algumas observações desnecessariamente. As principais estratégias são:
- **Remover registros duplicados**.
- **Remover colunas altamente correlacionadas** para identificar redundância).

Ao garantir que os dados estejam **limpos e estruturados**, evitamos distorções nos modelos e garantimos um **aprendizado mais eficiente**.

### **2.4 Como validar a limpeza de seus dados?**

O processo de limpeza de dados envolve muitas etapas para identificar e corrigir entradas de problemas. A primeira etapa é analisar os dados para identificar erros. Isso pode implicar o uso de ferramentas de análise qualitativa que utilizam regras, padrões e restrições para identificar valores inválidos. A próxima etapa é remover ou corrigir erros. 

As etapas de limpeza de dados geralmente incluem a correção de:

- **Dados duplicados: descarte informações duplicadas**
- **Dados ausentes: sinalize e descarte ou insira os dados ausentes**
- **Erros estruturais: corrija erros tipográficos e outras inconsistências e faça os dados cumprirem um padrão ou convenção comum**

## FeatureEngineer

O `FeatureEngineer` é o componente responsável pela engenharia de features.

### **Papel das Features no Machine Learning**
- **Quanto melhores as features, melhor o modelo!** A qualidade das features muitas vezes importa mais do que o próprio algoritmo escolhido.
- **Feature Engineering** (Engenharia de Features) é a técnica de criar, transformar e selecionar features relevantes para otimizar o desempenho do modelo.

### **Tipos de Features**
1. **Numéricas** (Contínuas ou Discretas)
   - Exemplo: idade, altura, preço, número de produtos comprados.
  
2. **Categóricas** 
   - Exemplo: cor do carro (vermelho, azul), tipo de imóvel (casa, apartamento).
   - Podem ser transformadas em números usando técnicas como **One-Hot Encoding**.

3. **Binárias (Booleanas)**
   - Exemplo: "Possui garagem?" (Sim/Não = 1/0).

4. **Textuais**
   - Exemplo: Comentários de clientes (podem ser transformados em vetores usando TF-IDF ou embeddings).

5. **Temporais**
   - Exemplo: Data da compra (pode ser decomposta em dia da semana, mês, estação do ano).

6. **Derivadas (Engenharia de Features)**
   - Criadas a partir das features originais para melhorar o desempenho do modelo.
   - Exemplo: Criar uma nova variável **"preço por m²"** a partir de "preço total" e "área".

### **Etapas do Feature Engineering**

O processo de Feature Engineering pode ser dividido em 6 etapas principais:

 - Transformações de Features
 - Criação de Novas Features
 - Seleção e Redução de Features
 - Normalização e Padronização

## **Transformações de Features**
**Objetivo:** Melhorar a relação entre as variáveis e a variável alvo, facilitando o aprendizado do modelo.

**Principais Transformações:**
#### **Codificação de Variáveis Categóricas**
   - **One-Hot Encoding** para variáveis com poucas categorias
   - **Target Encoding** para problemas de classificação

#### **Transformações Matemáticas**
   - **Log Transform** (para dados enviesados)
   - **Box-Cox Transformation** (para normalizar distribuições)
   - **Polynomial Features** (para incluir interações entre variáveis)

#### **Transformações de Texto (NLP)**
   - Tokenização, Stemização e Lematização
   - Vetorização com TF-IDF ou Embeddings (Word2Vec, BERT)

### **Criação de Novas Features**
**Objetivo:** Gerar **novas variáveis** que representem melhor o comportamento dos dados e melhorem o desempenho do modelo.

**Exemplos de Feature Engineering:**
#### **Combinação de Variáveis**
   - Criar **média móvel**, razão (`preço/unitário`), produto, etc.

#### **Extração de Informações de Datas**
   - Dia da semana, Mês, Estação do ano, Feriado

#### **Engenharia de Features em Séries Temporais**
   - Lags (valores passados como input)
   - Rolling Mean (médias móveis)

### **Seleção e Redução de Features**
**Objetivo:** Remover variáveis redundantes ou irrelevantes para evitar overfitting e melhorar a eficiência do modelo.

 **Técnicas de Seleção de Features:**
#### **Métodos Estatísticos**
   - Seleção baseada em **correlação** (`df.corr()`)
   - Testes estatísticos como **ANOVA, Chi-Square**

#### **Feature Importance**
   - Modelos como **Random Forest e XGBoost** ajudam a identificar as variáveis mais relevantes

#### **Redução de Dimensionalidade**
   - PCA (Principal Component Analysis) para compactar variáveis mantendo a variância
   - t-SNE/UMAP para visualização em 2D

### **Normalização e Padronização**
**Objetivo:** Normalizar os dados para que todas as variáveis sejam comparadas de forma justa.

#### **Técnicas de Normalização e Padronização**
   - **Correção de outliers**
    - Remover outliers extremos usando **Z-score** ou **IQR (Interquartile Range)**
    - Transformações como **Log Scaling** para reduzir impacto de valores extremos
   - **Normalização e Padronização**
    - **Min-Max Scaling** (para dados entre `[0,1]`)
    - **Z-score Standardization** (para dados distribuídos normalmente)

## Módulo Explorer

O `Explorer` é responsável por testar diversas configurações de transformação e identificar as mais eficazes para os dados. **Navegar no espaço de possíveis transformações** e selecionar as melhores combinações de features.

#### ***Árvore de Transformações**
**Árvore de Transformações** representa diferentes sequências de transformações aplicadas às features existentes. Essa abordagem permite:
- **Explorar um grande número de possibilidades de novas features** sem necessidade de intervenção manual.
- **Evitar redundância e sobrecarga de dados**, mantendo apenas as features mais informativas.
- **Facilitar a interpretação dos modelos**, permitindo a rastreabilidade de como cada feature foi criada.

### **Componentes do Explorer**

- **TransformationTree**: Constrói e mantém uma estrutura hierárquica de transformações
- **HeuristicSearch**: Executa busca eficiente para encontrar as melhores transformações
- **FeatureRefinement**: Elimina redundâncias e prioriza features interpretáveis