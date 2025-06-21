# 📚 Estudo de AI Engineering

> Descrição:  
> Este documento contém anotações sobre Engenharia de Inteligência Artificial, focando inicialmente em Machine Learning, suas técnicas e o ciclo de vida dos modelos.

---

## 📑 Sumário

- [1. Introdução](#1-introdução)
- [2. Machine Learning](#2-machine-learning)
    - [2.1 Técnicas de Machine Learning](#21-técnicas-de-machine-learning)
    - [2.2 Ciclo de Vida de um Modelo de Machine Learning](#22-ciclo-de-vida-de-um-modelo-de-machine-learning)
    - [2.3 Ferramentas de Machine Learning](#23-machine-learning-tools)

- [3. Deep Learning](#3-deep-learning)
    - [3.1 Ferramentas de Deep Learning](#31-deep-learning-tools)


- [0. Links Úteis e Referências](#0-links-úteis-e-referências)

---

## 1. Introdução

A Engenharia de IA abrange a construção, treinamento, manutenção e monitoramento de sistemas de Inteligência Artificial, com forte ênfase em práticas de software e ciclo de vida de modelos.

---

## 2. Machine Learning

### 2.1 Técnicas de Machine Learning

| Técnica | O que é | Para que serve | Exemplo de Cenário de Uso |
|:---|:---|:---|:---|
| **Classificação** | Prever rótulos categóricos. | Identificar a qual classe um dado pertence. | Diagnosticar se uma doença está presente (Sim/Não). |
| **Regressão** | Prever valores contínuos. | Estimar uma quantidade numérica. | Previsão de preços de imóveis. |
| **Clusterização** | Agrupar dados semelhantes sem rótulos. | Descobrir grupos naturais nos dados. | Segmentação de clientes em marketing. |
| **Redução de Dimensionalidade** | Simplificar dados mantendo características principais. | Aumentar performance e facilitar visualização. | PCA para reduzir variáveis financeiras. |
| **Detecção de Anomalias** | Identificar padrões fora do normal. | Detectar desvios ou irregularidades. | Fraude em transações bancárias. |
| **Sistemas de Recomendação** | Sugerir itens relevantes para usuários. | Personalizar a experiência do usuário. | Recomendação de filmes na Netflix. |
| **Regras de Associação** | Encontrar relações entre variáveis. | Descobrir padrões de coocorrência. | Analisar compras: quem compra pão também compra leite. |
| **Previsão de Séries Temporais** | Prever dados que variam no tempo. | Antecipar tendências e comportamentos futuros. | Previsão de demanda de energia elétrica. |


---

### 2.2 Ciclo de Vida de um Modelo de Machine Learning

O ciclo de vida de um modelo de Machine Learning normalmente segue as seguintes etapas:

1. **Definição do Problema**
   - Entender claramente o objetivo do projeto.
   - Exemplo: Como cliente de produtos de beleza, gostaria de receber recomendações de outros produtos com base no meu histórico de compras para poder atender às minhas necessidades de cuidados com a pele e melhorar a saúde geral da minha pele.

2. **Coleta e Análise de Dados**
   - Obter, explorar e compreender os dados disponíveis.
   - Exemplo: Devo determinar que tipo de dados a empresa possui e identificar as fontes de onde eles virão. Isso pode incluir dados do usuário, como demografia, histórico de compras e qualquer coisa relacionada a transações concluídas. Também posso obter dados do produto, ou seja, o inventário de produtos e o que eles fazem, seus ingredientes quão populares eles são, suas avaliações de clientes, e assim por diante.

3. **Preparação dos Dados (Data Preparation)**
   - Limpeza, transformação, seleção de características (features).
   - Exemplo: Na maioria das vezes, dados de múltiplas fontes conterão erros, formatações diferentes, e dados ausentes. Este processo se sobrepõe ao processo de coleta de dados pois podem ser realizados em conjunto. O foco aqui é preparar uma versão quase final dos dados . Eu precisarei garantir que os dados sejam limpos para filtrar dados irrelevantes, incluindo ATA e UO.

4. **Escolha ou desenvolva o Modelo**
   - Selecionar algoritmos de ML adequados ao problema.
   - Exemplo: Regressão logística para classificação binária.

5. **Treinamento do Modelo**
   - Alimentar o algoritmo com dados para aprender padrões.
   - Exemplo: Treinar 80% dos dados e validar em 20%.

6. **Avaliação do Modelo**
   - Medir desempenho usando métricas apropriadas (accuracy, F1-score, AUC).
   - Exemplo: Avaliar se o modelo é melhor que um baseline.

7. **Ajuste de Hiperparâmetros (Hyperparameter Tuning)**
   - Otimizar parâmetros para melhorar a performance.
   - Exemplo: Usar Grid Search ou Random Search.

8. **Deploy do Modelo**
   - Implantar o modelo em ambiente de produção (APIs, apps, sistemas).
   - Exemplo: Subir o modelo no servidor com FastAPI.

9. **Monitoramento e Manutenção**
   - Monitorar performance ao longo do tempo e re-treinar se necessário.
   - Exemplo: Detectar drift de dados que prejudique as previsões.

10. **Documentação e Atualização**
    - Registrar todo o processo e atualizar o ciclo conforme mudanças.
    - Exemplo: Criar artefatos de versionamento de modelo.

---

### 2.3 Machine Learning Tools

#### Library
- **Pandas**: Usada para Manipulação de dados e análise.
- **Scikit-learn**: Que fornece uma ampla gama de algoritmos de Aprendizagem supervisionada e Aprendizagem não supervisionada para regressão linear.
- **SciPy**: SciPy, construído sobre o NumPy, é usado para computação científica e possui módulos para otimização, integração, regressão linear, e mais. Scikit-learn é usado para construir modelos clássicos de aprendizado de máquina, oferecendo um conjunto completo de algoritmos de classificação, regressão, agrupamento, e redução de dimensionalidade.
- **NumPy**: NumPy fornece suporte fundamental para aprendizado de máquina com cálculos numéricos eficientes em grandes, matrizes de dados multidimensionais. O Pandas é usado para análise de dados, visualização, limpeza, e preparação de dados para aprendizado de máquina.

#### Linguagens
- **Python**: Python é uma linguagem amplamente utilizada devido à sua extensa coleção de bibliotecas para análise e processamento de dados e sua facilidade no desenvolvimento de modelos de Aprendizado de máquina.
- **R**: R é outra linguagem popular para aprendizado estatístico que também contém muitas bibliotecas para exploração de dados e Aprendizado de máquina.
- **Julia**: Julia é uma LAN de alto desempenho com suporte para computação numérica paralela e distribuída, usada por pesquisadores de AC, AI, CI e ADM.
- **SCALA**: Scala é uma linguagem escalável, amplamente utilizada para processar big data e construir Pipeline de aprendizado de máquina.
- **JAVA**: Java é uma linguagem multiuso que suporta aplicações de aprendizado de máquina escaláveis implantadas em produção.
- **JAVA-SCRIPT**: E JavaScript é usado para executar modelos de Aprendizado de máquina em navegadores web para servir aplicações do lado do cliente.

#### Data Processing and analytics
- PostgreSQL: É um poderoso sistema de banco de dados relacional de código máquina. aberto baseado em SQL, uma linguagem projetada para armazenar, manipular e recuperar dados em bancos de dados. 
- Hadoop: É uma solução de código aberto, altamente escalável e baseada em disco para armazenamento e processamento em lote de grandes volumes de dados, incluindo ATA, AC e UO.
- Spark: É uma estrutura de processamento de dados distribuída em RAM para processamento de Big data em tempo real. Ele suporta Data frame e SQL e é mais rápido e fácil de usar do que o Hadoop.
- Apache Kafka:  é uma plataforma de streaming distribuída para construir pipelines de big data e executar análises em tempo real.
- Pandas: é uma biblioteca popular Python para explorar e manipular dados. Central para o Pandas é o Data frame, uma Estrutura de dados tabular para limpar e transformar Dados estruturados.

#### Data Visualization
- Matplotlib é uma biblioteca fundamental extensa para gerar gráficos personalizáveis e visualizações interativas.
- Seaborn é uma biblioteca baseada em Matplotlib. Ela fornece uma interface de alto nível para desenhar gráficos estatísticos atraentes e informativos.
- ggplot2 é um pacote de visualização de dados de código aberto em R. Ele permite que você construa e adicione elementos aos seus gráficos e camadas.
- Tableau é uma ferramenta de Business Intelligence( BI) para painéis de visualização de dados interativos.

---

## 3. Deep Learning

### 3.1 Deep Learning Tools

#### Deep Learning
- **TensorFlow**: é uma biblioteca de código aberto para computação numérica e aprendizado de máquina em larga escala.
- **Keras**: é uma biblioteca de Deep Learning fácil de usar para implementar redes neurais.
- **Theano**: é usado para definir, otimizar, e avaliar eficientemente expressões matemáticas envolvendo matrizes.
- **PyTorch**: é uma biblioteca de código aberto para aplicações de Deep Learning e visão computacional em NLP. Ela também permite a experimentação para testar ideias .

#### Computer Vision

- **OpenCV**:  ou Biblioteca de Visão Computacional de Código Aberto, é uma biblioteca para aplicações de visão computacional em tempo real, como detecção de objetos, classificação de imagens, e realidade aumentada.
- **Scikit-image**:  construído sobre o SciPy e compatível com o Pandas, oferece algoritmos de processamento de imagens, como filtros, segmentação, Extração de recursos, e operações morfológicas.
- **TorchVision**: parte do projeto PyTorch, consiste em conjuntos de dados populares, carregamento de imagens, arquiteturas de IA pré-treinadas, e transformações comuns de imagens para visão computacional.

#### NLP
- **NLTK**: O Natural Language Toolkit, ou NLTK, é uma biblioteca abrangente que oferece ferramentas de processamento de texto, tokenização, e derivação. 
- **Textblob**: é uma biblioteca para tarefas como marcação de partes do discurso, extração de frases nominais, Análise de sentimento e tradução.
- **Stanza**:  é uma biblioteca de NLP do grupo Stanford NLP com modelos pré-treinados precisos para muitas tarefas de NLP, incluindo marcação de partes do discurso, reconhecimento de entidades nomeadas, e análise de dependência.

---





---



---




## 0. Links Úteis e Referências

- [Coursera - Machine Learning by Andrew Ng](https://www.coursera.org/learn/machine-learning)
- [Google AI Education](https://ai.google/education/)
- [Machine Learning Crash Course - Google](https://developers.google.com/machine-learning/crash-course)
- Livro: *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* - Aurélien Géron
- Livro: *Pattern Recognition and Machine Learning* - Christopher Bishop

---