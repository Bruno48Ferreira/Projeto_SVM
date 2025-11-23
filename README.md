````markdown
# Projeto SVM – Previsão da Qualidade da Água

Este projeto implementa um modelo de **Máquinas de Vetores de Suporte (SVM)** para
prever a classe de qualidade da água (ótima, boa, regular, ruim, péssima) a partir
de parâmetros físico-químicos que compõem o Índice de Qualidade da Água (IQA).

Os dados utilizados foram preparados a partir das séries históricas da
Agência Nacional de Águas e Saneamento Básico (ANA), gerando o arquivo
`data/raw/agua_iqa.csv` com as colunas de entrada e a classe alvo.

> **Obs.:** este repositório contém a versão em linha de comando, sem a interface gráfica.

---

## Estrutura do projeto

```text
Projeto_SVM/
├─ data/
│  └─ raw/
│     └─ agua_iqa.csv                # dataset final usado no treinamento
├─ imgs/                             # imanges referentes aos gráficos gerados após treinamento
│  ├─ Figure_1_Matriz_Validacao_Linear.png
│  ├─ Figure_2_Matriz_Validacao_RBF.png
│  ├─ Figure_3_Matriz_Teste_RBF.png
│  ├─ Figure_4_Metricas_Validacao_Linear.png
│  ├─ Figure_5_Metricas_Validacao_RBF.png
│  └─ Figure_6_Metricas_Validacao_Teste_RBF.png
├─ src/
│  └─ build_dataset.py               # script opcional para gerar o agua_iqa.csv
├─ Artigo_SVM.pdf                    # artigo escrito com base neste projeto
└─ Qualidade_Agua_SVM.py             # script principal (pipeline SVM)
````

No código, o arquivo de dados é referenciado pela constante `DATA_PATH`
em `Qualidade_Agua_SVM.py`. Caso o caminho seja absoluto, é recomendável
ajustá-lo para um caminho relativo apontando para `data/raw/agua_iqa.csv`.

---

## Requisitos

* Python 3.10+ (recomendado)
* Bibliotecas:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

---

## Como executar o modelo SVM

Na raiz do projeto:

```bash
python Qualidade_Agua_SVM.py
```

O script irá:

1. Carregar e limpar o dataset a partir de `DATA_PATH`.
2. Dividir os dados em treino, validação e teste.
3. Normalizar os atributos (escala [0, 1]).
4. Treinar dois modelos SVM (kernel **linear** e **RBF**).
5. Avaliar ambos na base de validação.
6. Escolher automaticamente o melhor modelo (pela acurácia).
7. Avaliar o melhor modelo na base de teste.
8. Exibir:

   * Matrizes de confusão (validação e teste).
   * Gráficos de precisão, recall e F1-score por classe.
9. Realizar a previsão de uma **amostra de exemplo** e imprimir a classe prevista.

---

## Script de preparação do dataset (`src/build_dataset.py`)

O arquivo `src/build_dataset.py` pode ser utilizado (ou adaptado) para
construir o arquivo `data/raw/agua_iqa.csv` a partir dos arquivos brutos
baixados do portal de dados abertos da ANA.

Em resumo, ele:

1. Lê os CSVs de indicadores (OD, fósforo total, turbidez, DBO, IQA etc.).
2. Converte as séries históricas `MED_ano` para o formato “uma linha por estação/ano”.
3. Faz o *merge* dos parâmetros com o IQA numérico.
4. Converte o IQA para classes categóricas (ótima, boa, regular, ruim, péssima).
5. Gera o arquivo consolidado `agua_iqa.csv`.

> Caso você já tenha `data/raw/agua_iqa.csv` pronto, não é obrigatório
> executar o `build_dataset.py`.

---

## Descrição das funções principais (`Qualidade_Agua_SVM.py`)

### `carregar_dados(caminho_csv: str) -> pd.DataFrame`

Carrega o arquivo CSV contendo o dataset.

* Verifica se o arquivo existe.
* Lê o CSV com `pandas.read_csv`.
* Retorna um `DataFrame` com todos os dados originais.

### `limpar_dados(df: pd.DataFrame) -> pd.DataFrame`

Realiza a limpeza básica do dataset.

* Remove linhas totalmente vazias (`dropna(how="all")`).
* Remove linhas com quaisquer valores ausentes.
* Mantém apenas as colunas indicadas em `FEATURE_COLUMNS` e `TARGET_COLUMN`.
* Retorna um `DataFrame` pronto para ser dividido em treino/validação/teste.

### `dividir_dados(df: pd.DataFrame)`

Separa o dataset em três subconjuntos:

* Entrada (`X`) = colunas em `FEATURE_COLUMNS`.
* Saída (`y`) = coluna `TARGET_COLUMN`.

Passos:

1. Primeira divisão:

   * `train_test_split` com `test_size=0.3` →
     70% **treino** (`X_train`, `y_train`) e 30% temporário (`X_temp`, `y_temp`).
2. Segunda divisão:

   * `train_test_split` em `X_temp`, `y_temp` com `test_size=0.5` →
     15% **validação** (`X_val`, `y_val`) e 15% **teste** (`X_test`, `y_test`).
3. Utiliza `stratify` para manter a proporção de classes.

Retorna:

```python
X_train, X_val, X_test, y_train, y_val, y_test
```

### `normalizar_dados(X_train, X_val, X_test)`

Aplica a normalização Min–Max ([0, 1]) em todas as features.

* Cria um `MinMaxScaler(feature_range=(0, 1))`.
* Ajusta **apenas no treino**: `scaler.fit_transform(X_train)`.
* Aplica a transformação em validação e teste: `scaler.transform(...)`.
* Retorna os dados normalizados e o scaler para uso futuro:

```python
X_train_scaled, X_val_scaled, X_test_scaled, scaler
```

### `treinar_svm(X_train, y_train, kernel="rbf", C=1.0, gamma="scale")`

Treina um classificador SVM.

* Utiliza `sklearn.svm.SVC`.
* Parâmetros:

  * `kernel`: tipo de kernel (`"linear"`, `"rbf"`, etc.).
  * `C`: parâmetro de regularização.
  * `gamma`: parâmetro do kernel RBF.
* Ajusta o modelo com `.fit(X_train, y_train)`.
* Retorna o modelo treinado.

### `plot_metricas_avaliacao(y_true, y_pred, labels, titulo="Métricas de Avaliação")`

Gera o gráfico de barras com **precisão**, **recall** e **F1-score** por classe.

* Usa `classification_report(..., output_dict=True)` para extrair as métricas.
* Para cada classe em `labels`, coleta:

  * `precision`
  * `recall`
  * `f1-score`
* Plota três barras por classe com `matplotlib`.
* Configura e exibe o gráfico com `plt.show()`.

### `plot_matriz_confusao(cm, labels, titulo="Matriz de Confusão")`

Desenha uma matriz de confusão usando `seaborn.heatmap`.

* `cm`: matriz de confusão (gerada por `confusion_matrix`).
* `labels`: lista de rótulos das classes, usada em `xticklabels` e `yticklabels`.
* Adiciona títulos e rótulos de eixos (“Previsto” e “Real”).
* Exibe o gráfico com `plt.show()`.

### `avaliar_modelo(model, X, y, conjunto="Teste") -> float`

Avalia um modelo SVM em um conjunto específico (treino, validação ou teste).

1. Calcula as predições: `y_pred = model.predict(X)`.
2. Calcula a acurácia com `accuracy_score(y, y_pred)`.
3. Imprime no terminal:

   * Acurácia.
   * Relatório de classificação.
4. Gera:

   * Matriz de confusão e gráfico.
   * Gráfico de métricas.
5. Retorna a acurácia.

### `prever_qualidade(model, scaler, nova_amostra: dict)`

Faz a previsão da classe de qualidade da água para uma nova amostra.

### `main()`

Orquestra todo o pipeline:

* Carrega, limpa, divide, normaliza.
* Treina SVM linear e RBF.
* Compara na validação e escolhe o melhor.
* Avalia no teste.
* Faz uma previsão de exemplo.
