# Projeto SVM – Previsão da Qualidade da Água

Este projeto implementa um modelo de **Máquinas de Vetores de Suporte (SVM)** para
prever a classe de qualidade da água (ótima, boa, regular, ruim, péssima) a partir
de parâmetros físico-químicos que compõem o Índice de Qualidade da Água (IQA).

Os dados utilizados foram preparados a partir das séries históricas da
Agência Nacional de Águas e Saneamento Básico (ANA), gerando o arquivo
`data/agua_iqa.csv` com as colunas de entrada e a classe alvo.


## Estrutura do projeto

```text
Projeto_SVM/
├─ data/
│  └─ raw/
│     └─ agua_iqa.csv                # dataset final usado no treinamento
├─ imgs/
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


No código, o arquivo é referenciado pela constante DATA_PATH.
Se necessário, ajuste esse caminho para o local correto do seu agua_iqa.csv.

Requisitos

Python 3.10+ (recomendado)

Bibliotecas:

pip install pandas numpy scikit-learn matplotlib seaborn

Como executar

Na raiz do projeto:

cd src
python main.py


O script irá:

Carregar e limpar o dataset.

Dividir os dados em treino, validação e teste.

Normalizar os atributos (escala [0, 1]).

Treinar dois modelos SVM (kernel linear e RBF).

Avaliar ambos na base de validação.

Escolher automaticamente o melhor modelo (pela acurácia).

Avaliar o melhor modelo na base de teste.

Exibir:

Matrizes de confusão (validação e teste).

Gráficos de precisão, recall e F1-score por classe.

Realizar a previsão de uma amostra de exemplo e imprimir a classe prevista.

Pipeline do algoritmo

Pré-processamento

Carregamento do CSV.

Remoção de linhas vazias ou com valores ausentes.

Seleção das colunas de interesse (features + classe).

Divisão dos dados

70% treino, 15% validação, 15% teste (com estratificação por classe).

Normalização

Escala Min–Max para o intervalo [0, 1], ajustada apenas com os dados de treino.

Treinamento

SVM com kernel linear.

SVM com kernel RBF.

Avaliação

Cálculo de acurácia, precisão, recall, F1-score.

Matrizes de confusão e gráficos de métricas por classe.

Previsão

Função para prever a classe de uma nova amostra informada via dicionário Python.

Descrição das funções (src/main.py)
carregar_dados(caminho_csv: str) -> pd.DataFrame

Carrega o arquivo CSV contendo o dataset.

Verifica se o arquivo existe.

Lê o CSV com pandas.read_csv.

Retorna um DataFrame com todos os dados originais.

limpar_dados(df: pd.DataFrame) -> pd.DataFrame

Realiza a limpeza básica do dataset.

Remove linhas totalmente vazias (dropna(how="all")).

Remove linhas com quaisquer valores ausentes.

Mantém apenas as colunas indicadas em FEATURE_COLUMNS e TARGET_COLUMN.

Retorna um DataFrame pronto para ser dividido em treino/validação/teste.

dividir_dados(df: pd.DataFrame)

Separa o dataset em três subconjuntos:

Entrada (X) = colunas em FEATURE_COLUMNS.

Saída (y) = coluna TARGET_COLUMN.

Passos:

Primeira divisão:

train_test_split com test_size=0.3 →
70% treino (X_train, y_train) e 30% temporário (X_temp, y_temp).

Segunda divisão:

train_test_split em X_temp, y_temp com test_size=0.5 →
15% validação (X_val, y_val) e 15% teste (X_test, y_test).

Utiliza stratify para manter a proporção de classes.

Retorna:

X_train, X_val, X_test, y_train, y_val, y_test

normalizar_dados(X_train, X_val, X_test)

Aplica a normalização Min–Max ([0, 1]) em todas as features.

Cria um MinMaxScaler(feature_range=(0, 1)).

Ajusta apenas no treino: scaler.fit_transform(X_train).

Aplica a transformação em validação e teste: scaler.transform(...).

Retorna os dados normalizados e o scaler para uso futuro:

X_train_scaled, X_val_scaled, X_test_scaled, scaler

treinar_svm(X_train, y_train, kernel="rbf", C=1.0, gamma="scale")

Treina um classificador SVM.

Utiliza sklearn.svm.SVC.

Parâmetros:

kernel: tipo de kernel ("linear", "rbf", etc.).

C: parâmetro de regularização.

gamma: parâmetro do kernel RBF.

Ajusta o modelo com .fit(X_train, y_train).

Retorna o modelo treinado.

plot_metricas_avaliacao(y_true, y_pred, labels, titulo="Métricas de Avaliação")

Gera o gráfico de barras com precisão, recall e F1-score por classe.

Usa classification_report(..., output_dict=True) para extrair as métricas.

Para cada classe em labels, coleta:

precision

recall

f1-score

Plota três barras por classe com matplotlib:

Azul: Precisão

Laranja: Recall

Verde: F1-score

Configura e exibe o gráfico com plt.show().

plot_matriz_confusao(cm, labels, titulo="Matriz de Confusão")

Desenha uma matriz de confusão usando seaborn.heatmap.

cm: matriz de confusão (gerada por confusion_matrix).

labels: lista de rótulos das classes, usada em xticklabels e yticklabels.

Adiciona títulos e rótulos de eixos (“Previsto” e “Real”).

Exibe o gráfico com plt.show().

avaliar_modelo(model, X, y, conjunto="Teste") -> float

Avalia um modelo SVM em um conjunto específico (treino, validação ou teste).

Passos:

Calcula as predições: y_pred = model.predict(X).

Calcula a acurácia com accuracy_score(y, y_pred).

Imprime no terminal:

Acurácia.

Relatório de classificação (classification_report).

Gera:

Matriz de confusão (confusion_matrix) e chama plot_matriz_confusao.

Gráfico de métricas chamando plot_metricas_avaliacao.

Retorna a acurácia (float).

É usada para comparar SVM linear e RBF na validação e avaliar o modelo escolhido no teste.

prever_qualidade(model, scaler, nova_amostra: dict)

Realiza a previsão da classe de qualidade da água para uma nova amostra.

nova_amostra deve ser um dicionário com todas as features:

{
    "oxigenio_dissolvido": ...,
    "coliformes_termotolerantes": ...,
    "pH": ...,
    "temperatura_agua": ...,
    "nitrogenio_total": ...,
    "fosforo_total": ...,
    "turbidez": ...,
    "solidos_totais": ...
}


Passos:

Constrói um vetor na ordem de FEATURE_COLUMNS.

Aplica o mesmo scaler treinado (scaler.transform).

Usa model.predict para obter a classe.

Retorna a string da classe prevista (ex.: "boa", "ruim").

main()

Função principal que orquestra todo o pipeline:

Carrega e imprime o shape inicial do dataset.

Limpa os dados e imprime o shape após limpeza.

Divide em treino/validação/teste.

Normaliza os dados.

Treina:

SVM Linear → svm_linear

SVM RBF → svm_rbf

Avalia ambos na validação:

acc_linear = avaliar_modelo(... "Validação (Linear)")

acc_rbf = avaliar_modelo(... "Validação (RBF)")

Compara as acurácias:

Seleciona automaticamente o melhor modelo.

Avalia o melhor modelo no conjunto de teste:

avaliar_modelo(melhor_modelo, X_test_s, y_test, ...)

Define uma amostra de exemplo no código e chama prever_qualidade:

Imprime no terminal a classe prevista para essa amostra.
