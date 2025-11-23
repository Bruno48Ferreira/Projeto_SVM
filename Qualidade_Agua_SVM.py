# src/main.py

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Caminho do dataset
DATA_PATH = os.path.join('d:\\Faculdade\\2°Semestre - 2025\\IA 2\\Projeto SVM', "data", "agua_iqa.csv")

# Colunas de entrada (features) baseadas no dataset
FEATURE_COLUMNS = [
    "oxigenio_dissolvido",
    "coliformes_termotolerantes",
    "pH",
    "temperatura_agua",
    "nitrogenio_total",
    "fosforo_total",
    "turbidez",
    "solidos_totais",
]

# Coluna alvo (classe da qualidade da água)
TARGET_COLUMN = "classe"

# Carrega o dataset para análise
def carregar_dados(caminho_csv: str) -> pd.DataFrame:
    if not os.path.exists(caminho_csv):
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho_csv}")
    df = pd.read_csv(caminho_csv, sep=",")
    return df

# Limpa o dataset de dados desnecessarios como linhas vazias ou com valor ausente
def limpar_dados(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(how="all")
    df = df.dropna()
    colunas_necessarias = FEATURE_COLUMNS + [TARGET_COLUMN]
    df = df[colunas_necessarias]
    return df

# Separa os dados do dataset entre validação e treinamento
def dividir_dados(df: pd.DataFrame):
    X = df[FEATURE_COLUMNS].values
    y = df[TARGET_COLUMN].values

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=0.3,        # 70% treino, 30% (val+teste)
        random_state=42,
        stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.5,        # metade para teste, metade para validação
        random_state=42,
        stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test

# Normaliza os dados deixando-os em um intervalo de [0 - 1]
def normalizar_dados(X_train, X_val, X_test):
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler

# Treina o modelo com base no kernel escolhido
def treinar_svm(X_train, y_train, kernel="rbf", C=1.0, gamma="scale"):
    model = SVC(kernel=kernel, C=C, gamma=gamma)
    model.fit(X_train, y_train)
    return model

# Função responsável por gerar o gráfico de metricas do modelo
def plot_metricas_avaliacao(y_true, y_pred, labels, titulo="Métricas de Avaliação"):
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0   # evita erro quando uma métrica é indefinida
    )
    classes = [str(l) for l in labels]
    precisions = [report[c]["precision"] for c in classes]
    recalls    = [report[c]["recall"]    for c in classes]
    f1_scores  = [report[c]["f1-score"]  for c in classes]
    x = np.arange(len(classes))
    largura = 0.25  # largura de cada barra
    plt.figure(figsize=(8, 5))
    plt.bar(x - largura, precisions, width=largura, label="Precisão")
    plt.bar(x,          recalls,   width=largura, label="Recall")
    plt.bar(x + largura, f1_scores, width=largura, label="F1-score")
    plt.xticks(x, classes)
    plt.ylim(0, 1.05)
    plt.xlabel("Classe")
    plt.ylabel("Valor da métrica")
    plt.title(titulo)
    plt.legend()
    plt.tight_layout()
    plt.show()

def avaliar_modelo(model, X, y, conjunto="Teste"):
    # Predições
    y_pred = model.predict(X)

    # Acurácia
    acc = accuracy_score(y, y_pred)
    print(f"\n=== Avaliação no conjunto de {conjunto} ===")
    print(f"Acurácia: {acc:.4f}")
    print("\nRelatório de classificação:")
    print(classification_report(y, y_pred))

    # Matriz de confusão
    labels = np.unique(y)
    cm = confusion_matrix(y, y_pred, labels=labels)
    plot_matriz_confusao(cm, labels, titulo=f"Matriz de Confusão - {conjunto}")
    # Gráfico de metricas
    plot_metricas_avaliacao(y, y_pred, labels, titulo=f"Métricas de Avaliação - {conjunto}")
    return acc

# Gera o gráfico das matrizes de confusão de cada modelo
def plot_matriz_confusao(cm, labels, titulo="Matriz de Confusão"):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels,yticklabels=labels)
    plt.xlabel("Previsto")
    plt.ylabel("Real")
    plt.title(titulo)
    plt.tight_layout()
    plt.show()

# Função para análise de novos dados
def prever_qualidade(model, scaler, nova_amostra: dict):
    vetor = np.array([[nova_amostra[col] for col in FEATURE_COLUMNS]])
    vetor_scaled = scaler.transform(vetor)
    classe_predita = model.predict(vetor_scaled)[0]
    return classe_predita

def main():
    # 1) Carregar dados
    df = carregar_dados(DATA_PATH)
    print("Shape original:", df.shape)

    # 2) Limpar
    df = limpar_dados(df)
    print("Shape após limpeza:", df.shape)

    # 3) Dividir em treino/validação/teste
    X_train, X_val, X_test, y_train, y_val, y_test = dividir_dados(df)

    # 4) Normalizar
    X_train_s, X_val_s, X_test_s, scaler = normalizar_dados(X_train, X_val, X_test)

    # 5) Treinar dois modelos: linear e RBF
    print("\nTreinando SVM com kernel linear...")
    svm_linear = treinar_svm(X_train_s, y_train, kernel="linear")
    print("\nTreinando SVM com kernel RBF...")
    svm_rbf = treinar_svm(X_train_s, y_train, kernel="rbf")

    # 6) Avalia melhor modelo com base na acuracia de cada um dos modelos
    acc_linear = avaliar_modelo(svm_linear, X_val_s, y_val, conjunto="Validação (Linear)")
    acc_rbf = avaliar_modelo(svm_rbf, X_val_s, y_val, conjunto="Validação (RBF)")

    if acc_rbf >= acc_linear:
        melhor_modelo = svm_rbf
        nome_modelo   = "RBF"
    else:
        melhor_modelo = svm_linear
        nome_modelo   = "Linear"

    print(f"\nUsando o modelo {nome_modelo} para avaliação final no teste...")
    avaliar_modelo(melhor_modelo, X_test_s, y_test, conjunto=f"Teste ({nome_modelo})")

    # 8) Exemplo de previsão com nova amostra
    exemplo = {
        "oxigenio_dissolvido": 7.5,
        "coliformes_termotolerantes": 200,
        "pH": 7.2,
        "temperatura_agua": 23,
        "nitrogenio_total": 1.2,
        "fosforo_total": 0.05,
        "turbidez": 10,
        "solidos_totais": 500
    }

    classe = prever_qualidade(svm_rbf, scaler, exemplo)
    print("\nExemplo de previsão para nova amostra:", classe)

main()