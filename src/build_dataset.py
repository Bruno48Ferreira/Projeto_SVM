import os
import pandas as pd
import numpy as np
from functools import reduce

# ==========================
# Caminhos
# ==========================

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
OUT_PATH = os.path.join(BASE_DIR, "data", "agua_iqa.csv")

FILES = {
    "od": os.path.join(RAW_DIR, "od_serie_historica.csv"),
    "fosforo_total": os.path.join(RAW_DIR, "fosforo_total_serie_historica.csv"),
    "turbidez": os.path.join(RAW_DIR, "turbidez_serie_historica.csv"),
    "dbo": os.path.join(RAW_DIR, "dbo_serie_historica.csv"),  # não usado no SVM por enquanto
    "iqa": os.path.join(RAW_DIR, "iqa_serie_historica.csv"),
}

# Chaves de identificação da amostra
KEY_COLS_BASE = [
    "CDESTACAO",
    "SGUF",
    "CORPODAGUA",
    "AMBIENTE",
    "LATITUDE",
    "LONGITUDE",
]

# ==========================
# Funções auxiliares
# ==========================

def carregar_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    return pd.read_csv(path, sep=",")


def melt_medidas(df: pd.DataFrame, valor_nome: str) -> pd.DataFrame:
    """
    Converte colunas MED_YYYY -> linhas.
    Saída: chaves + ANO + valor_nome
    """
    id_cols = [c for c in KEY_COLS_BASE if c in df.columns]
    med_cols = [c for c in df.columns if c.startswith("MED_")]

    if not med_cols:
        raise ValueError("Nenhuma coluna MED_ encontrada.")

    tmp = df[id_cols + med_cols].copy()

    long_df = tmp.melt(
        id_vars=id_cols,
        value_vars=med_cols,
        var_name="ANO_COL",
        value_name=valor_nome,
    )

    long_df["ANO"] = (
        long_df["ANO_COL"]
        .str.extract(r"MED_(\d+)", expand=False)
        .astype("Int64")
    )

    long_df = long_df.drop(columns=["ANO_COL"])
    return long_df


def classificar_iqa(iqa):
    """
    Exemplo de faixas (pode ajustar depois):
      - IQA <= 19  -> pessima
      - 19 < IQA <= 36 -> ruim
      - 36 < IQA <= 51 -> regular
      - 51 < IQA <= 79 -> boa
      - IQA > 79  -> otima
    """
    if pd.isna(iqa):
        return np.nan
    try:
        iqa = float(iqa)
    except Exception:
        return np.nan

    if iqa <= 19:
        return "pessima"
    elif iqa <= 36:
        return "ruim"
    elif iqa <= 51:
        return "regular"
    elif iqa <= 79:
        return "boa"
    else:
        return "otima"


# ==========================
# Construção do dataset
# ==========================

def construir_dataset():
    # 1) Processar OD, Fósforo e Turbidez (features reais)
    print("Lendo e transformando OD, Fósforo Total e Turbidez...")
    df_od = melt_medidas(carregar_csv(FILES["od"]), "od")
    df_p = melt_medidas(carregar_csv(FILES["fosforo_total"]), "fosforo_total")
    df_t = melt_medidas(carregar_csv(FILES["turbidez"]), "turbidez")

    # 2) Merge desses parâmetros pelas chaves + ANO
    def merge_on(a, b):
        return pd.merge(a, b, on=KEY_COLS_BASE + ["ANO"], how="inner")

    df_feat = reduce(merge_on, [df_od, df_p, df_t])

    print(f"Após merge de OD, P e Turbidez: {df_feat.shape[0]} linhas")

    # 3) Processar IQA
    print("Lendo e transformando IQA...")
    df_iqa = melt_medidas(carregar_csv(FILES["iqa"]), "IQA")

    # 4) Juntar IQA com as features
    df_full = pd.merge(
        df_feat,
        df_iqa,
        on=KEY_COLS_BASE + ["ANO"],
        how="inner",
    )

    print(f"Após adicionar IQA: {df_full.shape[0]} linhas")

    # 5) Remover linhas sem valor em OD, P, Turbidez ou IQA
    df_full = df_full.dropna(subset=["od", "fosforo_total", "turbidez", "IQA"])
    print(f"Após remover NaN: {df_full.shape[0]} linhas completas")

    # 6) Criar classe categórica
    df_full["classe"] = df_full["IQA"].apply(classificar_iqa)

    # 7) Mapear para colunas esperadas pelo modelo

    # oxigenio_dissolvido = od
    df_full["oxigenio_dissolvido"] = df_full["od"]

    # fosforo_total e turbidez já existem
    # Criar colunas faltantes como 0.0 (placeholders)
    df_full["coliformes_termotolerantes"] = 0.0
    df_full["pH"] = 0.0
    df_full["temperatura_agua"] = 0.0
    df_full["nitrogenio_total"] = 0.0
    df_full["solidos_totais"] = 0.0

    # 8) Selecionar apenas as colunas finais que o SVM usa

    final_cols = [
        "oxigenio_dissolvido",
        "coliformes_termotolerantes",
        "pH",
        "temperatura_agua",
        "nitrogenio_total",
        "fosforo_total",
        "turbidez",
        "solidos_totais",
        "classe",
    ]

    df_final = df_full[final_cols].copy()

    # 9) Salvar
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    print(f"Salvando dataset final em: {OUT_PATH}")
    df_final.to_csv(OUT_PATH, index=False)
    print("Concluído.")


if __name__ == "__main__":
    construir_dataset()
