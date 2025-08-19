# core/analizar.py
from __future__ import annotations

import pandas as pd
from pandas.api.types import (
    is_numeric_dtype,
    is_datetime64_any_dtype,
    is_bool_dtype,
    is_categorical_dtype,
)

def obtener_tipos_variables(df: pd.DataFrame) -> dict:
    """
    Clasifica las columnas del DataFrame según su tipo de dato.
    Retorna un dict con listas: 'numericas', 'categoricas', 'temporales'.
    """
    numericas, categoricas, temporales = [], [], []

    for col in df.columns:
        s = df[col]
        if is_datetime64_any_dtype(s):
            temporales.append(col)
        elif is_bool_dtype(s):
           
            categoricas.append(col)
        elif is_numeric_dtype(s):
            numericas.append(col)
        elif is_categorical_dtype(s) or s.dtype == object:
            categoricas.append(col)
        else:
           
            categoricas.append(col)

    return {
        "numericas": numericas,
        "categoricas": categoricas,
        "temporales": temporales,
    }

def obtener_estadisticas_descriptivas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula estadísticas descriptivas para todas las columnas.
    Devuelve un DataFrame transpuesto e incluye:
      - 'n_unique' (# de valores únicos)
      - 'missing_rate' (proporción de nulos)
    Compatible con pandas que NO soportan 'datetime_is_numeric'.
    """
    try:
        desc = df.describe(include="all", datetime_is_numeric=False).transpose()
    except TypeError:
        desc = df.describe(include="all").transpose()

    desc["n_unique"] = df.nunique(dropna=True)
    desc["missing_rate"] = df.isna().mean()

    return desc

def obtener_matriz_correlacion(df: pd.DataFrame, metodo: str = "pearson") -> pd.DataFrame:
    """
    Calcula la matriz de correlación entre columnas numéricas reales.
    Excluye columnas booleanas y castea a float64 para evitar errores.
    Si hay <2 columnas válidas, retorna DataFrame vacío.
    """
    # Filtrar numéricas reales (excluyendo booleanas)
    numeric_cols = [
        c for c in df.columns
        if is_numeric_dtype(df[c]) and not is_bool_dtype(df[c])
    ]

    if len(numeric_cols) < 2:
        return pd.DataFrame()

    num_df = df[numeric_cols].apply(pd.to_numeric, errors="coerce").astype("float64")

    num_df = num_df.dropna(axis=1, how="all")
    if num_df.shape[1] < 2:
        return pd.DataFrame()

    metodo = metodo if metodo in {"pearson", "spearman", "kendall"} else "pearson"
    return num_df.corr(method=metodo)
