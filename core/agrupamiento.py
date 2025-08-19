import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from pandas.api.types import is_numeric_dtype
import numpy as np

class Agrupamiento:
    """
    Clase encargada de aplicar técnicas de agrupamiento (clustering)
    sobre datos numéricos utilizando K-Means.
    """

    @staticmethod
    def por_kmeans(df: pd.DataFrame, num_clusters: int = 3) -> pd.Series:
        """
        Aplica el algoritmo K-Means a las columnas numéricas del DataFrame.

        Parámetros:
        - df: DataFrame con los datos originales.
        - num_clusters: cantidad de grupos (clusters) a formar.

        Retorna:
        - Una Serie de Pandas donde cada fila del DataFrame original
          está asignada a un grupo (cluster). El índice corresponde
          a las filas utilizadas (sin NaN tras el filtrado).
        """
        if not isinstance(num_clusters, int) or num_clusters < 2:
            raise ValueError("num_clusters debe ser un entero >= 2.")

        # 1) Selección robusta de columnas numéricas
        num_cols = [c] + [] if False else [c for c in df.columns if is_numeric_dtype(df[c])]
        if not num_cols:
            raise ValueError("No hay columnas numéricas para aplicar K-Means.")

        dfn = df[num_cols].apply(pd.to_numeric, errors="coerce")

        # 2) Eliminar columnas completamente NaN
        dfn = dfn.dropna(axis=1, how="all")
        if dfn.shape[1] < 2:
            raise ValueError("Se necesitan al menos 2 columnas numéricas no vacías para aplicar K-Means.")

        # 3) Eliminar filas con NaN (como en tu versión original)
        dfn = dfn.dropna(axis=0, how="any")
        if len(dfn) < num_clusters:
            raise ValueError(f"Filas insuficientes para {num_clusters} clusters después de limpiar NaN.")

        # 4) Quitar columnas de varianza cero (constantes)
        var = dfn.var(numeric_only=True)
        cols_const = var.index[var.fillna(0) == 0].tolist()
        if cols_const:
            dfn = dfn.drop(columns=cols_const)

        if dfn.shape[1] < 2:
            raise ValueError("Tras eliminar columnas constantes, faltan columnas para el clustering.")

        # 5) Escalado estándar
        scaler = StandardScaler()
        X = scaler.fit_transform(dfn)

        # 6) K-Means
        modelo = KMeans(n_clusters=num_clusters, n_init=10, random_state=42)
        etiquetas = modelo.fit_predict(X)

        return pd.Series(etiquetas, index=dfn.index, name="Cluster")
