import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

class DeteccionAtipicos:
    """
    Clase encargada de detectar valores atípicos en variables numéricas
    utilizando distintos métodos estadísticos y de aprendizaje automático.
    """

    @staticmethod
    def por_zscore(df: pd.DataFrame, umbral: float = 3.0) -> pd.DataFrame:
        """
        Detecta valores atípicos por columna utilizando el método de Z-score.

        Parámetros:
        - df: DataFrame con los datos originales
        - umbral: valor absoluto de Z-score a partir del cual se considera atípico

        Retorna:
        - Un DataFrame booleano indicando si cada valor es atípico (True) o no (False)
        """
        atipicos = pd.DataFrame(index=df.index)
        numericas = df.select_dtypes(include=['float64', 'int64'])

        for columna in numericas.columns:
            z_scores = (numericas[columna] - numericas[columna].mean()) / numericas[columna].std()
            atipicos[columna] = z_scores.abs() > umbral

        return atipicos

    @staticmethod
    def por_rango_intercuartil(df: pd.DataFrame) -> pd.DataFrame:
        """
        Detecta valores atípicos por columna utilizando el rango intercuartílico (IQR).

        Retorna:
        - Un DataFrame booleano indicando si cada valor es atípico según IQR
        """
        atipicos = pd.DataFrame(index=df.index)
        numericas = df.select_dtypes(include=['float64', 'int64'])

        for columna in numericas.columns:
            Q1 = numericas[columna].quantile(0.25)
            Q3 = numericas[columna].quantile(0.75)
            IQR = Q3 - Q1
            atipicos[columna] = (
                (numericas[columna] < (Q1 - 1.5 * IQR)) |
                (numericas[columna] > (Q3 + 1.5 * IQR))
            )

        return atipicos

    @staticmethod
    def por_isolation_forest(df: pd.DataFrame) -> pd.Series:
        """
        Detecta registros atípicos utilizando el algoritmo Isolation Forest.

        Retorna:
        - Una Serie booleana donde True indica que el registro completo es atípico
        """
        numericas = df.select_dtypes(include=['float64', 'int64']).dropna()

        if numericas.shape[1] < 2:
            # No hay suficientes variables numéricas para aplicar el modelo
            return pd.Series([False] * len(df), index=df.index)

        modelo = IsolationForest(contamination=0.05, random_state=42)
        modelo.fit(numericas)
        predicciones = modelo.predict(numericas)

        return pd.Series(predicciones == -1, index=numericas.index)
