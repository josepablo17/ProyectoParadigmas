import pandas as pd

class GeneradorResumen:
    """
    Clase que genera resúmenes en texto natural a partir de los resultados
    de agrupamientos, detección de atípicos y análisis de correlación.
    """

    @staticmethod
    def resumen_agrupamiento(clusters: pd.Series) -> str:
        """
        Genera un resumen textual a partir de los resultados del clustering.

        Parámetro:
        - clusters: Serie con los valores de cluster asignados a cada fila

        Retorna:
        - Cadena de texto que describe la cantidad de grupos encontrados y su tamaño
        """
        num_clusters = clusters.nunique()
        conteo = clusters.value_counts().sort_index()

        resumen = f"🔹 Se identificaron {num_clusters} grupos principales mediante K-Means:\n"
        for grupo, cantidad in conteo.items():
            resumen += f"  - Grupo {grupo}: {cantidad} registros\n"

        return resumen

    @staticmethod
    def resumen_atipicos(df: pd.DataFrame, zscore: pd.DataFrame, iqr: pd.DataFrame, forest: pd.Series) -> str:
        """
        Resume la cantidad de valores atípicos encontrados por cada método.

        Parámetros:
        - df: DataFrame original (no se usa directamente, pero se mantiene para consistencia)
        - zscore: DataFrame booleano con outliers detectados por Z-score
        - iqr: DataFrame booleano con outliers detectados por IQR
        - forest: Serie booleana con registros anómalos detectados por Isolation Forest

        Retorna:
        - Cadena de texto con los totales de valores atípicos encontrados
        """
        resumen = "🔹 Detección de valores atípicos:\n"
        resumen += f"  - Por Z-score: {int(zscore.sum().sum())} valores atípicos\n"
        resumen += f"  - Por IQR: {int(iqr.sum().sum())} valores atípicos\n"
        resumen += f"  - Por Isolation Forest: {forest.sum()} registros marcados como anómalos\n"

        return resumen

    @staticmethod
    def resumen_correlaciones(correlaciones: pd.DataFrame, umbral: float = 0.8) -> str:
        """
        Genera un resumen textual con las correlaciones fuertes detectadas entre variables.

        Parámetros:
        - correlaciones: matriz de correlación
        - umbral: valor mínimo absoluto para considerar una correlación como fuerte

        Retorna:
        - Cadena de texto con las correlaciones que superan el umbral
        """
        resumen = f"🔹 Correlaciones fuertes detectadas (|r| ≥ {umbral:.2f}):\n"
        contador = 0

        for i in range(len(correlaciones.columns)):
            for j in range(i + 1, len(correlaciones.columns)):
                col1 = correlaciones.columns[i]
                col2 = correlaciones.columns[j]
                coef = correlaciones.iloc[i, j]

                if abs(coef) >= umbral:
                    resumen += f"  - {col1} vs {col2}: r = {coef:.2f}\n"
                    contador += 1

        if contador == 0:
            resumen += "  - No se encontraron correlaciones fuertes.\n"

        return resumen
