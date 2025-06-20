import pandas as pd

def obtener_tipos_variables(df: pd.DataFrame) -> dict:
    """
    Clasifica las columnas del DataFrame según su tipo de dato.

    Parámetros:
    - df: DataFrame con los datos a analizar.

    Retorna:
    - Un diccionario con listas de nombres de columnas por tipo:
      numéricas, categóricas y temporales.
    """
    return {
        'numericas': df.select_dtypes(include=['int64', 'float64']).columns.tolist(),
        'categoricas': df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist(),
        'temporales': df.select_dtypes(include=['datetime64[ns]']).columns.tolist()
    }

def obtener_estadisticas_descriptivas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula estadísticas descriptivas de todas las columnas del DataFrame.

    Retorna:
    - Un DataFrame transpuesto con estadísticas como media, mediana,
      desviación estándar, valores únicos, etc.
    """
    return df.describe(include='all').transpose()

def obtener_matriz_correlacion(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula la matriz de correlación entre columnas numéricas.

    Retorna:
    - Un DataFrame con coeficientes de correlación de Pearson entre variables.
      Si hay menos de 2 columnas numéricas, retorna un DataFrame vacío.
    """
    numeric_df = df.select_dtypes(include=['int64', 'float64'])

    if numeric_df.shape[1] < 2:
        return pd.DataFrame()

    return numeric_df.corr(method='pearson')  # Cambiar a 'spearman' si lo deseas
