import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

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
          está asignada a un grupo (cluster).
        """

        # Filtrar solo columnas numéricas y eliminar filas con valores faltantes
        numericas = df.select_dtypes(include=['float64', 'int64']).dropna()

        # Validar que haya al menos 2 columnas numéricas
        if numericas.shape[1] < 2:
            raise ValueError("Se necesitan al menos 2 columnas numéricas para aplicar K-Means.")

        # Normalizar (escalar) los datos para que todas las variables contribuyan igual
        escalador = StandardScaler()
        datos_escalados = escalador.fit_transform(numericas)

        # Crear y ajustar el modelo K-Means
        modelo = KMeans(n_clusters=num_clusters, n_init=10, random_state=42)
        etiquetas = modelo.fit_predict(datos_escalados)

        # Retornar una Serie con la asignación de cluster por fila
        return pd.Series(etiquetas, index=numericas.index, name="Cluster")
