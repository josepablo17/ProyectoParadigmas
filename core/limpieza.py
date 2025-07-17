import pandas as pd

class LimpiadorDatos:
    @staticmethod
    def limpiar(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        resumen = {}

        # 1. Columnas con más del 50% de valores nulos
        umbral = 0.5
        cols_con_muchos_nulos = df.columns[df.isnull().mean() > umbral].tolist()
        resumen["columnas_eliminadas_por_nulos"] = len(cols_con_muchos_nulos)
        df = df.drop(columns=cols_con_muchos_nulos)

        # 2. Eliminar duplicados
        duplicados = df.duplicated().sum()
        resumen["duplicados_eliminados"] = duplicados
        df = df.drop_duplicates()

        # 3. Rellenar nulos restantes
        nulos_antes = df.isnull().sum().sum()
        for col in df.columns:
            if df[col].dtype in ['float64', 'int64']:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna("Desconocido")
        nulos_despues = df.isnull().sum().sum()
        resumen["nulos_rellenados"] = nulos_antes - nulos_despues

        return df, resumen

