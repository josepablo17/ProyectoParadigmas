import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype

class LimpiadorDatos:
    @staticmethod
    def limpiar(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        """
        Limpieza básica del DataFrame:
          1) Elimina columnas con >50% de nulos
          2) Elimina filas duplicadas
          3) Imputa nulos restantes: mediana en numéricas; "Desconocido" en no numéricas

        Retorna:
          (df_limpio, resumen_dict)
        """
        resumen: dict = {}
        df = df.copy()

        # Normalización ligera: reemplazar inf/-inf por NaN para permitir imputación
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # 1) Columnas con > 50% nulos
        umbral = 0.5
        null_ratio = df.isna().mean()
        cols_con_muchos_nulos = null_ratio[null_ratio > umbral].index.tolist()
        df.drop(columns=cols_con_muchos_nulos, inplace=True)

        resumen["columnas_eliminadas_por_nulos"] = len(cols_con_muchos_nulos)
        resumen["detalle_columnas_eliminadas"] = cols_con_muchos_nulos

        # 2) Eliminar duplicados
        duplicados = int(df.duplicated(keep="first").sum())
        df.drop_duplicates(keep="first", inplace=True)
        resumen["duplicados_eliminados"] = duplicados

        # 3) Rellenar nulos restantes
        nulos_antes = int(df.isna().sum().sum())

        # a) Numéricas → mediana
        num_cols = [c for c in df.columns if is_numeric_dtype(df[c])]
        if num_cols:
            medians = df[num_cols].median(numeric_only=True)
            df[num_cols] = df[num_cols].fillna(medians)

        # b) No numéricas → "Desconocido"
        non_num_cols = [c for c in df.columns if c not in num_cols]
        if non_num_cols:
            df[non_num_cols] = df[non_num_cols].fillna("Desconocido")

        nulos_despues = int(df.isna().sum().sum())
        resumen["nulos_rellenados"] = nulos_antes - nulos_despues

        return df, resumen
