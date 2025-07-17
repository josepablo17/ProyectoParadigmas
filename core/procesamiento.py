import pandas as pd
from core.limpieza import LimpiadorDatos

def preparar_datos(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Limpia el DataFrame y genera la columna 'Exito' si existe 'Units Sold'.

    Retorna:
    - El DataFrame limpio.
    - Un resumen de limpieza con métricas clave.
    """
    df_limpio, resumen = LimpiadorDatos.limpiar(df)

    if "Units Sold" in df_limpio.columns:
        umbral = df_limpio["Units Sold"].median()
        df_limpio["Exito"] = df_limpio["Units Sold"] >= umbral

    return df_limpio, resumen
