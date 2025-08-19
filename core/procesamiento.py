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
    df_limpio = df_limpio.copy()

    resumen.setdefault("target", {})
    resumen["target"]["creada"] = False
    resumen["target"]["columna_base"] = None
    resumen["target"]["umbral_mediana"] = None
    resumen["target"]["observaciones_validas"] = None

    # Solo si existe exactamente 'Units Sold' (mantenemos tu contrato actual)
    if "Units Sold" in df_limpio.columns:
        # Convertir a numérico de forma segura (si hay strings, comas, etc.)
        unidades = pd.to_numeric(df_limpio["Units Sold"], errors="coerce")
        n_valid = int(unidades.notna().sum())

        if n_valid > 0:
            umbral = float(unidades.median())  # mediana ignora NaN
            # Crear Exito comparando contra la mediana; mantener longitud original
            exito = (unidades >= umbral).astype("boolean")
            df_limpio["Exito"] = exito

            resumen["target"]["creada"] = True
            resumen["target"]["columna_base"] = "Units Sold"
            resumen["target"]["umbral_mediana"] = umbral
            resumen["target"]["observaciones_validas"] = n_valid
        else:
            # No crear 'Exito' si no hay datos numéricos válidos
            resumen["target"]["observaciones_validas"] = 0

    return df_limpio, resumen
