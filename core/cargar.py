import pandas as pd
import os

def CargarDatos(ruta_archivo: str) -> pd.DataFrame:
    """
    Carga un archivo de datos en formato CSV o Excel y lo convierte en un DataFrame de Pandas.

    Parámetros:
    - ruta_archivo: ruta del archivo a cargar (debe terminar en .csv, .xlsx o .xls)

    Retorna:
    - Un DataFrame con los datos cargados.

    Lanza:
    - FileNotFoundError si el archivo no existe.
    - ValueError si el formato del archivo no es soportado.
    """
    
    # Validar existencia del archivo
    if not os.path.exists(ruta_archivo):
        raise FileNotFoundError(f"El archivo no existe: {ruta_archivo}")

    # Cargar archivo según su extensión
    if ruta_archivo.endswith('.csv'):
        df = pd.read_csv(ruta_archivo)
    elif ruta_archivo.endswith('.xlsx') or ruta_archivo.endswith('.xls'):
        df = pd.read_excel(ruta_archivo)
    else:
        raise ValueError("⚠️ Formato de archivo no soportado. Usa .csv o .xlsx")

    return df
