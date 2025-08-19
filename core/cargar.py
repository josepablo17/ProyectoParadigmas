# core/cargar.py
from __future__ import annotations

import pandas as pd
from io import BytesIO

def cargar_datos(uploaded_file) -> pd.DataFrame:
    """
    Carga un archivo subido desde Streamlit (UploadedFile) en formato CSV o Excel.

    Parámetros:
    - uploaded_file: archivo subido desde st.file_uploader

    Retorna:
    - Un DataFrame con los datos cargados.
    """
    if uploaded_file is None or getattr(uploaded_file, "name", None) is None:
        raise ValueError("No se recibió un archivo válido.")

    name = uploaded_file.name.lower()

    try:
        if name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif name.endswith((".xls", ".xlsx")):
            bio = BytesIO(uploaded_file.getbuffer())
            df = pd.read_excel(bio)
        else:
            raise ValueError("Formato no soportado. Usa .csv, .xls o .xlsx.")
    except Exception as e:
        raise ValueError(f"No se pudo leer el archivo: {type(e).__name__}: {e}") from e

    # Normalización ligera de columnas
    df.columns = [str(c).strip() for c in df.columns]

    if df.empty:
        raise ValueError("El archivo se cargó pero no contiene datos.")

    return df
