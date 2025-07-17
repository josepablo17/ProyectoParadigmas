# core/cargar.py

import pandas as pd
from io import BytesIO, StringIO

def cargar_datos(uploaded_file):
    """
    Carga un archivo subido desde Streamlit (UploadedFile) en formato CSV o Excel.

    Parámetros:
    - uploaded_file: archivo subido desde st.file_uploader

    Retorna:
    - Un DataFrame con los datos cargados.
    """
    if uploaded_file.name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith((".xls", ".xlsx")):
        return pd.read_excel(BytesIO(uploaded_file.read()))
    else:
        raise ValueError("⚠️ Formato de archivo no soportado. Usa .csv o .xlsx")
