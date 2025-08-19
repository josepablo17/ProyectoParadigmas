from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict

def etiquetar_clusters(perfil_cluster: pd.DataFrame) -> Dict:
    """
    Asigna etiquetas legibles a cada cluster basadas en métricas promedio.
    Reglas (mismas que las tuyas):
      - Éxito en ventas: Revenue > 6000 y Units Returned < 0.1
      - Alta promoción: Discount > 0.15
      - Alta rotación: Units Sold > 140
      - Comportamiento promedio: en caso contrario

    Notas:
    - Robusto a columnas faltantes (usa 0 por defecto).
    - Convierte a numérico (errores→NaN→0) para evitar comparaciones inválidas.
    - Asegura unicidad: agrega sufijos (2), (3) si se repite.
    """
    if perfil_cluster is None or len(perfil_cluster) == 0:
        return {}

    # Asegurar que las columnas existan y sean numéricas
    cols = ["Revenue", "Units Returned", "Discount", "Units Sold"]
    safe = perfil_cluster.copy()

    for c in cols:
        if c not in safe.columns:
            # columna inexistente → asumir 0
            safe[c] = 0
        else:
            
            safe[c] = pd.to_numeric(safe[c], errors="coerce").fillna(0)

    etiquetas: Dict = {}
    usadas = set()

    # Iterar en orden determinista por índice 
    for idx, fila in safe.sort_index().iterrows():
        rev = float(fila["Revenue"])
        ret = float(fila["Units Returned"])
        disc = float(fila["Discount"])
        sold = float(fila["Units Sold"])

        if (rev > 6000) and (ret < 0.1):
            base = "Éxito en ventas"
        elif disc > 0.15:
            base = "Alta promoción"
        elif sold > 140:
            base = "Alta rotación"    
        else:
            base = "Comportamiento promedio"

        etiqueta = base
        k = 2
        while etiqueta in usadas:
            etiqueta = f"{base} ({k})"
            k += 1

        etiquetas[idx] = etiqueta
        usadas.add(etiqueta)

    return etiquetas
