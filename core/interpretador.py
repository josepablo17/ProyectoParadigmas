# core/interpretador.py
from __future__ import annotations
import pandas as pd
import numpy as np

class InterpretadorInteligente:

    @staticmethod
    def sugerencias_atipicos(zscore, iqr, forest) -> str:
        """
        zscore: DataFrame booleano (True = outlier por Z)
        iqr:    DataFrame booleano (True = outlier por IQR)
        forest: Serie/array booleano (True = fila atípica) o None
        """
        mensaje = "Sugerencias basadas en detección de atípicos:\n"

        # Contar con defensas
        def safe_sum_bool(x):
            if isinstance(x, pd.DataFrame):
                return int(x.sum(numeric_only=True).sum())
            if isinstance(x, pd.Series):
                return int(x.sum())
            if isinstance(x, (np.ndarray, list, tuple)):
                arr = np.asarray(x)
                return int(np.nansum(arr.astype(float)))
            return 0

        total_z = safe_sum_bool(zscore)
        total_iqr = safe_sum_bool(iqr)
        total_forest = safe_sum_bool(forest)

        # Estimar tamaño de referencia (filas x cols num)
        n_rows = 0
        n_cols_num = 0
        if isinstance(zscore, pd.DataFrame) and not zscore.empty:
            n_rows = len(zscore)
            n_cols_num = zscore.shape[1]
        elif isinstance(iqr, pd.DataFrame) and not iqr.empty:
            n_rows = len(iqr)
            n_cols_num = iqr.shape[1]

        n_cells = max(n_rows * max(n_cols_num, 1), 1)
        rate_z = total_z / n_cells if n_cells else 0.0
        rate_iqr = total_iqr / n_cells if n_cells else 0.0

        # Reglas (absolutas + relativas)
        if total_z > 100 or rate_z >= 0.05:
            mensaje += "- Hay muchos atípicos por Z-score. Revisa escalas/unidades y considera estandarizar o winsorizar.\n"
        if total_iqr > 100 or rate_iqr >= 0.05:
            mensaje += "- IQR detectó numerosos extremos. Revisa outliers por fuente de dato (errores de captura) o aplica límites por percentil.\n"
        if total_forest > 50 or (n_rows and (total_forest / n_rows) >= 0.05):
            mensaje += "- Isolation Forest marcó múltiples registros. Prioriza inspección manual de esos casos y valida reglas de negocio.\n"

        if (total_z + total_iqr + total_forest) == 0:
            mensaje += "- No se detectaron valores atípicos relevantes. El dataset luce estable.\n"

        return mensaje

    @staticmethod
    def sugerencias_correlaciones(correlaciones, umbral: float = 0.8) -> str:
        """
        correlaciones: DataFrame de correlación (p. ej., Pearson/Spearman)
        """
        mensaje = "Sugerencias basadas en correlaciones fuertes:\n"

        if not isinstance(correlaciones, pd.DataFrame) or correlaciones.empty or correlaciones.shape[1] < 2:
            mensaje += "- No hay suficientes variables numéricas o la matriz está vacía.\n"
            return mensaje

        # Tomar pares únicos i<j
        cols = list(correlaciones.columns)
        fuertes = []
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                coef = correlaciones.iloc[i, j]
                if pd.notna(coef) and abs(coef) >= umbral:
                    fuertes.append((cols[i], cols[j], float(coef)))

        if fuertes:
            # Ordenar por |r| descendente y limitar
            fuertes.sort(key=lambda t: abs(t[2]), reverse=True)
            fuertes = fuertes[:10]
            mensaje += "- Se detectaron relaciones muy fuertes. Considera eliminar/combinar variables para evitar multicolinealidad:\n"
            for v1, v2, r in fuertes:
                mensaje += f"  • {v1} ↔ {v2}: r = {r:.2f}\n"
            mensaje += "- Si vas a modelar, evalúa VIF/regularización o selecciona una sola variable por grupo correlacionado.\n"
        else:
            mensaje += "- No se encontraron correlaciones ≥ {:.2f}. Las variables parecen relativamente independientes.\n".format(umbral)

        return mensaje
