# core/resumen.py
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, Iterable

class GeneradorResumen:
    @staticmethod
    def _fmt_pct(x: float, decimals: int = 1) -> str:
        try:
            return f"{100 * float(x):.{decimals}f}%"
        except Exception:
            return "—"

    @staticmethod
    def _fmt_num(x, decimals: int = 2) -> str:
        try:
            xf = float(x)
            if abs(xf) >= 1000:
                return f"{xf:,.{decimals}f}"
            return f"{xf:.{decimals}f}"
        except Exception:
            return "—"

    @staticmethod
    def resumen_agrupamiento(clusters: Iterable, etiquetas: Dict) -> str:
        """
        clusters: array-like de asignaciones (int), puede contener -1 (sin cluster)
        etiquetas: dict {cluster_id: 'nombre legible'}
        """
        if clusters is None:
            return "• No se realizó agrupamiento."

        s = pd.Series(clusters)
        if s.empty:
            return "• No se realizó agrupamiento."

        total = len(s)
        conteo = s.value_counts(dropna=False).sort_index()
        partes = ["Agrupamiento (K-Means):"]

        for cid, cnt in conteo.items():
            etiqueta = etiquetas.get(cid, f"Cluster {cid}")
            partes.append(f"- {etiqueta}: {cnt} ({GeneradorResumen._fmt_pct(cnt/total)})")

        # pequeña pista de calidad (opcional si luego calculas silhouette aparte)
        if (-1 in conteo.index) and (conteo[-1] > 0):
            partes.append("- Nota: hay filas sin cluster asignado (−1).")

        return "\n".join(partes)

    @staticmethod
    def resumen_atipicos(
        df: pd.DataFrame,
        zscore_flags: pd.DataFrame | None,
        iqr_flags: pd.DataFrame | None,
        forest_flags: pd.Series | None,
        top_n: int = 5,
    ) -> str:
        """
        zscore_flags / iqr_flags: DataFrames booleanos por columna (True=outlier)
        forest_flags: Serie booleana por fila (True=registro atípico)
        """
        partes = ["Valores atípicos:"]

        # Z-score
        if isinstance(zscore_flags, pd.DataFrame) and not zscore_flags.empty:
            rates = zscore_flags.mean(numeric_only=True).sort_values(ascending=False)
            top = rates.head(top_n)
            if not top.empty:
                partes.append("- Z-score (por columna):")
                for col, rate in top.items():
                    partes.append(f"  • {col}: {GeneradorResumen._fmt_pct(rate)}")
        else:
            partes.append("- Z-score: no aplicable o sin columnas numéricas.")

        # IQR
        if isinstance(iqr_flags, pd.DataFrame) and not iqr_flags.empty:
            rates = iqr_flags.mean(numeric_only=True).sort_values(ascending=False)
            top = rates.head(top_n)
            if not top.empty:
                partes.append("- IQR (por columna):")
                for col, rate in top.items():
                    partes.append(f"  • {col}: {GeneradorResumen._fmt_pct(rate)}")
        else:
            partes.append("- IQR: no aplicable o sin columnas numéricas.")

        # Isolation Forest
        if isinstance(forest_flags, pd.Series) and not forest_flags.empty:
            rate = float(forest_flags.mean())
            partes.append(f"- Isolation Forest (por fila): {GeneradorResumen._fmt_pct(rate)} de registros atípicos.")
        else:
            partes.append("- Isolation Forest: no aplicable.")

        return "\n".join(partes)

    @staticmethod
    def resumen_correlaciones(corr_df: pd.DataFrame | None, top_n: int = 5) -> str:
        """
        corr_df: matriz de correlación (num_cols x num_cols).
        Reporta top-N correlaciones absolutas (excluyendo diagonal y duplicados).
        """
        if corr_df is None or corr_df.empty or corr_df.shape[0] < 2:
            return "Correlaciones: no hay suficientes variables numéricas para evaluar."

        # Convertir a formato largo y quitar duplicados (i<j)
        c = corr_df.copy()
        np.fill_diagonal(c.values, np.nan)
        long = (
            c.abs()
             .stack()
             .reset_index()
             .rename(columns={"level_0": "var1", "level_1": "var2", 0: "abs_corr"})
        )

        # Mantener sólo pares únicos (var1 < var2 en orden)
        mask = long["var1"] < long["var2"]
        long = long[mask]

        top = long.sort_values("abs_corr", ascending=False).head(top_n)

        if top.empty:
            return "Correlaciones: sin relaciones destacadas."

        partes = ["Correlaciones destacadas (|r|):"]
        for _, row in top.iterrows():
            v1, v2, r = row["var1"], row["var2"], row["abs_corr"]
            partes.append(f"- {v1} ↔ {v2}: {GeneradorResumen._fmt_num(r, 3)}")

        return "\n".join(partes)
