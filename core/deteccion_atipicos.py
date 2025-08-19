import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from pandas.api.types import is_numeric_dtype

class DeteccionAtipicos:
    """
    Métodos de detección de valores atípicos en variables numéricas.
    """

    @staticmethod
    def por_zscore(df: pd.DataFrame, umbral: float = 3.0) -> pd.DataFrame:
        """
        Devuelve un DataFrame booleano por columna indicando outliers por |Z| > umbral.
        """
        atipicos = pd.DataFrame(index=df.index)
        numericas = [c for c in df.columns if is_numeric_dtype(df[c])]

        if not numericas:
            return atipicos  # vacío, mismo índice

        dfn = df[numericas].apply(pd.to_numeric, errors="coerce")

        for col in dfn.columns:
            s = dfn[col]
            mu = s.mean()
            sd = s.std(ddof=0)
            if not np.isfinite(sd) or sd == 0:

                atipicos[col] = False
                continue
            z = (s - mu) / sd
            atipicos[col] = z.abs() > umbral

        return atipicos

    @staticmethod
    def por_rango_intercuartil(df: pd.DataFrame, mult: float = 1.5) -> pd.DataFrame:
        """
        Devuelve un DataFrame booleano por columna indicando outliers por IQR.
        """
        atipicos = pd.DataFrame(index=df.index)
        numericas = [c for c in df.columns if is_numeric_dtype(df[c])]

        if not numericas:
            return atipicos

        dfn = df[numericas].apply(pd.to_numeric, errors="coerce")

        for col in dfn.columns:
            s = dfn[col]
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            if not np.isfinite(iqr) or iqr == 0:
                atipicos[col] = False
                continue
            lo, hi = q1 - mult * iqr, q3 + mult * iqr
            atipicos[col] = (s < lo) | (s > hi)

        return atipicos

    @staticmethod
    def por_isolation_forest(
        df: pd.DataFrame,
        contamination: float = 0.05,
        random_state: int = 42,
        min_cols: int = 2,
        min_rows: int = 20,
    ) -> pd.Series:
        """
        Detecta registros atípicos con Isolation Forest.
        Retorna una Serie booleana alineada al índice del df original.
        """
        # Selección robusta de numéricas
        numericas = [c for c in df.columns if is_numeric_dtype(df[c])]
        if len(numericas) < min_cols or len(df) < min_rows:
            return pd.Series(False, index=df.index)

        dfn = df[numericas].apply(pd.to_numeric, errors="coerce")

        # Entrenar sólo con filas completas para estabilidad
        dfn_complete = dfn.dropna(axis=0, how="any")
        if len(dfn_complete) < min_rows:
            return pd.Series(False, index=df.index)

        # Entrenamiento
        modelo = IsolationForest(
            contamination=contamination,
            random_state=random_state,
        )
        pred = modelo.fit_predict(dfn_complete)

        # Serie booleana sobre las filas usadas
        out_complete = pd.Series(pred == -1, index=dfn_complete.index)
        out_full = out_complete.reindex(df.index, fill_value=False)

        return out_full
