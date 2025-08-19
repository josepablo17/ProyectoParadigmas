import os
import re
import unicodedata
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.api.types import is_numeric_dtype, is_bool_dtype, is_categorical_dtype

def _safe_name(name: str) -> str:
    # Normaliza: quita acentos, espacios → '_', solo [A-Za-z0-9_-]
    nfkd = unicodedata.normalize("NFKD", str(name))
    s = "".join(c for c in nfkd if not unicodedata.combining(c))
    s = re.sub(r"\s+", "_", s.strip())
    s = re.sub(r"[^A-Za-z0-9_\-]", "", s)
    return s or "col"

def _select_numeric(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in df.columns if is_numeric_dtype(df[c])]
    if not cols:
        return pd.DataFrame(index=df.index)
    dfn = df[cols].apply(pd.to_numeric, errors="coerce")
    # Reemplaza inf/-inf por NaN
    dfn = dfn.replace([np.inf, -np.inf], np.nan)
    # Quita columnas totalmente vacías/constantes
    non_all_nan = dfn.columns[dfn.notna().any()].tolist()
    dfn = dfn[non_all_nan]
    const = [c for c in dfn.columns if dfn[c].nunique(dropna=True) <= 1]
    dfn = dfn.drop(columns=const, errors="ignore")
    return dfn

def _select_categorical(df: pd.DataFrame) -> pd.DataFrame:
    cats = []
    for c in df.columns:
        s = df[c]
        if is_bool_dtype(s) or is_categorical_dtype(s) or s.dtype == object:
            cats.append(c)
    return df[cats] if cats else pd.DataFrame(index=df.index)

def CrearGraficos(df: pd.DataFrame, carpeta_salida: str = "output"):
    """
    Genera gráficos automáticos según el tipo de variable y los guarda como imágenes.
    """
    os.makedirs(carpeta_salida, exist_ok=True)

    numericas = _select_numeric(df)
    categoricas = _select_categorical(df)

    # ---------- Histogramas (numéricas) ----------
    for col in numericas.columns:
        try:
            serie = numericas[col].dropna()
            if serie.empty:
                continue
            plt.figure()
            # bins heurístico (Sturges) y kde solo si no es discreto
            is_discrete = (serie.dropna().round().eq(serie.dropna())).mean() > 0.95
            sns.histplot(serie, kde=not is_discrete, bins="sturges")
            plt.title(f"Histograma de {col}")
            plt.xlabel(col); plt.ylabel("Frecuencia")
            plt.tight_layout()
            fname = f"{carpeta_salida}/histograma_{_safe_name(col)}.png"
            plt.savefig(fname, dpi=140, bbox_inches="tight")
        except Exception:
            pass
        finally:
            plt.close()

    # ---------- Boxplots (numéricas) ----------
    for col in numericas.columns:
        try:
            serie = numericas[col].dropna()
            if serie.empty:
                continue
            plt.figure()
            sns.boxplot(x=serie, whis=1.5)
            plt.title(f"Boxplot de {col}")
            plt.tight_layout()
            fname = f"{carpeta_salida}/boxplot_{_safe_name(col)}.png"
            plt.savefig(fname, dpi=140, bbox_inches="tight")
        except Exception:
            pass
        finally:
            plt.close()

    # ---------- Barras (categóricas) ----------
    TOP_N = 20
    for col in categoricas.columns:
        try:
            s = categoricas[col].astype("string").fillna("Desconocido")
            vc = s.value_counts(dropna=False)
            if vc.empty:
                continue
            top = vc.head(TOP_N)
            otros = vc.iloc[TOP_N:].sum()
            if otros > 0:
                top.loc["Otros"] = otros
            plt.figure(figsize=(9, 4.5))
            top.sort_values(ascending=False).plot(kind="bar")
            plt.title(f"Frecuencia de {col}")
            plt.ylabel("Cantidad"); plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            fname = f"{carpeta_salida}/barras_{_safe_name(col)}.png"
            plt.savefig(fname, dpi=140, bbox_inches="tight")
        except Exception:
            pass
        finally:
            plt.close()

    # ---------- Dispersión (las 2 numéricas de mayor varianza) ----------
    if numericas.shape[1] >= 2:
        try:
            # selecciona top 2 por varianza
            var = numericas.var(numeric_only=True).sort_values(ascending=False)
            x_col, y_col = var.index[:2].tolist()
            plot_df = numericas[[x_col, y_col]].dropna()
            if len(plot_df) > 5000:
                plot_df = plot_df.sample(5000, random_state=42)
            plt.figure()
            sns.scatterplot(x=x_col, y=y_col, data=plot_df, alpha=0.6, edgecolor=None)
            plt.title(f"Dispersión: {x_col} vs {y_col}")
            plt.tight_layout()
            fname = f"{carpeta_salida}/dispersion_{_safe_name(x_col)}_vs_{_safe_name(y_col)}.png"
            plt.savefig(fname, dpi=140, bbox_inches="tight")
        except Exception:
            pass
        finally:
            plt.close()

    # ---------- Heatmap de correlación (Spearman + triángulo) ----------
    if numericas.shape[1] >= 2:
        try:
            corr = numericas.corr(method="spearman")
            annot_ok = corr.shape[0] <= 12
            mask = np.triu(np.ones_like(corr, dtype=bool))
            plt.figure(figsize=(min(1.2 * corr.shape[0], 12), min(1.2 * corr.shape[0], 12)))
            sns.heatmap(corr, mask=mask, annot=annot_ok, fmt=".2f", cmap="coolwarm", square=True)
            plt.title("Mapa de calor - Correlaciones (Spearman)")
            plt.tight_layout()
            fname = f"{carpeta_salida}/mapa_calor_correlaciones.png"
            plt.savefig(fname, dpi=160, bbox_inches="tight")
        except Exception:
            pass
        finally:
            plt.close()
