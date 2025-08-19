import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype, is_bool_dtype, is_categorical_dtype

def _num_cols(df: pd.DataFrame):
    cols = [c for c in df.columns if is_numeric_dtype(df[c])]
    dfn = df[cols].apply(pd.to_numeric, errors="coerce") if cols else pd.DataFrame(index=df.index)
    dfn = dfn.replace([np.inf, -np.inf], np.nan)
    return dfn.loc[:, dfn.notna().any(axis=0)]

def _cat_cols(df: pd.DataFrame):
    cats = []
    for c in df.columns:
        s = df[c]
        if is_bool_dtype(s) or is_categorical_dtype(s) or s.dtype == object:
            cats.append(c)
    return cats

def mostrar_dashboard(df: pd.DataFrame):
    st.title("Dashboard Interactivo de Visualización")

    numericas_df = _num_cols(df)
    columnas_numericas = numericas_df.columns.tolist()
    columnas_categoricas = _cat_cols(df)

    tipo_grafico = st.selectbox(
        "Selecciona el tipo de gráfico",
        ["Histograma", "Boxplot", "Dispersión", "Barras por categoría"]
    )

    if tipo_grafico in ["Histograma", "Boxplot", "Dispersión"] and len(columnas_numericas) == 0:
        st.warning("No hay columnas numéricas válidas para este tipo de gráfico.")
        return
    if tipo_grafico == "Barras por categoría" and (len(columnas_categoricas) == 0 or len(columnas_numericas) == 0):
        st.warning("Se requiere al menos una columna categórica y una numérica.")
        return

    if tipo_grafico in ["Histograma", "Boxplot"]:
        variable = st.selectbox("Variable numérica", columnas_numericas, key="dash_var_num")
        serie = numericas_df[variable].dropna()

        # Controles extra
        if tipo_grafico == "Histograma":
            bins = st.slider("Bins (número de contenedores)", min_value=5, max_value=100, value=30)
            kde = st.checkbox("Mostrar KDE (curva de densidad)", value=True)
            if st.button("Generar histograma", key="btn_hist"):
                if serie.empty:
                    st.info("No hay datos válidos para graficar.")
                    return
                fig, ax = plt.subplots()
                sns.histplot(serie, bins=bins, kde=kde, ax=ax)
                ax.set_title(f"Histograma de {variable}")
                ax.set_xlabel(variable); ax.set_ylabel("Frecuencia")
                st.pyplot(fig)
        else:  
            if st.button("Generar boxplot", key="btn_box"):
                if serie.empty:
                    st.info("No hay datos válidos para graficar.")
                    return
                fig, ax = plt.subplots()
                sns.boxplot(x=serie, ax=ax, whis=1.5)
                ax.set_title(f"Boxplot de {variable}")
                st.pyplot(fig)

    elif tipo_grafico == "Dispersión":
        x = st.selectbox("Eje X", columnas_numericas, key="dash_x")
        y = st.selectbox("Eje Y", columnas_numericas, key="dash_y")
        sample_max = st.slider("Muestreo máximo de puntos", 1000, 100_000, 5000, step=1000)
        alpha = st.slider("Transparencia (alpha)", 10, 100, 60, step=5) / 100.0

        if st.button("Generar dispersión", key="btn_scatter"):
            plot_df = numericas_df[[x, y]].dropna()
            if plot_df.empty:
                st.info("No hay datos válidos para graficar.")
                return
            if len(plot_df) > sample_max:
                plot_df = plot_df.sample(sample_max, random_state=42)
            fig, ax = plt.subplots()
            sns.scatterplot(data=plot_df, x=x, y=y, ax=ax, alpha=alpha, edgecolor=None)
            ax.set_title(f"Dispersión: {x} vs {y}")
            st.pyplot(fig)

    elif tipo_grafico == "Barras por categoría":
        cat = st.selectbox("Variable categórica", columnas_categoricas, key="dash_cat")
        num = st.selectbox("Variable numérica", columnas_numericas, key="dash_num")
        agg = st.selectbox("Función de agregación", ["mean", "sum", "count", "median"], index=0)
        top_n = st.slider("Top-N categorías", 5, 50, 20)

        if st.button("Generar gráfico de barras", key="btn_bar"):
            s_cat = df[cat].astype("string").fillna("Desconocido")
            s_num = numericas_df[num] if num in numericas_df.columns else pd.to_numeric(df[num], errors="coerce")
            tmp = pd.DataFrame({cat: s_cat, num: s_num})
            if agg == "count":
                g = tmp.groupby(cat, dropna=False)[num].count().sort_values(ascending=False)
            else:
                g = tmp.groupby(cat, dropna=False)[num].agg(agg).sort_values(ascending=False)

            if g.empty:
                st.info("No hay datos válidos para graficar.")
                return

            g_top = g.head(top_n)
            otros_val = g.iloc[top_n:].sum() if len(g) > top_n else 0
            if otros_val != 0:
                g_top.loc["Otros"] = otros_val

            fig, ax = plt.subplots(figsize=(9, 4.5))
            g_top.sort_values(ascending=False).plot(kind="bar", ax=ax)
            ax.set_title(f"{agg} de {num} por {cat}")
            ax.set_ylabel(agg.capitalize())
            plt.xticks(rotation=45, ha="right")
            st.pyplot(fig)
