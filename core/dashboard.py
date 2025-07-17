import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def mostrar_dashboard(df: pd.DataFrame):
    st.title("Dashboard Interactivo de Visualización")

    tipo_grafico = st.selectbox("Selecciona el tipo de gráfico", ["Histograma", "Boxplot", "Dispersión", "Barras por categoría"])

    columnas_numericas = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    columnas_categoricas = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    if tipo_grafico in ["Histograma", "Boxplot"]:
        variable = st.selectbox("Selecciona una variable numérica", columnas_numericas, key="var_numerica")

        if st.button(f"Generar {tipo_grafico.lower()}", key=f"{tipo_grafico}_btn"):
            fig, ax = plt.subplots()
            if tipo_grafico == "Histograma":
                sns.histplot(df[variable], kde=True, ax=ax)
            else:
                sns.boxplot(x=df[variable], ax=ax)
            st.pyplot(fig)

    elif tipo_grafico == "Dispersión":
        x = st.selectbox("Eje X", columnas_numericas, key="eje_x")
        y = st.selectbox("Eje Y", columnas_numericas, key="eje_y")

        if st.button("Generar dispersión", key="dispersion_btn"):
            fig, ax = plt.subplots()
            sns.scatterplot(x=df[x], y=df[y], ax=ax)
            st.pyplot(fig)

    elif tipo_grafico == "Barras por categoría":
        cat = st.selectbox("Variable categórica", columnas_categoricas, key="cat_var")
        num = st.selectbox("Variable numérica", columnas_numericas, key="num_var")

        if st.button("Generar gráfico de barras", key="barras_btn"):
            fig, ax = plt.subplots()
            sns.barplot(x=df[cat], y=df[num], ax=ax)
            plt.xticks(rotation=45)
            st.pyplot(fig)
