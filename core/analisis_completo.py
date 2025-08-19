import os
import streamlit as st
import pandas as pd

from core.analizar import (
    obtener_estadisticas_descriptivas,
    obtener_tipos_variables,
    obtener_matriz_correlacion,
)
from core.deteccion_atipicos import DeteccionAtipicos
from core.agrupamiento import Agrupamiento
from core.resumen import GeneradorResumen
from core.interpretador import InterpretadorInteligente
from core.visualizacion import CrearGraficos
from core.exportar import ExportadorPDF
from core.etiquetar_cluster import etiquetar_clusters


def ejecutar_analisis_completo(df: pd.DataFrame) -> None:
    """Orquesta el análisis completo y renderiza resultados en Streamlit."""
    df = df.copy()

    # --- Tipos de variables y stats generales ---
    tipos = obtener_tipos_variables(df)
    st.subheader("Tipos de variables")
    st.json(tipos)

    st.subheader("Estadísticas descriptivas")
    st.dataframe(obtener_estadisticas_descriptivas(df))

    # --- Correlaciones sobre numéricas ---
    st.subheader("Matriz de correlación")
    corr_df = obtener_matriz_correlacion(df)  
    if corr_df.empty:
        st.info("No hay suficientes columnas numéricas para calcular correlación.")
    else:
        st.dataframe(corr_df)

    # --- Detección de valores atípicos ---
    num_cols = tipos.get("numericas", [])
    df_num = df[num_cols].apply(pd.to_numeric, errors="coerce") if num_cols else pd.DataFrame()

    zscore = DeteccionAtipicos.por_zscore(df_num if not df_num.empty else df)
    iqr = DeteccionAtipicos.por_rango_intercuartil(df_num if not df_num.empty else df)
    forest = DeteccionAtipicos.por_isolation_forest(df_num if not df_num.empty else df)

    # --- Agrupamiento con KMeans
    if df_num.shape[1] >= 2 and len(df_num) >= 20:
        clusters = Agrupamiento.por_kmeans(df_num, num_clusters=3)  
       
        if hasattr(clusters, "shape") and len(clusters) != len(df):
            s = pd.Series(clusters, index=df_num.dropna().index)
            clusters = s.reindex(df.index, fill_value=-1).to_numpy()
        df["Cluster"] = clusters
    else:
        st.warning("No hay suficientes variables numéricas para agrupar (KMeans). Se omite clustering.")
        df["Cluster"] = -1  # etiqueta “sin cluster”

    # --- Perfil promedio por clúster + etiquetas ---
    if (df["Cluster"] != -1).any():
        perfil_cluster = df.groupby("Cluster").mean(numeric_only=True).round(2)
        etiquetas = etiquetar_clusters(perfil_cluster) or {}
        etiquetas_default = {k: f"Cluster {k}" for k in perfil_cluster.index}
        etiquetas_final = {**etiquetas_default, **etiquetas}

        perfil_cluster.index = perfil_cluster.index.map(etiquetas_final)
        df["Cluster Etiqueta"] = df["Cluster"].map(etiquetas_final)

        st.subheader("Clusters detectados")
        conteo_clusters = df["Cluster Etiqueta"].value_counts()
        st.bar_chart(conteo_clusters)

        st.subheader("Perfil promedio por clúster")
        st.dataframe(perfil_cluster)
    else:
        perfil_cluster = pd.DataFrame()
        etiquetas_final = {}
        st.info("Sin clusters válidos para perfilar/etiquetar.")

    # --- Resumen + interpretación ---
    resumen = (
        GeneradorResumen.resumen_agrupamiento(
            df.get("Cluster", pd.Series([-1] * len(df))).to_numpy(), etiquetas_final
        ) + "\n"
        + GeneradorResumen.resumen_atipicos(df, zscore, iqr, forest) + "\n"
        + GeneradorResumen.resumen_correlaciones(corr_df if not corr_df.empty else pd.DataFrame())
    ).replace("🔹", "-").replace("≥", ">=")

    interpretacion = (
        InterpretadorInteligente.sugerencias_atipicos(zscore, iqr, forest) + "\n"
        + InterpretadorInteligente.sugerencias_correlaciones(corr_df if not corr_df.empty else pd.DataFrame())
    )

    resumen_completo = resumen + "\n" + interpretacion

    st.subheader("Resumen generado")
    st.text_area("Resumen del análisis", resumen_completo, height=350)

    try:
        CrearGraficos(df)  
    except Exception as e:
        st.warning(f"No se pudieron generar algunos gráficos: {type(e).__name__}: {e}")

    # --- Exportación a PDF ---
    try:
        os.makedirs("output", exist_ok=True)
        exportador = ExportadorPDF("output")
        exportador.agregar_titulo("Informe de Análisis Automatizado")
        exportador.agregar_parrafo(resumen_completo)
        exportador.agregar_imagenes()

        ruta_pdf = exportador.guardar_pdf("informe_final.pdf")

        with open(ruta_pdf, "rb") as file:
            st.download_button("Descargar informe PDF", file, file_name="informe_final.pdf")
        st.success(f"PDF generado en: {ruta_pdf}")
    except Exception as e:
        st.error(f"No se pudo generar o descargar el PDF: {type(e).__name__}: {e}")
