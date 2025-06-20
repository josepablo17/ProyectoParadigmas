import streamlit as st
import pandas as pd

from core.cargar import CargarDatos
from core.analizar import (
    obtener_estadisticas_descriptivas,
    obtener_tipos_variables,
    obtener_matriz_correlacion
)
from core.deteccion_atipicos import DeteccionAtipicos
from core.agrupamiento import Agrupamiento
from core.resumen import GeneradorResumen
from core.visualizacion import CrearGraficos
from core.exportar import ExportadorPDF

st.set_page_config(page_title="Análisis Inteligente de Datos", layout="wide")
st.title("🧠 Sistema de Análisis Automatizado de Datos")

archivo = st.file_uploader("📁 Carga un archivo CSV o Excel", type=["csv", "xlsx"])

if archivo:
    try:
        df = pd.read_csv(archivo) if archivo.name.endswith("csv") else pd.read_excel(archivo)
        st.success("✅ Archivo cargado correctamente.")
        st.dataframe(df.head())

        if st.button("🚀 Ejecutar análisis completo"):
            tipos = obtener_tipos_variables(df)
            st.subheader("📌 Tipos de variables")
            st.json(tipos)

            st.subheader("📊 Estadísticas descriptivas")
            st.dataframe(obtener_estadisticas_descriptivas(df))

            st.subheader("📈 Matriz de correlación")
            correlaciones = obtener_matriz_correlacion(df)
            st.dataframe(correlaciones)

            zscore = DeteccionAtipicos.por_zscore(df)
            iqr = DeteccionAtipicos.por_rango_intercuartil(df)
            forest = DeteccionAtipicos.por_isolation_forest(df)

            clusters = Agrupamiento.por_kmeans(df, num_clusters=3)
            df["Cluster"] = clusters
            st.subheader("🧩 Clusters detectados")
            st.bar_chart(clusters.value_counts())

            resumen = (
                GeneradorResumen.resumen_agrupamiento(clusters) + "\n" +
                GeneradorResumen.resumen_atipicos(df, zscore, iqr, forest) + "\n" +
                GeneradorResumen.resumen_correlaciones(correlaciones)
            ).replace("🔹", "-").replace("≥", ">=")

            st.subheader("📋 Resumen generado")
            st.text_area("Resumen", resumen, height=300)

            CrearGraficos(df)

            exportador = ExportadorPDF("output")
            exportador.agregar_titulo("Informe de Análisis Automatizado")
            exportador.agregar_parrafo(resumen)
            exportador.agregar_imagenes()
            exportador.guardar_pdf("informe_final.pdf")

            with open("output/informe_final.pdf", "rb") as file:
                st.download_button("📄 Descargar informe PDF", file, file_name="informe_final.pdf")

    except Exception as e:
        st.error(f"❌ Error al procesar los datos: {e}")
