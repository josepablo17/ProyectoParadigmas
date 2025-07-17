import streamlit as st
from core.analizar import (
    obtener_estadisticas_descriptivas,
    obtener_tipos_variables,
    obtener_matriz_correlacion
)
from core.deteccion_atipicos import DeteccionAtipicos
from core.agrupamiento import Agrupamiento
from core.resumen import GeneradorResumen
from core.interpretador import InterpretadorInteligente
from core.visualizacion import CrearGraficos
from core.exportar import ExportadorPDF
from core.etiquetar_cluster import etiquetar_clusters 

def ejecutar_analisis_completo(df):
    tipos = obtener_tipos_variables(df)
    st.subheader("Tipos de variables")
    st.json(tipos)

    st.subheader("Estadísticas descriptivas")
    st.dataframe(obtener_estadisticas_descriptivas(df))

    st.subheader("Matriz de correlación")
    correlaciones = obtener_matriz_correlacion(df)
    st.dataframe(correlaciones)

    # Detección de valores atípicos
    zscore = DeteccionAtipicos.por_zscore(df)
    iqr = DeteccionAtipicos.por_rango_intercuartil(df)
    forest = DeteccionAtipicos.por_isolation_forest(df)

    # Agrupamiento con KMeans
    clusters = Agrupamiento.por_kmeans(df, num_clusters=3)
    df["Cluster"] = clusters

    # Crear perfil promedio por clúster
    perfil_cluster = df.groupby("Cluster").mean(numeric_only=True).round(2)

    # Etiquetar los clusters
    etiquetas = etiquetar_clusters(perfil_cluster)
    perfil_cluster.index = perfil_cluster.index.map(etiquetas)
    df["Cluster Etiqueta"] = df["Cluster"].map(etiquetas)

    # Visualización
    st.subheader("Clusters detectados")
    conteo_clusters = df["Cluster Etiqueta"].value_counts()
    st.bar_chart(conteo_clusters)

    st.subheader("Perfil promedio por clúster")
    st.dataframe(perfil_cluster)

    # Generar resumen
    resumen = (
        GeneradorResumen.resumen_agrupamiento(clusters,etiquetas) + "\n" +
        GeneradorResumen.resumen_atipicos(df, zscore, iqr, forest) + "\n" +
        GeneradorResumen.resumen_correlaciones(correlaciones)
    ).replace("🔹", "-").replace("≥", ">=")

    interpretacion = (
        InterpretadorInteligente.sugerencias_atipicos(zscore, iqr, forest) + "\n" +
        InterpretadorInteligente.sugerencias_correlaciones(correlaciones)
    )

    resumen_completo = resumen + "\n" + interpretacion

    st.subheader("Resumen generado")
    st.text_area("Resumen del análisis", resumen_completo, height=350)

    # Gráficos adicionales
    CrearGraficos(df)

    # Exportar informe
    exportador = ExportadorPDF("output")
    exportador.agregar_titulo("Informe de Análisis Automatizado")
    exportador.agregar_parrafo(resumen_completo)
    exportador.agregar_imagenes()
    exportador.guardar_pdf("informe_final.pdf")

    with open("output/informe_final.pdf", "rb") as file:
        st.download_button("Descargar informe PDF", file, file_name="informe_final.pdf")
