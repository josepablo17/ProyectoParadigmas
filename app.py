import streamlit as st

from core.cargar import cargar_datos
from core.analisis_completo import ejecutar_analisis_completo
from core.dashboard import mostrar_dashboard
from core.procesamiento import preparar_datos

# Configuración inicial de la página
st.set_page_config(page_title="Análisis Inteligente de Datos", layout="wide")

# Título y descripción
st.title("Sistema de Análisis Automatizado de Datos")
st.markdown("""
Bienvenido al sistema de análisis inteligente de datos. Aquí podrás:

- Cargar tus archivos en formato CSV o Excel
- Limpiar y preparar automáticamente tus datos
- Visualizar estadísticas, correlaciones y agrupamientos
- Generar reportes PDF detallados del análisis

Sube tu archivo para comenzar 
""")

# Subida de archivo
archivo = st.file_uploader("Carga un archivo CSV o Excel", type=["csv", "xlsx"])
st.divider()

# Procesamiento principal
if archivo:
    try:
        # Cargar datos
        df = cargar_datos(archivo)
        st.success("Archivo cargado correctamente.")

        # Mostrar vista previa dentro de un contenedor expandible
        with st.expander("Vista previa del archivo cargado"):
            st.dataframe(df.head())

        # Limpieza de datos
        with st.spinner("Limpiando y preparando datos..."):
            df, resumen_limpieza = preparar_datos(df)

        st.info("Datos limpiados automáticamente.")

        # Mostrar resumen de limpieza en un panel expandible
        with st.expander("Ver resumen de limpieza de datos"):
            st.markdown(f"""
            - Columnas eliminadas por alto porcentaje de nulos: **{resumen_limpieza['columnas_eliminadas_por_nulos']}**
            - Registros duplicados eliminados: **{resumen_limpieza['duplicados_eliminados']}**
            - Celdas con valores nulos que fueron rellenadas: **{resumen_limpieza['nulos_rellenados']}**
            """)

        st.divider()

        # Selector de acción
        opcion = st.radio("¿Qué deseas hacer?", ["Dashboard Interactivo", "Análisis Completo"])

        if opcion == "Dashboard Interactivo":
            mostrar_dashboard(df)

        elif opcion == "Análisis Completo":
            ejecutar_analisis_completo(df)

    except Exception as e:
        st.error(f"Error al procesar los datos: {e}")
