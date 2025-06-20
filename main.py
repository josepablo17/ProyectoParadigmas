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

def main():
    ruta = "data/Suplementos.csv"

    try:
        # Paso 1: Cargar datos
        df = CargarDatos(ruta)
        print("✅ Datos cargados correctamente. Vista previa:")
        print(df.head())

        # Paso 2: Identificar tipos de variables
        tipos = obtener_tipos_variables(df)
        print("\n🔍 Tipos de variables detectadas:")
        print(tipos)

        # Paso 3: Estadísticas descriptivas
        print("\n📊 Estadísticas descriptivas:")
        print(obtener_estadisticas_descriptivas(df))

        # Paso 4: Correlaciones
        print("\n📈 Matriz de correlación (Pearson):")
        correlaciones = obtener_matriz_correlacion(df)
        if not correlaciones.empty:
            print(correlaciones)
        else:
            print("⚠️ No hay suficientes columnas numéricas para calcular correlaciones.")

        # Paso 5: Detección de atípicos por 3 métodos
        print("\n🚨 Atípicos (Z-score):")
        zscore = DeteccionAtipicos.por_zscore(df)
        print(zscore.sum())

        print("\n🚨 Atípicos (Rango intercuartílico - IQR):")
        iqr = DeteccionAtipicos.por_rango_intercuartil(df)
        print(iqr.sum())

        print("\n🚨 Atípicos (Isolation Forest):")
        forest = DeteccionAtipicos.por_isolation_forest(df)
        print(f"Registros atípicos detectados: {forest.sum()}")

        # Paso 6: Clustering (agrupamiento)
        print("\n🧩 Agrupamiento automático (K-Means):")
        clusters = Agrupamiento.por_kmeans(df, num_clusters=3)
        df["Cluster"] = clusters
        print(clusters.value_counts())

        # Paso 7: Generar resumen del análisis
        print("\n📋 Resumen general del análisis:")
        print(GeneradorResumen.resumen_agrupamiento(clusters))
        print(GeneradorResumen.resumen_atipicos(df, zscore, iqr, forest))
        print(GeneradorResumen.resumen_correlaciones(correlaciones))

        print("\n🖼️ Generando gráficos automáticos...")
        CrearGraficos(df)
        print("✅ Gráficos guardados en la carpeta 'output/'")

        resumen_completo = (
        GeneradorResumen.resumen_agrupamiento(clusters) + "\n" +
        GeneradorResumen.resumen_atipicos(df, zscore, iqr, forest) + "\n" +
        GeneradorResumen.resumen_correlaciones(correlaciones)
        )
        resumen_completo = resumen_completo.replace("🔹", "-")
        resumen_completo = resumen_completo.replace("≥", ">=")

        print("\n📝 Exportando resumen y gráficos a PDF...")
        exportador = ExportadorPDF("output")
        exportador.agregar_titulo("Informe de Análisis Automatizado")
        exportador.agregar_parrafo(resumen_completo)
        exportador.agregar_imagenes()
        exportador.guardar_pdf("informe_final.pdf")
        print("✅ Informe guardado en 'output/informe_final.pdf'")


    except Exception as e:
        print(f"❌ Error al procesar los datos: {e}")

if __name__ == "__main__":
    main()
