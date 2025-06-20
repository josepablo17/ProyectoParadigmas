import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def CrearGraficos(df: pd.DataFrame, carpeta_salida: str = "output"):
    """
    Genera gráficos automáticos según el tipo de variable y los guarda como imágenes.
    """

    os.makedirs(carpeta_salida, exist_ok=True)
    numericas = df.select_dtypes(include=['int64', 'float64'])
    categoricas = df.select_dtypes(include=['object', 'category', 'bool'])

    # Histograma por cada columna numérica
    for col in numericas.columns:
        plt.figure()
        sns.histplot(df[col], kde=True)
        plt.title(f"Histograma de {col}")
        plt.xlabel(col)
        plt.ylabel("Frecuencia")
        plt.tight_layout()
        plt.savefig(f"{carpeta_salida}/histograma_{col}.png")
        plt.close()

    # Boxplot por cada columna numérica
    for col in numericas.columns:
        plt.figure()
        sns.boxplot(x=df[col])
        plt.title(f"Boxplot de {col}")
        plt.tight_layout()
        plt.savefig(f"{carpeta_salida}/boxplot_{col}.png")
        plt.close()

    # Gráfico de barras por cada variable categórica
    for col in categoricas.columns:
        plt.figure(figsize=(8, 4))
        df[col].value_counts().plot(kind='bar')
        plt.title(f"Frecuencia de {col}")
        plt.ylabel("Cantidad")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{carpeta_salida}/barras_{col}.png")
        plt.close()

    # Diagrama de dispersión de las primeras 2 columnas numéricas (si hay)
    if numericas.shape[1] >= 2:
        plt.figure()
        sns.scatterplot(x=numericas.columns[0], y=numericas.columns[1], data=df)
        plt.title(f"Dispersión: {numericas.columns[0]} vs {numericas.columns[1]}")
        plt.tight_layout()
        plt.savefig(f"{carpeta_salida}/dispersión_{numericas.columns[0]}_vs_{numericas.columns[1]}.png")
        plt.close()

    # Mapa de calor de correlación
    if numericas.shape[1] >= 2:
        plt.figure(figsize=(10, 8))
        correlacion = numericas.corr()
        sns.heatmap(correlacion, annot=True, cmap='coolwarm')
        plt.title("Mapa de calor - Correlaciones")
        plt.tight_layout()
        plt.savefig(f"{carpeta_salida}/mapa_calor_correlaciones.png")
        plt.close()
