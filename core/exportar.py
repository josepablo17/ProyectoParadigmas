from fpdf import FPDF
import os

class ExportadorPDF:
    """
    Clase para exportar el resumen y gráficos a un archivo PDF.
    """

    def __init__(self, carpeta_output: str = "output"):
        self.pdf = FPDF()
        self.pdf.set_auto_page_break(auto=True, margin=15)
        self.carpeta = carpeta_output

    def agregar_titulo(self, titulo: str):
        self.pdf.set_font("Arial", 'B', 16)
        self.pdf.add_page()
        self.pdf.cell(0, 10, titulo, ln=True, align="C")

    def agregar_parrafo(self, texto: str):
        self.pdf.set_font("Arial", size=12)
        self.pdf.multi_cell(0, 10, texto)

    def agregar_imagenes(self, extensiones: list = [".png"]):
        for archivo in sorted(os.listdir(self.carpeta)):
            if any(archivo.endswith(ext) for ext in extensiones):
                ruta = os.path.join(self.carpeta, archivo)
                self.pdf.add_page()
                self.pdf.image(ruta, w=180)
                self.pdf.set_font("Arial", size=10)
                self.pdf.ln(5)
                self.pdf.cell(0, 5, archivo, ln=True)

    def guardar_pdf(self, nombre_archivo: str = "informe.pdf"):
        self.pdf.output(os.path.join(self.carpeta, nombre_archivo))
