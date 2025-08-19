# core/exportar.py
from fpdf import FPDF
import os

def _to_latin1_safe(texto: str) -> str:
    """
    FPDF clásico no maneja UTF-8 por defecto con fuentes base.
    Convertimos a latin-1; caracteres fuera de rango se reemplazan.
    """
    if not isinstance(texto, str):
        texto = str(texto)
    return texto.encode("latin-1", "replace").decode("latin-1")

class ExportadorPDF:
    """
    Clase para exportar el resumen y gráficos a un archivo PDF.
    """

    def __init__(self, carpeta_output: str = "output"):
        self.pdf = FPDF()

        self.pdf.set_auto_page_break(auto=True, margin=15)
        self.carpeta = carpeta_output
        os.makedirs(self.carpeta, exist_ok=True)

    def _page_width_available(self) -> float:
        return self.pdf.w - self.pdf.l_margin - self.pdf.r_margin

    def agregar_titulo(self, titulo: str):
        self.pdf.add_page()
        self.pdf.set_font("Arial", 'B', 16)
        self.pdf.cell(0, 10, _to_latin1_safe(titulo), ln=True, align="C")
        self.pdf.ln(2)

    def agregar_parrafo(self, texto: str):

        self.pdf.set_font("Arial", size=12)
        self.pdf.multi_cell(0, 6, _to_latin1_safe(texto))
        self.pdf.ln(1)

    def agregar_imagenes(self, extensiones: list | None = None):
        """
        Inserta en nuevas páginas todas las imágenes en la carpeta de salida.
        - Escala al ancho disponible respetando márgenes.
        - Inserta el nombre del archivo como pie de figura.
        """
        if extensiones is None:
            extensiones = [".png", ".jpg", ".jpeg", ".webp"]

        exts = {ext.lower() for ext in extensiones}

        try:
            archivos = sorted(os.listdir(self.carpeta))
        except FileNotFoundError:
            return

        max_w = self._page_width_available()

        any_image = False
        for archivo in archivos:
            _, ext = os.path.splitext(archivo)
            if ext.lower() not in exts:
                continue

            ruta = os.path.join(self.carpeta, archivo)
            if not os.path.isfile(ruta):
                continue

            any_image = True
            self.pdf.add_page()
         
            self.pdf.image(ruta, x=self.pdf.l_margin, w=max_w)
            self.pdf.ln(3)
            self.pdf.set_font("Arial", size=10)
            self.pdf.cell(0, 5, _to_latin1_safe(archivo), ln=True, align="C")


        if not any_image:
            if self.pdf.page_no() > 0:
                self.pdf.set_font("Arial", size=11)
                self.pdf.ln(4)
                self.pdf.cell(0, 6, _to_latin1_safe("No se encontraron imágenes para adjuntar."), ln=True)

    def guardar_pdf(self, nombre_archivo: str = "informe.pdf") -> str:
        ruta = os.path.join(self.carpeta, nombre_archivo)
        self.pdf.output(ruta)
        return ruta
