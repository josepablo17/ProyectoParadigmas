def etiquetar_clusters(perfil_cluster) -> dict:
    etiquetas = {}
    usadas = set()

    for idx, fila in perfil_cluster.iterrows():
        if fila["Revenue"] > 6000 and fila["Units Returned"] < 0.1:
            base = "Éxito en ventas"
        elif fila["Discount"] > 0.15:
            base = "Alta promoción"
        elif fila["Units Sold"] > 140:
            base = " Alta rotación"
        else:
            base = "Comportamiento promedio"

        etiqueta = base
        contador = 2
        while etiqueta in usadas:
            etiqueta = f"{base} ({contador})"
            contador += 1

        etiquetas[idx] = etiqueta
        usadas.add(etiqueta)

    return etiquetas
