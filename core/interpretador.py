# core/interpretador.py

class InterpretadorInteligente:

    @staticmethod
    def sugerencias_atipicos(zscore, iqr, forest):
        total_z = int(zscore.sum().sum())
        total_iqr = int(iqr.sum().sum())
        total_forest = int(forest.sum())

        mensaje = "Sugerencias basadas en detección de atípicos:\n"
        
        if total_z > 100:
            mensaje += "- Hay una cantidad considerable de atípicos por Z-score. Revisa la escala de medición de algunas variables.\n"
        if total_iqr > 100:
            mensaje += "- El método IQR detectó muchos extremos. Podría haber errores de captura o valores no representativos.\n"
        if total_forest > 50:
            mensaje += "- Isolation Forest detectó anomalías en múltiples registros. Considera revisarlos individualmente.\n"
        if total_z < 5 and total_iqr < 5 and total_forest < 5:
            mensaje += "- No se detectaron muchos valores atípicos. El dataset parece estar bien distribuido.\n"

        return mensaje

    @staticmethod
    def sugerencias_correlaciones(correlaciones, umbral=0.8):
        mensaje = "Sugerencias basadas en correlaciones fuertes:\n"
        fuertes = []
        for i in range(len(correlaciones.columns)):
            for j in range(i + 1, len(correlaciones.columns)):
                coef = correlaciones.iloc[i, j]
                if abs(coef) >= umbral:
                    fuertes.append((correlaciones.columns[i], correlaciones.columns[j], coef))

        if fuertes:
            mensaje += "- Se detectaron variables con alta correlación. Podrías eliminar una de las dos para evitar redundancia:\n"
            for var1, var2, coef in fuertes:
                mensaje += f"  {var1} y {var2}: r = {coef:.2f}\n"
        else:
            mensaje += "- No se encontraron correlaciones significativas. Las variables parecen ser independientes.\n"

        return mensaje
