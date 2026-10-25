# Datos originales (no versionados)

Conjunto: **Credit Risk Analysis** (Lending Club), Kaggle.
Enlace: https://www.kaggle.com/datasets/ranadeep/credit-risk-dataset
Fecha de descarga: 22 de octubre de 2026.

El archivo `loan.csv` (~421 MB) excede el límite de 100 MB de GitHub, por eso no está en el repositorio
(ver `.gitignore`). Para regenerar los datos procesados:

1. Descargar el zip desde Kaggle y extraer `loan.csv` en esta carpeta (`data/raw/`).
2. Ejecutar el script de procesamiento (ver README principal) para generar `data/processed/`.

El diccionario de variables (`LCDataDictionary.xlsx`) sí está versionado aquí y es la fuente
que se cita para el significado de cada variable.
