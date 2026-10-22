"""Compara las columnas del CSV con las variables del diccionario (hoja LoanStats).

Solo lectura. Guarda results/comparacion_columnas_diccionario.csv.
Uso:  python scripts/01_comparar_diccionario.py
"""
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
CSV = RAIZ / "data" / "raw" / "loan.csv"
XLSX = RAIZ / "data" / "raw" / "LCDataDictionary.xlsx"

columnas_csv = list(pd.read_csv(CSV, nrows=0).columns)
dic = pd.read_excel(XLSX, sheet_name="LoanStats")
# la primera columna son los nombres; se quitan espacios y filas vacías o notas al pie
nombres_dic = dic.iloc[:, 0].dropna().astype(str).str.replace("\xa0", " ").str.strip()
nombres_dic = [n for n in nombres_dic if n and n != "LoanStatNew"]

filas = [{"variable": c, "en_csv": c in columnas_csv, "en_diccionario": c in nombres_dic}
         for c in sorted(set(columnas_csv) | set(nombres_dic))]
tabla = pd.DataFrame(filas)
tabla.to_csv(RAIZ / "results" / "comparacion_columnas_diccionario.csv", index=False)

print(f"Columnas en el CSV: {len(columnas_csv)} | variables en el diccionario: {len(nombres_dic)}")
print("\nEn el CSV pero SIN definición en el diccionario:")
print(tabla.loc[tabla.en_csv & ~tabla.en_diccionario, "variable"].to_string(index=False))
print("\nEn el diccionario pero NO en el CSV:")
print(tabla.loc[~tabla.en_csv & tabla.en_diccionario, "variable"].to_string(index=False))
