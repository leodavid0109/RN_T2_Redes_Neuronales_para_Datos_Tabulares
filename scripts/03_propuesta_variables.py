"""Une la propuesta de variables con el perfil y la definición textual del diccionario.

Solo lectura sobre los datos. Guarda results/propuesta_variables_pd.csv.
Uso:  python scripts/03_propuesta_variables.py
"""
import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from src.variables import NOMBRE_EN_DICCIONARIO, VARIABLES  # noqa: E402

perfil = pd.read_csv(RAIZ / "results" / "perfil_variables.csv")
dic = pd.read_excel(RAIZ / "data" / "raw" / "LCDataDictionary.xlsx", sheet_name="LoanStats")
dic = dic.iloc[:, :2].dropna()
dic.columns = ["variable", "definicion"]
dic["variable"] = dic["variable"].astype(str).str.replace("\xa0", " ").str.strip()
definiciones = dict(zip(dic["variable"], dic["definicion"].astype(str).str.strip()))

columnas_csv = list(perfil["variable"])
faltan = [c for c in columnas_csv if c not in VARIABLES]
sobran = [c for c in VARIABLES if c not in columnas_csv]
assert not faltan and not sobran, f"Sin clasificar: {faltan} | no están en el CSV: {sobran}"

filas = []
for _, r in perfil.iterrows():
    v = r["variable"]
    cat, just = VARIABLES[v]
    nombre_dic = NOMBRE_EN_DICCIONARIO.get(v, v)
    filas.append({
        "variable": v,
        "categoria": cat,
        "justificacion": just,
        "pct_faltantes": r["pct_faltantes"],
        "n_unicos": r["n_unicos"],
        "primera_fecha_con_dato": r["primera_fecha_con_dato"],
        "definicion_diccionario_en": definiciones.get(nombre_dic, "(sin definición en el diccionario)"),
        "nota_nombre": f"el diccionario la llama {nombre_dic} (equivalencia supuesta)" if v in NOMBRE_EN_DICCIONARIO else "",
    })
tabla = pd.DataFrame(filas)
tabla.to_csv(RAIZ / "results" / "propuesta_variables_pd.csv", index=False)

print(tabla["categoria"].value_counts().to_string())
print("\nTotal de columnas clasificadas:", len(tabla))

# Verificación de la justificación de funded_amnt
df = pd.read_csv(RAIZ / "data" / "raw" / "loan.csv", usecols=["loan_amnt", "funded_amnt"])
print("\n% de créditos con funded_amnt == loan_amnt:", round(100 * (df.loan_amnt == df.funded_amnt).mean(), 2))
