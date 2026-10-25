"""Tabla de coeficientes de la logística para las variables de las hipótesis (Tabla 10) y tabla final de hipótesis.

Combina: resultado descriptivo (results/tabla07_resultados_hipotesis.csv), coeficientes de la regresión logística
(results/regresion_logistica_coeficientes.csv) y, si existe, la importancia de variables de la interpretabilidad
(contrato: results/importancia_variables.csv con columnas variable [las 24 originales], importancia y metodo).
Las variables numéricas están estandarizadas: el coeficiente es el cambio del logit por desviación estándar.
Salidas: results/tabla10_coeficientes_hipotesis.csv y results/tabla_hipotesis_final.csv
Uso:  python scripts/12_tabla_hipotesis_final.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
UMBRAL_NULO = 0.02            # |coeficiente| menor a esto se considera casi nulo
# hipótesis -> (variable de la hipótesis, nombre en el diccionario/variables originales, signo esperado, columnas del preprocesador)
H = {"H1": ("dti", "dti", +1, ["dti"]),
     "H2": ("term", "term", +1, ["term_meses"]),
     "H3": ("revol_util", "revol_util", +1, ["revol_util"]),
     "H4": ("delinq_2yrs", "delinq_2yrs", +1, ["delinq_2yrs"]),
     "H5": ("inq_last_6mths", "inq_last_6mths", +1, ["inq_last_6mths"]),
     "H6": ("annual_inc", "annual_inc", -1, ["annual_inc"]),
     "H7": ("antiguedad_cred_meses", "earliest_cr_line", -1, ["antiguedad_cred_meses"]),
     "H8": ("home_ownership", "home_ownership", +1, ["home_ownership_RENT", "home_ownership_MORTGAGE"])}

coef = pd.read_csv(RAIZ / "results" / "regresion_logistica_coeficientes.csv")
w = coef.pivot(index="variable", columns="particion", values="coeficiente")
desc = pd.read_csv(RAIZ / "results" / "tabla07_resultados_hipotesis.csv").set_index("hipotesis")
imp = None
f_imp = RAIZ / "results" / "importancia_variables.csv"
if f_imp.exists():
    imp = pd.read_csv(f_imp)
    imp["rango"] = imp["importancia"].rank(ascending=False).astype(int)
    imp = imp.set_index("variable")

filas = []
for h, (var, original, signo, cols) in H.items():
    if h == "H8":                      # RENT menos MORTGAGE: positivo = alquilar es más riesgoso
        a = w.loc["home_ownership_RENT", "aleatoria"] - w.loc["home_ownership_MORTGAGE", "aleatoria"]
        t = w.loc["home_ownership_RENT", "temporal"] - w.loc["home_ownership_MORTGAGE", "temporal"]
        nombre = "home_ownership (RENT menos MORTGAGE)"
    else:
        a, t = w.loc[cols[0], "aleatoria"], w.loc[cols[0], "temporal"]
        nombre = var
    coincide = (np.sign(a) == signo) and (np.sign(t) == signo)
    casi_nulo = max(abs(a), abs(t)) < UMBRAL_NULO
    fila = {"hipotesis": h, "variable": nombre, "signo_esperado": "+" if signo > 0 else "-",
            "coef_aleatoria": a, "coef_temporal": t,
            "coincide": "Sí, casi nulo" if (coincide and casi_nulo) else ("Sí" if coincide else "No"),
            "diferencia_descriptiva_pp": desc.loc[h, "diferencia_pp"], "resultado_descriptivo": desc.loc[h, "veredicto_descriptivo"]}
    if imp is not None:
        fila["importancia_interpretabilidad"] = imp["importancia"].get(original, np.nan)
        fila["rango_importancia"] = imp["rango"].get(original, np.nan)
    filas.append(fila)

final = pd.DataFrame(filas)
final.to_csv(RAIZ / "results" / "tabla_hipotesis_final.csv", index=False)
final[["hipotesis", "variable", "signo_esperado", "coef_aleatoria", "coef_temporal", "coincide"]].round(3).to_csv(
    RAIZ / "results" / "tabla10_coeficientes_hipotesis.csv", index=False)
pd.set_option("display.width", 200)
print(final.round(3).to_string(index=False))
if imp is None:
    print("(sin results/importancia_variables.csv: la columna de interpretabilidad queda pendiente de Leonard)")
