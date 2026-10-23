"""Guarda los modelos finales de la regresión logística (los C ya elegidos en la búsqueda del script 08).

El ajuste es determinista; se comprueba que los coeficientes coinciden con los de
results/regresion_logistica_coeficientes.csv. Salidas: models/pd_logreg_aleatoria.joblib y
models/pd_logreg_temporal.joblib.
Uso:  python scripts/09_guardar_logistica.py
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

RAIZ = Path(__file__).resolve().parents[1]
df = pd.read_parquet(RAIZ / "data" / "processed" / "pd.parquet")
pre = joblib.load(RAIZ / "models" / "preprocessor.joblib")
X = pre.transform(df[list(pre.feature_names_in_)])
y = df["objetivo"].to_numpy()
resumen = json.loads((RAIZ / "results" / "regresion_logistica_resumen.json").read_text(encoding="utf-8"))
coefs = pd.read_csv(RAIZ / "results" / "regresion_logistica_coeficientes.csv")

for nombre, col in [("aleatoria", "split_random"), ("temporal", "split_temporal")]:
    C = resumen["particiones"][nombre]["C_elegido"]
    m = (df[col] == "entrenamiento").to_numpy()
    mod = LogisticRegression(penalty="l2", C=C, solver="lbfgs", max_iter=resumen["max_iter"], class_weight=None).fit(X[m], y[m])
    ref = coefs[coefs["particion"] == nombre]["coeficiente"].to_numpy()
    assert np.allclose(np.append(mod.coef_[0], mod.intercept_[0]), ref, rtol=0, atol=1e-9), nombre
    joblib.dump(mod, RAIZ / "models" / f"pd_logreg_{nombre}.joblib")
    print(nombre, "C =", C, "| guardado | coeficientes iguales a los del script 08")
