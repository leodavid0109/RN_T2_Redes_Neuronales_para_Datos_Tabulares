"""Modelo de referencia de la Parte 1: regresión logística con penalización L2.

Decisiones del equipo (22 oct 2026): solo L2; criterio de selección = log-loss en validación;
sin ponderar clases; rejilla de C de 10 valores logarítmicos entre 1e-4 y 1e2.
La búsqueda se hace por separado en cada partición (aleatoria y temporal) usando solo entrenamiento
y validación; la prueba solo se evalúa una vez, con el C ya elegido. El modelo no se reajusta con
entrenamiento + validación. Datos de entrada: únicamente data/processed/ y el preprocesador guardado.

Convención de sklearn: se minimiza  ½‖w‖² + C · Σ_i pérdida_i  (C grande = poca regularización).
La regresión logística (lbfgs) es determinista: no usa semilla y su varianza entre semillas es cero.

Salidas (results/ y figures/):
  regresion_logistica_busqueda.csv, regresion_logistica_metricas.csv, regresion_logistica_coeficientes.csv,
  curva_calibracion_logreg.csv, regresion_logistica_resumen.json, fig05_busqueda_regularizacion.png,
  fig06_calibracion_logistica.png
Uso:  python scripts/08_regresion_logistica.py
"""
import json
import os
import platform
import sys
import time
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.stats import ks_2samp
from sklearn.linear_model import LogisticRegression

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from src.metricas import curva_calibracion, metricas_pd  # noqa: E402
from src.variables import CATEGORICAS, INDICADOR_FALTANTE  # noqa: E402


def _nombre_cpu():
    """Obtiene nombre de CPU sin asumir que existe /proc/cpuinfo (p. ej., en Windows)."""
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for linea in cpuinfo.read_text(encoding="utf-8", errors="ignore").splitlines():
            if linea.lower().startswith("model name") and ":" in linea:
                return linea.split(":", 1)[1].strip()
    return platform.processor() or platform.machine() or "desconocido"

# ---------------- Parámetros ----------------
GRILLA_C = np.logspace(-4, 2, 10)
MAX_ITER = 2000
PARTICIONES = {"aleatoria": "split_random", "temporal": "split_temporal"}
COLORES = {"aleatoria": "#0072B2", "temporal": "#009E73"}
# --------------------------------------------

df = pd.read_parquet(RAIZ / "data" / "processed" / "pd.parquet")
pre = joblib.load(RAIZ / "models" / "preprocessor.joblib")
columnas_entrada = list(pre.feature_names_in_)
X = pre.transform(df[columnas_entrada])
nombres = list(pre.get_feature_names_out())
y = df["objetivo"].to_numpy()
assert X.shape[1] == len(nombres) == 99

busqueda, metricas, coefs, calib = [], [], [], []
resumen = {"particiones": {}}
t_total = 0.0
for nombre, col in PARTICIONES.items():
    m = {s: (df[col] == s).to_numpy() for s in ["entrenamiento", "validacion", "prueba"]}
    filas, modelos = [], {}
    t0 = time.perf_counter()
    for C in GRILLA_C:
        t1 = time.perf_counter()
        mod = LogisticRegression(penalty="l2", C=C, solver="lbfgs", max_iter=MAX_ITER, class_weight=None)
        mod.fit(X[m["entrenamiento"]], y[m["entrenamiento"]])
        dt = time.perf_counter() - t1
        assert mod.n_iter_[0] < MAX_ITER, f"no convergió con C={C}"
        mv = metricas_pd(y[m["validacion"]], mod.predict_proba(X[m["validacion"]])[:, 1])
        filas.append({"particion": nombre, "C": C, "logloss_val": mv["logloss"], "auc_val": mv["auc"],
                      "iteraciones": int(mod.n_iter_[0]), "tiempo_s": dt})
        modelos[C] = mod
    t_busqueda = time.perf_counter() - t0
    t_total += t_busqueda
    tabla = pd.DataFrame(filas)
    C_opt = float(tabla.loc[tabla["logloss_val"].idxmin(), "C"])
    tabla["elegida"] = tabla["C"] == C_opt
    busqueda.append(tabla)
    mod = modelos[C_opt]

    # determinismo: ajustar de nuevo da exactamente los mismos coeficientes
    otro = LogisticRegression(penalty="l2", C=C_opt, solver="lbfgs", max_iter=MAX_ITER).fit(X[m["entrenamiento"]], y[m["entrenamiento"]])
    determinista = bool(np.array_equal(otro.coef_, mod.coef_) and np.array_equal(otro.intercept_, mod.intercept_))
    assert determinista

    n_parametros = int(mod.coef_.size + mod.intercept_.size)   # variables + sesgo
    assert n_parametros == X.shape[1] + 1

    for s in ["entrenamiento", "validacion", "prueba"]:
        p = mod.predict_proba(X[m[s]])[:, 1]
        r = metricas_pd(y[m[s]], p)
        # comprobación independiente del KS (prueba de dos muestras de scipy)
        ks_ref = ks_2samp(p[y[m[s]] == 1], p[y[m[s]] == 0]).statistic
        assert abs(r["ks"] - ks_ref) < 1e-9, (r["ks"], ks_ref)
        metricas.append({"particion": nombre, "conjunto": s, "registros": int(m[s].sum()),
                         "tasa_malos_pct": 100 * y[m[s]].mean(), "prob_media_predicha_pct": 100 * p.mean(), **r})
        if s != "entrenamiento":
            c = curva_calibracion(y[m[s]], p)
            c.insert(0, "conjunto", s)
            c.insert(0, "particion", nombre)
            calib.append(c)
    coefs.append(pd.DataFrame({"particion": nombre, "variable": nombres + ["(sesgo)"],
                               "coeficiente": np.append(mod.coef_[0], mod.intercept_[0])}))
    resumen["particiones"][nombre] = {
        "configuraciones_entrenadas": len(GRILLA_C), "C_elegido": C_opt,
        "tiempo_busqueda_s": round(t_busqueda, 3), "parametros_entrenables": n_parametros,
        "iteraciones_modelo_elegido": int(mod.n_iter_[0]), "determinista": determinista}

pd.concat(busqueda).to_csv(RAIZ / "results" / "regresion_logistica_busqueda.csv", index=False)
pd.DataFrame(metricas).to_csv(RAIZ / "results" / "regresion_logistica_metricas.csv", index=False)
pd.concat(coefs).to_csv(RAIZ / "results" / "regresion_logistica_coeficientes.csv", index=False)
pd.concat(calib).to_csv(RAIZ / "results" / "curva_calibracion_logreg.csv", index=False)
resumen.update({
    "penalizacion": "L2", "criterio_seleccion": "log-loss en validación", "class_weight": None,
    "grilla_C": [float(c) for c in GRILLA_C], "max_iter": MAX_ITER, "solver": "lbfgs",
    "semillas": "ninguna (modelo determinista; varianza entre semillas = 0)",
    "tiempo_total_busqueda_s": round(t_total, 3),
    "hardware": {"cpu": _nombre_cpu(), "nucleos": os.cpu_count(), "so": platform.platform()},
    "versiones": {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
                  "scikit-learn": sklearn.__version__, "scipy": scipy.__version__},
    "columnas_de_entrada": X.shape[1]})
(RAIZ / "results" / "regresion_logistica_resumen.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")

# ---------------- Figuras ----------------
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
b = pd.concat(busqueda)
fig, ax = plt.subplots(figsize=(6.4, 4))
for nombre in PARTICIONES:
    t = b[b["particion"] == nombre]
    ax.plot(t["C"], t["logloss_val"], marker="o", color=COLORES[nombre], label=f"Partición {nombre}")
    e = t[t["elegida"]].iloc[0]
    ax.scatter([e["C"]], [e["logloss_val"]], s=140, facecolors="none", edgecolors=COLORES[nombre], linewidths=2)
ax.set_xscale("log")
ax.set_xlabel("C (inverso de la regularización L2)")
ax.set_ylabel("Log-loss en validación")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(RAIZ / "figures" / "fig05_busqueda_regularizacion.png", dpi=200)
plt.close(fig)

c = pd.concat(calib)
c = c[c["conjunto"] == "prueba"]
fig, ax = plt.subplots(figsize=(5, 5))
tope = float(max(c["prob_predicha_media"].max(), c["tasa_observada"].max())) * 1.05
ax.plot([0, tope], [0, tope], color="#8A8A8A", linestyle="--", label="Calibración perfecta")
for nombre in PARTICIONES:
    t = c[c["particion"] == nombre]
    ax.plot(t["prob_predicha_media"], t["tasa_observada"], marker="o", color=COLORES[nombre], label=f"Partición {nombre}")
ax.set_xlabel("Probabilidad predicha media por decil")
ax.set_ylabel("Tasa de incumplimiento observada")
ax.set_xlim(0, tope)
ax.set_ylim(0, tope)
ax.legend(frameon=False, loc="upper left")
fig.tight_layout()
fig.savefig(RAIZ / "figures" / "fig06_calibracion_logistica.png", dpi=200)
plt.close(fig)

pd.set_option("display.width", 200)
print(pd.concat(busqueda).to_string(index=False))
print(pd.DataFrame(metricas).round(4).to_string(index=False))
print(json.dumps(resumen, indent=2, ensure_ascii=False))
