"""Tabla de comparación justa de la PD (regresión logística frente a la red) y figura de calibración comparada.

La fila de la regresión logística sale de results/regresion_logistica_*.  La fila de la red se llena
cuando existen los archivos de Leonard (contrato, mismos nombres de columnas que los de la logística):
  results/pd_mlp_metricas_por_semilla.csv   semilla, particion, conjunto, auc, gini, ks, brier, logloss  (30 semillas)
  results/pd_mlp_resumen.json               configuraciones_entrenadas, tiempo_total_busqueda_s, parametros_entrenables
  results/curva_calibracion_mlp.csv         semilla, particion, conjunto, grupo, n, prob_predicha_media, tasa_observada
Sin esos archivos solo se escribe la fila de la logística. Estadísticos: media, desviación estándar muestral
(ddof = 1), mejor y peor sobre las semillas; la logística es determinista (varianza entre semillas = 0).
Salidas: results/tabla_comparacion_pd.csv, figures/fig09_calibracion_comparada.png (si hay red).
Uso:  python scripts/11_tabla_comparacion_pd.py [--mlp DIRECTORIO_CON_ARCHIVOS_DE_LA_RED] [--salida DIRECTORIO]
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--mlp", default=str(RAIZ / "results"))
ap.add_argument("--salida", default=str(RAIZ / "results"))
ap.add_argument("--figuras", default=str(RAIZ / "figures"))
args = ap.parse_args()
dir_mlp, dir_salida, dir_fig = Path(args.mlp), Path(args.salida), Path(args.figuras)
METRICAS = ["auc", "gini", "ks", "brier"]
MENOR_ES_MEJOR = {"brier"}
CONJUNTOS = ["validacion", "prueba"]


def filas_modelo(tarea, modelo, particion, conjunto, config, tiempo, params, semillas, valores):
    """valores: DataFrame con una fila por semilla y una columna por métrica."""
    out = []
    for m in METRICAS:
        v = valores[m].to_numpy(dtype=float)
        mejor, peor = (v.min(), v.max()) if m in MENOR_ES_MEJOR else (v.max(), v.min())
        out.append({"tarea": tarea, "modelo": modelo, "particion": particion, "conjunto": conjunto,
                    "configuraciones_entrenadas": config, "tiempo_total_s": tiempo, "parametros_entrenables": params,
                    "semillas": semillas, "metrica": m, "media": v.mean(), "desv_est": v.std(ddof=1) if len(v) > 1 else 0.0,
                    "mejor": mejor, "peor": peor})
    return out


# ---- Regresión logística ----
met = pd.read_csv(RAIZ / "results" / "regresion_logistica_metricas.csv")
res = json.loads((RAIZ / "results" / "regresion_logistica_resumen.json").read_text(encoding="utf-8"))
filas = []
for part, info in res["particiones"].items():
    for conj in CONJUNTOS:
        fila = met[(met["particion"] == part) & (met["conjunto"] == conj)]
        assert len(fila) == 1
        filas += filas_modelo("PD", "Regresión logística", part, conj, info["configuraciones_entrenadas"],
                              info["tiempo_busqueda_s"], info["parametros_entrenables"], "0 (determinista)", fila)

# ---- Red neuronal (si existen los archivos) ----
f_met, f_res, f_cal = (dir_mlp / "pd_mlp_metricas_por_semilla.csv", dir_mlp / "pd_mlp_resumen.json",
                       dir_mlp / "curva_calibracion_mlp.csv")
hay_red = f_met.exists() and f_res.exists()
if hay_red:
    mm = pd.read_csv(f_met)
    rr = json.loads(f_res.read_text(encoding="utf-8"))
    n_sem = mm["semilla"].nunique()
    assert n_sem >= 30, f"el enunciado exige al menos 30 semillas y hay {n_sem}"
    for part in sorted(mm["particion"].unique()):
        for conj in CONJUNTOS:
            sub = mm[(mm["particion"] == part) & (mm["conjunto"] == conj)]
            assert sub["semilla"].is_unique and len(sub) == n_sem, (part, conj)
            filas += filas_modelo("PD", "MLP", part, conj, rr["configuraciones_entrenadas"], rr["tiempo_total_busqueda_s"],
                                  rr["parametros_entrenables"], n_sem, sub)
else:
    print("No se encontraron los archivos de la red en", dir_mlp, "- se escribe solo la fila de la logística.")

tabla = pd.DataFrame(filas)
dir_salida.mkdir(parents=True, exist_ok=True)
tabla.to_csv(dir_salida / "tabla_comparacion_pd.csv", index=False)

# ---- Figura de calibración comparada (prueba) ----
if hay_red and f_cal.exists():
    lg = pd.read_csv(RAIZ / "results" / "curva_calibracion_logreg.csv")
    cm = pd.read_csv(f_cal)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    parts = sorted(cm["particion"].unique())
    fig, axs = plt.subplots(1, len(parts), figsize=(5 * len(parts), 4.8), squeeze=False)
    for ax, part in zip(axs[0], parts):
        l = lg[(lg["particion"] == part) & (lg["conjunto"] == "prueba")]
        c = cm[(cm["particion"] == part) & (cm["conjunto"] == "prueba")]
        g = c.groupby("grupo").agg(x=("prob_predicha_media", "mean"), y=("tasa_observada", "mean"),
                                   ymin=("tasa_observada", "min"), ymax=("tasa_observada", "max"))
        tope = float(max(l["prob_predicha_media"].max(), l["tasa_observada"].max(), g["x"].max(), g["ymax"].max())) * 1.05
        ax.plot([0, tope], [0, tope], color="#8A8A8A", linestyle="--", label="Calibración perfecta")
        ax.plot(l["prob_predicha_media"], l["tasa_observada"], marker="o", color="#0072B2", label="Regresión logística")
        ax.plot(g["x"], g["y"], marker="s", color="#D55E00", label="MLP (media de las semillas)")
        ax.fill_between(g["x"], g["ymin"], g["ymax"], color="#D55E00", alpha=0.2, label="MLP (mínimo a máximo)")
        ax.set_title(f"Partición {part}")
        ax.set_xlabel("Probabilidad predicha media por decil")
        ax.set_ylabel("Tasa de incumplimiento observada")
        ax.set_xlim(0, tope)
        ax.set_ylim(0, tope)
    axs[0][0].legend(frameon=False, loc="upper left", fontsize=8)
    fig.tight_layout()
    dir_fig.mkdir(parents=True, exist_ok=True)
    fig.savefig(dir_fig / "fig09_calibracion_comparada.png", dpi=200)
    plt.close(fig)

pd.set_option("display.width", 220)
print(tabla[tabla["metrica"] == "auc"].round(4).to_string(index=False))
